// GlProgram — the program builder that replaces THREE's `ShaderMaterial`/`MaterialBase`
// (plan/three-removal/02-gl-framework.md §2; verbatim fork: MaterialBase.js, 71 lines).
//
// Verified fork behaviours reproduced here:
//  1. Uniform-as-property aliasing (MaterialBase.js:20-38) → explicit setFloat/setVec*/…
//     + a cached name→WebGLUniformLocation map (lazy `#loc`).
//  2. Defines are construction-time immutable. Feature changes build a staged
//     replacement program graph in GlResources rather than mutating a linked program
//     in place (which would reset the accumulator).
//  3. GLSL3 is implicit in THREE (auto from `precision highp isampler2D` + `layout(location=N) out`).
//     We have no ShaderMaterial, so the preamble emits `#version 300 es` ourselves, and the
//     FRAGMENT additionally gets the precision qualifiers + `layout(location = 0) out vec4
//     pc_fragColor;` that THREE auto-injected (the fork shader assumes pc_fragColor at loc 0).
//
// Sampler units are assigned sequentially at link time (mirrors THREE's auto sampler-unit
// assignment; plan 02 §2 + the binding-convention remap in 04 §4).

/** Build the shared `#version` + `#define` preamble (exposed for GPU-free unit tests). */
export function buildPreamble(defines: ReadonlyMap<string, number>): string {
  let s = '#version 300 es\n';
  for (const [k, v] of defines) s += `#define ${k} ${v}\n`;
  return s;
}

/** The fragment-only header THREE auto-injects; we must emit it ourselves (plan 02 §2). */
const FRAG_HEADER =
  'precision highp float;\nprecision highp int;\nlayout(location = 0) out vec4 pc_fragColor;\n';

// THREE-compat GLSL1→GLSL3 keyword bridges. THREE's WebGLProgram silently rewrites the
// fork's `varying`/`attribute`/`texture2D`/`gl_FragColor` for GLSL3 via these #defines;
// our raw `#version 300 es` must emit them or the (verbatim-copied) fork kernels won't
// compile. Verified against a real GL driver (llvmpipe) — string/mock tests can't catch this.
const VERT_COMPAT = '#define attribute in\n#define varying out\n#define texture2D texture\n';
const FRAG_COMPAT =
  '#define varying in\n#define texture2D texture\n#define textureCube texture\n' +
  '#define texture2DLodEXT textureLod\n#define textureCubeLodEXT textureLod\n' +
  '#define gl_FragColor pc_fragColor\n';

/** Compose the full vertex source (preamble + GLSL3-compat defines + body). */
export function buildVertexSource(defines: ReadonlyMap<string, number>, vertBody: string): string {
  return `${buildPreamble(defines)}${VERT_COMPAT}${vertBody}`;
}

/** Compose the full fragment source (preamble + frag header + GLSL3-compat defines + body). */
export function buildFragmentSource(defines: ReadonlyMap<string, number>, fragBody: string): string {
  return `${buildPreamble(defines)}${FRAG_HEADER}${FRAG_COMPAT}${fragBody}`;
}

// GLSL sampler keywords whose declared uniforms consume a texture unit.
const SAMPLER_TYPE_RE = /\b(?:[iu]?sampler2D(?:Array)?|[iu]?sampler3D|[iu]?samplerCube)\b/;

interface ParallelShaderCompileExtension {
  readonly COMPLETION_STATUS_KHR: GLenum;
}

interface PendingProgramLink {
  readonly program: WebGLProgram;
  readonly vertexShader: WebGLShader;
  readonly fragmentShader: WebGLShader;
  readonly startedAtMs: number;
}

export interface GlProgramOptions {
  /**
   * Maximum elapsed monotonic wall time a KHR_parallel_shader_compile link may
   * remain incomplete. The deadline is observed on the next prepare()/use()
   * poll; no background timer is installed.
   */
  readonly linkTimeoutMs?: number;
}

/** Long enough for cold ANGLE compilation, but bounded so a broken driver cannot hang forever. */
export const DEFAULT_PROGRAM_LINK_TIMEOUT_MS = 120_000;

function monotonicNowMs(): number {
  return globalThis.performance?.now() ?? Date.now();
}

/**
 * Walk the active uniforms; cache every location and assign a sequential texture unit
 * to each sampler-typed uniform (mirrors THREE's link-time sampler-unit assignment).
 */
function assignSamplerUnits(
  gl: WebGL2RenderingContext,
  program: WebGLProgram,
  uniformLoc: Map<string, WebGLUniformLocation | null>,
  samplerUnit: Map<string, number>,
): void {
  const count = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS) as number;
  let nextUnit = 0;
  for (let i = 0; i < count; i += 1) {
    const info = gl.getActiveUniform(program, i);
    if (info == null) continue;
    // Array uniforms come back as `name[0]`; the base name resolves the location too.
    const baseName = info.name.endsWith('[0]') ? info.name.slice(0, -3) : info.name;
    const loc = gl.getUniformLocation(program, info.name);
    uniformLoc.set(baseName, loc);
    if (SAMPLER_TYPE_RE.test(glTypeName(gl, info.type))) {
      samplerUnit.set(baseName, nextUnit);
      nextUnit += 1;
    }
  }
}

/** Map a GL uniform-type enum to its GLSL keyword (only the sampler types matter here). */
function glTypeName(gl: WebGL2RenderingContext, type: GLenum): string {
  switch (type) {
    case gl.SAMPLER_2D: return 'sampler2D';
    case gl.INT_SAMPLER_2D: return 'isampler2D';
    case gl.UNSIGNED_INT_SAMPLER_2D: return 'usampler2D';
    case gl.SAMPLER_2D_ARRAY: return 'sampler2DArray';
    case gl.INT_SAMPLER_2D_ARRAY: return 'isampler2DArray';
    case gl.UNSIGNED_INT_SAMPLER_2D_ARRAY: return 'usampler2DArray';
    case gl.SAMPLER_3D: return 'sampler3D';
    case gl.SAMPLER_CUBE: return 'samplerCube';
    default: return 'non-sampler';
  }
}

export class GlProgram {
  readonly #gl: WebGL2RenderingContext;
  readonly #vertSrc: string;
  readonly #fragSrcBody: string;
  readonly #defines = new Map<string, number>();
  #program: WebGLProgram | null = null;
  #pendingLink: PendingProgramLink | null = null;
  #parallelCompile: ParallelShaderCompileExtension | null | undefined;
  #linkFailure: Error | null = null;
  readonly #linkTimeoutMs: number;
  readonly #uniformLoc = new Map<string, WebGLUniformLocation | null>();
  readonly #samplerUnit = new Map<string, number>();
  #dirty = true;

  constructor(
    gl: WebGL2RenderingContext,
    vertSrc: string,
    fragSrcBody: string,
    defines: Record<string, number>,
    options: GlProgramOptions = {},
  ) {
    this.#gl = gl;
    this.#vertSrc = vertSrc;
    this.#fragSrcBody = fragSrcBody;
    const linkTimeoutMs = options.linkTimeoutMs ?? DEFAULT_PROGRAM_LINK_TIMEOUT_MS;
    if (!Number.isFinite(linkTimeoutMs) || linkTimeoutMs <= 0) {
      throw new RangeError('pt-webgl2: GlProgram linkTimeoutMs must be finite and > 0');
    }
    this.#linkTimeoutMs = linkTimeoutMs;
    for (const [k, v] of Object.entries(defines)) this.#defines.set(k, v);
  }

  /**
   * Start or poll compilation/linking without blocking on LINK_STATUS when
   * KHR_parallel_shader_compile is available. Returns false while the driver is
   * still compiling and throws on a compile/link error or deadline expiry.
   */
  prepare(): boolean {
    if (this.#linkFailure != null) throw this.#linkFailure;
    if (!this.#dirty && this.#program != null) return true;
    if (this.#pendingLink == null) this.#beginLink();
    return this.#pollPendingLink();
  }

  /** Bind the program when ready; return false without changing GL state while pending. */
  use(): boolean {
    if (!this.prepare()) return false;
    this.#gl.useProgram(this.#program);
    return true;
  }

  setFloat(name: string, v: number): void {
    const l = this.#loc(name);
    if (l != null) this.#gl.uniform1f(l, v);
  }

  setInt(name: string, v: number): void {
    const l = this.#loc(name);
    if (l != null) this.#gl.uniform1i(l, v);
  }

  /**
   * Set a `uint` uniform (e.g. `lights.count`). WebGL2 requires `uniform1ui`
   * for unsigned-int uniforms — `uniform1i` on a `uint` location is a GL type
   * error, so before this setter existed the fork GLSL's `uint` uniforms could
   * not be driven at all (the `lights.count` analytic-light gate stayed `0u`).
   */
  setUint(name: string, v: number): void {
    const l = this.#loc(name);
    if (l != null) this.#gl.uniform1ui(l, v >>> 0);
  }

  setVec2(name: string, x: number, y: number): void {
    const l = this.#loc(name);
    if (l != null) this.#gl.uniform2f(l, x, y);
  }

  setVec3(name: string, x: number, y: number, z: number): void {
    const l = this.#loc(name);
    if (l != null) this.#gl.uniform3f(l, x, y, z);
  }

  /** Upload a column-major mat4 (THREE/GL convention — `transpose` is always false). */
  setMat4(name: string, m: Float32Array | readonly number[]): void {
    const l = this.#loc(name);
    if (l == null) return;
    const data = m instanceof Float32Array ? m : new Float32Array(m);
    this.#gl.uniformMatrix4fv(l, false, data);
  }

  /** Upload a float[] uniform (e.g. the spectral CMF tables uCmfX/uXCmfCdf). */
  setFloatArray(name: string, arr: Float32Array | readonly number[]): void {
    const l = this.#loc(name);
    if (l == null) return;
    const data = arr instanceof Float32Array ? arr : new Float32Array(arr);
    this.#gl.uniform1fv(l, data);
  }

  /**
   * Bind a texture to a sampler uniform: activate the unit assigned at link, bind the
   * texture to `target`, and point the sampler at that unit (THREE auto-bind equivalent;
   * plan 04 §4). No-op when the program does not declare the sampler.
   */
  bindTexture(name: string, tex: WebGLTexture, target: GLenum = this.#gl.TEXTURE_2D): void {
    const unit = this.#samplerUnit.get(name);
    if (unit == null) return;
    const gl = this.#gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(target, tex);
    const l = this.#loc(name);
    if (l != null) gl.uniform1i(l, unit);
  }

  dispose(): void {
    this.#discardPendingLink();
    if (this.#program != null) {
      this.#gl.deleteProgram(this.#program);
      this.#program = null;
    }
    this.#uniformLoc.clear();
    this.#samplerUnit.clear();
    this.#linkFailure = null;
    this.#dirty = true;
  }

  #beginLink(): void {
    const gl = this.#gl;
    const vert = buildVertexSource(this.#defines, this.#vertSrc);
    const frag = buildFragmentSource(this.#defines, this.#fragSrcBody);
    const vertexShader = gl.createShader(gl.VERTEX_SHADER);
    const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
    if (vertexShader == null || fragmentShader == null) {
      if (vertexShader != null) gl.deleteShader(vertexShader);
      if (fragmentShader != null) gl.deleteShader(fragmentShader);
      throw new Error('pt-webgl2: gl.createShader returned null');
    }
    const program = gl.createProgram();
    if (program == null) {
      gl.deleteShader(vertexShader);
      gl.deleteShader(fragmentShader);
      throw new Error('pt-webgl2: gl.createProgram returned null');
    }
    try {
      gl.shaderSource(vertexShader, vert);
      gl.compileShader(vertexShader);
      gl.shaderSource(fragmentShader, frag);
      gl.compileShader(fragmentShader);
      gl.attachShader(program, vertexShader);
      gl.attachShader(program, fragmentShader);
      gl.linkProgram(program);
      this.#pendingLink = {
        program,
        vertexShader,
        fragmentShader,
        startedAtMs: monotonicNowMs(),
      };
    } catch (error) {
      gl.deleteProgram(program);
      gl.deleteShader(vertexShader);
      gl.deleteShader(fragmentShader);
      throw error;
    }
  }

  #pollPendingLink(): boolean {
    const pending = this.#pendingLink;
    if (pending == null) return !this.#dirty && this.#program != null;
    const extension = this.#parallelShaderCompile();
    if (extension != null) {
      const complete = Boolean(
        this.#gl.getProgramParameter(pending.program, extension.COMPLETION_STATUS_KHR),
      );
      if (!complete) {
        const elapsedMs = monotonicNowMs() - pending.startedAtMs;
        if (elapsedMs < this.#linkTimeoutMs) return false;
        const error = new Error(
          `pt-webgl2: program link did not complete within ${this.#linkTimeoutMs} ms ` +
            `(fragment source ${this.#fragSrcBody.length} chars)`,
        );
        this.#discardPendingLink();
        this.#linkFailure = error;
        throw error;
      }
    }
    this.#finalizePendingLink(pending);
    return true;
  }

  #finalizePendingLink(pending: PendingProgramLink): void {
    const gl = this.#gl;
    const vertexOk = Boolean(gl.getShaderParameter(pending.vertexShader, gl.COMPILE_STATUS));
    const fragmentOk = Boolean(gl.getShaderParameter(pending.fragmentShader, gl.COMPILE_STATUS));
    const linkOk = Boolean(gl.getProgramParameter(pending.program, gl.LINK_STATUS));
    const vertexLog = gl.getShaderInfoLog(pending.vertexShader) ?? '';
    const fragmentLog = gl.getShaderInfoLog(pending.fragmentShader) ?? '';
    const programLog = gl.getProgramInfoLog(pending.program) ?? '';
    this.#releasePendingShaders(pending);
    this.#pendingLink = null;

    let failure: Error | null = null;
    if (!vertexOk) {
      failure = new Error(`pt-webgl2: vertex shader compile failed:\n${vertexLog || '(no info log)'}`);
    } else if (!fragmentOk) {
      failure = new Error(
        `pt-webgl2: fragment shader compile failed:\n${fragmentLog || '(no info log)'}`,
      );
    } else if (!linkOk) {
      failure = new Error(`pt-webgl2: program link failed:\n${programLog || '(no info log)'}`);
    }
    if (failure != null) {
      gl.deleteProgram(pending.program);
      this.#linkFailure = failure;
      throw failure;
    }

    const previous = this.#program;
    this.#program = pending.program;
    if (previous != null) gl.deleteProgram(previous);
    this.#uniformLoc.clear();
    this.#samplerUnit.clear();
    assignSamplerUnits(gl, this.#program, this.#uniformLoc, this.#samplerUnit);
    this.#dirty = false;
  }

  #parallelShaderCompile(): ParallelShaderCompileExtension | null {
    if (this.#parallelCompile === undefined) {
      this.#parallelCompile = this.#gl.getExtension(
        'KHR_parallel_shader_compile',
      );
    }
    return this.#parallelCompile;
  }

  #releasePendingShaders(pending: PendingProgramLink): void {
    const gl = this.#gl;
    gl.detachShader(pending.program, pending.vertexShader);
    gl.detachShader(pending.program, pending.fragmentShader);
    gl.deleteShader(pending.vertexShader);
    gl.deleteShader(pending.fragmentShader);
  }

  #discardPendingLink(): void {
    const pending = this.#pendingLink;
    if (pending == null) return;
    this.#pendingLink = null;
    this.#releasePendingShaders(pending);
    this.#gl.deleteProgram(pending.program);
  }

  #loc(name: string): WebGLUniformLocation | null {
    if (!this.#uniformLoc.has(name)) {
      const loc = this.#program == null ? null : this.#gl.getUniformLocation(this.#program, name);
      this.#uniformLoc.set(name, loc);
    }
    return this.#uniformLoc.get(name) ?? null;
  }
}
