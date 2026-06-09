// GlProgram — the program builder that replaces THREE's `ShaderMaterial`/`MaterialBase`
// (plan/three-removal/02-gl-framework.md §2; verbatim fork: MaterialBase.js, 71 lines).
//
// Verified fork behaviours reproduced here:
//  1. Uniform-as-property aliasing (MaterialBase.js:20-38) → explicit setFloat/setVec*/…
//     + a cached name→WebGLUniformLocation map (lazy `#loc`).
//  2. needsUpdate → 'recompilation' event (MaterialBase.js:5-14, 43-69) → `setDefine`
//     is CHANGE-GATED: a real change marks `#dirty`, and the relink happens on the NEXT
//     `use()`. An unchanged setDefine is a no-op (NEVER relink per-frame — a recompile
//     resets the accumulator, per PhysicalPathTracingMaterial.js:43-44).
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

function compileShader(gl: WebGL2RenderingContext, type: GLenum, src: string): WebGLShader {
  const sh = gl.createShader(type);
  if (sh == null) throw new Error('pt-webgl2: gl.createShader returned null');
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh) ?? '(no info log)';
    gl.deleteShader(sh);
    const kind = type === gl.VERTEX_SHADER ? 'vertex' : 'fragment';
    throw new Error(`pt-webgl2: ${kind} shader compile failed:\n${log}`);
  }
  return sh;
}

function linkProgram(gl: WebGL2RenderingContext, vertSrc: string, fragSrc: string): WebGLProgram {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vertSrc);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fragSrc);
  const program = gl.createProgram();
  if (program == null) {
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    throw new Error('pt-webgl2: gl.createProgram returned null');
  }
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  // Shaders can be detached + deleted once linked; the program retains the binary.
  gl.detachShader(program, vs);
  gl.detachShader(program, fs);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program) ?? '(no info log)';
    gl.deleteProgram(program);
    throw new Error(`pt-webgl2: program link failed:\n${log}`);
  }
  return program;
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
  readonly #uniformLoc = new Map<string, WebGLUniformLocation | null>();
  readonly #samplerUnit = new Map<string, number>();
  #dirty = true;

  constructor(
    gl: WebGL2RenderingContext,
    vertSrc: string,
    fragSrcBody: string,
    defines: Record<string, number>,
  ) {
    this.#gl = gl;
    this.#vertSrc = vertSrc;
    this.#fragSrcBody = fragSrcBody;
    for (const [k, v] of Object.entries(defines)) this.#defines.set(k, v);
  }

  /**
   * Change-gated `#define` set (MaterialBase.js:43-69). Returns true on an ACTUAL change,
   * marking the program dirty so the next `use()` relinks. Returns false (no-op) when the
   * value is unchanged — this is what prevents per-frame recompiles (which reset the accumulator).
   */
  setDefine(name: string, value: number): boolean {
    if (this.#defines.get(name) === value) return false;
    this.#defines.set(name, value);
    this.#dirty = true;
    return true;
  }

  /** Bind the program, relinking first if a define changed since the last use. */
  use(): void {
    if (this.#dirty || this.#program == null) this.#relink();
    this.#gl.useProgram(this.#program);
  }

  setFloat(name: string, v: number): void {
    const l = this.#loc(name);
    if (l != null) this.#gl.uniform1f(l, v);
  }

  setInt(name: string, v: number): void {
    const l = this.#loc(name);
    if (l != null) this.#gl.uniform1i(l, v);
  }

  setVec2(name: string, x: number, y: number): void {
    const l = this.#loc(name);
    if (l != null) this.#gl.uniform2f(l, x, y);
  }

  setVec3(name: string, x: number, y: number, z: number): void {
    const l = this.#loc(name);
    if (l != null) this.#gl.uniform3f(l, x, y, z);
  }

  setVec4(name: string, x: number, y: number, z: number, w: number): void {
    const l = this.#loc(name);
    if (l != null) this.#gl.uniform4f(l, x, y, z, w);
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

  /** The texture unit assigned to a sampler at link (null if absent) — for UBO/MRT wiring. */
  samplerUnit(name: string): number | null {
    return this.#samplerUnit.get(name) ?? null;
  }

  /** The linked GLProgram handle (null until first `use()`/relink) — for UBO block binding. */
  get program(): WebGLProgram | null {
    return this.#program;
  }

  dispose(): void {
    if (this.#program != null) {
      this.#gl.deleteProgram(this.#program);
      this.#program = null;
    }
    this.#uniformLoc.clear();
    this.#samplerUnit.clear();
    this.#dirty = true;
  }

  #relink(): void {
    const gl = this.#gl;
    if (this.#program != null) gl.deleteProgram(this.#program);
    const vert = buildVertexSource(this.#defines, this.#vertSrc);
    const frag = buildFragmentSource(this.#defines, this.#fragSrcBody);
    this.#program = linkProgram(gl, vert, frag);
    this.#uniformLoc.clear();
    this.#samplerUnit.clear();
    assignSamplerUnits(gl, this.#program, this.#uniformLoc, this.#samplerUnit);
    this.#dirty = false;
  }

  #loc(name: string): WebGLUniformLocation | null {
    if (!this.#uniformLoc.has(name)) {
      const loc = this.#program == null ? null : this.#gl.getUniformLocation(this.#program, name);
      this.#uniformLoc.set(name, loc);
    }
    return this.#uniformLoc.get(name) ?? null;
  }
}
