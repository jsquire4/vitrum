// PresentPass — the tonemap/exposure/outputColorSpace present pass (D10.1).
// Previously the #presentTex / #presentFbo / #presentProgram fields + the
// #runPresentPass / #ensurePresentProgram / buildPresentFragBody / BLEND_FRAG
// ... wait: BLEND_FRAG belongs to the blend-composite step, not the present pass.
// Only the present-pass pieces are extracted here:
//   buildPresentFragBody, #ensurePresentProgram, #runPresentPass,
//   #presentTex, #presentFbo, #presentProgram.
//
// The present-pass reads the portable running-mean ping-pong result and writes
// the tonemapped result to an
// allocated RGBA32F texture (presentTex).
//
// Usage pattern:
//   1. allocate(gl, w, h)      — call when ensureAccumResources reallocates
//   2. destroy()               — call on resize or engine dispose
//   3. run(srcTex, mode, exp, cs) — call once per frame after the accum step
//   4. tex                     — the result texture (null before first allocate)
//
// Provenance: extracted from glResources.ts, behavior-preserving (no logic changed).

import { GlProgram } from './glProgram.js';
import { FullscreenQuad, FULLSCREEN_VERT } from './fullscreenQuad.js';
import { createRenderTarget, type RenderTarget } from './framebuffer.js';
import * as TonemapFunctions from '../glsl/shader/common/tonemap_functions.glsl.js';

/**
 * Present-pass fragment shader body (no `#version`/preamble — GlProgram prepends those).
 * Reads the HDR accumulation texture (RGBA32F), applies exposure + the selected tonemap
 * operator, and optionally applies the sRGB OETF before writing to the present target.
 *
 * Uniforms:
 *   uAccumTex       — RGBA32F accumulation texture (sampler2D)
 *   uTonemapMode    — operator index (0=aces, 1=agx, 2=reinhard, 3=linear, 4=none)
 *   uExposure       — linear-exposure multiplier (default 1.0)
 *   uOutputColorSpace — 0=srgb (apply OETF, default), 1=linear (skip OETF)
 *
 * Wired 2026-06-10: FrameQualitySettings.tonemap / .exposure / .outputColorSpace.
 */
function buildPresentFragBody(tonemapGlsl: string): string {
  return /* glsl */ `
in vec2 vUv;
uniform sampler2D uAccumTex;
uniform int uTonemapMode;
uniform float uExposure;
uniform int uOutputColorSpace;

${tonemapGlsl}

void main() {
  vec3 hdr = texture(uAccumTex, vUv).rgb;
  // Guard against negative values that can appear from alpha-compositing precision.
  vec3 tonemapped = vitrumTonemap(max(hdr, vec3(0.0)), uTonemapMode, uExposure);
  // outputColorSpace 0 = srgb (default) — apply the IEC 61966-2-1 OETF before
  // writing the display-referred output (the framebuffer is RGBA32F, not auto-sRGB).
  // outputColorSpace 1 = linear — skip the OETF (useful for HDR/linear pipeline).
  if (uOutputColorSpace == 0) {
    pc_fragColor = vec4(vt_linearToSrgb(tonemapped), 1.0);
  } else {
    pc_fragColor = vec4(tonemapped, 1.0);
  }
}
`;
}

/**
 * Manages the single-attachment present-pass target (RGBA32F) + the tonemap
 * fullscreen-quad program. Owned by GlResources.
 *
 * Default dials match the contract (FrameQualitySettings) defaults and the
 * walkaround-hybrid orchestrator (HybridEngineFrameOrchestrator.ts:764):
 *   tonemapMode = 0 (aces), exposure = 1.0, outputColorSpace = 0 (srgb).
 */
export class PresentPass {
  readonly #gl: WebGL2RenderingContext;
  readonly #quad: FullscreenQuad;
  #target: RenderTarget | null = null;
  #program: GlProgram | null = null;

  constructor(gl: WebGL2RenderingContext, quad: FullscreenQuad) {
    this.#gl = gl;
    this.#quad = quad;
  }

  /** The tonemapped present texture (null before first allocate). */
  get tex(): WebGLTexture | null {
    return this.#target?.color ?? null;
  }

  /** Start or poll the present program without issuing a draw. */
  prepareProgram(): boolean {
    return this.#ensureProgram().prepare();
  }

  /**
   * Allocate (or reallocate) the present target at the given dimensions.
   * Call whenever ensureAccumResources reallocates (width or height changed).
   * RGBA32F — deliberate; the present texture is the public primaryRadiance
   * and must stay FLOAT-readable (see framebuffer.ts's present-format note).
   */
  allocate(w: number, h: number): void {
    const next = createRenderTarget(this.#gl, w, h, false);
    const previous = this.#target;
    this.#target = next;
    previous?.destroy();
  }

  /**
   * Run the present pass: blit the provided HDR source texture through the
   * tonemap + OETF chain into the present target.
   *
   * Called once per drawAccumStep, after the PT sample and running-mean
   * composite. The present target is already allocated by
   * `allocate` so this is a no-alloc hot path.
   */
  run(
    srcTex: WebGLTexture,
    width: number,
    height: number,
    tonemapMode: number,
    exposure: number,
    outputColorSpace: number,
  ): void {
    const gl = this.#gl;
    const fbo = this.#target?.fbo ?? null;
    if (fbo == null) return;

    const prog = this.#ensureProgram();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
    gl.viewport(0, 0, width, height);
    gl.disable(gl.BLEND); // no blending in the present pass — overwrite only

    if (!prog.use()) {
      throw new Error('pt-webgl2: present pass reached draw before its program was ready');
    }
    prog.bindTexture('uAccumTex', srcTex);
    prog.setInt('uTonemapMode', tonemapMode);
    prog.setFloat('uExposure', exposure);
    prog.setInt('uOutputColorSpace', outputColorSpace);
    this.#quad.draw(gl);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /** The present FBO (for readPixels source selection). */
  get fbo(): WebGLFramebuffer | null {
    return this.#target?.fbo ?? null;
  }

  /** Destroy the present target textures and FBO (NOT the shared quad/program). */
  destroy(): void {
    const target = this.#target;
    this.#target = null;
    target?.destroy();
  }

  /** Destroy everything including the program. Call on full engine dispose. */
  disposeProgram(): void {
    this.#program?.dispose();
    this.#program = null;
  }

  // ── private ────────────────────────────────────────────────────────────────

  #ensureProgram(): GlProgram {
    if (this.#program == null) {
      const tonemapGlsl = (TonemapFunctions as Record<string, unknown>)['tonemap_functions'];
      if (typeof tonemapGlsl !== 'string') throw new Error('pt-webgl2: tonemap_functions GLSL not found');
      this.#program = new GlProgram(
        this.#gl,
        FULLSCREEN_VERT,
        buildPresentFragBody(tonemapGlsl),
        {}, // no compile-time defines; all dials are uniforms
      );
    }
    return this.#program;
  }
}
