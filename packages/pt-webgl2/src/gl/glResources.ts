// GlResources — the GL resource owner (plan/three-removal/02-gl-framework.md §5, §6).
// Analog of pt-webgpu's GpuResources / the fork's PathTracingRenderer state. It owns:
//   - the accumulation FBO + RGBA32F color texture (+ optional MRT gNormalDepth/gAlbedo),
//   - the Regime-2 ping-pong blend pair (allocated lazily only when needed),
//   - the PT GlProgram (built from composeTraceGlsl + FULLSCREEN_VERT),
//   - the Regime-2 BlendMaterial composite quad program,
//   - the std140 FrameParams UBO (WS5 frameParamsPacker target),
//   - a FullscreenQuad.
//
// The per-sample draw (drawAccumStep) replaces the fork's renderTask generator: bind the
// accum FBO + MRT draw buffers, set the blend regime, bind scene textures + params UBO, and
// draw the fullscreen triangle (fork PathTracingRenderer.js:144-167, §6 of plan 02).

import type { TraceFeatures, AccumRegime } from '../featureTypes.js';
import { featureDefines } from '../featureTypes.js';
import type { UploadedSceneTextures } from '../scene/sceneTextures.js';
import { composeTraceGlsl } from '../glsl/composeTraceGlsl.js';
import { GlProgram } from './glProgram.js';
import { FullscreenQuad, FULLSCREEN_VERT } from './fullscreenQuad.js';
import {
  createRenderTarget,
  createColorTexture,
  bindRenderTarget,
  clearRenderTarget,
  type RenderTarget,
} from './framebuffer.js';
import { setBlendForRegime } from './blend.js';
import { probeGlCaps, type GlCaps } from './glCaps.js';
import {
  CIE_X_TABLE, CIE_Y_TABLE, CIE_Z_TABLE,
  X_CMF_CDF, Y_CMF_CDF, Z_CMF_CDF,
  X_CMF_INTEGRAL, Y_CMF_INTEGRAL, Z_CMF_INTEGRAL,
} from '@vitrum/shared-samplers';
// Present-pass tonemap GLSL functions (port of @vitrum/shared-samplers,
// kept numerically identical — see tonemap_functions.glsl.js for provenance).
import * as TonemapFunctions from '../glsl/shader/common/tonemap_functions.glsl.js';

// H2 — spectral CMF upload tables (constant; precomputed Float32 copies so the
// per-frame upload allocates nothing). The spectral path importance-samples the
// hero wavelength against these CIE 1931 tables/CDFs and reconstructs RGB via the
// integrals; without them every uniform is 0 → wavelengthPdf=0 → black. The CDFs
// are Float64Array in @vitrum/shared-samplers (length 82); GL needs Float32.
const CMF_X_F32 = Float32Array.from(CIE_X_TABLE);
const CMF_Y_F32 = Float32Array.from(CIE_Y_TABLE);
const CMF_Z_F32 = Float32Array.from(CIE_Z_TABLE);
const CMF_XCDF_F32 = Float32Array.from(X_CMF_CDF);
const CMF_YCDF_F32 = Float32Array.from(Y_CMF_CDF);
const CMF_ZCDF_F32 = Float32Array.from(Z_CMF_CDF);

/**
 * Per-frame INDIVIDUAL uniforms the (verbatim-copied) fork GLSL reads — it declares
 * `uniform vec2 resolution; uniform int bounces; uniform mat4 cameraWorldMatrix; ...`,
 * NOT a FrameParams UBO. Verified on a real driver: a UBO bind alone renders black
 * (camera matrices read as zero). The engine computes these each frame.
 */
export interface FrameUniforms {
  readonly resolution: readonly [number, number];
  readonly bounces: number;
  readonly transmissiveBounces: number;
  readonly filterGlossyFactor: number;
  readonly radianceClamp: number;
  readonly cameraWorldMatrix: Float32Array; // inverse(viewMatrix)
  readonly invProjectionMatrix: Float32Array; // inverse(projMatrix)
  readonly environmentIntensity: number;
  readonly environmentRotation: Float32Array; // mat4
  readonly spectralEnabled: boolean;
  readonly causticStrategy: number; // 0=none, 1=manifold-nee, 2=photon-map
  readonly mneeMaxIterations: number;
  readonly mneeMaxChainLength: number;
  readonly backgroundAlpha: number; // 1 = opaque visible env (default); <1 = transparent (alpha-composite)
  /**
   * A5 (2026-06-10): BDPT host-driver inputs. `bdpt` mirrors the FEATURE_BDPT
   * compile flag (so the driver knows to issue the light-subpath passes); when it
   * is false the BDPT path is never touched and the frame is byte-identical to the
   * unidirectional path. `maxLightBounces` is uploaded as `uBdptMaxLightBounces`
   * (number of stored light-subpath vertices the connection sweep attempts). It is
   * capped to BDPT_MAX_LIGHT_BOUNCES by the engine.
   */
  readonly bdpt: boolean;
  readonly bdptMaxLightBounces: number;
  /** Spectral global Cauchy IOR dispersion coefficients (H2 follow-on); 0/0/0 = no dispersion. */
  readonly iorCauchy: readonly [number, number, number];
  /** Spectral global Jakob & Hanika reflectance coefficients (H2 follow-on); (0,0,0) = flat S≡½ no-op. */
  readonly jakobCoeffs: readonly [number, number, number];
  /** Thin-lens DoF PhysicalCamera uniforms (flag-plumbing audit); null = pinhole (FEATURE_DOF off). */
  readonly dof: {
    readonly focusDistance: number;
    readonly bokehSize: number;
    readonly apertureBlades: number;
    readonly apertureRotation: number;
    readonly anamorphicRatio: number;
  } | null;
  // ── Tonemap / present-pass dials (2026-06-10) ──────────────────────────
  // Wired from FrameQualitySettings.tonemap / .exposure / .outputColorSpace.
  // These are NOT used by the PT accumulation shader — they drive the separate
  // present pass (glResources #runPresentPass) that blits the HDR accum
  // texture to a tonemapped output.
  //
  // CONTRACT-DEFAULT TENSION: the contract (FrameQualitySettings) documents
  // default 'aces' @ 1.0 @ 'srgb'. pt-webgl2 previously had NO present pass,
  // so the raw linear HDR was returned as primaryRadiance. The new default
  // (aces+srgb) changes the visual output of any host that was receiving the
  // raw HDR. The walkaround backend had the same default and documents this in
  // HybridEngineFrameOrchestrator.ts:764 — here we match that behaviour.
  // Hosts that need the raw HDR should pass quality.tonemap='none' and
  // quality.outputColorSpace='linear'.
  /** Tonemap operator mode (matches TONEMAP_MODE_INDEX: 0=aces,1=agx,2=reinhard,3=linear,4=none). Default: 0 (aces). */
  readonly tonemapMode: number;
  /** Linear-exposure multiplier applied before tonemapping. Default: 1.0. */
  readonly exposure: number;
  /** Output color space: 0 = srgb (OETF applied, default), 1 = linear (OETF skipped). */
  readonly outputColorSpace: number;
}

/**
 * A5 — BDPT light-path ping-pong dimensions. The light-subpath kernel writes a
 * (BDPT_MAX_LIGHT_BOUNCES columns × 3 rows) RGBA32F texture: one column per light
 * bounce, three rows per vertex (pos|kind, normal|pdfFwd, throughput|pdfRev). The
 * width MUST stay 3 — the connection sweep caps the merged path at BDPT_MAX_MERGED
 * (=19) with `n = c + e + 3`, and the kernel comment fixes the layout at width 3.
 */
const BDPT_LIGHT_PATH_COLS = 3;
const BDPT_LIGHT_PATH_ROWS = 3;

/**
 * The Regime-2 composite quad fragment (verbatim port of the fork's BlendMaterial.js:31-59,
 * GL_FragColor → pc_fragColor). Lerps target1/target2 by `opacity = 1/(samples+1)` with
 * alpha-weighted compositing, written into the ping-pong pair.
 */
const BLEND_FRAG = `
in vec2 vUv;
uniform float opacity;
uniform sampler2D target1;
uniform sampler2D target2;
void main() {
  vec4 color1 = texture(target1, vUv);
  vec4 color2 = texture(target2, vUv);
  float invOpacity = 1.0 - opacity;
  float totalAlpha = color1.a * invOpacity + color2.a * opacity;
  if (color1.a != 0.0 || color2.a != 0.0) {
    pc_fragColor.rgb = color1.rgb * (invOpacity * color1.a / totalAlpha)
                     + color2.rgb * (opacity * color2.a / totalAlpha);
    pc_fragColor.a = totalAlpha;
  } else {
    pc_fragColor = vec4(0.0);
  }
}
`;

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
  // writing to the 8-bit output (the framebuffer is RGBA8 unorm, not auto-sRGB).
  // outputColorSpace 1 = linear — skip the OETF (useful for HDR/linear pipeline).
  if (uOutputColorSpace == 0) {
    pc_fragColor = vec4(vt_linearToSrgb(tonemapped), 1.0);
  } else {
    pc_fragColor = vec4(tonemapped, 1.0);
  }
}
`;
}

export class GlResources {
  readonly #gl: WebGL2RenderingContext;
  readonly #caps: GlCaps;
  /** Whether MRT aux g-buffers (gNormalDepth/gAlbedo) are allocated + written. */
  readonly #auxBuffers: boolean;
  /** OES_draw_buffers_indexed — per-draw-buffer blend state, so the colour blend runs
   *  on attachment 0 while the aux g-buffer attachments overwrite (no blend). null when
   *  unavailable (the g-buffer is then best-effort; the colour path is always correct). */
  readonly #drawBuffersIndexed: { disableiOES(target: number, index: number): void } | null;
  readonly #quad: FullscreenQuad;

  /** The PT accumulation target (RGBA32F primary + optional MRT aux). */
  #accum: RenderTarget | null = null;
  /** Regime-2 ping-pong blend pair — allocated lazily on first alpha-composite step. */
  #blend: [RenderTarget, RenderTarget] | null = null;
  /** Which slot of the blend pair currently holds the readable result. */
  #blendReadIndex = 0;

  #ptProgram: GlProgram | null = null;
  #blendProgram: GlProgram | null = null;

  // ── Present pass (tonemap / exposure / outputColorSpace) ────────────────────
  // A separate single-attachment RGBA8 render target + fullscreen quad program
  // that blits the HDR accum texture through the tonemap+OETF chain.
  // Allocated lazily on the first drawAccumStep call (ensured once per
  // ensureAccumResources cycle); destroyed/reallocated on resize.
  #presentTex: WebGLTexture | null = null;
  #presentFbo: WebGLFramebuffer | null = null;
  #presentProgram: GlProgram | null = null;

  #accumWidth = 0;
  #accumHeight = 0;
  /** Samples already accumulated since the last clearAccum() — drives opacity 1/(N+1). */
  #samples = 0;

  /**
   * A5 — BDPT light-path ping-pong pair (3×3 single-attachment RGBA32F). Bounce 0
   * writes column 0 (emitter vertex; reads nothing); bounce k reads column k-1 from
   * the "read" texture and writes column k to the "write" texture. The light-subpath
   * kernel `discard`s every column except `uBdptVertexCol`, so to keep already-built
   * columns alive across passes we blit the read texture into the write texture
   * before each column draw, then overwrite just that column. After the last bounce
   * the read texture holds all columns and is bound as `uBdptLightPathTex` for the
   * connection sweep. Allocated lazily only when bdpt is on (null otherwise → the
   * unidirectional path never touches this). */
  #bdptLightPath: [RenderTarget, RenderTarget] | null = null;
  /** Scratch FBO for the copy-source side of the per-column blit. */
  #bdptCopyFbo: WebGLFramebuffer | null = null;

  /**
   * @param supportsAuxBuffers host policy for the MRT g-buffers; intersected here with the
   * device's MAX_DRAW_BUFFERS so a 1-MRT device degrades to attachment-0-only (plan 02 §4).
   */
  constructor(gl: WebGL2RenderingContext, supportsAuxBuffers: boolean) {
    this.#gl = gl;
    this.#caps = probeGlCaps(gl);
    this.#auxBuffers = supportsAuxBuffers && this.#caps.maxDrawBuffers >= 3;
    const dbi = gl.getExtension('OES_draw_buffers_indexed') as { disableiOES?: unknown } | null;
    this.#drawBuffersIndexed =
      dbi != null && typeof dbi.disableiOES === 'function'
        ? (dbi as { disableiOES(target: number, index: number): void })
        : null;
    this.#quad = new FullscreenQuad(gl);
  }

  /** Probed device capabilities (regime selection, MRT/sampler budgets). */
  get caps(): GlCaps {
    return this.#caps;
  }

  /**
   * Ensure the accumulation resources match w×h, reallocating on a size change.
   * Returns `recreated` — true when targets were (re)built, so the caller resets its
   * sample counter (fork PathTracingRenderer.setSize → reset, :358-374).
   */
  ensureAccumResources(w: number, h: number): boolean {
    if (w === this.#accumWidth && h === this.#accumHeight && this.#accum != null) return false;
    this.#destroyTargets();
    this.#accum = createRenderTarget(this.#gl, w, h, this.#auxBuffers);
    // Present target — RGBA8 (sufficient for display output; the HDR lives in #accum).
    this.#destroyPresentTarget();
    this.#presentTex = createColorTexture(this.#gl, w, h);
    // RGBA8 FBO for the present-pass output.
    const fbo = this.#gl.createFramebuffer();
    if (fbo == null) throw new Error('pt-webgl2: failed to create present-pass FBO');
    this.#gl.bindFramebuffer(this.#gl.FRAMEBUFFER, fbo);
    this.#gl.framebufferTexture2D(this.#gl.FRAMEBUFFER, this.#gl.COLOR_ATTACHMENT0, this.#gl.TEXTURE_2D, this.#presentTex, 0);
    this.#gl.bindFramebuffer(this.#gl.FRAMEBUFFER, null);
    this.#presentFbo = fbo;
    this.#accumWidth = w;
    this.#accumHeight = h;
    this.#samples = 0;
    return true;
  }

  /** Build the PT program from `composeTraceGlsl(features)` once (idempotent). */
  ensureProgram(features: TraceFeatures): void {
    this.#ptProgram ??= new GlProgram(
      this.#gl,
      FULLSCREEN_VERT,
      composeTraceGlsl(features),
      featureDefines(features),
    );
  }

  /** The PT program (null before ensureProgram) — for the host to set uniforms/defines. */
  get ptProgram(): GlProgram | null {
    return this.#ptProgram;
  }

  /** Clear all accumulation targets to (0,0,0,0) + reset the sample counter (fork reset()). */
  clearAccum(): void {
    if (this.#accum != null) clearRenderTarget(this.#gl, this.#accum);
    if (this.#blend != null) {
      clearRenderTarget(this.#gl, this.#blend[0]);
      clearRenderTarget(this.#gl, this.#blend[1]);
    }
    this.#blendReadIndex = 0;
    this.#samples = 0;
  }

  /**
   * One accumulation step (plan 02 §6). Binds the accum FBO + MRT draw buffers, sets the
   * blend regime, uploads the running-average opacity + seed, binds the scene textures and
   * the params UBO, and draws the fullscreen triangle. For the alpha-composite regime it then
   * runs the BlendMaterial composite into the ping-pong pair.
   */
  drawAccumStep(
    scene: UploadedSceneTextures,
    regime: AccumRegime,
    seed: number,
    frame: FrameUniforms,
  ): void {
    const gl = this.#gl;
    if (this.#accum == null) throw new Error('pt-webgl2: drawAccumStep before ensureAccumResources');
    if (this.#ptProgram == null) throw new Error('pt-webgl2: drawAccumStep before ensureProgram');
    const prog = this.#ptProgram;

    // A5 — BDPT light subpath. Build the light-path vertex texture for THIS sample
    // (the subpath is reseeded per frame via the `seed`/rand bank) BEFORE the eye
    // pass, then bind it as `uBdptLightPathTex`. Only runs when bdpt is on; the
    // unidirectional path skips this entirely (byte-identical when bdpt:false).
    const bdptResult =
      frame.bdpt ? this.#buildBdptLightSubpath(prog, scene, seed, frame) : null;

    bindRenderTarget(gl, this.#accum);
    gl.viewport(0, 0, this.#accumWidth, this.#accumHeight);
    setBlendForRegime(gl, regime, this.#samples);
    // The MRT g-buffer (gNormalDepth@1, gAlbedo@2) holds sample-invariant primary-hit
    // data and must NOT accumulate: WebGL2 weights each draw buffer by its OWN output
    // alpha, and gNormalDepth packs linear depth (>1) in alpha, so the colour-blend
    // recurrence diverges → NaN. Disable blend on JUST the aux attachments (per-draw-buffer
    // state) so they overwrite (latest sample = correct, primary-hit is deterministic)
    // while the colour attachment keeps its EXACT running-average blend untouched. No
    // extension → the g-buffer is best-effort (the colour path is always correct).
    if (this.#auxBuffers && this.#drawBuffersIndexed != null) {
      this.#drawBuffersIndexed.disableiOES(gl.BLEND, 1);
      this.#drawBuffersIndexed.disableiOES(gl.BLEND, 2);
    }

    prog.use();
    // The copied fork GLSL reads INDIVIDUAL uniforms (no FrameParams UBO).
    prog.setInt('seed', seed);
    prog.setFloat('opacity', 1 / (this.#samples + 1));
    prog.setVec2('resolution', frame.resolution[0], frame.resolution[1]);
    prog.setInt('bounces', frame.bounces);
    prog.setInt('transmissiveBounces', frame.transmissiveBounces);
    prog.setFloat('filterGlossyFactor', frame.filterGlossyFactor);
    prog.setFloat('uRadianceClamp', frame.radianceClamp);
    prog.setMat4('cameraWorldMatrix', frame.cameraWorldMatrix);
    prog.setMat4('invProjectionMatrix', frame.invProjectionMatrix);
    prog.setFloat('environmentIntensity', frame.environmentIntensity);
    // H3 FIX (2026-06-09): upload backgroundAlpha. Directly-visible background
    // (NO_HIT first ray) sets `pc_fragColor.a = backgroundAlpha`, then the running
    // average multiplies by `opacity`; in the 'normal' regime the SRC_ALPHA blend
    // weights the fragment by that alpha. Never uploaded → defaulted to 0 → the
    // background contributed `src*0 + dst*1` every frame and NEVER accumulated
    // (directly-visible sky/HDRI rendered black). 1 = opaque (accumulates like
    // geometry); <1 routes to the alpha-composite regime (see #regime).
    prog.setFloat('backgroundAlpha', frame.backgroundAlpha);
    prog.setMat4('environmentRotation', frame.environmentRotation);
    prog.setInt('uSpectralRendering', frame.spectralEnabled ? 1 : 0);
    if (frame.spectralEnabled) {
      // H2 FIX (2026-06-09): upload the CIE CMF tables + CDFs + integrals. Before
      // this, GlProgram.setFloatArray had ZERO callers, so uCmfX/Y/Z, the three
      // CDFs, and the integrals all defaulted to 0 → wavelengthPdf=0 →
      // wavelengthToRGB() returned vec3(0) → `spectral: true` rendered BLACK.
      // Constant data, cheap re-upload; gated so non-spectral frames skip it.
      // (u_jakobCoeffs / iorCauchy stay at their flat-spectrum / no-dispersion
      // defaults — those refine spectral reflectance colour, not black-vs-lit.)
      prog.setFloatArray('uCmfX', CMF_X_F32);
      prog.setFloatArray('uCmfY', CMF_Y_F32);
      prog.setFloatArray('uCmfZ', CMF_Z_F32);
      prog.setFloatArray('uXCmfCdf', CMF_XCDF_F32);
      prog.setFloatArray('uYCmfCdf', CMF_YCDF_F32);
      prog.setFloatArray('uZCmfCdf', CMF_ZCDF_F32);
      prog.setFloat('uXCmfIntegral', X_CMF_INTEGRAL);
      prog.setFloat('uYCmfIntegral', Y_CMF_INTEGRAL);
      prog.setFloat('uZCmfIntegral', Z_CMF_INTEGRAL);
    }
    prog.setInt('uCausticStrategy', frame.causticStrategy);
    prog.setFloat('uMneeMaxIterations', frame.mneeMaxIterations);
    prog.setFloat('uMneeMaxChainLength', frame.mneeMaxChainLength);
    // H2 follow-on: scene-global spectral dispersion + reflectance coefficients.
    // Default (0,0,0)/(0,0,0) keep the no-dispersion / flat-S≡½ no-op path, so a
    // non-dispersive spectral frame is unchanged. Set unconditionally (cheap scalar
    // uploads; gated to nothing-but-defaults when the host supplies no dispersion).
    prog.setVec3('u_jakobCoeffs', frame.jakobCoeffs[0], frame.jakobCoeffs[1], frame.jakobCoeffs[2]);
    prog.setFloat('iorCauchyA', frame.iorCauchy[0]);
    prog.setFloat('iorCauchyB', frame.iorCauchy[1]);
    prog.setFloat('iorCauchyC', frame.iorCauchy[2]);
    // Flag-plumbing audit (2026-06-10): upload the PhysicalCamera DoF uniforms when
    // dof is enabled. The FEATURE_DOF GLSL gate is compiled in only when opts.dof was
    // set (see #traceFeatures), so these setters are inactive no-ops otherwise — but
    // we still gate the upload to skip the work for the common pinhole path.
    if (frame.dof != null) {
      prog.setFloat('physicalCamera.focusDistance', frame.dof.focusDistance);
      prog.setFloat('physicalCamera.bokehSize', frame.dof.bokehSize);
      prog.setInt('physicalCamera.apertureBlades', frame.dof.apertureBlades);
      prog.setFloat('physicalCamera.apertureRotation', frame.dof.apertureRotation);
      prog.setFloat('physicalCamera.anamorphicRatio', frame.dof.anamorphicRatio);
    }
    // A5 — eye pass: light-subpath pass OFF, bind the built light-path texture, and
    // upload the bounce count the connection sweep iterates. `#bindSceneTextures`
    // binds a dummy for `uBdptLightPathTex` when bdpt is off (or the build failed);
    // here we override it with the real result so the connection pass reads vertices.
    if (frame.bdpt) {
      prog.setInt('uBdptLightSubpathPass', 0);
      prog.setInt('uBdptMaxLightBounces', frame.bdptMaxLightBounces);
    }
    this.#bindSceneTextures(prog, scene, bdptResult);
    this.#quad.draw(gl);

    if (regime === 'alpha-composite') this.#compositeBlendStep();

    this.#samples += 1;

    // Present pass — blit the HDR accum through tonemap + OETF into #presentTex.
    this.#runPresentPass(frame.tonemapMode, frame.exposure, frame.outputColorSpace);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /**
   * The tonemapped present texture (the output of the most-recent present pass).
   * Always points to the presentTex (RGBA32F) written by #runPresentPass after
   * each drawAccumStep. When no present pass has run yet (before the first
   * drawAccumStep), returns the raw HDR accum as a fallback.
   *
   * NOTE: the present texture is the TONEMAPPED output.  Hosts that need the
   * raw HDR accumulation can read #accum.color directly (internal API only).
   */
  resultTexture(): WebGLTexture | null {
    // Prefer the present-pass output (tonemapped) when it has been allocated.
    if (this.#presentTex != null) return this.#presentTex;
    // Fallback: raw HDR accum (pre-present, e.g. before the first frame).
    if (this.#blend != null) {
      const [a, b] = this.#blend;
      return (this.#blendReadIndex === 0 ? a : b).color;
    }
    return this.#accum?.color ?? null;
  }

  /** The packed normal+depth g-buffer (null when MRT disabled). */
  get normalDepthTex(): WebGLTexture | null {
    return this.#accum?.normalDepth ?? null;
  }

  /** The albedo g-buffer (null when MRT disabled). */
  get albedoTex(): WebGLTexture | null {
    return this.#accum?.albedo ?? null;
  }

  dispose(): void {
    const gl = this.#gl;
    this.#destroyTargets();
    this.#destroyPresentTarget();
    this.#ptProgram?.dispose();
    this.#ptProgram = null;
    this.#blendProgram?.dispose();
    this.#blendProgram = null;
    this.#presentProgram?.dispose();
    this.#presentProgram = null;
    this.#quad.dispose(gl);
    // H7 FIX (2026-06-09): delete the lazily-allocated dummy textures — dispose()
    // freed the programs/targets/quad but LEAKED these two GPU textures on every
    // engine teardown (Canvas remount / route change churn would accumulate them).
    if (this.#dummy2dTex != null) { gl.deleteTexture(this.#dummy2dTex); this.#dummy2dTex = null; }
    if (this.#dummy2dArrTex != null) { gl.deleteTexture(this.#dummy2dArrTex); this.#dummy2dArrTex = null; }
    // A5 — free the BDPT light-path ping-pong pair + copy FBO.
    if (this.#bdptLightPath != null) {
      this.#bdptLightPath[0].destroy();
      this.#bdptLightPath[1].destroy();
      this.#bdptLightPath = null;
    }
    if (this.#bdptCopyFbo != null) { gl.deleteFramebuffer(this.#bdptCopyFbo); this.#bdptCopyFbo = null; }
  }

  // ----- internals -------------------------------------------------------------------------

  /** Composite the latest PT sample into the ping-pong pair (Regime 2; fork :156-165). */
  #compositeBlendStep(): void {
    const gl = this.#gl;
    const accum = this.#accum;
    if (accum == null) return;
    this.#ensureBlendPair();
    const blend = this.#blend;
    if (blend == null) return;
    const [slot0, slot1] = blend;
    const readTarget = this.#blendReadIndex === 0 ? slot0 : slot1;
    const writeTarget = this.#blendReadIndex === 0 ? slot1 : slot0;

    const blendProg = this.#ensureBlendProgram();
    bindRenderTarget(gl, writeTarget);
    gl.viewport(0, 0, this.#accumWidth, this.#accumHeight);
    gl.disable(gl.BLEND); // the composite math is done in-shader, not by fixed-function blend.

    blendProg.use();
    blendProg.setFloat('opacity', 1 / (this.#samples + 1));
    // target1 = prior accumulated result; target2 = the just-rendered PT sample.
    blendProg.bindTexture('target1', readTarget.color);
    blendProg.bindTexture('target2', accum.color);
    this.#quad.draw(gl);

    this.#blendReadIndex = 1 - this.#blendReadIndex;
  }

  #ensureBlendProgram(): GlProgram {
    this.#blendProgram ??= new GlProgram(this.#gl, FULLSCREEN_VERT, BLEND_FRAG, {});
    return this.#blendProgram;
  }

  /**
   * A5 — build the BDPT light subpath for this sample and return the texture holding
   * all light-path vertex columns (to be bound as `uBdptLightPathTex` for the eye
   * pass's connection sweep). Returns null when there is nothing to connect to (no
   * analytic lights) — the caller then leaves the dummy bound and the frame renders
   * unidirectionally.
   *
   * Per-column protocol (one fullscreen draw over a 3×3 viewport per bounce):
   *   read  = the texture holding columns < col already built this frame
   *   write = the other ping-pong slot
   *   1. blit read → write (copy already-built columns forward; the kernel `discard`s
   *      every column != uBdptVertexCol, so without this they'd be lost on the swap)
   *   2. set uBdptLightSubpathPass=1, uBdptVertexCol=col, uBdptMaxLightBounces
   *   3. bind read as uBdptLightPathTex (bounce k reads column k-1 from it)
   *   4. draw → write column `col` is overwritten with the new vertex
   *   5. swap read/write
   * After the loop, `read` holds all columns. Reading and writing the SAME texture in
   * one draw is a WebGL2 feedback loop (undefined), which the read≠write ping-pong +
   * pre-blit avoids.
   */
  #buildBdptLightSubpath(
    prog: GlProgram,
    scene: UploadedSceneTextures,
    seed: number,
    frame: FrameUniforms,
  ): WebGLTexture | null {
    if (scene.lightCount === 0) return null; // nothing to sample → unidirectional fallback
    const gl = this.#gl;
    this.#ensureBdptLightPath();
    const pair = this.#bdptLightPath;
    if (pair == null) return null;
    const copyFbo = this.#bdptCopyFbo;
    if (copyFbo == null) return null;

    const cols = Math.max(1, Math.min(frame.bdptMaxLightBounces, BDPT_LIGHT_PATH_COLS));

    // Clear both slots so unbuilt columns read as (0,0,0,0); column 0 row 0 .w==0 is
    // BDPT_KIND_LIGHT, so an all-zero column is NOT auto-invalid — but the kernel only
    // ever connects to columns it actually wrote, and the connection sweep iterates
    // [0, uBdptMaxLightBounces); a failed sample writes BDPT_KIND_INVALID (.w==3)
    // explicitly. Clearing keeps stale prior-frame columns out.
    clearRenderTarget(gl, pair[0]);
    clearRenderTarget(gl, pair[1]);

    prog.use();
    // The light-subpath pass shares the eye program; flip the pass flag + upload the
    // per-pass scalars. The scene textures (BVH/materials/lights) are bound below.
    prog.setInt('seed', seed);
    prog.setInt('uBdptLightSubpathPass', 1);
    prog.setInt('uBdptMaxLightBounces', cols);
    // The light subpath traces scene rays → needs the same per-frame transforms the
    // eye pass reads (lightsDenom uses environmentIntensity; traceScene reads none of
    // the camera matrices but initRenderState / fog do touch a few). Upload the load-
    // bearing ones; the kernel ignores the rest in the subpath branch.
    prog.setVec2('resolution', BDPT_LIGHT_PATH_COLS, BDPT_LIGHT_PATH_ROWS);
    prog.setInt('bounces', frame.bounces);
    prog.setInt('transmissiveBounces', frame.transmissiveBounces);
    prog.setFloat('environmentIntensity', frame.environmentIntensity);
    prog.setMat4('environmentRotation', frame.environmentRotation);
    prog.setMat4('cameraWorldMatrix', frame.cameraWorldMatrix);
    prog.setMat4('invProjectionMatrix', frame.invProjectionMatrix);

    gl.disable(gl.BLEND); // vertex writes overwrite; no accumulation in the subpath.

    let readIdx = 0;
    for (let col = 0; col < cols; col += 1) {
      const read = pair[readIdx]!;
      const write = pair[1 - readIdx]!;

      // 1. Copy already-built columns (< col) read → write so they survive the swap.
      if (col > 0) this.#blitBdpt(read, write, copyFbo);

      // 2/3. Per-column scalars + the read texture as uBdptLightPathTex.
      prog.setInt('uBdptVertexCol', col);
      // Bind scene textures with the read slot as the light-path source. (For col 0
      // the kernel ignores the texture; binding the read slot is harmless.)
      this.#bindSceneTextures(prog, scene, read.color);

      // 4. Draw the 3×3 viewport into the write slot.
      bindRenderTarget(gl, write);
      gl.viewport(0, 0, BDPT_LIGHT_PATH_COLS, BDPT_LIGHT_PATH_ROWS);
      this.#quad.draw(gl);

      // 5. Swap — `write` now holds columns ≤ col.
      readIdx = 1 - readIdx;
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    // After the loop, pair[readIdx] is the most-recently-written slot (all columns).
    return pair[readIdx]!.color;
  }

  /** A5 — copy `src` color into `dst` color via a framebuffer blit (preserve built columns). */
  #blitBdpt(src: RenderTarget, dst: RenderTarget, copyFbo: WebGLFramebuffer): void {
    const gl = this.#gl;
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, copyFbo);
    gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, src.color, 0);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, dst.fbo);
    gl.blitFramebuffer(
      0, 0, BDPT_LIGHT_PATH_COLS, BDPT_LIGHT_PATH_ROWS,
      0, 0, BDPT_LIGHT_PATH_COLS, BDPT_LIGHT_PATH_ROWS,
      gl.COLOR_BUFFER_BIT, gl.NEAREST,
    );
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
  }

  /** A5 — lazily allocate the BDPT light-path ping-pong pair + copy FBO. */
  #ensureBdptLightPath(): void {
    if (this.#bdptLightPath != null) return;
    const gl = this.#gl;
    this.#bdptLightPath = [
      createRenderTarget(gl, BDPT_LIGHT_PATH_COLS, BDPT_LIGHT_PATH_ROWS, false),
      createRenderTarget(gl, BDPT_LIGHT_PATH_COLS, BDPT_LIGHT_PATH_ROWS, false),
    ];
    this.#bdptCopyFbo = gl.createFramebuffer();
    if (this.#bdptCopyFbo == null) throw new Error('pt-webgl2: failed to create BDPT copy FBO');
  }

  #ensureBlendPair(): void {
    if (this.#blend != null) return;
    this.#blend = [
      createRenderTarget(this.#gl, this.#accumWidth, this.#accumHeight, false),
      createRenderTarget(this.#gl, this.#accumWidth, this.#accumHeight, false),
    ];
    clearRenderTarget(this.#gl, this.#blend[0]);
    clearRenderTarget(this.#gl, this.#blend[1]);
    this.#blendReadIndex = 0;
  }

  /** Bind the scene texture bundle to the PT program's samplers (plan 04 §4 binding remap).
   *  `bdptLightPath` (A5) overrides the `uBdptLightPathTex` dummy with the built light-path
   *  vertex texture when BDPT is driven; null keeps the dummy (bdpt off / build failed). */
  #bindSceneTextures(
    prog: GlProgram,
    scene: UploadedSceneTextures,
    bdptLightPath: WebGLTexture | null = null,
  ): void {
    const gl = this.#gl;
    // BVH struct samplers (BVH { usampler2D index; sampler2D position; sampler2D bvhBounds;
    // usampler2D bvhContents; } — bvh_struct_definitions.glsl).
    prog.bindTexture('bvh.index', scene.bvhIndex);
    prog.bindTexture('bvh.position', scene.bvhPosition);
    prog.bindTexture('bvh.bvhBounds', scene.bvhBounds);
    prog.bindTexture('bvh.bvhContents', scene.bvhContents);
    prog.bindTexture('materialIndexAttribute', scene.materialIndex);
    prog.bindTexture('materials', scene.materials);
    prog.bindTexture('attributesArray', scene.attributesArray, gl.TEXTURE_2D_ARRAY);
    prog.bindTexture('lights.tex', scene.lights); // LightsInfo { sampler2D tex; uint count; }
    // H1 FIX (2026-06-09): upload the `lights.count` uint — without it the field
    // defaults to 0u and the ENTIRE analytic-light system is inert (the NEE gate
    // `rand(5) < count/lightsDenom`, the forward light-hit loop `i < lights.count`,
    // and the BDPT light-subpath all see zero lights). Only `mesh-area` emitters
    // lit anything, via the emissive fold. This restores point/spot/rect-area/
    // circ-area/directional NEE. The packed lights texture was already uploaded;
    // only its count uniform was missing.
    prog.setUint('lights.count', scene.lightCount);
    // B4 — mesh-area triangle lights (NEE). Bind the tri-light texture (dummy when
    // the scene has none, so the sampler stays valid) + the count + Σ-area scalars
    // the GLSL mesh-NEE branch and forward-hit MIS read. count==0 → both inert.
    prog.bindTexture('uMeshLights', scene.meshLights ?? this.#dummyTex2D());
    prog.setUint('uMeshLightCount', scene.meshLightCount);
    prog.setFloat('uTotalEmissiveArea', scene.totalEmissiveArea);
    // Every OPTIONAL sampler the fork GLSL declares must reference a valid texture of
    // the matching type — an unbound sampler defaults to unit 0 and collides with a
    // different-typed sampler there (GL_INVALID_OPERATION → black). bindTexture no-ops
    // for inactive samplers, so binding a dummy where there's no scene data is safe.
    const d2d = this.#dummyTex2D();
    const d2a = this.#dummyTex2DArray();
    prog.bindTexture('iesProfiles', scene.iesProfiles ?? d2a, gl.TEXTURE_2D_ARRAY);
    prog.bindTexture('textures', scene.textures2DArray ?? d2a, gl.TEXTURE_2D_ARRAY);
    prog.bindTexture('backgroundMap', d2d);
    prog.bindTexture('sobolTexture', d2d);
    prog.bindTexture('stratifiedTexture', d2d);
    prog.bindTexture('stratifiedOffsetTexture', d2d);
    // H5 FIX (2026-06-09): when FEATURE_BDPT is compiled in (`bdpt: true`), the GLSL
    // declares `uniform sampler2D uBdptLightPathTex` but it was MISSING from this
    // dummy list — so it stayed unbound, defaulted to unit 0, and collided with the
    // usampler there → GL_INVALID_OPERATION → black frame (exactly the failure mode
    // this block's comment warns about). Bind a dummy (no-op when bdpt is off, since
    // bindTexture skips inactive samplers). NOTE: BDPT is not yet host-driven (the
    // light-subpath passes are never issued — see index.ts), so with this bound the
    // frame renders unidirectionally rather than crashing; full BDPT orchestration is
    // tracked as a feature in items_to_fix §H5.
    // A5 (2026-06-10): BDPT is now host-driven — when a light-path texture was built
    // this frame we bind it here so the connection sweep reads real light vertices.
    // bdptLightPath is null when bdpt is off (or the per-frame build short-circuited,
    // e.g. no lights), so the dummy keeps the sampler valid and the connection sweep
    // sees only BDPT_KIND_INVALID vertices → adds nothing (unidirectional fallback).
    prog.bindTexture('uBdptLightPathTex', bdptLightPath ?? d2d);
    // EquirectHdrInfo { sampler2D marginalWeights; sampler2D conditionalWeights; sampler2D map; float totalSum; }
    prog.bindTexture('envMapInfo.map', scene.envMap ?? d2d);
    prog.bindTexture('envMapInfo.marginalWeights', scene.envMarginal ?? d2d);
    prog.bindTexture('envMapInfo.conditionalWeights', scene.envConditional ?? d2d);
    // The env-sampling GLSL early-outs on `envMapInfo.totalSum == 0.0`; bind the
    // scalar so a present environment is actually sampled (0 when absent → correct
    // no-env early-out). Without this the env textures upload but never light.
    prog.setFloat('envMapInfo.totalSum', scene.envTotalSum);
  }

  #dummy2dTex: WebGLTexture | null = null;
  #dummy2dArrTex: WebGLTexture | null = null;

  #dummyTex2D(): WebGLTexture {
    const gl = this.#gl;
    if (this.#dummy2dTex == null) {
      const t = gl.createTexture();
      if (t == null) throw new Error('pt-webgl2: failed to create dummy 2D texture');
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      this.#dummy2dTex = t;
    }
    return this.#dummy2dTex;
  }

  #dummyTex2DArray(): WebGLTexture {
    const gl = this.#gl;
    if (this.#dummy2dArrTex == null) {
      const t = gl.createTexture();
      if (t == null) throw new Error('pt-webgl2: failed to create dummy 2D-array texture');
      gl.bindTexture(gl.TEXTURE_2D_ARRAY, t);
      gl.texImage3D(gl.TEXTURE_2D_ARRAY, 0, gl.RGBA, 1, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));
      gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      this.#dummy2dArrTex = t;
    }
    return this.#dummy2dArrTex;
  }

  #destroyTargets(): void {
    this.#accum?.destroy();
    this.#accum = null;
    if (this.#blend != null) {
      this.#blend[0].destroy();
      this.#blend[1].destroy();
      this.#blend = null;
    }
    this.#accumWidth = 0;
    this.#accumHeight = 0;
    this.#blendReadIndex = 0;
  }

  #destroyPresentTarget(): void {
    const gl = this.#gl;
    if (this.#presentTex != null) { gl.deleteTexture(this.#presentTex); this.#presentTex = null; }
    if (this.#presentFbo != null) { gl.deleteFramebuffer(this.#presentFbo); this.#presentFbo = null; }
  }

  #ensurePresentProgram(): GlProgram {
    if (this.#presentProgram == null) {
      const tonemapGlsl = (TonemapFunctions as Record<string, unknown>)['tonemap_functions'];
      if (typeof tonemapGlsl !== 'string') throw new Error('pt-webgl2: tonemap_functions GLSL not found');
      this.#presentProgram = new GlProgram(
        this.#gl,
        FULLSCREEN_VERT,
        buildPresentFragBody(tonemapGlsl),
        {}, // no compile-time defines; all dials are uniforms
      );
    }
    return this.#presentProgram;
  }

  /**
   * Run the present pass: blit the current HDR accumulation (or ping-pong blend
   * result) through the tonemap + OETF chain into #presentTex.
   *
   * Called once per drawAccumStep, after the PT accumulation draw and the
   * optional alpha-composite step. The present target is already allocated by
   * ensureAccumResources so this is a no-alloc hot path.
   *
   * Default dials match the contract (FrameQualitySettings) defaults and the
   * walkaround-hybrid orchestrator (HybridEngineFrameOrchestrator.ts:764):
   *   tonemapMode = 0 (aces), exposure = 1.0, outputColorSpace = 0 (srgb).
   */
  #runPresentPass(tonemapMode: number, exposure: number, outputColorSpace: number): void {
    const gl = this.#gl;
    const presentFbo = this.#presentFbo;
    if (presentFbo == null) return;

    // The HDR source is the ping-pong read slot (Regime 2) or the primary accum.
    let srcTex: WebGLTexture | null = null;
    if (this.#blend != null) {
      const [a, b] = this.#blend;
      srcTex = (this.#blendReadIndex === 0 ? a : b).color;
    } else {
      srcTex = this.#accum?.color ?? null;
    }
    if (srcTex == null) return;

    const prog = this.#ensurePresentProgram();
    gl.bindFramebuffer(gl.FRAMEBUFFER, presentFbo);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
    gl.viewport(0, 0, this.#accumWidth, this.#accumHeight);
    gl.disable(gl.BLEND); // no blending in the present pass — overwrite only

    prog.use();
    prog.bindTexture('uAccumTex', srcTex);
    prog.setInt('uTonemapMode', tonemapMode);
    prog.setFloat('uExposure', exposure);
    prog.setInt('uOutputColorSpace', outputColorSpace);
    this.#quad.draw(gl);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }
}
