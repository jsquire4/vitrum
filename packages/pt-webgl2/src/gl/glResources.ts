// GlResources — the GL resource owner (plan/three-removal/02-gl-framework.md §5, §6).
// Analog of pt-webgpu's GpuResources / the fork's PathTracingRenderer state. It owns:
//   - the accumulation FBO + RGBA32F color texture (+ optional MRT gNormalDepth/gAlbedo),
//   - the Regime-2 ping-pong blend pair (allocated lazily only when needed),
//   - the PT GlProgram (built from composeTraceGlsl + FULLSCREEN_VERT),
//   - the Regime-2 BlendMaterial composite quad program,
//   - a FullscreenQuad.
//
// The per-sample draw (drawAccumStep) replaces the fork's renderTask generator: bind the
// accum FBO + MRT draw buffers, set the blend regime, bind scene textures + uniforms, and
// draw the fullscreen triangle (fork PathTracingRenderer.js:144-167, §6 of plan 02).
//
// D10.1 (2026-06-10): BDPT light-subpath machinery extracted to BdptSubpathBuilder.ts;
// present-pass (tonemap/exposure/outputColorSpace) extracted to PresentPass.ts.

import type { TraceFeatures, AccumRegime } from '../featureTypes.js';
import { featureDefines } from '../featureTypes.js';
import type { UploadedSceneTextures } from '../scene/sceneTextures.js';
import { composeTraceGlsl } from '../glsl/composeTraceGlsl.js';
import { GlProgram } from './glProgram.js';
import { FullscreenQuad, FULLSCREEN_VERT } from './fullscreenQuad.js';
import {
  createRenderTarget,
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
  SOBOL_TEXTURE_SIZE,
  generateSobolTextureData,
} from '@vitrum/shared-samplers';
import { BdptSubpathBuilder } from './BdptSubpathBuilder.js';
import { PresentPass } from './PresentPass.js';
import {
  readOidnInputsFromWebGlFbos,
  type WebGlOidnReadbackResult,
} from '../denoise/rgba32fReadback.js';

// ── D10.2: SCENE_TEXTURE_BINDINGS table ──────────────────────────────────────
// Typed table that drives #bindSceneTextures. Each entry describes:
//   name      — GLSL uniform name (as passed to GlProgram.bindTexture)
//   kind      — 'tex2d' | 'tex2dArray' (determines gl.TEXTURE_2D vs gl.TEXTURE_2D_ARRAY)
//   source    — accessor returning the WebGLTexture to bind, or null for a dummy
//
// TABLE ORDER IS LOAD-BEARING: GlProgram assigns sampler units in the order
// uniforms are bound at link time. The mock-GL tests pin sampler-unit assignment
// by this order; reordering entries breaks those tests without a golden re-pin.
//
// Non-texture uniforms (lights.count, uMeshLightCount, uTotalEmissiveArea,
// uTotalEmissivePower, envMapInfo.totalSum) are uploaded separately after the
// loop — they are scalar setters, not texture binds.
//
// A5 override: uBdptLightPathTex is included in the table (bound with a dummy
// here so the sampler unit is registered), then overridden AFTER the loop with
// the real light-path texture when BDPT is active. See #bindSceneTextures for
// the override comment.

type SceneTexKind = 'tex2d' | 'tex2dArray';

interface SceneTextureBinding {
  readonly name: string;
  readonly kind: SceneTexKind;
  /**
   * Accessor that returns the WebGLTexture for this slot given the scene bundle
   * and the lazily-allocated dummy textures (d2d = 2D dummy, d2a = 2DArray dummy).
   * Return null to fall through to the appropriate dummy for this kind.
   */
  readonly source: (scene: UploadedSceneTextures, d2d: WebGLTexture, d2a: WebGLTexture) => WebGLTexture;
}

const SCENE_TEXTURE_BINDINGS: readonly SceneTextureBinding[] = [
  // BVH struct samplers (bvh_struct_definitions.glsl: BVH { usampler2D index; sampler2D position; ... })
  { name: 'bvh.index',                  kind: 'tex2d',      source: (s) => s.bvhIndex },
  { name: 'bvh.position',               kind: 'tex2d',      source: (s) => s.bvhPosition },
  { name: 'bvh.bvhBounds',              kind: 'tex2d',      source: (s) => s.bvhBounds },
  { name: 'bvh.bvhContents',            kind: 'tex2d',      source: (s) => s.bvhContents },
  // Per-triangle / per-material tables
  { name: 'materialIndexAttribute',     kind: 'tex2d',      source: (s) => s.materialIndex },
  { name: 'materials',                  kind: 'tex2d',      source: (s) => s.materials },
  // Vertex attribute array (normal / tangent / uv / color layers)
  { name: 'attributesArray',            kind: 'tex2dArray', source: (s) => s.attributesArray },
  // Analytic lights (LightsInfo { sampler2D tex; uint count; })
  { name: 'lights.tex',                 kind: 'tex2d',      source: (s) => s.lights },
  // B4 — mesh-area triangle lights (dummy when scene has none → inert branch)
  { name: 'uMeshLights',               kind: 'tex2d',      source: (s, d2d) => s.meshLights ?? d2d },
  // Optional samplers the fork GLSL declares — must be bound to a valid texture of
  // the matching type to avoid unit-0 collision (GL_INVALID_OPERATION → black).
  // bindTexture no-ops for inactive samplers so the dummy binds are always safe.
  { name: 'textures',                   kind: 'tex2dArray', source: (s, _d2d, d2a) => s.textures2DArray ?? d2a },
  // iesProfiles removed — IES profiles not in @vitrum/core contract (item 20).
  { name: 'backgroundMap',              kind: 'tex2d',      source: (_s, d2d) => d2d },
  { name: 'sobolTexture',               kind: 'tex2d',      source: (_s, d2d) => d2d },
  { name: 'stratifiedTexture',          kind: 'tex2d',      source: (_s, d2d) => d2d },
  { name: 'stratifiedOffsetTexture',    kind: 'tex2d',      source: (_s, d2d) => d2d },
  // A5 BDPT light-path texture (dummy here; overridden after the loop when bdpt is active).
  // NOTE: unit-0 collision warning — this MUST appear in the table so the sampler
  // unit is registered at link time.  The after-loop override replaces the dummy with
  // the real light-path texture when FEATURE_BDPT is compiled in and bdpt is active.
  { name: 'uBdptLightPathTex',          kind: 'tex2d',      source: (_s, d2d) => d2d },
  // EquirectHdrInfo importance-sampling samplers
  { name: 'envMapInfo.map',             kind: 'tex2d',      source: (s, d2d) => s.envMap ?? d2d },
  { name: 'envMapInfo.marginalWeights', kind: 'tex2d',      source: (s, d2d) => s.envMarginal ?? d2d },
  { name: 'envMapInfo.conditionalWeights', kind: 'tex2d',   source: (s, d2d) => s.envConditional ?? d2d },
] as const;

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
  readonly materialLodDepth: number;
  readonly radianceClamp: number;
  readonly cameraWorldMatrix: Float32Array; // inverse(viewMatrix)
  readonly invProjectionMatrix: Float32Array; // inverse(projMatrix)
  readonly environmentIntensity: number;
  readonly environmentRotation: Float32Array; // mat4
  readonly backgroundBlur: number;
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
  // present pass (PresentPass) that blits the HDR accum texture to a tonemapped
  // output.
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
  #randomType: 0 | 1 | 2 = 0;

  // ── Present pass (D10.1: extracted to PresentPass) ────────────────────────
  readonly #presentPass: PresentPass;

  #accumWidth = 0;
  #accumHeight = 0;
  /** Samples already accumulated since the last clearAccum() — drives opacity 1/(N+1). */
  #samples = 0;

  // ── BDPT light-subpath (D10.1: extracted to BdptSubpathBuilder) ───────────
  readonly #bdptBuilder: BdptSubpathBuilder;

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
    this.#presentPass = new PresentPass(gl, this.#quad);
    this.#bdptBuilder = new BdptSubpathBuilder(gl);
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
    // Present target — RGBA32F (deliberate: it is the public primaryRadiance
    // and must stay FLOAT-readable; see createPresentTexture).
    this.#presentPass.allocate(w, h);
    this.#accumWidth = w;
    this.#accumHeight = h;
    this.#samples = 0;
    return true;
  }

  /** Build the PT program from `composeTraceGlsl(features)` once (idempotent). */
  ensureProgram(features: TraceFeatures): void {
    if (this.#ptProgram == null) {
      this.#randomType = features.randomType;
      this.#ptProgram = new GlProgram(
        this.#gl,
        FULLSCREEN_VERT,
        composeTraceGlsl(features),
        featureDefines(features),
      );
    }
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
      frame.bdpt
        ? this.#bdptBuilder.build(
            prog,
            scene,
            seed,
            frame,
            (p, s, t) => this.#bindSceneTextures(p, s, t),
            () => this.#quad.draw(gl),
          )
        : null;

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
    prog.setInt('materialLodDepth', frame.materialLodDepth);
    prog.setFloat('uRadianceClamp', frame.radianceClamp);
    prog.setMat4('cameraWorldMatrix', frame.cameraWorldMatrix);
    prog.setMat4('invProjectionMatrix', frame.invProjectionMatrix);
    prog.setFloat('environmentIntensity', frame.environmentIntensity);
    prog.setFloat('backgroundBlur', frame.backgroundBlur);
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

    // Present pass — blit the HDR accum through tonemap + OETF into the present target.
    let srcTex: WebGLTexture | null = null;
    if (this.#blend != null) {
      const [a, b] = this.#blend;
      srcTex = (this.#blendReadIndex === 0 ? a : b).color;
    } else {
      srcTex = this.#accum?.color ?? null;
    }
    if (srcTex != null) {
      this.#presentPass.run(srcTex, this.#accumWidth, this.#accumHeight, frame.tonemapMode, frame.exposure, frame.outputColorSpace);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /** Current accumulation dimensions (0×0 before first ensureAccumResources). */
  get accumDims(): { readonly width: number; readonly height: number } {
    return { width: this.#accumWidth, height: this.#accumHeight };
  }

  /**
   * The tonemapped present texture (the output of the most-recent present pass).
   * Always points to the presentTex (RGBA32F) written by PresentPass after
   * each drawAccumStep. When no present pass has run yet (before the first
   * drawAccumStep), returns the raw HDR accum as a fallback.
   *
   * NOTE: the present texture is the TONEMAPPED output.  Hosts that need the
   * raw HDR accumulation can read #accum.color directly (internal API only).
   */
  resultTexture(): WebGLTexture | null {
    // Prefer the present-pass output (tonemapped) when it has been allocated.
    if (this.#presentPass.tex != null) return this.#presentPass.tex;
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

  /**
   * Read the linear-HDR accumulator plus optional MRT aux attachments into the
   * CPU RGB layout consumed by the shared OIDN final-pass dispatcher.
   */
  readOidnInputsRgba32f(): WebGlOidnReadbackResult | null {
    const w = this.#accumWidth;
    const h = this.#accumHeight;
    if (w <= 0 || h <= 0) return null;
    const colorFbo = this.#linearReadFbo();
    const accum = this.#accum;
    const auxFbo = accum?.fbo ?? null;
    const hasAux = this.#auxBuffers && accum?.normalDepth != null && accum.albedo != null;
    return readOidnInputsFromWebGlFbos(this.#gl, {
      colorFbo,
      auxFbo,
      width: w,
      height: h,
      ...(hasAux
        ? {
            normalDepthAttachment: this.#gl.COLOR_ATTACHMENT1,
            albedoAttachment: this.#gl.COLOR_ATTACHMENT2,
          }
        : {}),
    });
  }

  dispose(): void {
    const gl = this.#gl;
    this.#destroyTargets();
    this.#presentPass.destroy();
    this.#presentPass.disposeProgram();
    this.#ptProgram?.dispose();
    this.#ptProgram = null;
    this.#blendProgram?.dispose();
    this.#blendProgram = null;
    this.#quad.dispose(gl);
    // H7 FIX (2026-06-09): delete the lazily-allocated dummy textures — dispose()
    // freed the programs/targets/quad but LEAKED these two GPU textures on every
    // engine teardown (Canvas remount / route change churn would accumulate them).
    if (this.#dummy2dTex != null) { gl.deleteTexture(this.#dummy2dTex); this.#dummy2dTex = null; }
    if (this.#dummy2dArrTex != null) { gl.deleteTexture(this.#dummy2dArrTex); this.#dummy2dArrTex = null; }
    if (this.#sobolTex != null) { gl.deleteTexture(this.#sobolTex); this.#sobolTex = null; }
    // A5 — free the BDPT light-path ping-pong pair + copy FBO (D10.1: via BdptSubpathBuilder).
    this.#bdptBuilder.dispose();
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
    // Dummies are created BEFORE any prog.bindTexture call — deliberately.
    // The factory does a raw gl.bindTexture on whatever texture unit is active;
    // the pre-refactor code evaluated `this.#dummyTex2D()` inline as a bind
    // argument, so on the FIRST frame the dummy creation clobbered the
    // still-active `lights.tex` unit with the 1×1 black dummy — sample 1 of
    // every accumulation rendered with a black lights texture (a permanent
    // ~1/spp under-bias of the direct-light term in the accumulated mean).
    // Found 2026-06-11 via the run-ptwebgl2-h1 GPU A/B when this table landed
    // (lit meanLum 0.26176 → 0.26931 at 24 spp = the corrected first sample).
    const d2d = this.#dummyTex2D();
    const d2a = this.#dummyTex2DArray();

    // D10.2: table-driven texture binding. Loop order matches the original hand-written
    // call order — GlProgram assigns sampler units in bind order at link time, so the
    // mock-GL tests' sampler-unit pin relies on this sequence being preserved.
    for (const binding of SCENE_TEXTURE_BINDINGS) {
      const tex = binding.source(scene, d2d, d2a);
      const target = binding.kind === 'tex2dArray' ? gl.TEXTURE_2D_ARRAY : gl.TEXTURE_2D;
      prog.bindTexture(binding.name, tex, target);
    }

    // A5 override (AFTER the table loop): replace the uBdptLightPathTex dummy with the
    // real light-path texture when BDPT was active this frame. The table loop bound the
    // dummy first so the sampler unit is registered; this second bind updates the active
    // unit to point at the real texture. bdptLightPath === null → dummy stays (unidirectional
    // fallback; the connection sweep sees only BDPT_KIND_INVALID vertices → adds nothing).
    if (bdptLightPath != null) {
      prog.bindTexture('uBdptLightPathTex', bdptLightPath);
    }
    if (this.#randomType === 1) {
      prog.bindTexture('sobolTexture', this.#sobolTex2D());
    }

    // Non-texture scalar uniforms — uploaded after the texture loop.
    // H1 FIX: lights.count must be set as uint (setUint) — without it the analytic-light
    // system is inert (count defaults to 0u, NEE gate and forward light loop both see 0).
    prog.setUint('lights.count', scene.lightCount);
    // B4: mesh-area NEE scalars (count==0 → inert branch, byte-identical to no-mesh-light).
    prog.setUint('uMeshLightCount', scene.meshLightCount);
    prog.setFloat('uTotalEmissiveArea', scene.totalEmissiveArea);
    prog.setFloat('uTotalEmissivePower', scene.totalEmissivePower);
    // Env-map importance-sampling total sum (0 → correct no-env early-out in GLSL).
    prog.setFloat('envMapInfo.totalSum', scene.envTotalSum);
  }

  #dummy2dTex: WebGLTexture | null = null;
  #dummy2dArrTex: WebGLTexture | null = null;
  #sobolTex: WebGLTexture | null = null;

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

  #sobolTex2D(): WebGLTexture {
    const gl = this.#gl;
    if (this.#sobolTex == null) {
      const t = gl.createTexture();
      if (t == null) throw new Error('pt-webgl2: failed to create Sobol texture');
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA32F,
        SOBOL_TEXTURE_SIZE,
        SOBOL_TEXTURE_SIZE,
        0,
        gl.RGBA,
        gl.FLOAT,
        generateSobolTextureData(),
      );
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      this.#sobolTex = t;
    }
    return this.#sobolTex;
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

  #linearReadFbo(): WebGLFramebuffer | null {
    if (this.#blend != null) {
      const [a, b] = this.#blend;
      return (this.#blendReadIndex === 0 ? a : b).fbo;
    }
    return this.#accum?.fbo ?? null;
  }

  /**
   * CPU readback of the HDR accumulation or present FBO, row-flipped to
   * top-left origin (WebGL uses bottom-left, so row 0 in `readPixels` is the
   * BOTTOM of the image).
   *
   * `source:'linear'` reads the RGBA32F accumulation FBO — the running mean of
   * all accumulated path-trace samples (linear-light HDR, scene radiance units).
   * EXT_color_buffer_float is required (enforced by resolveWebGl2TraceTier; the
   * engine never reaches this point without it).
   *
   * `source:'output'` reads the present FBO — the RGBA32F tonemapped output
   * written by PresentPass. DELIBERATELY RGBA32F: the present texture is the
   * public `primaryRadiance`, and hosts/harnesses read it with FLOAT
   * readPixels (see createPresentTexture's format note).
   *
   * Returns `null` when the requested FBO has not been allocated yet (before
   * the first frame).
   *
   * NOTE: readPixelsRgba32f has access to #blend/#accum/#presentPass.fbo and
   * cannot be extracted as a free function without passing those in — left here
   * as a method (D10.1 analysis: too coupled, leaving and reporting).
   */
  readPixelsRgba32f(source: 'linear' | 'output'): Float32Array | null {
    const w = this.#accumWidth;
    const h = this.#accumHeight;
    if (w <= 0 || h <= 0) return null;
    const gl = this.#gl;

    const fbo = source === 'output' ? this.#presentPass.fbo : this.#linearReadFbo();
    if (fbo == null) return null;

    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    // Both targets are RGBA32F (accum AND present — see createPresentTexture),
    // so both read with the FLOAT path.
    const pixels = new Float32Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, pixels);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    // WebGL readPixels writes bottom-left origin (row 0 = bottom).
    // Flip vertically so row 0 = top (the captureFrame contract: top-left origin).
    const rowBytes = w * 4;
    const tmp = new Float32Array(rowBytes);
    for (let top = 0, bot = h - 1; top < bot; top++, bot--) {
      const topOff = top * rowBytes;
      const botOff = bot * rowBytes;
      tmp.set(pixels.subarray(topOff, topOff + rowBytes));
      pixels.copyWithin(topOff, botOff, botOff + rowBytes);
      pixels.set(tmp, botOff);
    }
    return pixels;
  }
}
