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
  SOBOL_TEXTURE_SIZE,
  generateSobolTextureData,
} from '@vitrum/shared-samplers';
import { BdptSubpathBuilder } from './BdptSubpathBuilder.js';
import { PresentPass } from './PresentPass.js';
import { uploadFrameUniforms } from './uploadFrameUniforms.js';
import { BLEND_FRAG } from './blendFrag.js';
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
// TABLE ORDER IS LOAD-BEARING: GlProgram assigns sampler units by walking the
// linked program's active uniforms (getActiveUniform / ACTIVE_UNIFORMS), and this
// table must bind each sampler exactly once so every slot gets a distinct unit.
// The pre-push T1 GPU smoke compiles/links the real pass graph and is the true
// guard for sampler-unit order — the mock-GL suite does NOT exercise
// assignSamplerUnits (createMockGl.getActiveUniform returns null, so the sampler
// walk is skipped and mock sampler binds no-op). Do not rely on a mock-GL test to
// catch a reorder here; a reorder is caught by the T1 smoke, not vitest.
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

    // Upload the per-frame individual uniforms (D11-6: extracted to
    // uploadFrameUniforms). `prog.use()` + the whole setter sequence lives there,
    // byte-identical to the old inline body.
    uploadFrameUniforms(prog, this.#samples, seed, frame);
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
    // call order. GlProgram assigns sampler units by walking the linked program's active
    // uniforms; each sampler must be bound exactly once here so it gets a distinct unit.
    // The T1 GPU smoke links the real program and is the true sampler-order guard; the
    // mock-GL suite does not exercise assignSamplerUnits (mock getActiveUniform is null).
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
