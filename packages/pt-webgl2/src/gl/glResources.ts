// GlResources — the GL resource owner (plan/three-removal/02-gl-framework.md §5, §6).
// Analog of pt-webgpu's GpuResources / the fork's PathTracingRenderer state. It owns:
//   - the accumulation FBO + RGBA32F color texture (+ optional MRT gNormalDepth/gAlbedo),
//   - the radiance-only shader-composite ping-pong pair (allocated lazily),
//   - the PT GlProgram (built from composeTraceGlsl + FULLSCREEN_VERT),
//   - the portable running-mean composite quad program,
//   - a FullscreenQuad.
//
// The per-sample draw (drawAccumStep) replaces the fork's renderTask generator: bind the
// accum FBO + MRT draw buffers, bind scene textures + uniforms, and
// draw the fullscreen triangle (fork PathTracingRenderer.js:144-167, §6 of plan 02).
//
// D10.1 (2026-06-10): BDPT light-subpath machinery extracted to BdptSubpathBuilder.ts;
// present-pass (tonemap/exposure/outputColorSpace) extracted to PresentPass.ts.

import type { TraceFeatures } from '../featureTypes.js';
import { featureDefines } from '../featureTypes.js';
import type { UploadedSceneTextures } from '../scene/sceneTextures.js';
import {
  composeNeeCandidateGlsl,
  composeNeeResolveGlsl,
  composeTraceGlsl,
} from '../glsl/composeTraceGlsl.js';
import { GlProgram } from './glProgram.js';
import { FullscreenQuad, FULLSCREEN_VERT } from './fullscreenQuad.js';
import {
  createRenderTarget,
  createNeeCandidateTarget,
  bindRenderTarget,
  clearRenderTarget,
  type NeeCandidateTarget,
  type RenderTarget,
} from './framebuffer.js';
import { probeGlCaps, type GlCaps } from './glCaps.js';
import { SOBOL_TEXTURE_SIZE, generateSobolTextureData } from '@vitrum/shared-samplers';
import { BdptSubpathBuilder } from './BdptSubpathBuilder.js';
import { PresentPass, selectPresentSources } from './PresentPass.js';
import { uploadFrameUniforms } from './uploadFrameUniforms.js';
import { BLEND_FRAG } from './blendFrag.js';
import {
  readOidnInputsFromWebGlFbos,
  type WebGlOidnReadbackResult,
} from '../denoise/rgba32fReadback.js';
import { allocGlTexture } from './texAlloc.js';
import {
  assertWebGl2RenderTargetRequest,
  DEFAULT_RENDER_TARGET_BUDGET_BYTES,
  estimateWebGl2DenoisedTargetBytes,
  estimateWebGl2ResidentBytes,
} from './renderTargetBudget.js';
import { prepareProgramSequence } from './programPreparation.js';

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
// Non-texture uniforms (lights.count, uMeshLightCount, uTotalEmissivePower,
// envMapInfo.totalSum) are uploaded separately after the
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
  readonly source: (
    scene: UploadedSceneTextures,
    d2d: WebGLTexture,
    d2a: WebGLTexture,
  ) => WebGLTexture;
}

const SCENE_TEXTURE_BINDINGS: readonly SceneTextureBinding[] = [
  // BVH struct samplers (bvh_struct_definitions.glsl: BVH { usampler2D index; sampler2D position; ... })
  { name: 'bvh.index', kind: 'tex2d', source: (s) => s.bvhIndex },
  { name: 'bvh.position', kind: 'tex2d', source: (s) => s.bvhPosition },
  { name: 'bvh.bvhBounds', kind: 'tex2d', source: (s) => s.bvhBounds },
  { name: 'bvh.bvhContents', kind: 'tex2d', source: (s) => s.bvhContents },
  // Per-triangle / per-material tables
  { name: 'materialIndexAttribute', kind: 'tex2d', source: (s) => s.materialIndex },
  { name: 'materials', kind: 'tex2d', source: (s) => s.materials },
  // Vertex attribute array (normal / tangent / uv / color layers)
  { name: 'attributesArray', kind: 'tex2dArray', source: (s) => s.attributesArray },
  // Analytic lights (LightsInfo { sampler2D tex; uint count; })
  { name: 'lights.tex', kind: 'tex2d', source: (s) => s.lights },
  // B4 — mesh-area triangle lights (dummy when scene has none → inert branch)
  { name: 'uMeshLights', kind: 'tex2d', source: (s, d2d) => s.meshLights ?? d2d },
  // Optional samplers the fork GLSL declares — must be bound to a valid texture of
  // the matching type to avoid unit-0 collision (GL_INVALID_OPERATION → black).
  // bindTexture no-ops for inactive samplers so the dummy binds are always safe.
  { name: 'textures', kind: 'tex2dArray', source: (s, _d2d, d2a) => s.textures2DArray ?? d2a },
  {
    name: 'materialRadianceTextures',
    kind: 'tex2dArray',
    source: (s, _d2d, d2a) => s.materialHdrTextures2DArray ?? d2a,
  },
  // iesProfiles removed — IES profiles not in @vitrum/core contract (item 20).
  { name: 'sobolTexture', kind: 'tex2d', source: (_s, d2d) => d2d },
  // A5 BDPT light-path texture (dummy here; overridden after the loop when bdpt is active).
  // NOTE: unit-0 collision warning — this MUST appear in the table so the sampler
  // unit is registered at link time.  The after-loop override replaces the dummy with
  // the real light-path texture when FEATURE_BDPT is compiled in and bdpt is active.
  { name: 'uBdptLightPathTex', kind: 'tex2d', source: (_s, d2d) => d2d },
  // EquirectHdrInfo importance-sampling samplers
  { name: 'envMapInfo.map', kind: 'tex2d', source: (s, d2d) => s.envMap ?? d2d },
  { name: 'envMapInfo.marginalWeights', kind: 'tex2d', source: (s, d2d) => s.envMarginal ?? d2d },
  {
    name: 'envMapInfo.conditionalWeights',
    kind: 'tex2d',
    source: (s, d2d) => s.envConditional ?? d2d,
  },
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
  readonly cameraWorldMatrix: Float32Array; // inverse(viewMatrix)
  readonly invProjectionMatrix: Float32Array; // inverse(projMatrix)
  readonly environmentIntensity: number;
  readonly environmentRotation: Float32Array; // mat4
  readonly backgroundBlur: number;
  readonly spectralEnabled: boolean;
  readonly backgroundAlpha: number; // 1 = opaque visible env (default); <1 = transparent
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
  /** Bounding sphere used to launch directional/environment light paths on a disk. */
  readonly bdptSceneCenter: readonly [number, number, number];
  readonly bdptSceneRadius: number;
  /** Shared hero wavelength/PDF for the global light path when BDPT + spectral are both active. */
  readonly bdptSharedWavelengthNm: number;
  readonly bdptSharedWavelengthPdf: number;
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

interface ProgramGraph {
  readonly key: string;
  readonly pt: GlProgram;
  readonly candidate: GlProgram;
  readonly resolve: GlProgram;
  readonly blend: GlProgram;
  readonly randomType: TraceFeatures['randomType'];
  /**
   * The three path-tracing programs are intentionally linked serially.
   * ANGLE's D3D11 translator otherwise runs three six-figure-source links at
   * once and can spend minutes thrashing the same compiler worker pool.
   */
  prepareStage: number;
}

/** Compiler/relink identity. Scene light presence is deliberately absent:
 * analytic, mesh-area, and environment paths are runtime-count/texture branches
 * in the same GLSL and do not alter either composed source or preamble defines. */
export function programGraphKey(features: TraceFeatures): string {
  return [
    features.bdpt ? 1 : 0,
    features.dof ? 1 : 0,
    features.cameraType,
    features.fog ? 1 : 0,
    features.randomType,
    features.basicMaterials ? 1 : 0,
    features.scalarRichMaterials ? 1 : 0,
    features.mappedPbrMaterials ? 1 : 0,
    features.mappedRichMaterials ? 1 : 0,
  ].join(':');
}

export class GlResources {
  readonly #gl: WebGL2RenderingContext;
  readonly #caps: GlCaps;
  /** Whether MRT aux g-buffers (gNormalDepth/gAlbedo) are allocated + written. */
  readonly #auxBuffers: boolean;
  readonly #quad: FullscreenQuad;
  readonly #maxRenderTargetBytes: number;

  /** The PT accumulation target (RGBA32F primary + optional MRT aux). */
  #accum: RenderTarget | null = null;
  /** Four-attachment packed handoff from path replay to the no-loop NEE resolve. */
  #neeCandidate: NeeCandidateTarget | null = null;
  /** Radiance-only shader-composite pair, used for alpha compositing and when
   * MRT auxiliaries require the PT target itself to remain an unblended sample. */
  #blend: [RenderTarget, RenderTarget] | null = null;
  /** Which slot of the blend pair currently holds the readable result. */
  #blendReadIndex = 0;

  #ptProgram: GlProgram | null = null;
  #neeCandidateProgram: GlProgram | null = null;
  #neeResolveProgram: GlProgram | null = null;
  #blendProgram: GlProgram | null = null;
  #activeProgramKey: string | null = null;
  #pendingProgramGraph: ProgramGraph | null = null;
  #randomType: TraceFeatures['randomType'] = 0;

  // ── Present pass (D10.1: extracted to PresentPass) ────────────────────────
  readonly #presentPass: PresentPass;
  #denoisedLinearTex: WebGLTexture | null = null;

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
  constructor(
    gl: WebGL2RenderingContext,
    supportsAuxBuffers: boolean,
    maxRenderTargetBytes = DEFAULT_RENDER_TARGET_BUDGET_BYTES,
  ) {
    this.#gl = gl;
    this.#caps = probeGlCaps(gl);
    this.#auxBuffers = supportsAuxBuffers && this.#caps.maxDrawBuffers >= 3;
    this.#maxRenderTargetBytes = maxRenderTargetBytes;
    this.#quad = new FullscreenQuad(gl);
    this.#presentPass = new PresentPass(gl, this.#quad);
    this.#bdptBuilder = new BdptSubpathBuilder(gl);
  }

  /** Validate dimensions, device texture limits, safe arithmetic, and peak bytes. */
  validateAccumRequest(w: number, h: number): number {
    const replacingPublishedTargets =
      this.#accum != null && (w !== this.#accumWidth || h !== this.#accumHeight);
    return assertWebGl2RenderTargetRequest(
      w,
      h,
      this.#auxBuffers,
      this.#maxRenderTargetBytes,
      this.#caps.maxTexSize,
      replacingPublishedTargets ? this.#residentRenderTargetBytes() : 0,
    );
  }

  /**
   * Ensure the accumulation resources match w×h, reallocating on a size change.
   * Returns `recreated` — true when targets were (re)built, so the caller resets its
   * sample counter (fork PathTracingRenderer.setSize → reset, :358-374).
   */
  ensureAccumResources(w: number, h: number): boolean {
    this.validateAccumRequest(w, h);
    if (w === this.#accumWidth && h === this.#accumHeight && this.#accum != null) return false;
    const nextAccum = createRenderTarget(this.#gl, w, h, this.#auxBuffers);
    let nextNeeCandidate: NeeCandidateTarget | null = null;
    // Present target — RGBA32F (deliberate: it is the public primaryRadiance
    // and must stay FLOAT-readable; both render targets allocate RGBA32F).
    try {
      nextNeeCandidate = createNeeCandidateTarget(this.#gl, w, h);
      this.#presentPass.allocate(w, h);
    } catch (error) {
      nextAccum.destroy();
      nextNeeCandidate?.destroy();
      throw error;
    }

    const previousAccum = this.#accum;
    const previousNeeCandidate = this.#neeCandidate;
    const previousBlend = this.#blend;
    this.#accum = nextAccum;
    this.#neeCandidate = nextNeeCandidate;
    this.#blend = null;
    this.#accumWidth = w;
    this.#accumHeight = h;
    this.#blendReadIndex = 0;
    this.#samples = 0;
    this.#clearDenoisedSource();
    previousAccum?.destroy();
    previousNeeCandidate?.destroy();
    if (previousBlend != null) {
      previousBlend[0].destroy();
      previousBlend[1].destroy();
    }
    return true;
  }

  /**
   * Build/start/poll the complete draw-program graph. Returns false while any
   * KHR_parallel_shader_compile job is pending; no draw may start until all
   * programs are atomically ready for one complete MC sample.
   */
  ensureProgram(features: TraceFeatures): boolean {
    const key = programGraphKey(features);
    if (
      this.#activeProgramKey === key &&
      this.#ptProgram != null &&
      this.#neeCandidateProgram != null &&
      this.#neeResolveProgram != null &&
      this.#blendProgram != null
    ) {
      return true;
    }
    if (this.#pendingProgramGraph?.key !== key) {
      this.#disposeProgramGraph(this.#pendingProgramGraph);
      this.#pendingProgramGraph = null;
      const baseDefines = featureDefines(features);
      let ptProgram: GlProgram | null = null;
      let candidateProgram: GlProgram | null = null;
      let resolveProgram: GlProgram | null = null;
      let blendProgram: GlProgram | null = null;
      try {
        ptProgram = new GlProgram(this.#gl, FULLSCREEN_VERT, composeTraceGlsl(features), {
          ...baseDefines,
          NEE_CANDIDATE_PASS: 0,
        });
        candidateProgram = new GlProgram(
          this.#gl,
          FULLSCREEN_VERT,
          composeNeeCandidateGlsl(features),
          {
            ...baseDefines,
            NEE_CANDIDATE_PASS: 1,
          },
        );
        resolveProgram = new GlProgram(
          this.#gl,
          FULLSCREEN_VERT,
          composeNeeResolveGlsl(features),
          {
            ...baseDefines,
            NEE_CANDIDATE_PASS: 0,
          },
        );
        blendProgram = new GlProgram(this.#gl, FULLSCREEN_VERT, BLEND_FRAG, {});
      } catch (error) {
        ptProgram?.dispose();
        candidateProgram?.dispose();
        resolveProgram?.dispose();
        blendProgram?.dispose();
        throw error;
      }
      if (
        ptProgram == null ||
        candidateProgram == null ||
        resolveProgram == null ||
        blendProgram == null
      ) {
        throw new Error('pt-webgl2: program construction completed without all pass programs');
      }
      this.#pendingProgramGraph = {
        key,
        pt: ptProgram,
        candidate: candidateProgram,
        resolve: resolveProgram,
        blend: blendProgram,
        randomType: features.randomType,
        prepareStage: 0,
      };
    }
    const pending = this.#pendingProgramGraph;
    if (pending == null) {
      throw new Error('pt-webgl2: pending program graph is missing after construction');
    }
    try {
      // Do not overlap the three large ANGLE/D3D11 jobs. Native Edge can
      // compile each one successfully yet fail to finish any of their links
      // when they contend concurrently for the translator worker pool.
      const largePrograms = [pending.pt, pending.candidate, pending.resolve] as const;
      const preparation = prepareProgramSequence(largePrograms, pending.prepareStage);
      pending.prepareStage = preparation.nextIndex;
      if (!preparation.ready) return false;
      // These two programs are tiny and can be polled together after the large
      // graph has linked without materially increasing compiler pressure.
      const blendReady = pending.blend.prepare();
      const presentReady = this.#presentPass.prepareProgram();
      if (!blendReady || !presentReady) return false;
    } catch (error) {
      this.#disposeProgramGraph(pending);
      this.#pendingProgramGraph = null;
      throw error;
    }

    const previous: ProgramGraph | null =
      this.#ptProgram != null &&
      this.#neeCandidateProgram != null &&
      this.#neeResolveProgram != null &&
      this.#blendProgram != null &&
      this.#activeProgramKey != null
        ? {
            key: this.#activeProgramKey,
            pt: this.#ptProgram,
            candidate: this.#neeCandidateProgram,
            resolve: this.#neeResolveProgram,
            blend: this.#blendProgram,
            randomType: this.#randomType,
            prepareStage: 3,
          }
        : null;
    this.#ptProgram = pending.pt;
    this.#neeCandidateProgram = pending.candidate;
    this.#neeResolveProgram = pending.resolve;
    this.#blendProgram = pending.blend;
    this.#activeProgramKey = pending.key;
    this.#randomType = pending.randomType;
    this.#pendingProgramGraph = null;
    this.#disposeProgramGraph(previous);
    return true;
  }

  /** The PT program (null before ensureProgram) — for the host to set uniforms/defines. */
  get ptProgram(): GlProgram | null {
    return this.#ptProgram;
  }

  /** Clear all accumulation targets to (0,0,0,0) + reset the sample counter (fork reset()). */
  clearAccum(): void {
    this.#clearDenoisedSource();
    if (this.#accum != null) clearRenderTarget(this.#gl, this.#accum);
    if (this.#neeCandidate != null) {
      this.#gl.bindFramebuffer(this.#gl.FRAMEBUFFER, this.#neeCandidate.fbo);
      this.#gl.drawBuffers(this.#neeCandidate.drawBuffers);
      this.#gl.clearColor(0, 0, 0, 0);
      this.#gl.clear(this.#gl.COLOR_BUFFER_BIT);
    }
    if (this.#blend != null) {
      clearRenderTarget(this.#gl, this.#blend[0]);
      clearRenderTarget(this.#gl, this.#blend[1]);
    }
    this.#blendReadIndex = 0;
    this.#samples = 0;
  }

  /**
   * One accumulation step (plan 02 §6). Binds the accum FBO + MRT draw buffers,
   * uploads the sample seed, binds scene textures, and draws the fullscreen
   * triangle. After separately resolving NEE into the raw radiance sample, it
   * performs one portable shader running-mean update into the ping-pong pair.
   */
  drawAccumStep(
    scene: UploadedSceneTextures,
    seed: number,
    frame: FrameUniforms,
  ): void {
    const gl = this.#gl;
    if (this.#accum == null)
      throw new Error('pt-webgl2: drawAccumStep before ensureAccumResources');
    if (
      this.#ptProgram == null ||
      this.#neeCandidateProgram == null ||
      this.#neeResolveProgram == null
    ) {
      throw new Error('pt-webgl2: drawAccumStep before ensureProgram');
    }
    if (this.#neeCandidate == null) {
      throw new Error('pt-webgl2: drawAccumStep before NEE candidate allocation');
    }
    this.#clearDenoisedSource();
    const prog = this.#ptProgram;

    // A5 — BDPT light subpath. Build the light-path vertex texture for THIS sample
    // (the subpath is reseeded per frame via the `seed`/rand bank) BEFORE the eye
    // pass, then bind it as `uBdptLightPathTex`. Only runs when bdpt is on; the
    // unidirectional path skips this entirely (byte-identical when bdpt:false).
    const bdptResult = frame.bdpt
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
    // Core WebGL2 cannot control blending independently per draw buffer. MRT
    // samples therefore render unblended so attachments 1/2 overwrite with the
    // latest primary-hit normal/depth and albedo. Attachment 0 is accumulated
    // through the portable shader ping-pong pass below. This is an explicit
    // last-sample auxiliary contract (important for DoF, where primary hits can
    // vary); encoded normals and linear depth are never numerically blended.
    // Main radiance and the separately-resolved NEE term must form ONE complete
    // sample before the single running-mean update. Therefore every device uses
    // the portable shader composite. Fixed-function float blending is not a
    // valid alternate regime because it would blend every MRT attachment and
    // would accumulate main radiance before the NEE sample is complete.
    gl.disable(gl.BLEND);

    // Upload the per-frame individual uniforms (D11-6: extracted to
    // uploadFrameUniforms). `prog.use()` and the live setter sequence live there.
    const sampleOpacity = 1;
    uploadFrameUniforms(prog, sampleOpacity, seed, frame);
    this.#bindSceneTextures(prog, scene, bdptResult);
    this.#quad.draw(gl);

    // Replay the exact continuation path into the four packed candidate
    // attachments. Stateless reservoir replacement plus saved/restored light
    // RNG leaves every continuation draw bit-identical to the main pass.
    const candidateProgram = this.#neeCandidateProgram;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.#neeCandidate.fbo);
    gl.drawBuffers(this.#neeCandidate.drawBuffers);
    gl.viewport(0, 0, this.#accumWidth, this.#accumHeight);
    gl.disable(gl.BLEND);
    uploadFrameUniforms(candidateProgram, 1, seed, frame);
    this.#bindSceneTextures(candidateProgram, scene, bdptResult);
    this.#quad.draw(gl);

    // Resolve one retained vertex with no path loop and add it to the raw main
    // sample. Alpha is zero in the resolve output, so primary/background alpha
    // remains owned solely by the main trace.
    bindRenderTarget(gl, this.#accum);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
    gl.viewport(0, 0, this.#accumWidth, this.#accumHeight);
    gl.enable(gl.BLEND);
    gl.blendEquation(gl.FUNC_ADD);
    gl.blendFunc(gl.ONE, gl.ONE);
    const resolveProgram = this.#neeResolveProgram;
    uploadFrameUniforms(resolveProgram, 1, seed, frame);
    this.#bindSceneTextures(resolveProgram, scene, bdptResult);
    for (let attachment = 0; attachment < 4; attachment += 1) {
      resolveProgram.bindTexture(
        `uNeeCandidate${attachment}`,
        this.#neeCandidate.textures[attachment]!,
      );
    }
    this.#quad.draw(gl);
    gl.disable(gl.BLEND);
    gl.drawBuffers(this.#accum.drawBuffers);

    this.#compositeBlendStep();

    this.#samples += 1;

    this.presentAccumulation(frame.tonemapMode, frame.exposure, frame.outputColorSpace);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /** Current accumulation dimensions (0×0 before first ensureAccumResources). */
  get accumDims(): { readonly width: number; readonly height: number } {
    return { width: this.#accumWidth, height: this.#accumHeight };
  }

  /** Re-run only tonemap/exposure/output-space over the current linear accumulator. */
  presentAccumulation(tonemapMode: number, exposure: number, outputColorSpace: number): void {
    const sources = selectPresentSources(
      this.#linearResultTexture(),
      this.#denoisedLinearTex,
    );
    if (sources == null) return;
    this.#presentPass.run(
      sources.radiance,
      sources.coverage,
      this.#accumWidth,
      this.#accumHeight,
      tonemapMode,
      exposure,
      outputColorSpace,
    );
  }

  /** Upload accepted linear-HDR OIDN RGB and present it through the existing output transform. */
  presentDenoisedResult(
    frame: { readonly rgb: Float32Array; readonly width: number; readonly height: number },
    tonemapMode: number,
    exposure: number,
    outputColorSpace: number,
  ): void {
    if (frame.width !== this.#accumWidth || frame.height !== this.#accumHeight) {
      throw new RangeError(
        `pt-webgl2: OIDN result ${frame.width}×${frame.height} does not match accumulation ` +
          `${this.#accumWidth}×${this.#accumHeight}`,
      );
    }
    const pixelCount = frame.width * frame.height;
    if (frame.rgb.length !== pixelCount * 3) {
      throw new RangeError(
        `pt-webgl2: OIDN result expected ${pixelCount * 3} RGB floats, got ${frame.rgb.length}`,
      );
    }
    const nextDenoisedBytes = estimateWebGl2DenoisedTargetBytes(frame.width, frame.height);
    const residentBytes = this.#residentRenderTargetBytes();
    if (residentBytes > Number.MAX_SAFE_INTEGER - nextDenoisedBytes) {
      throw new RangeError(
        'pt-webgl2: OIDN target replacement peak overflows Number.MAX_SAFE_INTEGER',
      );
    }
    const replacementPeakBytes = residentBytes + nextDenoisedBytes;
    if (replacementPeakBytes > this.#maxRenderTargetBytes) {
      throw new RangeError(
        `pt-webgl2: OIDN target replacement peak requires ${replacementPeakBytes} bytes ` +
          `(${residentBytes} resident + ${nextDenoisedBytes} new), exceeding ` +
          `maxRenderTargetBytes=${this.#maxRenderTargetBytes}. ` +
          `The previous complete denoised frame remains active.`,
      );
    }
    const rgba = new Float32Array(pixelCount * 4);
    // OIDNDispatcherCore exposes CPU images in the engine-wide top-left
    // convention. texImage2D's first row is the texture's bottom row, and the
    // fullscreen present pass samples vUv.y=0 at the screen bottom, so flip the
    // rows back to GL orientation on upload.
    for (let dstY = 0; dstY < frame.height; dstY += 1) {
      const srcY = frame.height - 1 - dstY;
      for (let x = 0; x < frame.width; x += 1) {
        const src = (srcY * frame.width + x) * 3;
        const dst = (dstY * frame.width + x) * 4;
        rgba[dst] = frame.rgb[src]!;
        rgba[dst + 1] = frame.rgb[src + 1]!;
        rgba[dst + 2] = frame.rgb[src + 2]!;
        rgba[dst + 3] = 1;
      }
    }
    const next = allocGlTexture(this.#gl, {
      kind: 'rect',
      width: frame.width,
      height: frame.height,
      internalFormat: this.#gl.RGBA32F,
      format: this.#gl.RGBA,
      type: this.#gl.FLOAT,
      data: rgba,
      resourceName: 'OIDN denoised linear HDR',
    });
    const previous = this.#denoisedLinearTex;
    this.#denoisedLinearTex = next;
    try {
      this.presentAccumulation(tonemapMode, exposure, outputColorSpace);
    } catch (error) {
      this.#denoisedLinearTex = previous;
      this.#gl.deleteTexture(next);
      throw error;
    }
    if (previous != null) this.#gl.deleteTexture(previous);
  }

  /** Stop presenting the accepted OIDN source and return to the live accumulator. */
  clearDenoisedResult(): void {
    this.#clearDenoisedSource();
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

  #linearResultTexture(): WebGLTexture | null {
    if (this.#blend != null) {
      const [a, b] = this.#blend;
      return (this.#blendReadIndex === 0 ? a : b).color;
    }
    return this.#accum?.color ?? null;
  }

  /**
   * Packed normal+depth from the latest path-trace sample (null when MRT is
   * disabled). It is deliberately not averaged with progressive radiance.
   */
  get normalDepthTex(): WebGLTexture | null {
    return this.#accum?.normalDepth ?? null;
  }

  /** Albedo from the latest path-trace sample (null when MRT is disabled). */
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
    this.#disposeProgramGraph(this.#pendingProgramGraph);
    this.#pendingProgramGraph = null;
    this.#destroyTargets();
    this.#presentPass.destroy();
    this.#presentPass.disposeProgram();
    this.#ptProgram?.dispose();
    this.#ptProgram = null;
    this.#neeCandidateProgram?.dispose();
    this.#neeCandidateProgram = null;
    this.#neeResolveProgram?.dispose();
    this.#neeResolveProgram = null;
    this.#blendProgram?.dispose();
    this.#blendProgram = null;
    this.#activeProgramKey = null;
    this.#quad.dispose(gl);
    // H7 FIX (2026-06-09): delete the lazily-allocated dummy textures — dispose()
    // freed the programs/targets/quad but LEAKED these two GPU textures on every
    // engine teardown (Canvas remount / route change churn would accumulate them).
    if (this.#dummy2dTex != null) {
      gl.deleteTexture(this.#dummy2dTex);
      this.#dummy2dTex = null;
    }
    if (this.#dummy2dArrTex != null) {
      gl.deleteTexture(this.#dummy2dArrTex);
      this.#dummy2dArrTex = null;
    }
    if (this.#sobolTex != null) {
      gl.deleteTexture(this.#sobolTex);
      this.#sobolTex = null;
    }
    // A5 — free the BDPT light-path ping-pong pair + copy FBO (D10.1: via BdptSubpathBuilder).
    this.#bdptBuilder.dispose();
  }

  // ----- internals -------------------------------------------------------------------------

  #residentRenderTargetBytes(): number {
    if (this.#accum == null) return 0;
    return estimateWebGl2ResidentBytes(
      this.#accumWidth,
      this.#accumHeight,
      this.#auxBuffers,
      {
        blend: this.#blend != null,
        denoised: this.#denoisedLinearTex != null,
      },
    );
  }

  #disposeProgramGraph(graph: ProgramGraph | null): void {
    if (graph == null) return;
    graph.pt.dispose();
    graph.candidate.dispose();
    graph.resolve.dispose();
    graph.blend.dispose();
  }

  /** Composite the latest PT radiance sample into the radiance-only ping-pong pair. */
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

    if (!blendProg.use()) {
      throw new Error('pt-webgl2: blend pass reached draw before its program was ready');
    }
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
    const first = createRenderTarget(this.#gl, this.#accumWidth, this.#accumHeight, false);
    let second: RenderTarget | null = null;
    try {
      second = createRenderTarget(this.#gl, this.#accumWidth, this.#accumHeight, false);
      clearRenderTarget(this.#gl, first);
      clearRenderTarget(this.#gl, second);
    } catch (error) {
      first.destroy();
      second?.destroy();
      throw error;
    }
    this.#blend = [first, second];
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
      this.#dummy2dTex = allocGlTexture(gl, {
        kind: '2d',
        dim: 1,
        internalFormat: gl.RGBA,
        format: gl.RGBA,
        type: gl.UNSIGNED_BYTE,
        data: new Uint8Array([0, 0, 0, 255]),
        resourceName: 'dummy 2D',
      });
    }
    return this.#dummy2dTex;
  }

  #dummyTex2DArray(): WebGLTexture {
    const gl = this.#gl;
    if (this.#dummy2dArrTex == null) {
      this.#dummy2dArrTex = allocGlTexture(gl, {
        kind: 'array',
        dim: 1,
        layers: 1,
        internalFormat: gl.RGBA,
        format: gl.RGBA,
        type: gl.UNSIGNED_BYTE,
        data: new Uint8Array([0, 0, 0, 255]),
        resourceName: 'dummy 2D-array',
      });
    }
    return this.#dummy2dArrTex;
  }

  #sobolTex2D(): WebGLTexture {
    const gl = this.#gl;
    if (this.#sobolTex == null) {
      this.#sobolTex = allocGlTexture(gl, {
        kind: '2d',
        dim: SOBOL_TEXTURE_SIZE,
        internalFormat: gl.RGBA32F,
        format: gl.RGBA,
        type: gl.FLOAT,
        data: generateSobolTextureData(),
        resourceName: 'Sobol',
      });
    }
    return this.#sobolTex;
  }

  #clearDenoisedSource(): void {
    if (this.#denoisedLinearTex != null) {
      this.#gl.deleteTexture(this.#denoisedLinearTex);
      this.#denoisedLinearTex = null;
    }
  }

  #destroyTargets(): void {
    this.#clearDenoisedSource();
    this.#accum?.destroy();
    this.#accum = null;
    this.#neeCandidate?.destroy();
    this.#neeCandidate = null;
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
   * readPixels; the result target therefore stays RGBA32F.
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

    const previousFramebuffer = gl.getParameter(gl.FRAMEBUFFER_BINDING) as WebGLFramebuffer | null;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    const framebufferStatus = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (framebufferStatus !== gl.FRAMEBUFFER_COMPLETE) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, previousFramebuffer);
      throw new Error(
        `pt-webgl2: ${source} capture framebuffer is incomplete (status 0x${framebufferStatus.toString(16)})`,
      );
    }
    // Both targets are RGBA32F (accum AND present),
    // so both read with the FLOAT path.
    const pixels = new Float32Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, pixels);
    gl.bindFramebuffer(gl.FRAMEBUFFER, previousFramebuffer);

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
