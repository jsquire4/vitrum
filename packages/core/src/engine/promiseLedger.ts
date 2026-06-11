import type { AnalyticShape, ScenePrimitive } from '../scene/primitives.js';
import type { SceneEmitter } from '../scene/emitters.js';
import type { SceneEnvironment } from '../scene/environment.js';
import type {
  BackendSupportDetails,
  EngineCapabilities,
  FramePresentationMode,
  IncrementalPatchSupport,
} from './capabilities.js';

export type BackendId = 'walkaround-hybrid' | 'pt-webgl2' | 'pt-webgpu';

export interface BackendMethodPromises {
  readonly updatePrimitive: boolean;
  readonly updateEmitter: boolean;
  readonly updateEnvironment: boolean;
  readonly addPrimitive: boolean;
  readonly removePrimitive: boolean;
  readonly setSize: boolean;
  readonly updateLighting: boolean;
  readonly onFrame: boolean;
  readonly onProgress: boolean;
  readonly debug: boolean;
  /** Whether the backend implements the optional `Engine.getScene()` scene
   *  read-back (returns the retained canonical core {@link Scene}). All three
   *  shipping backends retain the core Scene and implement it. */
  readonly getScene: boolean;
  /** Whether the backend implements `Engine.onError()` — the GPU/runtime
   *  error subscription.  All three shipping backends wire this. */
  readonly onError: boolean;
  /**
   * Whether the backend implements `Engine.captureFrame()` — the GPU→CPU
   * pixel readback that returns a {@link CapturedFrame} (linear HDR RGBA
   * Float32, top-left origin).  All three shipping backends implement this.
   */
  readonly captureFrame: boolean;
}

export interface FrameInputPromises {
  readonly honorsViewportPerFrame: boolean;
  readonly requiresSwapChainView: boolean;
  readonly honorsPerFrameBounces: boolean;
}

export interface BackendPromiseRecord {
  readonly supportsIncrementalScene: boolean;
  readonly incrementalPatchSupport: IncrementalPatchSupport;
  /** Whether the backend implements the explicit whole-primitive add/remove API
   *  ({@link Engine.addPrimitive} / {@link Engine.removePrimitive}). Distinct
   *  from `incrementalPatchSupport.topology` (count-change patches on an
   *  EXISTING primitive). */
  readonly supportsAddRemovePrimitive: boolean;
  readonly supportsAuxBuffers: boolean;
  readonly accumulates: boolean;
  readonly supportedPrimitiveKinds: readonly ScenePrimitive['kind'][];
  readonly supportedEmitterKinds: readonly SceneEmitter['kind'][];
  readonly supportedEnvironmentKinds: readonly SceneEnvironment['kind'][];
  readonly supportedAnalyticShapes: readonly AnalyticShape[];
  readonly presentationMode: FramePresentationMode;
  readonly supportDetails: BackendSupportDetails;
  readonly methodPromises: BackendMethodPromises;
  readonly frameInputPromises: FrameInputPromises;
}

// ── Compile-time drift guard ─────────────────────────────────────────────────
//
// `BackendPromiseRecord` mirrors a subset of `EngineCapabilities`. The
// collection fields (supportedPrimitiveKinds / supportedEmitterKinds /
// supportedEnvironmentKinds / supportedAnalyticShapes) intentionally diverge in
// container type: the ledger uses readonly arrays (serialisable, easy to assert
// in tests) while `EngineCapabilities` uses `ReadonlySet` (O(1) has-check).
// That divergence is structural-by-design and cannot be bridged with a blanket
// `satisfies Partial<EngineCapabilities>`.
//
// Instead we assert that the SCALAR / STRUCT fields that mirror cap keys are
// structurally compatible. Uses the `AssertExtends<TExpected, TActual>` idiom:
// the generic `U extends T` constraint fires a real TS error (not a silent
// `never`) if `BackendPromiseRecord` drops or renames a guarded key.
//
// `_LedgerCapabilitySlice` picks exactly the cap keys whose types ARE compatible
// across both shapes. When a new scalar/struct cap is added to `EngineCapabilities`
// and it belongs in the ledger, add it here AND to `BackendPromiseRecord`.
type _LedgerCapabilitySlice = Pick<
  EngineCapabilities,
  | 'supportsIncrementalScene'
  | 'incrementalPatchSupport'
  | 'supportsAddRemovePrimitive'
  | 'supportsAuxBuffers'
  | 'accumulates'
  | 'presentationMode'
  | 'supportDetails'
>;
// `U extends T` in the type parameter position emits TS2344 if U is not
// assignable to T — unlike `declare const x: never` which is always valid.
type _AssertExtends<T, U extends T> = U;
// This line errors if BackendPromiseRecord drops or incompatibly changes any
// of the capability keys listed in _LedgerCapabilitySlice.
type _LedgerCoversCapabilities = _AssertExtends<_LedgerCapabilitySlice, BackendPromiseRecord>;
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Every incremental-patch facet supported. Shared frozen value referenced by
 * all three backend records below — they are byte-identical on this field, so
 * a single shared const prevents drift between them.
 *
 * Per-backend `incrementalPatchSupport.topology === true` rationale (what
 * "count-change patches on an EXISTING primitive are absorbed without a full
 * setScene" means for each backend):
 *
 *   • walkaround-hybrid — vertex/index-COUNT changes (positions/normals/uvs/
 *     tangents/indices) ARE absorbed: a geometry change invalidates every cached
 *     GI signal on this realtime stack, so the engine re-runs its BVH rebuild +
 *     temporal reset (reusing the packing path, not a targeted in-place edit).
 *     `instances`/`params`/`shape`/`fallbackMesh`/`kind` patches are ALSO absorbed
 *     (P5, 2026-06-03): `HybridEngine.updatePrimitive` intercepts these
 *     wholesale-replacement fields and routes them through a full setScene rebuild
 *     (mutate-Scene → setScene spine, like addPrimitive) — no longer a throw. So
 *     instance-COUNT changes work here too; the rebuild is the cost (GI is
 *     invalidated either way on a realtime stack), matching pt-webgl2/pt-webgpu's
 *     contract surface.
 *   • pt-webgl2 — mesh/skinned vertex/index-COUNT changes and instanced-mesh
 *     instance-COUNT changes route through the retained core scene and rebuild
 *     the backend's scene textures/BVH pack. Co-present `material` routes through
 *     the same setScene repack, so material/light indices cannot drift.
 *   • pt-webgpu — instanced-mesh instance-COUNT change → TLAS-only rebuild, BLAS
 *     reused (rebuildTlasReuseBlas + uploadScenePackTlasRealloc); mesh/skinned
 *     vertex/index-COUNT change → rebuild only the changed primitive's BLAS,
 *     splice into concat buffers, rebase offsets + TLAS roots
 *     (rebuildPrimitiveBlas + uploadScenePackGeometryRealloc).
 *
 * In all three, `id`/`kind` morphs throw in patchPrimitiveInScene, and
 * whole-primitive ADD/REMOVE is setScene (see supportsAddRemovePrimitive), not a
 * patch. So `topology` means "vertex/index-count patches on an existing primitive
 * are absorbed" — fully on all three backends, incl. the instance-count case
 * (walkaround does it via a full setScene rebuild, the PT backends via targeted
 * TLAS/BLAS realloc).
 *
 * Per-backend `supportsAddRemovePrimitive === true` rationale: addPrimitive
 * appends a new primitive and removePrimitive evicts one, each by routing a
 * fresh mutated `Scene` copy through the engine's existing setScene packing path
 * (convert→expand→repack). A new primitive almost always brings a NEW material,
 * and the targeted geometry-only regen SKIPS material re-pack; reusing the
 * shared setScene path re-packs the material + light arrays correctly by
 * construction — no fragile per-array index remap. Distinct from
 * incrementalPatchSupport.topology (count-change patches on an EXISTING primitive).
 */
const ALL_PATCHES_SUPPORTED: IncrementalPatchSupport = Object.freeze({
  transform: true,
  positions: true,
  material: true,
  emitter: true,
  topology: true,
});

const ALL_EMITTERS_NATIVE: BackendSupportDetails['emitters'] = Object.freeze({
  directional: 'native',
  'rect-area': 'native',
  'disc-area': 'native',
  point: 'native',
  spot: 'native',
  'mesh-area': 'native',
});

/**
 * Walkaround-hybrid emitter support: point/spot are routed through the DDGI
 * fixture-light path only (no ReSTIR-DI term) — they contribute to indirect
 * GI via probe radiance but are NOT directly sampled in the ReSTIR direct-
 * illumination reservoir. Road-to-100: H41 Option A adds an analytic NEE loop
 * for point/spot in the shade pass (W4).
 */
const WALKAROUND_EMITTERS: BackendSupportDetails['emitters'] = Object.freeze({
  directional: 'native',
  'rect-area': 'native',
  'disc-area': 'native',
  // H41 — additive analytic NEE loop in shade.wgsl (binding 13, separate from
  // the RIS area-emitter pool). Inverse-square + spot cone smoothstep falloff
  // with deterministic shadow rays. Grade promoted from 'approximate' to 'native'.
  point: 'native',
  spot: 'native',
  'mesh-area': 'native',
});

/**
 * pt-webgpu emitter support: all emitter kinds are now native.
 * disc-area was previously approximate (32-triangle fan); it is now packed natively
 * into the rect-area stream with a shape discriminator (RECT_DISC_SHAPE_DISC = 1.0
 * in emission.w) and sampled via the concentric-disc map (Shirley & Chiu 1997),
 * exactly matching pt-webgl2's CIRC_AREA_LIGHT handling.
 * Promoted approximate → native, 2026-06-10 (emitterPacking.ts `packDiscAsRect`).
 * A/B radiometric validation in R9-B.
 */
const PT_WEBGPU_EMITTERS: BackendSupportDetails['emitters'] = ALL_EMITTERS_NATIVE;

const NO_ANALYTIC_SHAPES: BackendSupportDetails['analyticShapes'] = Object.freeze({
  sphere: 'unsupported',
  box: 'unsupported',
  capsule: 'unsupported',
  cylinder: 'unsupported',
  'h-channel-came': 'unsupported',
});

const PT_WEBGPU_ANALYTIC_SHAPES_NATIVE: BackendSupportDetails['analyticShapes'] = Object.freeze({
  sphere: 'native',
  box: 'native',
  capsule: 'native',
  cylinder: 'native',
  'h-channel-came': 'native',
});

const ANALYTIC_SHAPES_FALLBACK_GENERATED_MESH: BackendSupportDetails['analyticShapes'] = Object.freeze({
  sphere: 'fallback-generated-mesh',
  box: 'fallback-generated-mesh',
  capsule: 'fallback-generated-mesh',
  cylinder: 'fallback-generated-mesh',
  'h-channel-came': 'fallback-generated-mesh',
});

// ── Shared mutation/method constants (D1.4) ──────────────────────────────────
//
// Extracted to eliminate copy-paste drift between the three backend records.
// Deep-equal to the prior inline literals (byte-identical ledger output).

/** pt-webgl2 mutations — all fallback-rebuild (no fast path; full scene-texture/
 *  BVH repack on every mutation kind).  resize and lighting are unsupported. */
const ALL_MUTATIONS_FALLBACK_REBUILD: BackendPromiseRecord['supportDetails']['mutations'] = Object.freeze({
  transform: 'fallback-rebuild',
  positions: 'fallback-rebuild',
  material: 'fallback-rebuild',
  emitter: 'fallback-rebuild',
  topology: 'fallback-rebuild',
  addPrimitive: 'fallback-rebuild',
  removePrimitive: 'fallback-rebuild',
  environment: 'fallback-rebuild',
  resize: 'unsupported',
  lighting: 'unsupported',
});

/** pt-webgpu mutations — native fast paths for all per-primitive mutation kinds;
 *  add/remove are fallback-rebuild (insert/evict forces a full BLAS/TLAS repack).
 *  resize and lighting are unsupported. */
const PT_WEBGPU_MUTATIONS: BackendPromiseRecord['supportDetails']['mutations'] = Object.freeze({
  transform: 'native',
  positions: 'native',
  material: 'native',
  emitter: 'native',
  topology: 'native',
  addPrimitive: 'fallback-rebuild',
  removePrimitive: 'fallback-rebuild',
  environment: 'native',
  resize: 'unsupported',
  lighting: 'unsupported',
});

/** Method-promise fields that are true (or false) identically across all three
 *  shipping backends.  Spread into each record; backend-specific fields
 *  (setSize, updateLighting, debug) are added after the spread. */
const COMMON_METHOD_PROMISES: Pick<
  BackendMethodPromises,
  | 'updatePrimitive'
  | 'updateEmitter'
  | 'updateEnvironment'
  | 'addPrimitive'
  | 'removePrimitive'
  | 'onFrame'
  | 'onProgress'
  | 'getScene'
  | 'onError'
  | 'captureFrame'
> = Object.freeze({
  updatePrimitive: true,
  updateEmitter: true,
  updateEnvironment: true,
  addPrimitive: true,
  removePrimitive: true,
  onFrame: true,
  onProgress: true,
  getScene: true,
  onError: true,
  captureFrame: true,
});
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Machine-checkable backend contract truth table.
 *
 * Tests in backend packages assert runtime behavior against this ledger so
 * capability/method drift fails mechanically.
 */
export const BACKEND_PROMISE_LEDGER: Readonly<Record<BackendId, BackendPromiseRecord>> = {
  'walkaround-hybrid': {
    supportsIncrementalScene: true,
    incrementalPatchSupport: ALL_PATCHES_SUPPORTED,
    supportsAddRemovePrimitive: true,
    // Interim (plan/v1-closure-plan-2026-06-10.md): walkaround-hybrid surfaces
    // normalDepth and motionVectors, but NOT variance — the contract's
    // supportsAuxBuffers flag means variance AND motionVectors, and variance is
    // never exposed from walkaround's FrameOutput wiring. Flipped to false until
    // the variance buffer is wired (Wave 2 or later).
    supportsAuxBuffers: false,
    accumulates: false,
    // The render-scene path ingests mesh / skinned-mesh / instanced-mesh;
    // analytic primitives are accepted in the authored scene and converted to
    // generated MeshPrimitive fallbacks before BVH/GI ingestion consumes them.
    // instanced-mesh IS genuine here — walkaround renders instances via the
    // TLAS per-instance traversal path.
    supportedPrimitiveKinds: ['mesh', 'skinned-mesh', 'instanced-mesh', 'analytic'],
    // rect-area/disc-area → ReSTIR-DI emitter tris + DDGI fixtures; mesh-area →
    // mesh emissive material; directional → DDGI `sun` light. point/spot →
    // DDGI fixture lights (DDGI-only routing; no ReSTIR-DI direct term —
    // grade 'approximate'; see WALKAROUND_EMITTERS). Spots carry real cone data
    // (spotAxis + cosInner/cosOuter) and evalPointLight in the probe shader
    // applies smoothstep cone falloff. See coreEmittersToDDGILights.
    // `directional` → coreEmittersToDDGILights maps it to a `sun` DDGILight
    // carrying its real direction + intensity + colour; the host single-counts it
    // by setting the DDGI sun-intensity multiplier to 1. ReSTIR-DI harvests no
    // directional emitter, so there is no DI double-count.
    supportedEmitterKinds: ['directional', 'rect-area', 'disc-area', 'point', 'spot', 'mesh-area'],
    // procedural-sky degrades to a scalar tint + irradiance via resolveHybridEnvironment
    // (mode: 'procedural-sky-approx'; turbidity/rayleigh/mieDirectionalG are ignored;
    // a warn is emitted). Promoted 'unsupported' → 'approximate' so the ledger matches
    // what the code actually does (silent degrade with warn ≠ hard drop).
    supportedEnvironmentKinds: ['none', 'hdri', 'procedural-sky'],
    supportedAnalyticShapes: ['sphere', 'box', 'capsule', 'cylinder', 'h-channel-came'],
    presentationMode: 'swapchain-required',
    supportDetails: {
      primitives: {
        mesh: 'native',
        'skinned-mesh': 'native',
        'instanced-mesh': 'native',
        analytic: 'fallback-generated-mesh',
      },
      emitters: WALKAROUND_EMITTERS,
      environments: {
        none: 'native',
        // B3 + Wave 4 (2026-06-10) — directional IBL COMPLETE: sinθ-weighted
        // equirect inverse-CDFs at scene-group bindings 15-19; envImportanceSample
        // is a DI NEE candidate in the RIS loop (`ris.wgsl`); GI-escape rays read
        // the real map (`risGi.wgsl` envRadiance) and the NRC variant matches;
        // DDGI probe misses sample the HDRI (procedural fallback when none); RC
        // last-cascade env is bound; `updateEnvironment` rebuilds the directional
        // CDFs at runtime via resolveHybridEnvironment → updateDirectionalEnvironment.
        // Promoted approximate→native. Radiometric A/B pending V28-B.
        hdri: 'native',
        // resolveHybridEnvironment handles procedural-sky (mode: 'procedural-sky-approx'):
        // turbidity/rayleigh/mieDirectionalG are not sampled; skyTint + skyIrradiance scalars
        // are derived from the sun direction + mieCoefficient heuristic; a warn is emitted.
        // 'approximate' is the honest grade — the backend degrades with signal, not silently.
        'procedural-sky': 'approximate',
      },
      analyticShapes: ANALYTIC_SHAPES_FALLBACK_GENERATED_MESH,
      mutations: {
        transform: 'native',
        positions: 'native',
        material: 'native',
        emitter: 'native',
        topology: 'fallback-rebuild',
        addPrimitive: 'fallback-rebuild',
        removePrimitive: 'fallback-rebuild',
        environment: 'approximate',
        resize: 'native',
        lighting: 'native',
      },
    },
    methodPromises: {
      ...COMMON_METHOD_PROMISES,
      // updateEnvironment note: env-only runtime update IS implemented
      // (HybridEngine.updateEnvironment): maps SceneEnvironment onto diffuse
      // sky-dome scalars (skyTint/skyIrradiance), invalidates the DDGI probe
      // cache + resets the temporal accumulator, NO BVH rebuild. HDRI is
      // intensity-only (no equirect sampling); see the method's JSDoc. ✓ via spread.
      //
      // addPrimitive/removePrimitive note: implemented via full setScene-rebuild;
      // see supportsAddRemovePrimitive above. ✓ via spread.
      //
      // GPU error surface: device.uncapturederror (throttled) + device.lost. ✓ via spread.
      //
      // GPU→CPU pixel readback: resolvedTexture for 'linear'; 'output' rejects
      // (swap-chain write, no engine-owned display buffer to read back). ✓ via spread.
      setSize: true,
      updateLighting: true,
      debug: true,
    },
    frameInputPromises: {
      honorsViewportPerFrame: false,
      requiresSwapChainView: true,
      honorsPerFrameBounces: false,
    },
  },
  'pt-webgl2': {
    supportsIncrementalScene: true,
    incrementalPatchSupport: ALL_PATCHES_SUPPORTED,
    supportsAddRemovePrimitive: true,
    supportsAuxBuffers: false,
    accumulates: true,
    // The native WebGL2 packer ingests mesh / skinned-mesh / instanced-mesh.
    // `analytic` is NOT in the runtime set (PT_WEBGL2_SUPPORT.supportedAnalyticShapes
    // is empty); analytic primitives are warned-and-skipped. No generated-mesh
    // fallback path exists in the current slice — road-to-100 D3.
    // instanced-mesh IS supported: the backend scene pack preserves each
    // instance at its real per-instance world transform.
    supportedPrimitiveKinds: ['mesh', 'skinned-mesh', 'instanced-mesh'],
    supportedEmitterKinds: ['directional', 'rect-area', 'disc-area', 'point', 'spot', 'mesh-area'],
    supportedEnvironmentKinds: ['none', 'hdri'],
    // Runtime supportedAnalyticShapes is an empty Set (PT_WEBGL2_SUPPORT). No
    // analytic shape is accepted — none here either.
    supportedAnalyticShapes: [],
    presentationMode: 'offscreen-texture',
    supportDetails: {
      primitives: {
        mesh: 'native',
        // 2026-06-10 (Wave 2): pt-webgl2 solves the pose at ingestion via core
        // `solveSkin` (bones + bindMatrix + morph targets; see
        // scene/solveSkinPrimitives.ts); `updatePrimitive({bones})` re-solves
        // through the full-rebuild path. Promoted approximate→native.
        'skinned-mesh': 'native',
        'instanced-mesh': 'native',
        // Analytic primitives are warned-and-skipped in the current slice.
        // The authored scene is cached for param/shape patches, but ingestion
        // skips the shape (no generated-mesh fallback). Road-to-100 D3.
        analytic: 'unsupported',
      },
      emitters: ALL_EMITTERS_NATIVE,
      environments: {
        none: 'native',
        hdri: 'native',
        'procedural-sky': 'unsupported',
      },
      analyticShapes: NO_ANALYTIC_SHAPES,
      // buildCapabilities() overrides ALL mutation kinds to 'fallback-rebuild'
      // (a full scene-texture/BVH repack, not a targeted in-place edit).
      // The incrementalPatchSupport flags above reflect the OUTCOME (patches
      // are absorbed without error), but the support GRADE is fallback-rebuild
      // because no fast-path exists in the current slice.
      mutations: ALL_MUTATIONS_FALLBACK_REBUILD,
    },
    methodPromises: {
      ...COMMON_METHOD_PROMISES,
      // Context-lost surface: webglcontextlost canvas event. ✓ via spread.
      // GPU→CPU pixel readback: accum FBO (RGBA32F, rows flipped to top-left)
      // for 'linear'; present FBO for 'output'. ✓ via spread.
      setSize: false,
      updateLighting: false,
      debug: false,
    },
    frameInputPromises: {
      honorsViewportPerFrame: true,
      requiresSwapChainView: false,
      honorsPerFrameBounces: true,
    },
  },
  'pt-webgpu': {
    supportsIncrementalScene: true,
    incrementalPatchSupport: ALL_PATCHES_SUPPORTED,
    supportsAddRemovePrimitive: true,
    supportsAuxBuffers: true,
    accumulates: true,
    supportedPrimitiveKinds: ['mesh', 'instanced-mesh', 'analytic', 'skinned-mesh'],
    supportedEmitterKinds: ['directional', 'point', 'spot', 'rect-area', 'disc-area', 'mesh-area'],
    supportedEnvironmentKinds: ['none', 'hdri', 'procedural-sky'],
    supportedAnalyticShapes: ['sphere', 'box', 'capsule', 'cylinder', 'h-channel-came'],
    presentationMode: 'offscreen-texture',
    supportDetails: {
      primitives: {
        mesh: 'native',
        // 2026-06-10 (Wave 2): pt-webgpu solves the pose at ingestion via core
        // `solveSkin` (bones + bindMatrix + morph targets; see
        // scene/uploadSceneBuffers.ts applySolveSkinToScene) and re-solves on
        // `updatePrimitive({bones})` via the mutation router's bones fast path.
        // Promoted approximate→native.
        'skinned-mesh': 'native',
        'instanced-mesh': 'native',
        analytic: 'native',
      },
      emitters: PT_WEBGPU_EMITTERS,
      environments: {
        none: 'native',
        hdri: 'native',
        // Preetham 1999 analytic daylight model (baked to 256×128 equirect,
        // routed through the HDRI importance-sampling path).  All scene fields
        // (turbidity, rayleigh, mieCoefficient, mieDirectionalG, sunDirection,
        // intensity) are consumed.  'approximate' reflects the bake resolution
        // (256×128) vs a continuous analytical eval in the kernel.
        'procedural-sky': 'approximate',
      },
      analyticShapes: PT_WEBGPU_ANALYTIC_SHAPES_NATIVE,
      mutations: PT_WEBGPU_MUTATIONS,
    },
    methodPromises: {
      ...COMMON_METHOD_PROMISES,
      // GPU error surface: device.uncapturederror (throttled) + device.lost. ✓ via spread.
      // GPU→CPU pixel readback: accumTexture (rgba16float decoded to f32) for
      // 'linear'; presentTexture (rgba16float) for 'output'. ✓ via spread.
      setSize: false,
      updateLighting: false,
      debug: true,
    },
    frameInputPromises: {
      honorsViewportPerFrame: true,
      requiresSwapChainView: false,
      honorsPerFrameBounces: true,
    },
  },
};
