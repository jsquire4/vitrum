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
    // FrameRendered surfaces normalDepth + demodulated albedo (rgba16float) +
    // motionVectors (rg32float) from the always-allocated G-buffer; hosts can
    // drive an external denoiser / post chain off them. (Variance is the RG32F
    // Welford buffer, not the contract's RGBA32F, so it's not exposed.)
    supportsAuxBuffers: true,
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
    supportedEnvironmentKinds: ['none', 'hdri'],
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
        hdri: 'approximate',
        'procedural-sky': 'unsupported',
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
      updatePrimitive: true,
      updateEmitter: true,
      // Env-only runtime update IS implemented (HybridEngine.updateEnvironment):
      // maps the SceneEnvironment onto the diffuse sky-dome scalars
      // (skyTint/skyIrradiance — this backend has no IBL baker), invalidates the
      // DDGI probe cache + resets the temporal accumulator, with NO BVH rebuild.
      // HDRI is intensity-only (no equirect sampling); see the method's JSDoc.
      updateEnvironment: true,
      // Implemented — see supportsAddRemovePrimitive above (full setScene-rebuild).
      addPrimitive: true,
      removePrimitive: true,
      setSize: true,
      updateLighting: true,
      onFrame: true,
      onProgress: true,
      debug: true,
      getScene: true,
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
      mutations: {
        // buildCapabilities() overrides ALL mutation kinds to 'fallback-rebuild'
        // (a full scene-texture/BVH repack, not a targeted in-place edit).
        // The incrementalPatchSupport flags above reflect the OUTCOME (patches
        // are absorbed without error), but the support GRADE is fallback-rebuild
        // because no fast-path exists in the current slice.
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
      },
    },
    methodPromises: {
      updatePrimitive: true,
      updateEmitter: true,
      updateEnvironment: true,
      addPrimitive: true,
      removePrimitive: true,
      setSize: false,
      updateLighting: false,
      onFrame: true,
      onProgress: true,
      debug: false,
      getScene: true,
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
        'skinned-mesh': 'native',
        'instanced-mesh': 'native',
        analytic: 'native',
      },
      emitters: ALL_EMITTERS_NATIVE,
      environments: {
        none: 'native',
        hdri: 'native',
        'procedural-sky': 'native',
      },
      analyticShapes: PT_WEBGPU_ANALYTIC_SHAPES_NATIVE,
      mutations: {
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
      },
    },
    methodPromises: {
      updatePrimitive: true,
      updateEmitter: true,
      updateEnvironment: true,
      addPrimitive: true,
      removePrimitive: true,
      setSize: false,
      updateLighting: false,
      onFrame: true,
      onProgress: true,
      debug: true,
      getScene: true,
    },
    frameInputPromises: {
      honorsViewportPerFrame: true,
      requiresSwapChainView: false,
      honorsPerFrameBounces: true,
    },
  },
};
