import type { AnalyticShape, ScenePrimitive } from '../scene/primitives.js';
import type { SceneEmitter } from '../scene/emitters.js';
import type { SceneEnvironment } from '../scene/environment.js';
import type { EngineCapabilities, FramePresentationMode, IncrementalPatchSupport } from './capabilities.js';

export type BackendId = 'walkaround-hybrid' | 'pt-webgl' | 'pt-webgpu';

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
 *     invalidated either way on a realtime stack), matching pt-webgl/pt-webgpu's
 *     contract surface.
 *   • pt-webgl — mesh/skinned vertex/index-COUNT change rebuilds that one mesh's
 *     THREE BufferGeometry in place (applyGeometryPatchToMesh) + the fork's
 *     targeted geometry+BVH regen (StaticGeometryGenerator force-rebuild on
 *     changed attribute lengths). instanced-mesh instance-COUNT change re-expands
 *     ONLY that primitive's baked THREE.Mesh children. Co-present `material`
 *     routes to a full setScene (MaterialsTexture re-pack).
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
 * shared setScene path re-packs the MaterialsTexture + light arrays correctly by
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
    // vitrumSceneToThree ingests mesh / skinned-mesh / instanced-mesh; analytic
    // has no THREE-conversion path (partitionSceneBySupport warn-skips it).
    // instanced-mesh IS genuine here — walkaround renders instances via the
    // TLAS per-instance traversal path.
    supportedPrimitiveKinds: ['mesh', 'skinned-mesh', 'instanced-mesh'],
    // rect-area/disc-area → ReSTIR-DI emitter tris + DDGI fixtures; mesh-area →
    // mesh emissive material; point/spot → DDGI fixture lights. Spots carry
    // real cone data (spotAxis + cosInner/cosOuter) and evalPointLight in the
    // probe shader applies smoothstep cone falloff. See coreEmittersToDDGILights.
    // `directional` → DDGI `sun` light: coreEmittersToDDGILights maps a scene
    // directional to a `sun` DDGILight carrying its real direction (negated to
    // a travel direction), intensity, and colour, and the host single-counts it
    // by setting the DDGI sun-intensity multiplier to 1 (so the config
    // primaryLightIntensity does not additionally scale the DDGI sun; it still
    // drives the shade-side Lo_emit). ReSTIR-DI harvests no directional emitter,
    // so there is no DI double-count.
    supportedEmitterKinds: ['directional', 'rect-area', 'disc-area', 'point', 'spot', 'mesh-area'],
    supportedEnvironmentKinds: ['none', 'hdri'],
    supportedAnalyticShapes: [],
    presentationMode: 'swapchain-required',
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
    },
    frameInputPromises: {
      honorsViewportPerFrame: false,
      requiresSwapChainView: true,
      honorsPerFrameBounces: false,
    },
  },
  'pt-webgl': {
    supportsIncrementalScene: true,
    incrementalPatchSupport: ALL_PATCHES_SUPPORTED,
    supportsAddRemovePrimitive: true,
    supportsAuxBuffers: false,
    accumulates: true,
    // vitrumSceneToThree ingests mesh / skinned-mesh / instanced-mesh;
    // analytic has no THREE-conversion path (partitionSceneBySupport
    // warn-skips it).
    // instanced-mesh IS supported — vitrumSceneToThree builds a single
    // THREE.InstancedMesh (shared with walkaround's TLAS path), and pt-webgl's
    // setScene expands it into N baked THREE.Mesh instances
    // (`expandInstancedMeshesInScene`) BEFORE the fork's geometry generator
    // runs, so each instance renders at its real per-instance world transform.
    // (The fork's convertToStaticGeometry bakes only mesh.matrixWorld and
    // ignores instanceMatrix, hence the pt-webgl-side pre-bake.)
    supportedPrimitiveKinds: ['mesh', 'skinned-mesh', 'instanced-mesh'],
    supportedEmitterKinds: ['directional', 'rect-area', 'disc-area', 'point', 'spot', 'mesh-area'],
    supportedEnvironmentKinds: ['none', 'hdri'],
    supportedAnalyticShapes: [],
    presentationMode: 'offscreen-texture',
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
    },
    frameInputPromises: {
      honorsViewportPerFrame: true,
      requiresSwapChainView: false,
      honorsPerFrameBounces: true,
    },
  },
};

