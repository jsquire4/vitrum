import type { AnalyticShape, ScenePrimitive } from '../scene/primitives.js';
import type { SceneEmitter } from '../scene/emitters.js';
import type { SceneEnvironment } from '../scene/environment.js';
import type { FramePresentationMode, IncrementalPatchSupport } from './capabilities.js';

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

/**
 * Machine-checkable backend contract truth table.
 *
 * Tests in backend packages assert runtime behavior against this ledger so
 * capability/method drift fails mechanically.
 */
export const BACKEND_PROMISE_LEDGER: Readonly<Record<BackendId, BackendPromiseRecord>> = {
  'walkaround-hybrid': {
    supportsIncrementalScene: true,
    incrementalPatchSupport: {
      transform: true,
      positions: true,
      material: true,
      emitter: true,
      topology: true,
    },
    // Whole-primitive add/remove is a deliberate follow-up on walkaround-hybrid
    // (the DDGI / ReSTIR / RC subsystems all index off a packed scene that would
    // need re-syncing). Hosts use `setScene` for now.
    supportsAddRemovePrimitive: false,
    supportsAuxBuffers: false,
    accumulates: false,
    // vitrumSceneToThree ingests mesh / skinned-mesh / instanced-mesh; analytic
    // has no THREE-conversion path (partitionSceneBySupport warn-skips it).
    // instanced-mesh IS genuine here — walkaround renders instances via the
    // TLAS per-instance traversal path.
    supportedPrimitiveKinds: ['mesh', 'skinned-mesh', 'instanced-mesh'],
    // rect-area/disc-area → ReSTIR-DI emitter tris + DDGI fixtures; mesh-area →
    // mesh emissive material; point/spot → DDGI fixture lights (spot
    // point-approximated). See coreEmittersToDDGILights.
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
      updateEnvironment: false,
      // Follow-up — see supportsAddRemovePrimitive above.
      addPrimitive: false,
      removePrimitive: false,
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
    // `topology: true` — pt-webgl absorbs BOTH topology-changing patch kinds
    // `updatePrimitive` can legally receive on an existing primitive, without a
    // full `setScene`:
    //   • mesh/skinned-mesh vertex/index-COUNT change → rebuild that one mesh's
    //     THREE BufferGeometry in place (`applyGeometryPatchToMesh`) + the fork's
    //     targeted geometry+BVH regen; the fork's StaticGeometryGenerator detects
    //     the changed attribute lengths and force-rebuilds (GEOMETRY_REBUILT);
    //   • instanced-mesh instance-COUNT change → re-expand ONLY that primitive's
    //     baked THREE.Mesh children in the live scene root (swap N → N', reusing
    //     the shared geometry + material) + the same targeted regen; the changed
    //     child set (count delta forces GEOMETRY_REBUILT) is picked up on the
    //     next generate().
    // Both stay same-material (a co-present `material` is blocked by the
    // geometry-only / instances-only classifiers and routes to a full `setScene`
    // so the MaterialsTexture is re-packed). `id`/`kind` morphs throw in
    // patchPrimitiveInScene, and whole-primitive ADD/REMOVE is `setScene`, not a
    // patch — so `topology` here means exactly "count-change patches on an
    // existing primitive are absorbed".
    incrementalPatchSupport: {
      transform: true,
      positions: true,
      material: true,
      emitter: true,
      topology: true,
    },
    // Explicit whole-primitive add/remove IS implemented: addPrimitive appends a
    // new primitive and removePrimitive evicts one, each via a full
    // `setScene`-equivalent rebuild of the mutated scene. A new primitive almost
    // always brings a NEW material, and the fork's targeted geometry-only regen
    // SKIPS updateMaterials(); routing add/remove through the shared setScene
    // packing path (convert→expand→tracer.setScene) re-packs the
    // MaterialsTexture + light arrays correctly by construction — no fragile
    // per-array index remap. Mirrors pt-webgpu's full-repack choice. Distinct
    // from incrementalPatchSupport.topology (count-change patches on an EXISTING
    // primitive).
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
    // `topology: true` — pt-webgpu absorbs EVERY topology-changing patch that
    // `updatePrimitive` can legally receive, without a full `setScene`:
    //   • instanced-mesh instance-COUNT change → TLAS-only rebuild, BLAS reused
    //     verbatim (slice-1: rebuildTlasReuseBlas + uploadScenePackTlasRealloc);
    //   • mesh/skinned-mesh vertex/index-COUNT change → rebuild ONLY the changed
    //     primitive's BLAS, splice it into the concat buffers, rebase downstream
    //     offsets + TLAS roots (slice-2: rebuildPrimitiveBlas resize splice +
    //     uploadScenePackGeometryRealloc).
    // `id`/`kind` morphs throw in patchPrimitiveInScene (contract violation), and
    // whole-primitive ADD/REMOVE is `setScene`, not a patch — so `topology` here
    // means exactly "count-change patches on an existing primitive are absorbed".
    incrementalPatchSupport: {
      transform: true,
      positions: true,
      material: true,
      emitter: true,
      topology: true,
    },
    // Explicit whole-primitive add/remove IS implemented: addPrimitive appends a
    // new primitive and removePrimitive evicts one, each via a full
    // buildPackedScene repack of the mutated scene (the dense material /
    // analytic / triMaterialId packing is reproduced correctly by reusing the
    // exact setScene packing path — no fragile per-array index remap).
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

