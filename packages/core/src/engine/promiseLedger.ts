import type { AnalyticShape, ScenePrimitive } from '../scene/primitives.js';
import type { SceneEmitter } from '../scene/emitters.js';
import type { SceneEnvironment } from '../scene/environment.js';
import type { FramePresentationMode, IncrementalPatchSupport } from './capabilities.js';

export type BackendId = 'walkaround-hybrid' | 'pt-webgl' | 'pt-webgpu';

export interface BackendMethodPromises {
  readonly updatePrimitive: boolean;
  readonly updateEmitter: boolean;
  readonly updateEnvironment: boolean;
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
      setSize: true,
      updateLighting: true,
      onFrame: true,
      onProgress: false,
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
    incrementalPatchSupport: {
      transform: true,
      positions: true,
      material: true,
      emitter: true,
      topology: false,
    },
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
    // `topology: false` stands even though pt-webgpu takes a transparent
    // TLAS-only fast path for instanced-mesh instance-COUNT changes (BLAS reused
    // verbatim; only the 5 TLAS buffers reallocate). That fast path is an
    // internal perf optimization — mesh vertex/index-count changes and
    // whole-primitive add/remove still rebuild through setScene — so the broad
    // "all topology patches absorbed" promise remains unmet.
    incrementalPatchSupport: {
      transform: true,
      positions: true,
      material: true,
      emitter: true,
      topology: false,
    },
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

