import type {
  AnalyticShape,
  EngineCapabilities,
  SceneEmitter,
  SceneEnvironment,
  ScenePrimitive,
  SupportSets,
} from '@vitrum/core';
import { BACKEND_PROMISE_LEDGER } from '@vitrum/core';

/**
 * The single source of truth for what this backend ingests. `buildCapabilities`
 * derives its `supported*Kinds` from THIS object, and `setScene` filters the scene
 * against it via `partitionSceneBySupport` — so advertised capability and ingestion
 * behaviour cannot drift (the pt-webgpu PT_WEBGPU_SUPPORT invariant).
 *
 * Slice 0 ships the mesh-focused set; analytic shapes + procedural-sky come online
 * in later slices (left as an empty/minimal set here, so they are warn-and-skipped).
 */
export const PT_WEBGL2_SUPPORT: Required<SupportSets> = {
  supportedPrimitiveKinds: new Set<ScenePrimitive['kind']>([
    'mesh',
    'instanced-mesh',
    'skinned-mesh',
  ]),
  supportedEmitterKinds: new Set<SceneEmitter['kind']>([
    'directional',
    'point',
    'spot',
    'rect-area',
    'disc-area',
    'mesh-area',
  ]),
  supportedAnalyticShapes: new Set<AnalyticShape>(),
  supportedEnvironmentKinds: new Set<SceneEnvironment['kind']>(['none', 'hdri']),
};

/**
 * Build the capabilities object. Slice 0 reuses the `pt-webgl` promise-ledger row's
 * `supportDetails` to avoid editing the closed core `BackendId` union before the
 * capability surface is final (master plan §5). A dedicated `'pt-webgl2'` ledger row
 * lands at cutover.
 */
export function buildCapabilities(
  causticStrategy: EngineCapabilities['causticStrategy'],
  maxBounces: number,
  maxSamplesPerPixel: number,
  supportsAuxBuffers: boolean,
): EngineCapabilities {
  return {
    supportsIncrementalScene: true,
    incrementalPatchSupport: {
      transform: true,
      positions: true,
      material: true,
      emitter: true,
      topology: false,
    },
    supportsAddRemovePrimitive: false,
    supportsAuxBuffers,
    accumulates: true,
    maxSamplesPerPixel,
    maxBounces,
    supportedAnalyticShapes: new Set(PT_WEBGL2_SUPPORT.supportedAnalyticShapes),
    supportedEmitterKinds: new Set(PT_WEBGL2_SUPPORT.supportedEmitterKinds),
    supportedPrimitiveKinds: new Set(PT_WEBGL2_SUPPORT.supportedPrimitiveKinds),
    supportedEnvironmentKinds: new Set(PT_WEBGL2_SUPPORT.supportedEnvironmentKinds),
    presentationMode: 'offscreen-texture',
    causticStrategy,
    supportDetails: BACKEND_PROMISE_LEDGER['pt-webgl'].supportDetails,
  };
}
