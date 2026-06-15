import type {
  AnalyticShape,
  EngineCapabilities,
  SceneEmitter,
  SceneEnvironment,
  ScenePrimitive,
  SupportSets,
} from '@vitrum/core';
import { BACKEND_PROMISE_LEDGER } from '@vitrum/core';

const PT_WEBGL2_BASE_SUPPORT_DETAILS = BACKEND_PROMISE_LEDGER['pt-webgl2'].supportDetails;

/**
 * The single source of truth for what this backend ingests. `buildCapabilities`
 * derives its `supported*Kinds` from THIS object, and `setScene` filters the scene
 * against it via `partitionSceneBySupport` — so advertised capability and ingestion
 * behaviour cannot drift (the pt-webgpu PT_WEBGPU_SUPPORT invariant).
 *
 * Slice 0 ships the mesh-focused set. Procedural sky is baked into the HDRI
 * equirect path; analytic shapes remain warn-and-skipped.
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
  supportedEnvironmentKinds: new Set<SceneEnvironment['kind']>(['none', 'hdri', 'procedural-sky']),
};

/**
 * Build the capabilities object from the explicit `pt-webgl2` promise-ledger row.
 */
export function buildCapabilities(
  causticStrategy: EngineCapabilities['causticStrategy'],
  maxBounces: number,
  maxSamplesPerPixel: number,
  supportsAuxBuffers: boolean,
  experimental?: { bdpt?: boolean; spectral?: boolean; oidn?: boolean },
): EngineCapabilities {
  // experimentalFeatures advertises OFF-default research paths that are HOST-DRIVEN
  // and live (not inert). A5 (2026-06-10): pt-webgl2-bdpt is added only when bdpt:true
  // because the light-subpath passes are now actually issued (see GlResources). The
  // photon-map/manifold-nee approximations are surfaced via causticStrategy + JSDoc,
  // not here. Undefined (no flags) → field omitted, matching the prior shape.
  const features = new Set<string>();
  if (experimental?.bdpt === true) features.add('pt-webgl2-bdpt');
  if (experimental?.spectral === true) features.add('pt-webgl2-spectral');
  if (experimental?.oidn === true) features.add('pt-webgl2-oidn-final');
  return {
    supportsIncrementalScene: true,
    incrementalPatchSupport: {
      transform: true,
      positions: true,
      material: true,
      emitter: true,
      topology: true,
    },
    supportsAddRemovePrimitive: true,
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
    // T3.G #30 — this backend exposes debug.pickPrimitive (CPU ray-cast click-to-pick).
    debugSurface: true,
    ...(features.size > 0 ? { experimentalFeatures: features } : {}),
    supportDetails: {
      ...PT_WEBGL2_BASE_SUPPORT_DETAILS,
      mutations: {
        ...PT_WEBGL2_BASE_SUPPORT_DETAILS.mutations,
        transform: 'fallback-rebuild',
        positions: 'fallback-rebuild',
        material: 'native',
        emitter: 'fallback-rebuild',
        topology: 'fallback-rebuild',
        addPrimitive: 'fallback-rebuild',
        removePrimitive: 'fallback-rebuild',
        environment: 'native',
      },
    },
  };
}
