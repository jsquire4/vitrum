import type {
  AnalyticShape,
  EngineCapabilities,
  EngineFeatureId,
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
 * Procedural sky is baked into the HDRI equirect path. Analytic primitives are
 * accepted at the contract boundary and tessellated to generated mesh fallbacks
 * before the WebGL2 triangle-BVH packer ingests the scene.
 */
export const PT_WEBGL2_SUPPORT: Required<SupportSets> = {
  supportedPrimitiveKinds: new Set<ScenePrimitive['kind']>([
    'mesh',
    'analytic',
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
  supportedAnalyticShapes: new Set<AnalyticShape>([
    'sphere',
    'box',
    'capsule',
    'cylinder',
    'h-channel-came',
  ]),
  supportedEnvironmentKinds: new Set<SceneEnvironment['kind']>(['none', 'hdri', 'procedural-sky']),
};

/**
 * Build the capabilities object from the explicit `pt-webgl2` promise-ledger row.
 */
export function buildCapabilities(
  denoiser: 'none' | 'oidn-final',
  maxBounces: number,
  maxSamplesPerPixel: number,
  _supportsAuxBuffers: boolean,
  selected?: { bdpt?: boolean; spectral?: boolean; sampling?: 'pcg' | 'sobol' },
): EngineCapabilities {
  const activeFeatures = new Set<EngineFeatureId>();
  if (selected?.bdpt === true) activeFeatures.add('pt-webgl2-bdpt');
  if (selected?.spectral === true) activeFeatures.add('pt-webgl2-spectral');
  if (selected?.sampling === 'sobol') activeFeatures.add('pt-webgl2-sobol-sampling');
  if (denoiser === 'oidn-final') activeFeatures.add('pt-webgl2-oidn-final');

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
    // The renderer may allocate normalDepth/albedo MRTs for denoisers, but the
    // EngineCapabilities flag is stricter: it requires variance + motionVectors.
    supportsAuxBuffers: false,
    accumulates: true,
    maxSamplesPerPixel,
    maxBounces,
    supportedAnalyticShapes: new Set(PT_WEBGL2_SUPPORT.supportedAnalyticShapes),
    supportedEmitterKinds: new Set(PT_WEBGL2_SUPPORT.supportedEmitterKinds),
    supportedPrimitiveKinds: new Set(PT_WEBGL2_SUPPORT.supportedPrimitiveKinds),
    supportedEnvironmentKinds: new Set(PT_WEBGL2_SUPPORT.supportedEnvironmentKinds),
    presentationMode: 'offscreen-texture',
    causticStrategy: selected?.bdpt === true ? 'bdpt' : 'none',
    // T3.G #30 — this backend exposes debug.pickPrimitive (CPU ray-cast click-to-pick).
    debugSurface: true,
    activeFeatures,
    supportDetails: {
      ...PT_WEBGL2_BASE_SUPPORT_DETAILS,
      causticStrategies: {
        ...PT_WEBGL2_BASE_SUPPORT_DETAILS.causticStrategies,
        bdpt: {
          mode: 'native',
          estimatorScope:
            'bounded general BDPT: power-weighted light subpaths store 1..8 vertices (default 4); finite-emitter c=0 plus c>=1 light/eye surface and participating-medium vertices carry an exact eight-entry nested homogeneous-medium stack, RGB or hero-wavelength Beer visibility, authored sigma_a/sigma_s, and Henyey-Greenstein phase transport through Veach power-heuristic MIS; thicknessMap modulates the core contract\'s authored surface-volume attenuation; distant paths are disjointly owned by primary c=0 NEE, camera/delta forward escape, or c>=1 BDPT',
          emitterKinds: {
            directional: 'native',
            point: 'native',
            spot: 'native',
            'rect-area': 'native',
            'disc-area': 'native',
            'mesh-area': 'native',
            environment: 'native',
          },
          // This is the complete spatial-medium domain expressible by the core
          // contract: closed homogeneous boundaries, authored absorption /
          // scattering / HG anisotropy, and texture-modulated thickness.
          volumeScattering: 'native',
          incompatibleFeatures: [],
        },
      },
      mutations: {
        ...PT_WEBGL2_BASE_SUPPORT_DETAILS.mutations,
        transform: 'native',
        positions: 'native',
        material: 'native',
        emitter: 'native',
        topology: 'fallback-rebuild',
        addPrimitive: 'fallback-rebuild',
        removePrimitive: 'fallback-rebuild',
        environment: 'native',
      },
    },
  };
}
