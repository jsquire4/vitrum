import type {
  EngineCapabilities,
  EngineFeatureId,
} from '@vitrum/core';
import {
  PT_WEBGL2_SUPPORT,
  PT_WEBGL2_SUPPORT_MANIFEST,
} from './supportManifest.js';

export { PT_WEBGL2_SUPPORT } from './supportManifest.js';

/**
 * Build live capabilities from the backend-local executable support manifest.
 */
export function buildCapabilities(
  denoiser: 'none' | 'oidn-final',
  maxBounces: number,
  maxSamplesPerPixel: number,
  selected?: {
    bdpt?: boolean;
    spectral?: boolean;
    sampling?: 'pcg' | 'sobol';
    debug?: boolean;
  },
): EngineCapabilities {
  const mutationDetails = PT_WEBGL2_SUPPORT_MANIFEST.mutations;
  const mutationAccepted = (
    mode: (typeof mutationDetails)[keyof typeof mutationDetails],
  ): boolean => mode !== 'unsupported';
  const activeFeatures = new Set<EngineFeatureId>();
  if (selected?.bdpt === true) activeFeatures.add('pt-webgl2-bdpt');
  if (selected?.spectral === true) activeFeatures.add('pt-webgl2-spectral');
  if (selected?.sampling === 'sobol') activeFeatures.add('pt-webgl2-sobol-sampling');
  if (denoiser === 'oidn-final') activeFeatures.add('pt-webgl2-oidn-final');

  return {
    supportsIncrementalScene:
      mutationAccepted(mutationDetails.transform) ||
      mutationAccepted(mutationDetails.positions) ||
      mutationAccepted(mutationDetails.material) ||
      mutationAccepted(mutationDetails.emitter) ||
      mutationAccepted(mutationDetails.topology),
    incrementalPatchSupport: {
      transform: mutationAccepted(mutationDetails.transform),
      positions: mutationAccepted(mutationDetails.positions),
      material: mutationAccepted(mutationDetails.material),
      emitter: mutationAccepted(mutationDetails.emitter),
      topology: mutationAccepted(mutationDetails.topology),
    },
    supportsAddRemovePrimitive:
      mutationAccepted(mutationDetails.addPrimitive) &&
      mutationAccepted(mutationDetails.removePrimitive),
    // The renderer may allocate normalDepth/albedo MRTs for denoisers, but the
    // EngineCapabilities flag is stricter: it requires variance + motionVectors.
    supportsAuxBuffers: PT_WEBGL2_SUPPORT_MANIFEST.motionVectors != null,
    accumulates: true,
    maxSamplesPerPixel,
    maxBounces,
    supportedAnalyticShapes: new Set(PT_WEBGL2_SUPPORT.supportedAnalyticShapes),
    supportedEmitterKinds: new Set(PT_WEBGL2_SUPPORT.supportedEmitterKinds),
    supportedPrimitiveKinds: new Set(PT_WEBGL2_SUPPORT.supportedPrimitiveKinds),
    supportedEnvironmentKinds: new Set(PT_WEBGL2_SUPPORT.supportedEnvironmentKinds),
    presentationMode: 'offscreen-texture',
    causticStrategy: selected?.bdpt === true ? 'bdpt' : 'none',
    // Inverse is a finite-difference niche on this backend. Path replay is
    // not implemented; requesting it throws at session construction.
    inverseRendering: {
      methods: {
        'finite-difference': 'native',
        'path-replay': 'unsupported',
      },
    },
    // T3.G #30 — exposed only for an explicitly debug-enabled engine.
    debugSurface: selected?.debug === true,
    activeFeatures,
    supportDetails: PT_WEBGL2_SUPPORT_MANIFEST,
  };
}
