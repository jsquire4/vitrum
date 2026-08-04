import {
  defineBackendSupportManifest,
  supportSetsFromManifest,
  type BackendSupportManifest,
  type BackendSupportMode,
  type EngineDenoiserMode,
} from '@vitrum/core';

import { VALID_DENOISERS } from './HybridEngineOptions.js';
import { WALKAROUND_NEURAL_DENOISER_SHAPE_REQUIREMENT } from './neural/shapeContract.js';
import { CONSUMED_MATERIAL_FIELD_DOCS } from './restir/consumedMaterialFields.js';

/**
 * Executable walkaround material contract.
 *
 * This table is deliberately exhaustive: adding a MaterialSpec field in core
 * fails this package's typecheck until walkaround classifies it. The runtime
 * parity guard below also ties every implemented row to the backend's actual
 * ingestion/transport documentation rather than to the static promise ledger.
 */
export const WALKAROUND_MATERIAL_SUPPORT: BackendSupportManifest['materials'] = Object.freeze({
  baseColor: 'approximate',
  roughness: 'approximate',
  metallic: 'approximate',
  emissive: 'native',
  emissiveIntensity: 'native',
  shadingModel: 'approximate',
  alphaMode: 'native',
  alphaCutoff: 'native',
  opacity: 'native',
  doubleSided: 'native',
  transmission: 'approximate',
  ior: 'approximate',
  attenuationColor: 'approximate',
  attenuationDistance: 'approximate',
  thickness: 'approximate',
  baseColorMap: 'approximate',
  normalMap: 'approximate',
  normalScale: 'approximate',
  roughnessMap: 'approximate',
  metallicMap: 'approximate',
  transmissionMap: 'approximate',
  thicknessMap: 'approximate',
  emissiveMap: 'native',
  alphaMap: 'native',
  aoMap: 'approximate',
  aoMapIntensity: 'approximate',
  clearcoatMap: 'native',
  clearcoatRoughnessMap: 'native',
  clearcoatNormalMap: 'native',
  clearcoatNormalScale: 'native',
  sheenColorMap: 'native',
  sheenRoughnessMap: 'native',
  iridescenceMap: 'native',
  iridescenceThicknessMap: 'native',
  anisotropyMap: 'native',
  specularColorMap: 'native',
  specularIntensityMap: 'native',
  bumpMap: 'approximate',
  bumpScale: 'approximate',
  displacementMap: 'approximate',
  displacementScale: 'approximate',
  displacementBias: 'approximate',
  displacementSubdivisions: 'approximate',
  lightMap: 'native',
  lightMapIntensity: 'native',
  sheen: 'native',
  sheenColor: 'native',
  sheenRoughness: 'native',
  clearcoat: 'native',
  clearcoatRoughness: 'native',
  iridescence: 'native',
  iridescenceIor: 'native',
  iridescenceThicknessRange: 'native',
  specularIntensity: 'native',
  specularColor: 'native',
  envMapIntensity: 'native',
  spectralAttenuation: 'approximate',
  dispersionAbbeNumber: 'approximate',
  scatteringCoefficient: 'approximate',
  scatteringAnisotropy: 'approximate',
  scatteringCoefficientRGB: 'approximate',
  frontLayer: 'approximate',
  backLayer: 'approximate',
  thinFilmStack: 'approximate',
  anisotropy: 'native',
  anisotropyRotation: 'native',
  extensions: 'native',
});

for (const [field, mode] of Object.entries(WALKAROUND_MATERIAL_SUPPORT)) {
  const documented = Object.prototype.hasOwnProperty.call(CONSUMED_MATERIAL_FIELD_DOCS, field);
  if ((mode !== 'unsupported') !== documented) {
    throw new Error(
      `walkaround support manifest drift: materials.${field}=${mode} but ` +
        `CONSUMED_MATERIAL_FIELD_DOCS ${documented ? 'contains' : 'omits'} it.`,
    );
  }
}

const WALKAROUND_PRIMITIVES = Object.freeze({
  mesh: 'native',
  'skinned-mesh': 'native',
  'instanced-mesh': 'native',
  analytic: 'fallback-generated-mesh',
} as const);

const WALKAROUND_EMITTERS = Object.freeze({
  directional: 'native',
  point: 'native',
  spot: 'native',
  'rect-area': 'native',
  'disc-area': 'native',
  'mesh-area': 'native',
} as const);

const WALKAROUND_ENVIRONMENTS = Object.freeze({
  none: 'native',
  hdri: 'native',
  'procedural-sky': 'approximate',
} as const);

const WALKAROUND_ANALYTIC_SHAPES = Object.freeze({
  sphere: 'fallback-generated-mesh',
  box: 'fallback-generated-mesh',
  capsule: 'fallback-generated-mesh',
  cylinder: 'fallback-generated-mesh',
  'h-channel-came': 'fallback-generated-mesh',
} as const);

const WALKAROUND_SHADOWS = Object.freeze({
  primitiveCastShadow: 'native',
  emitterCastShadow: 'native',
} as const);

const WALKAROUND_MUTATIONS = Object.freeze({
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
} as const);

const WALKAROUND_MATERIAL_PROFILES = Object.freeze({
  deltaTransmission: 'approximate',
  roughTransmission: 'approximate',
  layeredTransmission: 'approximate',
  normalMappedTransmission: 'approximate',
  participatingMedia: 'approximate',
  faceLayers: 'approximate',
} as const);

const WALKAROUND_CAUSTIC_STRATEGIES = Object.freeze({
  'refractive-trace': {
    mode: 'approximate',
    estimatorScope:
      'camera-visible diffuse receiver <- up-to-four specular transmission interfaces <- directional emitter; two bounded stratified RGB candidates, not Newton manifold NEE',
    emitterKinds: {
      directional: 'native',
      point: 'unsupported',
      spot: 'unsupported',
      'rect-area': 'unsupported',
      'disc-area': 'unsupported',
      'mesh-area': 'unsupported',
      environment: 'unsupported',
    },
    volumeScattering: 'unsupported',
    incompatibleFeatures: ['manifold-nee', 'photon-map'],
  },
  'manifold-nee': {
    mode: 'approximate',
    estimatorScope:
      'camera-visible diffuse receiver <- one-to-eight mapped rough/delta transmission events <- sampled explicit or environment endpoint; bounded SMS inverse-basin correction',
    emitterKinds: {
      directional: 'native',
      point: 'native',
      spot: 'native',
      'rect-area': 'native',
      'disc-area': 'native',
      'mesh-area': 'native',
      environment: 'native',
    },
    volumeScattering: 'unsupported',
    incompatibleFeatures: ['refractive-trace', 'photon-map'],
  },
} as const);

const WALKAROUND_BASE_DENOISERS = Object.freeze({
  none: 'native',
  auto: 'native',
  atrous: 'native',
  'atrous-variance': 'native',
  'svgf-real': 'native',
  bmfr: 'native',
  'oidn-final': 'native',
  neural: 'native',
} as const satisfies Readonly<
  Record<(typeof VALID_DENOISERS)[number], BackendSupportMode>
> satisfies Readonly<Record<EngineDenoiserMode, BackendSupportMode>>);

export type WalkaroundSupportTier = 'full' | 'lite';
export type WalkaroundNeuralCertification = 'absent' | 'uncertified' | 'certified';

export interface WalkaroundSupportProfile {
  readonly tier: WalkaroundSupportTier;
  readonly neuralCertification: WalkaroundNeuralCertification;
  readonly oidnModelAvailable: boolean;
  /** Instance-resolved end-to-end bulk-medium nesting capacity. */
  readonly maxNestedMedia?: 4 | 8;
}

function createWalkaroundSupportManifest(
  profile: WalkaroundSupportProfile,
): BackendSupportManifest {
  const neuralAvailable = profile.tier === 'full' && profile.neuralCertification === 'certified';
  const maxNestedMedia = profile.maxNestedMedia ?? 8;
  const denoisers = {
    ...WALKAROUND_BASE_DENOISERS,
    bmfr: profile.tier === 'lite' ? 'unsupported' : 'native',
    neural: neuralAvailable ? 'native' : 'unsupported',
    'oidn-final': profile.oidnModelAvailable ? 'native' : 'unsupported',
  } as const satisfies Readonly<Record<EngineDenoiserMode, BackendSupportMode>>;

  return defineBackendSupportManifest({
    primitives: WALKAROUND_PRIMITIVES,
    emitters: WALKAROUND_EMITTERS,
    environments: WALKAROUND_ENVIRONMENTS,
    analyticShapes: WALKAROUND_ANALYTIC_SHAPES,
    materials: WALKAROUND_MATERIAL_SUPPORT,
    materialProfiles: WALKAROUND_MATERIAL_PROFILES,
    shadows: WALKAROUND_SHADOWS,
    denoisers,
    denoiserSpatialShapeRequirements: {
      neural: WALKAROUND_NEURAL_DENOISER_SHAPE_REQUIREMENT,
    },
    motionVectors: {
      units: 'pixels',
      direction: 'previous-minus-current',
      geometry: 'camera-only',
      sceneMutationPolicy: 'reset-history',
    },
    causticStrategies: WALKAROUND_CAUSTIC_STRATEGIES,
    mutations: WALKAROUND_MUTATIONS,
    bounceSemantics: {
      kind: 'ddgi-feedback',
      directOnlyValue: 1,
      multiBounceEquilibriumValue: 2,
      inactiveWhenLayerDisabled: 'ddgi',
    },
    opticalMedia: {
      maxNestedMedia,
      topology: 'closed-oriented-disjoint-or-nested',
      overflowPolicy: 'reject-scene',
    },
  });
}

const SUPPORT_MANIFEST_CACHE = new Map<string, BackendSupportManifest>();

/**
 * Resolve the exact immutable manifest published by one engine instance.
 * The twelve tier/certification/model combinations are cached and shared.
 */
export function walkaroundSupportManifest(
  profile: WalkaroundSupportProfile,
): BackendSupportManifest {
  const key =
    `${profile.tier}:${profile.neuralCertification}:` +
    `${profile.oidnModelAvailable ? 'oidn' : 'no-oidn'}:` +
    `media-${profile.maxNestedMedia ?? 8}`;
  const cached = SUPPORT_MANIFEST_CACHE.get(key);
  if (cached != null) return cached;
  const manifest = createWalkaroundSupportManifest(profile);
  SUPPORT_MANIFEST_CACHE.set(key, manifest);
  return manifest;
}

export function walkaroundSupportSets(profile: WalkaroundSupportProfile) {
  return supportSetsFromManifest(walkaroundSupportManifest(profile));
}

/**
 * Static-ledger parity profile: every host-provisioned full-tier denoiser has
 * its required certified model asset.
 */
export const WALKAROUND_FULL_CERTIFIED_SUPPORT_MANIFEST = walkaroundSupportManifest({
  tier: 'full',
  neuralCertification: 'certified',
  oidnModelAvailable: true,
  // Static full-feature ledger includes the four-deep NRC/refractive suffix.
  maxNestedMedia: 4,
});
