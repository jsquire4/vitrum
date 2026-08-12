import {
  defineBackendSupportManifest,
  supportSetsFromManifest,
  type BackendSupportManifest,
  type BackendSupportMode,
  type EngineDenoiserMode,
} from '@vitrum/core';
import { SOBOL_DIMENSION_COUNT } from '@vitrum/shared-samplers';

import { THIN_FILM_LAYER_LIMIT } from './scene/materialPacking.js';
import type { PtWebgpuTraceTier } from './traceTier.js';

/**
 * Executable full-tier material contract.
 *
 * This backend-local table is intentionally exhaustive. Adding a MaterialSpec
 * field in core makes this file fail typecheck until pt-webgpu classifies the
 * new field. The cross-backend promise ledger is a static reference checked by
 * tests; runtime support, validation, and capability reporting use this table.
 */
const PT_WEBGPU_FULL_MATERIALS: BackendSupportManifest['materials'] =
  Object.freeze({
    baseColor: 'native',
    roughness: 'native',
    metallic: 'native',
    emissive: 'native',
    emissiveIntensity: 'native',
    shadingModel: 'approximate',
    alphaMode: 'native',
    alphaCutoff: 'native',
    opacity: 'native',
    doubleSided: 'native',
    transmission: 'native',
    ior: 'native',
    attenuationColor: 'native',
    attenuationDistance: 'native',
    thickness: 'approximate',
    baseColorMap: 'native',
    normalMap: 'native',
    normalScale: 'native',
    roughnessMap: 'native',
    metallicMap: 'native',
    transmissionMap: 'native',
    thicknessMap: 'approximate',
    emissiveMap: 'native',
    alphaMap: 'native',
    aoMap: 'native',
    aoMapIntensity: 'native',
    clearcoatMap: 'native',
    clearcoatRoughnessMap: 'native',
    clearcoatNormalMap: 'native',
    clearcoatNormalScale: 'native',
    sheenColorMap: 'native',
    sheenRoughnessMap: 'native',
    iridescenceMap: 'native',
    iridescenceThicknessMap: 'native',
    anisotropyMap: 'approximate',
    specularColorMap: 'native',
    specularIntensityMap: 'native',
    bumpMap: 'native',
    bumpScale: 'native',
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
    spectralAttenuation: 'native',
    dispersionAbbeNumber: 'native',
    scatteringCoefficient: 'native',
    scatteringAnisotropy: 'native',
    scatteringCoefficientRGB: 'native',
    frontLayer: 'native',
    backLayer: 'native',
    thinFilmStack: 'native',
    anisotropy: 'approximate',
    anisotropyRotation: 'approximate',
    // The shared `skipEmitter === true` lane suppresses implicit mesh-light
    // synthesis while preserving camera-visible surface emission.
    extensions: 'native',
  });

/**
 * Lite composes the scalar material payload but has no material texture-array
 * bindings. These overrides are the exact profile boundary enforced by
 * setScene/updatePrimitive and reported to hosts.
 */
const PT_WEBGPU_LITE_MATERIALS: BackendSupportManifest['materials'] =
  Object.freeze({
    ...PT_WEBGPU_FULL_MATERIALS,
    baseColorMap: 'unsupported',
    normalMap: 'unsupported',
    normalScale: 'unsupported',
    roughnessMap: 'unsupported',
    metallicMap: 'unsupported',
    transmissionMap: 'unsupported',
    thicknessMap: 'unsupported',
    emissiveMap: 'unsupported',
    alphaMap: 'unsupported',
    aoMap: 'unsupported',
    aoMapIntensity: 'unsupported',
    clearcoatMap: 'unsupported',
    clearcoatRoughnessMap: 'unsupported',
    clearcoatNormalMap: 'unsupported',
    clearcoatNormalScale: 'unsupported',
    sheenColorMap: 'unsupported',
    sheenRoughnessMap: 'unsupported',
    iridescenceMap: 'unsupported',
    iridescenceThicknessMap: 'unsupported',
    anisotropyMap: 'unsupported',
    specularColorMap: 'unsupported',
    specularIntensityMap: 'unsupported',
    bumpMap: 'unsupported',
    bumpScale: 'unsupported',
    lightMap: 'unsupported',
    lightMapIntensity: 'unsupported',
    alphaMode: 'unsupported',
    alphaCutoff: 'unsupported',
    opacity: 'unsupported',
    envMapIntensity: 'unsupported',
    anisotropy: 'unsupported',
    anisotropyRotation: 'unsupported',
    // Lite transports surface refraction plus absorption-only Beer attenuation.
    // It has no medium collision/phase walk, so authored scattering must select
    // the full tier instead of being accepted and silently discarded.
    scatteringCoefficient: 'unsupported',
    scatteringAnisotropy: 'unsupported',
    scatteringCoefficientRGB: 'unsupported',
    // Scalar tint/roughness are consumed, but lite omits per-face normal maps.
    frontLayer: 'approximate',
    backLayer: 'approximate',
  });

const PT_WEBGPU_SHADOWS = Object.freeze({
  primitiveCastShadow: 'native',
  emitterCastShadow: 'native',
} as const);

const PT_WEBGPU_DENOISERS = Object.freeze({
  none: 'native',
  auto: 'native',
  atrous: 'unsupported',
  'atrous-variance': 'unsupported',
  'svgf-real': 'unsupported',
  bmfr: 'unsupported',
  'oidn-final': 'native',
  neural: 'unsupported',
} as const satisfies Readonly<Record<EngineDenoiserMode, BackendSupportMode>>);

const PT_WEBGPU_SAMPLING_SEQUENCES = Object.freeze({
  default: 'pcg',
  modes: Object.freeze({
    pcg: 'native',
    sobol: 'native',
  }),
  sobol: Object.freeze({
    lowDiscrepancyDimensions: SOBOL_DIMENSION_COUNT,
    continuation: 'independent-pcg',
    sampleBlockSize: 65_536,
    frameIndexPeriod: 4_294_967_296,
  }),
} as const);

const PT_WEBGPU_FULL_MUTATIONS = Object.freeze({
  transform: 'native',
  positions: 'native',
  material: 'fallback-rebuild',
  emitter: 'native',
  topology: 'native',
  addPrimitive: 'fallback-rebuild',
  removePrimitive: 'fallback-rebuild',
  environment: 'native',
  resize: 'native',
  lighting: 'native',
} as const);

const PT_WEBGPU_LITE_MUTATIONS = Object.freeze({
  ...PT_WEBGPU_FULL_MUTATIONS,
  transform: 'fallback-rebuild',
  positions: 'fallback-rebuild',
  material: 'fallback-rebuild',
  topology: 'fallback-rebuild',
} as const);

/**
 * The bounded BDPT strategy implemented by the full-tier allocation.
 *
 * Construction validation and capability activation consume this same object;
 * keep numeric limits and strategy requirements here rather than duplicating
 * them in option gates.
 */
export const PT_WEBGPU_BDPT_SUPPORT = Object.freeze({
  mode: 'bounded-explicit-connections',
  maxLightVertices: 8,
  maxEyeVertices: 8,
  pureEyeStrategy: 'partitioned-eye-estimator',
  cameraSplatStrategy: 'native',
  misDenominator: 'sampled-strategies-only',
} as const satisfies NonNullable<
  BackendSupportManifest['bidirectionalPathTracing']
>);

/** Full-tier executable support and fidelity evidence. */
export const PT_WEBGPU_FULL_SUPPORT_MANIFEST = defineBackendSupportManifest({
  bounceSemantics: {
    kind: 'path-depth',
    perFrameControl: 'finite-path-depth',
  },
  primitives: {
    mesh: 'native',
    'skinned-mesh': 'native',
    'instanced-mesh': 'native',
    analytic: 'native',
  },
  emitters: {
    directional: 'native',
    point: 'native',
    spot: 'native',
    'rect-area': 'native',
    'disc-area': 'native',
    'mesh-area': 'native',
  },
  environments: {
    none: 'native',
    hdri: 'native',
    // Preetham sky is baked to the finite-resolution environment map.
    'procedural-sky': 'approximate',
  },
  analyticShapes: {
    sphere: 'native',
    box: 'native',
    capsule: 'native',
    cylinder: 'native',
    'h-channel-came': 'native',
  },
  materials: PT_WEBGPU_FULL_MATERIALS,
  shadows: PT_WEBGPU_SHADOWS,
  denoisers: PT_WEBGPU_DENOISERS,
  motionVectors: {
    units: 'pixels',
    direction: 'current-minus-previous',
    geometry: 'camera-only',
    sceneMutationPolicy: 'reset-history',
  },
  mutations: PT_WEBGPU_FULL_MUTATIONS,
  causticStrategies: {
    'manifold-nee': {
      mode: 'native',
      estimatorScope:
        'camera-visible finite surface receiver <- one-to-eight planar geometric-normal mesh/instanced/skinned delta interfaces <- sampled explicit or environment endpoint; bounded Newton/SMS solve; analytic delta interfaces, varying interface normals, and normal/bump/layer-normal mapped interfaces fail closed before upload',
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
      incompatibleFeatures: [],
    },
    'photon-map': {
      mode: 'native',
      estimatorScope:
        'camera-visible finite surface or homogeneous-medium collision receiver <- one-or-more production delta surface events <- sampled light; surface uses a disk density and volume uses a medium-identity-filtered HG sphere density with independent progressive state',
      emitterKinds: {
        directional: 'native',
        point: 'native',
        spot: 'native',
        'rect-area': 'native',
        'disc-area': 'native',
        'mesh-area': 'native',
        environment: 'native',
      },
      volumeScattering: 'native',
      incompatibleFeatures: [],
    },
  },
  bidirectionalPathTracing: PT_WEBGPU_BDPT_SUPPORT,
  samplingSequences: PT_WEBGPU_SAMPLING_SEQUENCES,
  thinFilmLayerLimit: THIN_FILM_LAYER_LIMIT,
});

/** Lite-tier executable support and fidelity evidence. */
export const PT_WEBGPU_LITE_SUPPORT_MANIFEST = defineBackendSupportManifest({
  bounceSemantics: {
    kind: 'path-depth',
    perFrameControl: 'finite-path-depth',
  },
  primitives: {
    mesh: 'native',
    'skinned-mesh': 'native',
    'instanced-mesh': 'native',
    analytic: 'fallback-generated-mesh',
  },
  emitters: {
    directional: 'native',
    point: 'native',
    spot: 'native',
    'rect-area': 'native',
    'disc-area': 'native',
    'mesh-area': 'unsupported',
  },
  environments: {
    none: 'native',
    hdri: 'native',
    'procedural-sky': 'approximate',
  },
  analyticShapes: {
    sphere: 'fallback-generated-mesh',
    box: 'fallback-generated-mesh',
    capsule: 'fallback-generated-mesh',
    cylinder: 'fallback-generated-mesh',
    'h-channel-came': 'fallback-generated-mesh',
  },
  materials: PT_WEBGPU_LITE_MATERIALS,
  shadows: PT_WEBGPU_SHADOWS,
  denoisers: PT_WEBGPU_DENOISERS,
  mutations: PT_WEBGPU_LITE_MUTATIONS,
  samplingSequences: PT_WEBGPU_SAMPLING_SEQUENCES,
  thinFilmLayerLimit: THIN_FILM_LAYER_LIMIT,
});

export const PT_WEBGPU_FULL_SUPPORT =
  supportSetsFromManifest(PT_WEBGPU_FULL_SUPPORT_MANIFEST);
export const PT_WEBGPU_LITE_SUPPORT =
  supportSetsFromManifest(PT_WEBGPU_LITE_SUPPORT_MANIFEST);

export function ptWebgpuSupportManifest(
  traceTier: PtWebgpuTraceTier,
): BackendSupportManifest {
  return traceTier === 'lite'
    ? PT_WEBGPU_LITE_SUPPORT_MANIFEST
    : PT_WEBGPU_FULL_SUPPORT_MANIFEST;
}

export function ptWebgpuSupportSets(traceTier: PtWebgpuTraceTier) {
  return traceTier === 'lite'
    ? PT_WEBGPU_LITE_SUPPORT
    : PT_WEBGPU_FULL_SUPPORT;
}

function isImplemented(mode: BackendSupportMode): boolean {
  return mode !== 'unsupported';
}

/** All public denoiser enum values, and the subset this backend accepts. */
export const PT_WEBGPU_DENOISER_VALUES = Object.freeze(
  Object.keys(PT_WEBGPU_FULL_SUPPORT_MANIFEST.denoisers) as EngineDenoiserMode[],
);
export const PT_WEBGPU_IMPLEMENTED_DENOISER_VALUES = Object.freeze(
  PT_WEBGPU_DENOISER_VALUES.filter(
    (mode) => isImplemented(PT_WEBGPU_FULL_SUPPORT_MANIFEST.denoisers[mode]),
  ),
);

/** Creation-time sampling modes are taken from the same live manifest. */
export const PT_WEBGPU_SAMPLING_VALUES = Object.freeze(
  Object.entries(PT_WEBGPU_FULL_SUPPORT_MANIFEST.samplingSequences!.modes)
    .filter(([, mode]) => mode != null && isImplemented(mode))
    .map(([mode]) => mode as 'pcg' | 'sobol'),
);
