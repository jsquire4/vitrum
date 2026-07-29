import {
  defineBackendSupportManifest,
  supportSetsFromManifest,
  type BackendSupportManifest,
} from '@vitrum/core';

/**
 * Executable pt-webgl2 material contract. The materials texture packer,
 * texture-atlas builder, mutation warning path, live capabilities, and
 * conformance tests all consume this backend-local classification.
 *
 * This record is deliberately exhaustive. A new `MaterialSpec` key breaks this
 * package's typecheck until the WebGL2 renderer classifies it.
 */
export const PT_WEBGL2_MATERIAL_SUPPORT: BackendSupportManifest['materials'] =
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
    anisotropyMap: 'native',
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
    scatteringCoefficientRGB: 'approximate',
    frontLayer: 'native',
    backLayer: 'native',
    thinFilmStack: 'native',
    anisotropy: 'native',
    anisotropyRotation: 'native',
    // The shared `skipEmitter === true` lane suppresses implicit mesh-light
    // classification while preserving camera-visible surface emission.
    extensions: 'native',
  });

const ALL_EMITTERS_NATIVE = Object.freeze({
  directional: 'native',
  point: 'native',
  spot: 'native',
  'rect-area': 'native',
  'disc-area': 'native',
  'mesh-area': 'native',
} as const);

const ANALYTIC_FALLBACKS = Object.freeze({
  sphere: 'fallback-generated-mesh',
  box: 'fallback-generated-mesh',
  capsule: 'fallback-generated-mesh',
  cylinder: 'fallback-generated-mesh',
  'h-channel-came': 'fallback-generated-mesh',
} as const);

/**
 * Backend-local source of truth for live capability reporting and scene
 * acceptance. The core promise ledger is intentionally not read here.
 */
export const PT_WEBGL2_SUPPORT_MANIFEST = defineBackendSupportManifest({
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
  emitters: ALL_EMITTERS_NATIVE,
  environments: {
    none: 'native',
    hdri: 'native',
    'procedural-sky': 'approximate',
  },
  analyticShapes: ANALYTIC_FALLBACKS,
  materials: PT_WEBGL2_MATERIAL_SUPPORT,
  shadows: {
    primitiveCastShadow: 'native',
    emitterCastShadow: 'native',
  },
  denoisers: {
    none: 'native',
    auto: 'native',
    atrous: 'unsupported',
    'atrous-variance': 'unsupported',
    'svgf-real': 'unsupported',
    bmfr: 'unsupported',
    'oidn-final': 'native',
    neural: 'unsupported',
  },
  causticStrategies: {
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
      volumeScattering: 'native',
      incompatibleFeatures: [],
    },
  },
  thinFilmLayerLimit: 35,
  mutations: {
    transform: 'native',
    positions: 'native',
    material: 'native',
    emitter: 'native',
    topology: 'fallback-rebuild',
    addPrimitive: 'fallback-rebuild',
    removePrimitive: 'fallback-rebuild',
    environment: 'native',
    resize: 'native',
    lighting: 'fallback-rebuild',
  },
});

/**
 * Coarse sets are derived, not separately declared. `setScene` partitions
 * against this exact object and `buildCapabilities` republishes copies of the
 * same sets.
 */
export const PT_WEBGL2_SUPPORT =
  supportSetsFromManifest(PT_WEBGL2_SUPPORT_MANIFEST);
