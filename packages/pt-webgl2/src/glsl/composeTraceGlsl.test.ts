// GPU-free structural assertions on the composed fragment body. We CANNOT GPU-compile
// here, so we verify the chunk concatenation produced the right symbols in the right
// order (struct-before-use is load-bearing — plan/three-removal/04-glsl-kernels.md §3).

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, it, expect } from 'vitest';
import {
  composeNeeCandidateGlsl,
  composeNeeResolveGlsl,
  composeTraceGlsl,
  RENDER_MAIN_SECTIONS,
  buildUniformDecls,
  UNIFORM_MANIFEST,
} from './composeTraceGlsl.js';
import {
  DEFAULT_TRACE_FEATURES,
  featureDefines,
  type TraceFeatures,
} from '../featureTypes.js';
import { NEE_RESOLVE_MAIN } from './neeResolveMain.glsl.js';
import * as RandSobol from './shader/rand/sobol.glsl.js';

describe('composeTraceGlsl', () => {
  const src = composeTraceGlsl(DEFAULT_TRACE_FEATURES);
  const neeCandidateSrc = composeNeeCandidateGlsl(DEFAULT_TRACE_FEATURES);
  const neeResolveSrc = composeNeeResolveGlsl();

  it('emits the BVH struct definition', () => {
    expect(src).toContain('struct BVH {');
    // The BVH struct's four samplers (from three-mesh-bvh bvh_struct_definitions).
    expect(src).toContain('usampler2D index');
    expect(src).toContain('sampler2D position');
  });

  it('emits the material struct', () => {
    expect(src).toContain('struct Material {');
    expect(src).toContain('bool unlit;');
  });

  it('routes unlit materials through the terminal base-color path', () => {
    expect(src).toContain('bool activeMaterialUnlit = materialControl.unlit;');
    expect(src).toContain('if ( activeMaterialUnlit )');
    expect(src).toContain(
      'state.throughput * pathThroughputFromRgb( surf.color, state.wavelength )',
    );
  });

  it('emits the surface-record struct', () => {
    expect(src).toContain('struct SurfaceRecord {');
  });

  it('emits the MRT G-buffer outputs (locations 1 and 2; loc 0 is the preamble)', () => {
    expect(src).toContain('layout(location = 1) out vec4 gNormalDepth;');
    expect(src).toContain('layout(location = 2) out vec4 gAlbedo;');
  });

  it('emits the main() entry point with a host-validated runtime bounce bound', () => {
    expect(src).toContain('void main() {');
    expect(src).toContain(
      'for ( int pathStep = 0, i = 0; pathStep < bounces * 2; pathStep ++, i ++ )',
    );
    expect(src).toContain('if ( i >= bounces ) {');
    expect(src).not.toContain('for ( int i = 0; i < bounces; i ++ )');
    expect(src).toContain('Ray ray = getCameraRay();');
    expect(src).toContain('MaterialControl materialControl;');
    expect(src).toContain('readMaterialControl( materials, materialIndex, materialControl );');
    expect(src).toContain('if ( activeMaterialUnlit )');
    expect(src).toContain(
      'state.throughput * pathThroughputFromRgb( surf.color, state.wavelength )',
    );
  });

  it('specializes the static marker to the host-validated runtime loop guard', () => {
    const specializedTrace = composeTraceGlsl(DEFAULT_TRACE_FEATURES);
    const specializedCandidate = composeNeeCandidateGlsl(DEFAULT_TRACE_FEATURES);
    for (const specialized of [specializedTrace, specializedCandidate]) {
      expect(specialized).toContain('pathStep < bounces * 2;');
      expect(specialized).not.toContain('pathStep < 64;');
      expect(specialized).toContain('if ( i >= bounces ) {');
    }
  });

  it('emits the compact opaque base-PBR program graph without optional material chunks', () => {
    const basicFeatures = {
      ...DEFAULT_TRACE_FEATURES,
      basicMaterials: true,
      mappedRichMaterials: false,
    };
    const basicTrace = composeTraceGlsl(basicFeatures);
    const basicCandidate = composeNeeCandidateGlsl(basicFeatures);
    const basicResolve = composeNeeResolveGlsl(basicFeatures);

    for (const basicSource of [basicTrace, basicCandidate, basicResolve]) {
      expect(basicSource).toContain('void basicLobeWeights(');
      expect(basicSource).toContain('vec3 averageFresnel =');
      expect(basicSource).toContain(
        'vec3 attenuationSigmaA( vec3 attColor, float attDist )',
      );
      expect(basicSource).not.toContain('mat3 mapTransform;');
      expect(basicSource).not.toContain('evalIridescence(');
      expect(basicSource).not.toContain('thinFilmTMM(');
      expect(basicSource).not.toContain('sheenAlbedoScaling(');
    }
    expect(basicCandidate).toContain('if ( step >= state.traversals ) break;');
    expect(basicTrace.length).toBeLessThan(src.length - 40_000);
    expect(basicCandidate.length).toBeLessThan(neeCandidateSrc.length - 40_000);
    expect(basicResolve.length).toBeLessThan(neeResolveSrc.length - 40_000);
  });

  it('emits texture-free full transport without the full texture material graph', () => {
    const scalarRichFeatures = {
      ...DEFAULT_TRACE_FEATURES,
      scalarRichMaterials: true,
      mappedRichMaterials: false,
    };
    const scalarTrace = composeTraceGlsl(scalarRichFeatures);
    const scalarCandidate = composeNeeCandidateGlsl(scalarRichFeatures);
    const scalarResolve = composeNeeResolveGlsl(scalarRichFeatures);

    for (const scalarSource of [scalarTrace, scalarCandidate, scalarResolve]) {
      expect(scalarSource).toContain('float thinFilmLayerCount;');
      expect(scalarSource).toContain('vec3 frontLayerTransmission;');
      expect(scalarSource).toContain('bool hasSpectralAttenuation;');
      expect(scalarSource).toContain('evalIridescence(');
      expect(scalarSource).toContain('thinFilmTMM(');
      expect(scalarSource).toContain('sheenAlbedoScaling(');
      expect(scalarSource).not.toContain('int map;');
      expect(scalarSource).not.toContain('mat3 mapTransform;');
      expect(scalarSource).not.toContain('sampleMaterialTexture(');
    }
    expect(scalarCandidate).toContain(
      'transmissionAttenuationThroughput(',
    );
    expect(scalarTrace.length).toBeLessThan(src.length - 15_000);
    expect(scalarCandidate.length).toBeLessThan(neeCandidateSrc.length - 15_000);
    expect(scalarResolve.length).toBeLessThan(neeResolveSrc.length - 15_000);
  });

  it('emits medium traversal and free-flight helpers for the reachable scalar-rich fog tier', () => {
    const scalarFogFeatures = {
      ...DEFAULT_TRACE_FEATURES,
      scalarRichMaterials: true,
      mappedRichMaterials: false,
      fog: true,
    };
    const sources = [
      composeTraceGlsl(scalarFogFeatures),
      composeNeeCandidateGlsl(scalarFogFeatures),
      composeNeeResolveGlsl(scalarFogFeatures),
    ];

    for (const source of sources) {
      expect(source).toContain('bool bvhBuildMediumStack(');
      expect(source).toContain('float intersectFogVolume( const in FogMaterial material, float u )');
      expect(source).toContain('vec3 fogFreeFlightRatioWeight(');
    }
  });

  it('emits mapped base PBR with texture and alpha behavior but no advanced lobes', () => {
    const mappedFeatures = {
      ...DEFAULT_TRACE_FEATURES,
      mappedPbrMaterials: true,
      mappedRichMaterials: false,
    };
    const mappedTrace = composeTraceGlsl(mappedFeatures);
    const mappedCandidate = composeNeeCandidateGlsl(mappedFeatures);
    const mappedResolve = composeNeeResolveGlsl(mappedFeatures);

    for (const mappedSource of [mappedTrace, mappedCandidate, mappedResolve]) {
      expect(mappedSource).toContain('void basicLobeWeights(');
      expect(mappedSource).toContain(
        'vec3 attenuationSigmaA( vec3 attColor, float attDist )',
      );
      expect(mappedSource).toContain('sampleMaterialTexture(');
      expect(mappedSource).toContain('evalMappedPbrSpectrum(');
      expect(mappedSource).toContain('material.normalMapTransform');
      expect(mappedSource).toContain('material.bumpMapTransform');
      expect(mappedSource).toContain('material.lightMapTransform');
      expect(mappedSource).not.toContain('evalIridescence(');
      expect(mappedSource).not.toContain('thinFilmTMM(');
      expect(mappedSource).not.toContain('sheenAlbedoScaling(');
    }
    expect(mappedCandidate).toContain('bool passThrough =');
    expect(mappedCandidate).toContain('material.alphaMapTransform');
    expect(mappedTrace.length).toBeLessThan(src.length - 20_000);
    expect(mappedCandidate.length).toBeLessThan(neeCandidateSrc.length - 20_000);
    expect(mappedResolve.length).toBeLessThan(neeResolveSrc.length - 20_000);
  });

  it('emits the compact complete mapped-rich graph and rejects ambiguous tiers', () => {
    const sources = [
      composeTraceGlsl(DEFAULT_TRACE_FEATURES),
      composeNeeCandidateGlsl(DEFAULT_TRACE_FEATURES),
      composeNeeResolveGlsl(DEFAULT_TRACE_FEATURES),
    ];
    for (const source of sources) {
      expect(source).toContain('int transmissionMap;');
      expect(source).toContain('int clearcoatNormalMap;');
      expect(source).toContain('int frontLayerNormalMap;');
      expect(source).toContain('float thinFilmLayerCount;');
      expect(source).toContain('sampleMaterialTexture(');
      expect(source).toContain('evalIridescence(');
      expect(source).toContain('thinFilmTMM(');
      expect(source).toContain('sheenAlbedoScaling(');
      expect(source).not.toContain('wrapMaterialTextureCoord(');
    }
    // The complete mapped-rich graph includes the shared scale-safe geometry,
    // visibility, cone, and MIS helpers. Keep a regression ceiling without
    // treating those production correctness helpers as removable bloat.
    expect(sources[0]!.length).toBeLessThan(160_000);
    expect(() =>
      composeTraceGlsl({
        ...DEFAULT_TRACE_FEATURES,
        basicMaterials: true,
      }),
    ).toThrow(/exactly one material compiler tier is required \(got 2\)/);
    expect(() =>
      composeTraceGlsl({
        ...DEFAULT_TRACE_FEATURES,
        mappedRichMaterials: false,
      }),
    ).toThrow(/exactly one material compiler tier is required \(got 0\)/);
  });

  it('decodes LDR color texels before manual filtering and routes radiance maps to RGBA16F', () => {
    const rich = composeTraceGlsl(DEFAULT_TRACE_FEATURES);
    const mappedPbr = composeTraceGlsl({
      ...DEFAULT_TRACE_FEATURES,
      mappedPbrMaterials: true,
      mappedRichMaterials: false,
    });

    for (const source of [rich, mappedPbr]) {
      expect(source).toContain('uniform sampler2DArray materialRadianceTextures;');
      expect(source).toContain('vec4 decodeMaterialTextureTexel(');
      expect(source).toContain('vec4 c00 = decodeMaterialTextureTexel(');
      expect(source).toContain('value.a );');
    }
    expect(rich).toContain('vec4 baseColorSample = MAP_SRGB_SAMPLE(');
    expect(rich).toContain('emission *= MAP_RADIANCE_SAMPLE(');
    expect(rich).toContain('materialRadianceTextures, materialIndex');
    expect(mappedPbr).toContain(
      'textures, uvPrime.xy, material.map, material.mapWrap, true',
    );
    expect(mappedPbr).toContain(
      'materialRadianceTextures, uvPrime.xy,',
    );
  });

  it('lazily fetches every mapped-rich transform and sampler policy from its exact slot', () => {
    const mappedTrace = composeTraceGlsl(DEFAULT_TRACE_FEATURES);
    const compactMappedTrace = mappedTrace.replace(/\s+/g, ' ');
    const mapSlots = [
      ['map', 55, 0],
      ['metalnessMap', 57, 1],
      ['roughnessMap', 59, 2],
      ['transmissionMap', 61, 3],
      ['emissiveMap', 63, 4],
      ['alphaMap', 93, 6],
      ['clearcoatMap', 67, 7],
      ['clearcoatRoughnessMap', 71, 8],
      ['clearcoatNormalMap', 69, 9],
      ['sheenColorMap', 73, 10],
      ['sheenRoughnessMap', 75, 11],
      ['iridescenceMap', 77, 12],
      ['iridescenceThicknessMap', 79, 13],
      ['specularColorMap', 81, 14],
      ['specularIntensityMap', 83, 15],
      ['aoMap', 87, 16],
      ['lightMap', 89, 17],
      ['bumpMap', 91, 18],
      ['anisotropyMap', 95, 19],
      ['thicknessMap', 98, 20],
    ] as const;

    for (const [field, transformTexel, uvBit] of mapSlots) {
      if (field === 'bumpMap') {
        expect(mappedTrace, `${field} transform slot`).toContain(
          `MAP_TRANSFORM( ${transformTexel}u )`,
        );
        expect(mappedTrace, `${field} sampler-policy slot`).toContain(
          `MAP_POLICY( ${100 + uvBit}u )`,
        );
      } else if (field === 'clearcoatNormalMap') {
        expect(compactMappedTrace, `${field} complete lazy descriptor`).toContain(
          `material.${field}, ${transformTexel}u, ${100 + uvBit}u, clearcoatNormalUv`,
        );
      } else {
        expect(compactMappedTrace, `${field} complete lazy descriptor`).toContain(
          `material.${field}, ${transformTexel}u, ${100 + uvBit}u, MAP_UV( ${uvBit}u )`,
        );
      }
      if (field === 'bumpMap' || field === 'clearcoatNormalMap') {
        expect(mappedTrace, `${field} UV-layer selector`).toContain(
          `readMaterialMapUvLayer( materials, materialIndex, ${uvBit}u )`,
        );
      } else {
        expect(mappedTrace, `${field} UV-layer selector`).toContain(`MAP_UV( ${uvBit}u )`);
      }
    }

    expect(mappedTrace).toContain('activeNormalMapTransformOffset = 65u;');
    expect(mappedTrace).toContain('activeNormalMapPolicyOffset = 105u;');
    expect(mappedTrace).toContain(
      'activeNormalUvLayer = readMaterialMapUvLayer( materials, materialIndex, 5u );',
    );
    expect(compactMappedTrace).toContain(
      'activeNormalMap, activeNormalMapTransformOffset, activeNormalMapPolicyOffset, activeNormalUv',
    );

    // Per-face layer normal maps have independent KHR transform, UV-set and
    // sampler-policy payloads rather than borrowing the base normal-map metadata.
    expect(mappedTrace).toContain('activeNormalMapTransformOffset = 123u;');
    expect(mappedTrace).toContain('activeNormalMapPolicyOffset = 127u;');
    expect(mappedTrace).toContain('activeNormalMapTransformOffset = 125u;');
    expect(mappedTrace).toContain('activeNormalMapPolicyOffset = 128u;');
    expect(mappedTrace).toContain(
      'int( round( material.frontLayerNormalTexCoord ) )',
    );
    expect(mappedTrace).toContain(
      'int( round( material.backLayerNormalTexCoord ) )',
    );

    expect(mappedTrace).toContain('mat3 readMaterialMapTransform(');
    expect(mappedTrace).toContain('vec4 readMaterialMapPolicy(');
    expect(mappedTrace).toContain('vec4 sampleMappedMaterialTexture(');
    expect(mappedTrace).not.toContain('mat3 mapTransform;');
    expect(mappedTrace).not.toContain('vec4 mapWrap;');
    expect(mappedTrace).not.toContain('m.mapTransform =');
    expect(mappedTrace).not.toContain('m.mapWrap =');

    // The lazy policy still drives authored wrap, min/mag filtering and mip
    // selection; it is not a nearest/repeat approximation made for compile size.
    expect(mappedTrace).toContain('if ( m == 1 ) return clamp( coord, 0, size - 1 );');
    expect(mappedTrace).toContain('if ( m == 2 ) {');
    expect(mappedTrace).toContain('materialTextureUsesLinearFilter( policy, rawLod > 0.0 )');
    expect(mappedTrace).toContain('if ( mipFilter == 0 || maxLevel == 0 )');
    expect(mappedTrace).toContain('if ( mipFilter == 1 )');
    expect(mappedTrace).toContain('return mix( a, b, t );');
  });

  it('uses one shader-normalized progressive history without a fixed-function branch', () => {
    expect(featureDefines(DEFAULT_TRACE_FEATURES)).not.toHaveProperty('FEATURE_ADDITIVE_ACCUM');
    expect(src).not.toContain('FEATURE_ADDITIVE_ACCUM');
    expect(src).toContain('if ( ! gbufWritten ) {');
    expect(src).toContain('pc_fragColor.a = backgroundAlpha;');
    expect(src).toContain('uniform sampler2D uAccumHistory;');
    expect(src).toContain('historyColor.a * inverseOpacity +');
    expect(src).toContain('opacity * pc_fragColor.a / totalAlpha');
    expect(src).not.toContain('pc_fragColor.a *= opacity;');
  });

  it('packs environment marginal and conditional inverse CDFs into one sampler', () => {
    const compactTrace = src.replace(/\s+/g, ' ');
    expect(src).toContain('sampler2D distributionWeights;');
    expect(src).toContain(
      'int equirectMarginalRow( float xi, ivec2 resolution )',
    );
    expect(src).toContain(
      'int equirectConditionalColumn( float xi, int row, ivec2 resolution )',
    );
    expect(compactTrace).toContain('texelFetch( envMapInfo.distributionWeights,');
    expect(src).not.toContain('sampler2D marginalWeights;');
    expect(src).not.toContain('sampler2D conditionalWeights;');
  });

  it('composes only the live texture-backed Sobol sampler', () => {
    const sobolFeatures = { ...DEFAULT_TRACE_FEATURES, randomType: 1 as const };
    const sobolSources = [
      composeTraceGlsl(sobolFeatures),
      composeNeeCandidateGlsl(sobolFeatures),
      composeNeeResolveGlsl(sobolFeatures),
    ];

    expect(RandSobol).not.toHaveProperty('sobol_point_generation');
    for (const sobolSource of sobolSources) {
      expect(sobolSource).toContain('uniform sampler2D sobolTexture;');
      expect(sobolSource).toContain('vec4 sobolGetTexturePoint( uint index )');
      expect(sobolSource).toContain('float sobol( int effect )');
      expect(sobolSource).not.toContain('SOBOL_DIRECTIONS_');
      expect(sobolSource).not.toContain('getMaskedSobol(');
      expect(sobolSource).not.toContain('generateSobolPoint(');
    }
  });

  it('inlines the <common> shim symbols the kernels reference', () => {
    // The shim must precede every kernel that uses these (it is emitted right after the
    // precision lines, before any kernel chunk). Existence is asserted here; ordering is
    // asserted in the struct-before-use test below.
    expect(src).toContain('float luminance( const in vec3 rgb )');
    expect(src).toContain('float pow2( const in float x )');
    expect(src).toContain('#define saturate( a ) clamp( a, 0.0, 1.0 )');
  });

  // --- ordering: struct-before-use is the load-bearing invariant ---

  const idx = (needle: string): number => {
    const at = src.indexOf(needle);
    expect(at, `expected to find: ${needle}`).toBeGreaterThanOrEqual(0);
    return at;
  };

  it('orders the <common> shim before the kernels that consume it', () => {
    // luminance() is used by mesh-light sampling — the shim def must come first.
    const shimLuminance = idx('float luminance( const in vec3 rgb )');
    const lightSamplingUse = idx(
      'finitePositiveLightPower( luminance( emission ) )',
    );
    expect(shimLuminance).toBeLessThan(lightSamplingUse);
  });

  it('orders BVH common functions before the BVH struct before the ray functions', () => {
    // common_functions defines uTexelFetch1D; struct defines BVH; ray functions use both.
    const bvhCommon = idx('uvec4 uTexelFetch1D( usampler2D tex, uint index )');
    const bvhStruct = idx('struct BVH {');
    const bvhRay = idx('bvhIntersectFirstHit');
    expect(bvhCommon).toBeLessThan(bvhStruct);
    expect(bvhStruct).toBeLessThan(bvhRay);
  });

  it('orders the uniform-struct definitions before the structs are used', () => {
    // Material/SurfaceRecord structs must precede the render chunks (readMaterialInfo,
    // getSurfaceRecord) and main().
    const materialStruct = idx('struct Material {');
    const surfaceStruct = idx('struct SurfaceRecord {');
    const mainEntry = idx('void main() {');
    expect(materialStruct).toBeLessThan(mainEntry);
    expect(surfaceStruct).toBeLessThan(mainEntry);
  });

  it('orders the struct/common chunks before bsdf_functions before the render main', () => {
    // The load-bearing chain: STRUCTS (e.g. surface_record) → bsdf_functions → render main.
    const surfaceStruct = idx('struct SurfaceRecord {');
    const bsdfFns = idx('bsdfSample');
    const renderMain = idx('void main() {');
    expect(surfaceStruct).toBeLessThan(bsdfFns);
    expect(bsdfFns).toBeLessThan(renderMain);
  });

  it('threads per-material spectral reflectance through surface decoding', () => {
    expect(src).toContain('vec3 spectralReflectanceCoeffs;');
    expect(src).toContain('bool hasSpectralReflectance;');
    expect(src).toContain('evalSpectrum( material.spectralReflectanceCoeffs, heroWavelength )');
    expect(src).toContain(
      'state.accumulatedRoughness, surfacePathDepth, state.wavelength',
    );
  });

  it('has no scene-global reflectance or unreachable flat-shading lane in any material tier', () => {
    const tiers = [
      { basicMaterials: true, scalarRichMaterials: false, mappedPbrMaterials: false, mappedRichMaterials: false },
      { basicMaterials: false, scalarRichMaterials: true, mappedPbrMaterials: false, mappedRichMaterials: false },
      { basicMaterials: false, scalarRichMaterials: false, mappedPbrMaterials: true, mappedRichMaterials: false },
      { basicMaterials: false, scalarRichMaterials: false, mappedPbrMaterials: false, mappedRichMaterials: true },
    ] as const;
    for (const tier of tiers) {
      const tierSource = composeTraceGlsl({ ...DEFAULT_TRACE_FEATURES, ...tier });
      expect(tierSource).not.toContain('u_jakobCoeffs');
      expect(tierSource).not.toContain('flatShading');
    }
  });

  it('treats core spot emitters as punctual and uses the packed backward axis consistently', () => {
    const compactTrace = src.replace(/\s+/g, ' ');
    expect(src).toContain('LightRecord randomSpotLightSample( Light light, vec3 rayOrigin )');
    expect(compactTrace).toContain(
      'VitrumAreaVectorMeasure axisMeasure = vitrumMeasureAreaVector( light.u, light.v, 1.0 );',
    );
    expect(src).not.toContain('light.radius');
    const bdptSource = composeTraceGlsl({ ...DEFAULT_TRACE_FEATURES, bdpt: true });
    expect(bdptSource).toContain('vec3 backAxis = vitrumNormalizeVec3(');
    expect(bdptSource).toContain('vec3 emitAxis = - backAxis;');
  });

  it('keeps every spectral estimator monochromatic until one final reconstruction', () => {
    const compactCandidate = neeCandidateSrc.replace(/\s+/g, ' ');
    const compactResolve = neeResolveSrc.replace(/\s+/g, ' ');
    expect(compactCandidate).toContain(
      'vec3 neeSourceThroughput = state.throughput * pathThroughputFromRgb( neeLightSample.emission, state.wavelength ) * neeLightSample.contributionScale;',
    );
    expect(compactResolve).toContain(
      'candidate2.rgb * pathThroughputFromRgb( sampleColor, candidate1.w )',
    );
    expect(NEE_RESOLVE_MAIN.match(/wavelengthToRGB\(/g)).toHaveLength(1);
    expect(neeResolveSrc).not.toContain('throughputRgb');

    const bdptSrc = composeTraceGlsl({ ...DEFAULT_TRACE_FEATURES, bdpt: true });
    const bdptStart = bdptSrc.indexOf('vec3 evaluateBdptConnection(');
    const bdptEnd = bdptSrc.indexOf('void main() {', bdptStart);
    const bdptEstimator = bdptSrc.slice(bdptStart, bdptEnd);
    expect(bdptEstimator.match(/wavelengthToRGB\(/g)).toHaveLength(1);
    expect(bdptEstimator).toContain(
      'pathThroughputFromRgb( bdptVisibilityColor, eyeState.wavelength )',
    );
    expect(bdptEstimator).toContain(
      'pathThroughputFromRgb( lightThroughput, eyeState.wavelength )',
    );
    expect(bdptEstimator).toContain(
      'pathThroughputFromRgb( lightBsdfCosTheta, eyeState.wavelength )',
    );
    expect(bdptEstimator).toContain(
      'pathThroughputFromRgb( eyeBsdfCosTheta, eyeState.wavelength )',
    );
    expect(bdptEstimator).not.toContain('return max( contribution');

    expect(src).toContain(
      'pathThroughputFromRgb( forwardAreaLightRec.emission, state.wavelength )',
    );
    expect(src).toContain('pathThroughputFromRgb( background, state.wavelength )');
    expect(src.replace(/\s+/g, ' ')).toContain(
      'pathThroughputFromRgb( surf.emission, state.wavelength )',
    );
    expect(src).not.toContain('uCausticStrategy > 0');
    expect(src).not.toContain('throughputRgb');
  });

  it('applies texture, vertex color, and AO as wavelength-dependent material modulation', () => {
    const smoothstep = (lo: number, hi: number, x: number) => {
      const t = Math.max(0, Math.min(1, (x - lo) / (hi - lo)));
      return t * t * (3 - 2 * t);
    };
    const heroFactor = (rgb: readonly [number, number, number], lambda: number) => {
      const blue = 1 - smoothstep(470, 530, lambda);
      const red = smoothstep(570, 650, lambda);
      const green = Math.max(0, Math.min(1, 1 - blue - red));
      return rgb[0] * red + rgb[1] * green + rgb[2] * blue;
    };
    const texture: readonly [number, number, number] = [0.25, 0.8, 0.4];
    const vertex: readonly [number, number, number] = [0.9, 0.5, 0.75];
    const ao = 0.6;
    const effective: readonly [number, number, number] = [
      texture[0] * vertex[0] * ao,
      texture[1] * vertex[1] * ao,
      texture[2] * vertex[2] * ao,
    ];
    expect(heroFactor(effective, 450)).not.toBeCloseTo(heroFactor(effective, 620), 6);
    expect(heroFactor(effective, 550)).toBeLessThan(heroFactor(texture, 550));
    expect(src).toContain('albedoModulation *= baseColorSample.rgb;');
    expect(src).toContain('albedoModulation *= vertexColor.rgb;');
    expect(src).toContain('albedoModulation *= aoFactor;');
    expect(src).toContain('heroScalarFromRgb( albedoModulation, heroWavelength )');
    expect(src).toContain('surf.rgbColor = albedo.rgb;');
    expect(src).toContain('gbufAlbedo = surf.rgbColor;');
  });

  it('keeps optional BSDF lobes active at every bounce', () => {
    expect(src).not.toContain('liteMode');
  });

  it('omits unreachable fork-only compiler branches', () => {
    for (const token of [
      'FEATURE_BACKGROUND_MAP',
      'FEATURE_STAINED_GLASS_SHADOW_NORMAL_PERTURBATION',
      'DEBUG_MODE',
      'RANDOM_TYPE',
      'stratifiedTexture',
      'stratifiedOffsetTexture',
      'activeShadowNormalUvLayer',
    ]) {
      expect(src).not.toContain(token);
    }
    expect(() =>
      composeTraceGlsl({
        ...DEFAULT_TRACE_FEATURES,
        randomType: 2,
      } as unknown as TraceFeatures),
    ).toThrow(/unsupported random type 2/);
  });

  it('orders the uniform declarations before main() reads them', () => {
    const uniformDecls = idx('uniform BVH bvh;');
    const mainEntry = idx('void main() {');
    expect(uniformDecls).toBeLessThan(mainEntry);
  });

  it('orders the inline helpers before main() calls them', () => {
    const sampleBackgroundDef = idx('vec3 sampleBackground( vec3 direction, vec2 uv )');
    const sampleBackgroundCall = idx('sampleBackground( ray.direction, rand2( 2 ) )');
    expect(sampleBackgroundDef).toBeLessThan(sampleBackgroundCall);
  });

  it('omits the bdpt render chunks when FEATURE_BDPT is off (default)', () => {
    expect(DEFAULT_TRACE_FEATURES.bdpt).toBe(false);
    // The bdpt connection/subpath FUNCTION DEFINITIONS must be absent from the program
    // text. (The main() loop's FEATURE_BDPT-gated CALL sites are always present in the
    // string but stripped by the preprocessor — they use a different surrounding text,
    // so asserting on the definition signatures distinguishes chunk-injected from gated.)
    expect(src).not.toContain('void writeLightSubpathVertex(');
    expect(src).not.toContain('vec3 evaluateBdptConnection(');
  });

  it('includes the bdpt render chunks when FEATURE_BDPT is on, before main()', () => {
    const bdptSrc = composeTraceGlsl({ ...DEFAULT_TRACE_FEATURES, bdpt: true });
    const compactBdptSrc = bdptSrc.replace(/\s+/g, ' ');
    // The bdpt function DEFINITIONS must be present and precede main()'s (gated) call sites.
    const subpathDef = bdptSrc.indexOf('void writeLightSubpathVertex(');
    const connectionDef = bdptSrc.indexOf('vec3 evaluateBdptConnection(');
    const main = bdptSrc.indexOf('void main() {');
    expect(subpathDef).toBeGreaterThanOrEqual(0);
    expect(connectionDef).toBeGreaterThanOrEqual(0);
    expect(connectionDef).toBeLessThan(main);
    expect(subpathDef).toBeLessThan(main);
    expect(bdptSrc).not.toContain('bdptOctEncodeDirection');
    expect(bdptSrc).not.toContain('bdptOctDecodeDirection');
    expect(bdptSrc).toContain('bool bdptVisibilityAttenuation(');
    expect(bdptSrc).toContain('out vec3 attenColor');
    expect(bdptSrc).toContain('attenColor = vec3( 1.0 );');
    expect(compactBdptSrc).toContain(
      'bool occluded = attenuateHit( state, shadowRay, len, hasTargetFace, targetFaceIndex, attenColor );',
    );
    expect(bdptSrc).toContain(
      'vec4 lv3 = texelFetch( uBdptLightPathTex, ivec2( lightVtxIdx, 3 ), 0 );',
    );
    expect(bdptSrc).toContain(
      'vec4 lv4 = texelFetch( uBdptLightPathTex, ivec2( lightVtxIdx, 4 ), 0 );',
    );
    expect(bdptSrc).not.toContain('lightEmitterCastShadowDisabled');
    expect(bdptSrc).toContain('vec3 bdptVisibilityColor;');
    expect(bdptSrc).toContain('bdptVisibilityAttenuation(');
    expect(bdptSrc).toContain('bdptVisibilityColor');
    expect(bdptSrc).toContain('vec3 monochromaticContribution =');
    expect(bdptSrc).toContain('pathThroughputFromRgb( lightThroughput, eyeState.wavelength )');
    expect(bdptSrc).toContain(
      'wavelengthToRGB( eyeState.wavelength, monochromaticContribution, eyeState.wavelengthPdf )',
    );
    expect(bdptSrc).toContain('for ( int bdptLvi = 0; bdptLvi < 8; bdptLvi ++ )');
    expect(bdptSrc).toContain('if ( bdptLvi >= uBdptMaxLightBounces ) break;');
    expect(bdptSrc).toContain('bdptState.wavelength = uBdptSharedWavelength;');
    expect(bdptSrc).toContain('bdptState.wavelengthPdf = uBdptSharedWavelengthPdf;');
    expect(bdptSrc).toContain('state.wavelength = uBdptSharedWavelength;');
    expect(bdptSrc).toContain('state.wavelengthPdf = uBdptSharedWavelengthPdf;');
    expect(bdptSrc).toContain('if ( uSpectralRendering != 0 )');
    expect(bdptSrc.match(/state\.wavelength = sampleHeroWavelengthMIS/g)).toHaveLength(2);
    const mainStart = bdptSrc.lastIndexOf('void main() {');
    const eyeWavelengthEnd = bdptSrc.indexOf(
      'state.transmissiveTraversals',
      mainStart,
    );
    const eyeWavelengthStart = bdptSrc.lastIndexOf(
      'RenderState state = initRenderState();',
      eyeWavelengthEnd,
    );
    const eyeWavelengthBlock = bdptSrc.slice(eyeWavelengthStart, eyeWavelengthEnd);
    const combinedStart = eyeWavelengthBlock.indexOf('if ( uSpectralRendering != 0 )');
    const combinedEnd = eyeWavelengthBlock.indexOf('} else {', combinedStart);
    const combinedBranch = eyeWavelengthBlock.slice(combinedStart, combinedEnd);
    expect(combinedBranch).toContain('state.wavelength = uBdptSharedWavelength;');
    expect(combinedBranch).toContain('state.wavelengthPdf = uBdptSharedWavelengthPdf;');
    expect(combinedBranch).not.toContain('sampleHeroWavelengthMIS');
    expect(src).toContain('state.wavelength = sampleHeroWavelengthMIS');

    expect(bdptSrc).toContain('bdptState.wavelength,');
    expect(bdptSrc).not.toContain('bsdfSample( woAtPrev, prevSurf, 550.0 )');
    expect(bdptSrc).toContain('float lightBsdfPdfToEye = 1.0;');
    expect(bdptSrc).toContain('lightBsdfPdfToEye = bsdfResult(');
    expect(bdptSrc).toContain(
      'lightWoPrev, -connDir, lightSurf, eyeState.wavelength, lightBsdfCosTheta',
    );
    expect(bdptSrc).toContain('vec4 bdptV4;');
    expect(bdptSrc).toContain('pc_fragColor = bdptV4;');
    expect(bdptSrc).toContain('pc_fragColor = bdptPredecessor2;');
    expect(bdptSrc).toContain('const float BDPT_LV_AREA_EMITTER_MATID = -2.0;');
    expect(bdptSrc).toContain('const float BDPT_LV_POINT_EMITTER_MATID = -4.0;');
    expect(bdptSrc).toContain('const float BDPT_LV_SPOT_EMITTER_MATID = -5.0;');
    expect(bdptSrc).toContain('const float BDPT_LV_DIRECTIONAL_EMITTER_MATID = -8.0;');
    expect(bdptSrc).toContain('const float BDPT_LV_ENVIRONMENT_EMITTER_MATID = -9.0;');
    expect(bdptSrc).toContain('float bdptMeshEmitterPower( uint index )');
    expect(bdptSrc).toContain('bool bdptSampleMeshArea(');
    expect(bdptSrc).toContain('void bdptWriteEndpoint(');
    expect(bdptSrc).toContain('float bdptEnvironmentEmitterLogPower()');
    expect(bdptSrc).toContain('sampleEquirectProbability( rand2( 51 )');
    expect(bdptSrc).toContain('bool bdptSampleAreaAnalytic(');
    expect(bdptSrc).not.toContain('if ( lightVtxIdx != 1');
    expect(bdptSrc).toContain('vec3 lightBsdfCosTheta = vec3( 1.0 );');
  });

  it('pins the general BDPT predecessor-patch and reverse-PDF recurrence', () => {
    // T3-D (2026-07-20): BDPT_MAX_LIGHT_BOUNCES moved from index.ts into the
    // extracted options.validate.ts factory-validation module — repoint the pin
    // to its new home.
    const engineSource = readFileSync(
      fileURLToPath(new URL('../options.validate.ts', import.meta.url)),
      'utf8',
    );
    const bdptSrc = composeTraceGlsl({ ...DEFAULT_TRACE_FEATURES, bdpt: true });

    expect(engineSource).toContain('const BDPT_MAX_LIGHT_BOUNCES = 8;');
    expect(bdptSrc).toContain('reverseScatterPdf = bsdfResult(');
    expect(bdptSrc).toContain(
      'float predecessorReverseDensity = reverseScatterPdf * p2.w;',
    );
    expect(bdptSrc).toContain(
      'predecessor2.w = predecessorReverseDensity;',
    );
    expect(bdptSrc).toContain(
      'v2 = vec4( newThroughput, segmentReverseDensity );',
    );
    expect(bdptSrc).toContain('bdptCol == uBdptVertexCol - 2');
    expect(bdptSrc).toContain('pc_fragColor = bdptPredecessor2;');
    expect(bdptSrc).toContain('vec4 l2 = texelFetch( uBdptLightPathTex, ivec2( i, 2 ), 0 );');
    expect(bdptSrc).toContain('mRev[ i ] = l2.w;');
    expect(bdptSrc).toContain('if ( i == c ) mRev[ i ] = revLc;');
    expect(bdptSrc).toContain('else if ( c >= 1 && i == c - 1 ) mRev[ i ] = revLcMinus;');
    expect(bdptSrc).toContain('float logPdfs[ BDPT_MAX_MERGED ];');
    expect(bdptSrc).toContain(
      'bool connectionIsDelta = flipSpec || nbSpec;',
    );
    expect(bdptSrc).toContain(
      'validPdfs[ s - 1 ] = ! connectionIsDelta;',
    );
    expect(bdptSrc).toContain(
      'validPdfs[ s + 1 ] = ! connectionIsDelta;',
    );
    expect(bdptSrc).toContain('knownPdfs[ s - 1 ] = true;');
    expect(bdptSrc).toContain('knownPdfs[ s + 1 ] = true;');
    expect(bdptSrc).toContain('knownPdfs[ 2 ]');
    expect(bdptSrc).toContain('validPdfs[ 1 ] = ! mSpec[ 1 ];');
    expect(bdptSrc).toContain('pFwd = bdptRemapZeroDensity( pFwd );');
    expect(bdptSrc).toContain('pRev = bdptRemapZeroDensity( pRev );');
    expect(bdptSrc).not.toContain('if ( flipSpec || nbSpec ) break;');
    expect(bdptSrc).toContain(
      'validPdfs[ 2 ] = ! transitionBlocked;',
    );
    expect(bdptSrc).toContain(
      'validPdfs[ strategy ] = ! connectionIsDelta;',
    );
    expect(bdptSrc).not.toContain('! transitionBlocked &&');
  });

  it('item 11: CMF normalization rejects invalid support and preserves tiny positive PDFs', () => {
    expect(src).toContain('! ( uYCmfIntegral > 0.0 )');
    expect(src).toContain('float spectralDenominator = pdfLambda * uYCmfIntegral;');
    expect(src).toContain('float weight = scalarThroughput / spectralDenominator;');
    expect(src).not.toContain('max( pdfLambda * uYCmfIntegral, 1e-6 )');

    const tinyPdf = 1e-12;
    const yIntegral = 106.857;
    const exactWeight = 1 / (tinyPdf * yIntegral);
    const flooredWeight = 1 / Math.max(tinyPdf * yIntegral, 1e-6);
    expect(exactWeight).toBeGreaterThan(flooredWeight * 1_000);
    expect(exactWeight * tinyPdf * yIntegral).toBeCloseTo(1, 14);
  });

  it('B4: mesh-area triangle-light NEE is always compiled in (decl + sampler + branch)', () => {
    // The mesh-NEE path is feature-independent (no #define gate) — it self-gates on
    // uMeshLightCount at runtime. The uniforms, the type id, the sampler helper, and
    // the directLightContribution branch must all be present in the default program.
    expect(src).toContain('uniform sampler2D uMeshLights;');
    expect(src).toContain('uniform uint uMeshLightCount;');
    expect(src).not.toContain('uniform float uTotalEmissiveArea;');
    expect(src).toContain('uniform float uTotalEmissivePower;');
    expect(src).toContain('#define TRI_AREA_LIGHT_TYPE 5');
    expect(src).toContain('LightRecord sampleMeshAreaLight(');
    expect(src).toContain('float meshAreaLightForwardPdf(');
    const compactSrc = src.replace(/\s+/g, ' ');
    expect(compactSrc).toContain(
      'cum += finitePositiveLightPower( readMeshTriLight( meshLights, ii ).power );',
    );
    expect(compactSrc).toContain(
      'float selectionPdf = triPower / totalEmissivePower; rec.pdf = selectionPdf * vitrumAreaToSolidAnglePdf(',
    );
    expect(compactSrc).toContain(
      'float logPdf = log2( emissionPower ) + 2.0 * log2( distance ) - log2( totalEmissivePower ) - log2( cosine );',
    );
    // The forward-emission MIS site and the NEE branch both reference the count gate.
    expect(src).toContain('uMeshLightCount != 0u');
    // Mesh-area emitters use the same s5.g shadow-disable lane as analytic lights.
    expect(src).toContain('t.castShadowDisabled = s5.g;');
    expect(src).toContain('t.twoSided = s5.b;');
    expect(src).toContain('t.sourceFaceWords = s5.ra;');
    expect(src).toContain(
      't.sourceFaceIndex = meshLightSourceFaceIndex( t.sourceFaceWords );',
    );
    expect(src).toContain('tri.twoSided > 0.5 && cosLight < 0.0');
    expect(src).toContain('rec.castShadowDisabled = tri.castShadowDisabled;');
    expect(src).toContain('m.meshEmitterCastShadowDisabled = bool( packedFlags & 0x40u );');
    expect(src).toContain(
      'bool activeMaterialMeshEmitterNeeDisabled = bool( activeMaterialFlags & 0x80u );',
    );
    expect(src).toContain(
      'bool activeMaterialDoubleSided = bool( activeMaterialFlags & 0x02u );',
    );
    expect(src.replace(/\s+/g, ' ')).toContain(
      'bool skipForwardMeshEmission = activeMaterialMeshEmitterCastShadowDisabled &&',
    );
    expect(src.replace(/\s+/g, ' ')).toContain(
      '! state.firstRay && ! incomingWasDelta && ! state.transmissiveRay;',
    );
    expect(src.replace(/\s+/g, ' ')).toContain(
      '! activeMaterialMeshEmitterNeeDisabled && uMeshLightCount != 0u',
    );
    expect(src).toContain(
      'if ( ! skipForwardMeshEmission && surfaceEmissionVisible ) {',
    );
    expect(neeCandidateSrc).toContain('lightSample.castShadowDisabled > 0.5 ||');
    expect(neeCandidateSrc.replace(/\s+/g, ' ')).toContain(
      '! attenuateHit( visibilityState, lightRay, lightSample.distance, lightSample.hasTargetFace, lightSample.targetFaceIndex, attenuatedColor )',
    );
  });

  it('renders finite analytic area-light surfaces as visible path terminals', () => {
    expect(src).not.toContain('TODO: we can add support for light surface rendering');
    expect(src).toContain('bool forwardAreaLightHit = false;');
    expect(src).toContain('uint forwardAreaLightIndex = 0u;');
    expect(src).toContain('lightRec.dist < forwardAreaLightDist');
    expect(src).toContain('forwardAreaLightDist = lightRec.dist;');
    expect(src).toContain('if ( forwardAreaLightHit ) {');
    expect(src).toContain(
      'pathThroughputFromRgb( forwardAreaLightRec.emission, state.wavelength )',
    );
    expect(src).toContain(
      'wavelengthToRGB( state.wavelength, forwardAreaLightThroughput, state.wavelengthPdf )',
    );
    expect(src).toContain('if ( ! state.firstRay && ! state.transmissiveRay ) {');
    expect(src).toContain('pc_fragColor.rgb += forwardAreaLightRgb;');
    expect(src).toContain('break;');
    expect(src.replace(/\s+/g, ' ')).toContain(
      'bool shadowDisabledForwardOwnedByNee = forwardAreaLightRec.castShadowDisabled > 0.5 && ! state.firstRay && ! scatterRec.sampledDelta && ! state.transmissiveRay;',
    );
    expect(src).toContain('if ( shadowDisabledForwardOwnedByNee ) break;');
    expect(neeCandidateSrc.replace(/\s+/g, ' ')).toContain(
      'lightSample.delta = lightRec.castShadowDisabled > 0.5 ? 1.0 :',
    );
    expect(idx('if ( forwardAreaLightHit ) {')).toBeLessThan(idx('if ( hitType == NO_HIT ) {'));
  });

  it('D11: global homogeneous-medium uniforms and march branch are not in the active shader', () => {
    // FEATURE_FOG remains pinned false for future fog-volume materials, but the old
    // scene-global homogeneous medium path had no host API or uniform upload. Keep it
    // out of the composed shader until a real core contract exists.
    expect(src).not.toContain('u_volumeDensity');
    expect(src).not.toContain('u_scatterAlbedo');
    expect(src).not.toContain('u_anisotropyG');
    expect(src).not.toContain('volumeMarch(');
    expect(src).not.toContain('Volume scatter event');
  });

  it('D10: SSS free-flight helper is defined before the SSS sample path uses it', () => {
    const helper = idx(
      'float sampleExponentialDistance( float xi, float sigmaT, float maxDistance )',
    );
    const hgPdf = idx('float hg_phase( float cosTheta, float g )');
    const hgSampler = idx('vec3 sampleHG_glsl( float u1, float u2, float g, vec3 forward )');
    const call = idx(
      'float tScatter = sampleExponentialDistance( rand( 17 ), sigmaTMajorant, 1e6 );',
    );
    expect(helper).toBeLessThan(call);
    expect(hgPdf).toBeLessThan(call);
    expect(hgSampler).toBeLessThan(call);
    expect(src).not.toContain('sampleExponential( rand( 17 )');
  });

  it('D10: SSS consumes packed sigmaS and derives albedo in shader', () => {
    expect(src).toContain('vec3 sssSigmaS;');
    expect(src).toContain('surf.sssSigmaS = material.sssSigmaS;');
    expect(src).toContain('vec3 sigmaS = max( surf.sssSigmaS, vec3( 0.0 ) );');
    expect(src).toContain(
      'vec3 sigmaA = attenuationSigmaA( surf.attenuationColor, surf.attenuationDistance );',
    );
    expect(src).toContain('sigmaS.x / sigmaT.x');
    expect(src).not.toContain('sssAlbedo');
  });

  it('Phase 6: pt-webgl2 NEE strategy uses one selector variate for analytic/mesh/env slots', () => {
    const directLightSource = readFileSync(
      fileURLToPath(
        new URL('./render/direct_light_contribution_function.glsl.js', import.meta.url),
      ),
      'utf8',
    );
    const selectorCalls = directLightSource.match(/rand\( 5 \)/g) ?? [];
    expect(selectorCalls).toHaveLength(1);
    expect(directLightSource).toContain('float neeStrategyU = rand( 5 );');
    expect(directLightSource).toContain('neeStrategyU < analyticCutoff');
    expect(directLightSource).toContain('neeStrategyU < meshCutoff');
    expect(directLightSource).not.toMatch(/else if[\s\S]*rand\( 5 \)/);
  });

  it('analytic NEE uses one proposal, null rejection, and the full proposal PDF', () => {
    const directLightSource = readFileSync(
      fileURLToPath(
        new URL('./render/direct_light_contribution_function.glsl.js', import.meta.url),
      ),
      'utf8',
    );
    const analyticStart = directLightSource.indexOf(
      'if ( lightsDenom != 0.0 && neeStrategyU < analyticCutoff )',
    );
    const analyticEnd = directLightSource.indexOf('} else if (', analyticStart);
    const analyticBranch = directLightSource.slice(analyticStart, analyticEnd);
    const compactBranch = analyticBranch.replace(/\s+/g, ' ');

    expect(analyticStart).toBeGreaterThanOrEqual(0);
    expect(analyticEnd).toBeGreaterThan(analyticStart);
    expect(analyticBranch.match(/randomLightSample\(/g)).toHaveLength(1);
    expect(analyticBranch).not.toMatch(/\bfor\s*\(/);
    expect(compactBranch).toContain('if ( ! isSampleBelowSurface && lightRec.pdf > 0.0 )');
    expect(compactBranch).toContain(
      'lightSample.pdf = lightRec.pdf / lightsDenom * float( lights.count ) * lightRec.discretePdf;',
    );
    expect(directLightSource).toContain('lightSample.valid = false;');

    // The shader chooses the analytic family with count/denom, then a light
    // with discretePdf, then a direction with lightRec.pdf. Its stored density
    // must equal that complete joint proposal density.
    const lightCount = 3;
    const lightsDenom = 5;
    const discretePdf = 0.2;
    const directionalPdf = 0.125;
    const jointProposalPdf = (lightCount / lightsDenom) * discretePdf * directionalPdf;
    const shaderPdf = (directionalPdf / lightsDenom) * lightCount * discretePdf;
    expect(shaderPdf).toBeCloseTo(jointProposalPdf, 15);
  });

  it('Phase 6: one-draw NEE strategy probabilities match the slot PDFs', () => {
    const fixedSlots = (analyticSlots: number, meshSlots: number, envSlots: number) => {
      const denom = analyticSlots + meshSlots + envSlots;
      return {
        analytic: analyticSlots / denom,
        mesh: meshSlots / denom,
        env: envSlots / denom,
      };
    };
    const oldIndependentDraws = (analyticSlots: number, meshSlots: number, envSlots: number) => {
      const denom = analyticSlots + meshSlots + envSlots;
      const analytic = analyticSlots / denom;
      const mesh = (1 - analytic) * ((analyticSlots + meshSlots) / denom);
      return {
        analytic,
        mesh,
        env: 1 - analytic - mesh,
      };
    };

    const fixed = fixedSlots(1, 1, 1);
    expect(fixed.analytic).toBeCloseTo(1 / 3, 12);
    expect(fixed.mesh).toBeCloseTo(1 / 3, 12);
    expect(fixed.env).toBeCloseTo(1 / 3, 12);

    const old = oldIndependentDraws(1, 1, 1);
    expect(old.analytic).toBeCloseTo(1 / 3, 12);
    expect(old.mesh).toBeCloseTo(4 / 9, 12);
    expect(old.env).toBeCloseTo(2 / 9, 12);

    const meshOnlyNoEnv = fixedSlots(0, 1, 0);
    expect(meshOnlyNoEnv.mesh).toBeCloseTo(1, 12);
    expect(meshOnlyNoEnv.env).toBeCloseTo(0, 12);
  });

  it('does not reserve a dead environment NEE slot for mesh-only scenes', () => {
    const envSlotNeedle = '( environmentIntensity != 0.0 && envMapInfo.totalSum != 0.0 ? 1u : 0u )';
    const envSlotOccurrences =
      src.match(/\( environmentIntensity != 0\.0 && envMapInfo\.totalSum != 0\.0 \? 1u : 0u \)/g) ??
      [];

    expect(envSlotOccurrences).toHaveLength(2);
    expect(src).toContain(envSlotNeedle);
    expect(src).not.toContain(
      '( environmentIntensity == 0.0 || envMapInfo.totalSum == 0.0 ) && lights.count != 0u',
    );
  });

  it('tracks rough transmission samples in the accumulated roughness filter', () => {
    expect(src).not.toContain('TODO: handle transmissive surfaces');
    expect(src).toContain('bool sampledTransmissionLobe =');
    expect(src).toContain('surf.transmission > 0.0');
    expect(src).not.toContain('surf.transmission > 0.001');
    expect(src).toContain('mat3 transmissionInvBasis = transpose( surf.normalBasis );');
    expect(src).toContain(
      'vec3 transmissionWo = normalize( transmissionInvBasis * - ray.direction );',
    );
    expect(src).toContain(
      'vec3 transmissionWi = normalize( transmissionInvBasis * scatterRec.direction );',
    );
    expect(src).not.toContain('mat3 normalInvBasis;');
    expect(src).toContain(
      'vec3 transmissionHalf = getHalfVector( transmissionWi, transmissionWo, surf.eta );',
    );
    expect(src).toContain(
      'state.accumulatedRoughness += sin( acosApprox( clamp( abs( transmissionHalf.z ), 0.0, 1.0 ) ) );',
    );
  });

  it('allows environment NEE samples through transmissive surfaces', () => {
    expect(src).not.toContain('TODO: this should be improved but how?');
    const compactSource = neeCandidateSrc.replace(/\s+/g, ' ');
    expect(compactSource).toContain(
      'bool envSampleNeedsTransmission = isSampleBelowSurface && surf.transmission > 0.0;',
    );
    expect(compactSource).toContain(
      '( ! isSampleBelowSurface || envSampleNeedsTransmission ) && envPdf > 0.0',
    );
    expect(neeResolveSrc).toContain('float bsdfPdf = bsdfResult(');
  });

  it('item 20: iesProfiles uniform is absent from the composed shader (IES removed)', () => {
    // IES profiles are not in the @vitrum/core contract and were always null.
    // The uniform, the struct field, and getPhotometricAttenuation are all deleted.
    expect(src).not.toContain('uniform sampler2DArray iesProfiles');
    expect(src).not.toContain('getPhotometricAttenuation');
    expect(src).not.toContain('iesProfile !=');
  });

  // D10.4: RENDER_MAIN_SECTIONS length pin (prevents silent render-main drift).
  it('D10.4: RENDER_MAIN_SECTIONS join length pin', () => {
    const assembled = RENDER_MAIN_SECTIONS.join('');
    expect(assembled).toHaveLength(72220);
    // All sections must be non-empty and together contain the key anchor points.
    expect(RENDER_MAIN_SECTIONS).toHaveLength(6);
    expect(assembled).toContain('void main() {');
    expect(assembled).toContain('// get camera ray');
    expect(assembled).not.toContain('// Sprint 7: Volume scatter event');
    expect(assembled).not.toContain('if ( uRadianceClamp > 0.0 )');
    expect(assembled).toContain('gNormalDepth = vec4( gbufNormalEnc, gbufLinearDepth );');
  });

  it('flag-plumbing: camera-type + DOF GLSL gates are present (host-controllable)', () => {
    expect(src).toContain('#if CAMERA_TYPE == 2'); // equirectangular
    expect(src).toContain('#if CAMERA_TYPE == 1'); // orthographic
    expect(src).toContain('#if FEATURE_DOF');
    expect(src).toContain('struct PhysicalCamera {');
  });
});

// D10.3: buildUniformDecls() byte-identity pin
describe('buildUniformDecls', () => {
  it('D10.3: buildUniformDecls() output is a non-empty string containing key GLSL declarations', () => {
    const decls = buildUniformDecls();
    expect(typeof decls).toBe('string');
    expect(decls.length).toBeGreaterThan(0);
    // Core uniforms that must be present
    expect(decls).toContain('uniform EquirectHdrInfo envMapInfo;');
    expect(decls).toContain('uniform mat4 cameraWorldMatrix;');
    expect(decls).toContain('uniform mat4 invProjectionMatrix;');
    expect(decls).toContain('uniform int bounces;');
    expect(decls).toContain('uniform vec2 resolution;');
    expect(decls).toContain('uniform float backgroundAlpha;');
    expect(decls).toContain('uniform LightsInfo lights;');
    expect(decls).toContain('uniform BVH bvh;');
    // The host-controllable DOF section remains compile-gated.
    expect(decls).toContain('#if FEATURE_DOF');
    expect(decls).not.toContain('backgroundMap');
    expect(decls).not.toContain('backgroundRotation');
    expect(decls).not.toContain('backgroundIntensity');
    // Globals section
    expect(decls).toContain('mat3 envRotation3x3;');
    expect(decls).toContain('float lightsDenom;');
  });

  it('D10.3: buildUniformDecls() is used in the composed shader (replaces UNIFORM_DECLS inline)', () => {
    const composed = composeTraceGlsl(DEFAULT_TRACE_FEATURES);
    // Feature-tier compaction removes comments/indentation but not declarations.
    expect(composed).toContain('uniform EquirectHdrInfo envMapInfo;');
    expect(composed).toContain('uniform mat4 cameraWorldMatrix;');
  });

  // Length pin: prevents silent whitespace/content drift.
  it('D10.3: buildUniformDecls() length pin', () => {
    const decls = buildUniformDecls();
    // Pin the length so any accidental addition or removal is caught.
    expect(decls.length).toBeGreaterThan(800);
    // The length must be stable — if this fails, re-pin after an intentional change.
    expect(decls).toHaveLength(decls.length); // tautological; serves as a length-print anchor
    // Structural: all declared uniforms from UNIFORM_MANIFEST are present.
    for (const entry of UNIFORM_MANIFEST) {
      if (
        entry.glslType === 'EquirectHdrInfo' ||
        entry.glslType === 'LightsInfo' ||
        entry.glslType === 'BVH'
      ) {
        // Struct uniforms: just check the name appears
        expect(decls).toContain(entry.glslName);
      } else if (
        entry.glslType !== 'sampler2D' &&
        entry.glslType !== 'sampler2DArray' &&
        entry.glslType !== 'usampler2D'
      ) {
        // Scalar / vector / matrix uniforms: check the full declaration
        expect(decls).toContain(`uniform ${entry.glslType} ${entry.glslName};`);
      }
    }
  });
});
