// D3 stride-parity guard (2026-06-10). The packer↔shader material stride is the
// repo's recurring upload-gap bug class (H1 / H41 / the D3 85-vs-93 working-tree
// break, which the entire suite stayed GREEN through — this file is the fix for
// that blind spot). The stride is single-sourced in materialStride.js; this test
// asserts every composed-shader fetch site actually carries that value and that
// no stale hardcoded stride survives anywhere in the composed GLSL.
import { describe, expect, it } from 'vitest';
import type { MaterialSpec } from '@vitrum/core';
import { MATERIAL_PIXELS, packMaterialsTexture } from './materialsTexture.js';
import { LIGHT_PIXELS } from './lightsTexture.js';
import { composeTraceGlsl } from '../glsl/composeTraceGlsl.js';
import { DEFAULT_TRACE_FEATURES } from '../featureTypes.js';

function composedShader(): string {
  return composeTraceGlsl({ ...DEFAULT_TRACE_FEATURES });
}

describe('material stride parity (packer ↔ composed GLSL)', () => {
  it('readMaterialInfo fetches at the packer stride', () => {
    expect(composedShader()).toContain(`uint i = index * ${MATERIAL_PIXELS}u;`);
  });

  it('every GLSL MATERIAL_PIXELS const equals the packer stride', () => {
    const matches = [...composedShader().matchAll(/MATERIAL_PIXELS\s*=\s*(\d+)u/g)];
    // thin_film_tmm + inside_fog_volume + util_functions all declare it.
    expect(matches.length).toBeGreaterThanOrEqual(3);
    for (const m of matches) {
      expect(Number(m[1])).toBe(MATERIAL_PIXELS);
    }
  });

  it('no stale hardcoded material stride survives in the composed shader', () => {
    // Every bare `index * <N>u` / `materialIndex * <N>u` multiply (the names the
    // material- and lights-texture decoders use) must carry one of the two
    // packer-exported strides — a mismatched literal is exactly the D3 break.
    // The lookbehind excludes other *Index identifiers (triIndex, faceIndex…).
    const knownStrides = new Set([MATERIAL_PIXELS, LIGHT_PIXELS]);
    const staleFetch = [...composedShader().matchAll(/(?<![A-Za-z])(?:material)?[iI]ndex\s*\*\s*(\d+)u/g)]
      .map((m) => Number(m[1]))
      .filter((n) => !knownStrides.has(n));
    expect(staleFetch).toEqual([]);
  });

  it('packer writes the D3 texels inside the stride (smoke)', () => {
    const mats: MaterialSpec[] = [
      { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0, envMapIntensity: 2.5 },
      { baseColor: [1, 0, 0], roughness: 0.1, metallic: 1, aoMapIntensity: 0.25 },
    ];
    const packed = packMaterialsTexture(mats);
    // material 0: texel 85 .a = envMapIntensity; texel 86 .r = aoMapIntensity.
    expect(packed.data[85 * 4 + 3]).toBe(2.5);
    expect(packed.data[86 * 4 + 0]).toBe(1.0); // default aoMapIntensity
    // material 1 decodes at base = MATERIAL_PIXELS — the exact D3 break shape.
    const base1 = MATERIAL_PIXELS * 4;
    expect(packed.data[base1 + 85 * 4 + 3]).toBe(1.0); // default envMapIntensity
    expect(packed.data[base1 + 86 * 4 + 0]).toBe(0.25);
  });

  it('packer keeps the former UV1 mirror lane reserved at texel 86.a', () => {
    const noUv1Mat: MaterialSpec[] = [
      { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 },
    ];
    const noUv1 = packMaterialsTexture(noUv1Mat);
    expect(noUv1.data[86 * 4 + 3]).toBe(0);

    // Per-map UV selection lives only in the scalable selector table.
    const handle = {};
    const layerOf = new Map<unknown, number>([[handle, 0]]);
    const uv1Mat: MaterialSpec[] = [
      { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0, baseColorMap: { handle, texCoord: 1 } },
    ];
    const withUv1 = packMaterialsTexture(uv1Mat, layerOf);
    expect(withUv1.data[86 * 4 + 3]).toBe(0);
  });

  it('GLSL omits the superseded uvTexCoordMask field', () => {
    const shader = composedShader();
    expect(shader).toContain('s = texelFetch1D( tex, i + 86u );');
    expect(shader).not.toContain('uvTexCoordMask');
  });

  it('GLSL decodes scalar anisotropy and routes GGX through anisotropic helpers', () => {
    const shader = composedShader();
    expect(shader).toContain('s = texelFetch1D( tex, i + 6u );');
    expect(shader).toContain('m.anisotropyMap = int( round( s.b ) );');
    expect(shader).toContain('s = texelFetch1D( tex, i + 11u );');
    expect(shader).toContain('m.thinFilm = bool( s.b ); m.anisotropy = clamp( s.a, 0.0, 1.0 );');
    expect(shader).toContain('s = texelFetch1D( tex, i + 17u );');
    expect(shader).toContain('m.thinFilmAngleDependent = s.g > 0.5; m.anisotropyRotation = s.b;');
    expect(shader).toContain('s = texelFetch1D( tex, i + 97u );');
    expect(shader).toContain('m.thickness = max( s.r, 0.0 ); m.thicknessMap = int( round( s.g ) );');
    expect(shader).toContain('mat3 readMaterialMapTransform(');
    expect(shader).toContain('vec4 readMaterialMapPolicy(');
    expect(shader).not.toContain('m.anisotropyMapTransform =');
    expect(shader).not.toContain('m.thicknessMapTransform =');
    expect(shader).toContain('m.frontLayerNormalMap = int( round( layerNormal.r ) );');
    expect(shader).toContain('m.backLayerNormalMap = int( round( layerNormal.b ) );');
    expect(shader).toContain('activeNormalMapTransformOffset = 123u;');
    expect(shader).toContain('activeNormalMapTransformOffset = 125u;');
    expect(shader).toContain('activeNormalMap = material.frontLayerNormalMap;');
    expect(shader).toContain('activeNormalMap = material.backLayerNormalMap;');
    expect(shader).toContain('int( round( material.frontLayerNormalTexCoord ) )');
    expect(shader).toContain('readMaterialMapUvLayer( materials, materialIndex, mapIndex )');
    expect(shader.replace(/\s+/g, ' ')).toContain(
      'activeNormalMap, activeNormalMapTransformOffset, activeNormalMapPolicyOffset, activeNormalUv',
    );
    expect(shader).not.toContain('m.anisotropyMapWrap =');
    expect(shader).not.toContain('m.thicknessMapWrap =');
    expect(shader.replace(/\s+/g, ' ')).toContain(
      'material.anisotropyMap, 95u, 119u, MAP_UV( 19u )',
    );
    expect(shader).toContain('anisotropy *= anisotropySample.b;');
    expect(shader).toContain('anisotropyRotation = rotation;');
    expect(shader.replace(/\s+/g, ' ')).toContain(
      'int uvLayer = readMaterialMapUvLayer( materials, materialIndex, 20u );',
    );
    expect(shader.replace(/\s+/g, ' ')).toContain(
      'sampleMappedMaterialTexture( materials, textures, materialIndex, material.thicknessMap, 98u, 120u, uv, thicknessSample ) ) attenuationThickness *= thicknessSample.g;',
    );
    expect(shader.replace(/\s+/g, ' ')).toContain(
      'if ( stack.hasAttenuationThicknesses[ top ] ) { attenuationDist = min( attenuationDist, max( stack.attenuationThicknesses[ top ], 0.0 ) );',
    );
    expect(shader).toContain('surf.anisotropy = clamp( anisotropy, 0.0, 1.0 );');
    expect(shader).toContain('mat3 getBasisFromNormalAndTangent( vec3 normal, vec4 tangentSample )');
    expect(shader).toContain('vec3 tangent = tangentSample.xyz - n * dot( tangentSample.xyz, n );');
    expect(shader).toContain('vec4 bsdfTangentSample = textureSampleBarycoord(');
    expect(shader.replace(/\s+/g, ' ')).toContain(
      'int bsdfBasisUvLayer = anisotropyMapApplied ? readMaterialMapUvLayer( materials, materialIndex, 19u ) : ATTR_UV;',
    );
    expect(shader.replace(/\s+/g, ' ')).toContain(
      'surf.normalBasis = getBasisFromSelectedUv( bvh.position, attributesArray, bsdfBasisUvLayer, surfaceHit.faceIndices.xyz, surf.normal, bsdfTangentSample );',
    );
    expect(shader.replace(/\s+/g, ' ')).toContain(
      'int clearcoatBasisUvLayer = material.clearcoatNormalMap != - 1 ? readMaterialMapUvLayer( materials, materialIndex, 9u ) : ATTR_UV;',
    );
    expect(shader.replace(/\s+/g, ' ')).toContain(
      'surf.clearcoatBasis = getBasisFromSelectedUv( bvh.position, attributesArray, clearcoatBasisUvLayer, surfaceHit.faceIndices.xyz, surf.clearcoatNormal, bsdfTangentSample );',
    );
    expect(shader).toContain('vec2 anisotropicRoughnessAxes( const in SurfaceRecord surf )');
    expect(shader).toContain('ggxDirectionForSurface( wo, surf, rand2( 12 ) )');
    expect(shader).toContain('ggxDistributionForSurface( wh, surf )');
    expect(shader).toContain('ggxPdfForSurface( wo, wh, surf )');
  });
});
