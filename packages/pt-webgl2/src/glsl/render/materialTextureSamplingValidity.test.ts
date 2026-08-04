import { describe, expect, it } from 'vitest';
import { ATTENUATE_HIT_MAPPED_PBR_GLSL } from './attenuate_hit_mapped_pbr.glsl.js';
import { GET_SURFACE_RECORD_MAPPED_PBR_GLSL } from './get_surface_record_mapped_pbr.glsl.js';
import * as AttenuateRichSource from './attenuate_hit_function.glsl.js';
import * as SurfaceRichSource from './get_surface_record_function.glsl.js';
import * as BsdfRichSource from '../shader/bsdf/bsdf_functions.glsl.js';
import { MATERIAL_MAPPED_PBR_GLSL } from '../shader/structs/material_mapped_pbr.glsl.js';
import { MATERIAL_MAPPED_RICH_GLSL } from '../shader/structs/material_mapped_rich.glsl.js';
import { MATERIAL_TEXTURE_SAMPLING_GLSL } from '../shader/structs/material_texture_sampling.glsl.js';

const attenuateHitRich = (
  AttenuateRichSource as unknown as Record<string, string>
).attenuate_hit_function!;
const getSurfaceRecordRich = (
  SurfaceRichSource as unknown as Record<string, string>
).get_surface_record_function!;
const bsdfRich = (
  BsdfRichSource as unknown as Record<string, string>
).bsdf_functions!;

describe('mapped material texture explicit validity', () => {
  it('validates descriptors, every lower tap, and both trilinear levels', () => {
    expect(MATERIAL_TEXTURE_SAMPLING_GLSL).toContain('bool decodeMaterialTexturePolicy(');
    expect(MATERIAL_TEXTURE_SAMPLING_GLSL).toContain('bool sampleMaterialTexture(');
    expect(MATERIAL_TEXTURE_SAMPLING_GLSL).toContain(
      'if ( ! valid00 || ! valid10 || ! valid01 || ! valid11 ) return false;',
    );
    expect(MATERIAL_TEXTURE_SAMPLING_GLSL).toContain(
      'if ( ! validA || ! validB ) return false;',
    );
    expect(MATERIAL_TEXTURE_SAMPLING_GLSL).toContain(
      'return materialTextureFiniteVec4( value );',
    );
    expect(MATERIAL_TEXTURE_SAMPLING_GLSL).not.toContain('return vec4( 1.0 )');

    for (const decoder of [MATERIAL_MAPPED_PBR_GLSL, MATERIAL_MAPPED_RICH_GLSL]) {
      expect(decoder).toContain('bool sampleMaterialTexture(');
      expect(decoder).not.toContain('return vec4( 1.0 )');
    }
  });

  it('lets each PBR role choose its physical fallback', () => {
    expect(GET_SURFACE_RECORD_MAPPED_PBR_GLSL).toContain('vec4 albedo = vec4( material.color, material.opacity );');
    expect(GET_SURFACE_RECORD_MAPPED_PBR_GLSL).toContain('vec4 lightSample;');
    expect(GET_SURFACE_RECORD_MAPPED_PBR_GLSL).toContain('material.lightMapWrap, lightSample');
    expect(GET_SURFACE_RECORD_MAPPED_PBR_GLSL).toContain(
      'decodeMaterialTextureNormal(',
    );
    expect(GET_SURFACE_RECORD_MAPPED_PBR_GLSL).toContain(
      'centerValid && uValid && vValid',
    );
    expect(GET_SURFACE_RECORD_MAPPED_PBR_GLSL).toContain(
      'normal / absScale - sign( bumpScale ) * slope',
    );
    expect(ATTENUATE_HIT_MAPPED_PBR_GLSL).toContain('vec4 mapSample;');
    expect(ATTENUATE_HIT_MAPPED_PBR_GLSL).toContain('vec4 alphaSample;');
  });

  it('pins rich identity, black-radiance, normal, bump, and anisotropy fallbacks', () => {
    expect(getSurfaceRecordRich).toContain('float attenuationThickness = material.thickness;');
    expect(getSurfaceRecordRich).toContain('vec4 emissiveSample;');
    expect(getSurfaceRecordRich).toContain('vec4 lightSample;');
    expect(getSurfaceRecordRich).toContain('decodeMaterialTextureNormal(');
    expect(getSurfaceRecordRich).toContain('centerValid && uValid && vValid');
    expect(getSurfaceRecordRich).toContain('float directionScale = max( abs( rg.x ), abs( rg.y ) );');
    expect(getSurfaceRecordRich).toContain('bool anisotropyMapApplied = false;');
    expect(getSurfaceRecordRich).toContain('int bsdfBasisUvLayer = anisotropyMapApplied');
    expect(attenuateHitRich).toContain('vec4 baseColorSample;');
    expect(attenuateHitRich).toContain('vec4 alphaSample;');
  });

  it('derives and consumes the opposite thin-sheet interface independently', () => {
    expect(getSurfaceRecordRich).toContain(
      'oppositeNormalMap = material.backLayerNormalMap;',
    );
    expect(getSurfaceRecordRich).toContain('vec4 oppositeNormalSample;');
    expect(getSurfaceRecordRich).toContain(
      'oppositeNormalSample, oppositeNormalScale, oppositeTexNormal',
    );
    expect(getSurfaceRecordRich).toContain(
      'oppositeNormal, oppositeBumpBasis, bumpSize',
    );
    expect(bsdfRich).toContain('result.normal = surf.oppositeNormal;');
    expect(bsdfRich).toContain('result.normalBasis = surf.oppositeNormalBasis;');
    expect(bsdfRich).toContain('SurfaceRecord exitSurf = oppositeFacingSurface( surf, false );');
  });
});
