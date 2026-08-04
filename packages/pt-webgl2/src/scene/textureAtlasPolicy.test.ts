import { describe, expect, it } from 'vitest';
import type { MaterialSpec } from '@vitrum/core';
import {
  MATERIAL_MAP_FIELD_ORDER,
  MATERIAL_PIXELS,
  MATERIAL_WRAP_TEXEL_OFFSET,
  packMaterialsTexture,
} from './materialsTexture.js';
import { packTextureAtlas } from './texturesArray.js';
import { MATERIAL_MAPPED_PBR_GLSL } from '../glsl/shader/structs/material_mapped_pbr.glsl.js';
import { MATERIAL_MAPPED_RICH_GLSL } from '../glsl/shader/structs/material_mapped_rich.glsl.js';

function handle(size: number, value: number): unknown {
  return {
    width: size,
    height: size,
    data: new Float32Array(size * size * 4).fill(value),
    __vitrum_hint__: {
      channels: 4,
      dataType: 'float32',
      colorSpace: 'linear',
    },
  };
}

function baseMapMaterial(
  textureHandle: unknown,
  sampler: Partial<NonNullable<MaterialSpec['baseColorMap']>> = {},
): MaterialSpec {
  return {
    baseColor: [1, 1, 1],
    roughness: 1,
    metallic: 0,
    baseColorMap: { handle: textureHandle, ...sampler },
  };
}

describe('tiled material-atlas policy', () => {
  it('encodes native extent and both source offsets without changing sampler modes', () => {
    const anchor = handle(8, 0);
    const fillers = [handle(2, 0.1), handle(2, 0.2), handle(2, 0.3)];
    const target = handle(2, 0.4);
    const materials = [
      baseMapMaterial(anchor),
      ...fillers.map((texture) => baseMapMaterial(texture)),
      baseMapMaterial(target, {
        wrapS: 'mirrored-repeat',
        wrapT: 'clamp-to-edge',
        magFilter: 'linear',
        minFilter: 'linear',
        mipFilter: 'linear',
      }),
    ];
    const atlas = packTextureAtlas(materials)!;
    const placement = atlas.layerOfByColorSpace.placements?.srgb.get(target);
    expect(placement).toMatchObject({ layer: 1, x: 2, y: 2, width: 2, height: 2 });

    const packed = packMaterialsTexture(materials, atlas.layerOfByColorSpace);
    const baseMapIndex = MATERIAL_MAP_FIELD_ORDER.indexOf('baseColorMap');
    const policyOffset =
      (4 * MATERIAL_PIXELS + MATERIAL_WRAP_TEXEL_OFFSET + baseMapIndex) * 4;
    expect(Array.from(packed.data.slice(policyOffset, policyOffset + 4))).toEqual([
      2 * 4 + 2, // width + mirrored-repeat
      2 * 4 + 1, // height + clamp-to-edge
      2 * 4 + 2, // source x + linear mip filter
      2 * 4 + 3, // source y + linear mag/min filters
    ]);
  });

  it.each([
    ['mapped PBR', MATERIAL_MAPPED_PBR_GLSL],
    ['mapped rich', MATERIAL_MAPPED_RICH_GLSL],
  ])('%s shader decodes offset and sampler policy lanes independently', (_name, glsl) => {
    expect(glsl).toContain('bool decodeMaterialTexturePolicy(');
    expect(glsl).toContain('baseOffset = ivec2( packedMip / 4, packedFilter / 4 );');
    expect(glsl).toContain('sourceOffset = baseOffset / divisor;');
    expect(glsl).toContain(
      'filterPair = packedFilter - ( packedFilter / 4 ) * 4;',
    );
    expect(glsl).toContain(
      'mipFilter = packedMip - ( packedMip / 4 ) * 4;',
    );
  });
});
