import { describe, expect, it } from 'vitest';
import type { MaterialSpec } from '@vitrum/core';
import { TLAS_TRAVERSAL_CORE_WGSL } from '@vitrum/shared-bvh';

import {
  MATERIAL_MAP_META_TEXEL_OFFSETS,
  MATERIAL_MAP_META_TEXELS_PER_TRI,
  packMaterialTextureAtlas,
} from '../bvh/materialTextureAtlasPack.js';
import {
  packBVHBeerColorsFromCore,
  packBVHRoughMetalFromCore,
} from '../restir/packingHelpers.js';
import { makeProbeUpdateRaysWGSL } from '../ddgi/wgsl/probeUpdateRays.wgsl.js';
import { MATERIAL_ATLAS_WGSL } from '../shaders/materialAtlas.wgsl.js';
import { NRC_INDEPENDENT_SUFFIX_WGSL } from '../shaders/nrcIndependentSuffix.wgsl.js';
import { REFRACTIVE_CAUSTICS_WGSL } from '../shaders/refractiveCaustics.wgsl.js';
import { RIS_GI_WGSL } from '../shaders/risGi.wgsl.js';
import { NATIVE_GLASS_GI_WGSL } from '../shaders/risGiGlassWalk.wgsl.js';
import { SCENE_TRAVERSAL_WGSL } from '../shaders/sceneTraversal.wgsl.js';
import { SURFACE_TEXTURES_WGSL } from '../shaders/surfaceTextures.wgsl.js';
import { TRANSPARENT_OIT_WGSL } from '../shaders/transparentOit.wgsl.js';

const base = {
  baseColor: [0.4, 0.5, 0.6],
  roughness: 0.5,
  metallic: 0,
} as const satisfies MaterialSpec;

describe('walkaround double-sided transport', () => {
  it('packs a dedicated per-triangle side payload without consuming AO bits', () => {
    const aoMap = { handle: { test: true } } as NonNullable<MaterialSpec['aoMap']>;
    const materials: MaterialSpec[] = [
      { ...base, aoMap, aoMapIntensity: 0.67 },
      { ...base, aoMap, aoMapIntensity: 0.67, doubleSided: true },
    ];
    const triMaterials = new Uint32Array([0, 1]);

    const beer = packBVHBeerColorsFromCore(triMaterials, materials, 2);
    expect(beer[0]! & 0xff).toBe(0);
    expect(beer[1]! & 0xff).toBe(1);

    const roughMetal = packBVHRoughMetalFromCore(triMaterials, materials, 2);
    const aoBits = (Math.round(0.67 * 31) & 0x1f) << 3;
    expect(roughMetal[0]! & 0xf8).toBe(aoBits);
    expect(roughMetal[1]! & 0xf8).toBe(aoBits);

    const atlas = packMaterialTextureAtlas(materials, triMaterials, 2);
    expect(MATERIAL_MAP_META_TEXELS_PER_TRI).toBe(157);
    expect(MATERIAL_MAP_META_TEXEL_OFFSETS.SIDE_FLAGS).toBe(156);
    const side0 = (MATERIAL_MAP_META_TEXEL_OFFSETS.SIDE_FLAGS * 4);
    const side1 = ((MATERIAL_MAP_META_TEXELS_PER_TRI + MATERIAL_MAP_META_TEXEL_OFFSETS.SIDE_FLAGS) * 4);
    expect(atlas.baseColorMetaData[side0]).toBe(0);
    expect(atlas.baseColorMetaData[side1]).toBe(1);
  });

  it('corrects mirrored TLAS winding and face-forwards smooth normals', () => {
    expect(TLAS_TRAVERSAL_CORE_WGSL).toContain('fn tlasLinearOrientationSign(');
    expect(TLAS_TRAVERSAL_CORE_WGSL).toContain(
      '(*best).side = localHit.side * tlasLinearOrientationSign(l2w0, l2w1, l2w2);',
    );
    expect(SCENE_TRAVERSAL_WGSL).toContain(
      'n = safe_normalize(worldN) * tlasLinearOrientationSign(w2l0, w2l1, w2l2);',
    );

    const determinantSign = (c0: readonly number[], c1: readonly number[], c2: readonly number[]): number => {
      const cross = [
        c1[1]! * c2[2]! - c1[2]! * c2[1]!,
        c1[2]! * c2[0]! - c1[0]! * c2[2]!,
        c1[0]! * c2[1]! - c1[1]! * c2[0]!,
      ];
      const determinant = c0[0]! * cross[0]! + c0[1]! * cross[1]! + c0[2]! * cross[2]!;
      return determinant >= 0 ? 1 : -1;
    };
    expect(determinantSign([-1, 0, 0], [0, 1, 0], [0, 0, 1])).toBe(-1);
    expect(determinantSign([1, 0, 0], [0, 1, 0], [0, 0, 1])).toBe(1);
  });

  it('filters opaque one-sided backfaces but preserves double-sided and volume exits', () => {
    expect(MATERIAL_ATLAS_WGSL).toContain('fn materialSideAdmittedForHit(');
    expect(MATERIAL_ATLAS_WGSL).toContain('return doubleSided || transmissive;');
    expect(MATERIAL_ATLAS_WGSL).toContain('if (!materialSideAdmittedForHit(hit))');
    expect(TRANSPARENT_OIT_WGSL).toContain('if (!materialSideAdmittedForHit(hit))');
    expect(SURFACE_TEXTURES_WGSL).toContain(
      'let alphaT = materialShadowTransmittanceForHit(hit, word, false);',
    );

    const ddgi = makeProbeUpdateRaysWGSL(8);
    expect(ddgi).toContain('fn ddgiMaterialSideAdmittedForHit(');
    expect(ddgi).toContain('(mat.flags & MATERIAL_FLAG_DOUBLE_SIDED) != 0u');
    expect(ddgi).toContain('mat.transmission > 0.0');
  });

  it('evaluates the authored opposite-face normal map at reciprocal sheet interfaces', () => {
    expect(MATERIAL_ATLAS_WGSL).toContain(
      'fn applyFaceLayerNormalMapForSideForHit(',
    );
    expect(MATERIAL_ATLAS_WGSL).toContain(
      'fn applyNormalMapForSideForHit(',
    );
    for (const source of [
      RIS_GI_WGSL,
      NRC_INDEPENDENT_SUFFIX_WGSL,
      NATIVE_GLASS_GI_WGSL,
      REFRACTIVE_CAUSTICS_WGSL,
    ]) {
      expect(source).toContain('applyNormalMapForSideForHit(');
    }
    expect(RIS_GI_WGSL).toContain('slabFrontFacing = !slabFrontFacing;');
    expect(NRC_INDEPENDENT_SUFFIX_WGSL).toContain(
      'slabFrontFacing = !slabFrontFacing;',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'primaryHit, primarySmoothNormal, slabFrontFacing,',
    );
    expect(NATIVE_GLASS_GI_WGSL).toContain(
      'walkHit, walkSmoothNormal, slabFrontFacing,',
    );
    expect(NATIVE_GLASS_GI_WGSL).not.toContain('slabNormal = -slabNormal;');
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain(
      'let exitRoughness = faceLayerRoughness(mappedBaseRoughness, exitLayer);',
    );
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain(
      'hit, interfaceSmoothNormal, hit.side < 0.0,',
    );
  });

  it('matches an actual reciprocal non-flat tangent frame, including mirrored TLAS handedness', () => {
    type Vec3 = readonly [number, number, number];
    const cross = (a: Vec3, b: Vec3): Vec3 => [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
    ];
    const normalize = (v: Vec3): Vec3 => {
      const inv = 1 / Math.hypot(...v);
      return [v[0] * inv, v[1] * inv, v[2] * inv];
    };
    const mapped = (
      normal: Vec3,
      tangent: Vec3,
      handedness: number,
      sample: Vec3,
    ): Vec3 => {
      const bitangent = cross(normal, tangent).map(
        (lane) => lane * handedness,
      ) as unknown as Vec3;
      return normalize([
        tangent[0] * sample[0] + bitangent[0] * sample[1] + normal[0] * sample[2],
        tangent[1] * sample[0] + bitangent[1] * sample[1] + normal[1] * sample[2],
        tangent[2] * sample[0] + bitangent[2] * sample[1] + normal[2] * sample[2],
      ]);
    };
    const nonFlatSample = normalize([0.37, -0.41, 0.83]);
    const physicalFront: Vec3 = [0, 0, 1];
    const actualBack: Vec3 = [0, 0, -1];

    for (const mirrorSign of [1, -1]) {
      for (const authoredHandedness of [1, -1]) {
        const tangent: Vec3 = [mirrorSign, 0, 0];
        const worldHandedness = authoredHandedness * mirrorSign;
        const actual = mapped(
          actualBack,
          tangent,
          worldHandedness,
          nonFlatSample,
        );
        // materialNormalFrameForSideForHit flips the physical face-forward
        // normal before materialTangentFrameForHit derives its bitangent.
        const virtual = mapped(
          physicalFront.map((lane) => -lane) as unknown as Vec3,
          tangent,
          worldHandedness,
          nonFlatSample,
        );
        expect(virtual).toEqual(actual);
        expect(mapped(
          physicalFront,
          tangent,
          worldHandedness,
          nonFlatSample,
        )).not.toEqual(actual);
      }
    }

    expect(MATERIAL_ATLAS_WGSL).toContain(
      'fn materialNormalFrameForSideForHit(',
    );
    expect(MATERIAL_ATLAS_WGSL).toContain(
      'isFrontFace == (hit.side >= 0.0)',
    );
    expect(MATERIAL_ATLAS_WGSL).toContain(
      'let sideNormal = materialNormalFrameForSideForHit(',
    );
    expect(MATERIAL_ATLAS_WGSL).toContain(
      'authoredHandedness = authoredHandedness * tangentHandednessForLocalToWorld(',
    );
  });
});
