import type { MaterialSpec, MeshPrimitive, Scene } from '@vitrum/core';
import { mergeWorldSpaceFromCore } from '@vitrum/shared-bvh';
import { describe, expect, it } from 'vitest';
import { DEFAULT_TRACE_FEATURES } from '../featureTypes.js';
import {
  ATTR_LAYER_TANGENT,
  packAttributesArray,
} from '../scene/attributesTextureArray.js';
import { materialUsesScalarRichWebGl2Shader } from '../scene/sceneTraceFeatures.js';
import { composeTraceGlsl } from './composeTraceGlsl.js';

const BASE_MATERIAL: MaterialSpec = {
  baseColor: [0.7, 0.6, 0.5],
  roughness: 0.35,
  metallic: 0.8,
  anisotropy: 0.75,
};

function uvRotatedTriangle(): Scene {
  const primitive: MeshPrimitive = {
    kind: 'mesh',
    id: 'uv-rotated',
    positions: new Float32Array([
      0, 0, 0,
      1, 0, 0,
      0, 1, 0,
    ]),
    normals: new Float32Array([
      0, 0, 1,
      0, 0, 1,
      0, 0, 1,
    ]),
    // dP/du points along +Y, unlike the arbitrary +X normal-only basis.
    uvs: new Float32Array([
      0, 0,
      0, 1,
      1, 0,
    ]),
    indices: new Uint32Array([0, 1, 2]),
    material: BASE_MATERIAL,
  };
  return {
    primitives: [primitive],
    emitters: [],
    environment: { kind: 'none' },
  };
}

describe('pt-webgl2 scalar-rich tangent fallback evidence', () => {
  it('derives the missing tangent from UV gradients and consumes that attribute', () => {
    const merged = mergeWorldSpaceFromCore(uvRotatedTriangle(), {
      positionStride: 4,
    });
    const attributes = packAttributesArray(merged);
    const floatsPerLayer = attributes.dim * attributes.dim * 4;
    const tangentBase = ATTR_LAYER_TANGENT * floatsPerLayer;

    for (let vertex = 0; vertex < 3; vertex += 1) {
      const offset = tangentBase + vertex * 4;
      expect(attributes.data[offset]).toBeCloseTo(0, 6);
      expect(attributes.data[offset + 1]).toBeCloseTo(1, 6);
      expect(attributes.data[offset + 2]).toBeCloseTo(0, 6);
      expect(attributes.data[offset + 3]).toBe(-1);
    }

    const scalarRich = composeTraceGlsl({
      ...DEFAULT_TRACE_FEATURES,
      scalarRichMaterials: true,
      mappedRichMaterials: false,
    });
    expect(scalarRich).toContain(
      'attributesArray, ATTR_TANGENT, surfaceHit.barycoord, surfaceHit.faceIndices.xyz',
    );
    expect(scalarRich).toContain(
      'surf.normalBasis = getBasisFromNormalAndTangent( surf.normal, tangentSample );',
    );
  });

  it('routes every normal-affecting material map out of scalar-rich', () => {
    const texture = {
      handle: {
        width: 1,
        height: 1,
        data: new Uint8Array([128, 128, 255, 255]),
        __vitrum_hint__: { channels: 4 as const },
      },
    };
    const mappedMaterials: readonly MaterialSpec[] = [
      { ...BASE_MATERIAL, normalMap: texture },
      { ...BASE_MATERIAL, bumpMap: texture },
      { ...BASE_MATERIAL, clearcoatNormalMap: texture },
      {
        ...BASE_MATERIAL,
        frontLayer: { transmission: [1, 1, 1], normalMap: texture },
      },
      {
        ...BASE_MATERIAL,
        backLayer: { transmission: [1, 1, 1], normalMap: texture },
      },
    ];

    for (const material of mappedMaterials) {
      expect(materialUsesScalarRichWebGl2Shader(material)).toBe(false);
    }
  });
});
