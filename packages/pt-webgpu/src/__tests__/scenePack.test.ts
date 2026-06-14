import { describe, expect, it } from 'vitest';
import type { Scene } from '@vitrum/core';
import { asMat4 } from '@vitrum/core';
import { buildPackedScene, rebuildTlasForSceneTransforms } from '../scene/uploadSceneBuffers.js';

function makeScene(): Scene {
  return {
    primitives: [{
      kind: 'mesh',
      id: 'tri',
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      material: {
        baseColor: [0.25, 0.5, 0.75],
        roughness: 0.4,
        metallic: 0.2,
        emissive: [0.2, 0.1, 0],
        emissiveIntensity: 2,
      },
    }],
    emitters: [{
      kind: 'directional',
      id: 'sun',
      direction: [0, -1, 0],
      color: [1, 0.8, 0.6],
      intensity: 3,
    }],
    environment: { kind: 'none' },
  };
}

describe('buildPackedScene core packing', () => {
  it('packs one triangle and material payload', () => {
    const packed = buildPackedScene(makeScene());
    expect(packed.triangleCount).toBe(1);
    expect(Array.from(packed.indices)).toEqual([0, 1, 2, 0]);
    expect(packed.tangents.length).toBe(packed.positions.length);
    expect(Array.from(packed.tangents)).toEqual(new Array(packed.positions.length).fill(0));
    expect(packed.colors.length).toBe(packed.positions.length);
    expect(Array.from(packed.colors)).toEqual(new Array(packed.positions.length).fill(1));
    expect(packed.materials.length).toBe(112); // SPEC-01: MATERIAL_FLOAT_STRIDE 108 → 112 (KHR specular scalars)
    expect(packed.materials[0]).toBeCloseTo(0.25);
    expect(packed.materials[4]).toBeCloseTo(0.4);
  });

  it('preserves authored COLOR_0 colors for full-tier baseColor modulation', () => {
    const scene = makeScene();
    const mesh = scene.primitives[0]!;
    if (mesh.kind !== 'mesh') throw new Error('expected mesh');
    const withColors: Scene = {
      ...scene,
      primitives: [{
        ...mesh,
        colors: new Float32Array([
          1, 0, 0, 0.25,
          0, 1, 0, 0.5,
          0, 0, 1, 0.75,
        ]),
      }],
    };
    const packed = buildPackedScene(withColors);
    expect(Array.from(packed.colors.slice(0, 12))).toEqual([
      1, 0, 0, 0.25,
      0, 1, 0, 0.5,
      0, 0, 1, 0.75,
    ]);
  });

  it('preserves authored tangents for full-tier material normal reconstruction', () => {
    const scene = makeScene();
    const mesh = scene.primitives[0]!;
    if (mesh.kind !== 'mesh') throw new Error('expected mesh');
    const withTangents: Scene = {
      ...scene,
      primitives: [{
        ...mesh,
        tangents: new Float32Array([
          1, 0, 0, -1,
          1, 0, 0, -1,
          1, 0, 0, -1,
        ]),
      }],
    };
    const packed = buildPackedScene(withTangents);
    expect(Array.from(packed.tangents.slice(0, 12))).toEqual([
      1, 0, 0, -1,
      1, 0, 0, -1,
      1, 0, 0, -1,
    ]);
  });

  it('packs analytic primitive payload for shader intersections', () => {
    const base = makeScene();
    const scene: Scene = {
      ...base,
      primitives: [
        ...base.primitives,
        {
          kind: 'analytic',
          id: 'a-sphere',
          shape: 'sphere',
          params: new Float32Array([0, 0, 0, 0.5]),
          material: { baseColor: [0.8, 0.2, 0.1], roughness: 0.5, metallic: 0.1 },
        },
      ],
    };
    const packed = buildPackedScene(scene);
    expect(packed.analyticCount).toBe(1);
    expect(packed.analyticHeaders.length).toBe(4);
    expect(packed.analyticParams.length).toBe(8);
  });

  it('builds and uploads TLAS metadata buffers', () => {
    const packed = buildPackedScene(makeScene());
    expect(packed.tlasNodes.length).toBeGreaterThan(0);
    expect(packed.tlasBlasRoots[0]).toBe(0);
    expect(packed.tlasInstanceWorldToLocal.length).toBe(16);
    expect(packed.tlasInstanceLocalToWorld.length).toBe(16);
  });

  it('packs instanced meshes as one BLAS with multiple TLAS instances', () => {
    const instancedScene: Scene = {
      primitives: [{
        kind: 'instanced-mesh',
        id: 'inst-tri',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 },
        instances: [
          asMat4(new Float32Array([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
          ])),
          asMat4(new Float32Array([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            2, 0, 0, 1,
          ])),
        ],
      }],
      emitters: [],
      environment: { kind: 'none' },
    };

    const packed = buildPackedScene(instancedScene);
    expect(packed.triangleCount).toBe(1);
    expect(packed.tlasBlasRoots.length).toBe(2);
    expect(packed.tlasBlasRoots[0]).toBe(0);
    expect(packed.tlasBlasRoots[1]).toBe(0);
    expect(packed.tlasInstanceWorldToLocal.length).toBe(32);
    expect(packed.tlasInstanceLocalToWorld.length).toBe(32);
    // Translation lives in mat4[12] in this column-major pack.
    expect(packed.tlasInstanceLocalToWorld[28]).toBeCloseTo(2, 5);
    expect(packed.tlasInstanceWorldToLocal[28]).toBeCloseTo(-2, 5);
  });

  it('keeps mesh geometry local and expresses mesh transform via TLAS matrices', () => {
    const translatedScene: Scene = {
      primitives: [{
        kind: 'mesh',
        id: 'translated-tri',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.4, metallic: 0.1 },
        transform: asMat4(new Float32Array([
          1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1, 0,
          3, 0, 0, 1,
        ])),
      }],
      emitters: [],
      environment: { kind: 'none' },
    };

    const packed = buildPackedScene(translatedScene);
    expect(packed.triangleCount).toBe(1);
    // Vertex payload remains in local space; TLAS matrices carry placement.
    expect(Array.from(packed.positions.slice(0, 12))).toEqual([
      0, 0, 0, 0,
      1, 0, 0, 0,
      0, 1, 0, 0,
    ]);
    expect(packed.tlasBlasRoots.length).toBe(1);
    expect(packed.tlasBlasRoots[0]).toBe(0);
    expect(packed.tlasInstanceLocalToWorld[12]).toBeCloseTo(3, 5);
    expect(packed.tlasInstanceWorldToLocal[12]).toBeCloseTo(-3, 5);
  });

  it('refits existing TLAS nodes for transform-only scene changes', () => {
    const base = makeScene();
    const baseMesh = base.primitives[0];
    if (baseMesh == null || baseMesh.kind !== 'mesh') {
      throw new Error('test setup error: expected mesh primitive');
    }
    const packed = buildPackedScene(base);
    const moved: Scene = {
      ...base,
      primitives: [
        {
          ...baseMesh,
          transform: asMat4(new Float32Array([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            2, 0, 0, 1,
          ])),
        },
      ],
    };
    const rebuilt = rebuildTlasForSceneTransforms(
      moved,
      packed.primitiveTlasBindings,
      {
        tlasNodes: packed.tlasNodes,
        tlasInstanceIndices: packed.tlasInstanceIndices,
        tlasBlasRoots: packed.tlasBlasRoots,
        tlasInstanceWorldToLocal: packed.tlasInstanceWorldToLocal,
      },
    );
    expect(rebuilt.ok).toBe(true);
    if (!rebuilt.ok) return;
    expect(rebuilt.tlasNodes.length).toBe(packed.tlasNodes.length);
    expect(Array.from(rebuilt.tlasInstanceIndices)).toEqual(Array.from(packed.tlasInstanceIndices));
    expect(rebuilt.tlasInstanceLocalToWorld[12]).toBeCloseTo(2, 5);
    expect(rebuilt.tlasInstanceWorldToLocal[12]).toBeCloseTo(-2, 5);
    expect(rebuilt.tlasNodes[0]).not.toBe(packed.tlasNodes[0]);
  });
});
