import { describe, expect, it } from 'vitest';
import type { Scene } from '@vitrum/core';
import { asMat4 } from '@vitrum/core';
import {
  buildPackedScene,
  rebuildTlasForSceneTransforms,
  sceneCenterRadiusForPackedGeometry,
} from '../scene/uploadSceneBuffers.js';

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
    expect(packed.materials.length).toBe(116); // VOL-THICKNESS: MATERIAL_FLOAT_STRIDE 112 -> 116
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
    expect(packed.warnings.some((warning) => warning.includes('a-sphere') && warning.includes('skipped'))).toBe(false);
  });

  it('packs every analytic shape into the shader two-vec4 ABI', () => {
    const material = {
      baseColor: [0.8, 0.2, 0.1] as const,
      roughness: 0.5,
      metallic: 0.1,
    };
    const scene: Scene = {
      primitives: [
        {
          kind: 'analytic',
          id: 'sphere',
          shape: 'sphere',
          params: new Float32Array([1, 2, 3, 4]),
          material,
        },
        {
          kind: 'analytic',
          id: 'box',
          shape: 'box',
          params: new Float32Array([5, 6, 7, 8, 9, 10]),
          material,
        },
        {
          kind: 'analytic',
          id: 'capsule',
          shape: 'capsule',
          params: new Float32Array([11, 12, 13, 14, 15, 16, 17]),
          material,
        },
        {
          kind: 'analytic',
          id: 'cylinder',
          shape: 'cylinder',
          params: new Float32Array([18, 19, 20, 21, 22]),
          material,
        },
        {
          kind: 'analytic',
          id: 'came',
          shape: 'h-channel-came',
          params: new Float32Array([30, 12, 14, 2]),
          material,
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    };

    const packed = buildPackedScene(scene);
    expect(packed.analyticCount).toBe(5);
    expect(Array.from(packed.analyticHeaders)).toEqual([
      1, 0, 0, 0,
      2, 1, 2, 0,
      3, 2, 4, 0,
      4, 3, 6, 0,
      5, 4, 8, 0,
    ]);
    expect(Array.from(packed.analyticParams)).toEqual([
      1, 2, 3, 4, 0, 0, 0, 0,
      5, 6, 7, 0, 8, 9, 10, 0,
      11, 12, 13, 0, 14, 15, 16, 17,
      18, 19, 20, 21, 22, 0, 0, 0,
      30, 12, 14, 2, 0, 0, 0, 0,
    ]);
  });

  it('includes transformed analytics in initial and incremental scene bounds', () => {
    const analyticScene: Scene = {
      primitives: [{
        kind: 'analytic',
        id: 'bounded-sphere',
        shape: 'sphere',
        params: new Float32Array([0, 0, 0, 2]),
        transform: asMat4(new Float32Array([
          2, 0, 0, 0,
          0, 3, 0, 0,
          0, 0, 4, 0,
          100, -20, 7, 1,
        ])),
        material: {
          baseColor: [1, 1, 1],
          roughness: 0.5,
          metallic: 0,
        },
      }],
      emitters: [],
      environment: { kind: 'none' },
    };
    const packed = buildPackedScene(analyticScene);
    expect(packed.sceneCenter).toEqual([100, -20, 7]);
    expect(packed.sceneRadius).toBeCloseTo(Math.hypot(4, 6, 8), 12);

    const movedTransforms = new Float32Array(packed.analyticLocalToWorld);
    movedTransforms[12] = -50;
    movedTransforms[13] = 30;
    movedTransforms[14] = 9;
    const moved = sceneCenterRadiusForPackedGeometry({
      bvhNodes: packed.bvhNodes,
      tlasNodes: packed.tlasNodes,
      analyticHeaders: packed.analyticHeaders,
      analyticParams: packed.analyticParams,
      analyticLocalToWorld: movedTransforms,
    });
    expect(moved.center).toEqual([-50, 30, 9]);
    expect(moved.radius).toBeCloseTo(Math.hypot(4, 6, 8), 12);
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
    // BDPT pseudo-distant emitters use world-space scene bounds, not the local
    // BLAS root around the origin.
    expect(packed.sceneCenter[0]).toBeCloseTo(3.5, 5);
    expect(packed.sceneCenter[1]).toBeCloseTo(0.5, 5);
    expect(packed.sceneCenter[2]).toBeCloseTo(0, 5);
    expect(packed.sceneRadius).toBeCloseTo(Math.SQRT2 * 0.5, 5);
  });

  it('packs lite merged geometry as one world-space BLAS for multi-mesh scenes', () => {
    const scene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'left',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          material: { baseColor: [1, 0, 0], roughness: 0.4, metallic: 0 },
        },
        {
          kind: 'mesh',
          id: 'right',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          transform: asMat4(new Float32Array([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            2, 0, 0, 1,
          ])),
          material: { baseColor: [0, 1, 0], roughness: 0.4, metallic: 0 },
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    };

    const packed = buildPackedScene(scene, { geometryMode: 'merged' });
    expect(packed.triangleCount).toBe(2);
    expect(packed.tlasNodes.length).toBe(0);
    expect(packed.tlasBlasRoots.length).toBe(0);
    expect(packed.primitiveTlasBindings).toEqual([]);
    expect(packed.indices.length).toBe(8);
    expect(packed.indices[3]).toBe(0);
    expect(packed.indices[7]).toBe(0);
    expect(packed.triMaterialIds.length).toBe(2);
    expect(new Set(Array.from(packed.triMaterialIds))).toEqual(new Set([0, 1]));
    expect(packed.materials.length).toBe(232);
    expect(Array.from(packed.positions.slice(12, 24))).toEqual([
      2, 0, 0, 0,
      3, 0, 0, 0,
      2, 1, 0, 0,
    ]);
  });

  it('preserves authored tangents and handedness in lite merged geometry', () => {
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

    const packed = buildPackedScene(withTangents, { geometryMode: 'merged' });
    expect(Array.from(packed.tangents.slice(0, 12))).toEqual([
      1, 0, 0, -1,
      1, 0, 0, -1,
      1, 0, 0, -1,
    ]);
  });

  it('bakes primitive-constant COLOR_0 RGB into lite merged material baseColor', () => {
    const scene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'constant-colored',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          colors: new Float32Array([
            0.5, 0.25, 1,
            0.5, 0.25, 1,
            0.5, 0.25, 1,
          ]),
          material: { baseColor: [0.8, 0.4, 0.2], roughness: 0.4, metallic: 0 },
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    };

    const packed = buildPackedScene(scene, { geometryMode: 'merged' });
    expect(packed.materials[0]).toBeCloseTo(0.4);
    expect(packed.materials[1]).toBeCloseTo(0.1);
    expect(packed.materials[2]).toBeCloseTo(0.2);
    expect(Array.from(packed.colors.slice(0, 12))).toEqual([
      0.5, 0.25, 1, 1,
      0.5, 0.25, 1, 1,
      0.5, 0.25, 1, 1,
    ]);
  });

  it('packs lite merged geometry by baking instanced meshes into root-zero BLAS geometry', () => {
    const scene: Scene = {
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
            4, 0, 0, 1,
          ])),
        ],
      }],
      emitters: [],
      environment: { kind: 'none' },
    };

    const packed = buildPackedScene(scene, { geometryMode: 'merged' });
    expect(packed.triangleCount).toBe(2);
    expect(packed.tlasNodes.length).toBe(0);
    expect(packed.indices.length).toBe(8);
    expect(packed.triMaterialIds.length).toBe(2);
    expect(new Set(Array.from(packed.triMaterialIds))).toEqual(new Set([0]));
    expect(Array.from(packed.positions.slice(12, 24))).toEqual([
      4, 0, 0, 0,
      5, 0, 0, 0,
      4, 1, 0, 0,
    ]);
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
