import { describe, expect, it } from 'vitest';
import { asMat4, type Scene } from '@vitrum/core';
import { luminance } from '@vitrum/shared-samplers';
import { meshTriangleArea } from '../bdpt/flatEmitterWalk.js';
import { buildLightTreeInputForScene } from '../scene/emitterPacking.js';
import { buildPackedScene } from '../scene/uploadSceneBuffers.js';

function baseScene(): Scene {
  return {
    primitives: [{
      kind: 'mesh',
      id: 'tri',
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      material: { baseColor: [1, 1, 1], roughness: 0.4, metallic: 0 },
    }],
    emitters: [],
    environment: { kind: 'none' },
  };
}

function quadScene(kind: 'mesh' | 'instanced-mesh' = 'mesh'): Scene {
  const primitive = {
    kind,
    id: 'quad',
    positions: new Float32Array([
      0, 0, 0,
      1, 0, 0,
      1, 1, 0,
      0, 1, 0,
    ]),
    normals: new Float32Array([
      0, 0, 1,
      0, 0, 1,
      0, 0, 1,
      0, 0, 1,
    ]),
    indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    material: { baseColor: [1, 1, 1], roughness: 0.4, metallic: 0 },
    ...(kind === 'instanced-mesh'
      ? {
          instances: [
            asMat4([
              1, 0, 0, 0,
              0, 1, 0, 0,
              0, 0, 1, 0,
              0, 0, 0, 1,
            ]),
            asMat4([
              1, 0, 0, 0,
              0, 1, 0, 0,
              0, 0, 1, 0,
              10, 0, 0, 1,
            ]),
          ],
        }
      : {}),
  } as Scene['primitives'][number];
  return {
    primitives: [primitive],
    emitters: [{ kind: 'mesh-area', id: 'm', meshId: 'quad', color: [0.25, 0.5, 1], intensity: 4 }],
    environment: { kind: 'none' },
  };
}

function triAt(data: Float32Array, tri: number): {
  a: [number, number, number];
  b: [number, number, number];
  c: [number, number, number];
  r: [number, number, number];
} {
  const o = tri * 16;
  return {
    a: [data[o]!, data[o + 1]!, data[o + 2]!],
    b: [data[o + 4]!, data[o + 5]!, data[o + 6]!],
    c: [data[o + 8]!, data[o + 9]!, data[o + 10]!],
    r: [data[o + 12]!, data[o + 13]!, data[o + 14]!],
  };
}

function expectVec3Close(actual: readonly [number, number, number], expected: readonly [number, number, number]): void {
  expect(actual[0]).toBeCloseTo(expected[0], 6);
  expect(actual[1]).toBeCloseTo(expected[1], 6);
  expect(actual[2]).toBeCloseTo(expected[2], 6);
}

describe('buildPackedScene emitter + environment packing', () => {
  it('packs point, spot, rect-area, and mesh-area lights', () => {
    const scene: Scene = {
      ...baseScene(),
      emitters: [
        { kind: 'point', id: 'p', position: [2, 3, 4], color: [0.5, 1, 0.25], intensity: 8 },
        { kind: 'spot', id: 's', position: [5, 6, 7], direction: [0, -1, 0], angle: 0.5, color: [1, 0.5, 0.25], intensity: 4 },
        { kind: 'rect-area', id: 'r', position: [0, 1, 0], uAxis: [0.5, 0, 0], vAxis: [0, 0.5, 0], color: [1, 1, 1], intensity: 10 },
        { kind: 'mesh-area', id: 'm', meshId: 'tri', color: [0.5, 0.25, 1], intensity: 6 },
      ],
    };
    const packed = buildPackedScene(scene);
    expect(packed.pointLightCount).toBe(1);
    expect(packed.spotLightCount).toBe(1);
    expect(packed.rectAreaLightCount).toBe(1);
    expect(packed.meshAreaLightCount).toBe(1);
  });

  it('expands mesh-area emitters to every referenced triangle', () => {
    const scene = quadScene('mesh');
    const packed = buildPackedScene(scene);
    expect(packed.meshAreaLightCount).toBe(2);
    expect(packed.meshAreaLightsData.length).toBe(2 * 16);
    const t0 = triAt(packed.meshAreaLightsData, 0);
    const t1 = triAt(packed.meshAreaLightsData, 1);
    expectVec3Close(t0.a, [0, 0, 0]);
    expectVec3Close(t0.b, [1, 0, 0]);
    expectVec3Close(t0.c, [1, 1, 0]);
    expectVec3Close(t1.a, [0, 0, 0]);
    expectVec3Close(t1.b, [1, 1, 0]);
    expectVec3Close(t1.c, [0, 1, 0]);
    expectVec3Close(t0.r, [1, 2, 4]);
    expectVec3Close(t1.r, [1, 2, 4]);

    const tree = buildLightTreeInputForScene(scene);
    expect(tree.powers.length).toBe(2);
    expectVec3Close(tree.centroids[0]!, [2 / 3, 1 / 3, 0]);
    expectVec3Close(tree.centroids[1]!, [1 / 3, 2 / 3, 0]);
    expect(tree.powers[0]).toBeCloseTo(luminance(1, 2, 4) * 0.5, 6);
    expect(tree.powers[1]).toBeCloseTo(luminance(1, 2, 4) * 0.5, 6);
  });

  it('expands instanced mesh-area emitters across every instance and triangle', () => {
    const packed = buildPackedScene(quadScene('instanced-mesh'));
    expect(packed.meshAreaLightCount).toBe(4);
    const firstInstanceFirstTri = triAt(packed.meshAreaLightsData, 0);
    const secondInstanceFirstTri = triAt(packed.meshAreaLightsData, 2);
    expectVec3Close(firstInstanceFirstTri.a, [0, 0, 0]);
    expectVec3Close(firstInstanceFirstTri.b, [1, 0, 0]);
    expectVec3Close(firstInstanceFirstTri.c, [1, 1, 0]);
    expectVec3Close(secondInstanceFirstTri.a, [10, 0, 0]);
    expectVec3Close(secondInstanceFirstTri.b, [11, 0, 0]);
    expectVec3Close(secondInstanceFirstTri.c, [11, 1, 0]);
  });

  it('lowers disc-area emitters into equal-area mesh triangle records', () => {
    const radius = 2;
    const radiance: [number, number, number] = [3, 1.5, 0.75];
    const scene: Scene = {
      ...baseScene(),
      emitters: [{
        kind: 'disc-area',
        id: 'disc',
        position: [0, 2, 0],
        normal: [0, 1, 0],
        radius,
        color: [1, 0.5, 0.25],
        intensity: 3,
      }],
    };
    const packed = buildPackedScene(scene);
    expect(packed.rectAreaLightCount).toBe(0);
    expect(packed.meshAreaLightCount).toBe(32);
    let area = 0;
    for (let i = 0; i < packed.meshAreaLightCount; i += 1) {
      const tri = triAt(packed.meshAreaLightsData, i);
      expectVec3Close(tri.r, radiance);
      area += meshTriangleArea(tri.a, tri.b, tri.c);
    }
    expect(area).toBeCloseTo(Math.PI * radius * radius, 5);

    const tree = buildLightTreeInputForScene(scene);
    expect(tree.powers.length).toBe(32);
    const totalPower = tree.powers.reduce((sum, p) => sum + p, 0);
    expect(totalPower).toBeCloseTo(
      luminance(radiance[0], radiance[1], radiance[2]) * Math.PI * radius * radius,
      4,
    );
  });

  it('cameraVisibleEmitters re-attaches mesh-area emitter radiance onto the primitive material (color·intensity)', () => {
    // This scene models an emissive mesh as a sampled mesh-area emitter while the
    // primitive material remains non-emissive. With cameraVisibleEmitters the
    // packer re-attaches the emitter radiance so the primitive glows to the
    // camera. The re-attached emissive (packed floats 4..6 of material slot 0,
    // pre-multiplied by emissiveIntensity) must EXACTLY equal the mesh-area NEE
    // radiance color·intensity = [0.5,0.25,1]·6 — so camera glow matches the
    // lit appearance.
    const scene: Scene = {
      ...baseScene(),
      emitters: [{ kind: 'mesh-area', id: 'm', meshId: 'tri', color: [0.5, 0.25, 1], intensity: 6 }],
    };
    const off = buildPackedScene(scene);
    // Default (off): the primitive material emissive stays at its scene value (0).
    expect([off.materials[4], off.materials[5], off.materials[6]]).toEqual([0, 0, 0]);
    const on = buildPackedScene(scene, { cameraVisibleEmitters: true });
    expect(on.materials[4]).toBeCloseTo(0.5 * 6, 5);
    expect(on.materials[5]).toBeCloseTo(0.25 * 6, 5);
    expect(on.materials[6]).toBeCloseTo(1 * 6, 5);
    // Everything else byte-identical (only emissive changed).
    expect(on.materials[0]).toBe(off.materials[0]); // baseColor.r
    expect(on.materials.length).toBe(off.materials.length);
  });

  it('cameraVisibleEmitters does NOT re-attach when the emitter has no matching primitive (meshId mismatch)', () => {
    const scene: Scene = {
      ...baseScene(),
      emitters: [{ kind: 'mesh-area', id: 'm', meshId: 'no-such-mesh', color: [1, 1, 1], intensity: 9 }],
    };
    const on = buildPackedScene(scene, { cameraVisibleEmitters: true });
    expect([on.materials[4], on.materials[5], on.materials[6]]).toEqual([0, 0, 0]);
  });

  it('packs HDRI map payload and CDF', () => {
    const scene: Scene = {
      ...baseScene(),
      environment: {
        kind: 'hdri',
        hdri: { width: 2, height: 2, data: new Float32Array([4,1,1, 1,4,1, 1,1,4, 2,2,2]) },
      },
    };
    const packed = buildPackedScene(scene);
    expect(packed.hasEnvironmentMap).toBe(true);
    expect(packed.environmentMapTexels.length).toBe(16);
    expect(packed.environmentMapCdf.length).toBe(5);
  });

  it('warns and falls back when HDRI payload is opaque', () => {
    const scene: Scene = { ...baseScene(), environment: { kind: 'hdri', hdri: { mock: true } as never } };
    const packed = buildPackedScene(scene);
    expect(packed.hasEnvironmentMap).toBe(false);
    expect(packed.warnings.some((w) => w.includes('HDRI environment'))).toBe(true);
  });
});
