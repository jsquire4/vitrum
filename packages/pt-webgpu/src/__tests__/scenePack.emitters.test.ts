import { describe, expect, it } from 'vitest';
import { asMat4, type Scene } from '@vitrum/core';
import { luminance } from '@vitrum/shared-samplers';
import {
  buildLightTreeInputForScene,
  defaultDirectionalAngularDiameter,
  meshAreaEmitterAdjointRangeForScene,
} from '../scene/emitterPacking.js';
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

  it('reports a stable adjoint range for uncapped explicit mesh-area emitters', () => {
    const scene = quadScene('mesh');
    const range = meshAreaEmitterAdjointRangeForScene(scene, 'm');
    expect(range).toEqual({
      start: 0,
      count: 2,
      totalMeshAreaTriangles: 2,
      capped: false,
    });
  });

  it('subdivides implicit emissive-map mesh lights through the packed-scene path', () => {
    const scene: Scene = {
      primitives: [{
        kind: 'mesh',
        id: 'mapped-tri',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        uvs: new Float32Array([0.75, 0, 0.75, 0, 0.75, 0]),
        material: {
          baseColor: [1, 1, 1],
          roughness: 0.4,
          metallic: 0,
          emissive: [2, 2, 2],
          emissiveIntensity: 3,
          emissiveMap: {
            handle: {
              width: 2,
              height: 1,
              data: new Uint8Array([
                255, 0, 0, 255,
                0, 255, 0, 255,
              ]),
            },
          },
        },
      }],
      emitters: [],
      environment: { kind: 'none' },
    };

    const packed = buildPackedScene(scene);
    expect(packed.meshAreaLightCount).toBe(4);
    expect(packed.meshAreaLightsData.length).toBe(4 * 16);
    expectVec3Close(triAt(packed.meshAreaLightsData, 0).r, [0, 6, 0]);
    expect(packed.warnings).toEqual([]);
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

  it('packs disc-area emitters natively into the rect stream (shape tag = 1.0, π·r² area)', () => {
    // Native analytic disc emitters replace the 32-triangle fan, 2026-06-10 —
    // RENDER-CHANGING for disc-lit scenes, A/B in R9-B.
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
    // Disc is now packed into the rect stream, NOT the mesh stream.
    expect(packed.rectAreaLightCount).toBe(1);
    expect(packed.meshAreaLightCount).toBe(0);

    // Verify record layout: stride = 16 floats (4 vec4f).
    expect(packed.rectAreaLightsData.length).toBe(16);
    const d = packed.rectAreaLightsData;
    // vec4 0: center
    expect(d[0]).toBeCloseTo(0, 5);
    expect(d[1]).toBeCloseTo(2, 5);
    expect(d[2]).toBeCloseTo(0, 5);
    // vec4 1: uAxis = tangent × radius; |uAxis| must equal radius
    const uLen = Math.hypot(d[4]!, d[5]!, d[6]!);
    expect(uLen).toBeCloseTo(radius, 5);
    // vec4 2: vAxis = bitangent × radius; |vAxis| must equal radius
    const vLen = Math.hypot(d[8]!, d[9]!, d[10]!);
    expect(vLen).toBeCloseTo(radius, 5);
    // u and v must be orthogonal (packed in disc plane)
    const uvDot = d[4]! * d[8]! + d[5]! * d[9]! + d[6]! * d[10]!;
    expect(uvDot).toBeCloseTo(0, 5);
    // vec4 3: radiance.rgb matches color × intensity
    expect(d[12]).toBeCloseTo(radiance[0], 5);
    expect(d[13]).toBeCloseTo(radiance[1], 5);
    expect(d[14]).toBeCloseTo(radiance[2], 5);
    // shape tag = 1.0 (disc)
    expect(d[15]).toBeCloseTo(1.0, 5);

    // Light tree: one leaf, power = luminance × π·r²
    const tree = buildLightTreeInputForScene(scene);
    expect(tree.powers.length).toBe(1);
    const expectedPower = luminance(radiance[0], radiance[1], radiance[2]) * Math.PI * radius * radius;
    expect(tree.powers[0]).toBeCloseTo(expectedPower, 4);
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
    const scene: Scene = { ...baseScene(), environment: { kind: 'hdri', hdri: { mock: true } } };
    const packed = buildPackedScene(scene);
    expect(packed.hasEnvironmentMap).toBe(false);
    expect(packed.warnings.some((w) => w.includes('HDRI environment'))).toBe(true);
  });
});

// ── SHADOW-01 (2026-06-11) — emitter castShadowDisabled lanes ────────────────
//
// Lane map (emitterPacking.ts layout docs):
//   directional — angularDiameter lane SIGN-ENCODED: packed = -1 - ad when
//                 castShadow:false (raw ad >= 0 when true/undefined).
//   point       — vec4 2 .z (float index 10).
//   spot        — vec4 3 .z (float index 14).
//   rect/disc   — vec4 0 .w (float index 3).
//   mesh-area   — radiance vec4 .w (float index 15 of each 16-float record).
describe('SHADOW-01 emitter castShadowDisabled lanes', () => {
  it('packs castShadow:false into every light kind lane; defaults pack 0 / non-negative', () => {
    const scene: Scene = {
      ...baseScene(),
      emitters: [
        { kind: 'directional', id: 'd0', direction: [0, -1, 0], color: [1, 1, 1], intensity: 1, angularDiameter: 0.25, castShadow: false },
        { kind: 'directional', id: 'd1', direction: [0, -1, 0], color: [1, 1, 1], intensity: 1, angularDiameter: 0.25 },
        { kind: 'point', id: 'p0', position: [0, 1, 0], color: [1, 1, 1], intensity: 1, castShadow: false },
        { kind: 'point', id: 'p1', position: [0, 1, 0], color: [1, 1, 1], intensity: 1 },
        { kind: 'spot', id: 's0', position: [0, 1, 0], direction: [0, -1, 0], angle: 0.5, color: [1, 1, 1], intensity: 1, castShadow: false },
        { kind: 'spot', id: 's1', position: [0, 1, 0], direction: [0, -1, 0], angle: 0.5, color: [1, 1, 1], intensity: 1 },
        { kind: 'rect-area', id: 'r0', position: [0, 1, 0], uAxis: [1, 0, 0], vAxis: [0, 1, 0], color: [1, 1, 1], intensity: 1, castShadow: false },
        { kind: 'rect-area', id: 'r1', position: [0, 1, 0], uAxis: [1, 0, 0], vAxis: [0, 1, 0], color: [1, 1, 1], intensity: 1 },
        { kind: 'disc-area', id: 'c0', position: [0, 1, 0], normal: [0, -1, 0], radius: 0.5, color: [1, 1, 1], intensity: 1, castShadow: false },
        { kind: 'mesh-area', id: 'm0', meshId: 'tri', color: [1, 1, 1], intensity: 1, castShadow: false },
      ],
    };
    const packed = buildPackedScene(scene);

    // directional — sign-encoded angularDiameter (stride 8 floats / light).
    expect(packed.directionalLightsData[3]).toBeCloseTo(-1.25, 6);  // -1 - 0.25
    expect(packed.directionalLightsData[8 + 3]).toBeCloseTo(0.25, 6);
    expect(defaultDirectionalAngularDiameter(scene)).toBeCloseTo(-1.25, 6);

    // point — stride 12, lane 10.
    expect(packed.pointLightsData[10]).toBe(1);
    expect(packed.pointLightsData[12 + 10]).toBe(0);

    // spot — stride 16, lane 14.
    expect(packed.spotLightsData[14]).toBe(1);
    expect(packed.spotLightsData[16 + 14]).toBe(0);

    // rect — stride 16, lane 3; disc is appended after rects in the same stream.
    expect(packed.rectAreaLightsData[3]).toBe(1);
    expect(packed.rectAreaLightsData[16 + 3]).toBe(0);
    expect(packed.rectAreaLightsData[2 * 16 + 3]).toBe(1);  // the disc record

    // mesh-area — radiance .w of each 16-float triangle record.
    expect(packed.meshAreaLightCount).toBeGreaterThan(0);
    expect(packed.meshAreaLightsData[15]).toBe(1);
  });

  it('DEFAULT-PATH INVARIANT: flag-less emitters pack byte-identically (all lanes 0 / raw ad)', () => {
    const scene: Scene = {
      ...baseScene(),
      emitters: [
        { kind: 'directional', id: 'd', direction: [0, -1, 0], color: [1, 1, 1], intensity: 1 },
        { kind: 'point', id: 'p', position: [0, 1, 0], color: [1, 1, 1], intensity: 1 },
        { kind: 'spot', id: 's', position: [0, 1, 0], direction: [0, -1, 0], angle: 0.5, color: [1, 1, 1], intensity: 1 },
        { kind: 'rect-area', id: 'r', position: [0, 1, 0], uAxis: [1, 0, 0], vAxis: [0, 1, 0], color: [1, 1, 1], intensity: 1 },
        { kind: 'mesh-area', id: 'm', meshId: 'tri', color: [1, 1, 1], intensity: 1 },
      ],
    };
    const packed = buildPackedScene(scene);
    expect(packed.directionalLightsData[3]).toBe(0);   // no angularDiameter → 0, non-negative
    expect(defaultDirectionalAngularDiameter(scene)).toBe(0);
    expect(packed.pointLightsData[10]).toBe(0);
    expect(packed.spotLightsData[14]).toBe(0);
    expect(packed.rectAreaLightsData[3]).toBe(0);
    expect(packed.meshAreaLightsData[15]).toBe(0);
  });

  it('keeps first-directional angularDiameter non-negative when castShadow is enabled', () => {
    const scene: Scene = {
      ...baseScene(),
      emitters: [
        { kind: 'directional', id: 'd', direction: [0, -1, 0], color: [1, 1, 1], intensity: 1, angularDiameter: 0.125 },
      ],
    };
    const packed = buildPackedScene(scene);
    expect(packed.directionalLightsData[3]).toBeCloseTo(0.125, 6);
    expect(defaultDirectionalAngularDiameter(scene)).toBeCloseTo(0.125, 6);
  });
});
