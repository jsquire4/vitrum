import { describe, expect, it } from 'vitest';
import type { MaterialSpec, MeshPrimitive, Scene } from '@vitrum/core';
import type { WorldSpaceMergeResult } from '@vitrum/shared-bvh';
import {
  hasMeshAreaLightForPrimitive,
  packMeshAreaLights,
  TRI_AREA_LIGHT_TYPE,
  TRI_LIGHT_PIXELS,
} from './meshAreaLights.js';
import { materialEmissionExcludedFromMeshNee } from './meshEmitterPolicy.js';
import {
  FLOAT16_HALF_MIN_SUBNORMAL,
  float16BitsToFloat32,
} from './halfFloat.js';
import { packTextureAtlas } from './texturesArray.js';

// B4 — mesh-area triangle-light packer. Pure-CPU: builds a fake merged geometry
// stream + scene and checks triangle extraction, area, radiance, and the layout.

function fakeMerged(overrides: Partial<WorldSpaceMergeResult> = {}): WorldSpaceMergeResult {
  // One unit quad in the y=0 plane (2 tris, total area 1), merge-order indices.
  const positions = new Float32Array([
    0, 0, 0, 0, // v0 (stride 4)
    1, 0, 0, 0, // v1
    1, 0, 1, 0, // v2
    0, 0, 1, 0, // v3
  ]);
  const mergedIndices = new Uint32Array([0, 1, 2, 0, 2, 3]);
  return {
    bvhNodes: new Float32Array(),
    positions,
    positionStrideFloats: 4,
    indices: mergedIndices,
    bvhIndexStride: 3,
    triMaterialId: new Uint32Array([0, 0]),
    bvhTriToMergedTri: new Uint32Array([0, 1]),
    normals: new Float32Array(positions.length),
    tangents: new Float32Array(positions.length),
    colors: new Float32Array([
      1, 1, 1, 1,
      1, 1, 1, 1,
      1, 1, 1, 1,
      1, 1, 1, 1,
    ]),
    uvs: new Float32Array(8),
    mergedIndices,
    mergedTriMaterialId: new Uint32Array([0, 0]),
    materials: [],
    boundingBox: { min: [0, 0, 0], max: [1, 0, 1] },
    meshVertexRanges: [
      { name: 'panel', vertexStart: 0, vertexCount: 4, triStart: 0, triCount: 2 },
    ],
    warnings: [],
    vertexCount: 4,
    triangleCount: 2,
    ...overrides,
  };
}

function sceneWith(emitters: Scene['emitters']): Scene {
  return { primitives: [], emitters, environment: { kind: 'none' } };
}

function material(overrides: Partial<MaterialSpec>): MaterialSpec {
  const resolved: MaterialSpec = {
    baseColor: [0.5, 0.5, 0.5],
    roughness: 1,
    metallic: 0,
    ...overrides,
  };
  // Exact-cell fixtures must opt out of the public trilinear mip default.
  return resolved.emissiveMap != null && resolved.emissiveMap.mipFilter === undefined
    ? { ...resolved, emissiveMap: { ...resolved.emissiveMap, mipFilter: 'none' } }
    : resolved;
}

function luminance(rgb: readonly [number, number, number]): number {
  return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
}

function packedTargetFace(data: Float32Array, lightIndex: number): number {
  const base = lightIndex * TRI_LIGHT_PIXELS * 4;
  return data[base + 20]! + data[base + 23]! * 0x1_0000;
}

function srgbToLinearForTest(value: number): number {
  return value <= 0.04045
    ? value / 12.92
    : ((value + 0.055) / 1.055) ** 2.4;
}

function panelPrimitive(mat: MaterialSpec, castShadow = true): MeshPrimitive {
  return {
    kind: 'mesh',
    id: 'panel',
    positions: new Float32Array([
      0, 0, 0,
      1, 0, 0,
      1, 0, 1,
      0, 0, 1,
    ]),
    normals: new Float32Array([
      0, 1, 0,
      0, 1, 0,
      0, 1, 0,
      0, 1, 0,
    ]),
    uvs: new Float32Array(8),
    indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    material: mat,
    castShadow,
  };
}

function sceneWithPrimitive(primitive: MeshPrimitive, emitters: Scene['emitters'] = []): Scene {
  return { primitives: [primitive], emitters, environment: { kind: 'none' } };
}

describe('packMeshAreaLights (B4)', () => {
  it('treats omitted mipFilter exactly like linear and explicit none as exact-cell eligible', () => {
    const mapped = (mipFilter?: 'linear' | 'none'): MaterialSpec => ({
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      emissive: [1, 1, 1],
      emissiveMap: {
        handle: {},
        ...(mipFilter !== undefined ? { mipFilter } : {}),
      },
    });
    expect(materialEmissionExcludedFromMeshNee(mapped())).toBe(true);
    expect(materialEmissionExcludedFromMeshNee(mapped('linear'))).toBe(true);
    expect(materialEmissionExcludedFromMeshNee(mapped('none'))).toBe(false);
  });

  it('returns null/empty when there are no mesh-area emitters', () => {
    const out = packMeshAreaLights(sceneWith([]), fakeMerged());
    expect(out.data).toBeNull();
    expect(out.triLightCount).toBe(0);
    expect(out.totalEmissiveArea).toBe(0);
  });

  it('extracts every triangle of the referenced mesh with correct area + radiance', () => {
    const out = packMeshAreaLights(
      sceneWith([{ kind: 'mesh-area', id: 'm', meshId: 'panel', color: [0.2, 0.4, 0.8], intensity: 10 }]),
      fakeMerged(),
    );
    expect(out.triLightCount).toBe(2);
    // Two right triangles of a unit quad → total area 1.
    expect(out.totalEmissiveArea).toBeCloseTo(1, 6);
    expect(out.data).not.toBeNull();
    const d = out.data!;
    // First triangle: s0 = (v0, type), s1 = (radiance, 0), s3.a = triArea.
    expect(d[3]).toBe(TRI_AREA_LIGHT_TYPE); // type id
    expect(d[4]).toBeCloseTo(0.2 * 10, 5); // radiance.r = color*intensity
    expect(d[5]).toBeCloseTo(0.4 * 10, 5);
    expect(d[6]).toBeCloseTo(0.8 * 10, 5);
    expect(d[15]).toBeCloseTo(0.5, 6); // each tri area = 0.5
    expect(d[16]).toBeCloseTo(luminance([2, 4, 8]) * 0.5, 6);
    expect(out.totalEmissivePower).toBeCloseTo(luminance([2, 4, 8]), 6);
  });

  it('represents cumulative mesh-emitter mass beyond float32 without dropping support', () => {
    const repeatedIndices = new Uint32Array([
      0, 1, 2,
      0, 2, 3,
      0, 1, 2,
      0, 2, 3,
    ]);
    const merged = fakeMerged({
      indices: repeatedIndices,
      mergedIndices: repeatedIndices,
      triMaterialId: new Uint32Array(4),
      mergedTriMaterialId: new Uint32Array(4),
      bvhTriToMergedTri: new Uint32Array([0, 1, 2, 3]),
      triangleCount: 4,
      meshVertexRanges: [
        {
          name: 'panel',
          vertexStart: 0,
          vertexCount: 4,
          triStart: 0,
          triCount: 4,
        },
      ],
    });
    const packed = packMeshAreaLights(
      sceneWith([{
        kind: 'mesh-area',
        id: 'overflowing-panel',
        meshId: 'panel',
        color: [1, 1, 1],
        intensity: 2e38,
      }]),
      merged,
    );
    expect(packed.totalEmissivePower).toBeGreaterThan(3.402823466e38);
    for (let i = 0; i < packed.triLightCount; i += 1) {
      expect(packed.data![i * 24 + 17]).toBe(0.25);
    }
  });

  it('retains positive mesh-emitter support across adversarial dynamic range', () => {
    const twoSources = fakeMerged({
      meshVertexRanges: [
        { name: 'bright', vertexStart: 0, vertexCount: 3, triStart: 0, triCount: 1 },
        { name: 'dim', vertexStart: 0, vertexCount: 3, triStart: 1, triCount: 1 },
      ],
    });
    const emitter = (id: string, meshId: string, intensity: number) => ({
      kind: 'mesh-area' as const,
      id,
      meshId,
      color: [1, 1, 1] as [number, number, number],
      intensity,
    });
    for (const emitters of [
      [emitter('dominant', 'bright', 1e30), emitter('retained', 'dim', 1e-30)],
      [emitter('retained-first', 'bright', 1e-30), emitter('dominant-last', 'dim', 1e30)],
    ]) {
      const packed = packMeshAreaLights(sceneWith(emitters), twoSources);
      const pmf0 = packed.data![17]!;
      const pmf1 = packed.data![24 + 17]!;
      expect(pmf0).toBeGreaterThan(0);
      expect(pmf1).toBeGreaterThan(0);
      expect(Math.fround(pmf0 + pmf1)).toBe(1);
    }
  });

  it('skips meshes with no matching emitter (only emissive meshes contribute)', () => {
    const merged = fakeMerged({
      meshVertexRanges: [
        { name: 'panel', vertexStart: 0, vertexCount: 4, triStart: 0, triCount: 1 },
        { name: 'wall', vertexStart: 0, vertexCount: 4, triStart: 1, triCount: 1 },
      ],
    });
    const out = packMeshAreaLights(
      sceneWith([{ kind: 'mesh-area', id: 'm', meshId: 'panel', color: [1, 1, 1], intensity: 1 }]),
      merged,
    );
    expect(out.triLightCount).toBe(1); // only 'panel' emits
  });

  it('uses the 6-texel stride per triangle light', () => {
    const out = packMeshAreaLights(
      sceneWith([{ kind: 'mesh-area', id: 'm', meshId: 'panel', color: [1, 1, 1], intensity: 1 }]),
      fakeMerged(),
    );
    expect(TRI_LIGHT_PIXELS).toBe(6);
    // 2 lights × 6 texels = 12 texels → square dim ≥ ceil(sqrt(12)) = 4.
    expect(out.dim).toBeGreaterThanOrEqual(4);
    expect(out.data!.length).toBe(out.dim * out.dim * 4);
  });

  it('packs the inverse-mapped BVH target face losslessly as low/high 16-bit words', () => {
    const scene = sceneWith([
      { kind: 'mesh-area', id: 'm', meshId: 'panel', color: [1, 1, 1], intensity: 1 },
    ]);
    const permuted = packMeshAreaLights(
      scene,
      fakeMerged({ bvhTriToMergedTri: new Uint32Array([1, 0]) }),
    );
    expect(packedTargetFace(permuted.data!, 0)).toBe(1);
    expect(packedTargetFace(permuted.data!, 1)).toBe(0);

    const triangleCount = 0x1_0001;
    const highWordPermutation = Uint32Array.from(
      { length: triangleCount },
      (_, index) => index,
    );
    highWordPermutation[0] = 0x1_0000;
    highWordPermutation[0x1_0000] = 0;
    const highWord = packMeshAreaLights(
      scene,
      fakeMerged({
        triangleCount,
        bvhTriToMergedTri: highWordPermutation,
        meshVertexRanges: [
          { name: 'panel', vertexStart: 0, vertexCount: 4, triStart: 0, triCount: 1 },
        ],
      }),
    );
    expect(highWord.data![20]).toBe(0);
    expect(highWord.data![23]).toBe(1);
    expect(packedTargetFace(highWord.data!, 0)).toBe(0x1_0000);
  });

  it('rejects incomplete, out-of-range, and non-bijective BVH face mappings', () => {
    const scene = sceneWith([
      { kind: 'mesh-area', id: 'm', meshId: 'panel', color: [1, 1, 1], intensity: 1 },
    ]);
    expect(() =>
      packMeshAreaLights(
        scene,
        fakeMerged({ bvhTriToMergedTri: new Uint32Array([0]) }),
      ),
    ).toThrow(/mapping is incomplete/);
    expect(() =>
      packMeshAreaLights(
        scene,
        fakeMerged({ bvhTriToMergedTri: new Uint32Array([0, 2]) }),
      ),
    ).toThrow(/out-of-range face/);
    expect(() =>
      packMeshAreaLights(
        scene,
        fakeMerged({ bvhTriToMergedTri: new Uint32Array([0, 0]) }),
      ),
    ).toThrow(/mapping is non-bijective/);
  });

  it('packs mesh-area emitter castShadow:false into the shared s5.g shadow-disable lane', () => {
    const out = packMeshAreaLights(
      sceneWith([
        {
          kind: 'mesh-area',
          id: 'm',
          meshId: 'panel',
          color: [1, 1, 1],
          intensity: 1,
          castShadow: false,
        },
      ]),
      fakeMerged(),
    );
    expect(out.triLightCount).toBe(2);
    const d = out.data!;
    expect(d[21]).toBe(1);
    expect(d[TRI_LIGHT_PIXELS * 4 + 21]).toBe(1);

    const defaultOut = packMeshAreaLights(
      sceneWith([{ kind: 'mesh-area', id: 'm', meshId: 'panel', color: [1, 1, 1], intensity: 1 }]),
      fakeMerged(),
    );
    expect(defaultOut.data![21]).toBe(0);
    expect(defaultOut.data![TRI_LIGHT_PIXELS * 4 + 21]).toBe(0);
  });

  it('packs the backing material doubleSided contract into s5.b for every triangle', () => {
    const oneSided = packMeshAreaLights(
      sceneWithPrimitive(panelPrimitive(material({ emissive: [1, 1, 1] }))),
      fakeMerged(),
    );
    const twoSided = packMeshAreaLights(
      sceneWithPrimitive(panelPrimitive(material({
        emissive: [1, 1, 1],
        doubleSided: true,
      }))),
      fakeMerged(),
    );

    expect(oneSided.data![22]).toBe(0);
    expect(twoSided.data![22]).toBe(1);
    expect(twoSided.data![TRI_LIGHT_PIXELS * 4 + 22]).toBe(1);
  });

  it('synthesizes triangle-light NEE for emissive mesh materials without explicit emitters', () => {
    const out = packMeshAreaLights(
      sceneWithPrimitive(panelPrimitive(material({ emissive: [0.25, 0.5, 1], emissiveIntensity: 4 }))),
      fakeMerged(),
    );

    expect(out.triLightCount).toBe(2);
    expect(out.totalEmissiveArea).toBeCloseTo(1, 6);
    expect(out.data![4]).toBeCloseTo(1, 6);
    expect(out.data![5]).toBeCloseTo(2, 6);
    expect(out.data![6]).toBeCloseTo(4, 6);
    expect(hasMeshAreaLightForPrimitive(
      sceneWithPrimitive(panelPrimitive(material({ emissive: [0.25, 0.5, 1], emissiveIntensity: 4 }))),
      'panel',
    )).toBe(true);
  });

  it('honors skipEmitter before implicit emissive-map CPU-readability checks', () => {
    const skipped = sceneWithPrimitive(panelPrimitive(material({
      emissive: [1, 1, 1],
      emissiveMap: { handle: {} },
      extensions: { skipEmitter: true },
    })));

    expect(() => packMeshAreaLights(skipped, fakeMerged())).not.toThrow();
    const out = packMeshAreaLights(skipped, fakeMerged());
    expect(out.triLightCount).toBe(0);
    expect(out.data).toBeNull();
    expect(hasMeshAreaLightForPrimitive(skipped, 'panel')).toBe(false);
  });

  it('keeps an explicit mesh-area emitter authoritative over material skipEmitter', () => {
    const explicit = sceneWithPrimitive(
      panelPrimitive(material({
        emissive: [1, 1, 1],
        extensions: { skipEmitter: true },
      })),
      [{
        kind: 'mesh-area',
        id: 'explicit-panel',
        meshId: 'panel',
        color: [0.25, 0.5, 1],
        intensity: 4,
      }],
    );

    const out = packMeshAreaLights(explicit, fakeMerged());
    expect(out.triLightCount).toBe(2);
    expect(out.data?.[4]).toBeCloseTo(1, 6);
    expect(out.data?.[5]).toBeCloseTo(2, 6);
    expect(out.data?.[6]).toBeCloseTo(4, 6);
    expect(hasMeshAreaLightForPrimitive(explicit, 'panel')).toBe(true);
  });

  it('keeps a black explicit mesh-area emitter authoritative over stale implicit emission', () => {
    const explicit = sceneWithPrimitive(
      panelPrimitive(material({ emissive: [4, 2, 1], emissiveIntensity: 3 })),
      [{
        kind: 'mesh-area',
        id: 'disabled-panel',
        meshId: 'panel',
        color: [1, 1, 1],
        intensity: 0,
      }],
    );

    const out = packMeshAreaLights(explicit, fakeMerged());
    expect(out.triLightCount).toBe(0);
    expect(out.data).toBeNull();
    expect(hasMeshAreaLightForPrimitive(explicit, 'panel')).toBe(false);
  });

  it('retains positive emitters below the former luminance cutoff', () => {
    const out = packMeshAreaLights(
      sceneWithPrimitive(panelPrimitive(material({ emissive: [5e-7, 5e-7, 5e-7] }))),
      fakeMerged(),
    );
    expect(out.triLightCount).toBe(2);
    expect(out.totalEmissivePower).toBeGreaterThan(0);
    expect(out.data![4]).toBeGreaterThan(0);
  });

  it('subdivides CPU-readable emissiveMap implicit triangle lights with UV-local radiance', () => {
    const emissiveMap = {
      handle: {
        width: 2,
        height: 1,
        data: new Uint8Array([
          255, 0, 0, 255,
          0, 255, 0, 255,
        ]),
      },
    };
    const out = packMeshAreaLights(
      sceneWithPrimitive(panelPrimitive(material({
        emissive: [2, 2, 2],
        emissiveIntensity: 3,
        emissiveMap,
      }))),
      fakeMerged(),
    );

    expect(out.triLightCount).toBe(2);
    expect(out.totalEmissiveArea).toBeCloseTo(1, 6);
    // This fixture's fake merged UVs are all (0,0), so every sub-triangle samples
    // the red texel; power is luminance([6,0,0]) over total area 1.
    expect(out.totalEmissivePower).toBeCloseTo(luminance([6, 0, 0]), 6);
    expect(out.data![4]).toBeCloseTo(6, 6);
    expect(out.data![5]).toBeCloseTo(0, 6);
    expect(out.data![6]).toBeCloseTo(0, 6);
  });

  it('uses the exact level-0 RGBA16F texel for byte-128 forward-hit and NEE radiance', () => {
    const handle = {
      width: 1,
      height: 1,
      data: new Uint8Array([128, 128, 128, 255]),
      __vitrum_hint__: { channels: 4, dataType: 'uint8' } as const,
    };
    const mappedMaterial = material({
      emissive: [1, 1, 1],
      emissiveIntensity: 1,
      emissiveMap: { handle },
    });
    const atlas = packTextureAtlas(
      [mappedMaterial],
      { storageClass: 'hdr' },
    )!;
    const forwardLevel0Texel = float16BitsToFloat32(atlas.data[0]!);
    const out = packMeshAreaLights(
      sceneWithPrimitive(panelPrimitive(mappedMaterial)),
      fakeMerged(),
    );

    expect(forwardLevel0Texel).not.toBe(srgbToLinearForTest(128 / 255));
    expect(out.data![4]).toBe(forwardLevel0Texel);
    expect(out.data![5]).toBe(forwardLevel0Texel);
    expect(out.data![6]).toBe(forwardLevel0Texel);
  });

  it('matches explicit mesh-area forward f32 operand order for tiny emissive × huge intensity', () => {
    const handle = {
      width: 1,
      height: 1,
      data: new Uint8Array([128, 0, 0, 255]),
      __vitrum_hint__: { channels: 4, dataType: 'uint8' } as const,
    };
    const tinyEmissive = 2 ** -149;
    const hugeIntensity = 2 ** 120;
    const primitive = panelPrimitive(material({ emissiveMap: { handle } }));
    const mappedMaterial = material({
      emissive: [tinyEmissive, 0, 0],
      emissiveIntensity: hugeIntensity,
      emissiveMap: { handle },
    });
    const atlas = packTextureAtlas(
      [mappedMaterial],
      { storageClass: 'hdr' },
    )!;
    const mapTexel = float16BitsToFloat32(atlas.data[0]!);
    const expectedScalar = Math.fround(
      Math.fround(hugeIntensity) * Math.fround(tinyEmissive),
    );
    const expectedRadiance = Math.fround(expectedScalar * mapTexel);
    const out = packMeshAreaLights(
      sceneWithPrimitive(primitive, [{
        kind: 'mesh-area',
        id: 'mapped-explicit-f32',
        meshId: 'panel',
        color: [tinyEmissive, 0, 0],
        intensity: hugeIntensity,
      }]),
      fakeMerged(),
    );

    expect(expectedRadiance).toBeGreaterThan(0);
    expect(out.data![4]).toBe(expectedRadiance);
    expect(out.data![5]).toBe(0);
    expect(out.data![6]).toBe(0);
  });

  it('fails both mapped forward-atlas and NEE paths when a level-0 texel underflows f16', () => {
    const handle = {
      width: 1,
      height: 1,
      data: new Float32Array([FLOAT16_HALF_MIN_SUBNORMAL, 0, 0, 1]),
      __vitrum_hint__: {
        channels: 4,
        dataType: 'float32',
        colorSpace: 'linear',
      } as const,
    };
    const mappedMaterial = material({
      emissive: [1, 1, 1],
      emissiveMap: { handle },
    });

    expect(() =>
      packTextureAtlas(
        [mappedMaterial],
        { storageClass: 'hdr' },
      ),
    ).toThrow(/underflows to \+0 in RGBA16F storage/);
    expect(() =>
      packMeshAreaLights(
        sceneWithPrimitive(panelPrimitive(mappedMaterial)),
        fakeMerged(),
      ),
    ).toThrow(/underflows to \+0 in RGBA16F storage/);
  });

  it('fails both mapped forward-atlas and NEE paths on negative outgoing radiance', () => {
    const handle = {
      width: 1,
      height: 1,
      data: new Float32Array([-(2 ** -24), 1, 1, 1]),
      __vitrum_hint__: {
        channels: 4,
        dataType: 'float32',
        colorSpace: 'linear',
      } as const,
    };
    const mappedMaterial = material({
      emissive: [1, 1, 1],
      emissiveMap: { handle },
    });

    expect(() =>
      packTextureAtlas(
        [mappedMaterial],
        { storageClass: 'hdr' },
      ),
    ).toThrow(/outgoing-radiance RGB value .* must be non-negative/);
    expect(() =>
      packMeshAreaLights(
        sceneWithPrimitive(panelPrimitive(mappedMaterial)),
        fakeMerged(),
      ),
    ).toThrow(/must be non-negative outgoing radiance/);
  });

  it.each([
    [
      'implicit emissive component underflow despite a compensating intensity',
      sceneWithPrimitive(panelPrimitive(material({
        emissive: [2 ** -150, 0, 0],
        emissiveIntensity: 2 ** 100,
      }))),
      /emissive\[0\] underflows material RGBA32F storage/,
    ],
    [
      'implicit emissive component overflow',
      sceneWithPrimitive(panelPrimitive(material({
        emissive: [3.5e38, 0, 0],
        emissiveIntensity: 1,
      }))),
      /emissive\[0\] overflows material RGBA32F storage/,
    ],
    [
      'implicit emissive intensity underflow',
      sceneWithPrimitive(panelPrimitive(material({
        emissive: [2 ** 100, 0, 0],
        emissiveIntensity: 2 ** -150,
      }))),
      /emissiveIntensity underflows material RGBA32F storage/,
    ],
    [
      'implicit emissive intensity overflow',
      sceneWithPrimitive(panelPrimitive(material({
        emissive: [1, 0, 0],
        emissiveIntensity: 3.5e38,
      }))),
      /emissiveIntensity overflows material RGBA32F storage/,
    ],
  ])('fails closed on %s', (_label, scene, message) => {
    expect(() => packMeshAreaLights(scene, fakeMerged())).toThrow(message);
  });

  it('applies the same material-slot underflow guard to explicit mesh-area folding', () => {
    const scene = sceneWithPrimitive(
      panelPrimitive(material({})),
      [{
        kind: 'mesh-area',
        id: 'explicit-underflow',
        meshId: 'panel',
        color: [2 ** -150, 0, 0],
        intensity: 2 ** 100,
      }],
    );
    expect(() => packMeshAreaLights(scene, fakeMerged())).toThrow(
      /mesh-area emitter explicit-underflow emissive\[0\] underflows material RGBA32F storage/,
    );
  });

  it('clips CPU-readable emissiveMap UV footprints to exact texel cells', () => {
    const emissiveMap = {
      handle: {
        width: 2,
        height: 1,
        data: new Uint8Array([
          255, 0, 0, 255,
          0, 255, 0, 255,
        ]),
      },
    };
    const oneTri = new Uint32Array([0, 1, 3]);
    const out = packMeshAreaLights(
      sceneWithPrimitive(panelPrimitive(material({
        emissive: [2, 2, 2],
        emissiveMap,
      }))),
      fakeMerged({
        indices: oneTri,
        mergedIndices: oneTri,
        bvhTriToMergedTri: new Uint32Array([0]),
        triMaterialId: new Uint32Array([0]),
        mergedTriMaterialId: new Uint32Array([0]),
        meshVertexRanges: [
          { name: 'panel', vertexStart: 0, vertexCount: 4, triStart: 0, triCount: 1 },
        ],
        triangleCount: 1,
        uvs: new Float32Array([
          0, 0,
          1, 0,
          0, 0,
          0, 1,
        ]),
      }),
    );

    expect(out.triLightCount).toBe(3);
    expect(out.totalEmissiveArea).toBeCloseTo(0.5, 6);
    expect(out.totalEmissivePower).toBeGreaterThan(0);
    expect(out.data![4]).toBeCloseTo(2, 6);
    expect(out.data![5]).toBeCloseTo(0, 6);
    expect(out.data![6]).toBeCloseTo(0, 6);
    const radianceRecords = Array.from({ length: out.triLightCount }, (_, i) => {
      const base = i * TRI_LIGHT_PIXELS * 4;
      return [out.data![base + 4], out.data![base + 5], out.data![base + 6]].join(',');
    });
    expect(radianceRecords).toContain('0,2,0');
    expect(
      Array.from({ length: out.triLightCount }, (_, i) => packedTargetFace(out.data!, i)),
    ).toEqual([0, 0, 0]);
  });

  it('subdivides explicit mesh-area triangle lights through the referenced material emissiveMap', () => {
    const emissiveMap = {
      handle: {
        width: 2,
        height: 1,
        data: new Uint8Array([
          255, 0, 0, 255,
          0, 255, 0, 255,
        ]),
      },
    };
    const scene: Scene = {
      primitives: [panelPrimitive(material({
        emissiveMap,
      }))],
      emitters: [{
        kind: 'mesh-area',
        id: 'mapped-explicit',
        meshId: 'panel',
        color: [2, 2, 2],
        intensity: 1,
      }],
      environment: { kind: 'none' },
    };
    const out = packMeshAreaLights(
      scene,
      fakeMerged({ uvs: new Float32Array([0.75, 0, 0.75, 0, 0.75, 0, 0.75, 0]) }),
    );

    expect(out.triLightCount).toBe(2);
    expect(out.totalEmissiveArea).toBeCloseTo(1, 6);
    expect(out.data![4]).toBeCloseTo(0, 6);
    expect(out.data![5]).toBeCloseTo(2, 6);
    expect(out.data![6]).toBeCloseTo(0, 6);
  });

  it('samples implicit emissive maps from an arbitrary sparse texCoord stream', () => {
    const texCoord = 37;
    const uvSets: Array<Float32Array | undefined> = [];
    uvSets[texCoord] = new Float32Array([
      0.75, 0,
      0.75, 0,
      0.75, 0,
      0.75, 0,
    ]);
    const primitive: MeshPrimitive = {
      ...panelPrimitive(material({
        emissive: [2, 2, 2],
        emissiveMap: {
          texCoord,
          handle: {
            width: 2,
            height: 1,
            data: new Uint8Array([
              255, 0, 0, 255,
              0, 255, 0, 255,
            ]),
          },
        },
      })),
      uvSets,
    };
    const out = packMeshAreaLights(sceneWithPrimitive(primitive), fakeMerged());

    expect(out.triLightCount).toBe(2);
    expect(out.totalEmissiveArea).toBeCloseTo(1, 6);
    expect(out.data![4]).toBeCloseTo(0, 6);
    expect(out.data![5]).toBeCloseTo(2, 6);
    expect(out.data![6]).toBeCloseTo(0, 6);
  });

  it('suppresses implicit mesh-light synthesis when a readable emissiveMap is black', () => {
    const out = packMeshAreaLights(
      sceneWithPrimitive(panelPrimitive(material({
        emissive: [10, 10, 10],
        emissiveIntensity: 5,
        emissiveMap: {
          handle: {
            width: 1,
            height: 1,
            data: new Uint8Array([0, 0, 0, 255]),
          },
        },
      }))),
      fakeMerged(),
    );

    expect(out.data).toBeNull();
    expect(out.triLightCount).toBe(0);
    expect(out.totalEmissiveArea).toBe(0);
    expect(hasMeshAreaLightForPrimitive(
      sceneWithPrimitive(panelPrimitive(material({
        emissive: [10, 10, 10],
        emissiveIntensity: 5,
        emissiveMap: {
          handle: {
            width: 1,
            height: 1,
            data: new Uint8Array([0, 0, 0, 255]),
          },
        },
      }))),
      'panel',
    )).toBe(false);
  });

  it('fails closed for unreadable emissiveMap handles instead of biasing MIS', () => {
    expect(() => packMeshAreaLights(
      sceneWithPrimitive(panelPrimitive(material({
        emissive: [0.5, 0.25, 0.125],
        emissiveIntensity: 8,
        emissiveMap: { handle: { id: 'gpu-only-texture' } },
      }))),
      fakeMerged(),
    )).toThrow(/emissiveMap without complete CPU-readable texels/);
  });

  it.each([
    ['linear minification', { minFilter: 'linear' as const }],
    ['linear magnification', { magFilter: 'linear' as const }],
    ['mip filtering', { mipFilter: 'nearest' as const }],
  ])('keeps %s mapped emission valid through forward hits while excluding it from NEE', (_label, sampler) => {
    const scene = sceneWithPrimitive(panelPrimitive(material({
        emissive: [1, 1, 1],
        emissiveMap: {
          handle: {
            width: 2,
            height: 1,
            data: new Uint8Array([255, 0, 0, 255, 0, 255, 0, 255]),
          },
          ...sampler,
        },
      })));
    expect(() => packMeshAreaLights(scene, fakeMerged())).not.toThrow();
    expect(packMeshAreaLights(scene, fakeMerged())).toMatchObject({
      data: null,
      triLightCount: 0,
      totalEmissivePower: 0,
    });
    expect(hasMeshAreaLightForPrimitive(scene, 'panel')).toBe(false);
  });

  it.each([
    ['linear minification', { minFilter: 'linear' as const }],
    ['linear magnification', { magFilter: 'linear' as const }],
    ['mip filtering', { mipFilter: 'nearest' as const }],
  ])('does not require CPU texels for forward-only %s emission', (_label, sampler) => {
    const scene = sceneWithPrimitive(panelPrimitive(material({
      emissive: [1, 1, 1],
      emissiveMap: {
        handle: { id: 'gpu-only-forward-emission' },
        ...sampler,
      },
    })));
    expect(() => packMeshAreaLights(scene, fakeMerged())).not.toThrow();
    expect(packMeshAreaLights(scene, fakeMerged()).triLightCount).toBe(0);
  });

  it('keeps an explicit filtered mesh-area emitter forward-only as one coherent estimator', () => {
    const scene = sceneWithPrimitive(
      panelPrimitive(material({
        emissiveMap: {
          handle: {
            width: 2,
            height: 1,
            data: new Uint8Array([255, 255, 255, 255, 128, 128, 128, 255]),
          },
          magFilter: 'linear',
        },
      })),
      [{
        kind: 'mesh-area',
        id: 'mapped-panel',
        meshId: 'panel',
        color: [1, 0.5, 0.25],
        intensity: 4,
      }],
    );

    expect(packMeshAreaLights(scene, fakeMerged())).toMatchObject({
      data: null,
      triLightCount: 0,
      totalEmissivePower: 0,
    });
    expect(hasMeshAreaLightForPrimitive(scene, 'panel')).toBe(false);
  });

  it('accepts an exact linear cpuMirror for an otherwise opaque handle', () => {
    const out = packMeshAreaLights(
      sceneWithPrimitive(panelPrimitive(material({
        emissive: [2, 2, 2],
        emissiveIntensity: 1,
        emissiveMap: {
          handle: {
            id: 'opaque-with-mirror',
            cpuMirror: {
              width: 1,
              height: 1,
              channels: 4,
              dataType: 'float32',
              colorSpace: 'linear',
              data: new Float32Array([0.25, 0.5, 1, 1]),
            },
          },
        },
      }))),
      fakeMerged(),
    );

    expect(out.triLightCount).toBe(2);
    expect(out.data![4]).toBeCloseTo(0.5, 6);
    expect(out.data![5]).toBeCloseTo(1, 6);
    expect(out.data![6]).toBeCloseTo(2, 6);
  });
});
