import { describe, expect, it, vi } from 'vitest';
import type { EngineWarning, MaterialSpec, MeshPrimitive, Scene } from '@vitrum/core';
import { buildReSTIRSceneBVHForCoreScene } from '../bvhCore.js';
import type { ReSTIRBvhMode } from '../bvhTypes.js';
import { MATERIAL_MAP_META_TEXEL_OFFSETS } from '../../bvh/materialTextureAtlasPack.js';

function material(baseColor: readonly [number, number, number]): MaterialSpec {
  return { baseColor, roughness: 1, metallic: 0 };
}

function triangle(id: string, xOffset: number, mat: MaterialSpec): MeshPrimitive {
  return {
    kind: 'mesh',
    id,
    positions: new Float32Array([
      xOffset, 0, 0,
      xOffset + 1, 0, 0,
      xOffset, 1, 0,
    ]),
    normals: new Float32Array([
      0, 0, 1,
      0, 0, 1,
      0, 0, 1,
    ]),
    uvs: new Float32Array([0, 0, 1, 0, 0, 1]),
    indices: new Uint32Array([0, 1, 2]),
    material: mat,
  };
}

function emissiveRows(cpuData: ArrayBuffer): [number, number, number, number][] {
  const data = new Float32Array(cpuData);
  const rows: [number, number, number, number][] = [];
  for (let i = 0; i < data.length; i += 4) {
    rows.push([data[i]!, data[i + 1]!, data[i + 2]!, data[i + 3]!]);
  }
  return rows;
}

function expectOnlyMaterialSlotEmissive(
  buffers: ReturnType<typeof buildReSTIRSceneBVHForCoreScene>,
  materialSlot: number,
  expected: readonly [number, number, number],
): void {
  const triMaterialIds = [...new Uint32Array(buffers.triangleMaterialIds.cpuData)];
  const rows = emissiveRows(buffers.bvhEmissiveLe.cpuData);
  let matchedRows = 0;
  for (let i = 0; i < rows.length; i += 1) {
    const row = rows[i]!;
    if (triMaterialIds[i] === materialSlot) {
      matchedRows += 1;
      expect(row[0]).toBeCloseTo(expected[0], 6);
      expect(row[1]).toBeCloseTo(expected[1], 6);
      expect(row[2]).toBeCloseTo(expected[2], 6);
      expect(row[3]).toBe(0);
    } else {
      expect(row).toEqual([0, 0, 0, 0]);
    }
  }
  expect(matchedRows).toBeGreaterThan(0);
}


function withTextureSourcePath<T extends MaterialSpec['displacementMap']>(ref: T, sourcePath: string): T {
  Object.defineProperty(ref as object, Symbol('vitrum.gltf.textureRefSource'), {
    value: { path: sourcePath },
    enumerable: false,
  });
  return ref;
}

function scene(primitives: MeshPrimitive[]): Scene {
  return {
    primitives,
    emitters: [],
    environment: { kind: 'none' },
  };
}

describe('ReSTIR bvhCore material resolver', () => {
  it.each<ReSTIRBvhMode>(['merged', 'tlas'])(
    'packs and importance-samples emissive texCoord 7 in %s mode',
    (bvhMode) => {
      const emissiveMap = {
        handle: {
          width: 2,
          height: 2,
          data: new Uint8Array([
            255, 255, 255, 255, 128, 64, 32, 255,
            32, 64, 128, 255, 255, 255, 255, 255,
          ]),
          __vitrum_hint__: { channels: 4 as const, dataType: 'uint8' as const },
        },
        texCoord: 7,
      };
      const primitive: MeshPrimitive = {
        ...triangle('uv7-emitter', 0, {
          ...material([1, 1, 1]),
          emissive: [1, 1, 1],
          emissiveIntensity: 2,
          emissiveMap,
        }),
        uvs: new Float32Array([0, 0, 1, 0, 0, 1]),
        uvSets: [
          new Float32Array([0, 0, 1, 0, 0, 1]),
          undefined,
          undefined,
          undefined,
          undefined,
          undefined,
          undefined,
          new Float32Array([0, 0, 1, 0, 0, 1]),
        ],
      };

      const buffers = buildReSTIRSceneBVHForCoreScene(scene([primitive]), { bvhMode });
      const emissiveMeta = MATERIAL_MAP_META_TEXEL_OFFSETS.EMISSIVE * 4;
      expect(buffers.materialTextureAtlas.baseColorMetaData[emissiveMeta]).toBe(0);
      expect(buffers.materialTextureAtlas.baseColorMetaData[emissiveMeta + 1]).toBe(2 * 16);
      expect(buffers.materialTextureAtlas.triangleUvs?.uvSets?.get(7)).toBeDefined();
      expect(buffers.emitterCount).toBeGreaterThan(0);
      expect(buffers.totalEmissivePower).toBeGreaterThan(0);
    },
  );

  it.each<ReSTIRBvhMode>(['merged', 'tlas'])(
    'packs sparse UV ids at and above the native array-index ceiling in %s mode',
    (bvhMode) => {
      const nativeCeilingIndex = 0xffff_fffe;
      const ordinaryPropertyIndex = 0x1_0000_0001;
      const highUv = new Float32Array([0, 0, 1, 0, 0, 1]);
      const uvSets: Array<Float32Array | undefined> = [];
      uvSets[nativeCeilingIndex] = highUv;
      uvSets[ordinaryPropertyIndex] = highUv;
      const primitive: MeshPrimitive = {
        ...triangle('array-boundary-uv', 0, {
          ...material([1, 1, 1]),
          baseColorMap: {
            handle: {
              width: 1,
              height: 1,
              data: new Uint8Array([255, 255, 255, 255]),
              __vitrum_hint__: { channels: 4 as const, dataType: 'uint8' as const },
            },
            texCoord: ordinaryPropertyIndex,
          },
        }),
        uvSets,
      };

      const buffers = buildReSTIRSceneBVHForCoreScene(scene([primitive]), { bvhMode });
      expect(
        buffers.materialTextureAtlas.triangleUvs?.uvSets?.get(nativeCeilingIndex),
      ).toBeDefined();
      expect(
        buffers.materialTextureAtlas.triangleUvs?.uvSets?.get(ordinaryPropertyIndex),
      ).toBeDefined();
    },
  );

  it.each<ReSTIRBvhMode>(['merged', 'tlas'])(
    'rejects a high-UV material on a primitive missing that stream in %s mode',
    (bvhMode) => {
      const highMapMaterial: MaterialSpec = {
        ...material([1, 1, 1]),
        baseColorMap: {
          handle: {
            width: 1,
            height: 1,
            data: new Uint8Array([255, 255, 255, 255]),
            __vitrum_hint__: { channels: 4, dataType: 'uint8' },
          },
          texCoord: 7,
        },
      };
      const missing = {
        ...triangle('missing-uv7', 0, highMapMaterial),
        uvs: new Float32Array([0, 0, 1, 0, 0, 1]),
      } satisfies MeshPrimitive;
      const suppliesElsewhere = {
        ...triangle('supplies-uv7', 2, material([1, 1, 1])),
        uvSets: [
          undefined, undefined, undefined, undefined,
          undefined, undefined, undefined,
          new Float32Array([0, 0, 1, 0, 0, 1]),
        ],
      } satisfies MeshPrimitive;

      const expectedError = bvhMode === 'merged'
        ? /references TEXCOORD_7.*does not provide that UV stream/
        : /packMaterialTextureAtlas: triangle 0 material references texCoord 7, but that primitive does not supply the UV stream/;
      expect(() => buildReSTIRSceneBVHForCoreScene(
        scene([missing, suppliesElsewhere]),
        { bvhMode },
      )).toThrow(expectedError);
    },
  );

  it('packs one material slot per unique mesh-like primitive in TLAS mode', () => {
    const buffers = buildReSTIRSceneBVHForCoreScene(scene([
      triangle('red-panel', 0, material([1, 0, 0])),
      triangle('green-panel', 2, material([0, 1, 0])),
    ]), { bvhMode: 'tlas' });

    expect([...new Uint32Array(buffers.triangleMaterialIds.cpuData)]).toEqual([0, 1]);
    expect(buffers.coreMaterials.map((m) => m.baseColor)).toEqual([
      [1, 0, 0],
      [0, 1, 0],
    ]);
  });

  it('rejects duplicate mesh-like primitive ids instead of reusing the first material slot', () => {
    const duplicateScene = scene([
      triangle('panel', 0, material([1, 0, 0])),
      triangle('panel', 2, material([0, 1, 0])),
    ]);

    expect(() => buildReSTIRSceneBVHForCoreScene(duplicateScene, { bvhMode: 'tlas' }))
      .toThrow(/duplicate mesh-like primitive id\(s\): panel/);
  });

  it('packs TLAS bvhIndex.xyz from stride-4 scene indices, not tri*3 offsets', () => {
    const buffers = buildReSTIRSceneBVHForCoreScene(scene([
      triangle('red-panel', 0, material([1, 0, 0])),
      triangle('green-panel', 2, material([0, 1, 0])),
    ]), { bvhMode: 'tlas' });

    const source = buffers.scenePack!.indices;
    const packed = new Uint32Array(buffers.bvhIndex.cpuData);
    expect(buffers.scenePack!.triangleCount).toBe(2);
    expect([...source.slice(0, 8)]).toEqual([0, 1, 2, 0, 3, 4, 5, 0]);

    for (let t = 0; t < buffers.scenePack!.triangleCount; t += 1) {
      expect([...packed.slice(t * 4, t * 4 + 3)]).toEqual([
        source[t * 4]!,
        source[t * 4 + 1]!,
        source[t * 4 + 2]!,
      ]);
    }

    // Regression guard for the old F-TLAS1 class: tri 1 used to read
    // [source[3], source[4], source[5]] = [0, 3, 4].
    expect([...packed.slice(4, 7)]).toEqual([3, 4, 5]);
  });

  it.each<ReSTIRBvhMode>(['merged', 'tlas'])(
    'rejects dangling mesh-area emitter references in %s mode',
    (bvhMode) => {
      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
      const warnings: EngineWarning[] = [];
      const sourceScene: Scene = {
        primitives: [triangle('panel', 0, material([1, 1, 1]))],
        emitters: [{
          kind: 'mesh-area',
          id: 'missing-emitter',
          meshId: 'missing-panel',
          color: [1, 1, 1],
          intensity: 3,
        }],
        environment: { kind: 'none' },
      };

      expect(() => buildReSTIRSceneBVHForCoreScene(sourceScene, {
        bvhMode,
        onWarning: (warning) => warnings.push(warning),
        warningPhase: 'setScene',
        warningMethod: 'setScene',
      })).toThrow(/meshId references missing primitive "missing-panel"/);

      expect(warnSpy).not.toHaveBeenCalled();
      expect(warnings).toEqual([]);
      warnSpy.mockRestore();
    },
  );

  it.each<ReSTIRBvhMode>(['merged', 'tlas'])(
    'rejects unreadable vertex-displacement sources in %s mode',
    (bvhMode) => {
      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
      const warnings: EngineWarning[] = [];
      const sourceScene = scene([
        triangle('panel', 0, {
          ...material([1, 1, 1]),
          displacementMap: { handle: { id: 'height' } },
          displacementScale: 0.25,
          displacementBias: -0.05,
        }),
      ]);

      expect(() => buildReSTIRSceneBVHForCoreScene(sourceScene, {
        bvhMode,
        onWarning: (warning) => warnings.push(warning),
        warningPhase: 'setScene',
        warningMethod: 'setScene',
      })).toThrow(/Primitive "panel" displacementMap handle is not CPU-readable/);
      expect(warnSpy).not.toHaveBeenCalled();
      expect(warnings).toEqual([]);
      warnSpy.mockRestore();
    },
  );



  it.each<ReSTIRBvhMode>(['merged', 'tlas'])(
    'preserves source-path details when rejecting capitalized displacement sources in %s mode',
    (bvhMode) => {
      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
      const warnings: EngineWarning[] = [];
      const sourceScene = scene([{
        ...triangle('panel-high-uv', 0, {
          ...material([1, 1, 1]),
          displacementMap: withTextureSourcePath({
            handle: { id: 'height' },
            texCoord: 2,
          }, 'materials[0].extensions.VITRUM_displacement.displacementTexture'),
          displacementScale: 0.25,
        }),
        uvSets: [
          undefined,
          undefined,
          new Float32Array([0, 0, 1, 0, 0, 1]),
        ],
      }]);

      expect(() => buildReSTIRSceneBVHForCoreScene(sourceScene, {
        bvhMode,
        onWarning: (warning) => warnings.push(warning),
        warningPhase: 'setScene',
        warningMethod: 'setScene',
      })).toThrow(/materials\[0\]\.extensions\.VITRUM_displacement\.displacementTexture handle is not CPU-readable/);
      expect(warnings).toEqual([]);
      expect(warnSpy).not.toHaveBeenCalled();
      warnSpy.mockRestore();
    },
  );

  it.each<ReSTIRBvhMode>(['merged', 'tlas'])(
    'rejects missing-high-UV microdisplacement input in %s mode',
    (bvhMode) => {
      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
      const warnings: EngineWarning[] = [];
      const sourceScene = scene([
        triangle('panel-micro-high-uv', 0, {
          ...material([1, 1, 1]),
          displacementMap: {
            handle: {
              width: 1,
              height: 1,
              data: new Uint8Array([128]),
              __vitrum_hint__: { channels: 1, dataType: 'uint8' },
            },
            texCoord: 2,
          },
          displacementScale: 0.25,
          displacementSubdivisions: 1,
        }),
      ]);

      const expectedError = bvhMode === 'merged'
        ? /displacementMap\.texCoord references TEXCOORD_2.*does not provide that UV stream/
        : /Primitive "panel-micro-high-uv" displacementMap requests TEXCOORD_2, but that exact UV channel is absent/;
      expect(() => buildReSTIRSceneBVHForCoreScene(sourceScene, {
        bvhMode,
        onWarning: (warning) => warnings.push(warning),
        warningPhase: 'setScene',
        warningMethod: 'setScene',
      })).toThrow(expectedError);

      expect(warnings).toEqual([]);
      expect(warnSpy).not.toHaveBeenCalled();
      warnSpy.mockRestore();
    },
  );

  it('routes mesh-area Le overrides by full material signatures when maps differ', () => {
    const handleA = { name: 'alpha-a' };
    const handleB = { name: 'alpha-b' };
    const matA: MaterialSpec = { ...material([1, 1, 1]), alphaMap: { handle: handleA } };
    const matB: MaterialSpec = { ...material([1, 1, 1]), alphaMap: { handle: handleB } };
    const sourceScene: Scene = {
      primitives: [
        triangle('alpha-a-panel', 0, matA),
        triangle('alpha-b-panel', 2, matB),
      ],
      emitters: [{
        kind: 'mesh-area',
        id: 'alpha-b-emitter',
        meshId: 'alpha-b-panel',
        color: [0.25, 0.5, 1],
        intensity: 4,
      }],
      environment: { kind: 'none' },
    };

    const buffers = buildReSTIRSceneBVHForCoreScene(sourceScene, { bvhMode: 'merged' });
    const slotB = buffers.coreMaterials.findIndex((m) => m.alphaMap === matB.alphaMap);

    expect(slotB).toBeGreaterThanOrEqual(0);
    expectOnlyMaterialSlotEmissive(buffers, slotB, [1, 2, 4]);
  });

  it('routes mesh-area Le overrides to the castShadow-split material slot', () => {
    const shared = material([1, 1, 1]);
    const shadowless = {
      ...triangle('shadowless-panel', 2, shared),
      castShadow: false,
    } satisfies MeshPrimitive;
    const sourceScene: Scene = {
      primitives: [
        triangle('caster-panel', 0, shared),
        shadowless,
      ],
      emitters: [{
        kind: 'mesh-area',
        id: 'shadowless-emitter',
        meshId: 'shadowless-panel',
        color: [1, 0.25, 0.5],
        intensity: 3,
      }],
      environment: { kind: 'none' },
    };

    const buffers = buildReSTIRSceneBVHForCoreScene(sourceScene, { bvhMode: 'merged' });
    const shadowlessSlot = buffers.coreMaterials.findIndex((m) =>
      (m as MaterialSpec & { readonly castShadow?: boolean }).castShadow === false,
    );

    expect(shadowlessSlot).toBeGreaterThanOrEqual(0);
    expectOnlyMaterialSlotEmissive(buffers, shadowlessSlot, [3, 0.75, 1.5]);
  });
});
