import { describe, expect, it, vi } from 'vitest';
import type { EngineWarning, MaterialSpec, MeshPrimitive, Scene } from '@vitrum/core';
import { buildReSTIRSceneBVHForCoreScene } from '../bvhCore.js';
import type { ReSTIRBvhMode } from '../bvhTypes.js';

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
    'routes missing mesh-area emitter references through structured warnings in %s mode',
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

      const buffers = buildReSTIRSceneBVHForCoreScene(sourceScene, {
        bvhMode,
        onWarning: (warning) => warnings.push(warning),
        warningPhase: 'setScene',
        warningMethod: 'setScene',
      });

      expect(buffers.emitterCount).toBeGreaterThanOrEqual(1);
      expect(warnSpy).not.toHaveBeenCalled();
      expect(warnings).toEqual([
        expect.objectContaining({
          code: 'walkaround-hybrid.mesh-area-emitter-missing-mesh',
          backend: 'walkaround-hybrid',
          phase: 'setScene',
          method: 'setScene',
          details: {
            emitterId: 'missing-emitter',
            meshId: 'missing-panel',
            source: 'bvh-emissive-override',
            fallback: 'emitter skipped',
          },
        }),
      ]);
      warnSpy.mockRestore();
    },
  );

  it.each<ReSTIRBvhMode>(['merged', 'tlas'])(
    'routes scene-pack vertex-displacement skips through structured warnings in %s mode',
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

      const buffers = buildReSTIRSceneBVHForCoreScene(sourceScene, {
        bvhMode,
        onWarning: (warning) => warnings.push(warning),
        warningPhase: 'setScene',
        warningMethod: 'setScene',
      });

      const scenePackWarnings = buffers.warnings ?? [];
      expect(scenePackWarnings.some((warning) =>
        warning.includes('Primitive "panel" displacementMap') &&
        warning.includes('displacement skipped'),
      )).toBe(true);
      expect(warnSpy).not.toHaveBeenCalled();
      expect(warnings).toContainEqual(expect.objectContaining({
        code: 'walkaround-hybrid.vertex-displacement-skipped',
        backend: 'walkaround-hybrid',
        phase: 'setScene',
        method: 'setScene',
        details: expect.objectContaining({
          source: 'shared-bvh',
          fallback: 'displacement skipped',
          warning: expect.stringContaining('Primitive "panel" displacementMap'),
        }),
      }));
      warnSpy.mockRestore();
    },
  );



  it.each<ReSTIRBvhMode>(['merged', 'tlas'])(
    'preserves source-path details for capitalized displacement skips in %s mode',
    (bvhMode) => {
      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
      const warnings: EngineWarning[] = [];
      const sourceScene = scene([
        triangle('panel-high-uv', 0, {
          ...material([1, 1, 1]),
          displacementMap: withTextureSourcePath({
            handle: { id: 'height' },
            texCoord: 2,
          }, 'materials[0].extensions.VITRUM_displacement.displacementTexture'),
          displacementScale: 0.25,
        }),
      ]);

      buildReSTIRSceneBVHForCoreScene(sourceScene, {
        bvhMode,
        onWarning: (warning) => warnings.push(warning),
        warningPhase: 'setScene',
        warningMethod: 'setScene',
      });

      expect(warnings).toContainEqual(expect.objectContaining({
        code: 'walkaround-hybrid.vertex-displacement-skipped',
        details: expect.objectContaining({
          source: 'shared-bvh',
          fallback: 'displacement skipped',
          sourcePath: 'materials[0].extensions.VITRUM_displacement.displacementTexture',
          warning: expect.stringContaining('requests TEXCOORD_2'),
        }),
      }));
      expect(warnSpy).not.toHaveBeenCalled();
      warnSpy.mockRestore();
    },
  );

  it.each<ReSTIRBvhMode>(['merged', 'tlas'])(
    'classifies microdisplacement fallback warnings in %s mode',
    (bvhMode) => {
      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
      const warnings: EngineWarning[] = [];
      const sourceScene = scene([
        triangle('panel-micro-high-uv', 0, {
          ...material([1, 1, 1]),
          displacementMap: { handle: { id: 'height' }, texCoord: 2 },
          displacementScale: 0.25,
          displacementSubdivisions: 1,
        }),
      ]);

      buildReSTIRSceneBVHForCoreScene(sourceScene, {
        bvhMode,
        onWarning: (warning) => warnings.push(warning),
        warningPhase: 'setScene',
        warningMethod: 'setScene',
      });

      expect(warnings).toContainEqual(expect.objectContaining({
        code: 'walkaround-hybrid.vertex-displacement-skipped',
        details: expect.objectContaining({
          source: 'shared-bvh',
          fallback: 'microdisplacement fallback to vertex displacement',
          warning: expect.stringContaining('Falling back to vertex displacement'),
        }),
      }));
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
