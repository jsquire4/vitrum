import { describe, expect, it } from 'vitest';
import {
  asMat4,
  type AnalyticPrimitive,
  type MaterialSpec,
  type MaterialSpecPatch,
  type MeshPrimitive,
  type Scene,
} from '@vitrum/core';
import {
  materialDefinesBulkOpticalMedium,
  mergeWorldSpaceFromCore,
  OpticalMediumTopologyError,
} from '@vitrum/shared-bvh';
import { createPTEngine_WebGL2 } from '../index.js';
import { createMockGl } from '../__tests__/mockGl.js';
import { packBvhTextureData } from './bvhTextureAdapter.js';
import { expandAnalyticPrimitiveFallbacks } from './uploadSceneTextures.js';
import {
  assertWebGl2OpticalMediumTopology,
  buildWebGl2OpticalComponentIds,
  buildWebGl2RepresentedPrimitiveInstanceIds,
} from './opticalMediumPolicy.js';

const CLEAR_SHEET: MaterialSpec = {
  baseColor: [1, 1, 1],
  roughness: 0,
  metallic: 0,
  transmission: 1,
  thickness: 0,
};

const BULK_GLASS: MaterialSpec = {
  ...CLEAR_SHEET,
  thickness: 1,
};

const OUTWARD_CUBE_INDICES = new Uint32Array([
  0, 2, 1, 0, 3, 2,
  4, 5, 6, 4, 6, 7,
  0, 1, 5, 0, 5, 4,
  3, 7, 6, 3, 6, 2,
  0, 4, 7, 0, 7, 3,
  1, 2, 6, 1, 6, 5,
]);

function cubePrimitive(
  id: string,
  halfExtent: number,
  material: MaterialSpec,
): MeshPrimitive {
  const h = halfExtent;
  const positions = new Float32Array([
    -h, -h, -h,
    h, -h, -h,
    h, h, -h,
    -h, h, -h,
    -h, -h, h,
    h, -h, h,
    h, h, h,
    -h, h, h,
  ]);
  const normals = new Float32Array(positions.length);
  for (let vertex = 0; vertex < positions.length / 3; vertex += 1) {
    const x = positions[vertex * 3]!;
    const y = positions[vertex * 3 + 1]!;
    const z = positions[vertex * 3 + 2]!;
    const inverseLength = 1 / Math.hypot(x, y, z);
    normals[vertex * 3] = x * inverseLength;
    normals[vertex * 3 + 1] = y * inverseLength;
    normals[vertex * 3 + 2] = z * inverseLength;
  }
  return {
    kind: 'mesh',
    id,
    positions,
    normals,
    indices: OUTWARD_CUBE_INDICES,
    material,
  };
}

function nestedScene(materials: readonly MaterialSpec[]): Scene {
  return {
    primitives: materials.map((material, index) =>
      cubePrimitive(`medium-${index}`, materials.length - index, material)),
    emitters: [],
    environment: { kind: 'none' },
  };
}

function translation(
  x: number,
  y = 0,
  z = 0,
): ReturnType<typeof asMat4> {
  return asMat4(new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    x, y, z, 1,
  ]));
}

function disconnectedCubePrimitive(id: string): MeshPrimitive {
  const first = cubePrimitive(id, 1, BULK_GLASS);
  const positions = new Float32Array(first.positions.length * 2);
  positions.set(first.positions, 0);
  positions.set(first.positions, first.positions.length);
  for (let vertex = 8; vertex < 16; vertex += 1) {
    positions[vertex * 3] = positions[vertex * 3]! + 5;
  }
  const normals = new Float32Array(first.normals.length * 2);
  normals.set(first.normals, 0);
  normals.set(first.normals, first.normals.length);
  const indices = new Uint32Array(OUTWARD_CUBE_INDICES.length * 2);
  indices.set(OUTWARD_CUBE_INDICES, 0);
  for (let index = 0; index < OUTWARD_CUBE_INDICES.length; index += 1) {
    indices[OUTWARD_CUBE_INDICES.length + index] =
      OUTWARD_CUBE_INDICES[index]! + 8;
  }
  return { ...first, positions, normals, indices };
}

function panelWithDuplicateSeam(id: string): MeshPrimitive {
  return {
    kind: 'mesh',
    id,
    positions: new Float32Array([
      0, 0, 0, 1, 0, 0, 1, 1, 0,
      0, 0, 0, 1, 1, 0, 0, 1, 0,
    ]),
    normals: new Float32Array([
      0, 0, 1, 0, 0, 1, 0, 0, 1,
      0, 0, 1, 0, 0, 1, 0, 0, 1,
    ]),
    indices: new Uint32Array([0, 1, 2, 3, 4, 5]),
    material: CLEAR_SHEET,
  };
}

function analyticSphere(
  id: string,
  material: MaterialSpec,
  fallback = false,
): AnalyticPrimitive {
  const fallbackSource = cubePrimitive(`${id}-fallback-source`, 1, material);
  return {
    kind: 'analytic',
    id,
    shape: 'sphere',
    params: new Float32Array([0, 0, 0, 1]),
    material,
    ...(fallback
      ? {
          fallbackMesh: {
            positions: fallbackSource.positions,
            normals: fallbackSource.normals,
            indices: fallbackSource.indices!,
          },
        }
      : {}),
  };
}

describe('pt-webgl2 optical-medium topology policy', () => {
  it.each([
    ['canonical generated triangles', false],
    ['explicit triangle fallback', true],
  ] as const)('maps %s for bulk and thin transmissive analytics', (_label, fallback) => {
    for (const [kind, material, expectedComponent] of [
      ['bulk', BULK_GLASS, 1],
      ['thin', CLEAR_SHEET, 0],
    ] as const) {
      const original: Scene = {
        primitives: [analyticSphere(`${kind}-${String(fallback)}`, material, fallback)],
        emitters: [],
        environment: { kind: 'none' },
      };
      const analysis = assertWebGl2OpticalMediumTopology(original, `test-${kind}`);
      expect(analysis.componentCount).toBe(expectedComponent);
      if (kind === 'bulk') {
        expect(analysis.components[0]?.representation.kind).toBe('triangle-mesh');
      }

      const represented = expandAnalyticPrimitiveFallbacks(original).scene;
      expect(represented.primitives[0]?.kind).toBe('mesh');
      const merged = mergeWorldSpaceFromCore(represented, { positionStride: 4 });
      expect(merged.triangleCount).toBeGreaterThan(12);
      // Deliberately pass the authored analytic scene: policy must lower it to
      // the same canonical transmissive mesh used by the merged range, even
      // when an unrelated explicit fallback was supplied.
      const componentIds = buildWebGl2OpticalComponentIds(
        original,
        merged,
        `test-${kind}-${String(fallback)}`,
      );
      expect(new Set(componentIds)).toEqual(new Set([expectedComponent]));
      expect(new Set(buildWebGl2RepresentedPrimitiveInstanceIds(
        merged,
        `test-${kind}-${String(fallback)}-range`,
      ))).toEqual(new Set([1]));
    }
  });

  it('proves topology in the exact merged-world transform arithmetic', () => {
    const cancellationMatrix = asMat4(new Float32Array([
      1.2345670461654663, 0, 0, 0,
      -0.9876539707183838, 1, 0, 0,
      0.33333298563957214, 0, 1, 0,
      43731892, 0, 0, 1,
    ]));
    const positions = new Float32Array([
      33856616, -23691874, 19167270,
      33856620, -23691874, 19167270,
      33856620, -23691872, 19167270,
      33856616, -23691872, 19167270,
      33856616, -23691874, 19167272,
      33856620, -23691874, 19167272,
      33856620, -23691872, 19167272,
      33856616, -23691872, 19167272,
    ]);
    const primitive: MeshPrimitive = {
      kind: 'mesh',
      id: 'merged-cancellation-cube',
      positions,
      normals: new Float32Array(positions.length).fill(1),
      indices: OUTWARD_CUBE_INDICES,
      material: BULK_GLASS,
      transform: cancellationMatrix,
    };
    const scene: Scene = {
      primitives: [primitive], emitters: [], environment: { kind: 'none' },
    };

    expect(assertWebGl2OpticalMediumTopology(scene, 'setScene').componentCount).toBe(1);
    const merged = mergeWorldSpaceFromCore(scene, { positionStride: 4 });
    expect(Array.from(merged.positions.slice(0, 3))).toEqual([
      115318608, -23691874, 19167270,
    ]);
    expect(new Set(buildWebGl2OpticalComponentIds(
      scene,
      merged,
      'test-merged-transform-arithmetic',
    ))).toEqual(new Set([1]));
  });

  it.each([
    ['bulk', 1, 1],
    ['thin sheet', 0, 0],
  ] as const)(
    'rebuilds an explicit-fallback analytic atomically across %s transmission 0↔1',
    async (_label, thickness, expectedComponent) => {
      const engine = await createPTEngine_WebGL2({ device: createMockGl() });
      try {
        const authored = analyticSphere('mutable-analytic', {
          ...CLEAR_SHEET,
          transmission: 0,
          thickness,
        }, true);
        engine.setScene({
          primitives: [authored], emitters: [], environment: { kind: 'none' },
        });
        expect(engine.getScene!()!.primitives[0]?.kind).toBe('analytic');
        expect(engine._debugGeoPack?.triangleCount).toBe(12);

        const opaquePack = engine._debugGeoPack;
        engine.updatePrimitive!('mutable-analytic', {
          material: { transmission: 1 },
        });
        const transmissivePack = engine._debugGeoPack!;
        expect(transmissivePack).not.toBe(opaquePack);
        expect(transmissivePack.triangleCount).toBeGreaterThan(12);
        expect(new Set(buildWebGl2OpticalComponentIds(
          engine.getScene!()!,
          transmissivePack,
          'test-analytic-transmission-enable',
        ))).toEqual(new Set([expectedComponent]));
        expect(new Set(buildWebGl2RepresentedPrimitiveInstanceIds(
          transmissivePack,
          'test-analytic-transmission-enable-range',
        ))).toEqual(new Set([1]));

        engine.updatePrimitive!('mutable-analytic', {
          material: { transmission: 0 },
        });
        const disabledPack = engine._debugGeoPack!;
        expect(disabledPack).not.toBe(transmissivePack);
        expect(disabledPack.triangleCount).toBe(12);
        expect(new Set(buildWebGl2OpticalComponentIds(
          engine.getScene!()!,
          disabledPack,
          'test-analytic-transmission-disable',
        ))).toEqual(new Set([0]));
      } finally {
        engine.dispose();
      }
    },
  );

  it('projects shared disconnected and instanced component membership into triangle ABI', () => {
    const disconnected: Scene = {
      primitives: [disconnectedCubePrimitive('disconnected')],
      emitters: [],
      environment: { kind: 'none' },
    };
    const base = cubePrimitive('instanced', 1, BULK_GLASS);
    const instanced: Scene = {
      primitives: [{
        ...base,
        kind: 'instanced-mesh',
        instances: [translation(-3), translation(3)],
      }],
      emitters: [],
      environment: { kind: 'none' },
    };

    for (const [label, scene] of [
      ['disconnected', disconnected],
      ['instanced', instanced],
    ] as const) {
      const merged = mergeWorldSpaceFromCore(scene, { positionStride: 4 });
      const componentIds = buildWebGl2OpticalComponentIds(
        scene,
        merged,
        `test-${label}`,
      );
      expect([...new Set(componentIds)].sort(), label).toEqual([1, 2]);
      expect(componentIds.filter((id) => id === 1), label).toHaveLength(12);
      expect(componentIds.filter((id) => id === 2), label).toHaveLength(12);
      const packed = packBvhTextureData(merged, componentIds);
      for (let triangle = 0; triangle < merged.triangleCount; triangle += 1) {
        expect(packed.materialIndex[triangle * 4 + 1], label).toBe(
          componentIds[triangle],
        );
      }
    }
  });

  it('keeps coincident thin-sheet primitive instances distinct from UV-seam identity', () => {
    const seamSheet = panelWithDuplicateSeam('sheet-a');
    const secondSheet = { ...panelWithDuplicateSeam('sheet-b') };
    const scene: Scene = {
      primitives: [seamSheet, secondSheet],
      emitters: [],
      environment: { kind: 'none' },
    };
    const merged = mergeWorldSpaceFromCore(scene, { positionStride: 4 });
    const rangeIds = buildWebGl2RepresentedPrimitiveInstanceIds(
      merged,
      'test-sheet-source-feature',
    );
    const firstRange = merged.meshVertexRanges[0]!;
    const secondRange = merged.meshVertexRanges[1]!;
    const mergedIds = new Uint32Array(merged.triangleCount);
    for (let bvhTriangle = 0; bvhTriangle < merged.triangleCount; bvhTriangle += 1) {
      mergedIds[merged.bvhTriToMergedTri[bvhTriangle]!] = rangeIds[bvhTriangle]!;
    }
    expect(new Set(
      mergedIds.slice(firstRange.triStart, firstRange.triStart + firstRange.triCount),
    )).toEqual(new Set([1]));
    expect(new Set(
      mergedIds.slice(secondRange.triStart, secondRange.triStart + secondRange.triCount),
    )).toEqual(new Set([2]));

    const packed = packBvhTextureData(
      merged,
      new Uint32Array(merged.triangleCount),
      rangeIds,
    );
    for (let triangle = 0; triangle < merged.triangleCount; triangle += 1) {
      expect(packed.materialIndex[triangle * 4 + 2]).toBe(rangeIds[triangle]);
    }
  });

  it('rejects opaqueâ†’thin activation at a coincident range before publishing mutation state', async () => {
    const record = new Map<string, unknown>();
    const engine = await createPTEngine_WebGL2({ device: createMockGl(record) });
    try {
      const opaqueSheet = {
        ...CLEAR_SHEET,
        transmission: 0,
      } satisfies MaterialSpec;
      engine.setScene({
        primitives: [
          { ...panelWithDuplicateSeam('activating-sheet'), material: opaqueSheet },
          panelWithDuplicateSeam('coincident-thin'),
        ],
        emitters: [],
        environment: { kind: 'none' },
      });
      const publishedScene = engine.getScene!();
      const publishedPack = engine._debugGeoPack;
      const allocationCount =
        (record.get('__texImage2D') as unknown[] | undefined)?.length ?? 0;

      expect(() => engine.updatePrimitive!('activating-sheet', {
        material: { transmission: 1 },
      })).toThrow(/distinct transmissive ranges must not share an exact first-hit event/);
      expect(engine.getScene!()).toBe(publishedScene);
      expect(engine._debugGeoPack).toBe(publishedPack);
      expect((record.get('__texImage2D') as unknown[] | undefined)?.length ?? 0)
        .toBe(allocationCount);
    } finally {
      engine.dispose();
    }
  });

  it('rejects a thin-sheet transform into exact contact before publishing geometry state', async () => {
    const record = new Map<string, unknown>();
    const engine = await createPTEngine_WebGL2({ device: createMockGl(record) });
    try {
      engine.setScene({
        primitives: [
          panelWithDuplicateSeam('fixed-sheet'),
          { ...panelWithDuplicateSeam('moving-sheet'), transform: translation(0, 0, 2) },
        ],
        emitters: [],
        environment: { kind: 'none' },
      });
      const publishedScene = engine.getScene!();
      const publishedPack = engine._debugGeoPack;
      const allocationCount =
        (record.get('__texImage2D') as unknown[] | undefined)?.length ?? 0;

      expect(() => engine.updatePrimitive!('moving-sheet', {
        transform: translation(0),
      })).toThrow(/distinct transmissive ranges must not share an exact first-hit event/);
      expect(engine.getScene!()).toBe(publishedScene);
      expect(engine._debugGeoPack).toBe(publishedPack);
      expect((record.get('__texImage2D') as unknown[] | undefined)?.length ?? 0)
        .toBe(allocationCount);
    } finally {
      engine.dispose();
    }
  });

  it('accepts exactly eight nested bulk media and rejects the ninth', () => {
    const eight = nestedScene(Array.from({ length: 8 }, () => BULK_GLASS));
    expect(assertWebGl2OpticalMediumTopology(eight, 'setScene')).toMatchObject({
      componentCount: 8,
      maxNestedMedia: 8,
    });

    const nine = nestedScene(Array.from({ length: 9 }, () => BULK_GLASS));
    expect(() => assertWebGl2OpticalMediumTopology(nine, 'setScene')).toThrowError(
      expect.objectContaining<Partial<OpticalMediumTopologyError>>({
        code: 'capacity-exceeded',
      }),
    );
  });

  it('uses the complete production bulk predicate without classifying clear sheets', () => {
    const variants: ReadonlyArray<readonly [string, MaterialSpec, boolean]> = [
      ['positive thickness', { ...CLEAR_SHEET, thickness: 0.25 }, true],
      ['scalar scattering', { ...CLEAR_SHEET, scatteringCoefficient: 0.1 }, true],
      ['RGB scattering', { ...CLEAR_SHEET, scatteringCoefficientRGB: [0, 0.1, 0] }, true],
      [
        'RGB absorption',
        { ...CLEAR_SHEET, attenuationColor: [0.8, 1, 1], attenuationDistance: 2 },
        true,
      ],
      [
        'spectral absorption',
        {
          ...CLEAR_SHEET,
          spectralAttenuation: {
            wavelengthStart: 400,
            wavelengthEnd: 700,
            values: new Float32Array([0, 0.1, 0]),
          },
        },
        true,
      ],
      ['clear sheet', CLEAR_SHEET, false],
      [
        'white no-op attenuation',
        { ...CLEAR_SHEET, attenuationColor: [1, 1, 1], attenuationDistance: 2 },
        false,
      ],
      [
        'transmission gate',
        { ...CLEAR_SHEET, transmission: 0, thickness: 1, scatteringCoefficient: 0.1 },
        false,
      ],
    ];
    for (const [label, material, expected] of variants) {
      expect(materialDefinesBulkOpticalMedium(material), label).toBe(expected);
    }
  });

  it.each<readonly [string, MaterialSpecPatch]>([
    ['thickness', { thickness: 0.25 }],
    ['scalar scattering', { scatteringCoefficient: 0.1 }],
    ['RGB scattering', { scatteringCoefficientRGB: [0, 0.1, 0] }],
    ['RGB absorption', { attenuationColor: [0.8, 1, 1], attenuationDistance: 2 }],
    [
      'spectral absorption',
      {
        spectralAttenuation: {
          wavelengthStart: 400,
          wavelengthEnd: 700,
          values: new Float32Array([0, 0.1, 0]),
        },
      },
    ],
  ])('rejects a ninth %s medium before publishing mutation or GPU state', async (_label, patch) => {
    const record = new Map<string, unknown>();
    const engine = await createPTEngine_WebGL2({ device: createMockGl(record) });
    try {
      const scene = nestedScene([
        ...Array.from({ length: 8 }, () => BULK_GLASS),
        CLEAR_SHEET,
      ]);
      engine.setScene(scene);
      const publishedScene = engine.getScene!();
      const publishedPack = engine._debugGeoPack;
      const shaderSourceCount = (record.get('__shaderSources') as string[] | undefined)?.length ?? 0;

      expect(() =>
        engine.updatePrimitive!('medium-8', { material: patch }),
      ).toThrow(/requires 9 simultaneously live bulk optical media/);
      expect(engine.getScene!()).toBe(publishedScene);
      expect(engine._debugGeoPack).toBe(publishedPack);
      expect((record.get('__shaderSources') as string[] | undefined)?.length ?? 0).toBe(
        shaderSourceCount,
      );
    } finally {
      engine.dispose();
    }
  });

  it('keeps authored bulk coefficients inert until transmission is enabled', async () => {
    const engine = await createPTEngine_WebGL2({ device: createMockGl() });
    try {
      const inactive = {
        ...CLEAR_SHEET,
        transmission: 0,
        thickness: 1,
        scatteringCoefficient: 0.1,
      } satisfies MaterialSpec;
      engine.setScene(nestedScene([
        ...Array.from({ length: 8 }, () => BULK_GLASS),
        inactive,
      ]));
      const publishedScene = engine.getScene!();
      expect(() =>
        engine.updatePrimitive!('medium-8', { material: { transmission: 1 } }),
      ).toThrow(/requires 9 simultaneously live bulk optical media/);
      expect(engine.getScene!()).toBe(publishedScene);
    } finally {
      engine.dispose();
    }
  });

  it('rejects an over-capacity addPrimitive before publishing scene or BVH state', async () => {
    const engine = await createPTEngine_WebGL2({ device: createMockGl() });
    try {
      engine.setScene(nestedScene(Array.from({ length: 8 }, () => BULK_GLASS)));
      const publishedScene = engine.getScene!();
      const publishedPack = engine._debugGeoPack;
      expect(() =>
        engine.addPrimitive!(cubePrimitive('medium-8', 0.5, BULK_GLASS)),
      ).toThrow(/requires 9 simultaneously live bulk optical media/);
      expect(engine.getScene!()).toBe(publishedScene);
      expect(engine._debugGeoPack).toBe(publishedPack);
    } finally {
      engine.dispose();
    }
  });

  it('rejects malformed and nondeterministic bulk boundaries before allocation', async () => {
    const record = new Map<string, unknown>();
    const engine = await createPTEngine_WebGL2({ device: createMockGl(record) });
    try {
      const complete = cubePrimitive('open', 1, BULK_GLASS);
      const open: MeshPrimitive = {
        ...complete,
        indices: new Uint32Array(OUTWARD_CUBE_INDICES.slice(3)),
      };
      expect(() => engine.setScene({
        primitives: [open], emitters: [], environment: { kind: 'none' },
      })).toThrow(/every edge must be used exactly twice/);
      expect(engine.getScene!()).toBeNull();
      expect(engine._debugGeoPack).toBeNull();

      const blended = cubePrimitive('blend', 1, {
        ...BULK_GLASS,
        alphaMode: 'blend',
        opacity: 0.5,
      });
      expect(() => engine.setScene({
        primitives: [blended], emitters: [], environment: { kind: 'none' },
      })).toThrow(/deterministically solid/);
      expect(engine.getScene!()).toBeNull();
      expect(engine._debugGeoPack).toBeNull();
      expect(record.has('__shaderSources')).toBe(false);
    } finally {
      engine.dispose();
    }
  });
});
