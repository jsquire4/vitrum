import { describe, expect, it } from 'vitest';
import { asMat4, type MaterialSpec, type Scene } from '@vitrum/core';
import type { PrimitiveTlasBinding } from '@vitrum/shared-bvh';
import {
  assertMneeInterfaceDomainSupported,
  buildMneeFacetCandidateTable,
  materialMayProduceMneeDelta,
  MNEE_FACET_TABLE_MAGIC,
  mneeBoundedU32Index,
  mneeBoundedU32Threshold,
  mneeConditionalFacetPmf,
  mneeScaleAwareEpsilon,
  mneeSolverToleranceOracle,
} from '../scene/mneeFacetCandidates.js';
import { MNEE_BOUNDED_CHAIN_CORE_WGSL } from '../wgsl/pathTrace/mneeNewton.wgsl.js';
import { buildPackedScene } from '../scene/uploadSceneBuffers.js';

const IDENTITY = asMat4([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
]);

const DELTA_MATERIAL: MaterialSpec = {
  baseColor: [1, 1, 1],
  roughness: 0,
  metallic: 0,
  transmission: 1,
  emissive: [0, 0, 0],
  emissiveIntensity: 0,
};

const ROUGH_MATERIAL: MaterialSpec = {
  ...DELTA_MATERIAL,
  roughness: 0.8,
  transmission: 0,
};

function trianglePrimitive(
  id: string,
  material: MaterialSpec = DELTA_MATERIAL,
): Extract<Scene['primitives'][number], { kind: 'mesh' }> {
  return {
    kind: 'mesh',
    id,
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    material,
  };
}

function binding(
  primitiveId: string,
  triStart: number,
  triCount: number,
  instanceCount: number,
): PrimitiveTlasBinding {
  return {
    primitiveId,
    primitiveKind: 'mesh',
    blasRoot: 0,
    instanceSourceIndices: Array.from({ length: instanceCount }, (_, index) => index),
    instanceCount,
    vertexStart: 0,
    vertexCount: triCount * 3,
    triStart,
    triCount,
    localAabbMin: [0, 0, 0],
    localAabbMax: [1, 1, 0],
  };
}

describe('MNEE facet proposal closure', () => {
  it('keeps every tiny positive smooth transmissive interface in the candidate domain', () => {
    expect(materialMayProduceMneeDelta({
      ...DELTA_MATERIAL,
      transmission: 1e-12,
    })).toBe(true);
    expect(materialMayProduceMneeDelta({
      ...DELTA_MATERIAL,
      transmission: 1e-12,
      metallic: 0.5,
    })).toBe(false);
    expect(materialMayProduceMneeDelta({
      ...DELTA_MATERIAL,
      transmission: 1e-12,
      metallic: 0.5,
      metallicMap: { handle: {} },
    })).toBe(true);
    expect(materialMayProduceMneeDelta({
      ...DELTA_MATERIAL,
      transmission: 1e-12,
      metallic: 1,
    })).toBe(false);
  });

  it('pins guided-mixture PMFs for same, different, missing, and N=1 guides', () => {
    expect(mneeConditionalFacetPmf(8, 3, 3)).toBeCloseTo(0.5 + 0.5 / 8, 15);
    expect(mneeConditionalFacetPmf(8, 3, 5)).toBeCloseTo(0.5 / 8, 15);
    expect(mneeConditionalFacetPmf(8, null, 5)).toBeCloseTo(1 / 8, 15);
    expect(mneeConditionalFacetPmf(1, 0, 0)).toBe(1);
    expect(mneeConditionalFacetPmf(1, null, 0)).toBe(1);
  });

  it('maps the full u32 domain without the 24-bit float-index ceiling', () => {
    const aboveFloatMantissa = 2 ** 24 + 1;
    expect(mneeBoundedU32Threshold(aboveFloatMantissa)).toBeGreaterThan(0);
    const wordForLast = aboveFloatMantissa * 2 - 1;
    expect(mneeBoundedU32Index(wordForLast, aboveFloatMantissa))
      .toBe(aboveFloatMantissa - 1);

    const maxU32 = 0xffff_ffff;
    expect(mneeBoundedU32Threshold(maxU32)).toBe(1);
    expect(mneeBoundedU32Index(0, maxU32)).toBeNull();
    expect(mneeBoundedU32Index(maxU32 - 1, maxU32)).toBe(maxU32 - 1);
    expect(mneeBoundedU32Index(maxU32, maxU32)).toBe(0);
  });

  it('packs more than 100k records without an argument-count ceiling', () => {
    const scene: Scene = {
      primitives: [trianglePrimitive('large')],
      emitters: [],
      environment: { kind: 'none' },
    };
    const candidateCount = 100_001;
    const table = buildMneeFacetCandidateTable(
      scene,
      [binding('large', 7, candidateCount, 1)],
      (candidateCount + 1) * 16,
    );
    expect(table.candidateCount).toBe(candidateCount);
    expect(table.records.length).toBe((candidateCount + 1) * 4);
    const words = new Uint32Array(table.records.buffer);
    expect(words[0]).toBe(MNEE_FACET_TABLE_MAGIC);
    expect(words[1]).toBe(candidateCount);
    expect(words[4]).toBe(7);
    expect(words[candidateCount * 4]).toBe(7 + candidateCount - 1);
  });

  it('uses cumulative TLAS insertion order across mixed and zero-count bindings', () => {
    const scene: Scene = {
      primitives: [
        trianglePrimitive('a'),
        trianglePrimitive('rough', ROUGH_MATERIAL),
        trianglePrimitive('zero'),
        trianglePrimitive('d'),
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    const table = buildMneeFacetCandidateTable(
      scene,
      [
        binding('a', 10, 2, 2),
        binding('rough', 12, 1, 1),
        binding('zero', 20, 4, 0),
        binding('d', 30, 1, 2),
      ],
      1 << 20,
    );
    const words = new Uint32Array(table.records.buffer);
    const identities: Array<readonly [number, number]> = [];
    for (let record = 1; record <= table.candidateCount; record += 1) {
      identities.push([words[record * 4]!, words[record * 4 + 1]!]);
    }
    expect(identities).toEqual([
      [10, 0], [11, 0], [10, 1], [11, 1],
      [30, 3], [30, 4],
    ]);
  });

  it('matches actual TLAS instance indices for mixed eligible primitives', () => {
    const base = trianglePrimitive('base');
    const scene: Scene = {
      primitives: [
        {
          ...base,
          kind: 'instanced-mesh',
          id: 'a',
          instances: [IDENTITY, asMat4([...IDENTITY.slice(0, 12), 2, 0, 0, 1])],
        },
        trianglePrimitive('rough', ROUGH_MATERIAL),
        {
          ...base,
          kind: 'instanced-mesh',
          id: 'zero',
          instances: [],
        },
        {
          ...base,
          kind: 'instanced-mesh',
          id: 'd',
          instances: [
            asMat4([...IDENTITY.slice(0, 12), 4, 0, 0, 1]),
            asMat4([...IDENTITY.slice(0, 12), 6, 0, 0, 1]),
          ],
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    const packed = buildPackedScene(scene, {
      includeMneeFacetCandidates: true,
      mneeFacetCandidateStorageLimitBytes: 1 << 20,
    });
    expect(packed.primitiveTlasBindings.map((entry) => [
      entry.primitiveId, entry.instanceCount,
    ])).toEqual([['a', 2], ['rough', 1], ['d', 2]]);
    expect(new Set(packed.tlasInstanceIndices)).toEqual(new Set([0, 1, 2, 3, 4]));

    const baseRecord = packed.analyticCount * 2;
    const words = new Uint32Array(packed.analyticParams.buffer);
    const wordBase = baseRecord * 4;
    expect(words[wordBase]).toBe(MNEE_FACET_TABLE_MAGIC);
    const count = words[wordBase + 1]!;
    const instanceIndices: number[] = [];
    for (let record = 0; record < count; record += 1) {
      instanceIndices.push(words[wordBase + (record + 1) * 4 + 1]!);
    }
    expect(instanceIndices).toEqual([0, 1, 3, 4]);
  });

  it('pins micro-scale and large-coordinate tolerance behavior', () => {
    const micro = mneeScaleAwareEpsilon(
      1e-12, [1e-6, -2e-6, 3e-6], 1e-6,
    );
    expect(micro).toBeGreaterThan(0);
    expect(micro).toBeLessThan(1e-9);

    const large = mneeScaleAwareEpsilon(
      1e-5, [1e9, -1e9, 1e9], 100,
    );
    expect(Number.isFinite(large)).toBe(true);
    expect(large).toBeGreaterThanOrEqual(1e9 * 4 * 2 ** -23);
    const kilometreSpan = mneeScaleAwareEpsilon(1e-5, [0, 0, 0], 1_000);
    expect(kilometreSpan).toBeLessThan(1e-3);
    expect(kilometreSpan).toBeGreaterThanOrEqual(1_000 * 4 * 2 ** -23);
  });

  it('pins scale-aware Newton FD and residual tolerances', () => {
    const micro = mneeSolverToleranceOracle(1e-12, [
      [0, 0, 0],
      [1e-6, 0, 0],
      [0, 1e-6, 0],
    ]);
    expect(micro.fdStep).toBeGreaterThan(micro.lengthFloor);
    expect(micro.fdStep).toBeLessThan(1e-8);
    expect(micro.residualTolerance).toBeLessThan(1e-4);
    expect(micro.representable).toBe(true);

    const translated = mneeSolverToleranceOracle(1e-5, [
      [1e9, 1e9, 1e9],
      [1e9 + 131_072, 1e9, 1e9],
      [1e9, 1e9 + 65_536, 1e9],
    ]);
    expect(translated.fdStep).toBeGreaterThanOrEqual(
      translated.coordinateScale * 4 * 2 ** -23,
    );
    expect(translated.fdStep).toBeLessThan(translated.localSpan);
    expect(translated.residualTolerance).toBeLessThan(0.01);
    expect(translated.representable).toBe(true);

    const collapsedCoordinate = Math.fround(1e9);
    const collapsed = mneeSolverToleranceOracle(1e-5, [
      [collapsedCoordinate, 0, 0],
      [Math.fround(1e9 + 1), 0, 0],
    ]);
    expect(collapsed.localSpan).toBe(0);
    expect(collapsed.representable).toBe(false);

    const absoluteEpsilonDominatesFeature = mneeSolverToleranceOracle(1e-3, [
      [0, 0, 0],
      [1e-5, 0, 0],
    ]);
    expect(absoluteEpsilonDominatesFeature.representable).toBe(false);

    expect(MNEE_BOUNDED_CHAIN_CORE_WGSL).toContain('fn mneeFdStepFromScales(');
    expect(MNEE_BOUNDED_CHAIN_CORE_WGSL).toContain('fn mneeResidualToleranceFromScales(');
    expect(MNEE_BOUNDED_CHAIN_CORE_WGSL).toContain('fn mneeScalesRepresentable(');
    expect(MNEE_BOUNDED_CHAIN_CORE_WGSL).not.toContain('let eps = 1e-3');
    expect(MNEE_BOUNDED_CHAIN_CORE_WGSL).not.toContain('out.residual <= 1e-4');
  });

  it('fails closed for shading-normal fields outside the planar solver domain', () => {
    const mapped: Scene = {
      primitives: [trianglePrimitive('mapped', {
        ...DELTA_MATERIAL,
        normalMap: { handle: {} },
      })],
      emitters: [],
      environment: { kind: 'none' },
    };
    expect(() => assertMneeInterfaceDomainSupported(mapped))
      .toThrow(/normal\/bump\/layer-normal maps/);

    const smooth = trianglePrimitive('smooth');
    const varying: Scene = {
      primitives: [{
        ...smooth,
        normals: new Float32Array([0, 0, 1, 0, 0.5, 0.866, 0, 0, 1]),
      }],
      emitters: [],
      environment: { kind: 'none' },
    };
    expect(() => assertMneeInterfaceDomainSupported(varying))
      .toThrow(/varying vertex normal/);
  });
});
