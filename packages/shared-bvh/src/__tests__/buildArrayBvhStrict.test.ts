import { describe, expect, it } from 'vitest';
import { buildArrayBvh } from '../buildArrayBvh.js';

const positions = new Float32Array([
  0, 0, 0, 0,
  1, 0, 0, 0,
  0, 1, 0, 0,
]);
const indices = new Uint32Array([0, 1, 2, 0]);
const materials = new Uint32Array([0]);

describe('buildArrayBvh direct public validation', () => {
  it('uses intrinsic typed-array brands rather than spoofable tags', () => {
    const wrongPositions = new Uint32Array(positions.length);
    Object.defineProperty(wrongPositions, Symbol.toStringTag, { value: 'Float32Array' });
    expect(() => buildArrayBvh(
      wrongPositions as never,
      indices,
      materials,
    )).toThrow(/positions.*exact Float32Array/);

    const wrongIndices = new Float32Array(indices.length);
    Object.defineProperty(wrongIndices, Symbol.toStringTag, { value: 'Uint32Array' });
    expect(() => buildArrayBvh(
      positions,
      wrongIndices as never,
      materials,
    )).toThrow(/indices.*exact Uint32Array/);
  });

  it('rejects misaligned streams and material-count mismatches', () => {
    expect(() => buildArrayBvh(
      positions.subarray(0, positions.length - 1),
      indices,
      materials,
    )).toThrow(/positions\.length.*positionStride/);
    expect(() => buildArrayBvh(
      positions,
      indices.subarray(0, indices.length - 1),
      materials,
    )).toThrow(/indices\.length.*indexStride/);
    expect(() => buildArrayBvh(
      positions,
      indices,
      new Uint32Array(0),
    )).toThrow(/triMaterialIds\.length 0.*triangle count 1/);
    expect(() => buildArrayBvh(
      new Float32Array(0),
      new Uint32Array(0),
      new Uint32Array([0]),
    )).toThrow(/triMaterialIds\.length 1.*triangle count 0/);
  });

  it('rejects unknown and allocation-amplifying options', () => {
    const build = (opts: unknown) => () => buildArrayBvh(
      positions,
      indices,
      materials,
      opts as never,
    );
    expect(build({ futureTypo: true })).toThrow(/unknown option.*futureTypo/);
    expect(build({ positionStride: 2 })).toThrow(/positionStride/);
    expect(build({ indexStride: 5 })).toThrow(/indexStride/);
    expect(build({ maxLeafTriangles: 0 })).toThrow(/maxLeafTriangles/);
    expect(build({ maxLeafTriangles: 0x10000 })).toThrow(/maxLeafTriangles/);
    expect(build({ binCount: 1 })).toThrow(/binCount/);
    expect(build({ binCount: 257 })).toThrow(/binCount/);
    expect(build({ binCount: 2.5 })).toThrow(/binCount/);
  });

  it('rejects non-finite data in every position lane', () => {
    const invalidPadding = positions.slice();
    invalidPadding[3] = Number.NaN;
    expect(() => buildArrayBvh(invalidPadding, indices, materials)).toThrow(
      /positions\[3\].*finite/,
    );
  });
});
