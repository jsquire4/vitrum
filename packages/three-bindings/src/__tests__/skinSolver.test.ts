import { describe, it, expect } from 'vitest';
import type { SkinnedMeshPrimitive } from '@vitrum/core';
import { solveSkin } from '../skinSolver.js';

// Column-major identity matrix.
const IDENT4 = (): Float32Array => new Float32Array([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
]);

/** Build a single-bone test primitive at a given `bonesMatrix` pose with
 *  rest positions/normals supplied verbatim. boneInverse = identity. */
function singleBonePrim(opts: {
  positions: Float32Array;
  normals: Float32Array;
  bonesMatrix: Float32Array;
}): SkinnedMeshPrimitive {
  const vCount = opts.positions.length / 3;
  const skinIndices = new Uint32Array(vCount * 4);     // all bone 0
  const skinWeights = new Float32Array(vCount * 4);
  for (let i = 0; i < vCount; i++) skinWeights[i * 4 + 0] = 1.0;
  return {
    kind: 'skinned-mesh',
    id: 'test',
    positions: opts.positions,
    normals: opts.normals,
    skinIndices,
    skinWeights,
    bones: opts.bonesMatrix,
    boneInverses: IDENT4(),
    material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0 },
  };
}

describe('solveSkin', () => {
  it('identity pose = rest pose (positions + normals unchanged)', () => {
    const prim = singleBonePrim({
      positions: new Float32Array([1, 2, 3, -4, 5, -6]),
      normals: new Float32Array([1, 0, 0, 0, 1, 0]),
      bonesMatrix: IDENT4(),
    });
    const { positions, normals } = solveSkin(prim);
    expect(Array.from(positions)).toEqual([1, 2, 3, -4, 5, -6]);
    // Normals are normalized; rest normals already unit-length so unchanged.
    expect(positions).toBeInstanceOf(Float32Array);
    expect(normals[0]).toBeCloseTo(1);
    expect(normals[1]).toBeCloseTo(0);
    expect(normals[3]).toBeCloseTo(0);
    expect(normals[4]).toBeCloseTo(1);
  });

  it('translation bone: positions translate, normals stay', () => {
    // Column-major translate(10, -2, 5).
    const trans = new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      10, -2, 5, 1,
    ]);
    const prim = singleBonePrim({
      positions: new Float32Array([0, 0, 0, 1, 1, 1]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1]),
      bonesMatrix: trans,
    });
    const { positions, normals } = solveSkin(prim);
    expect(positions[0]).toBeCloseTo(10);
    expect(positions[1]).toBeCloseTo(-2);
    expect(positions[2]).toBeCloseTo(5);
    expect(positions[3]).toBeCloseTo(11);
    expect(positions[4]).toBeCloseTo(-1);
    expect(positions[5]).toBeCloseTo(6);
    // Normal transform ignores translation column.
    expect(normals[2]).toBeCloseTo(1);
    expect(normals[5]).toBeCloseTo(1);
  });

  it('90-deg rotation about Z: (1,0,0) → (0,1,0)', () => {
    // Column-major rotZ(90°):
    //   [  0  1  0  0 ]
    //   [ -1  0  0  0 ]
    //   [  0  0  1  0 ]
    //   [  0  0  0  1 ]  (rotates +X → -Y? no — let's be explicit)
    //
    // Standard rotation about Z by θ:  x' = cosθ x - sinθ y,  y' = sinθ x + cosθ y
    // For θ = +90°: x' = -y, y' = x. So (1,0,0) → (0,1,0). ✓
    // Column-major (m[r + c*4]): col 0 = first column of the rotation.
    const rotZ = new Float32Array([
      0,  1, 0, 0,   // column 0  → (m00=0, m10=1, m20=0)
     -1,  0, 0, 0,   // column 1
      0,  0, 1, 0,
      0,  0, 0, 1,
    ]);
    const prim = singleBonePrim({
      positions: new Float32Array([1, 0, 0]),
      normals: new Float32Array([1, 0, 0]),
      bonesMatrix: rotZ,
    });
    const { positions, normals } = solveSkin(prim);
    expect(positions[0]).toBeCloseTo(0);
    expect(positions[1]).toBeCloseTo(1);
    expect(positions[2]).toBeCloseTo(0);
    expect(normals[0]).toBeCloseTo(0);
    expect(normals[1]).toBeCloseTo(1);
    expect(normals[2]).toBeCloseTo(0);
  });

  it('two-bone weighted blend: 50/50 averages the two bone transforms', () => {
    // Bone 0 = identity, Bone 1 = translate(10, 0, 0).
    const bones = new Float32Array([
      ...IDENT4(),
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      10, 0, 0, 1,
    ]);
    const boneInverses = new Float32Array([...IDENT4(), ...IDENT4()]);
    const prim: SkinnedMeshPrimitive = {
      kind: 'skinned-mesh',
      id: 'twobone',
      positions: new Float32Array([0, 0, 0]),
      normals: new Float32Array([0, 1, 0]),
      skinIndices: new Uint32Array([0, 1, 0, 0]),   // bones 0 and 1 used
      skinWeights: new Float32Array([0.5, 0.5, 0, 0]),
      bones,
      boneInverses,
      material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0 },
    };
    const { positions } = solveSkin(prim);
    // 0.5 * (0,0,0) + 0.5 * (10,0,0) = (5,0,0)
    expect(positions[0]).toBeCloseTo(5);
    expect(positions[1]).toBeCloseTo(0);
    expect(positions[2]).toBeCloseTo(0);
  });

  it('writes in-place into caller-provided buffers when supplied', () => {
    const prim = singleBonePrim({
      positions: new Float32Array([7, 8, 9]),
      normals: new Float32Array([0, 1, 0]),
      bonesMatrix: IDENT4(),
    });
    const outP = new Float32Array(3);
    const outN = new Float32Array(3);
    const r = solveSkin(prim, outP, outN);
    expect(r.positions).toBe(outP);
    expect(r.normals).toBe(outN);
    expect(Array.from(outP)).toEqual([7, 8, 9]);
  });

  it('morph target with weight 0 is a no-op (identity skinning result)', () => {
    const prim: SkinnedMeshPrimitive = {
      ...singleBonePrim({
        positions: new Float32Array([0, 0, 0]),
        normals: new Float32Array([0, 1, 0]),
        bonesMatrix: IDENT4(),
      }),
      morphTargets: [new Float32Array([100, 100, 100])],
      morphWeights: new Float32Array([0]),
    };
    const { positions } = solveSkin(prim);
    expect(Array.from(positions)).toEqual([0, 0, 0]);
  });

  it('morph target with weight 1 fully applies the delta before skinning', () => {
    const prim: SkinnedMeshPrimitive = {
      ...singleBonePrim({
        positions: new Float32Array([0, 0, 0]),
        normals: new Float32Array([0, 1, 0]),
        bonesMatrix: IDENT4(),
      }),
      morphTargets: [new Float32Array([5, -3, 7])],
      morphWeights: new Float32Array([1]),
    };
    const { positions } = solveSkin(prim);
    expect(positions[0]).toBeCloseTo(5);
    expect(positions[1]).toBeCloseTo(-3);
    expect(positions[2]).toBeCloseTo(7);
  });

  it('two morph targets blend linearly by weight', () => {
    const prim: SkinnedMeshPrimitive = {
      ...singleBonePrim({
        positions: new Float32Array([0, 0, 0]),
        normals: new Float32Array([0, 1, 0]),
        bonesMatrix: IDENT4(),
      }),
      morphTargets: [
        new Float32Array([10, 0, 0]),       // +X target
        new Float32Array([0, 20, 0]),       // +Y target
      ],
      morphWeights: new Float32Array([0.5, 0.25]),
    };
    const { positions } = solveSkin(prim);
    expect(positions[0]).toBeCloseTo(5);    // 0.5 * 10
    expect(positions[1]).toBeCloseTo(5);    // 0.25 * 20
    expect(positions[2]).toBeCloseTo(0);
  });

  it('morphs are applied BEFORE bone transform', () => {
    // Bone = translate(0, 100, 0).  Morph = +1 along X.
    // Expected: morphedRest = (1, 0, 0) → bone applies → (1, 100, 0).
    const trans = new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 100, 0, 1,
    ]);
    const prim: SkinnedMeshPrimitive = {
      ...singleBonePrim({
        positions: new Float32Array([0, 0, 0]),
        normals: new Float32Array([0, 1, 0]),
        bonesMatrix: trans,
      }),
      morphTargets: [new Float32Array([1, 0, 0])],
      morphWeights: new Float32Array([1]),
    };
    const { positions } = solveSkin(prim);
    expect(positions[0]).toBeCloseTo(1);
    expect(positions[1]).toBeCloseTo(100);
    expect(positions[2]).toBeCloseTo(0);
  });

  it('bindMatrix is honored — non-identity bind round-trips to identity skinning', () => {
    // bindMatrix = translate(5, 0, 0); bindMatrixInverse = translate(-5, 0, 0).
    // bones[0].matrixWorld = bindMatrix; boneInverse[0] = bindMatrixInverse.
    // For a vertex at rest position (0,0,0):
    //   skinVertex   = bindMatrix · (0,0,0) = (5, 0, 0)
    //   skinned      = (bones · boneInv) · (5,0,0) = identity · (5,0,0) = (5, 0, 0)
    //   bindMatrixInverse · skinned = (5,0,0) + (-5,0,0,1)_col = (0, 0, 0)
    // → solver should yield rest pose unchanged (the canonical bind-pose
    //   identity property).
    const trans = (tx: number): Float32Array => new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      tx, 0, 0, 1,
    ]);
    const prim: SkinnedMeshPrimitive = {
      kind: 'skinned-mesh',
      id: 'bind-test',
      positions: new Float32Array([0, 0, 0]),
      normals: new Float32Array([0, 1, 0]),
      skinIndices: new Uint32Array([0, 0, 0, 0]),
      skinWeights: new Float32Array([1, 0, 0, 0]),
      bones: trans(5),
      boneInverses: trans(-5),
      bindMatrix: trans(5),
      bindMatrixInverse: trans(-5),
      material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0 },
    };
    const { positions } = solveSkin(prim);
    expect(positions[0]).toBeCloseTo(0);
    expect(positions[1]).toBeCloseTo(0);
    expect(positions[2]).toBeCloseTo(0);
  });

  it('throws when bindMatrix has wrong length', () => {
    const prim: SkinnedMeshPrimitive = {
      ...singleBonePrim({
        positions: new Float32Array([0, 0, 0]),
        normals: new Float32Array([0, 1, 0]),
        bonesMatrix: IDENT4(),
      }),
      bindMatrix: new Float32Array(15),
      bindMatrixInverse: IDENT4(),
    };
    expect(() => solveSkin(prim)).toThrow(/bindMatrix/);
  });

  it('throws when input buffer lengths are inconsistent', () => {
    const bad: SkinnedMeshPrimitive = {
      kind: 'skinned-mesh',
      id: 'bad',
      positions: new Float32Array(6),                   // 2 verts
      normals: new Float32Array(3),                     // wrong: 1 vert
      skinIndices: new Uint32Array(8),
      skinWeights: new Float32Array(8),
      bones: IDENT4(),
      boneInverses: IDENT4(),
      material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0 },
    };
    expect(() => solveSkin(bad)).toThrow(/normals length/);
  });
});
