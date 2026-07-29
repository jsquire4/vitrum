import { describe, it, expect } from 'vitest';
import type { SkinnedMeshPrimitive } from '@vitrum/core';
import { mat3InverseTranspose, solveSkin } from '../skinSolver.js';

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

  it('preserves and solves more than four influences per vertex', () => {
    const bones = new Float32Array(6 * 16);
    const inverses = new Float32Array(6 * 16);
    for (let bone = 0; bone < 6; bone += 1) {
      bones.set(IDENT4(), bone * 16);
      inverses.set(IDENT4(), bone * 16);
      bones[bone * 16 + 12] = bone;
    }
    const prim: SkinnedMeshPrimitive = {
      kind: 'skinned-mesh',
      id: 'six-influences',
      positions: new Float32Array([0, 0, 0]),
      normals: new Float32Array([0, 1, 0]),
      skinIndices: new Uint32Array([0, 1, 2, 3, 4, 5]),
      skinWeights: new Float32Array([0.1, 0.1, 0.1, 0.1, 0.1, 0.5]),
      skinInfluencesPerVertex: 6,
      bones,
      boneInverses: inverses,
      material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 },
    };

    const solved = solveSkin(prim);

    expect(solved.positions[0]).toBeCloseTo(3.5, 6);
    expect(solved.positions[1]).toBeCloseTo(0, 6);
    expect(solved.normals[1]).toBeCloseTo(1, 6);
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

  it('skins rest tangents and preserves handedness', () => {
    const rotZ = new Float32Array([
      0, 1, 0, 0,
      -1, 0, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ]);
    const prim: SkinnedMeshPrimitive = {
      ...singleBonePrim({
        positions: new Float32Array([0, 0, 0]),
        normals: new Float32Array([0, 0, 1]),
        bonesMatrix: rotZ,
      }),
      tangents: new Float32Array([1, 0, 0, -1]),
    };

    const { tangents } = solveSkin(prim);

    expect(tangents).toBeDefined();
    expect(tangents![0]).toBeCloseTo(0, 6);
    expect(tangents![1]).toBeCloseTo(1, 6);
    expect(tangents![2]).toBeCloseTo(0, 6);
    expect(tangents![3]).toBe(-1);
  });

  it('applies morphTargetTangents before skinning', () => {
    const prim: SkinnedMeshPrimitive = {
      ...singleBonePrim({
        positions: new Float32Array([0, 0, 0]),
        normals: new Float32Array([0, 0, 1]),
        bonesMatrix: IDENT4(),
      }),
      tangents: new Float32Array([1, 0, 0, 1]),
      morphTargets: [new Float32Array([0, 0, 0])],
      morphTargetTangents: [new Float32Array([0, 1, 0])],
      morphWeights: new Float32Array([1]),
    };

    const { tangents } = solveSkin(prim);
    const invSqrt2 = 1 / Math.sqrt(2);

    expect(tangents).toBeDefined();
    expect(tangents![0]).toBeCloseTo(invSqrt2, 6);
    expect(tangents![1]).toBeCloseTo(invSqrt2, 6);
    expect(tangents![2]).toBeCloseTo(0, 6);
    expect(tangents![3]).toBe(1);
  });

  it('applies morphTargetUvs and morphTargetUv1s without skin-transforming texture coordinates', () => {
    const prim: SkinnedMeshPrimitive = {
      ...singleBonePrim({
        positions: new Float32Array([0, 0, 0, 1, 0, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1]),
        bonesMatrix: new Float32Array([
          1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1, 0,
          5, 7, 9, 1,
        ]),
      }),
      uvs: new Float32Array([0, 0, 0.5, 0.5]),
      uv1: new Float32Array([0.25, 0.25, 0.75, 0.75]),
      morphTargets: [new Float32Array([0, 0, 0, 0, 0, 0])],
      morphTargetUvs: [new Float32Array([0.1, 0.2, -0.2, 0.1])],
      morphTargetUv1s: [new Float32Array([0.5, 0, 0, -0.25])],
      morphWeights: new Float32Array([0.5]),
    };

    const { positions, uvs, uv1 } = solveSkin(prim);

    expect(positions[0]).toBeCloseTo(5);
    expect(positions[1]).toBeCloseTo(7);
    expect(Array.from(uvs ?? [])).toEqual(Array.from(new Float32Array([0.05, 0.1, 0.4, 0.55])));
    expect(Array.from(uv1 ?? [])).toEqual(Array.from(new Float32Array([0.5, 0.25, 0.75, 0.625])));
  });

  it('applies arbitrary sparse morphTargetUvSets and reuses caller output storage', () => {
    const uv2 = new Float32Array([0.2, 0.4, 0.6, 0.8]);
    const unchangedUv3 = new Float32Array([0.1, 0.3, 0.5, 0.7]);
    const prim: SkinnedMeshPrimitive = {
      ...singleBonePrim({
        positions: new Float32Array([0, 0, 0, 1, 0, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1]),
        bonesMatrix: IDENT4(),
      }),
      uvSets: [undefined, undefined, uv2, unchangedUv3],
      morphTargets: [new Float32Array(6)],
      morphTargetUvSets: [
        undefined,
        undefined,
        [new Float32Array([0.4, -0.2, -0.2, 0.4])],
      ],
      morphWeights: new Float32Array([0.5]),
    };
    const outUv2 = new Float32Array(4);

    const solved = solveSkin(
      prim,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      [undefined, undefined, outUv2],
    );

    expect(solved.uvSets?.[2]).toBe(outUv2);
    expect(solved.uvSets?.[3]).toBe(unchangedUv3);
    expect(Array.from(outUv2)).toEqual(Array.from(new Float32Array([0.4, 0.3, 0.5, 1])));
    expect(solved.uvs).toBeUndefined();
    expect(solved.uv1).toBeUndefined();
  });

  it('keeps scalable TEXCOORD_0/1 aliases coherent for legacy morph streams', () => {
    const uvs = new Float32Array([0, 0, 0.5, 0.5]);
    const uv1 = new Float32Array([0.25, 0.25, 0.75, 0.75]);
    const unchangedUv3 = new Float32Array([0.1, 0.3, 0.5, 0.7]);
    const prim: SkinnedMeshPrimitive = {
      ...singleBonePrim({
        positions: new Float32Array([0, 0, 0, 1, 0, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1]),
        bonesMatrix: IDENT4(),
      }),
      uvs,
      uv1,
      uvSets: [uvs, uv1, undefined, unchangedUv3],
      morphTargets: [new Float32Array(6)],
      morphTargetUvs: [new Float32Array([0.2, 0, 0, -0.2])],
      morphTargetUv1s: [new Float32Array([0, 0.2, -0.2, 0])],
      morphWeights: new Float32Array([0.5]),
    };

    const solved = solveSkin(prim);

    expect(solved.uvSets?.[0]).toBe(solved.uvs);
    expect(solved.uvSets?.[1]).toBe(solved.uv1);
    expect(solved.uvSets?.[3]).toBe(unchangedUv3);
  });

  it('applies additive RGB/RGBA COLOR_n morph deltas and reuses caller output storage', () => {
    const colors0 = new Float32Array([0.2, 0.4, 0.6, 1, 0.8, 0.6, 0.4, 0.5]);
    const colors2 = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    const delta0 = new Float32Array([0.2, -0.2, 0.4, 0, -0.4, 0.2, 0.2, 0.4]);
    const delta2 = new Float32Array([0.4, 0.2, -0.2, -0.2, 0.4, 0.2]);
    const prim: SkinnedMeshPrimitive = {
      ...singleBonePrim({
        positions: new Float32Array([0, 0, 0, 1, 0, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1]),
        bonesMatrix: IDENT4(),
      }),
      colors: colors0,
      colorSets: [colors0, undefined, colors2],
      morphTargets: [new Float32Array(6)],
      morphTargetColors: [delta0],
      morphTargetColorSets: [[delta0], undefined, [delta2]],
      morphWeights: new Float32Array([0.5]),
    };
    const out0 = new Float32Array(8);
    const out2 = new Float32Array(6);

    const solved = solveSkin(
      prim,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      out0,
      [out0, undefined, out2],
    );

    expect(solved.colors).toBe(out0);
    expect(solved.colorSets?.[0]).toBe(out0);
    expect(solved.colorSets?.[2]).toBe(out2);
    [0.3, 0.3, 0.8, 1, 0.6, 0.7, 0.5, 0.7].forEach((value, index) => {
      expect(out0[index]).toBeCloseTo(value);
    });
    [0.3, 0.3, 0.2, 0.3, 0.7, 0.7].forEach((value, index) => {
      expect(out2[index]).toBeCloseTo(value);
    });
  });

  it('preserves unchanged scalable color lanes and keeps the COLOR_0 alias coherent', () => {
    const colors0 = new Float32Array([0.2, 0.4, 0.6, 0.8, 0.6, 0.4]);
    const colors2 = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    const activeColors3 = new Float32Array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4]);
    const delta0 = new Float32Array([0.2, 0, -0.2, -0.2, 0, 0.2]);
    const delta2 = new Float32Array([0.4, 0.2, 0, 0, -0.2, -0.4]);
    const base = {
      ...singleBonePrim({
        positions: new Float32Array([0, 0, 0, 1, 0, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1]),
        bonesMatrix: IDENT4(),
      }),
      colors: colors0,
      colorSets: [colors0, undefined, colors2, activeColors3],
      vertexColorSet: 3,
      morphTargets: [new Float32Array(6)],
      morphWeights: new Float32Array([0.5]),
    } satisfies SkinnedMeshPrimitive;

    const scalable = solveSkin({
      ...base,
      morphTargetColorSets: [undefined, undefined, [delta2]],
    });
    expect(scalable.colorSets?.[3]).toBe(activeColors3);
    expect(scalable.colorSets?.[0]).toBe(colors0);
    [0.3, 0.3, 0.3, 0.4, 0.4, 0.4].forEach((value, index) => {
      expect(scalable.colorSets?.[2]?.[index]).toBeCloseTo(value);
    });

    const legacy = solveSkin({
      ...base,
      morphTargetColors: [delta0],
    });
    expect(legacy.colorSets?.[3]).toBe(activeColors3);
    expect(legacy.colorSets?.[0]).toBe(legacy.colors);
    [0.3, 0.4, 0.5, 0.7, 0.6, 0.5].forEach((value, index) => {
      expect(legacy.colors?.[index]).toBeCloseTo(value);
    });
  });

  it('solves morph UV lanes at the native index ceiling and above 2^32-1', () => {
    const indices = [0xffff_fffe, 0x1_0000_0001] as const;
    const uvSets: Array<Float32Array | undefined> = [];
    const morphTargetUvSets: Array<readonly Float32Array[] | undefined> = [];
    const outUvSets: Array<Float32Array | undefined> = [];
    for (const texCoord of indices) {
      uvSets[texCoord] = new Float32Array([0.2, 0.4, 0.6, 0.8]);
      morphTargetUvSets[texCoord] = [
        new Float32Array([0.4, -0.2, -0.2, 0.4]),
      ];
      outUvSets[texCoord] = new Float32Array(4);
    }
    const prim: SkinnedMeshPrimitive = {
      ...singleBonePrim({
        positions: new Float32Array([0, 0, 0, 1, 0, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1]),
        bonesMatrix: IDENT4(),
      }),
      uvSets,
      morphTargets: [new Float32Array(6)],
      morphTargetUvSets,
      morphWeights: new Float32Array([0.5]),
    };

    const solved = solveSkin(
      prim,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      outUvSets,
    );

    for (const texCoord of indices) {
      expect(solved.uvSets?.[texCoord]).toBe(outUvSets[texCoord]);
      expect(Array.from(outUvSets[texCoord] ?? [])).toEqual(
        Array.from(new Float32Array([0.4, 0.3, 0.5, 1])),
      );
    }
    expect(Object.keys(solved.uvSets ?? []).sort()).toEqual(
      indices.map(String).sort(),
    );
  });

  it('throws when morphTargetNormals count does not match morphTargets for active morphs', () => {
    const prim: SkinnedMeshPrimitive = {
      ...singleBonePrim({
        positions: new Float32Array([0, 0, 0]),
        normals: new Float32Array([0, 1, 0]),
        bonesMatrix: IDENT4(),
      }),
      morphTargets: [
        new Float32Array([1, 0, 0]),
        new Float32Array([0, 1, 0]),
      ],
      morphTargetNormals: [new Float32Array([0, 0, 1])],
      morphWeights: new Float32Array([1, 0]),
    };

    expect(() => solveSkin(prim)).toThrow(/morphTargetNormals length 1 != morphTargets 2/);
  });

  it('throws when an active morphTargetNormals entry has the wrong length', () => {
    const prim: SkinnedMeshPrimitive = {
      ...singleBonePrim({
        positions: new Float32Array([0, 0, 0]),
        normals: new Float32Array([0, 1, 0]),
        bonesMatrix: IDENT4(),
      }),
      morphTargets: [new Float32Array([1, 0, 0])],
      morphTargetNormals: [new Float32Array([0, 0])],
      morphWeights: new Float32Array([1]),
    };

    expect(() => solveSkin(prim)).toThrow(/morphTargetNormals\[0\] length 2 != normals 3/);
  });

  it('throws when morphTargetTangents count or entry length does not match active morphs', () => {
    const base = {
      ...singleBonePrim({
        positions: new Float32Array([0, 0, 0]),
        normals: new Float32Array([0, 1, 0]),
        bonesMatrix: IDENT4(),
      }),
      tangents: new Float32Array([1, 0, 0, 1]),
      morphTargets: [
        new Float32Array([1, 0, 0]),
        new Float32Array([0, 1, 0]),
      ],
      morphWeights: new Float32Array([1, 0]),
    };

    expect(() => solveSkin({
      ...base,
      morphTargetTangents: [new Float32Array([0, 0, 1])],
    })).toThrow(/morphTargetTangents length 1 != morphTargets 2/);

    expect(() => solveSkin({
      ...base,
      morphTargetTangents: [
        new Float32Array([0, 0]),
        new Float32Array([0, 0, 0]),
      ],
    })).toThrow(/morphTargetTangents\[0\] length 2 != tangents 3/);
  });

  it('throws when morph UV target counts or lengths do not match active morphs', () => {
    const base: SkinnedMeshPrimitive = {
      ...singleBonePrim({
        positions: new Float32Array([0, 0, 0]),
        normals: new Float32Array([0, 1, 0]),
        bonesMatrix: IDENT4(),
      }),
      uvs: new Float32Array([0, 0]),
      morphTargets: [
        new Float32Array([1, 0, 0]),
        new Float32Array([0, 1, 0]),
      ],
      morphWeights: new Float32Array([1, 0]),
    };

    expect(() => solveSkin({
      ...base,
      morphTargetUvs: [new Float32Array([0, 0])],
    })).toThrow(/morphTargetUvs length 1 != morphTargets 2/);

    expect(() => solveSkin({
      ...base,
      morphTargetUvs: [
        new Float32Array([0]),
        new Float32Array([0, 0]),
      ],
    })).toThrow(/morphTargetUvs\[0\] length 1 != uvs 2/);

    const withoutUvs: SkinnedMeshPrimitive = {
      kind: 'skinned-mesh',
      id: 'test-without-uvs',
      positions: base.positions,
      normals: base.normals,
      skinIndices: base.skinIndices,
      skinWeights: base.skinWeights,
      bones: base.bones,
      boneInverses: base.boneInverses,
      morphTargets: base.morphTargets!,
      morphWeights: base.morphWeights!,
      material: base.material,
    };
    expect(() => solveSkin({
      ...withoutUvs,
      morphTargetUvs: [
        new Float32Array([0, 0]),
        new Float32Array([0, 0]),
      ],
    })).toThrow(/morphTargetUvs supplied but primitive has no uvs stream/);
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

  it('throws when per-vertex skin weights do not sum to one', () => {
    const prim: SkinnedMeshPrimitive = {
      ...singleBonePrim({
        positions: new Float32Array([0, 0, 0]),
        normals: new Float32Array([0, 1, 0]),
        bonesMatrix: IDENT4(),
      }),
      skinWeights: new Float32Array([0.5, 0, 0, 0]),
    };
    expect(() => solveSkin(prim)).toThrow(/skinWeights for vertex 0 sum to 0.5; expected 1/);
  });

  // ── Theme 2 — scaled-bone inverse-transpose normals ─────────────────────
  // The C1 baseline transformed normals by the plain upper-3×3 of the skin
  // matrix, which is correct only for rigid bones. For a non-uniformly-scaled
  // bone the surface normal must transform by the INVERSE-TRANSPOSE; these
  // tests pin that the two answers genuinely differ and that solveSkin emits
  // the inverse-transpose result.

  it('non-uniform bone scale: an off-axis normal uses inverse-transpose (≠ plain matrix)', () => {
    // Column-major scale(2, 1, 1) — stretches X by 2×, Y/Z unchanged.
    const scaleX2 = new Float32Array([
      2, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ]);
    // Rest normal (1,1,0)/√2 — a 45° normal in the XY plane (off any axis so
    // the plain-matrix and inverse-transpose answers diverge).
    const inv2 = 1 / Math.sqrt(2);
    const prim = singleBonePrim({
      positions: new Float32Array([0, 0, 0]),
      normals: new Float32Array([inv2, inv2, 0]),
      bonesMatrix: scaleX2,
    });
    const { normals } = solveSkin(prim);

    // INVERSE-TRANSPOSE of scale(2,1,1) = scale(1/2,1,1). Applied to
    // (1,1,0)/√2 → (0.5/√2, 1/√2, 0), then normalized.
    const itx = 0.5 * inv2, ity = 1 * inv2, itz = 0;
    const itLen = Math.sqrt(itx * itx + ity * ity + itz * itz);
    expect(normals[0]).toBeCloseTo(itx / itLen, 5);
    expect(normals[1]).toBeCloseTo(ity / itLen, 5);
    expect(normals[2]).toBeCloseTo(itz / itLen, 5);

    // The (WRONG) plain-upper-3×3 transform would have produced
    // (2,1,0)/|(2,1,0)| — assert solveSkin's answer is NOT that, proving the
    // inverse-transpose is actually in effect.
    const plainX = 2 * inv2, plainY = 1 * inv2;
    const plainLen = Math.sqrt(plainX * plainX + plainY * plainY);
    expect(normals[0]).not.toBeCloseTo(plainX / plainLen, 3);
    expect(normals[1]).not.toBeCloseTo(plainY / plainLen, 3);
  });

  it('non-uniform scale keeps the deformed normal perpendicular to the deformed surface', () => {
    // A right-triangle tangent edge (1,1,0)/√2 with a normal (−1,1,0)/√2
    // (perpendicular in the XY plane). Under scale(3,1,1) on positions, the
    // edge maps to (3,1,0); the inverse-transpose normal must stay
    // perpendicular to that deformed edge (dot ≈ 0). A plain-matrix normal
    // would NOT.
    const scaleX3 = new Float32Array([
      3, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ]);
    const inv2 = 1 / Math.sqrt(2);
    const prim = singleBonePrim({
      positions: new Float32Array([0, 0, 0]),
      normals: new Float32Array([-inv2, inv2, 0]),
      bonesMatrix: scaleX3,
    });
    const { normals } = solveSkin(prim);

    // Deformed tangent edge = scale(3,1,1) · (1,1,0) = (3,1,0).
    const ex = 3, ey = 1, ez = 0;
    const dot = normals[0]! * ex + normals[1]! * ey + normals[2]! * ez;
    expect(dot).toBeCloseTo(0, 5);
  });

  it('mat3InverseTranspose: equals input for a pure rotation; is the scale-reciprocal for a diagonal', () => {
    // Rotation about Z by 90° (row-major). For an orthonormal R, (R⁻¹)ᵀ = R.
    const rotZ = [0, -1, 0, 1, 0, 0, 0, 0, 1];
    const itR = mat3InverseTranspose(rotZ);
    for (let i = 0; i < 9; i++) expect(itR[i]).toBeCloseTo(rotZ[i]!, 6);

    // Diagonal scale(2,4,5). Inverse-transpose of a diagonal is the
    // element-wise reciprocal (still diagonal).
    const scl = [2, 0, 0, 0, 4, 0, 0, 0, 5];
    const itS = mat3InverseTranspose(scl);
    expect(itS[0]).toBeCloseTo(1 / 2, 6);
    expect(itS[4]).toBeCloseTo(1 / 4, 6);
    expect(itS[8]).toBeCloseTo(1 / 5, 6);
    expect(itS[1]).toBeCloseTo(0, 6);
    expect(itS[3]).toBeCloseTo(0, 6);
  });

  it('mat3InverseTranspose: falls back to the input on a singular matrix (no NaN)', () => {
    // A rank-2 matrix (third row = first row) → det 0.
    const singular = [1, 2, 3, 0, 1, 0, 1, 2, 3];
    const it = mat3InverseTranspose(singular);
    for (let i = 0; i < 9; i++) {
      expect(Number.isFinite(it[i]!)).toBe(true);
      expect(it[i]).toBe(singular[i]);
    }
  });

  // ── GPU WGSL parity — the with-normals skin kernel
  // (walkaround-hybrid/src/skin/gpuSkinBvh.wgsl.ts) reimplements the
  // inverse-transpose with a DIFFERENT algebra than this CPU function: it works
  // on column-major inputs and returns the cofactor columns directly as
  // (c1×c2)/det, (c2×c0)/det, (c0×c1)/det. The CPU function uses the row-major
  // adjugate-over-determinant form. Both are mathematically (M⁻¹)ᵀ, but the WGSL
  // is GPU-only (no device in CI) and its tests only string-match. This pins the
  // ALGEBRA: a faithful TS port of the WGSL formula must produce the SAME matrix
  // as the trusted CPU function on the same matrix — catching a swapped column or
  // a flipped cross-product order in the shader. ───────────────────────────────
  it('mat3InverseTranspose: WGSL column-cross formula agrees with the CPU adjugate form', () => {
    const cross = (a: number[], b: number[]): number[] => [
      a[1]! * b[2]! - a[2]! * b[1]!,
      a[2]! * b[0]! - a[0]! * b[2]!,
      a[0]! * b[1]! - a[1]! * b[0]!,
    ];
    const dot = (a: number[], b: number[]): number =>
      a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;

    // Faithful port of GPU mat3InverseTranspose(c0,c1,c2) → returns the result
    // as a ROW-major 3×3 so we can compare it slot-for-slot to the CPU output.
    // GPU returns COLUMNS (c1×c2)/det, (c2×c0)/det, (c0×c1)/det; row-major[r][c]
    // = column[c][r].
    const gpuInverseTransposeRowMajor = (
      c0: number[], c1: number[], c2: number[],
    ): number[] => {
      const det = dot(c0, cross(c1, c2));
      const col0 = cross(c1, c2);
      const col1 = cross(c2, c0);
      const col2 = cross(c0, c1);
      const inv = 1 / det;
      // row-major [r0c0,r0c1,r0c2, r1c0,...]: out[r*3+c] = col_c[r]/det
      return [
        col0[0]! * inv, col1[0]! * inv, col2[0]! * inv,
        col0[1]! * inv, col1[1]! * inv, col2[1]! * inv,
        col0[2]! * inv, col1[2]! * inv, col2[2]! * inv,
      ];
    };

    // Test matrices: a non-uniform diagonal, a shear, and a rotation+scale.
    // Each given as column-major columns c0,c1,c2 (matching the WGSL input) and
    // the equivalent row-major buffer (matching the CPU input).
    const cases: { c0: number[]; c1: number[]; c2: number[] }[] = [
      // diag(2,3,5)
      { c0: [2, 0, 0], c1: [0, 3, 0], c2: [0, 0, 5] },
      // shear: x += 0.5y, with a y-stretch
      { c0: [1, 0, 0], c1: [0.5, 2, 0], c2: [0, 0, 1] },
      // arbitrary non-symmetric, full-rank
      { c0: [1, 2, 0], c1: [0, 1, 3], c2: [4, 0, 1] },
    ];

    for (const { c0, c1, c2 } of cases) {
      // Column-major M columns → row-major buffer [m00,m01,m02, m10,...].
      // m[r][c] = column_c[r].
      const rowMajor = [
        c0[0]!, c1[0]!, c2[0]!,
        c0[1]!, c1[1]!, c2[1]!,
        c0[2]!, c1[2]!, c2[2]!,
      ];
      const cpu = mat3InverseTranspose(rowMajor);
      const gpu = gpuInverseTransposeRowMajor(c0, c1, c2);
      for (let i = 0; i < 9; i++) {
        expect(gpu[i]!).toBeCloseTo(cpu[i]!, 6);
      }
    }
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

  it('throws when a skin index references a missing bone', () => {
    const bad = {
      ...singleBonePrim({
        positions: new Float32Array([0, 0, 0]),
        normals: new Float32Array([0, 1, 0]),
        bonesMatrix: IDENT4(),
      }),
      skinIndices: new Uint32Array([1, 0, 0, 0]),
    };
    expect(() => solveSkin(bad)).toThrow(/skinIndices\[0\] references bone 1/);
  });

  it('throws when skin weights are non-finite or negative', () => {
    const bad = {
      ...singleBonePrim({
        positions: new Float32Array([0, 0, 0]),
        normals: new Float32Array([0, 1, 0]),
        bonesMatrix: IDENT4(),
      }),
      skinWeights: new Float32Array([Number.NaN, 0, 0, 0]),
    };
    expect(() => solveSkin(bad)).toThrow(/skinWeights\[0\] is invalid/);
  });

  it('rejects a positions stream whose length is not divisible by three', () => {
    const bad = {
      ...singleBonePrim({
        positions: new Float32Array([0, 0, 0]),
        normals: new Float32Array([0, 1, 0]),
        bonesMatrix: IDENT4(),
      }),
      positions: new Float32Array([0, 0, 0, 1]),
      normals: new Float32Array([0, 1, 0, 0]),
    };
    expect(() => solveSkin(bad)).toThrow(/positions length 4.*multiple of 3/);
  });

  it('rejects runtime skin streams with the wrong typed-array representation', () => {
    const base = singleBonePrim({
      positions: new Float32Array([0, 0, 0]),
      normals: new Float32Array([0, 1, 0]),
      bonesMatrix: IDENT4(),
    });
    expect(() => solveSkin({
      ...base,
      skinIndices: new Uint16Array([0, 0, 0, 0]) as unknown as Uint32Array,
    })).toThrow(/skinIndices must be a Uint32Array/);
    expect(() => solveSkin({
      ...base,
      skinWeights: [1, 0, 0, 0] as unknown as Float32Array,
    })).toThrow(/skinWeights must be a Float32Array/);
  });

  it('rejects bone buffers with a trailing partial matrix', () => {
    const bad = {
      ...singleBonePrim({
        positions: new Float32Array([0, 0, 0]),
        normals: new Float32Array([0, 1, 0]),
        bonesMatrix: IDENT4(),
      }),
      bones: new Float32Array(17),
    };
    expect(() => solveSkin(bad)).toThrow(/bones length 17 not a multiple of 16/);
  });

  it('rejects non-finite geometry and bone-matrix values', () => {
    const geometry = singleBonePrim({
      positions: new Float32Array([Number.NaN, 0, 0]),
      normals: new Float32Array([0, 1, 0]),
      bonesMatrix: IDENT4(),
    });
    expect(() => solveSkin(geometry)).toThrow(/positions\[0\].*not finite/);
    const matrix = singleBonePrim({
      positions: new Float32Array([0, 0, 0]),
      normals: new Float32Array([0, 1, 0]),
      bonesMatrix: new Float32Array([
        Number.POSITIVE_INFINITY, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
      ]),
    });
    expect(() => solveSkin(matrix)).toThrow(/bones\[0\].*not finite/);
  });

  it('validates inactive morph targets and optional streams', () => {
    const base = singleBonePrim({
      positions: new Float32Array([0, 0, 0]),
      normals: new Float32Array([0, 1, 0]),
      bonesMatrix: IDENT4(),
    });
    expect(() => solveSkin({
      ...base,
      morphTargets: [new Float32Array(2)],
      morphWeights: new Float32Array([0]),
    })).toThrow(/morphTargets\[0\] length 2 != positions 3/);
    expect(() => solveSkin({
      ...base,
      morphTargets: [new Float32Array(3)],
      morphTargetNormals: [new Float32Array([0, Number.NaN, 0])],
      morphWeights: new Float32Array([0]),
    })).toThrow(/morphTargetNormals\[0\]\[1\].*not finite/);
    expect(() => solveSkin({
      ...base,
      morphWeights: new Float32Array([0]),
    })).toThrow(/morphTargets are required/);
  });

  it('requires bind matrices as a complete finite pair', () => {
    const base = singleBonePrim({
      positions: new Float32Array([0, 0, 0]),
      normals: new Float32Array([0, 1, 0]),
      bonesMatrix: IDENT4(),
    });
    expect(() => solveSkin({ ...base, bindMatrix: IDENT4() }))
      .toThrow(/must be supplied together/);
    const inverse = IDENT4();
    inverse[4] = Number.NaN;
    expect(() => solveSkin({ ...base, bindMatrix: IDENT4(), bindMatrixInverse: inverse }))
      .toThrow(/bindMatrixInverse\[4\].*not finite/);
  });
});
