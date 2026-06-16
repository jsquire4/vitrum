// skinSolve.test.ts — verifies that pt-webgl2 scene ingestion applies solveSkin
// to skinned-mesh primitives and leaves unskinned meshes byte-identical.
//
// Fixture design mirrors core's skinSolver.test.ts: a two-bone skeleton where
// bone 0 = identity and bone 1 = translate(4, 0, 0), so we can compute the
// expected world positions analytically and compare them against what the packer
// sees through `_debugGeoPack`.
//
// The test does NOT assert pixel correctness (that is the real-GPU A/B) — it
// asserts that the PACKED BVH positions for a posed skinned-mesh match an
// independent CPU expectation and differ from the rest pose.

import { describe, it, expect, vi } from 'vitest';
import type {
  MaterialSpec,
  MeshPrimitive,
  Scene,
  SkinnedMeshPrimitive,
} from '@vitrum/core';
import { solveSkin } from '@vitrum/core';
import { createPTEngine_WebGL2 } from '../index.js';
import type { PTEngineWebGL2Options } from '../index.js';
import { createMockGl } from './mockGl.js';
import { solveSkinPrimitives } from '../scene/solveSkinPrimitives.js';

// ── Helpers ──────────────────────────────────────────────────────────────────

const GREY: MaterialSpec = { baseColor: [0.5, 0.5, 0.5], roughness: 1, metallic: 0 };

/** Column-major 4×4 identity. */
function ident4(): Float32Array {
  return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
}

/** Column-major translate(tx, ty, tz). */
function translate4(tx: number, ty: number, tz: number): Float32Array {
  return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, tx, ty, tz, 1]);
}

/**
 * Two-bone skinned-mesh:
 *   bone 0 = identity  (boneInverse = identity)
 *   bone 1 = translate(4, 0, 0)  (boneInverse = identity)
 *
 * v0 = weight 100% bone 0 → stays at rest pos
 * v1 = weight 100% bone 1 → shifts +4 in X
 * v2 = weight 50/50       → shifts +2 in X
 *
 * Rest positions:
 *   v0 = (0, 0, 0)
 *   v1 = (0, 0, 1)
 *   v2 = (0, 0, 2)
 *
 * Expected SOLVED positions:
 *   v0 = (0,  0, 0)  (bone 0, identity)
 *   v1 = (4,  0, 1)  (bone 1, translate +4 in X)
 *   v2 = (2,  0, 2)  (50% bone0 + 50% bone1 → 0.5×0 + 0.5×4 = 2)
 *
 * Normals (all pointing +Z in rest pose, should stay +Z for pure translation):
 *   v0 = v1 = v2 = (0, 0, 1)
 *
 * Triangle: v0-v1-v2  (one triangle).
 */
function twoBoneSkinnedPrim(id = 'skinned'): SkinnedMeshPrimitive {
  const bones = new Float32Array([
    ...ident4(),          // bone 0 = identity
    ...translate4(4, 0, 0), // bone 1 = translate(4,0,0)
  ]);
  const boneInverses = new Float32Array([...ident4(), ...ident4()]);

  // 3 vertices × 4 influences each:
  const skinIndices = new Uint32Array([
    0, 0, 0, 0,  // v0: bone 0 w=1, rest w=0
    1, 0, 0, 0,  // v1: bone 1 w=1, rest w=0
    0, 1, 0, 0,  // v2: bone 0 w=0.5, bone 1 w=0.5
  ]);
  const skinWeights = new Float32Array([
    1.0, 0, 0, 0,         // v0
    1.0, 0, 0, 0,         // v1
    0.5, 0.5, 0, 0,       // v2
  ]);

  return {
    kind: 'skinned-mesh',
    id,
    positions: new Float32Array([0, 0, 0, 0, 0, 1, 0, 0, 2]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    indices: new Uint32Array([0, 1, 2]),
    skinIndices,
    skinWeights,
    bones,
    boneInverses,
    material: GREY,
  };
}

function opts(): PTEngineWebGL2Options {
  return { device: createMockGl() };
}

// ── Tests ────────────────────────────────────────────────────────────────────

describe('pt-webgl2 skinned-mesh ingestion', () => {
  it('posed two-bone skeleton: packed BVH positions match independent solveSkin output', async () => {
    const prim = twoBoneSkinnedPrim('sk');
    const scene: Scene = {
      primitives: [prim],
      emitters: [],
      environment: { kind: 'none' },
    };

    // Independent expectation — run the solver directly (not through the engine).
    const expected = solveSkin(prim);

    const e = await createPTEngine_WebGL2(opts());
    e.setScene(scene);

    const geoPack = e._debugGeoPack;
    expect(geoPack).not.toBeNull();
    // The merged stream has exactly 1 triangle = 3 vertices.
    expect(geoPack!.triangleCount).toBe(1);

    // Extract the 3 world-space XYZ positions from the packed buffer.
    // mergeWorldSpaceFromCore uses positionStride:4 so each vertex = 4 floats.
    const stride = 4;
    const packed = geoPack!.positions;

    const px0 = packed[0 * stride + 0]!;
    const py0 = packed[0 * stride + 1]!;
    const pz0 = packed[0 * stride + 2]!;
    const px1 = packed[1 * stride + 0]!;
    const py1 = packed[1 * stride + 1]!;
    const pz1 = packed[1 * stride + 2]!;
    const px2 = packed[2 * stride + 0]!;
    const py2 = packed[2 * stride + 1]!;
    const pz2 = packed[2 * stride + 2]!;

    // Match the independent solveSkin result (within float tolerance).
    expect(px0).toBeCloseTo(expected.positions[0]!);
    expect(py0).toBeCloseTo(expected.positions[1]!);
    expect(pz0).toBeCloseTo(expected.positions[2]!);
    expect(px1).toBeCloseTo(expected.positions[3]!);
    expect(py1).toBeCloseTo(expected.positions[4]!);
    expect(pz1).toBeCloseTo(expected.positions[5]!);
    expect(px2).toBeCloseTo(expected.positions[6]!);
    expect(py2).toBeCloseTo(expected.positions[7]!);
    expect(pz2).toBeCloseTo(expected.positions[8]!);

    // Numeric expectation (two-bone translate: no rotation → easy to verify by hand).
    // v0: 100% bone0 (identity) → rest pos (0,0,0)
    expect(px0).toBeCloseTo(0);
    expect(py0).toBeCloseTo(0);
    expect(pz0).toBeCloseTo(0);
    // v1: 100% bone1 (translate +4 in X) → rest pos (0,0,1) → solved (4,0,1)
    expect(px1).toBeCloseTo(4);
    expect(py1).toBeCloseTo(0);
    expect(pz1).toBeCloseTo(1);
    // v2: 50/50 blend → (2, 0, 2)
    expect(px2).toBeCloseTo(2);
    expect(py2).toBeCloseTo(0);
    expect(pz2).toBeCloseTo(2);

    e.dispose();
  });

  it('posed two-bone skeleton: packed positions differ from rest pose', async () => {
    const prim = twoBoneSkinnedPrim('sk2');
    const scene: Scene = {
      primitives: [prim],
      emitters: [],
      environment: { kind: 'none' },
    };

    const e = await createPTEngine_WebGL2(opts());
    e.setScene(scene);

    const geoPack = e._debugGeoPack!;
    const stride = 4;
    const packed = geoPack.positions;

    // v1 is bone-1-only with translate(4,0,0); rest X = 0, solved X = 4.
    const solvedX1 = packed[1 * stride + 0]!;
    expect(solvedX1).toBeCloseTo(4); // NOT the rest-pose X (0)
    expect(solvedX1).not.toBeCloseTo(0);

    e.dispose();
  });

  it('unskinned mesh is byte-identical through the skinning pre-pass', async () => {
    // Baseline: a plain MeshPrimitive with no skinning data.
    const plain: MeshPrimitive = {
      kind: 'mesh',
      id: 'plain',
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      indices: new Uint32Array([0, 1, 2]),
      material: GREY,
    };

    const scene: Scene = {
      primitives: [plain],
      emitters: [],
      environment: { kind: 'none' },
    };

    const e = await createPTEngine_WebGL2(opts());
    e.setScene(scene);

    const geoPack = e._debugGeoPack!;
    expect(geoPack.triangleCount).toBe(1);

    // Positions should match the input exactly (stride 4, w=0 pad).
    const stride = 4;
    const packed = geoPack.positions;
    expect(packed[0 * stride + 0]!).toBeCloseTo(0); // v0 X
    expect(packed[1 * stride + 0]!).toBeCloseTo(1); // v1 X
    expect(packed[2 * stride + 0]!).toBeCloseTo(0); // v2 X
    expect(packed[0 * stride + 1]!).toBeCloseTo(0); // v0 Y
    expect(packed[2 * stride + 1]!).toBeCloseTo(1); // v2 Y

    e.dispose();
  });

  it('preserves CPU-solved posed tangents after skinning and morph tangent blend', () => {
    const prim: SkinnedMeshPrimitive = {
      ...twoBoneSkinnedPrim('sk-tangent'),
      tangents: new Float32Array([
        1, 0, 0, 1,
        1, 0, 0, 1,
        1, 0, 0, 1,
      ]),
      morphTargets: [new Float32Array(9)],
      morphWeights: new Float32Array([1]),
      morphTargetTangents: [new Float32Array([
        0, 1, 0,
        0, 1, 0,
        0, 1, 0,
      ])],
    };
    const scene: Scene = {
      primitives: [prim],
      emitters: [],
      environment: { kind: 'none' },
    };

    const expected = solveSkin(prim).tangents;
    const posed = solveSkinPrimitives(scene).primitives[0];
    expect(posed?.kind).toBe('skinned-mesh');
    if (posed?.kind !== 'skinned-mesh') {
      throw new Error('expected solved primitive to remain skinned-mesh');
    }
    expect(posed.tangents).toBeInstanceOf(Float32Array);
    expect(Array.from(posed.tangents!)).toEqual(
      Array.from(expected!, (v) => expect.closeTo(v, 6)),
    );
    expect(posed.tangents![1]).toBeGreaterThan(0);
  });

  it('updatePrimitive with new bones re-solves skinning', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    // Initial pose: bone 0 = identity (v1 at bone0 = rest pos X=0).
    try {
      const prim = twoBoneSkinnedPrim('sk3');
      const scene: Scene = {
        primitives: [prim],
        emitters: [],
        environment: { kind: 'none' },
      };

      const e = await createPTEngine_WebGL2(opts());
      e.setScene(scene);

      const before = e._debugGeoPack!.positions[1 * 4 + 0]!; // v1 solved X after initial pose

      // New pose: bone 0 = translate(10, 0, 0), bone 1 = translate(4, 0, 0).
      // v1 is 100% bone 1 -> solved X = 4 (unchanged from before).
      // v0 is 100% bone 0 -> rest X = 0; new solved X = 10.
      const newBones = new Float32Array([
        ...translate4(10, 0, 0), // bone 0 -> translate +10 in X
        ...translate4(4, 0, 0),  // bone 1 -> same as before
      ]);

      e.updatePrimitive?.('sk3', { bones: newBones });

      const after = e._debugGeoPack!.positions[0 * 4 + 0]!; // v0 solved X after re-pose
      expect(after).toBeCloseTo(10); // re-solved: bone 0 now translates +10
      expect(before).toBeCloseTo(4);  // v1 was bone-1-only = +4
      expect(warn.mock.calls.flat().map(String).filter((m) =>
        m.includes('updatePrimitive("sk3") fields [bones]'),
      )).toHaveLength(1);

      e.dispose();
    } finally {
      warn.mockRestore();
    }
  });
});
