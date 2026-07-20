import { describe, expect, it } from 'vitest';
import type { Mat4, Scene, ScenePrimitive } from '@vitrum/core';
import { asMat4 } from '@vitrum/core';
import { mergeUv1FromCore, mergeWorldSpaceFromCore } from '../worldSpaceMerge.js';

// ─────────────────────────────────────────────────────────────────────────────
// R5 / V2-4 characterization: mergeWorldSpaceFromCore SKIPS an instance whose
// transformed triangles all filter out (validTriangles.length === 0 → no range
// pushed). mergeUv1FromCore re-derived a per-instance skip loop that MISSED that
// guard, so `rangeIdx` desynced after an all-filtered predecessor and UV1
// attached to the WRONG primitive. The fix drives the consumer off ranges by a
// recorded source-primitive id instead of replicating the skip logic.
// ─────────────────────────────────────────────────────────────────────────────

function ident(): Mat4 {
  return asMat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]));
}

/** A translation with a non-finite (Inf) component → every transformed vertex is
 *  non-finite → all triangles filter out → the instance pushes NO range. */
function infTranslate(): Mat4 {
  return asMat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, Infinity, 0, 0, 1]));
}

/** A distinct-per-primitive uv1 so we can tell whose UV1 landed where. */
function meshWithUv1(id: string, u: number): ScenePrimitive {
  return {
    kind: 'mesh',
    id,
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    uvs: new Float32Array([0, 0, 1, 0, 0, 1]),
    // All three vertices carry the same marker so any vertex reveals the source.
    uv1: new Float32Array([u, u, u, u, u, u]),
    material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 },
  };
}

/** An instanced-mesh whose single instance is all-filtered (Inf transform) →
 *  contributes NO merged range. Carries its own uv1 so the desync (if any) would
 *  be observable, but its instance never lands. */
function allFilteredInstancedMesh(id: string): ScenePrimitive {
  return {
    kind: 'instanced-mesh',
    id,
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    uvs: new Float32Array([0, 0, 1, 0, 0, 1]),
    uv1: new Float32Array([9, 9, 9, 9, 9, 9]),
    material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 },
    instances: [infTranslate()],
  };
}

describe('mergeUv1FromCore — range-by-id after an all-filtered predecessor (R5 / V2-4)', () => {
  it('attaches each primitive UV1 to its OWN merged vertices despite a filtered middle primitive', () => {
    // A (uv1=1) → B (all-filtered instanced) → C (uv1=3).
    // B pushes NO range, so `merged.meshVertexRanges` is [A, C]. The old
    // consumer advanced rangeIdx for B's instance and mis-assigned C's UV1 to
    // A's slice (and left C's slice at 0).
    const scene: Scene = {
      primitives: [
        meshWithUv1('A', 1),
        allFilteredInstancedMesh('B'),
        meshWithUv1('C', 3),
      ],
      emitters: [],
      environment: { kind: 'none' },
    };

    const merged = mergeWorldSpaceFromCore(scene);
    // Sanity: B contributed no range.
    expect(merged.meshVertexRanges.map((r) => r.name)).toEqual(['A', 'C']);

    const uv1 = mergeUv1FromCore(scene, merged.meshVertexRanges, merged.vertexCount);
    expect(uv1).not.toBeUndefined();
    if (uv1 == null) return;

    const rangeA = merged.meshVertexRanges.find((r) => r.name === 'A')!;
    const rangeC = merged.meshVertexRanges.find((r) => r.name === 'C')!;

    // A's merged vertices must all read A's marker (1).
    for (let v = 0; v < rangeA.vertexCount; v += 1) {
      expect(uv1[(rangeA.vertexStart + v) * 2]).toBe(1);
      expect(uv1[(rangeA.vertexStart + v) * 2 + 1]).toBe(1);
    }
    // C's merged vertices must all read C's marker (3) — NOT 0 (old desync) and
    // NOT A's marker (old mis-shift).
    for (let v = 0; v < rangeC.vertexCount; v += 1) {
      expect(uv1[(rangeC.vertexStart + v) * 2]).toBe(3);
      expect(uv1[(rangeC.vertexStart + v) * 2 + 1]).toBe(3);
    }
  });

  it('unfiltered baseline stays correct (no regression for the common path)', () => {
    const scene: Scene = {
      primitives: [
        meshWithUv1('A', 1),
        { ...(meshWithUv1('B', 2)), transform: ident() } as ScenePrimitive,
        meshWithUv1('C', 3),
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    const merged = mergeWorldSpaceFromCore(scene);
    expect(merged.meshVertexRanges.map((r) => r.name)).toEqual(['A', 'B', 'C']);
    const uv1 = mergeUv1FromCore(scene, merged.meshVertexRanges, merged.vertexCount);
    expect(uv1).not.toBeUndefined();
    if (uv1 == null) return;
    for (const [name, marker] of [['A', 1], ['B', 2], ['C', 3]] as const) {
      const r = merged.meshVertexRanges.find((x) => x.name === name)!;
      for (let v = 0; v < r.vertexCount; v += 1) {
        expect(uv1[(r.vertexStart + v) * 2]).toBe(marker);
      }
    }
  });
});
