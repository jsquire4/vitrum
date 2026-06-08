/**
 * CPU SET-EQUIVALENCE gate — the THREE-decoupled core-first DDGI merged-BVH path
 * (`SceneBvh.updateFromCore` → `mergeWorldSpaceFromCore` + the THREE-free
 * `packDDGIMaterialsFromCoreN` material packer) vs the THREE merged-BVH path it
 * mirrors (`vitrumSceneToThree` → `SceneBvh.update` → `buildSceneBVH` +
 * `packDDGIMaterialsN`), for the SAME core `Scene`.
 *
 * This is the DDGI analogue of `restir/__tests__/emitterListCoreEquivalence.test.ts`
 * (the ReSTIR-DI emitter decouple's CPU gate, commit `46a0078`). Both producers
 * start from ONE core `Scene`; the THREE side routes through `vitrumSceneToThree`
 * (the production THREE round-trip — which forces the emissive-as-final-radiance /
 * `emissiveIntensity = 1` convention), the core side through `updateFromCore` /
 * the core material packer (which reproduces that convention via
 * `toProductionEmissiveRadiance`). So the material radiometry is identical BY
 * CONSTRUCTION; the only remaining difference is the SAH builder (three-mesh-bvh
 * `MeshBVH` vs `buildArrayBvh`), which PERMUTES the BVH triangle ORDER but not the
 * triangle SET.
 *
 * Two things are pinned:
 *
 *  (1) GEOMETRY — the merged world-space triangle SET + per-triangle material id.
 *      A naive index-by-index `bvhNodes`/`indices` compare WOULD FAIL on the SAH
 *      ordering permutation even though the producers are geometrically
 *      identical. So both sides are decoded to a winding-insensitive, quantised
 *      world-vertex tri-key → materialId map and compared as SETS (mirroring
 *      `worldSpaceMerge.test.ts`'s `triKeyToMatId`). The world AABB (which DDGI's
 *      probe grid is sized from) is pinned float-close.
 *
 *  (2) MATERIALS — the packed `MaterialEntry` byte SET. The two paths dedup the
 *      same materials but MAY order the LUT slots differently (first-seen over a
 *      possibly-different primitive-visit order), and `triMaterialId` indexes into
 *      it, so a slot-by-slot compare could also spuriously differ. We therefore
 *      decode each material slot to its semantic fields and compare as a SET. The
 *      ei-collapse case (`emissiveIntensity ≠ 1`) is pinned explicitly: the core
 *      path must pack `emissive · 1` (NOT `emissive · ei`), matching the THREE
 *      path — the exact divergence the emitter decouple's GPU A/B caught.
 *
 * This is the no-GPU half of the validation; the CONVERGED GPU render A/B (high
 * frame count, comparing means/PSNR — NOT pixel identity) lives in the wsl-gpu
 * harness (`scripts/ddgi-core-bvh-ab.mjs`).
 */

import { describe, it, expect } from 'vitest';
import type * as THREE from 'three';
import type { MaterialSpec, Scene, MeshPrimitive } from '@vitrum/core';
import { asMat4 } from '@vitrum/core';
import { SceneBvh, type SceneBvhBuffers } from '@vitrum/shared-bvh';
import { LegacyThreeSceneBvh } from '@vitrum/shared-bvh/legacy/three';
import { vitrumSceneToThree } from '@vitrum/three-bindings';
import { packDDGIMaterialsN, packDDGIMaterialsFromCoreN, DDGI_MAX_MATERIALS } from '../probeUpdateMaterials.js';

// ──────────────────────────────────────────────────────────────────────────
// Core-Scene mesh helpers (a 4-vertex quad as two tris, matching the emitter
// test's `quad`).
// ──────────────────────────────────────────────────────────────────────────

function quad(
  id: string,
  verts: [number, number, number][],
  normal: [number, number, number],
  material: MaterialSpec,
  transform?: Float32Array,
): MeshPrimitive {
  return {
    kind: 'mesh',
    id,
    positions: new Float32Array([...verts[0]!, ...verts[1]!, ...verts[2]!, ...verts[3]!]),
    normals: new Float32Array([...normal, ...normal, ...normal, ...normal]),
    uvs: new Float32Array(8),
    indices: new Uint32Array([0, 2, 1, 2, 0, 3]),
    material,
    ...(transform ? { transform: asMat4(transform) } : {}),
  };
}

function translation(x: number, y: number, z: number): Float32Array {
  return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1]);
}

function emissiveMat(emissive: [number, number, number], ei: number, base: [number, number, number] = [0.1, 0.1, 0.1]): MaterialSpec {
  return { baseColor: base, roughness: 1, metallic: 0, emissive, emissiveIntensity: ei };
}
function opaqueMat(base: [number, number, number], roughness = 1, metallic = 0): MaterialSpec {
  return { baseColor: base, roughness, metallic };
}
function glassMat(base: [number, number, number], transmission: number, atten: [number, number, number]): MaterialSpec {
  return { baseColor: base, roughness: 0.1, metallic: 0, transmission, attenuationColor: atten, ior: 1.5 };
}

// ──────────────────────────────────────────────────────────────────────────
// (1) Geometry: merged world-space tri SET + per-tri materialId.
// ──────────────────────────────────────────────────────────────────────────

const STRIDE = 4; // DDGI builds at positionStride 4 (16-byte vec3f layout).

/** Quantise so f32-tail noise doesn't break the SET match (1e-3 grid >> last-ULP
 *  drift — same tolerance `worldSpaceMerge.test.ts` uses). */
function q(x: number): number {
  return Math.round(x * 1e3);
}

/** Winding-insensitive sorted world-vertex key of a triangle. */
function triKey(positions: Float32Array, i0: number, i1: number, i2: number): string {
  const v = (i: number): string => {
    const b = i * STRIDE;
    return `${q(positions[b]!)},${q(positions[b + 1]!)},${q(positions[b + 2]!)}`;
  };
  return [v(i0), v(i1), v(i2)].sort().join('|');
}

function triKeyToMatId(positions: Float32Array, indices: Uint32Array, triMaterialId: Uint32Array, triCount: number): Map<string, number> {
  const map = new Map<string, number>();
  for (let t = 0; t < triCount; t += 1) {
    const key = triKey(positions, indices[t * 3]!, indices[t * 3 + 1]!, indices[t * 3 + 2]!);
    map.set(key, triMaterialId[t]!);
  }
  return map;
}

/**
 * Build the merged BVH inputs for BOTH paths from one core Scene:
 *   - THREE side: vitrumSceneToThree(scene) → SceneBvh.update (buildSceneBVH).
 *   - core side:  SceneBvh.updateFromCore(scene) (mergeWorldSpaceFromCore).
 * Returns each path's tri-key→matId map + the materialId remap that makes the two
 * LUT orderings comparable (THREE matId X corresponds to core matId Y when they
 * tag the SAME world triangle).
 */
function buildBoth(scene: Scene): {
  threeBuffers: SceneBvhBuffers;
  coreBuffers: SceneBvhBuffers;
} {
  const threeBvh = new LegacyThreeSceneBvh();
  threeBvh.update(vitrumSceneToThree(scene));
  const coreBvh = new SceneBvh();
  coreBvh.updateFromCore(scene);
  const threeBuffers = threeBvh.buffers;
  const coreBuffers = coreBvh.buffers;
  if (threeBuffers == null || coreBuffers == null) {
    throw new Error('expected non-null buffers from both SceneBvh paths');
  }
  return { threeBuffers, coreBuffers };
}

function assertGeometrySetsEqual(scene: Scene): {
  /** Map from core LUT slot → THREE LUT slot (so the material SET test can align
   *  per-triangle ids across the two dedup orderings). */
  coreSlotToThreeSlot: Map<number, number>;
} {
  const { threeBuffers, coreBuffers } = buildBoth(scene);

  const threeMap = triKeyToMatId(
    threeBuffers.positions,
    threeBuffers.indices,
    threeBuffers.triMaterialId,
    threeBuffers.triMaterialId.length,
  );
  const coreTriCount = coreBuffers.triMaterialId.length;
  const coreMap = triKeyToMatId(coreBuffers.positions, coreBuffers.indices, coreBuffers.triMaterialId, coreTriCount);

  // Same tri COUNT + same world-tri SET.
  expect(coreTriCount).toBe(threeBuffers.triMaterialId.length);
  expect(coreMap.size).toBe(threeMap.size);
  expect([...coreMap.keys()].sort()).toEqual([...threeMap.keys()].sort());

  // Per-world-triangle the two paths must agree on WHICH material slot (modulo the
  // two LUTs' independent first-seen orderings — captured as a consistent
  // bijection coreSlot↔threeSlot).
  const coreSlotToThreeSlot = new Map<number, number>();
  for (const [key, coreSlot] of coreMap) {
    const threeSlot = threeMap.get(key)!;
    const existing = coreSlotToThreeSlot.get(coreSlot);
    if (existing === undefined) {
      coreSlotToThreeSlot.set(coreSlot, threeSlot);
    } else {
      // The mapping must be consistent — the same core slot can't map to two
      // different THREE slots (would mean the dedup grouped differently).
      expect(threeSlot).toBe(existing);
    }
  }

  // World AABB (DDGI probe grid sizing) is float-close.
  const tb = threeBuffers.boundingBox;
  const cb = coreBuffers.boundingBox;
  expect(cb.min[0]).toBeCloseTo(tb.min[0], 4);
  expect(cb.min[1]).toBeCloseTo(tb.min[1], 4);
  expect(cb.min[2]).toBeCloseTo(tb.min[2], 4);
  expect(cb.max[0]).toBeCloseTo(tb.max[0], 4);
  expect(cb.max[1]).toBeCloseTo(tb.max[1], 4);
  expect(cb.max[2]).toBeCloseTo(tb.max[2], 4);

  // The core path fills core-native `materials`; the THREE path is quarantined
  // behind the neutral `sourceMaterials` field.
  expect(coreBuffers.coreMaterials).not.toBeUndefined();
  expect(coreBuffers.materials.length).toBeGreaterThan(0);
  expect(threeBuffers.coreMaterials).toBeUndefined();
  expect(threeBuffers.materials.length).toBe(0);
  expect(threeBuffers.sourceMaterials?.length).toBeGreaterThan(0);

  return { coreSlotToThreeSlot };
}

// ──────────────────────────────────────────────────────────────────────────
// (2) Materials: packed MaterialEntry byte SET.
// ──────────────────────────────────────────────────────────────────────────

const MAT_FLOATS = 16; // 64-byte stride / 4.

interface DecodedMatEntry {
  baseColor: [number, number, number];
  roughness: number;
  emissive: [number, number, number];
  metalness: number;
  ior: number;
  transmission: number;
  attenuationDistance: number;
  thickness: number;
  attenuationColor: [number, number, number];
  flags: number;
}

function decodeMatEntries(buf: ArrayBuffer, count: number): DecodedMatEntry[] {
  const f = new Float32Array(buf);
  const u = new Uint32Array(buf);
  const out: DecodedMatEntry[] = [];
  for (let i = 0; i < count; i += 1) {
    const b = i * MAT_FLOATS;
    out.push({
      baseColor: [f[b]!, f[b + 1]!, f[b + 2]!],
      roughness: f[b + 3]!,
      emissive: [f[b + 4]!, f[b + 5]!, f[b + 6]!],
      metalness: f[b + 7]!,
      ior: f[b + 8]!,
      transmission: f[b + 9]!,
      attenuationDistance: f[b + 10]!,
      thickness: f[b + 11]!,
      attenuationColor: [f[b + 12]!, f[b + 13]!, f[b + 14]!],
      flags: u[b + 15]!,
    });
  }
  return out;
}

function matKey(m: DecodedMatEntry): string {
  const q4 = (x: number): string => Math.round(x * 1e4).toString();
  return [
    ...m.baseColor.map(q4), q4(m.roughness), ...m.emissive.map(q4), q4(m.metalness),
    q4(m.ior), q4(m.transmission), q4(m.thickness), ...m.attenuationColor.map(q4), m.flags,
  ].join(',');
}

/** Compare the USED material slots (count = number of distinct materials in the
 *  scene) as a SET. The padding slots beyond `usedCount` are the same library
 *  default on both sides, so we only compare the real entries. */
function assertMaterialSetsEqual(scene: Scene, usedCount: number): void {
  const { threeBuffers, coreBuffers } = buildBoth(scene);
  const threeMaterials = [...(threeBuffers.sourceMaterials ?? [])] as THREE.Material[];
  const threeBuf = packDDGIMaterialsN(threeMaterials, DDGI_MAX_MATERIALS);
  const coreBuf = packDDGIMaterialsFromCoreN(coreBuffers.materials, DDGI_MAX_MATERIALS);

  expect(threeMaterials.length).toBe(usedCount);
  expect(coreBuffers.materials.length).toBe(usedCount);

  const threeEntries = decodeMatEntries(threeBuf, usedCount).map(matKey).sort();
  const coreEntries = decodeMatEntries(coreBuf, usedCount).map(matKey).sort();
  expect(coreEntries).toEqual(threeEntries);
}

// ──────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────

describe('DDGI merged-BVH core path ≡ THREE path — geometry SET', () => {
  it('single opaque quad', () => {
    const scene: Scene = {
      primitives: [quad('q', [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], [0, 0, 1], opaqueMat([0.7, 0.2, 0.2]))],
      emitters: [],
      environment: { kind: 'none' },
    };
    assertGeometrySetsEqual(scene);
  });

  it('full emissive Cornell (5 walls + emissive mesh) — the GPU A/B scene', () => {
    const primitives: MeshPrimitive[] = [
      quad('floor', [[-1, -1, -1], [1, -1, -1], [1, -1, 1], [-1, -1, 1]], [0, 1, 0], opaqueMat([0.8, 0.8, 0.8])),
      quad('ceiling', [[-1, 1, -1], [-1, 1, 1], [1, 1, 1], [1, 1, -1]], [0, -1, 0], opaqueMat([0.8, 0.8, 0.8])),
      quad('back-wall', [[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], [0, 0, -1], opaqueMat([0.8, 0.8, 0.8])),
      quad('left-wall', [[-1, -1, -1], [-1, -1, 1], [-1, 1, 1], [-1, 1, -1]], [1, 0, 0], opaqueMat([0.75, 0.1, 0.1])),
      quad('right-wall', [[1, -1, 1], [1, -1, -1], [1, 1, -1], [1, 1, 1]], [-1, 0, 0], opaqueMat([0.1, 0.6, 0.1])),
      quad('emitter', [[-0.4, -0.4, 0.5], [0.4, -0.4, 0.5], [0.4, 0.4, 0.5], [-0.4, 0.4, 0.5]], [0, 0, -1], emissiveMat([1.0, 0.45, 0.15], 4, [0, 0, 0])),
    ];
    const scene: Scene = { primitives, emitters: [], environment: { kind: 'none' } };
    assertGeometrySetsEqual(scene);
  });

  it('multiple transformed meshes (translations)', () => {
    const scene: Scene = {
      primitives: [
        quad('a', [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], [0, 0, 1], opaqueMat([0.7, 0.1, 0.1]), translation(-0.8, 0.6, 0.2)),
        quad('b', [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], [0, 0, 1], opaqueMat([0.1, 0.7, 0.1]), translation(0.9, -0.3, -0.4)),
        quad('c', [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], [0, 0, 1], glassMat([0.95, 0.97, 1.0], 0.9, [0.6, 0.8, 0.95]), translation(0, 0, 1)),
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    assertGeometrySetsEqual(scene);
  });
});

describe('DDGI merged-BVH core materials ≡ THREE materials — packed MaterialEntry SET', () => {
  it('opaque + glass materials pack byte-identically (the fields the probe kernel reads: flags/transmission/attenuationColor/baseColor)', () => {
    const scene: Scene = {
      primitives: [
        quad('opaque', [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], [0, 0, 1], opaqueMat([0.7, 0.2, 0.2], 0.4, 0.0)),
        quad('metal', [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], [0, 0, 1], opaqueMat([0.9, 0.9, 0.9], 0.2, 1.0), translation(3, 0, 0)),
        quad('glass', [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], [0, 0, 1], glassMat([0.95, 0.97, 1.0], 0.9, [0.6, 0.8, 0.95]), translation(6, 0, 0)),
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    assertMaterialSetsEqual(scene, 3);
  });

  it('emissive material with emissiveIntensity ≠ 1 — the ei-collapse footgun (Le = emissive·1, NOT emissive·4)', () => {
    // ei=4 is EXACTLY the case the emitter decouple's GPU A/B caught: vitrumSceneToThree
    // forces THREE ei=1 → packed emissive = emissive·1; the core packer must reproduce
    // that (NOT emissive·4) via toProductionEmissiveRadiance.
    const scene: Scene = {
      primitives: [
        quad('lamp', [[-0.4, -0.4, 0.5], [0.4, -0.4, 0.5], [0.4, 0.4, 0.5], [-0.4, 0.4, 0.5]], [0, 0, -1], emissiveMat([1.0, 0.45, 0.15], 4, [0, 0, 0])),
        quad('wall', [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], [0, 0, 1], opaqueMat([0.8, 0.8, 0.8]), translation(0, 0, 2)),
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    assertMaterialSetsEqual(scene, 2);

    // Explicit pin: decode the core lamp slot and assert emissive·1.
    const { coreBuffers } = buildBoth(scene);
    const coreBuf = packDDGIMaterialsFromCoreN(coreBuffers.materials, DDGI_MAX_MATERIALS);
    const entries = decodeMatEntries(coreBuf, coreBuffers.materials.length);
    const lamp = entries.find((e) => e.emissive[0] > 0.5)!;
    expect(lamp.emissive[0]).toBeCloseTo(1.0, 4); // emissive·1, NOT 4.0
    expect(lamp.emissive[1]).toBeCloseTo(0.45, 4); // NOT 1.8
    expect(lamp.emissive[2]).toBeCloseTo(0.15, 4); // NOT 0.6
  });

  it('emissiveIntensity undefined (emissive present) — vitrumSceneToThree still forces ei=1', () => {
    // A core material with `emissive` but NO `emissiveIntensity`: vitrumSceneToThree
    // emits THREE ei=1 → Le=emissive·1. The core packer's toProductionEmissiveRadiance
    // forces ei=1 too (so the packed bytes match), even though a raw
    // coreMaterialToMaterialEntry would default the missing ei to ×1 anyway here —
    // the SET test pins they agree.
    const m: MaterialSpec = { baseColor: [0, 0, 0], roughness: 1, metallic: 0, emissive: [0.3, 0.6, 0.9] };
    const scene: Scene = {
      primitives: [quad('lamp', [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], [0, 0, 1], m)],
      emitters: [],
      environment: { kind: 'none' },
    };
    assertMaterialSetsEqual(scene, 1);
  });
});

describe('documents WHY the converged-but-not-pixel-identical framing holds (DDGI)', () => {
  it('the BVH triangle ORDER may differ even when the world-tri SET + materials are identical', () => {
    // Same contract as the emitter decouple: the SAH builders (MeshBVH vs
    // buildArrayBvh) permute the triangle order, so a low-spp probe-ray A/B differs
    // on stratification noise while the CONVERGED probe atlas matches. The CPU gate
    // is SET-equality, which holds regardless of whether the BVH orderings coincide.
    const scene: Scene = {
      primitives: [
        quad('a', [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], [0, 0, 1], opaqueMat([0.8, 0.2, 0.2]), translation(-2, 0, 0)),
        quad('b', [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], [0, 0, 1], opaqueMat([0.2, 0.2, 0.8]), translation(2, 0, 0)),
        quad('c', [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], [0, 0, 1], opaqueMat([0.2, 0.8, 0.2]), translation(0, 2, 0)),
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    assertGeometrySetsEqual(scene);
    assertMaterialSetsEqual(scene, 3);
  });
});
