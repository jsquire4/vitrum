/**
 * CPU SET-EQUIVALENCE gate — the THREE-decoupled core-first RC merged-BVH path
 * (`buildRCSceneBVHFromCore` → `mergeWorldSpaceFromCore` + the THREE-free
 * `packCascadeMaterialsFromCore` packer) vs the THREE merged-BVH path it mirrors
 * (`vitrumSceneToThree` → `buildRCSceneBVH` → `buildSceneBVH` +
 * `packCascadeMaterials`), for the SAME core `Scene` (items_to_fix F-RC2).
 *
 * The RC analogue of `ddgi/__tests__/ddgiMergedBvhCoreEquivalence.test.ts` (the
 * standalone-DDGI decouple's CPU gate, `15070cd`) and
 * `restir/__tests__/emitterListCoreEquivalence.test.ts` (the ReSTIR-DI emitter
 * decouple's gate, `46a0078`). Both producers start from ONE core `Scene`; the
 * THREE side routes through `vitrumSceneToThree` (the production THREE round-trip
 * — which forces the emissive-as-final-radiance / `emissiveIntensity = 1`
 * convention), the core side through `buildRCSceneBVHFromCore` /
 * `packCascadeMaterialsFromCore` (which reproduce that convention via
 * `toProductionEmissiveRadiance`). So the material radiometry is identical BY
 * CONSTRUCTION; the only remaining difference is the SAH builder (three-mesh-bvh
 * `MeshBVH` vs `buildArrayBvh`), which PERMUTES the BVH triangle ORDER but not the
 * triangle SET.
 *
 * Two things are pinned:
 *
 *  (1) GEOMETRY — the merged world-space triangle SET + per-triangle material id.
 *      An index-by-index compare WOULD FAIL on the SAH-ordering permutation even
 *      though the producers are geometrically identical, so both sides are decoded
 *      to a winding-insensitive, quantised world-vertex tri-key → materialId map
 *      and compared as SETS (mirroring `worldSpaceMerge.test.ts`'s `triKeyToMatId`).
 *      The world AABB (which RC's cascade probe grid is sized from — see
 *      `RCSubsystem._setSceneFromBVH`) is pinned float-close.
 *
 *  (2) MATERIALS — the packed RC `MaterialEntry` byte SET. RC's probe kernel reads
 *      `mat.emissive` (`probeRayCast.wgsl.ts:265,271,303`), so the ei-collapse fix
 *      is LOAD-BEARING here (unlike DDGI). Both paths dedup the same materials but
 *      MAY order the LUT slots differently, and `triMaterialId` indexes into it,
 *      so a slot-by-slot compare could spuriously differ — we decode each slot to
 *      its semantic fields and compare as a SET. Pinned explicitly: the ei=4 case
 *      (core must pack `emissive · 1`, NOT `emissive · 4`) AND RC's `thickness → 0.1`
 *      floor (both paths floor a missing thickness to 0.1, so they MATCH at 0.1).
 *
 * The no-GPU half of the validation; the CONVERGED GPU render A/B (CHECK 1
 * byte-identical + CHECK 2 converged) lives in the wsl-gpu harness
 * (`scripts/rc-core-bvh-ab.ts`), and the decisive geometry gate is the RC
 * brute-force oracle (`scripts/rc-merged-bvh-bruteforce-ab.ts`).
 */

import { describe, it, expect } from 'vitest';
import type { MaterialSpec, Scene, MeshPrimitive } from '@vitrum/core';
import { asMat4 } from '@vitrum/core';
import { vitrumSceneToThree } from '@vitrum/three-bindings';
import type * as THREE from 'three';
import {
  buildRCSceneBVH,
  buildRCSceneBVHFromCore,
  packCascadeMaterials,
  packCascadeMaterialsFromCore,
  type SceneBVH,
} from '../bvhCompute.js';

// ──────────────────────────────────────────────────────────────────────────
// Core-Scene mesh helpers (a 4-vertex quad as two tris).
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
function glassMat(base: [number, number, number], transmission: number, atten: [number, number, number], thickness?: number): MaterialSpec {
  return { baseColor: base, roughness: 0.1, metallic: 0, transmission, attenuationColor: atten, ior: 1.5, ...(thickness !== undefined ? { thickness } : {}) };
}

// ──────────────────────────────────────────────────────────────────────────
// Build the merged RC BVH for BOTH paths from one core Scene through their REAL
// production code:
//   - THREE: vitrumSceneToThree(scene) → buildRCSceneBVH(threeRoot, allMeshes).
//   - core:  buildRCSceneBVHFromCore(scene).
// The THREE filter mirrors RCSubsystem.setScene's `allMeshesFilter`
// (obj.isMesh === true); the core filter defaults to RC_CORE_MESH_FILTER.
// ──────────────────────────────────────────────────────────────────────────

function buildBoth(scene: Scene): { threeBvh: SceneBVH; coreBvh: SceneBVH } {
  const allMeshesFilter = (obj: THREE.Object3D): boolean =>
    (obj as THREE.Mesh).isMesh === true;
  const threeRoot = vitrumSceneToThree(scene);
  threeRoot.updateMatrixWorld(true);
  const threeBvh = buildRCSceneBVH(threeRoot, { filter: allMeshesFilter });
  const coreBvh = buildRCSceneBVHFromCore(scene);
  return { threeBvh, coreBvh };
}

// ── (1) Geometry: merged world-space tri SET + per-tri materialId. ──────────

const STRIDE = 4; // RC builds at positionStride 4 (16-byte vec3f layout).

/** Quantise so f32-tail noise doesn't break the SET match. */
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

function assertGeometrySetsEqual(scene: Scene): void {
  const { threeBvh, coreBvh } = buildBoth(scene);

  const threePos = threeBvh.positions.array as Float32Array;
  const threeIdx = threeBvh.indices.array as Uint32Array;
  const threeMat = threeBvh.triMaterialId.array as Uint32Array;
  const corePos = coreBvh.positions.array as Float32Array;
  const coreIdx = coreBvh.indices.array as Uint32Array;
  const coreMat = coreBvh.triMaterialId.array as Uint32Array;

  const threeTriCount = threeMat.length;
  const coreTriCount = coreMat.length;

  const threeMap = triKeyToMatId(threePos, threeIdx, threeMat, threeTriCount);
  const coreMap = triKeyToMatId(corePos, coreIdx, coreMat, coreTriCount);

  // Same tri COUNT + same world-tri SET.
  expect(coreTriCount).toBe(threeTriCount);
  expect(coreMap.size).toBe(threeMap.size);
  expect([...coreMap.keys()].sort()).toEqual([...threeMap.keys()].sort());

  // Per-world-triangle the two paths must agree on WHICH material slot (modulo
  // the two LUTs' independent first-seen orderings — captured as a consistent
  // bijection coreSlot↔threeSlot).
  const coreSlotToThreeSlot = new Map<number, number>();
  for (const [key, coreSlot] of coreMap) {
    const threeSlot = threeMap.get(key)!;
    const existing = coreSlotToThreeSlot.get(coreSlot);
    if (existing === undefined) coreSlotToThreeSlot.set(coreSlot, threeSlot);
    else expect(threeSlot).toBe(existing);
  }

  // World AABB (RC cascade probe grid sizing) is float-close.
  const tb = threeBvh.bounds;
  const cb = coreBvh.bounds;
  expect(cb.min.x).toBeCloseTo(tb.min.x, 4);
  expect(cb.min.y).toBeCloseTo(tb.min.y, 4);
  expect(cb.min.z).toBeCloseTo(tb.min.z, 4);
  expect(cb.max.x).toBeCloseTo(tb.max.x, 4);
  expect(cb.max.y).toBeCloseTo(tb.max.y, 4);
  expect(cb.max.z).toBeCloseTo(tb.max.z, 4);
}

// ── (2) Materials: packed RC MaterialEntry byte SET. ────────────────────────

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

function decodeMatEntries(buf: Float32Array, count: number): DecodedMatEntry[] {
  const f = buf;
  const u = new Uint32Array(buf.buffer, buf.byteOffset, buf.length);
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
    q4(m.ior), q4(m.transmission), q4(m.attenuationDistance), q4(m.thickness),
    ...m.attenuationColor.map(q4), m.flags,
  ].join(',');
}

/**
 * The THREE side packs from the deduped `THREE.Material[]` that `buildRCSceneBVH`
 * surfaces; the core side from the deduped `MaterialSpec[]` that
 * `buildRCSceneBVHFromCore`'s merge surfaces. Both go through the REAL RC packers
 * (`packCascadeMaterials` / `packCascadeMaterialsFromCore`). We can't read the
 * THREE material list off `SceneBVH` (it's wrapped into the packed
 * `materials` StorageBufferAttribute), so we compare the PACKED buffers directly
 * — which is exactly what gets uploaded.
 */
function assertMaterialSetsEqual(scene: Scene, usedCount: number): void {
  const { threeBvh, coreBvh } = buildBoth(scene);
  const threeBuf = threeBvh.materials.array as Float32Array;
  const coreBuf = coreBvh.materials.array as Float32Array;

  // Both packers emit exactly `usedCount` entries (RC packs the deduped list with
  // no maxCount pad — `packMaterials(list)` → list.length entries).
  expect(threeBuf.length).toBe(usedCount * MAT_FLOATS);
  expect(coreBuf.length).toBe(usedCount * MAT_FLOATS);

  const threeEntries = decodeMatEntries(threeBuf, usedCount).map(matKey).sort();
  const coreEntries = decodeMatEntries(coreBuf, usedCount).map(matKey).sort();
  expect(coreEntries).toEqual(threeEntries);
}

// ──────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────

describe('RC merged-BVH core path ≡ THREE path — geometry SET', () => {
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

describe('RC merged-BVH core materials ≡ THREE materials — packed MaterialEntry SET', () => {
  it('opaque + metal + glass materials pack byte-identically', () => {
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

  it('emissive material with emissiveIntensity ≠ 1 — the ei-collapse footgun (Le = emissive·1, NOT emissive·4); RC READS mat.emissive so this is load-bearing', () => {
    const scene: Scene = {
      primitives: [
        quad('lamp', [[-0.4, -0.4, 0.5], [0.4, -0.4, 0.5], [0.4, 0.4, 0.5], [-0.4, 0.4, 0.5]], [0, 0, -1], emissiveMat([1.0, 0.45, 0.15], 4, [0, 0, 0])),
        quad('wall', [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], [0, 0, 1], opaqueMat([0.8, 0.8, 0.8]), translation(0, 0, 2)),
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    assertMaterialSetsEqual(scene, 2);

    // Explicit pin: decode the core lamp slot and assert emissive·1 (NOT ·4).
    const { coreBvh } = buildBoth(scene);
    const coreBuf = coreBvh.materials.array as Float32Array;
    const entries = decodeMatEntries(coreBuf, coreBuf.length / MAT_FLOATS);
    const lamp = entries.find((e) => e.emissive[0] > 0.5)!;
    expect(lamp.emissive[0]).toBeCloseTo(1.0, 4); // emissive·1, NOT 4.0
    expect(lamp.emissive[1]).toBeCloseTo(0.45, 4); // NOT 1.8
    expect(lamp.emissive[2]).toBeCloseTo(0.15, 4); // NOT 0.6
  });

  it("RC's deliberate thickness → 0.1 floor: a material with no thickness packs 0.1 on BOTH paths", () => {
    // Opaque + glass, neither carrying an explicit thickness. RC floors a missing
    // thickness to 0.1 (its per-tri Beer-Lambert needs a non-zero numerator), in
    // BOTH packers. So both pack 0.1 and they MATCH — the set test already covers
    // equality; this pins the FLOOR VALUE explicitly (a bare core map would leave
    // thickness 0 here, diverging from THREE).
    const scene: Scene = {
      primitives: [
        quad('opaque', [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], [0, 0, 1], opaqueMat([0.5, 0.5, 0.5])),
        quad('glass', [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], [0, 0, 1], glassMat([0.95, 0.97, 1.0], 0.9, [0.6, 0.8, 0.95]), translation(3, 0, 0)),
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    assertMaterialSetsEqual(scene, 2);

    const { threeBvh, coreBvh } = buildBoth(scene);
    const coreEntries = decodeMatEntries(coreBvh.materials.array as Float32Array, 2);
    const threeEntries = decodeMatEntries(threeBvh.materials.array as Float32Array, 2);
    for (const e of coreEntries) expect(e.thickness).toBeCloseTo(0.1, 4);
    for (const e of threeEntries) expect(e.thickness).toBeCloseTo(0.1, 4);
  });

  it('explicit thickness is preserved (NOT floored) on both paths', () => {
    // A glass material WITH an explicit thickness > 0 keeps it (the floor only
    // fills a missing/zero thickness). Pins that the floor doesn't clobber real data.
    const scene: Scene = {
      primitives: [
        quad('glass', [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], [0, 0, 1], glassMat([0.95, 0.97, 1.0], 0.9, [0.6, 0.8, 0.95], 0.35)),
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    assertMaterialSetsEqual(scene, 1);

    const { coreBvh } = buildBoth(scene);
    const entries = decodeMatEntries(coreBvh.materials.array as Float32Array, 1);
    expect(entries[0]!.thickness).toBeCloseTo(0.35, 4);
  });
});

describe('documents WHY the converged-but-not-pixel-identical framing holds (RC)', () => {
  it('the BVH triangle ORDER may differ even when the world-tri SET + materials are identical', () => {
    // The SAH builders (MeshBVH vs buildArrayBvh) permute the triangle order, so a
    // low-spp cascade A/B differs on stratification noise while the CONVERGED
    // cascade matches. The CPU gate is SET-equality, which holds regardless of
    // whether the BVH orderings coincide. (Pre-F-RC1 the GPU traversal rendered
    // the two trees DIFFERENTLY — that was the stride bug, now fixed; a correct
    // traversal returns identical closest-hits on either valid tree.)
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

  it('cross-check: packCascadeMaterialsFromCore on an empty list emits a single zeroed 64-byte entry (parity with packCascadeMaterials)', () => {
    expect(packCascadeMaterialsFromCore([]).byteLength).toBe(64);
    expect(packCascadeMaterials([]).byteLength).toBe(64);
  });
});
