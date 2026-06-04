/**
 * BYTE-IDENTITY gate — the THREE-free core per-triangle MATERIAL packers (the
 * production `*FromCore` path: `packBVHIndexWFromCore` /
 * `packBVHBeerColorsFromCore` / `packBVHEmissiveLeFromCore`, driven by the
 * `buildMaterialResolver` parallel `coreMaterials[]`) vs the THREE `*Tri` packers
 * they replace (`packBVHIndexW` / `packBVHBeerColors` / `packBVHEmissiveLe`,
 * reading `vitrumSceneToThree`-converted THREE materials), for the SAME core
 * `Scene` — compared PER-TRIANGLE, byte-for-byte.
 *
 * Why BYTE-IDENTITY here (vs the SET-equivalence of the emitter / DDGI decouples,
 * commits `46a0078` / `15070cd`):
 *   The emitter list is built off `mergeWorldSpaceFromCore`'s STRUCTURAL dedup +
 *   `buildArrayBvh`'s SAH order, which permutes vs the THREE side — so only the
 *   SET matches. The per-triangle MATERIAL packers, by contrast, index by
 *   `geo.triMaterialIds`, which is produced by `packSceneFromCore` driven by
 *   `buildMaterialResolver`'s THREE-OBJECT-IDENTITY dedup ordering. The core
 *   `coreMaterials[]` is built in LOCKSTEP at the SAME slot index as the THREE
 *   `materials[]` (see `sceneBvhFromCore.ts:buildMaterialResolver`). So both
 *   producers share ONE triangle order AND ONE materialId→slot mapping → the
 *   packed bytes are EXACTLY equal per triangle, not merely set-equal.
 *
 * To make the comparison airtight we build the production buffers ONCE via
 * `buildReSTIRSceneBVHFromVitrumScene` (which emits the CORE-packed `bvhIndex` /
 * `bvhBeerColors` / `bvhEmissiveLe`) and re-pack the THREE reference over the
 * SAME `geo` it carries (`bvhIndicesStride3` + `triangleMaterialIds` +
 * `buildMaterials`). Both sides then provably address the identical triangle
 * stream + LUT, so any per-triangle byte difference is a real packer divergence.
 *
 * This is the no-GPU gate; the converged production ReSTIR/DDGI GPU output is
 * mechanically guaranteed identical when this byte-identity holds (the T1 GPU
 * smoke confirms no PSNR regression). Cases pinned (per the task): an ei=4
 * emissive (asserts `emissive·1`, NOT `·4`), a glass/transmissive surface, and a
 * multi-material multi-primitive scene (exercises the resolver ordering). Plus a
 * `packDDGIMaterialsFromCoreN(snap.coreMaterials)` byte-equals
 * `packDDGIMaterialsN(snap.materials)` cross-check (the production DDGI material
 * source the decouple newly routes through `snap.coreMaterials`).
 */

import { describe, it, expect } from 'vitest';
import type { MaterialSpec, Scene, MeshPrimitive } from '@vitrum/core';
import { asMat4 } from '@vitrum/core';
import { vitrumSceneToThree } from '@vitrum/three-bindings';
import { buildReSTIRSceneBVHFromVitrumScene } from '../sceneBvhFromCore.js';
import { makeRestirBvhSnapshot } from '../restirBvhSnapshot.js';
import {
  packBVHIndexW,
  packBVHBeerColors,
  packBVHEmissiveLe,
} from '../packingHelpers.js';
import {
  packDDGIMaterialsN,
  packDDGIMaterialsFromCoreN,
  DDGI_MAX_MATERIALS,
} from '../../ddgi/probeUpdateMaterials.js';

// ──────────────────────────────────────────────────────────────────────────
// Core-Scene mesh helpers (a 4-vertex quad as two tris — matches the emitter /
// DDGI equivalence tests' `quad`).
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
// Build the production buffers ONCE; re-pack the THREE reference over the SAME
// `geo` (triangle stream + materialId LUT + THREE material list) it carries.
// ──────────────────────────────────────────────────────────────────────────

interface PackedPair {
  triCount: number;
  /** Core path (production) — exactly the bytes `buffersFromScenePack` uploads. */
  coreIndexW: Uint32Array;
  coreBeer: Uint32Array;
  coreEmissive: Float32Array;
  /** THREE reference, packed over the SAME `geo`. */
  threeIndexW: Uint32Array;
  threeBeer: Uint32Array;
  threeEmissive: Float32Array;
}

function packBoth(scene: Scene): PackedPair {
  // `sceneRoots` is a pure round-trip in production (`vitrumSceneToThree(scene)`).
  const root = vitrumSceneToThree(scene);
  const buffers = buildReSTIRSceneBVHFromVitrumScene(scene, [root], {});

  const triCount = buffers.bvhIndex.count;
  const triMatIds = new Uint32Array(buffers.triangleMaterialIds.cpuData);
  // The core path packs bvhIndex.w over `geo.indices` (the stride-4 `ScenePackResult`
  // index buffer — `buffersFromScenePack` calls `packBVHIndexWFromCore(geo.indices,…)`),
  // so the THREE reference MUST pack over the SAME `geo.indices` for the index lanes
  // to be byte-comparable. `geo` is exposed as `buffers.scenePack` (= the
  // ScenePackResult). (`packBVHIndexW*` read `indices[tri*3+k]`; both packers see the
  // identical source, so any read-stride quirk is shared and cancels in the compare.)
  const geoIndices = buffers.scenePack!.indices;
  const threeMaterials = [...buffers.buildMaterials];

  // THREE reference over the SAME geo the core path packed.
  const threeIndexW = packBVHIndexW(geoIndices, triMatIds, threeMaterials, triCount);
  const threeBeer = packBVHBeerColors(triMatIds, threeMaterials, triCount);
  const threeEmissive = packBVHEmissiveLe(triMatIds, threeMaterials, triCount);

  return {
    triCount,
    coreIndexW: new Uint32Array(buffers.bvhIndex.cpuData),
    coreBeer: new Uint32Array(buffers.bvhBeerColors.cpuData),
    coreEmissive: new Float32Array(buffers.bvhEmissiveLe.cpuData),
    threeIndexW,
    threeBeer,
    threeEmissive,
  };
}

/** Per-triangle byte-identity of all three material lanes. */
function assertPerTriangleByteIdentity(p: PackedPair): void {
  expect(p.triCount).toBeGreaterThan(0);
  // bvhIndex.w u32 — vertex indices [0..2] + packed RGBA8 | trans4|isMetal|texType.
  // The whole vec4u (incl. the index lanes) must match.
  expect(p.coreIndexW.length).toBe(p.triCount * 4);
  expect(p.threeIndexW.length).toBe(p.triCount * 4);
  for (let i = 0; i < p.coreIndexW.length; i++) {
    expect(p.coreIndexW[i]).toBe(p.threeIndexW[i]);
  }
  // bvhBeerColors u32 (one per triangle).
  expect(p.coreBeer.length).toBe(p.triCount);
  for (let t = 0; t < p.triCount; t++) {
    expect(p.coreBeer[t]).toBe(p.threeBeer[t]);
  }
  // bvhEmissiveLe f32 (stride-4 per triangle). Exact f32 equality — both packers
  // write the identical IEEE-754 bits (same `emissive·1` source value).
  expect(p.coreEmissive.length).toBe(p.triCount * 4);
  for (let i = 0; i < p.coreEmissive.length; i++) {
    expect(p.coreEmissive[i]).toBe(p.threeEmissive[i]);
  }
}

// ──────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────

describe('ReSTIR per-triangle material packing: core path ≡ THREE path — BYTE-IDENTICAL', () => {
  it('ei=4 emissive quad — emissive Le packs as emissive·1 (NOT ·4)', () => {
    // The exact ei-collapse footgun the emitter decouple's GPU A/B caught:
    // vitrumSceneToThree forces THREE ei=1 → Le=emissive·1; the core packer must
    // reproduce that via toProductionEmissiveRadiance, NOT pack emissive·4.
    const scene: Scene = {
      primitives: [quad('lamp', [[-0.4, -0.4, 0.5], [0.4, -0.4, 0.5], [0.4, 0.4, 0.5], [-0.4, 0.4, 0.5]], [0, 0, -1], emissiveMat([1.0, 0.45, 0.15], 4, [0, 0, 0]))],
      emitters: [],
      environment: { kind: 'none' },
    };
    const p = packBoth(scene);
    assertPerTriangleByteIdentity(p);
    // Explicit pin: the non-zero emissive triangles carry emissive·1 (NOT ·4).
    let firedEmissive = 0;
    for (let t = 0; t < p.triCount; t++) {
      const r = p.coreEmissive[t * 4 + 0]!;
      if (r > 0) {
        firedEmissive++;
        expect(r).toBeCloseTo(1.0, 5); // emissive·1 — NOT 4.0
        expect(p.coreEmissive[t * 4 + 1]!).toBeCloseTo(0.45, 5); // NOT 1.8
        expect(p.coreEmissive[t * 4 + 2]!).toBeCloseTo(0.15, 5); // NOT 0.6
      }
    }
    expect(firedEmissive).toBe(2); // both tris of the emissive quad
  });

  it('glass / transmissive quad — raw attenuation color (bvhIndex.w) + Beer-Lambert color (bvh_beer) + glass low-byte bits', () => {
    // Transmissive: bvhIndex.w packs the RAW attenuation color + trans4; bvh_beer
    // packs the Beer-Lambert-tinted color. Both must match the THREE packers.
    const scene: Scene = {
      primitives: [quad('glass', [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], [0, 0, 1], glassMat([0.95, 0.97, 1.0], 0.9, [0.6, 0.8, 0.95]))],
      emitters: [],
      environment: { kind: 'none' },
    };
    const p = packBoth(scene);
    assertPerTriangleByteIdentity(p);
    // Sanity: a transmissive surface packs a non-zero trans4 nibble (bits 4..7 of
    // the low byte), so the index.w lanes are genuinely exercising the glass path.
    const lowByte = p.coreIndexW[3]! & 0xFF; // first triangle's packed material byte
    const trans4 = (lowByte >> 4) & 0xF;
    expect(trans4).toBeGreaterThan(0);
  });

  it('multi-material multi-primitive scene — exercises buildMaterialResolver slot ordering', () => {
    // Five distinct materials across six transformed primitives — the resolver's
    // THREE-identity dedup assigns slots in mesh-visit order; the parallel
    // coreMaterials[] must claim the SAME slot for the SAME mesh so every
    // triangle's materialId addresses byte-identical material data on both sides.
    const primitives: MeshPrimitive[] = [
      quad('floor', [[-1, -1, -1], [1, -1, -1], [1, -1, 1], [-1, -1, 1]], [0, 1, 0], opaqueMat([0.8, 0.8, 0.8])),
      quad('left', [[-1, -1, -1], [-1, -1, 1], [-1, 1, 1], [-1, 1, -1]], [1, 0, 0], opaqueMat([0.75, 0.1, 0.1])),
      quad('right', [[1, -1, 1], [1, -1, -1], [1, 1, -1], [1, 1, 1]], [-1, 0, 0], opaqueMat([0.1, 0.6, 0.1])),
      quad('metal', [[-0.5, -1, 0], [0.5, -1, 0], [0.5, 0, 0], [-0.5, 0, 0]], [0, 0, 1], opaqueMat([0.9, 0.9, 0.9], 0.2, 1.0), translation(0, 0.5, 0)),
      quad('glass', [[-0.4, -0.4, 0.5], [0.4, -0.4, 0.5], [0.4, 0.4, 0.5], [-0.4, 0.4, 0.5]], [0, 0, -1], glassMat([0.95, 0.97, 1.0], 0.9, [0.6, 0.8, 0.95]), translation(0.3, 0, 0)),
      quad('lamp', [[-0.3, 0.9, -0.3], [0.3, 0.9, -0.3], [0.3, 0.9, 0.3], [-0.3, 0.9, 0.3]], [0, -1, 0], emissiveMat([1.0, 0.9, 0.7], 6, [0, 0, 0])),
    ];
    const scene: Scene = { primitives, emitters: [], environment: { kind: 'none' } };
    const p = packBoth(scene);
    assertPerTriangleByteIdentity(p);
    // The emissive 'lamp' material has ei=6 → Le must STILL be emissive·1.
    let firedEmissive = 0;
    for (let t = 0; t < p.triCount; t++) {
      if (p.coreEmissive[t * 4 + 0]! > 0) {
        firedEmissive++;
        expect(p.coreEmissive[t * 4 + 0]!).toBeCloseTo(1.0, 5);
        expect(p.coreEmissive[t * 4 + 1]!).toBeCloseTo(0.9, 5);
        expect(p.coreEmissive[t * 4 + 2]!).toBeCloseTo(0.7, 5);
      }
    }
    expect(firedEmissive).toBe(2); // the lamp's two tris
  });

  it('opaque material with non-default roughness + metalness — isMetal bit + RGBA8 baseColor', () => {
    const scene: Scene = {
      primitives: [
        quad('matte', [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], [0, 0, 1], opaqueMat([0.2, 0.4, 0.6], 0.8, 0.0)),
        quad('metal', [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], [0, 0, 1], opaqueMat([0.7, 0.7, 0.7], 0.2, 1.0), translation(3, 0, 0)),
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    assertPerTriangleByteIdentity(packBoth(scene));
  });
});

// ──────────────────────────────────────────────────────────────────────────
// Cross-check: the production DDGI material source. The decouple routes
// production DDGI through `snap.coreMaterials` (packDDGIMaterialsFromCoreN);
// assert that packs byte-equal to the THREE `snap.materials`
// (packDDGIMaterialsN) it replaces.
// ──────────────────────────────────────────────────────────────────────────

describe('DDGI snapshot materials: packDDGIMaterialsFromCoreN(coreMaterials) ≡ packDDGIMaterialsN(materials) — byte-equal', () => {
  function assertDDGIMaterialBytesEqual(scene: Scene): void {
    const root = vitrumSceneToThree(scene);
    const buffers = buildReSTIRSceneBVHFromVitrumScene(scene, [root], {});
    const snap = makeRestirBvhSnapshot(buffers, scene);

    // The snapshot's core list is slot-aligned with its THREE list — same length,
    // same per-slot material — so packing each over DDGI_MAX_MATERIALS yields the
    // identical buffer (the production DDGI prefers snap.coreMaterials).
    expect(snap.coreMaterials.length).toBe(snap.materials.length);
    expect(snap.coreMaterials.length).toBeGreaterThan(0);

    const threeBuf = new Uint8Array(packDDGIMaterialsN([...snap.materials], DDGI_MAX_MATERIALS));
    const coreBuf = new Uint8Array(packDDGIMaterialsFromCoreN(snap.coreMaterials, DDGI_MAX_MATERIALS));
    expect(coreBuf.length).toBe(threeBuf.length);
    for (let i = 0; i < threeBuf.length; i++) {
      expect(coreBuf[i]).toBe(threeBuf[i]);
    }
  }

  it('opaque + metal + glass + emissive(ei=4) — full DDGI material struct byte-equal', () => {
    const scene: Scene = {
      primitives: [
        quad('opaque', [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], [0, 0, 1], opaqueMat([0.7, 0.2, 0.2], 0.4, 0.0)),
        quad('metal', [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], [0, 0, 1], opaqueMat([0.9, 0.9, 0.9], 0.2, 1.0), translation(3, 0, 0)),
        quad('glass', [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], [0, 0, 1], glassMat([0.95, 0.97, 1.0], 0.9, [0.6, 0.8, 0.95]), translation(6, 0, 0)),
        quad('lamp', [[-0.4, -0.4, 0.5], [0.4, -0.4, 0.5], [0.4, 0.4, 0.5], [-0.4, 0.4, 0.5]], [0, 0, -1], emissiveMat([1.0, 0.45, 0.15], 4, [0, 0, 0]), translation(9, 0, 0)),
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    assertDDGIMaterialBytesEqual(scene);
  });
});
