/**
 * CPU SET-EQUIVALENCE gate — the THREE-free core emitter path (the real
 * production `buildReSTIRSceneBVHFromVitrumScene` → `mergeWorldSpaceFromCore` +
 * `buildEmitterListFromCore`) vs the THREE emitter path it replaces
 * (`vitrumSceneToThree` → `buildSceneBVH` → `buildEmitterList`), for the SAME
 * core `Scene`.
 *
 * Both producers start from ONE core `Scene` and BOTH consume the SAME
 * `vitrumSceneToThree(scene)` THREE root for the THREE-reference side — so the
 * material radiometry is identical by construction (`vitrumSceneToThree` forces
 * the emissive-as-final-radiance / `emissiveIntensity = 1` convention; the core
 * path reproduces it via `toProductionEmissiveRadiance`). This is the faithful
 * production A/B at CPU level: the ONLY remaining difference is the SAH builder
 * (three-mesh-bvh `MeshBVH` vs `buildArrayBvh`), which permutes the emitter array
 * ORDER but not the emitter SET.
 *
 * Why a SET comparison (the load-bearing subtlety): the emitter list is built by
 * iterating the BVH-reordered triangle stream, so the emitter array ORDER differs
 * between the two producers (hence the CDF indexing and per-sample RIS selection
 * differ). A naive index-by-index compare would FAIL on ordering even though the
 * producers are radiometrically identical. So this test decodes both emitter
 * `Float32Array`s, sorts each by a canonical key (centroid ⊕ color), and asserts:
 *   (1) equal emitter COUNT,
 *   (2) per-emitter FLOAT-EQUALITY of every packed field after the sort,
 *   (3) equal TOTAL emissive power (the CDF normaliser),
 *   (4) the per-emitter power MULTISET is equal (the CDF step sizes agree).
 *
 * This is the no-GPU half of the validation; the CONVERGED GPU render A/B (high
 * frame count, comparing means/PSNR — NOT pixel identity) lives in the wsl-gpu
 * harness (`scripts/emitter-core-ab.mjs`).
 *
 * NOTE: routing the THREE side through `vitrumSceneToThree` is deliberate — an
 * earlier version of this test fed hand-matched THREE materials with the SAME
 * `emissiveIntensity` as the core material, which masked the production
 * `vitrumSceneToThree` ei-collapse and let an `emissive · emissiveIntensity` vs
 * `emissive · 1` divergence slip past CPU validation (the GPU A/B caught it). The
 * `vitrumSceneToThree`-routed form below is the one that actually pins production.
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import type { MaterialSpec, Scene, MeshPrimitive } from '@vitrum/core';
import { asMat4 } from '@vitrum/core';
import { buildSceneBVH } from '@vitrum/shared-bvh/legacy/three';
import { vitrumSceneToThree } from '@vitrum/three-bindings';
import { buildEmitterList } from '../emitterList.js';
import { collectRectAreaLightEmitterTris, collectRectAreaEmitterTrisFromCore } from '../bvhSceneHelpers.js';
import { buildReSTIRSceneBVHFromVitrumScene } from '../../legacy/three/restirSceneBvhFromCore.js';

const PRIMARY_LIGHT_DIR = new THREE.Vector3(0.3, -0.8, 0.5);
const PRIMARY_LIGHT_INTENSITY = 0.6;

// ──────────────────────────────────────────────────────────────────────────
// Core-Scene mesh helpers
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

// ──────────────────────────────────────────────────────────────────────────
// Emitter-list decode + canonical-set comparison
// ──────────────────────────────────────────────────────────────────────────

const EMITTER_FLOATS = 20; // 80-byte stride / 4

interface DecodedEmitter {
  vA: [number, number, number];
  vB: [number, number, number];
  vC: [number, number, number];
  normal: [number, number, number];
  area: number;
  color: [number, number, number];
  intensity: number;
  centroid: [number, number, number];
}

function decodeEmitters(emitterFloats: Float32Array): DecodedEmitter[] {
  const n = Math.floor(emitterFloats.length / EMITTER_FLOATS);
  const out: DecodedEmitter[] = [];
  for (let i = 0; i < n; i++) {
    const b = i * EMITTER_FLOATS;
    const vA: [number, number, number] = [emitterFloats[b]!, emitterFloats[b + 1]!, emitterFloats[b + 2]!];
    const vB: [number, number, number] = [emitterFloats[b + 4]!, emitterFloats[b + 5]!, emitterFloats[b + 6]!];
    const vC: [number, number, number] = [emitterFloats[b + 8]!, emitterFloats[b + 9]!, emitterFloats[b + 10]!];
    const normal: [number, number, number] = [emitterFloats[b + 12]!, emitterFloats[b + 13]!, emitterFloats[b + 14]!];
    const area = emitterFloats[b + 15]!;
    const color: [number, number, number] = [emitterFloats[b + 16]!, emitterFloats[b + 17]!, emitterFloats[b + 18]!];
    const intensity = emitterFloats[b + 19]!;
    const centroid: [number, number, number] = [
      (vA[0] + vB[0] + vC[0]) / 3,
      (vA[1] + vB[1] + vC[1]) / 3,
      (vA[2] + vB[2] + vC[2]) / 3,
    ];
    out.push({ vA, vB, vC, normal, area, color, intensity, centroid });
  }
  return out;
}

/** Canonical sort key: centroid ⊕ color, quantised so any last-ULP f32 drift
 *  between the two producers doesn't reorder the set. */
function emitterKey(e: DecodedEmitter): string {
  const q = (x: number): string => Math.round(x * 1e4).toString();
  return [q(e.centroid[0]), q(e.centroid[1]), q(e.centroid[2]), q(e.color[0]), q(e.color[1]), q(e.color[2])].join(',');
}

function sortCanonical(es: DecodedEmitter[]): DecodedEmitter[] {
  return [...es].sort((a, b) => emitterKey(a).localeCompare(emitterKey(b)));
}

/** Drop the synthetic placeholder emitter inserted when a scene has zero real
 *  emitters (vA=[0,10,0], color=[1,1,1], area=0.5). It is a GPU-non-empty-buffer
 *  sentinel, identical in both producers, present only when the real list is
 *  empty. */
function stripPlaceholder(es: DecodedEmitter[]): DecodedEmitter[] {
  if (es.length !== 1) return es;
  const e = es[0]!;
  const isPlaceholder =
    e.vA[0] === 0 && e.vA[1] === 10 && e.vA[2] === 0 &&
    e.color[0] === 1 && e.color[1] === 1 && e.color[2] === 1 && e.area === 0.5;
  return isPlaceholder ? [] : es;
}

function assertEmitterSetsEqual(threeFloats: Float32Array, coreFloats: Float32Array): void {
  const a = stripPlaceholder(decodeEmitters(threeFloats));
  const b = stripPlaceholder(decodeEmitters(coreFloats));
  expect(b.length).toBe(a.length);
  const as = sortCanonical(a);
  const bs = sortCanonical(b);
  for (let i = 0; i < as.length; i++) {
    const ea = as[i]!;
    const eb = bs[i]!;
    expect(emitterKey(eb)).toBe(emitterKey(ea));
    for (let k = 0; k < 3; k++) {
      expect(eb.vA[k]).toBeCloseTo(ea.vA[k]!, 4);
      expect(eb.vB[k]).toBeCloseTo(ea.vB[k]!, 4);
      expect(eb.vC[k]).toBeCloseTo(ea.vC[k]!, 4);
      expect(eb.normal[k]).toBeCloseTo(ea.normal[k]!, 4);
      expect(eb.color[k]).toBeCloseTo(ea.color[k]!, 4);
    }
    expect(eb.area).toBeCloseTo(ea.area, 4);
    expect(eb.intensity).toBeCloseTo(ea.intensity, 4);
  }
}

// ──────────────────────────────────────────────────────────────────────────
// Producers — both start from the SAME core Scene; the THREE side routes through
// vitrumSceneToThree (the production THREE emitter path), the core side through
// buildReSTIRSceneBVHFromVitrumScene (the production core-first path).
// ──────────────────────────────────────────────────────────────────────────

/** THREE reference: vitrumSceneToThree(scene) → buildSceneBVH → buildEmitterList. */
function threeEmitterFloats(scene: Scene): { emitterFloats: Float32Array; totalEmissivePower: number; cdfArray: Float32Array } {
  const root = vitrumSceneToThree(scene);
  const shared = buildSceneBVH([root], {
    positionStride: 4,
    filter: (obj: THREE.Object3D) => obj instanceof THREE.Mesh && (obj as THREE.InstancedMesh).isInstancedMesh !== true,
  });
  const extraEmitters = collectRectAreaLightEmitterTris([root]);
  return buildEmitterList(shared.indices, shared.positions, shared.normals, shared.triMaterialId, shared.materials, {
    primaryLightDir: PRIMARY_LIGHT_DIR,
    primaryLightIntensity: PRIMARY_LIGHT_INTENSITY,
    extraEmitters,
  });
}

/** Core path: the real production `buildReSTIRSceneBVHFromVitrumScene` (drives the
 *  emitter list from mergeWorldSpaceFromCore + buildEmitterListFromCore +
 *  toProductionEmissiveRadiance). Returns the packed emitter buffer it uploads. */
function coreEmitterFloats(scene: Scene): { emitterFloats: Float32Array; totalEmissivePower: number } {
  const root = vitrumSceneToThree(scene); // sceneRoots is a pure round-trip in production
  const buffers = buildReSTIRSceneBVHFromVitrumScene(scene, [root], {
    primaryLightDir: PRIMARY_LIGHT_DIR,
    primaryLightIntensity: PRIMARY_LIGHT_INTENSITY,
  });
  return {
    emitterFloats: new Float32Array(buffers.emitters.cpuData),
    totalEmissivePower: buffers.totalEmissivePower,
  };
}

// ──────────────────────────────────────────────────────────────────────────
// Scenes
// ──────────────────────────────────────────────────────────────────────────

function emissiveMat(emissive: [number, number, number], ei: number, base: [number, number, number] = [0.1, 0.1, 0.1]): MaterialSpec {
  return { baseColor: base, roughness: 1, metallic: 0, emissive, emissiveIntensity: ei };
}
function opaqueMat(base: [number, number, number]): MaterialSpec {
  return { baseColor: base, roughness: 1, metallic: 0 };
}
function glassMat(base: [number, number, number], transmission: number, atten: [number, number, number]): MaterialSpec {
  return { baseColor: base, roughness: 0.1, metallic: 0, transmission, attenuationColor: atten };
}

// ──────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────

describe('emitter list: production core path ≡ production THREE path — as a SET', () => {
  it('single emissive quad (emissiveIntensity ≠ 1 — the vitrumSceneToThree ei-collapse case)', () => {
    // ei=4 is exactly the case the GPU A/B caught: vitrumSceneToThree forces THREE
    // ei=1 → Le=emissive·1; the core path must reproduce that (NOT emissive·4).
    const scene: Scene = {
      primitives: [quad('lamp', [[-0.4, -0.4, 0.5], [0.4, -0.4, 0.5], [0.4, 0.4, 0.5], [-0.4, 0.4, 0.5]], [0, 0, -1], emissiveMat([1.0, 0.45, 0.15], 4, [0, 0, 0]))],
      emitters: [],
      environment: { kind: 'none' },
    };
    const t = threeEmitterFloats(scene);
    const c = coreEmitterFloats(scene);
    assertEmitterSetsEqual(t.emitterFloats, c.emitterFloats);
    expect(c.totalEmissivePower).toBeCloseTo(t.totalEmissivePower, 4);
    // Pin the actual Le: emissive·1 = [1,0.45,0.15] (NOT emissive·4).
    const e = stripPlaceholder(decodeEmitters(c.emitterFloats))[0]!;
    expect(e.color[0]).toBeCloseTo(1.0, 4);
    expect(e.color[1]).toBeCloseTo(0.45, 4);
    expect(e.color[2]).toBeCloseTo(0.15, 4);
  });

  it('full emissive Cornell (5 walls + emissive mesh + rect-area light) — the GPU A/B scene', () => {
    const primitives: MeshPrimitive[] = [
      quad('floor', [[-1, -1, -1], [1, -1, -1], [1, -1, 1], [-1, -1, 1]], [0, 1, 0], opaqueMat([0.8, 0.8, 0.8])),
      quad('ceiling', [[-1, 1, -1], [-1, 1, 1], [1, 1, 1], [1, 1, -1]], [0, -1, 0], opaqueMat([0.8, 0.8, 0.8])),
      quad('back-wall', [[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], [0, 0, -1], opaqueMat([0.8, 0.8, 0.8])),
      quad('left-wall', [[-1, -1, -1], [-1, -1, 1], [-1, 1, 1], [-1, 1, -1]], [1, 0, 0], opaqueMat([0.75, 0.1, 0.1])),
      quad('right-wall', [[1, -1, 1], [1, -1, -1], [1, 1, -1], [1, 1, 1]], [-1, 0, 0], opaqueMat([0.1, 0.6, 0.1])),
      quad('emitter', [[-0.4, -0.4, 0.5], [0.4, -0.4, 0.5], [0.4, 0.4, 0.5], [-0.4, 0.4, 0.5]], [0, 0, -1], emissiveMat([1.0, 0.45, 0.15], 4, [0, 0, 0])),
    ];
    const scene: Scene = {
      primitives,
      emitters: [{ kind: 'rect-area', id: 'ceiling-light', position: [0, 0.95, 0], uAxis: [0.2, 0, 0], vAxis: [0, 0, 0.2], color: [1, 1, 1], intensity: 12.0 }],
      environment: { kind: 'none' },
    };
    const t = threeEmitterFloats(scene);
    const c = coreEmitterFloats(scene);
    assertEmitterSetsEqual(t.emitterFloats, c.emitterFloats);
    expect(c.totalEmissivePower).toBeCloseTo(t.totalEmissivePower, 4);
  });

  it('multiple transformed emissive + opaque meshes — set equal up to ordering', () => {
    const scene: Scene = {
      primitives: [
        quad('lamp', [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], [0, 0, 1], emissiveMat([1.0, 0.9, 0.7], 6, [0.05, 0.05, 0.05]), translation(-0.8, 0.6, 0.2)),
        quad('wall', [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], [0, 0, 1], opaqueMat([0.7, 0.1, 0.1]), translation(0, 0, 1)),
        quad('lamp2', [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], [0, 0, 1], emissiveMat([0.2, 0.5, 1.0], 3), translation(0.9, -0.3, -0.4)),
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    const t = threeEmitterFloats(scene);
    const c = coreEmitterFloats(scene);
    assertEmitterSetsEqual(t.emitterFloats, c.emitterFloats);
    expect(c.totalEmissivePower).toBeCloseTo(t.totalEmissivePower, 4);
  });

  it('transmissive (sun-attenuated secondary) emitter — set + power parity', () => {
    // Tilt the glass so |dot(sun, normal)| > 0.05 (sun-eligible). Transmissive
    // branch never reads emissiveIntensity, so this validates the OTHER classifier arm.
    const tilt = new THREE.Matrix4().makeRotationX(Math.PI / 3);
    const scene: Scene = {
      primitives: [quad('glass', [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], [0, 0, 1], glassMat([0.95, 0.97, 1.0], 0.9, [0.6, 0.8, 0.95]), new Float32Array(tilt.elements))],
      emitters: [],
      environment: { kind: 'none' },
    };
    const t = threeEmitterFloats(scene);
    const c = coreEmitterFloats(scene);
    assertEmitterSetsEqual(t.emitterFloats, c.emitterFloats);
    expect(c.totalEmissivePower).toBeGreaterThan(0);
    expect(c.totalEmissivePower).toBeCloseTo(t.totalEmissivePower, 4);
  });
});

describe('rect-area extra emitters: collectRectAreaEmitterTrisFromCore ≡ THREE round-trip', () => {
  function assertRectTrisEqual(scene: Scene): void {
    const threeRoot = vitrumSceneToThree(scene);
    const threeTris = collectRectAreaLightEmitterTris([threeRoot]);
    const coreTris = collectRectAreaEmitterTrisFromCore(scene);
    expect(coreTris.length).toBe(threeTris.length);
    const key = (t: typeof coreTris[number]): string => {
      const cx = (t.vA[0] + t.vB[0] + t.vC[0]) / 3;
      const cy = (t.vA[1] + t.vB[1] + t.vC[1]) / 3;
      const cz = (t.vA[2] + t.vB[2] + t.vC[2]) / 3;
      return [cx, cy, cz].map((v) => Math.round(v * 1e4)).join(',');
    };
    const ts = [...threeTris].sort((a, b) => key(a).localeCompare(key(b)));
    const cs = [...coreTris].sort((a, b) => key(a).localeCompare(key(b)));
    for (let i = 0; i < ts.length; i++) {
      const a = ts[i]!;
      const b = cs[i]!;
      for (let k = 0; k < 3; k++) {
        expect(b.vA[k]).toBeCloseTo(a.vA[k]!, 4);
        expect(b.vB[k]).toBeCloseTo(a.vB[k]!, 4);
        expect(b.vC[k]).toBeCloseTo(a.vC[k]!, 4);
        expect(b.normal[k]).toBeCloseTo(a.normal[k]!, 4);
        expect(b.Le[k]).toBeCloseTo(a.Le[k]!, 4);
      }
      expect(b.area).toBeCloseTo(a.area, 4);
    }
  }

  it('axis-aligned ceiling rect-area light (Cornell-style)', () => {
    assertRectTrisEqual({
      primitives: [],
      emitters: [{ kind: 'rect-area', id: 'ceiling-light', position: [0, 0.95, 0], uAxis: [0.2, 0, 0], vAxis: [0, 0, 0.2], color: [1, 1, 1], intensity: 12.0 }],
      environment: { kind: 'none' },
    });
  });

  it('tilted, chromatic rect-area light (non-axis-aligned u/v)', () => {
    assertRectTrisEqual({
      primitives: [],
      emitters: [{ kind: 'rect-area', id: 'wall-light', position: [0.4, 0.3, -0.2], uAxis: [0.3, 0.1, 0.0], vAxis: [-0.05, 0.2, 0.15], color: [0.9, 0.4, 0.2], intensity: 7.5 }],
      environment: { kind: 'none' },
    });
  });
});

describe('documents WHY the converged-but-not-pixel-identical framing holds', () => {
  it('emitter ARRAY ORDER may differ even when the SET is identical', () => {
    // Two emitters with different powers; the producers MAY order them differently
    // (different SAH permutation) — the very reason a low-frame pixel A/B differs on
    // noise while the converged result matches. The contract is SET-equality, which
    // holds regardless of whether the order coincides.
    const scene: Scene = {
      primitives: [
        quad('lampA', [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], [0, 0, 1], emissiveMat([1.0, 0.0, 0.0], 8), translation(-0.7, 0.5, 0)),
        quad('lampB', [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], [0, 0, 1], emissiveMat([0.0, 0.0, 1.0], 2), translation(0.7, 0.5, 0)),
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    const t = threeEmitterFloats(scene);
    const c = coreEmitterFloats(scene);
    assertEmitterSetsEqual(t.emitterFloats, c.emitterFloats);
    expect(c.totalEmissivePower).toBeCloseTo(t.totalEmissivePower, 4);
  });
});
