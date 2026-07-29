/**
 * Pins that the ReSTIR emitter list is built in WORLD space from the canonical
 * `@vitrum/core` `MeshPrimitive.transform`.
 *
 * Why the invariant still matters: `packSceneFromCore` (`geo`) stores per-
 * primitive BLAS positions in LOCAL/object space (the world transform lives in
 * the TLAS instance matrices), so the emitter list must NOT be sourced from
 * `geo` — it needs world-space geometry (triangle area, world face normal, the
 * world-space sun dot, world centroids/AABBs, world rect-area tris). These tests
 * assert the emitter geometry is WORLD-space (driven by `primitive.transform`)
 * and DIVERGES from `geo`'s local-space pack, so any future "just use geo"
 * reroute breaks here.
 */

import { describe, expect, it } from 'vitest';
import type { MaterialSpec, Scene } from '@vitrum/core';
import { asMat4 } from '@vitrum/core';
import { packSceneFromCore } from '@vitrum/shared-bvh';
import { buildReSTIRSceneBVHForCoreScene } from '../src/restir/bvhCore.js';
import { buildEmitterListFromCore } from '../src/restir/emitterList.js';

const EMITTER_FLOATS = 20; // 80-byte EmitterTri stride / 4

/** Column-major 4×4 pure-translation matrix. */
function translation(x: number, y: number, z: number): Float32Array {
  return new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    x, y, z, 1,
  ]);
}

/**
 * Single emissive triangle whose local vertices sit at the origin. The world
 * placement is carried by the core `transform` (post-decouple: the authoritative
 * world transform), so LOCAL and WORLD space diverge whenever `transform` is
 * non-identity.
 */
function emissiveTriScene(transform?: Float32Array): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'lamp',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        indices: new Uint32Array([0, 1, 2]),
        material: {
          baseColor: [1, 1, 1],
          roughness: 0.5,
          metallic: 0,
          emissive: [1, 1, 1],
          emissiveIntensity: 5,
        },
        ...(transform ? { transform: asMat4(transform) } : {}),
      },
    ],
    emitters: [],
    environment: { kind: 'none' },
  };
}

function firstEmitterVertexA(buffers: { emitters: { cpuData: ArrayBuffer } }): [number, number, number] {
  const floats = new Float32Array(buffers.emitters.cpuData);
  return [floats[0]!, floats[1]!, floats[2]!];
}

describe('emitter list is built in world space (driven by core transform)', () => {
  it('emitter triangle vertex follows the core primitive transform, not local space', () => {
    // World translation carried by the core transform (the production source of
    // truth for emitter-list world geometry).
    const scene = emissiveTriScene(translation(10, 20, 30));
    const buffers = buildReSTIRSceneBVHForCoreScene(scene, { bvhMode: 'tlas' });

    // Exactly one emissive triangle ⇒ exactly one emitter (no synthetic
    // placeholder, no extra rect-area-light tris).
    expect(buffers.emitterCount).toBe(1);

    // vertexA in the emitter list = LOCAL (0,0,0) shifted by the world transform
    // → (10,20,30). If the emitter list were sourced from `geo` (local space) it
    // would read (0,0,0) and this assertion fails.
    const [ax, ay, az] = firstEmitterVertexA(buffers);
    expect(ax).toBeCloseTo(10, 5);
    expect(ay).toBeCloseTo(20, 5);
    expect(az).toBeCloseTo(30, 5);
  });

  it('geo (packSceneFromCore) carries LOCAL-space positions — proves the divergence', () => {
    const scene = emissiveTriScene(translation(10, 20, 30));
    // geo positions are local/object space regardless of the world transform
    // (the transform lives in the TLAS instance matrices, not the BLAS verts).
    const geo = packSceneFromCore(scene, { tlas: true, resolveMaterialId: () => 0 });
    // First vertex is the local (0,0,0) — NOT the world (10,20,30) the emitter
    // list above resolved to. Same primitive, two different spaces.
    expect(geo.positions[0]).toBeCloseTo(0, 6);
    expect(geo.positions[1]).toBeCloseTo(0, 6);
    expect(geo.positions[2]).toBeCloseTo(0, 6);
  });

  it('identity transform: emitter vertex matches local (sanity baseline)', () => {
    const scene = emissiveTriScene(); // no transform → identity
    const buffers = buildReSTIRSceneBVHForCoreScene(scene, { bvhMode: 'tlas' });
    expect(buffers.emitterCount).toBe(1);
    const [ax, ay, az] = firstEmitterVertexA(buffers);
    expect(ax).toBeCloseTo(0, 5);
    expect(ay).toBeCloseTo(0, 5);
    expect(az).toBeCloseTo(0, 5);
  });
});

describe('emitter source-index fallback', () => {
  it('retains castShadow:false when an invalid scalar source index falls back to packed radiance', () => {
    const material = {
      baseColor: [1, 1, 1],
      roughness: 0.5,
      metallic: 0,
      emissive: [1, 1, 1],
      emissiveIntensity: 5,
      castShadow: false,
    } as unknown as MaterialSpec;
    const result = buildEmitterListFromCore(
      new Uint32Array([0, 1, 2]),
      new Float32Array([
        0, 0, 0, 0,
        1, 0, 0, 0,
        0, 1, 0, 0,
      ]),
      new Float32Array([
        0, 0, 1, 0,
        0, 0, 1, 0,
        0, 0, 1, 0,
      ]),
      new Uint32Array([0]),
      [material],
      {
        packSourceTriIndex: true,
        sourceTriIndexForTriangle: () => 1.5,
      },
    );

    expect(result.emitterFloats[3]).toBe(-1);
    expect(result.emitterFloats[19]).toBe(1);
    expect(Array.from(result.emitterFloats.slice(16, 19))).toEqual([5, 5, 5]);
  });
});

// Reference the stride constant so it documents the EmitterTri layout the
// decode above relies on (vertexA occupies floats 0..2 of the 20-float entry).
void EMITTER_FLOATS;
