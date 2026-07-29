/**
 * H34-h unit tests — SceneBvh.updateFromCore sceneVersionTag fast path
 *
 * Verifies:
 *   1. Same tag → merge skipped (mergeWorldSpaceFromCore not called again)
 *   2. Changed tag → merge runs
 *   3. No tag → fallback fingerprint path (existing behaviour unchanged)
 */

import { describe, expect, it } from 'vitest';
import type { MaterialSpec, Scene } from '@vitrum/core';
import { SceneBvh } from '../sceneBvh.js';

// We spy on mergeWorldSpaceFromCore via the module; vitest's module mocking
// requires the import to be in the same module scope. Instead, we count
// merge invocations indirectly: since the merge is the only place that can
// populate _buffers on a real SceneBvh, we use a subclass to instrument it.

function minimalScene(material: number | Partial<MaterialSpec> = 0.5): Scene {
  const materialOverrides = typeof material === 'number'
    ? { roughness: material }
    : material;
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'tri',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0, ...materialOverrides },
      },
    ],
    emitters: [],
    environment: { kind: 'none' },
  };
}

function equalLengthEditedVertexScene(y: number): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'tri',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, y, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 },
      },
    ],
    emitters: [],
    environment: { kind: 'none' },
  };
}

function f32FromBits(bits: number): number {
  const raw = new Uint32Array([bits >>> 0]);
  return new Float32Array(raw.buffer)[0] ?? 0;
}

function largeSceneWithUnsampledYByte(yValue: number = 0): Scene {
  const triangleCount = 2731;
  const vertexCount = triangleCount * 3;
  const positions = new Float32Array(vertexCount * 3);
  const normals = new Float32Array(vertexCount * 3);

  for (let t = 0; t < triangleCount; t += 1) {
    const v = t * 9;
    positions[v] = 0; positions[v + 1] = 0; positions[v + 2] = t;
    positions[v + 3] = 1; positions[v + 4] = 0; positions[v + 5] = t;
    positions[v + 6] = 0; positions[v + 7] = 1; positions[v + 8] = t;

    normals[v + 2] = 1;
    normals[v + 5] = 1;
    normals[v + 8] = 1;
  }

  // With positionStride:4, this lands in byte offset vertex*16+4. The chosen
  // subnormal f32 has byte pattern 00 01 00 00, so only the odd byte changes.
  // The old sampled fingerprint path (stride=2 for this buffer size) missed it.
  positions[9 * 3 + 1] = yValue;

  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'large-unsampled-byte',
        positions,
        normals,
        material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 },
      },
    ],
    emitters: [],
    environment: { kind: 'none' },
  };
}

/**
 * Instrumented subclass: counts how many times the expensive work (the merge)
 * was actually invoked by tracking `_lastCoreFingerprint` transitions.
 *
 * We proxy the protected `_buffers` setter to count rebuilds — a rebuild
 * always sets _buffers to a non-null value (for non-empty scenes).  Using the
 * `updateFromCore` public API we can verify skip vs run by checking whether
 * the returned buffers object reference is the SAME (skip) or a new one (run).
 */
describe('H34-h: SceneBvh sceneVersionTag fast path', () => {
  it('same tag → merge skipped, buffers reference unchanged', () => {
    const bvh = new SceneBvh();
    const scene = minimalScene();

    bvh.updateFromCore(scene, { sceneVersionTag: 1 });
    const first = bvh.buffers;
    expect(first).not.toBeNull();

    bvh.updateFromCore(scene, { sceneVersionTag: 1 });
    const second = bvh.buffers;

    // Same reference — merge was skipped entirely.
    expect(second).toBe(first);
  });

  it('changed tag → merge runs, buffers reference updated', () => {
    const bvh = new SceneBvh();
    const scene1 = minimalScene(0.5);
    const scene2 = minimalScene(0.8); // different material → different fingerprint

    bvh.updateFromCore(scene1, { sceneVersionTag: 1 });
    const first = bvh.buffers;
    expect(first).not.toBeNull();

    bvh.updateFromCore(scene2, { sceneVersionTag: 2 });
    const second = bvh.buffers;

    // Different reference — merge ran.
    expect(second).not.toBe(first);
    expect(second).not.toBeNull();
  });

  it('same tag, different scene → merge still skipped (tag wins over content)', () => {
    const bvh = new SceneBvh();
    bvh.updateFromCore(minimalScene(0.5), { sceneVersionTag: 'v42' });
    const first = bvh.buffers;

    // Different scene but SAME tag — the contract says "tag unchanged → skip".
    bvh.updateFromCore(minimalScene(0.9), { sceneVersionTag: 'v42' });
    expect(bvh.buffers).toBe(first);
  });

  it('same tag skips the merge after an empty-scene publication', () => {
    const bvh = new SceneBvh();
    const empty: Scene = {
      primitives: [],
      emitters: [],
      environment: { kind: 'none' },
    };
    bvh.updateFromCore(empty, { sceneVersionTag: 'empty-v1' });
    expect(bvh.buffers).toBeNull();

    const poisonedScene = {
      get primitives(): never {
        throw new Error('same-tag empty scene must not be traversed again');
      },
      emitters: [],
      environment: { kind: 'none' },
    } as unknown as Scene;
    expect(() => {
      bvh.updateFromCore(poisonedScene, { sceneVersionTag: 'empty-v1' });
    }).not.toThrow();
    expect(bvh.buffers).toBeNull();
  });

  it('no tag → fingerprint fallback, identical scene skipped', () => {
    const bvh = new SceneBvh();
    const scene = minimalScene();

    bvh.updateFromCore(scene);
    const first = bvh.buffers;
    expect(first).not.toBeNull();

    // Same geometry + materials → fingerprint match → skip.
    bvh.updateFromCore(scene);
    expect(bvh.buffers).toBe(first);
  });

  it('no tag → fingerprint fallback, changed scene triggers rebuild', () => {
    const bvh = new SceneBvh();
    bvh.updateFromCore(minimalScene(0.5));
    const first = bvh.buffers;

    bvh.updateFromCore(minimalScene(0.9));
    // Fingerprint differs → rebuild.
    expect(bvh.buffers).not.toBe(first);
  });

  it('H24: no tag → equal-length vertex edit triggers rebuild', () => {
    const bvh = new SceneBvh();
    bvh.updateFromCore(equalLengthEditedVertexScene(1));
    const first = bvh.buffers;
    expect(first).not.toBeNull();

    bvh.updateFromCore(equalLengthEditedVertexScene(0.25));

    expect(bvh.buffers).not.toBe(first);
    expect(bvh.buffers?.positions[9]).toBeCloseTo(0.25, 6);
  });

  it('no tag → large unsampled-byte geometry edit triggers rebuild', () => {
    const bvh = new SceneBvh({ onSlowRebuild: () => undefined });
    bvh.updateFromCore(largeSceneWithUnsampledYByte(0));
    const first = bvh.buffers;
    expect(first).not.toBeNull();

    bvh.updateFromCore(largeSceneWithUnsampledYByte(f32FromBits(0x00000100)));

    expect(bvh.buffers).not.toBe(first);
  });

  it('H33: no tag → attenuationDistance-only material edit triggers rebuild', () => {
    const bvh = new SceneBvh();
    bvh.updateFromCore(minimalScene({
      transmission: 1,
      attenuationColor: [0.5, 0.5, 0.5],
      attenuationDistance: 1,
      thickness: 0.25,
    }));
    const first = bvh.buffers;
    expect(first).not.toBeNull();

    bvh.updateFromCore(minimalScene({
      transmission: 1,
      attenuationColor: [0.5, 0.5, 0.5],
      attenuationDistance: 10,
      thickness: 0.25,
    }));

    expect(bvh.buffers).not.toBe(first);
  });

  it('H33: no tag → thickness-only material edit triggers rebuild', () => {
    const bvh = new SceneBvh();
    bvh.updateFromCore(minimalScene({
      transmission: 1,
      attenuationColor: [0.5, 0.5, 0.5],
      attenuationDistance: 3,
      thickness: 0.25,
    }));
    const first = bvh.buffers;
    expect(first).not.toBeNull();

    bvh.updateFromCore(minimalScene({
      transmission: 1,
      attenuationColor: [0.5, 0.5, 0.5],
      attenuationDistance: 3,
      thickness: 0.75,
    }));

    expect(bvh.buffers).not.toBe(first);
  });

  it('dispose clears tag so next call always rebuilds', () => {
    const bvh = new SceneBvh();
    const scene = minimalScene();

    bvh.updateFromCore(scene, { sceneVersionTag: 7 });
    const first = bvh.buffers;
    bvh.dispose();
    expect(bvh.buffers).toBeNull();

    // After dispose, the tag was cleared. Same tag+scene → full rebuild.
    bvh.updateFromCore(scene, { sceneVersionTag: 7 });
    const second = bvh.buffers;
    expect(second).not.toBeNull();
    // Reference is a new object (rebuilt), though it may be structurally equal.
    // The key invariant: buffers are not null after a fresh rebuild post-dispose.
    expect(second).not.toBe(first);
  });

  it('string tags work identically to number tags', () => {
    const bvh = new SceneBvh();
    const scene1 = minimalScene(0.3);
    const scene2 = minimalScene(0.7); // different content

    bvh.updateFromCore(scene1, { sceneVersionTag: 'scene-v1' });
    const first = bvh.buffers;

    // Same tag → skip even though we pass a different scene object.
    bvh.updateFromCore(scene2, { sceneVersionTag: 'scene-v1' });
    expect(bvh.buffers).toBe(first);

    // Changed tag + different content → merge runs → new buffers object.
    bvh.updateFromCore(scene2, { sceneVersionTag: 'scene-v2' });
    expect(bvh.buffers).not.toBe(first);
  });
});
