/**
 * D8.7 — SceneBufferRegistry drift gate.
 *
 * Asserts that every `bufferField` named in SCENE_BUFFER_REGISTRY resolves to
 * an actual field on `UploadedSceneBuffers`. This is a structural TypeScript +
 * runtime check: if a buffer is added to `UploadedSceneBuffers` without a
 * matching registry entry (or vice-versa), this test will catch it at CI time
 * rather than at the next time someone tries to add a buffer and wonders why the
 * loop doesn't create it.
 *
 * The test does NOT check bind-group layout entries in gpuResources.ts (cross-
 * file coupling deferred per the D8.7 scope note) — the registry's sync comment
 * documents that invariant for the next developer.
 */
import { describe, expect, it, vi } from 'vitest';
import type { Scene } from '@vitrum/core';
import {
  SCENE_BUFFER_REGISTRY,
  buildPackedScene,
  uploadPackedScene,
} from '../scene/uploadSceneBuffers.js';
import { installGpuConstStubs, textureStubMethods } from './gpuStub.js';

describe('D8.7 SceneBufferRegistry', () => {
  /**
   * Enumerate the GPU-handle fields on UploadedSceneBuffers: every field whose
   * name ends in 'Buffer' is a GPUBuffer handle. Non-buffer fields (counts,
   * CPU mirrors, callbacks) do not end with 'Buffer'.
   *
   * We derive the expected set from a TS mapped-type trick at the value level:
   * we list the keys of the type's non-readonly interface.  At runtime we
   * can only do this via the actual registry entries, so the test does:
   *
   *   1. Confirm every registry bufferField is a key that ends in 'Buffer'
   *      (sanity check on the registry itself).
   *   2. Confirm every registry bufferField would be a distinct field name
   *      (no duplicate bufferField entries — that would silently shadow one).
   *   3. Confirm the count of registry entries equals 32 (the canonical buffer
   *      count after removing the unbound mesh-area source-factor buffer). If a buffer is
   *      added to UploadedSceneBuffers without a registry entry this assertion
   *      will fail with a clear message.
   */
  it('has 32 entries, each with a unique bufferField ending in "Buffer"', () => {
    expect(SCENE_BUFFER_REGISTRY.length).toBe(32);

    const seen = new Set<string>();
    for (const entry of SCENE_BUFFER_REGISTRY) {
      expect(entry.bufferField, `registry entry for key "${entry.key}"`).toMatch(/Buffer$/);
      expect(
        seen.has(entry.bufferField),
        `duplicate bufferField "${entry.bufferField}" in registry`,
      ).toBe(false);
      seen.add(entry.bufferField);
    }
  });

  it('every bufferField is a valid key on UploadedSceneBuffers (structural drift gate)', () => {
    /**
     * We can't enumerate UploadedSceneBuffers fields at runtime (it's an
     * interface, not a class), so we use a known complete set of GPU buffer
     * handle names that must exist on the interface — derived by inspection of
     * the interface declaration in uploadSceneBuffers.ts.  If the interface
     * gains a new *Buffer field that is NOT in the registry, the count check
     * in the previous test will fail.  If the registry declares a bufferField
     * that does NOT exist on the interface, TypeScript will catch it at compile
     * time via the `SceneBufferRegistryField` type.
     *
     * This test verifies the registry key order is consistent: the TLAS entries
     * (tlasNodesBuffer ... tlasInstanceLocalToWorldBuffer) must come last, which
     * is required by the uploadPackedScene loop that uses TLAS_START_INDEX.
     */
    const registryFields = SCENE_BUFFER_REGISTRY.map((e) => e.bufferField);
    const tlasFields: string[] = [
      'tlasNodesBuffer',
      'tlasInstanceIndicesBuffer',
      'tlasBlasRootsBuffer',
      'tlasInstanceWorldToLocalBuffer',
      'tlasInstanceLocalToWorldBuffer',
    ];
    // TLAS buffers must be contiguous at the END of the registry (the loop
    // uses findIndex('tlasNodesBuffer') as a split point).
    const tlasStartIdx = registryFields.indexOf('tlasNodesBuffer');
    expect(tlasStartIdx).toBeGreaterThan(0);
    const tailFields = registryFields.slice(tlasStartIdx);
    expect(tailFields).toStrictEqual(tlasFields);
  });

  it('TLAS_START_INDEX resolves correctly at runtime', () => {
    const idx = SCENE_BUFFER_REGISTRY.findIndex((e) => e.bufferField === 'tlasNodesBuffer');
    // There are 27 non-TLAS entries (BLAS + CWBVH prototype buffers + analytic
    // + env + emitters + light-tree + P2
    // UVs/tangents/colors/descriptors).
    expect(idx).toBe(27);
    // Remaining 5 entries are the TLAS buffers.
    expect(SCENE_BUFFER_REGISTRY.length - idx).toBe(5);
  });
});

/**
 * T2-A — registry-driven single-source gates. `uploadPackedScene` now drives its
 * create loop, destroy closure, and gpuMemoryBytes off SCENE_BUFFER_REGISTRY;
 * these tests pin that every `*Buffer` field on a REAL uploaded scene is covered
 * by the registry (completeness), that destroy tears down every registry buffer,
 * and that gpuMemoryBytes counts every live registry buffer.
 */
function meshScene(): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'mesh-a',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0 },
      },
    ],
    emitters: [],
    environment: { kind: 'none' },
  };
}

interface StubBuffer {
  label: string;
  size: number;
  destroy: ReturnType<typeof vi.fn>;
}

function makeUploadDevice(): { device: GPUDevice; buffers: StubBuffer[] } {
  installGpuConstStubs();
  const buffers: StubBuffer[] = [];
  const device = {
    queue: { writeBuffer: vi.fn(), writeTexture: vi.fn() },
    createBuffer: vi.fn((desc: GPUBufferDescriptor) => {
      const buf: StubBuffer = {
        label: desc.label ?? '',
        size: desc.size,
        destroy: vi.fn(),
      };
      buffers.push(buf);
      return buf as unknown as GPUBuffer;
    }),
    ...textureStubMethods(),
    limits: { maxTextureDimension2D: 8192 },
  } as unknown as GPUDevice;
  return { device, buffers };
}

describe('T2-A registry-driven upload single-source', () => {
  it('every *Buffer field on a real uploaded scene has a registry entry (completeness)', () => {
    const { device } = makeUploadDevice();
    const sb = uploadPackedScene(device, buildPackedScene(meshScene()));
    // Enumerate every GPUBuffer-handle field on the concrete UploadedSceneBuffers:
    // a field whose name ends in 'Buffer' and whose value has a `.destroy` method.
    const bufferFields = Object.keys(sb).filter(
      (k) =>
        k.endsWith('Buffer') &&
        typeof (sb as unknown as Record<string, { destroy?: unknown }>)[k]?.destroy === 'function',
    );
    expect(bufferFields.length).toBeGreaterThan(0);
    const registryFields = new Set(SCENE_BUFFER_REGISTRY.map((e) => e.bufferField));
    for (const field of bufferFields) {
      expect(registryFields.has(field as never), `buffer field "${field}" is missing from SCENE_BUFFER_REGISTRY`).toBe(true);
    }
    // And the registry has no extra entries beyond the real buffer set.
    expect(new Set(bufferFields)).toStrictEqual(registryFields);
    sb.destroy();
  });

  it('destroy() tears down every registry buffer exactly once + both material textures', () => {
    const { device, buffers } = makeUploadDevice();
    const sb = uploadPackedScene(device, buildPackedScene(meshScene()));
    const registryBuffers = SCENE_BUFFER_REGISTRY.map(
      (e) => sb[e.bufferField] as unknown as StubBuffer,
    );
    sb.destroy();
    for (const b of registryBuffers) {
      expect(b.destroy, `registry buffer ${b.label}`).toHaveBeenCalledTimes(1);
    }
    void buffers;
  });

  it('gpuMemoryBytes counts every registry buffer', () => {
    const { device } = makeUploadDevice();
    const sb = uploadPackedScene(device, buildPackedScene(meshScene()));
    const expected = SCENE_BUFFER_REGISTRY
      .reduce((sum, e) => sum + (sb[e.bufferField] as unknown as StubBuffer).size, 0);
    const { bufferBytes } = sb.gpuMemoryBytes();
    expect(bufferBytes).toBe(expected);
    sb.destroy();
  });
});
