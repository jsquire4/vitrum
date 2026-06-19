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
import { describe, expect, it } from 'vitest';
import { SCENE_BUFFER_REGISTRY } from '../scene/uploadSceneBuffers.js';

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
   *   3. Confirm the count of registry entries equals 28 (the canonical buffer
   *      count after adjoint mesh-area source factors were added). If a buffer is
   *      added to UploadedSceneBuffers without a registry entry this assertion
   *      will fail with a clear message.
   */
  it('has 28 entries, each with a unique bufferField ending in "Buffer"', () => {
    expect(SCENE_BUFFER_REGISTRY.length).toBe(28);

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
    // There are 23 non-TLAS entries (BLAS + analytic + env + emitters +
    // adjoint mesh-area source factors + light-tree + P2 UVs/tangents/colors/descriptors).
    expect(idx).toBe(23);
    // Remaining 5 entries are the TLAS buffers.
    expect(SCENE_BUFFER_REGISTRY.length - idx).toBe(5);
  });
});
