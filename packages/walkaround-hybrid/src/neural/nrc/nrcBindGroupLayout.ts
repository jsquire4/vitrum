/**
 * NRC @group(4) bind group layout factory — NRC subsystem owns its own layout.
 *
 * Moved from `pipeline/bindGroupLayouts.ts` (I5.2, R6 E sweep, 2026-06-11) so
 * the NRC subsystem (neural/nrc/) does not need to reach into the pipeline layer
 * for a pure-data layout definition. `pipeline/bindGroupLayouts.ts` re-exports
 * `getNrcBindGroupLayout` from here for back-compatibility.
 *
 * NRC (Müller et al. 2021) gi-ris @group(4) bind group layout. Present ONLY on
 * the gi-ris pipeline when `nrcEnabled` is compile-time on; the default gi-ris
 * pipeline (4 groups) never references it, so the default pipeline structure is
 * byte-for-byte pre-NRC (the GRIS-class regression discipline — f8df9a4).
 *
 *   0 — MLP weights      (read-only storage, f32) — concatenated weight matrices
 *   1 — MLP biases       (read-only storage, f32)
 *   2 — hash-grid tables (read-only storage, f32) — trainable feature tables
 *   3 — level descriptors (read-only storage, NrcLevelDesc)
 *   4 — record gather     (read_write storage, f32) — self-training records
 *   5 — encoding config   (uniform, NrcCfgUBO)
 *   6 — slot claim flags  (read_write storage, atomic<u32>) — torn-record guard
 *
 * Storage-buffer budget on the gi-ris pipeline (NRC ON): gi-ris binds the
 * risGi-frame storage buffer, the shared scene group, the hybrid-layer group,
 * and this NRC group. The NRC group adds 6 storage buffers, so the opt-in
 * capability gate requires maxStorageBuffersPerShaderStage >= 22 before
 * compilePipelines reaches createComputePipeline. The default NRC-OFF path never
 * references this group.
 */

import type { BGLCache } from '../../bglTypes.js';

export function getNrcBindGroupLayout(device: GPUDevice, cache: BGLCache): GPUBindGroupLayout {
  if (cache.nrc) return cache.nrc;
  cache.nrc = device.createBindGroupLayout({
    label: 'nrc-bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      // H27 — per-slot atomic claim flags (one u32 per recordCap slot). The
      // host clears this buffer to zero each frame. The GPU shader uses
      // atomicCompareExchangeWeak to claim a slot before writing the record,
      // preventing torn records when two invocations alias to the same slot.
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });
  return cache.nrc;
}
