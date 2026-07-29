/**
 * BVH / TLAS GPU buffer rebuild helpers for {@link ProbeUpdatePass} (W4c).
 */

import type { SceneBvhBuffers } from '@vitrum/shared-bvh';
import type { RestirBvhSnapshot } from '../restir/restirBvhSnapshot.js';
import { padTriangleIndicesToVec4 } from './probeUpdateMaterials.js';

const RO = 0x80 | 0x08; // STORAGE | COPY_DST — literal for Node vitest import chain
const COPY_STAGING = 0x04 | 0x08; // COPY_SRC | COPY_DST
const BVH_NODE_PLACEHOLDER_BYTES = 32;
const STORAGE_PLACEHOLDER_BYTES = 16;

export interface ProbeUpdateBvhGpuBuffers {
  bvhBuf: GPUBuffer;
  posBuf: GPUBuffer;
  idxBuf: GPUBuffer;
  normBuf: GPUBuffer;
  matIdBuf: GPUBuffer;
  tlasNodesBuf: GPUBuffer;
  tlasInstIdxBuf: GPUBuffer;
  tlasBlasRootsBuf: GPUBuffer;
  tlasW2lBuf: GPUBuffer;
  tlasL2wBuf: GPUBuffer;
}

type ProbeBvhBufferKey = keyof ProbeUpdateBvhGpuBuffers;
type ProbeCpuBufferData = ArrayBufferLike | ArrayBufferView;

interface GpuWriteSpan {
  readonly buffer: ArrayBuffer;
  readonly byteOffset: number;
  readonly byteLength: number;
}

/**
 * WebGPU does not accept SharedArrayBuffer-backed BufferSource values. Preserve
 * zero-copy uploads for ordinary ArrayBuffers, but copy the exact visible span
 * for SABs (and preserve typed-array byte offsets in both cases).
 */
function gpuWriteSpan(data: ProbeCpuBufferData): GpuWriteSpan {
  const backing = ArrayBuffer.isView(data) ? data.buffer : data;
  const byteOffset = ArrayBuffer.isView(data) ? data.byteOffset : 0;
  const byteLength = data.byteLength;
  if (backing instanceof ArrayBuffer) {
    return { buffer: backing, byteOffset, byteLength };
  }
  const copy = new Uint8Array(byteLength);
  copy.set(new Uint8Array(backing, byteOffset, byteLength));
  return { buffer: copy.buffer, byteOffset: 0, byteLength };
}

function destroyBuffersBestEffort(
  buffers: Iterable<GPUBuffer>,
  preserved: ReadonlySet<GPUBuffer> = new Set(),
): void {
  const destroyed = new Set<GPUBuffer>(preserved);
  for (const buffer of buffers) {
    if (destroyed.has(buffer)) continue;
    destroyed.add(buffer);
    try { buffer.destroy(); } catch { /* preserve the transaction outcome */ }
  }
}

function rebuildProbeBvhBuffers(
  device: GPUDevice,
  g: ProbeUpdateBvhGpuBuffers,
  entries: readonly (readonly [ProbeBvhBufferKey, ProbeCpuBufferData, number])[],
): void {
  const previous = new Map<ProbeBvhBufferKey, GPUBuffer>(
    entries.map(([key]) => [key, g[key]]),
  );
  const previousSet = new Set(previous.values());
  const candidates = new Map<ProbeBvhBufferKey, GPUBuffer>();
  const candidateSet = new Set<GPUBuffer>();
  try {
    for (const [key, data, minimumSize] of entries) {
      const candidate = device.createBuffer({
        label: `ddgi.${String(key)}.candidate`,
        size: Math.max(data.byteLength, minimumSize),
        usage: RO,
      });
      if (previousSet.has(candidate) || candidateSet.has(candidate)) {
        throw new Error(`DDGI BVH candidate ${String(key)} aliases a live resource.`);
      }
      candidates.set(key, candidate);
      candidateSet.add(candidate);
      if (data.byteLength > 0) {
        const source = gpuWriteSpan(data);
        device.queue.writeBuffer(
          candidate,
          0,
          source.buffer,
          source.byteOffset,
          source.byteLength,
        );
      }
    }
  } catch (error) {
    destroyBuffersBestEffort(candidateSet, previousSet);
    throw error;
  }

  // All writes have succeeded; publish the ten-buffer cohort without a
  // fallible operation between fields, then retire the old cohort.
  for (const [key] of entries) g[key] = candidates.get(key)!;
  destroyBuffersBestEffort(previousSet, candidateSet);
}

/**
 * C2 — TLAS transform refit.
 *
 * Refresh the complete five-stream traversal cohort. Even a transform-only
 * producer update fingerprints all five streams, and the TLAS builder is free
 * to change its leaf permutation or BLAS-root table while refitting. Capacity
 * growth is published only after one command buffer containing every upload is
 * accepted; same-capacity streams retain their public GPUBuffer identities.
 */
export function refitProbeTlasBuffersInPlace(
  device: GPUDevice,
  g: ProbeUpdateBvhGpuBuffers,
  tlas: NonNullable<RestirBvhSnapshot['tlas']>,
): void {
  const entries = [
    ['tlasNodesBuf', tlas.nodes, BVH_NODE_PLACEHOLDER_BYTES],
    ['tlasInstIdxBuf', tlas.instanceIndices, STORAGE_PLACEHOLDER_BYTES],
    ['tlasBlasRootsBuf', tlas.blasRoots, STORAGE_PLACEHOLDER_BYTES],
    ['tlasW2lBuf', tlas.worldToLocal, STORAGE_PLACEHOLDER_BYTES],
    ['tlasL2wBuf', tlas.localToWorld, STORAGE_PLACEHOLDER_BYTES],
  ] as const satisfies readonly (readonly [ProbeBvhBufferKey, ProbeCpuBufferData, number])[];
  const liveSet = new Set<GPUBuffer>(Object.values(g));
  const replacements = new Map<ProbeBvhBufferKey, GPUBuffer>();
  const destinations = new Map<ProbeBvhBufferKey, GPUBuffer>();
  const staging = new Map<ProbeBvhBufferKey, GPUBuffer>();
  const candidateSet = new Set<GPUBuffer>();
  let submitted = false;

  try {
    // A capacity increase gets a private destination. Same-capacity refits keep
    // their stable public buffers, but still upload through private staging.
    for (const [key, data, minimumSize] of entries) {
      const live = g[key];
      if (live.size >= data.byteLength) {
        destinations.set(key, live);
        continue;
      }
      const replacement = device.createBuffer({
        label: `ddgi.${String(key)}.refit-destination`,
        size: Math.max(data.byteLength, minimumSize),
        usage: RO,
      });
      if (liveSet.has(replacement) || candidateSet.has(replacement)) {
        throw new Error(`DDGI TLAS destination ${String(key)} aliases a live resource.`);
      }
      replacements.set(key, replacement);
      destinations.set(key, replacement);
      candidateSet.add(replacement);
    }

    // Queue writes target staging only. A failure at any write therefore leaves
    // every live TLAS buffer byte-for-byte untouched.
    for (const [key, data] of entries) {
      const stage = device.createBuffer({
        label: `ddgi.${String(key)}.refit-staging`,
        size: Math.max(data.byteLength, 4),
        usage: COPY_STAGING,
      });
      if (liveSet.has(stage) || candidateSet.has(stage)) {
        throw new Error(`DDGI TLAS staging ${String(key)} aliases another resource.`);
      }
      staging.set(key, stage);
      candidateSet.add(stage);
      if (data.byteLength > 0) {
        const source = gpuWriteSpan(data);
        device.queue.writeBuffer(
          stage,
          0,
          source.buffer,
          source.byteOffset,
          source.byteLength,
        );
      }
    }

    // All five staging→destination copies enter the queue in one command
    // buffer. Encoder/finish/submit failure cannot expose a mixed TLAS
    // generation, and public replacement identities are published only after
    // the queue accepts that complete command buffer.
    const encoder = device.createCommandEncoder({ label: 'ddgi.tlas-refit.transaction' });
    for (const [key, data] of entries) {
      if (data.byteLength === 0) continue;
      encoder.copyBufferToBuffer(
        staging.get(key)!,
        0,
        destinations.get(key)!,
        0,
        data.byteLength,
      );
    }
    const commandBuffer = encoder.finish();
    device.queue.submit([commandBuffer]);
    submitted = true;
  } catch (error) {
    destroyBuffersBestEffort(candidateSet, liveSet);
    throw error;
  }

  const retiredDestinations = new Set<GPUBuffer>();
  for (const [key, replacement] of replacements) {
    retiredDestinations.add(g[key]);
    g[key] = replacement;
  }
  // Retire as one cohort so aliased prior fields are destroyed at most once.
  // Preserve any prior buffer that another, non-replaced public field still
  // references; a later replacement will retire it when its final alias leaves.
  const publishedSet = new Set<GPUBuffer>(Object.values(g));
  destroyBuffersBestEffort(
    retiredDestinations,
    new Set<GPUBuffer>([...candidateSet, ...publishedSet]),
  );

  const stagingSet = new Set(staging.values());
  const retireStaging = (): void => destroyBuffersBestEffort(stagingSet, liveSet);
  if (submitted && typeof device.queue.onSubmittedWorkDone === 'function') {
    try {
      void device.queue.onSubmittedWorkDone().then(retireStaging, retireStaging);
    } catch {
      // Submission was already accepted. WebGPU defers the physical release of
      // buffers referenced by queued work, so destroy is both safe here and
      // necessary: a synchronous completion-tracker failure must not leak all
      // five private staging buffers.
      retireStaging();
    }
  } else {
    retireStaging();
  }
}

export function rebuildProbeBvhFromRestir(
  device: GPUDevice,
  g: ProbeUpdateBvhGpuBuffers,
  snap: RestirBvhSnapshot,
): void {
  // No-TLAS snapshots still bind the declared TLAS storage arrays. `tlasNodes`
  // is `array<BVHNode>` (32-byte stride), so strict WebGPU backends reject the
  // old generic 16-byte dummy at bind-group creation even when bvhMode=merged.
  const emptyTlasNodes = new ArrayBuffer(BVH_NODE_PLACEHOLDER_BYTES);
  const empty = new ArrayBuffer(STORAGE_PLACEHOLDER_BYTES);
  const tlas = snap.tlas;
  rebuildProbeBvhBuffers(device, g, [
    ['bvhBuf', snap.bvhNodes, BVH_NODE_PLACEHOLDER_BYTES],
    ['posBuf', snap.positions, STORAGE_PLACEHOLDER_BYTES],
    ['idxBuf', snap.bvhIndex, STORAGE_PLACEHOLDER_BYTES],
    ['normBuf', snap.normals, STORAGE_PLACEHOLDER_BYTES],
    ['matIdBuf', snap.triMaterialIds, STORAGE_PLACEHOLDER_BYTES],
    ['tlasNodesBuf', tlas?.nodes ?? emptyTlasNodes, BVH_NODE_PLACEHOLDER_BYTES],
    ['tlasInstIdxBuf', tlas?.instanceIndices ?? empty, STORAGE_PLACEHOLDER_BYTES],
    ['tlasBlasRootsBuf', tlas?.blasRoots ?? empty, STORAGE_PLACEHOLDER_BYTES],
    ['tlasW2lBuf', tlas?.worldToLocal ?? empty, STORAGE_PLACEHOLDER_BYTES],
    ['tlasL2wBuf', tlas?.localToWorld ?? empty, STORAGE_PLACEHOLDER_BYTES],
  ]);
}

export function rebuildProbeBvhFromScene(
  device: GPUDevice,
  g: ProbeUpdateBvhGpuBuffers,
  buffers: SceneBvhBuffers,
): void {
  const idx4 = padTriangleIndicesToVec4(buffers.indices);
  // Merged mode does not traverse the TLAS, but the probe-rays shader STILL
  // declares the five TLAS bindings (group 0, bindings 5–9). The first of them,
  // `tlasNodes: array<BVHNode>`, has a 32-byte struct stride → a minimum binding
  // size of 32 bytes. A 16-byte placeholder is REJECTED by strict backends
  // ("Binding size 16 … less than minimum 32" on lavapipe AND dzn), which makes
  // the bind group invalid and silently zeroes the probe atlas. Use a 32-byte
  // empty placeholder so the merged-mode bind group is valid on every backend.
  // (The TLAS-mode path uploads real, larger buffers via `rebuildProbeBvhFromRestir`,
  // so it was never affected.)
  const emptyTlasNodes = new ArrayBuffer(BVH_NODE_PLACEHOLDER_BYTES);
  const empty = new ArrayBuffer(STORAGE_PLACEHOLDER_BYTES);
  rebuildProbeBvhBuffers(device, g, [
    ['bvhBuf', buffers.bvhNodes, BVH_NODE_PLACEHOLDER_BYTES],
    ['posBuf', buffers.positions, STORAGE_PLACEHOLDER_BYTES],
    ['idxBuf', idx4, STORAGE_PLACEHOLDER_BYTES],
    ['normBuf', buffers.normals, STORAGE_PLACEHOLDER_BYTES],
    ['matIdBuf', buffers.triMaterialId, STORAGE_PLACEHOLDER_BYTES],
    ['tlasNodesBuf', emptyTlasNodes, BVH_NODE_PLACEHOLDER_BYTES],
    ['tlasInstIdxBuf', empty, STORAGE_PLACEHOLDER_BYTES],
    ['tlasBlasRootsBuf', empty, STORAGE_PLACEHOLDER_BYTES],
    ['tlasW2lBuf', empty, STORAGE_PLACEHOLDER_BYTES],
    ['tlasL2wBuf', empty, STORAGE_PLACEHOLDER_BYTES],
  ]);
}
