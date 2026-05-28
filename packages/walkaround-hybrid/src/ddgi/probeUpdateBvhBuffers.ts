/**
 * BVH / TLAS GPU buffer rebuild helpers for {@link ProbeUpdatePass} (W4c).
 */

import type { SceneBvhBuffers } from '@vitrum/shared-bvh';
import type { DdgiRestirBvhSnapshot } from './ddgiRestirBvh.js';
import { padTriangleIndicesToVec4 } from './probeUpdateMaterials.js';

const RO = 0x80 | 0x08; // STORAGE | COPY_DST — literal for Node vitest import chain

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

function replaceStorageBuffer(
  device: GPUDevice,
  oldBuf: GPUBuffer,
  data: ArrayBufferLike,
): GPUBuffer {
  oldBuf.destroy();
  const arr = data instanceof ArrayBuffer ? data : new Uint8Array(data).buffer;
  const buf = device.createBuffer({
    size: Math.max(arr.byteLength, 16),
    usage: RO,
  });
  device.queue.writeBuffer(buf, 0, arr);
  return buf;
}

/** C2 — TLAS transform refit: upload nodes + instance matrices only. */
export function refitProbeTlasBuffersInPlace(
  device: GPUDevice,
  g: ProbeUpdateBvhGpuBuffers,
  tlas: NonNullable<DdgiRestirBvhSnapshot['tlas']>,
): void {
  device.queue.writeBuffer(g.tlasNodesBuf, 0, tlas.nodes);
  device.queue.writeBuffer(g.tlasW2lBuf, 0, tlas.worldToLocal);
  device.queue.writeBuffer(g.tlasL2wBuf, 0, tlas.localToWorld);
}

export function rebuildProbeBvhFromRestir(
  device: GPUDevice,
  g: ProbeUpdateBvhGpuBuffers,
  snap: DdgiRestirBvhSnapshot,
): void {
  const upload = (old: GPUBuffer, data: ArrayBufferLike) => replaceStorageBuffer(device, old, data);
  g.bvhBuf = upload(g.bvhBuf, snap.bvhNodes);
  g.posBuf = upload(g.posBuf, snap.positions);
  g.idxBuf = upload(g.idxBuf, snap.bvhIndex);
  g.normBuf = upload(g.normBuf, snap.normals);
  g.matIdBuf = upload(g.matIdBuf, snap.triMaterialIds);
  const empty = new ArrayBuffer(16);
  const tlas = snap.tlas;
  g.tlasNodesBuf = upload(g.tlasNodesBuf, tlas?.nodes ?? empty);
  g.tlasInstIdxBuf = upload(g.tlasInstIdxBuf, tlas?.instanceIndices ?? empty);
  g.tlasBlasRootsBuf = upload(g.tlasBlasRootsBuf, tlas?.blasRoots ?? empty);
  g.tlasW2lBuf = upload(g.tlasW2lBuf, tlas?.worldToLocal ?? empty);
  g.tlasL2wBuf = upload(g.tlasL2wBuf, tlas?.localToWorld ?? empty);
}

export function rebuildProbeBvhFromScene(
  device: GPUDevice,
  g: ProbeUpdateBvhGpuBuffers,
  buffers: SceneBvhBuffers,
): void {
  const upload = (old: GPUBuffer, data: ArrayBufferLike) => replaceStorageBuffer(device, old, data);
  const idx4 = padTriangleIndicesToVec4(buffers.indices);
  g.bvhBuf = upload(g.bvhBuf, buffers.bvhNodes.buffer);
  g.posBuf = upload(g.posBuf, buffers.positions.buffer);
  g.idxBuf = upload(g.idxBuf, idx4.buffer);
  g.normBuf = upload(g.normBuf, buffers.normals.buffer);
  g.matIdBuf = upload(g.matIdBuf, buffers.triMaterialId.buffer);
  const empty = new ArrayBuffer(16);
  g.tlasNodesBuf = upload(g.tlasNodesBuf, empty);
  g.tlasInstIdxBuf = upload(g.tlasInstIdxBuf, empty);
  g.tlasBlasRootsBuf = upload(g.tlasBlasRootsBuf, empty);
  g.tlasW2lBuf = upload(g.tlasW2lBuf, empty);
  g.tlasL2wBuf = upload(g.tlasL2wBuf, empty);
}
