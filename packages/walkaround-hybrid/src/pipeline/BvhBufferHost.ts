/**
 * Owns merged-BVH + TLAS + emitter GPU buffers for {@link WalkaroundGPUPipeline}.
 * W4b — extracted from the pipeline god-file so upload/refit paths stay testable.
 */

import type { SceneBVHBuffers } from '../restir/bvhCompute.js';
import { createDummyStorageBuffer, uploadBuffer } from './resourceManager.js';

/** Mirrors `buildSceneBindGroup` resource bundle in bindGroupBuilders.ts. */
export interface SceneBindGroupResources {
  bvhNodesBuffer: GPUBuffer;
  bvhIndexBuffer: GPUBuffer;
  bvhPositionBuffer: GPUBuffer;
  emitterBuffer: GPUBuffer;
  emitterCdfBuffer: GPUBuffer;
  bvhBeerBuffer: GPUBuffer;
  tlasNodesBuffer: GPUBuffer;
  tlasInstanceIndicesBuffer: GPUBuffer;
  tlasBlasRootsBuffer: GPUBuffer;
  tlasInstanceWorldToLocalBuffer: GPUBuffer;
  tlasInstanceLocalToWorldBuffer: GPUBuffer;
}

/** `GPUBufferUsage.STORAGE` — literal avoids top-level `GPUBufferUsage` (Node vitest). */
const STORAGE = 0x80;

export class BvhBufferHost {
  private _bvhNodesBuffer: GPUBuffer | null = null;
  private _bvhIndexBuffer: GPUBuffer | null = null;
  private _bvhBeerBuffer: GPUBuffer | null = null;
  private _bvhPositionBuffer: GPUBuffer | null = null;
  private _tlasNodesBuffer: GPUBuffer | null = null;
  private _tlasInstanceIndicesBuffer: GPUBuffer | null = null;
  private _tlasBlasRootsBuffer: GPUBuffer | null = null;
  private _tlasInstanceWorldToLocalBuffer: GPUBuffer | null = null;
  private _tlasInstanceLocalToWorldBuffer: GPUBuffer | null = null;
  private _emitterBuffer: GPUBuffer | null = null;
  private _emitterCdfBuffer: GPUBuffer | null = null;
  private _lightTreeBuffer: GPUBuffer | null = null;

  get initialized(): boolean {
    return this._bvhNodesBuffer != null;
  }

  uploadInitial(device: GPUDevice, bvhBuffers: SceneBVHBuffers): void {
    this._bvhNodesBuffer = uploadBuffer(device, bvhBuffers.bvhNodes.cpuData, STORAGE);
    this._bvhIndexBuffer = uploadBuffer(device, bvhBuffers.bvhIndex.cpuData, STORAGE);
    this._bvhBeerBuffer = uploadBuffer(device, bvhBuffers.bvhBeerColors.cpuData, STORAGE);
    this._bvhPositionBuffer = uploadBuffer(device, bvhBuffers.bvhPositions.cpuData, STORAGE);
    this._emitterBuffer = uploadBuffer(device, bvhBuffers.emitters.cpuData, STORAGE);
    this._emitterCdfBuffer = uploadBuffer(device, bvhBuffers.emitterCdf.cpuData, STORAGE);
    this._lightTreeBuffer = uploadBuffer(device, bvhBuffers.lightTree.cpuData, STORAGE);
    this._uploadTlasBuffers(device, bvhBuffers);
  }

  /** RIS-only light-tree storage buffer (group 3 binding 0). Always non-null
   *  after `uploadInitial` (a 1-node placeholder backs it when the tree is
   *  disabled). */
  lightTreeBuffer(): GPUBuffer {
    if (this._lightTreeBuffer == null) {
      throw new Error('[BvhBufferHost] uploadInitial must run before lightTreeBuffer');
    }
    return this._lightTreeBuffer;
  }

  sceneBindGroupResources(): SceneBindGroupResources {
    if (!this.initialized) {
      throw new Error('[BvhBufferHost] uploadInitial must run before sceneBindGroupResources');
    }
    return {
      bvhNodesBuffer: this._bvhNodesBuffer!,
      bvhIndexBuffer: this._bvhIndexBuffer!,
      bvhPositionBuffer: this._bvhPositionBuffer!,
      emitterBuffer: this._emitterBuffer!,
      emitterCdfBuffer: this._emitterCdfBuffer!,
      bvhBeerBuffer: this._bvhBeerBuffer!,
      tlasNodesBuffer: this._tlasNodesBuffer!,
      tlasInstanceIndicesBuffer: this._tlasInstanceIndicesBuffer!,
      tlasBlasRootsBuffer: this._tlasBlasRootsBuffer!,
      tlasInstanceWorldToLocalBuffer: this._tlasInstanceWorldToLocalBuffer!,
      tlasInstanceLocalToWorldBuffer: this._tlasInstanceLocalToWorldBuffer!,
    };
  }

  updateEmitters(
    device: GPUDevice,
    bvhBuffers: Pick<SceneBVHBuffers, 'emitters' | 'emitterCdf' | 'lightTree'>,
  ): void {
    this._emitterBuffer?.destroy();
    this._emitterCdfBuffer?.destroy();
    this._lightTreeBuffer?.destroy();
    this._emitterBuffer = uploadBuffer(device, bvhBuffers.emitters.cpuData, STORAGE);
    this._emitterCdfBuffer = uploadBuffer(device, bvhBuffers.emitterCdf.cpuData, STORAGE);
    // Re-upload the selection tree: emitters changed, so the tree's leaf
    // emitterIndex → emitter array mapping (and powers) changed with them.
    this._lightTreeBuffer = uploadBuffer(device, bvhBuffers.lightTree.cpuData, STORAGE);
  }

  refreshBvhRefit(
    device: GPUDevice,
    bvhNodesBytes: ArrayBuffer,
    positionsSlice: { byteOffset: number; data: ArrayBuffer },
  ): void {
    if (!this.initialized) return;
    device.queue.writeBuffer(this._bvhNodesBuffer!, 0, bvhNodesBytes);
    device.queue.writeBuffer(this._bvhPositionBuffer!, positionsSlice.byteOffset, positionsSlice.data);
  }

  refreshBvhNodesOnly(device: GPUDevice, bvhNodesBytes: ArrayBuffer): void {
    if (!this.initialized) return;
    device.queue.writeBuffer(this._bvhNodesBuffer!, 0, bvhNodesBytes);
  }

  getBvhPositionBuffer(): GPUBuffer | null {
    return this._bvhPositionBuffer;
  }

  refreshTlasRefit(
    device: GPUDevice,
    tlasNodes: ArrayBuffer,
    worldToLocal: ArrayBuffer,
    localToWorld: ArrayBuffer,
  ): void {
    if (!this.initialized) return;
    device.queue.writeBuffer(this._tlasNodesBuffer!, 0, tlasNodes);
    device.queue.writeBuffer(this._tlasInstanceWorldToLocalBuffer!, 0, worldToLocal);
    device.queue.writeBuffer(this._tlasInstanceLocalToWorldBuffer!, 0, localToWorld);
  }

  refreshBvhMaterialSlice(
    device: GPUDevice,
    indexSlice: { byteOffset: number; data: ArrayBuffer },
    beerSlice: { byteOffset: number; data: ArrayBuffer },
  ): void {
    if (!this.initialized) return;
    device.queue.writeBuffer(this._bvhIndexBuffer!, indexSlice.byteOffset, indexSlice.data);
    device.queue.writeBuffer(this._bvhBeerBuffer!, beerSlice.byteOffset, beerSlice.data);
  }

  refreshBvhFullRebuild(
    device: GPUDevice,
    bvhBuffers: Pick<
      SceneBVHBuffers,
      'bvhNodes' | 'bvhIndex' | 'bvhBeerColors' | 'bvhPositions' | 'bvhMode' | 'tlas'
    >,
  ): void {
    if (!this.initialized) return;
    this._bvhNodesBuffer!.destroy();
    this._bvhIndexBuffer!.destroy();
    this._bvhBeerBuffer!.destroy();
    this._bvhPositionBuffer!.destroy();
    this._destroyTlasBuffers();
    this._bvhNodesBuffer = uploadBuffer(device, bvhBuffers.bvhNodes.cpuData, STORAGE);
    this._bvhIndexBuffer = uploadBuffer(device, bvhBuffers.bvhIndex.cpuData, STORAGE);
    this._bvhBeerBuffer = uploadBuffer(device, bvhBuffers.bvhBeerColors.cpuData, STORAGE);
    this._bvhPositionBuffer = uploadBuffer(device, bvhBuffers.bvhPositions.cpuData, STORAGE);
    this._uploadTlasBuffers(device, bvhBuffers as SceneBVHBuffers);
  }

  dispose(): void {
    this._bvhNodesBuffer?.destroy();
    this._bvhIndexBuffer?.destroy();
    this._bvhBeerBuffer?.destroy();
    this._bvhPositionBuffer?.destroy();
    this._destroyTlasBuffers();
    this._emitterBuffer?.destroy();
    this._emitterCdfBuffer?.destroy();
    this._lightTreeBuffer?.destroy();
    this._bvhNodesBuffer = null;
    this._bvhIndexBuffer = null;
    this._bvhBeerBuffer = null;
    this._bvhPositionBuffer = null;
    this._emitterBuffer = null;
    this._emitterCdfBuffer = null;
    this._lightTreeBuffer = null;
  }

  private _destroyTlasBuffers(): void {
    this._tlasNodesBuffer?.destroy();
    this._tlasInstanceIndicesBuffer?.destroy();
    this._tlasBlasRootsBuffer?.destroy();
    this._tlasInstanceWorldToLocalBuffer?.destroy();
    this._tlasInstanceLocalToWorldBuffer?.destroy();
    this._tlasNodesBuffer = null;
    this._tlasInstanceIndicesBuffer = null;
    this._tlasBlasRootsBuffer = null;
    this._tlasInstanceWorldToLocalBuffer = null;
    this._tlasInstanceLocalToWorldBuffer = null;
  }

  private _uploadTlasBuffers(device: GPUDevice, bvh: SceneBVHBuffers): void {
    const dummy = () => createDummyStorageBuffer(device, 'tlas-dummy');
    if (bvh.bvhMode === 'tlas' && bvh.tlas != null) {
      const t = bvh.tlas;
      this._tlasNodesBuffer = uploadBuffer(device, t.nodes.cpuData, STORAGE);
      this._tlasInstanceIndicesBuffer = uploadBuffer(device, t.instanceIndices.cpuData, STORAGE);
      this._tlasBlasRootsBuffer = uploadBuffer(device, t.blasRoots.cpuData, STORAGE);
      this._tlasInstanceWorldToLocalBuffer = uploadBuffer(device, t.worldToLocal.cpuData, STORAGE);
      this._tlasInstanceLocalToWorldBuffer = uploadBuffer(device, t.localToWorld.cpuData, STORAGE);
    } else {
      this._tlasNodesBuffer = dummy();
      this._tlasInstanceIndicesBuffer = dummy();
      this._tlasBlasRootsBuffer = dummy();
      this._tlasInstanceWorldToLocalBuffer = dummy();
      this._tlasInstanceLocalToWorldBuffer = dummy();
    }
  }
}
