/**
 * Owns merged-BVH + TLAS + emitter GPU buffers for {@link WalkaroundGPUPipeline}.
 * W4b — extracted from the pipeline god-file so upload/refit paths stay testable.
 */

import type { SceneBVHBuffers } from '../restir/bvhTypes.js';
import { createDummyStorageBuffer, uploadBuffer, uploadBufferPadded } from './resourceManager.js';
import {
  packAnalyticPointSpotEmitters,
} from '../restir/bvhSceneHelpers.js';
import { EMITTER_TRI_STRIDE_BYTES } from '../restir/emitterList.js';
import type { Scene } from '@vitrum/core';
import {
  uploadBeerTexture,
  refreshBeerTexture,
  type BeerTexture,
} from './bvhBeerTexture.js';
import {
  uploadEmissiveTexture,
  refreshEmissiveTexture,
  type EmissiveTexture,
} from './bvhEmissiveTexture.js';
import {
  uploadAnalyticLightsTexture,
  type AnalyticLightsTexture,
} from './analyticLightsTexture.js';
import type { GpuMemoryExternalSections } from './gpuMemoryEstimate.js';
import { PipelineResourceCache } from './PipelineResourceCache.js';
import {
  createPlaceholderEnvironment,
  uploadEnvironment,
  clearEnvironment,
  disposeEnvironment,
  type EnvironmentTextures,
} from './environmentTexture.js';
import type { DirectionalEnvData } from '../environment/equirectDirectional.js';

/** Mirrors `buildSceneBindGroup` resource bundle in bindGroupBuilders.ts. */
export interface SceneBindGroupResources {
  bvhNodesBuffer: GPUBuffer;
  bvhIndexBuffer: GPUBuffer;
  bvhPositionBuffer: GPUBuffer;
  emitterBuffer: GPUBuffer;
  emitterCdfBuffer: GPUBuffer;
  /** WS1 — per-tri Beer-Lambert visible color, r32uint texture (was a storage
   *  buffer). Shade reads it via `textureLoad`; the swap freed a storage slot
   *  for `bvhNormalBuffer`. */
  bvhBeerTextureView: GPUTextureView;
  /** Camera-visible emitters — per-tri HDR emissive Le, rgba32float texture
   *  (binding 12). Shade reads it via `textureLoad` (lo_emitterGlow). */
  bvhEmissiveTextureView: GPUTextureView;
  /** B1 — per-tri roughness+metalness, r32uint texture (binding 14). The
   *  ReSTIR/shade GGX BRDF + glossy/metal GI target read it via
   *  `decodeRoughMetal(triIndex)`. Same r32uint layout as `bvhBeer`. */
  bvhRoughMetalTextureView: GPUTextureView;
  /** WS1 — per-vertex world-space normals (stride-4 vec4f). Barycentric-blended
   *  in the primary passes for a smooth shading normal. */
  bvhNormalBuffer: GPUBuffer;
  tlasNodesBuffer: GPUBuffer;
  tlasInstanceIndicesBuffer: GPUBuffer;
  tlasBlasRootsBuffer: GPUBuffer;
  tlasInstanceWorldToLocalBuffer: GPUBuffer;
  tlasInstanceLocalToWorldBuffer: GPUBuffer;
  analyticLightsTextureView: GPUTextureView;
  /** B3 — directional IBL resources (bindings 15-19). Placeholders + hasEnv=0
   *  for non-HDRI scenes (scalar-tint fallback). */
  envMapTextureView: GPUTextureView;
  envMarginalTextureView: GPUTextureView;
  envConditionalTextureView: GPUTextureView;
  envSampler: GPUSampler;
  envParamsBuffer: GPUBuffer;
}

/** `GPUBufferUsage.STORAGE` — literal avoids top-level `GPUBufferUsage` (Node vitest). */
const STORAGE = 0x80;

export class BvhBufferHost {
  private readonly _resourceCache = new PipelineResourceCache();
  private _bvhNodesBuffer: GPUBuffer | null = null;
  private _bvhIndexBuffer: GPUBuffer | null = null;
  /** WS1 — beer is now a texture; track triCount so refit can re-upload it. */
  private _bvhBeerTexture: BeerTexture | null = null;
  private _bvhBeerTriCount = 0;
  /** Camera-visible emitters — per-tri HDR emissive Le, rgba32float texture. */
  private _bvhEmissiveTexture: EmissiveTexture | null = null;
  private _bvhEmissiveTriCount = 0;
  /** B1 — per-tri roughness+metalness, r32uint texture (reuses BeerTexture
   *  helpers; identical r32uint/one-u32-per-triangle layout). */
  private _bvhRoughMetalTexture: BeerTexture | null = null;
  private _bvhRoughMetalTriCount = 0;
  /** WS1 — per-vertex world-space normals for the smooth shading-normal blend. */
  private _bvhNormalBuffer: GPUBuffer | null = null;
  private _bvhPositionBuffer: GPUBuffer | null = null;
  private _tlasNodesBuffer: GPUBuffer | null = null;
  private _tlasInstanceIndicesBuffer: GPUBuffer | null = null;
  private _tlasBlasRootsBuffer: GPUBuffer | null = null;
  private _tlasInstanceWorldToLocalBuffer: GPUBuffer | null = null;
  private _tlasInstanceLocalToWorldBuffer: GPUBuffer | null = null;
  private _emitterBuffer: GPUBuffer | null = null;
  /** Number of EmitterTri entries in `_emitterBuffer` (for RC NEE). */
  private _emitterCount = 0;
  private _emitterCdfBuffer: GPUBuffer | null = null;
  private _lightTreeBuffer: GPUBuffer | null = null;
  private _analyticLightsTexture: AnalyticLightsTexture | null = null;
  /** B3 — directional IBL resources (bindings 15-19). Placeholder until a raw
   *  HDRI is supplied via updateEnvironment. */
  private _env: EnvironmentTextures | null = null;

  /**
   * Extra bytes appended to the light-tree storage buffer to hold the ReGIR
   * grid region (the grid-build pass writes it; RIS reads it from the SAME
   * @group(3) buffer so RIS stays at 16 storage buffers). `0` ⇒ ReGIR off, the
   * light-tree buffer is sized exactly as before (byte-identical). Stable for
   * the buffer's lifetime — set once by the pipeline before `uploadInitial`.
   */
  private _regirGridBytes = 0;

  /** Set the ReGIR grid byte count appended to the light-tree buffer. Must be
   *  called BEFORE `uploadInitial` (and before any `updateEmitters`). `0`
   *  disables ReGIR co-location (default — byte-identical to pre-ReGIR). */
  setRegirGridBytes(bytes: number): void {
    this._regirGridBytes = Math.max(0, bytes | 0);
  }

  get initialized(): boolean {
    return this._bvhNodesBuffer != null;
  }

  uploadInitial(device: GPUDevice, bvhBuffers: SceneBVHBuffers, scene?: Scene): void {
    this._bvhNodesBuffer = uploadBuffer(device, bvhBuffers.bvhNodes.cpuData, STORAGE);
    this._bvhIndexBuffer = uploadBuffer(device, bvhBuffers.bvhIndex.cpuData, STORAGE);
    this._bvhBeerTriCount = bvhBuffers.bvhBeerColors.count;
    this._bvhBeerTexture = uploadBeerTexture(
      device, bvhBuffers.bvhBeerColors.cpuData, this._bvhBeerTriCount);
    // Camera-visible emitters — per-tri HDR emissive Le (rgba32float texture).
    this._bvhEmissiveTriCount = bvhBuffers.bvhEmissiveLe.count;
    this._bvhEmissiveTexture = uploadEmissiveTexture(
      device,
      new Float32Array(bvhBuffers.bvhEmissiveLe.cpuData),
      this._bvhEmissiveTriCount);
    // B1 — per-tri roughness+metalness r32uint texture (same helper as beer).
    this._bvhRoughMetalTriCount = bvhBuffers.bvhRoughMetal.count;
    this._bvhRoughMetalTexture = uploadBeerTexture(
      device, bvhBuffers.bvhRoughMetal.cpuData, this._bvhRoughMetalTriCount);
    // WS1 — per-vertex world-space normals (stride-4 vec4f, .w unused). Same
    // data the DDGI / emitter paths already use (shared.normals).
    this._bvhNormalBuffer = uploadBuffer(device, bvhBuffers.bvhNormals.cpuData, STORAGE);
    this._bvhPositionBuffer = uploadBuffer(device, bvhBuffers.bvhPositions.cpuData, STORAGE);
    const emitterCount = validateEmitterPayload('uploadInitial', bvhBuffers.emitters);
    if (bvhBuffers.emitterCount !== emitterCount) {
      throw new RangeError(
        `[BvhBufferHost] uploadInitial emitterCount ${bvhBuffers.emitterCount} does not match ` +
        `emitters.count ${emitterCount}.`,
      );
    }
    this._emitterBuffer = uploadBuffer(device, bvhBuffers.emitters.cpuData, STORAGE);
    this._emitterCount = emitterCount;
    this._emitterCdfBuffer = uploadBuffer(device, bvhBuffers.emitterCdf.cpuData, STORAGE);
    // Combined light-tree + ReGIR-grid buffer (tree nodes in front, grid region
    // zeroed at the tail). `_regirGridBytes == 0` ⇒ exactly `uploadBuffer`.
    this._lightTreeBuffer = uploadBufferPadded(
      device, bvhBuffers.lightTree.cpuData, this._regirGridBytes, STORAGE);
    const analyticPacked = scene != null
      ? packAnalyticPointSpotEmitters(scene)
      : { data: new Float32Array(16), count: 0 };
    this._analyticLightsTexture = uploadAnalyticLightsTexture(
      device,
      analyticPacked.data,
      analyticPacked.count,
    );
    // B3 — directional IBL placeholder (hasEnv=0). updateEnvironment swaps in the
    // real map+CDFs when a raw HDRI is resolved (the WGSL falls back to the scalar
    // sky while this is the placeholder → no-HDRI byte-identity).
    if (this._env == null) {
      this._env = createPlaceholderEnvironment(device);
    }
    this._uploadTlasBuffers(device, bvhBuffers);
  }

  /**
   * B3 — swap the directional IBL resources. `data == null` resets to the no-HDRI
   * placeholder (hasEnv=0). Safe to call before `uploadInitial` (lazily creates
   * the placeholder first). After a swap the caller MUST invalidate the cached
   * scene bind group (the texture views changed) — WalkaroundGPUPipeline does
   * this via its scene-bind-group rebuild on setScene/updateEnvironment.
   */
  updateEnvironment(
    device: GPUDevice,
    data: DirectionalEnvData | null,
    rotationY: number,
    intensity: number,
  ): void {
    if (this._env == null) {
      this._env = createPlaceholderEnvironment(device);
    }
    if (data == null) {
      this._env = clearEnvironment(device, this._env);
    } else {
      this._env = uploadEnvironment(device, this._env, data, rotationY, intensity);
    }
  }

  updateAnalyticLights(device: GPUDevice, scene: Scene): void {
    const packed = packAnalyticPointSpotEmitters(scene);
    if (this._analyticLightsTexture != null) {
      this._analyticLightsTexture.texture.destroy();
    }
    this._analyticLightsTexture = uploadAnalyticLightsTexture(device, packed.data, packed.count);
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

  /** Shared rect-area emitter buffer + tri count for RC NEE (group-agnostic —
   *  emitters are world-space triangles). Null before `uploadInitial`. */
  emitterBufferAndCount(): { buffer: GPUBuffer; count: number } | null {
    if (this._emitterBuffer == null) return null;
    return { buffer: this._emitterBuffer, count: this._emitterCount };
  }

  /** A7 (2026-06-10): equirectangular env map texture + sampler for RC probe
   *  env sampling. Returns null before `uploadInitial`. The map is the
   *  placeholder (black) when no HDRI has been uploaded; RC should bind it
   *  regardless — `envIntensity` stays 1 but the placeholder just returns
   *  (0,0,0,1) so the last-cascade env sample produces zero radiance, which
   *  is correct when no env is present. */
  envBindings(): { textureView: GPUTextureView; sampler: GPUSampler } | null {
    if (this._env == null) return null;
    return {
      textureView: this._resourceCache.textureView(this._env.map),
      sampler: this._env.sampler,
    };
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
      bvhBeerTextureView: this._resourceCache.textureView(this._bvhBeerTexture!.texture),
      bvhEmissiveTextureView: this._resourceCache.textureView(this._bvhEmissiveTexture!.texture),
      bvhRoughMetalTextureView: this._resourceCache.textureView(this._bvhRoughMetalTexture!.texture),
      bvhNormalBuffer: this._bvhNormalBuffer!,
      tlasNodesBuffer: this._tlasNodesBuffer!,
      tlasInstanceIndicesBuffer: this._tlasInstanceIndicesBuffer!,
      tlasBlasRootsBuffer: this._tlasBlasRootsBuffer!,
      tlasInstanceWorldToLocalBuffer: this._tlasInstanceWorldToLocalBuffer!,
      tlasInstanceLocalToWorldBuffer: this._tlasInstanceLocalToWorldBuffer!,
      analyticLightsTextureView: this._resourceCache.textureView(this._analyticLightsTexture!.texture),
      envMapTextureView: this._resourceCache.textureView(this._env!.map),
      envMarginalTextureView: this._resourceCache.textureView(this._env!.marginal),
      envConditionalTextureView: this._resourceCache.textureView(this._env!.conditional),
      envSampler: this._env!.sampler,
      envParamsBuffer: this._env!.paramsBuffer,
    };
  }

  gpuMemorySections(): GpuMemoryExternalSections {
    const section: Record<string, unknown> = {};
    const add = (name: string, resource: unknown): void => {
      if (resource != null) section[name] = resource;
    };

    add('bvhNodesBuffer', this._bvhNodesBuffer);
    add('bvhIndexBuffer', this._bvhIndexBuffer);
    add('bvhNormalBuffer', this._bvhNormalBuffer);
    add('bvhPositionBuffer', this._bvhPositionBuffer);
    add('emitterBuffer', this._emitterBuffer);
    add('emitterCdfBuffer', this._emitterCdfBuffer);
    add('lightTreeBuffer', this._lightTreeBuffer);
    add('tlasNodesBuffer', this._tlasNodesBuffer);
    add('tlasInstanceIndicesBuffer', this._tlasInstanceIndicesBuffer);
    add('tlasBlasRootsBuffer', this._tlasBlasRootsBuffer);
    add('tlasInstanceWorldToLocalBuffer', this._tlasInstanceWorldToLocalBuffer);
    add('tlasInstanceLocalToWorldBuffer', this._tlasInstanceLocalToWorldBuffer);

    if (this._bvhBeerTexture != null) {
      section.bvhBeerTexture = {
        width: this._bvhBeerTexture.width,
        height: this._bvhBeerTexture.height,
        depthOrArrayLayers: 1,
        format: 'r32uint' as GPUTextureFormat,
      };
    }
    if (this._bvhEmissiveTexture != null) {
      section.bvhEmissiveTexture = {
        width: this._bvhEmissiveTexture.width,
        height: this._bvhEmissiveTexture.height,
        depthOrArrayLayers: 1,
        format: 'rgba32float' as GPUTextureFormat,
      };
    }
    if (this._bvhRoughMetalTexture != null) {
      section.bvhRoughMetalTexture = {
        width: this._bvhRoughMetalTexture.width,
        height: this._bvhRoughMetalTexture.height,
        depthOrArrayLayers: 1,
        format: 'r32uint' as GPUTextureFormat,
      };
    }
    if (this._analyticLightsTexture != null) {
      section.analyticLightsTexture = {
        width: this._analyticLightsTexture.width,
        height: this._analyticLightsTexture.height,
        depthOrArrayLayers: 1,
        format: 'rgba32float' as GPUTextureFormat,
      };
    }

    return { staticScene: section };
  }

  updateEmitters(
    device: GPUDevice,
    bvhBuffers: Pick<SceneBVHBuffers, 'emitters' | 'emitterCdf' | 'lightTree'>,
  ): void {
    const nextEmitterCount = validateEmitterPayload('updateEmitters', bvhBuffers.emitters);
    this._emitterBuffer?.destroy();
    this._emitterCdfBuffer?.destroy();
    this._lightTreeBuffer?.destroy();
    this._emitterBuffer = uploadBuffer(device, bvhBuffers.emitters.cpuData, STORAGE);
    // updateEmitters' Pick omits SceneBVHBuffers.emitterCount, but the
    // StorageBufferHandle carries the canonical count and has already been
    // byte-validated against EMITTER_TRI_STRIDE_BYTES above.
    this._emitterCount = nextEmitterCount;
    this._emitterCdfBuffer = uploadBuffer(device, bvhBuffers.emitterCdf.cpuData, STORAGE);
    // Re-upload the selection tree: emitters changed, so the tree's leaf
    // emitterIndex → emitter array mapping (and powers) changed with them.
    // Re-pad for the ReGIR grid region (zeroed; the grid-build pass refills it
    // next frame). The tree node count may have changed, so the grid region's
    // float offset (lightTreeNodeCount × LIGHT_TREE_FLOATS_PER_NODE = ×16) is recomputed by the pipeline.
    this._lightTreeBuffer = uploadBufferPadded(
      device, bvhBuffers.lightTree.cpuData, this._regirGridBytes, STORAGE);
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

  /** H19 — upload a per-vertex normals slice (vec4f stride, .w unused). */
  refreshBvhNormalsSlice(
    device: GPUDevice,
    normalsSlice: { byteOffset: number; data: ArrayBuffer },
  ): void {
    if (!this.initialized) return;
    device.queue.writeBuffer(this._bvhNormalBuffer!, normalsSlice.byteOffset, normalsSlice.data);
  }

  getBvhPositionBuffer(): GPUBuffer | null {
    return this._bvhPositionBuffer;
  }

  /** WS1 — live merged per-vertex normal buffer. The GPU-skin kernel writes
   *  inverse-transpose skinned normals directly into it at `baseVertex+vi`. */
  getBvhNormalBuffer(): GPUBuffer | null {
    return this._bvhNormalBuffer;
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
    /** WS1 — beer is a texture now: pass the FULL beer data + triCount so the
     *  texture is re-uploaded wholesale (a contiguous triangle slice is not a
     *  rectangular texture region unless it spans full rows). Cheap: 4 B/tri. */
    beerFull: { data: ArrayBuffer; triCount: number },
    /** Camera-visible emitters — FULL per-tri emissive Le re-upload (same
     *  wholesale rationale as beer; a triangle slice is not a rectangular
     *  texture region). */
    emissiveFull: { data: ArrayBuffer; triCount: number },
    /** B1 — FULL per-tri roughness+metalness re-upload (same wholesale
     *  rationale as beer/emissive). Optional so legacy callers stay valid. */
    roughMetalFull?: { data: ArrayBuffer; triCount: number },
  ): void {
    if (!this.initialized) return;
    device.queue.writeBuffer(this._bvhIndexBuffer!, indexSlice.byteOffset, indexSlice.data);
    refreshBeerTexture(device, this._bvhBeerTexture!, beerFull.data, beerFull.triCount);
    refreshEmissiveTexture(
      device, this._bvhEmissiveTexture!, new Float32Array(emissiveFull.data), emissiveFull.triCount);
    if (roughMetalFull && this._bvhRoughMetalTexture) {
      refreshBeerTexture(
        device, this._bvhRoughMetalTexture, roughMetalFull.data, roughMetalFull.triCount);
    }
  }

  refreshBvhFullRebuild(
    device: GPUDevice,
    bvhBuffers: Pick<
      SceneBVHBuffers,
      'bvhNodes' | 'bvhIndex' | 'bvhBeerColors' | 'bvhEmissiveLe' | 'bvhRoughMetal' | 'bvhNormals' | 'bvhPositions' | 'bvhMode' | 'tlas'
    >,
  ): void {
    if (!this.initialized) return;
    this._bvhNodesBuffer!.destroy();
    this._bvhIndexBuffer!.destroy();
    this._bvhBeerTexture!.texture.destroy();
    this._bvhEmissiveTexture!.texture.destroy();
    this._bvhRoughMetalTexture?.texture.destroy();
    this._bvhNormalBuffer!.destroy();
    this._bvhPositionBuffer!.destroy();
    this._destroyTlasBuffers();
    this._bvhNodesBuffer = uploadBuffer(device, bvhBuffers.bvhNodes.cpuData, STORAGE);
    this._bvhIndexBuffer = uploadBuffer(device, bvhBuffers.bvhIndex.cpuData, STORAGE);
    this._bvhBeerTriCount = bvhBuffers.bvhBeerColors.count;
    this._bvhBeerTexture = uploadBeerTexture(
      device, bvhBuffers.bvhBeerColors.cpuData, this._bvhBeerTriCount);
    this._bvhRoughMetalTriCount = bvhBuffers.bvhRoughMetal.count;
    this._bvhRoughMetalTexture = uploadBeerTexture(
      device, bvhBuffers.bvhRoughMetal.cpuData, this._bvhRoughMetalTriCount);
    this._bvhEmissiveTriCount = bvhBuffers.bvhEmissiveLe.count;
    this._bvhEmissiveTexture = uploadEmissiveTexture(
      device, new Float32Array(bvhBuffers.bvhEmissiveLe.cpuData), this._bvhEmissiveTriCount);
    this._bvhNormalBuffer = uploadBuffer(device, bvhBuffers.bvhNormals.cpuData, STORAGE);
    this._bvhPositionBuffer = uploadBuffer(device, bvhBuffers.bvhPositions.cpuData, STORAGE);
    this._uploadTlasBuffers(device, bvhBuffers as SceneBVHBuffers);
  }

  dispose(): void {
    this._bvhNodesBuffer?.destroy();
    this._bvhIndexBuffer?.destroy();
    this._bvhBeerTexture?.texture.destroy();
    this._bvhEmissiveTexture?.texture.destroy();
    this._bvhRoughMetalTexture?.texture.destroy();
    this._bvhNormalBuffer?.destroy();
    this._bvhPositionBuffer?.destroy();
    this._destroyTlasBuffers();
    this._emitterBuffer?.destroy();
    this._emitterCdfBuffer?.destroy();
    this._lightTreeBuffer?.destroy();
    this._analyticLightsTexture?.texture.destroy();
    if (this._env != null) { disposeEnvironment(this._env); this._env = null; }
    this._bvhNodesBuffer = null;
    this._bvhIndexBuffer = null;
    this._bvhBeerTexture = null;
    this._bvhEmissiveTexture = null;
    this._bvhRoughMetalTexture = null;
    this._bvhNormalBuffer = null;
    this._bvhPositionBuffer = null;
    this._emitterBuffer = null;
    this._emitterCdfBuffer = null;
    this._lightTreeBuffer = null;
    this._analyticLightsTexture = null;
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

function validateEmitterPayload(
  source: 'uploadInitial' | 'updateEmitters',
  emitters: Pick<SceneBVHBuffers['emitters'], 'cpuData' | 'count'>,
): number {
  const byteLength = emitters.cpuData.byteLength;
  if (byteLength % EMITTER_TRI_STRIDE_BYTES !== 0) {
    throw new RangeError(
      `[BvhBufferHost] ${source} emitters payload has ${byteLength} bytes, which is not aligned ` +
      `to the ${EMITTER_TRI_STRIDE_BYTES}-byte EmitterTri stride.`,
    );
  }
  const derivedCount = byteLength / EMITTER_TRI_STRIDE_BYTES;
  if (!Number.isInteger(emitters.count) || emitters.count < 0) {
    throw new RangeError(`[BvhBufferHost] ${source} emitters.count must be a non-negative integer.`);
  }
  if (emitters.count !== derivedCount) {
    throw new RangeError(
      `[BvhBufferHost] ${source} emitters.count ${emitters.count} does not match ` +
      `${byteLength} packed bytes (${derivedCount} EmitterTri entries).`,
    );
  }
  return emitters.count;
}
