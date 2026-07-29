/**
 * Owns merged-BVH + TLAS + emitter GPU buffers for {@link WalkaroundGPUPipeline}.
 * W4b — extracted from the pipeline god-file so upload/refit paths stay testable.
 */

import type { SceneBVHBuffers } from '../restir/bvhTypes.js';
import { uploadBuffer, uploadBufferPadded } from './resourceManager.js';
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
  uploadMaterialTextureAtlas,
  type MaterialTextureAtlasPayload,
  type MaterialTextureAtlasGpu,
} from './materialTextureAtlas.js';
import {
  uploadTangentTexture,
  type TangentTexture,
} from './bvhTangentTexture.js';
import {
  uploadVertexColorTexture,
  type VertexColorTexture,
} from './bvhVertexColorTexture.js';
import {
  uploadAnalyticLightsTexture,
  type AnalyticLightsTexture,
} from './analyticLightsTexture.js';
import type { GpuMemoryExternalSections } from './gpuMemoryEstimate.js';
import type { CollectedBvhMutation } from './CollectingBvhUpdateSink.js';
import { PipelineResourceCache } from './PipelineResourceCache.js';
import {
  createPlaceholderEnvironment,
  uploadEnvironment,
  clearEnvironment,
  type EnvironmentTextures,
} from './environmentTexture.js';
import type { DirectionalEnvData } from '../environment/equirectDirectional.js';
import {
  rethrowWithSceneMutationCleanup,
  runSceneMutationCleanups,
  type PreparedSceneMutation,
  type SceneMutationCleanup,
} from '../SceneMutationTransaction.js';
import {
  SCENE_STORAGE_SHARD_SEGMENTS,
  assertSceneStorageArenaFits,
  buildSceneStorageArena,
  emptySceneStorageSegmentSources,
  patchSceneStorageArenaSource,
  patchSceneStorageArenaSources,
  nextSceneStorageArenaEpoch,
  retainSceneStorageArenaSources,
  sceneGeometryStorageSources,
  sceneLightingStorageSources,
  sceneStorageArenaHeaderBytes,
  sceneStorageArenaShard,
  sceneStorageSegmentSources,
  type SceneStorageArenaPayload,
  type SceneStorageSegment,
  type SceneStorageSegmentSource,
  type SceneStorageArenaSourcePatch,
} from './sceneStorageArena.js';

/** Mirrors `buildSceneBindGroup` resource bundle in bindGroupBuilders.ts. */
export interface SceneBindGroupResources {
  sceneStorageArenaBuffers: readonly [GPUBuffer, GPUBuffer, GPUBuffer];
  /** WS1 — per-tri Beer-Lambert visible color, r32uint texture (was a storage
   *  buffer). Shade reads it via `textureLoad`; the swap freed a storage slot
   *  for `bvhNormalBuffer`. */
  bvhBeerTextureView: GPUTextureView;
  /** Camera-visible/GI-suffix emitters — per-tri HDR emissive Le, rgba32float
   *  texture (binding 12). Shade/OIT read it for visible glow; ReSTIR-GI suffix
   *  hits combine it with readable emissive maps at hit UV. */
  bvhEmissiveTextureView: GPUTextureView;
  /** B1 — per-tri roughness+metalness, r32uint texture (binding 14). The
   *  ReSTIR/shade GGX BRDF + glossy/metal GI target read it via
   *  `decodeRoughMetal(triIndex)`. Same r32uint layout as `bvhBeer`. */
  bvhRoughMetalTextureView: GPUTextureView;
  /** Phase-3D material-map RGBA32F array texture. */
  materialTextureAtlasView: GPUTextureView;
  /** Phase-3D first slice — per-triangle baseColorMap layer/UV metadata. */
  baseColorMapMetaTextureView: GPUTextureView;
  /** Per-vertex authored/generated tangents, exposed as a texture so the scene
   *  group does not spend another storage-buffer slot. */
  bvhTangentTextureView: GPUTextureView;
  /** Per-vertex COLOR_0 colors (rgba32float texture, binding 23). */
  bvhVertexColorTextureView: GPUTextureView;
  analyticLightsTextureView: GPUTextureView;
  /** B3 — directional IBL resources (bindings 15-19). Placeholders + hasEnv=0
   *  for non-HDRI scenes (scalar-tint fallback). */
  envMapTextureView: GPUTextureView;
  envMarginalTextureView: GPUTextureView;
  envConditionalTextureView: GPUTextureView;
  envParamsBuffer: GPUBuffer;
}

/**
 * Canonical geometry windows shared with Radiance Cascades. Each range points
 * into this host's scene-storage arena; RC binds the ranges directly and must
 * never destroy their buffers.
 */
export interface SceneGeometryBufferBindings {
  readonly bvhNodes: GPUBufferBinding;
  readonly bvhIndices: GPUBufferBinding;
  readonly bvhPositions: GPUBufferBinding;
  readonly bvhNormals: GPUBufferBinding;
}

/** GPU resource retirement is best-effort: one hostile/mock destroy must not
 * prevent the remaining generation from being released. */
function destroyResource(resource: { destroy(): void } | null | undefined): void {
  try {
    resource?.destroy();
  } catch {
    // Continue retirement: cleanup must never mask the mutation's real failure.
  }
}

/** `GPUBufferUsage.STORAGE` — literal avoids top-level `GPUBufferUsage` (Node vitest). */
const STORAGE = 0x80;

export class BvhBufferHost {
  private readonly _resourceCache = new PipelineResourceCache();
  private _sceneStorageArenaBuffers: [
    GPUBuffer | null,
    GPUBuffer | null,
    GPUBuffer | null,
  ] = [null, null, null];
  private _sceneStorageArenaPayload: SceneStorageArenaPayload | null = null;
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
  /** Phase-3D first slice — baseColorMap atlas + per-tri metadata. */
  private _materialTextureAtlas: MaterialTextureAtlasGpu | null = null;
  /** WS1 — per-vertex world-space normals for the smooth shading-normal blend. */
  private _bvhTangentTexture: TangentTexture | null = null;
  private _bvhVertexColorTexture: VertexColorTexture | null = null;
  /** Number of EmitterTri entries in the lighting arena (for RC NEE). */
  private _emitterCount = 0;
  private _lightTreeBuffer: GPUBuffer | null = null;
  private _analyticLightsTexture: AnalyticLightsTexture | null = null;
  /** B3 — directional IBL resources (bindings 15-19). Placeholder until a raw
   *  HDRI is supplied via updateEnvironment. */
  private _env: EnvironmentTextures | null = null;

  /**
   * Extra bytes appended to the light-tree storage buffer to hold the ReGIR
   * grid region (the grid-build pass writes it; RIS reads it from the SAME
   * @group(3) buffer so RIS stays at the derived storage-buffer floor). `0` ⇒ ReGIR off, the
   * light-tree buffer is sized exactly as before (byte-identical). Stable for
   * the buffer's lifetime — set once by the pipeline before `uploadInitial`.
   */
  private _regirGridBytes = 0;

  /** Set the ReGIR grid byte count appended to the light-tree buffer. Must be
   *  called BEFORE `uploadInitial` (and before any `updateEmitters`). `0`
   *  disables ReGIR co-location (default — byte-identical to pre-ReGIR). */
  setRegirGridBytes(bytes: number): void {
    if (!Number.isSafeInteger(bytes) || bytes < 0 || bytes % 4 !== 0) {
      throw new RangeError(
        `[BvhBufferHost] ReGIR grid padding must be a non-negative, 4-byte-aligned ` +
          `safe integer; received ${String(bytes)}.`,
      );
    }
    this._regirGridBytes = bytes;
  }

  get initialized(): boolean {
    return this._sceneStorageArenaBuffers[0] != null;
  }

  private _uploadSceneStorageArena(
    device: GPUDevice,
    replacements: Readonly<
      Partial<Record<SceneStorageSegment, SceneStorageSegmentSource>>
    >,
    shardIndices: readonly number[],
  ): void {
    const previous = this._sceneStorageArenaPayload;
    const replacesGeometry =
      shardIndices.includes(0) || shardIndices.includes(1);
    const base = previous ?? buildSceneStorageArena(
      emptySceneStorageSegmentSources(),
    );
    const payload = retainSceneStorageArenaSources(
      base,
      replacements,
      previous == null ? 1 : nextSceneStorageArenaEpoch(previous.sceneEpoch),
      previous == null
        ? 1
        : previous.geometryGeneration + (replacesGeometry ? 1 : 0),
    );
    assertSceneStorageArenaFits(payload, device.limits, shardIndices);
    for (const shardIndex of shardIndices) {
      const shard = sceneStorageArenaShard(payload, shardIndex);
      this._sceneStorageArenaBuffers[shardIndex] = uploadBuffer(
        device,
        shard.bytes,
        STORAGE,
      );
    }
    this._sceneStorageArenaPayload = payload;
  }

  private _swapSceneStorageArenaShards(
    other: BvhBufferHost,
    shardIndices: readonly number[],
    device?: GPUDevice,
  ): void {
    const left = this._sceneStorageArenaPayload;
    const right = other._sceneStorageArenaPayload;
    const key = shardIndices.join(',');
    if (key !== '0,1' && key !== '2' && key !== '0,1,2') {
      throw new Error('scene-storage shards must publish as geometry, lighting, or all');
    }
    if (key === '0,1,2') {
      for (const shardIndex of shardIndices) {
        [this._sceneStorageArenaBuffers[shardIndex], other._sceneStorageArenaBuffers[shardIndex]] = [
          other._sceneStorageArenaBuffers[shardIndex] ?? null,
          this._sceneStorageArenaBuffers[shardIndex] ?? null,
        ];
      }
      [this._sceneStorageArenaPayload, other._sceneStorageArenaPayload] = [right, left];
      return;
    }
    if (left == null || right == null) {
      throw new Error('scene-storage arena must be initialized before partial publication');
    }
    if (device == null) {
      throw new Error('partial scene-storage publication requires a GPUDevice for epoch synchronization');
    }
    if (key === '2' && left.geometryGeneration !== right.geometryGeneration) {
      throw new Error(
        'lighting scene-storage shard is incompatible with retained geometry/TLAS shards',
      );
    }
    const leftSources = { ...left.sources };
    const rightSources = { ...right.sources };
    for (const shardIndex of shardIndices) {
      [
        this._sceneStorageArenaBuffers[shardIndex],
        other._sceneStorageArenaBuffers[shardIndex],
      ] = [
        other._sceneStorageArenaBuffers[shardIndex] ?? null,
        this._sceneStorageArenaBuffers[shardIndex] ?? null,
      ];
      const names = SCENE_STORAGE_SHARD_SEGMENTS[shardIndex];
      if (names == null) throw new RangeError('invalid scene-storage shard index');
      for (const name of names) {
        [leftSources[name], rightSources[name]] = [
          rightSources[name],
          leftSources[name],
        ];
      }
    }
    this._sceneStorageArenaPayload = retainSceneStorageArenaSources(
      left,
      leftSources,
      right.sceneEpoch,
      key === '2' ? left.geometryGeneration : right.geometryGeneration,
    );
    other._sceneStorageArenaPayload = retainSceneStorageArenaSources(
      right,
      rightSources,
      left.sceneEpoch,
      key === '2' ? right.geometryGeneration : left.geometryGeneration,
    );
    this._writeSceneArenaHeaders(device, this._sceneStorageArenaPayload);
  }

  private _writeSceneArenaHeaders(
    device: GPUDevice,
    payload: SceneStorageArenaPayload,
  ): void {
    for (let index = 0; index < this._sceneStorageArenaBuffers.length; index += 1) {
      const buffer = this._sceneStorageArenaBuffers[index];
      if (buffer != null) {
        device.queue.writeBuffer(buffer, 0, sceneStorageArenaHeaderBytes(payload, index));
      }
    }
  }

  private _sceneArenaTarget(
    segment: SceneStorageSegment,
    byteOffset = 0,
  ): { buffer: GPUBuffer; byteOffset: number } {
    const payload = this._sceneStorageArenaPayload;
    if (payload == null) throw new Error('scene-storage arena is not initialized');
    const layout = payload.segments[segment];
    const buffer = this._sceneStorageArenaBuffers[layout.shard];
    if (buffer == null) throw new Error('scene-storage arena shard is not initialized');
    return { buffer, byteOffset: layout.byteOffset + byteOffset };
  }

  private _sceneArenaBinding(segment: SceneStorageSegment): GPUBufferBinding {
    const payload = this._sceneStorageArenaPayload;
    if (payload == null) throw new Error('scene-storage arena is not initialized');
    const layout = payload.segments[segment];
    const target = this._sceneArenaTarget(segment);
    return {
      buffer: target.buffer,
      offset: target.byteOffset,
      // Empty emitter payloads still bind one complete EmitterTri-sized range;
      // shard padding makes that range resident without a duplicate dummy SSBO.
      size: Math.max(layout.byteLength, segment === 'emitters' ? 80 : 16),
    };
  }

  private _writeSceneArenaSegment(
    device: GPUDevice,
    segment: SceneStorageSegment,
    byteOffset: number,
    data: ArrayBuffer,
  ): void {
    const target = this._sceneArenaTarget(segment, byteOffset);
    device.queue.writeBuffer(target.buffer, target.byteOffset, data);
    const next = patchSceneStorageArenaSource(
      this._sceneStorageArenaPayload!,
      segment,
      byteOffset,
      data,
    );
    this._writeSceneArenaHeaders(device, next);
    this._sceneStorageArenaPayload = next;
  }

  uploadInitial(device: GPUDevice, bvhBuffers: SceneBVHBuffers, scene?: Scene): void {
    const candidate = new BvhBufferHost();
    candidate._regirGridBytes = this._regirGridBytes;
    try {
      candidate._uploadInitialUnsafe(device, bvhBuffers, scene);
    } catch (error) {
      candidate.dispose();
      throw error;
    }
    this._swapAllResources(candidate);
    candidate.dispose();
  }

  private _uploadInitialUnsafe(device: GPUDevice, bvhBuffers: SceneBVHBuffers, scene?: Scene): void {
    this._uploadSceneStorageArena(
      device,
      sceneStorageSegmentSources(bvhBuffers),
      [0, 1, 2],
    );
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
    this._materialTextureAtlas = uploadMaterialTextureAtlas(device, bvhBuffers.materialTextureAtlas);
    // WS1 — per-vertex world-space normals (stride-4 vec4f, .w unused). Same
    // data the DDGI / emitter paths already use (shared.normals).
    this._bvhTangentTexture = uploadTangentTexture(
      device,
      new Float32Array(bvhBuffers.bvhTangents.cpuData),
      bvhBuffers.bvhTangents.count,
    );
    this._bvhVertexColorTexture = uploadVertexColorTexture(
      device,
      new Float32Array(bvhBuffers.bvhColors.cpuData),
      bvhBuffers.bvhColors.count,
    );
    const emitterCount = validateEmitterPayload('uploadInitial', bvhBuffers.emitters);
    if (bvhBuffers.emitterCount !== emitterCount) {
      throw new RangeError(
        `[BvhBufferHost] uploadInitial emitterCount ${bvhBuffers.emitterCount} does not match ` +
        `emitters.count ${emitterCount}.`,
      );
    }
    this._emitterCount = emitterCount;
    // Combined light-tree + ReGIR-grid buffer (tree nodes in front, grid region
    // zeroed at the tail). `_regirGridBytes == 0` ⇒ exactly `uploadBuffer`.
    this._lightTreeBuffer = uploadBufferPadded(
      device, bvhBuffers.lightTree.cpuData, this._regirGridBytes, STORAGE);
    const analyticPacked = scene != null
      ? packAnalyticPointSpotEmitters(scene)
      : { data: new Float32Array(0), count: 0 };
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
    const next = uploadAnalyticLightsTexture(device, packed.data, packed.count);
    const previous = this._analyticLightsTexture;
    this._analyticLightsTexture = next;
    previous?.texture.destroy();
  }

  /**
   * Stage every pipeline-owned resource affected by an emitter mutation:
   * emitter triangles/CDF/light tree, camera-visible emissive radiance, and the
   * analytic point/spot texture.
   */
  prepareEmitterLightingReplacement(
    device: GPUDevice,
    bvhBuffers: Pick<
      SceneBVHBuffers,
      'emitters' | 'emitterCdf' | 'emitterAlias' | 'lightTree' | 'bvhEmissiveLe'
    >,
    scene: Scene,
  ): PreparedSceneMutation {
    const analyticPacked = packAnalyticPointSpotEmitters(scene);
    const candidate = new BvhBufferHost();
    candidate._regirGridBytes = this._regirGridBytes;
    candidate._sceneStorageArenaPayload = this._sceneStorageArenaPayload;
    try {
      candidate._updateEmittersUnsafe(device, bvhBuffers);
      candidate._bvhEmissiveTriCount = bvhBuffers.bvhEmissiveLe.count;
      candidate._bvhEmissiveTexture = uploadEmissiveTexture(
        device,
        new Float32Array(bvhBuffers.bvhEmissiveLe.cpuData),
        candidate._bvhEmissiveTriCount,
      );
      candidate._analyticLightsTexture = uploadAnalyticLightsTexture(
        device,
        analyticPacked.data,
        analyticPacked.count,
      );
    } catch (error) {
      candidate.dispose();
      throw error;
    }

    const swap = (): void => {
      this._swapSceneStorageArenaShards(candidate, [2], device);
      [this._emitterCount, candidate._emitterCount] =
        [candidate._emitterCount, this._emitterCount];
      [this._lightTreeBuffer, candidate._lightTreeBuffer] =
        [candidate._lightTreeBuffer, this._lightTreeBuffer];
      [this._bvhEmissiveTexture, candidate._bvhEmissiveTexture] =
        [candidate._bvhEmissiveTexture, this._bvhEmissiveTexture];
      [this._bvhEmissiveTriCount, candidate._bvhEmissiveTriCount] =
        [candidate._bvhEmissiveTriCount, this._bvhEmissiveTriCount];
      [this._analyticLightsTexture, candidate._analyticLightsTexture] =
        [candidate._analyticLightsTexture, this._analyticLightsTexture];
    };

    let committed = false;
    let closed = false;
    return {
      commit: () => {
        if (closed || committed) return;
        swap();
        committed = true;
      },
      rollback: () => {
        if (closed) return;
        if (committed) swap();
        candidate.dispose();
        closed = true;
      },
      finalize: () => {
        if (closed) return;
        candidate.dispose();
        closed = true;
      },
    };
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
  emitterBufferAndCount(): (GPUBufferBinding & { count: number }) | null {
    if (!this.initialized) return null;
    return { ...this._sceneArenaBinding('emitters'), count: this._emitterCount };
  }

  /** Smallest aligned contiguous window containing emitter records and their
   * RC alias table. Relative offsets are encoded into CascadeUniforms. */
  emitterSamplingBufferAndCount(): (GPUBufferBinding & {
    count: number;
    emitterDataOffset: number;
    emitterAliasOffset: number;
  }) | null {
    if (!this.initialized || this._sceneStorageArenaPayload == null) return null;
    const emitters = this._sceneStorageArenaPayload.segments.emitters;
    const alias = this._sceneStorageArenaPayload.segments.emitterAlias;
    if (emitters.shard !== alias.shard) {
      throw new Error('[BvhBufferHost] emitter records and alias table must share one arena shard');
    }
    const buffer = this._sceneStorageArenaBuffers[emitters.shard];
    if (buffer == null) throw new Error('[BvhBufferHost] scene-storage arena shard is not initialized');
    const aliasEnd = alias.byteOffset + Math.max(alias.byteLength, 16);
    return {
      buffer,
      offset: emitters.byteOffset,
      size: aliasEnd - emitters.byteOffset,
      count: this._emitterCount,
      emitterDataOffset: 0,
      emitterAliasOffset: alias.byteOffset - emitters.byteOffset,
    };
  }

  /** A7 (2026-06-10): equirectangular env map texture + sampler for RC probe
   *  env sampling. Returns null before `uploadInitial`. The map is the
   *  placeholder (black, zero intensity) when no HDRI has been uploaded; RC
   *  still binds it, but `hasDirectionalEnvironment=false` selects the same
   *  flat scalar-sky fallback used by the main/ReSTIR miss paths. */
  envBindings(): {
    textureView: GPUTextureView;
    sampler: GPUSampler;
    rotationY: number;
    intensity: number;
    hasDirectionalEnvironment: boolean;
  } | null {
    if (this._env == null) return null;
    return {
      textureView: this._resourceCache.textureView(this._env.map),
      sampler: this._env.sampler,
      rotationY: this._env.rotationY,
      intensity: this._env.intensity,
      hasDirectionalEnvironment: this._env.hasDirectionalEnvironment,
    };
  }

  /** Material atlas views for RC/DDGI side-channel material sampling. Null before
   *  `uploadInitial`. These are the exact views used by the main scene bind
   *  group, so RC material-backed emitter NEE sees the same texture decode,
   *  wrap, and transform metadata as shade/ReSTIR. */
  materialAtlasBindings(): {
    materialTextureAtlasView: GPUTextureView;
    materialMapMetaTextureView: GPUTextureView;
    bvhTangentTextureView: GPUTextureView;
    bvhVertexColorTextureView: GPUTextureView;
  } | null {
    if (this._materialTextureAtlas == null) return null;
    return {
      materialTextureAtlasView: this._materialTextureAtlas.atlasTextureView,
      materialMapMetaTextureView: this._materialTextureAtlas.baseColorMetaTextureView,
      bvhTangentTextureView: this._resourceCache.textureView(this._bvhTangentTexture!.texture),
      bvhVertexColorTextureView: this._resourceCache.textureView(this._bvhVertexColorTexture!.texture),
    };
  }

  sceneBindGroupResources(): SceneBindGroupResources {
    if (!this.initialized) {
      throw new Error('[BvhBufferHost] uploadInitial must run before sceneBindGroupResources');
    }
    return {
      sceneStorageArenaBuffers: [
        this._sceneStorageArenaBuffers[0]!,
        this._sceneStorageArenaBuffers[1]!,
        this._sceneStorageArenaBuffers[2]!,
      ],
      bvhBeerTextureView: this._resourceCache.textureView(this._bvhBeerTexture!.texture),
      bvhEmissiveTextureView: this._resourceCache.textureView(this._bvhEmissiveTexture!.texture),
      bvhRoughMetalTextureView: this._resourceCache.textureView(this._bvhRoughMetalTexture!.texture),
      materialTextureAtlasView: this._materialTextureAtlas!.atlasTextureView,
      baseColorMapMetaTextureView: this._materialTextureAtlas!.baseColorMetaTextureView,
      bvhTangentTextureView: this._resourceCache.textureView(this._bvhTangentTexture!.texture),
      bvhVertexColorTextureView: this._resourceCache.textureView(this._bvhVertexColorTexture!.texture),
      analyticLightsTextureView: this._resourceCache.textureView(this._analyticLightsTexture!.texture),
      envMapTextureView: this._resourceCache.textureView(this._env!.map),
      envMarginalTextureView: this._resourceCache.textureView(this._env!.marginal),
      envConditionalTextureView: this._resourceCache.textureView(this._env!.conditional),
      envParamsBuffer: this._env!.paramsBuffer,
    };
  }

  gpuMemorySections(): GpuMemoryExternalSections {
    const section: Record<string, unknown> = {};
    const add = (name: string, resource: unknown): void => {
      if (resource != null) section[name] = resource;
    };

    add('lightTreeBuffer', this._lightTreeBuffer);
    this._sceneStorageArenaBuffers.forEach((buffer, index) => {
      add('sceneStorageArenaShard' + index, buffer);
    });

    // Single-layer texture memory sections — all share the
    // {width, height, depthOrArrayLayers: 1, format} shape; the descriptor
    // table keeps this list drift-free with dispose/transactional replacement.
    const addTex = (
      name: string,
      tex: { width: number; height: number } | null,
      format: GPUTextureFormat,
    ): void => {
      if (tex != null) {
        section[name] = { width: tex.width, height: tex.height, depthOrArrayLayers: 1, format };
      }
    };
    addTex('bvhBeerTexture', this._bvhBeerTexture, 'r32uint');
    addTex('bvhEmissiveTexture', this._bvhEmissiveTexture, 'rgba32float');
    addTex('bvhRoughMetalTexture', this._bvhRoughMetalTexture, 'r32uint');
    if (this._materialTextureAtlas != null) {
      section.materialTextureAtlas = {
        width: this._materialTextureAtlas.atlasDim,
        height: this._materialTextureAtlas.atlasDim,
        depthOrArrayLayers: this._materialTextureAtlas.atlasLayerCount,
        format: 'rgba32float' as GPUTextureFormat,
      };
      section.baseColorMapMetaTexture = {
        width: this._materialTextureAtlas.baseColorMetaWidth,
        height: this._materialTextureAtlas.baseColorMetaHeight,
        depthOrArrayLayers: 1,
        format: 'rgba32float' as GPUTextureFormat,
      };
    }
    addTex('bvhTangentTexture', this._bvhTangentTexture, 'rgba32float');
    addTex('bvhVertexColorTexture', this._bvhVertexColorTexture, 'rgba32float');
    addTex('analyticLightsTexture', this._analyticLightsTexture, 'rgba32float');

    return { staticScene: section };
  }

  updateEmitters(
    device: GPUDevice,
    bvhBuffers: Pick<SceneBVHBuffers, 'emitters' | 'emitterCdf' | 'emitterAlias' | 'lightTree'>,
  ): void {
    const candidate = new BvhBufferHost();
    candidate._regirGridBytes = this._regirGridBytes;
    candidate._sceneStorageArenaPayload = this._sceneStorageArenaPayload;
    try {
      candidate._updateEmittersUnsafe(device, bvhBuffers);
    } catch (error) {
      candidate.dispose();
      throw error;
    }
    this._swapSceneStorageArenaShards(candidate, [2], device);
    [this._emitterCount, candidate._emitterCount] = [candidate._emitterCount, this._emitterCount];
    [this._lightTreeBuffer, candidate._lightTreeBuffer] = [candidate._lightTreeBuffer, this._lightTreeBuffer];
    candidate.dispose();
  }

  private _updateEmittersUnsafe(
    device: GPUDevice,
    bvhBuffers: Pick<SceneBVHBuffers, 'emitters' | 'emitterCdf' | 'emitterAlias' | 'lightTree'>,
  ): void {
    const nextEmitterCount = validateEmitterPayload('updateEmitters', bvhBuffers.emitters);
    this._uploadSceneStorageArena(
      device,
      sceneLightingStorageSources(bvhBuffers),
      [2],
    );
    // updateEmitters' Pick omits SceneBVHBuffers.emitterCount, but the
    // StorageBufferHandle carries the canonical count and has already been
    // byte-validated against EMITTER_TRI_STRIDE_BYTES above.
    this._emitterCount = nextEmitterCount;
    // Re-upload the selection tree: emitters changed, so the tree's leaf
    // emitterIndex → emitter array mapping (and powers) changed with them.
    // Re-pad for the ReGIR grid region (zeroed; the grid-build pass refills it
    // next frame). The tree node count may have changed, so the grid region's
    // float offset (lightTreeNodeCount × LIGHT_TREE_FLOATS_PER_NODE = ×16) is recomputed by the pipeline.
    this._lightTreeBuffer = uploadBufferPadded(
      device, bvhBuffers.lightTree.cpuData, this._regirGridBytes, STORAGE);
  }

  replaceBvhAndEmitters(device: GPUDevice, bvhBuffers: SceneBVHBuffers): void {
    if (!this.initialized) return;
    const candidate = new BvhBufferHost();
    candidate._regirGridBytes = this._regirGridBytes;
    candidate._sceneStorageArenaPayload = this._sceneStorageArenaPayload;
    try {
      candidate._uploadGeometryReplacementUnsafe(device, bvhBuffers);
      candidate._updateEmittersUnsafe(device, bvhBuffers);
    } catch (error) {
      candidate.dispose();
      throw error;
    }
    this._swapGeometryResources(candidate, false);
    this._swapSceneStorageArenaShards(candidate, [0, 1, 2]);
    [this._emitterCount, candidate._emitterCount] = [candidate._emitterCount, this._emitterCount];
    [this._lightTreeBuffer, candidate._lightTreeBuffer] = [candidate._lightTreeBuffer, this._lightTreeBuffer];
    candidate.dispose();
  }

  prepareBvhAndEmitterReplacement(
    device: GPUDevice,
    bvhBuffers: SceneBVHBuffers,
  ): PreparedSceneMutation {
    const candidate = new BvhBufferHost();
    candidate._regirGridBytes = this._regirGridBytes;
    candidate._sceneStorageArenaPayload = this._sceneStorageArenaPayload;
    try {
      candidate._uploadGeometryReplacementUnsafe(device, bvhBuffers);
      candidate._updateEmittersUnsafe(device, bvhBuffers);
    } catch (error) {
      candidate.dispose();
      throw error;
    }
    const swap = (): void => {
      this._swapGeometryResources(candidate, false);
      this._swapSceneStorageArenaShards(candidate, [0, 1, 2]);
      [this._emitterCount, candidate._emitterCount] =
        [candidate._emitterCount, this._emitterCount];
      [this._lightTreeBuffer, candidate._lightTreeBuffer] =
        [candidate._lightTreeBuffer, this._lightTreeBuffer];
    };
    let committed = false;
    let closed = false;
    return {
      commit: () => {
        if (closed || committed) return;
        swap();
        committed = true;
      },
      rollback: () => {
        if (closed) return;
        if (committed) swap();
        candidate.dispose();
        closed = true;
      },
      finalize: () => {
        if (closed) return;
        candidate.dispose();
        closed = true;
      },
    };
  }

  /**
   * Prepare an incremental BVH publication without touching any live resource.
   * Buffer overwrites are encoded from private staging buffers. Resource-size
   * changes use private candidate handles. The caller must make its single
   * queue.submit the final irreversible transaction step.
   */
  prepareMutation(
    device: GPUDevice,
    encoder: GPUCommandEncoder,
    mutation: CollectedBvhMutation,
  ): PreparedSceneMutation {
    if (!this.initialized) {
      return { commit: () => undefined, rollback: () => undefined, finalize: () => undefined };
    }
    if (mutation.replacement) {
      return this.prepareBvhAndEmitterReplacement(device, mutation.replacement);
    }

    const staging: GPUBuffer[] = [];
    const candidate = new BvhBufferHost();
    candidate._regirGridBytes = this._regirGridBytes;
    let hasBeer = false;
    let hasEmissive = false;
    let hasRoughMetal = false;
    let hasAtlas = false;
    let hasEmitters = false;
    const arenaPrevious = this._sceneStorageArenaPayload;
    let arenaNext = arenaPrevious;
    const arenaPatches: SceneStorageArenaSourcePatch[] = [];
    const arenaMutationEpoch = arenaPrevious == null
      ? 1
      : nextSceneStorageArenaEpoch(arenaPrevious.sceneEpoch);
    let arenaChanged = false;
    candidate._sceneStorageArenaPayload = arenaPrevious;

    const stageCopy = (
      destination: GPUBuffer | null,
      destinationOffset: number,
      data: ArrayBuffer,
    ): void => {
      if (destination == null) throw new Error('BVH mutation destination is not initialized.');
      if ((destinationOffset & 3) !== 0 || (data.byteLength & 3) !== 0) {
        throw new RangeError('BVH staged copies require four-byte aligned offsets and sizes.');
      }
      const upload = device.createBuffer({
        label: 'bvh-mutation-staging',
        size: Math.max(4, data.byteLength),
        usage: 0x4,
        mappedAtCreation: true,
      });
      staging.push(upload);
      new Uint8Array(upload.getMappedRange()).set(new Uint8Array(data));
      upload.unmap();
      encoder.copyBufferToBuffer(upload, 0, destination, destinationOffset, data.byteLength);
    };
    const stageSceneArenaCopy = (
      segment: SceneStorageSegment,
      byteOffset: number,
      data: ArrayBuffer,
    ): void => {
      if (arenaNext == null) throw new Error('scene-storage arena is not initialized');
      const target = this._sceneArenaTarget(segment, byteOffset);
      stageCopy(target.buffer, target.byteOffset, data);
      arenaPatches.push({ segment, byteOffset, data });
      arenaChanged = true;
    };

    try {
      for (const nodes of mutation.nodes ?? []) {
        stageSceneArenaCopy(
          'bvhNodes',
          nodes.byteOffset,
          nodes.data,
        );
      }
      for (const position of mutation.positions ?? []) {
        stageSceneArenaCopy('bvhPositions', position.byteOffset, position.data);
      }
      for (const position of mutation.learningPositions ?? []) {
        stageSceneArenaCopy('bvhPositions', position.byteOffset, position.data);
      }
      for (const normal of mutation.normals ?? []) {
        stageSceneArenaCopy('bvhNormals', normal.byteOffset, normal.data);
      }
      if (mutation.tlas) {
        for (const nodes of mutation.tlas.nodes) {
          stageSceneArenaCopy('tlasNodes', nodes.byteOffset, nodes.data);
        }
        for (const matrix of mutation.tlas.worldToLocal) {
          stageSceneArenaCopy(
            'tlasInstanceWorldToLocal',
            matrix.byteOffset,
            matrix.data,
          );
        }
        for (const matrix of mutation.tlas.localToWorld) {
          stageSceneArenaCopy(
            'tlasInstanceLocalToWorld',
            matrix.byteOffset,
            matrix.data,
          );
        }
      }
      if (mutation.material) {
        stageSceneArenaCopy(
          'bvhIndex',
          mutation.material.index.byteOffset,
          mutation.material.index.data,
        );
        candidate._bvhBeerTriCount = mutation.material.beer.triCount;
        candidate._bvhBeerTexture = uploadBeerTexture(
          device,
          mutation.material.beer.data,
          mutation.material.beer.triCount,
        );
        hasBeer = true;
        candidate._bvhEmissiveTriCount = mutation.material.emissive.triCount;
        candidate._bvhEmissiveTexture = uploadEmissiveTexture(
          device,
          new Float32Array(mutation.material.emissive.data),
          mutation.material.emissive.triCount,
        );
        hasEmissive = true;
        if (mutation.material.roughMetal) {
          candidate._bvhRoughMetalTriCount = mutation.material.roughMetal.triCount;
          candidate._bvhRoughMetalTexture = uploadBeerTexture(
            device,
            mutation.material.roughMetal.data,
            mutation.material.roughMetal.triCount,
          );
          hasRoughMetal = true;
        }
      }
      if (mutation.atlas) {
        candidate._materialTextureAtlas = uploadMaterialTextureAtlas(device, mutation.atlas);
        hasAtlas = true;
      }
      if (arenaPatches.length > 0) {
        if (arenaPrevious == null) {
          throw new Error('scene-storage arena is not initialized');
        }
        arenaNext = patchSceneStorageArenaSources(
          arenaPrevious,
          arenaPatches,
          arenaMutationEpoch,
        );
        candidate._sceneStorageArenaPayload = arenaNext;
      }
      if (mutation.emitters) {
        candidate._updateEmittersUnsafe(device, mutation.emitters);
        hasEmitters = true;
      }
      if (arenaChanged) {
        if (arenaNext == null) {
          throw new Error('scene-storage arena is not initialized');
        }
        for (let index = 0; index < this._sceneStorageArenaBuffers.length; index += 1) {
          stageCopy(
            this._sceneStorageArenaBuffers[index] ?? null,
            0,
            sceneStorageArenaHeaderBytes(arenaNext, index),
          );
        }
      }
    } catch (error) {
      rethrowWithSceneMutationCleanup(
        error,
        [
          ...staging.map((buffer) => () => buffer.destroy()),
          () => candidate.dispose(),
        ],
        'BVH mutation preparation failed and candidate cleanup also failed',
      );
    }

    const swapCandidates = (): void => {
      if (hasBeer) {
        [this._bvhBeerTexture, candidate._bvhBeerTexture] =
          [candidate._bvhBeerTexture, this._bvhBeerTexture];
        [this._bvhBeerTriCount, candidate._bvhBeerTriCount] =
          [candidate._bvhBeerTriCount, this._bvhBeerTriCount];
      }
      if (hasEmissive) {
        [this._bvhEmissiveTexture, candidate._bvhEmissiveTexture] =
          [candidate._bvhEmissiveTexture, this._bvhEmissiveTexture];
        [this._bvhEmissiveTriCount, candidate._bvhEmissiveTriCount] =
          [candidate._bvhEmissiveTriCount, this._bvhEmissiveTriCount];
      }
      if (hasRoughMetal) {
        [this._bvhRoughMetalTexture, candidate._bvhRoughMetalTexture] =
          [candidate._bvhRoughMetalTexture, this._bvhRoughMetalTexture];
        [this._bvhRoughMetalTriCount, candidate._bvhRoughMetalTriCount] =
          [candidate._bvhRoughMetalTriCount, this._bvhRoughMetalTriCount];
      }
      if (hasAtlas) {
        [this._materialTextureAtlas, candidate._materialTextureAtlas] =
          [candidate._materialTextureAtlas, this._materialTextureAtlas];
      }
      if (hasEmitters) {
        this._swapSceneStorageArenaShards(candidate, [2], device);
        [this._emitterCount, candidate._emitterCount] =
          [candidate._emitterCount, this._emitterCount];
        [this._lightTreeBuffer, candidate._lightTreeBuffer] =
          [candidate._lightTreeBuffer, this._lightTreeBuffer];
      }
    };
    const releaseStaging = (): void => {
      runSceneMutationCleanups(
        staging.map((buffer) => () => buffer.destroy()),
        'BVH mutation staging-buffer retirement failed',
      );
    };
    let committed = false;
    let closed = false;
    return {
      commit: () => {
        if (closed || committed) return;
        this._sceneStorageArenaPayload = arenaNext;
        swapCandidates();
        committed = true;
      },
      rollback: () => {
        if (closed) return;
        closed = true;
        const cleanups: SceneMutationCleanup[] = [];
        if (committed) {
          cleanups.push(() => {
            swapCandidates();
            this._sceneStorageArenaPayload = arenaPrevious;
          });
        }
        cleanups.push(releaseStaging, () => candidate.dispose());
        runSceneMutationCleanups(cleanups, 'BVH mutation rollback failed');
      },
      finalize: () => {
        if (closed) return;
        closed = true;
        runSceneMutationCleanups(
          [releaseStaging, () => candidate.dispose()],
          'BVH mutation retirement failed',
        );
      },
    };
  }
  refreshBvhRefit(
    device: GPUDevice,
    bvhNodesBytes: ArrayBuffer,
    positionsSlice: { byteOffset: number; data: ArrayBuffer },
    bvhNodesByteOffset = 0,
  ): void {
    if (!this.initialized) return;
    this._writeSceneArenaSegment(
      device,
      'bvhNodes',
      bvhNodesByteOffset,
      bvhNodesBytes,
    );
    this._writeSceneArenaSegment(
      device,
      'bvhPositions',
      positionsSlice.byteOffset,
      positionsSlice.data,
    );
  }

  refreshBvhNodesOnly(
    device: GPUDevice,
    bvhNodesBytes: ArrayBuffer,
    bvhNodesByteOffset = 0,
  ): void {
    if (!this.initialized) return;
    this._writeSceneArenaSegment(
      device,
      'bvhNodes',
      bvhNodesByteOffset,
      bvhNodesBytes,
    );
  }

  /** H19 — upload a per-vertex normals slice (vec4f stride, .w unused). */
  refreshBvhNormalsSlice(
    device: GPUDevice,
    normalsSlice: { byteOffset: number; data: ArrayBuffer },
  ): void {
    if (!this.initialized) return;
    this._writeSceneArenaSegment(
      device,
      'bvhNormals',
      normalsSlice.byteOffset,
      normalsSlice.data,
    );
  }

  getBvhPositionBuffer(): GPUBuffer | null {
    return this.initialized ? this._sceneArenaBinding('bvhPositions').buffer : null;
  }

  getBvhPositionBinding(): GPUBufferBinding | null {
    return this.initialized ? this._sceneArenaBinding('bvhPositions') : null;
  }

  /** WS1 — live merged per-vertex normal buffer. The GPU-skin kernel writes
   *  inverse-transpose skinned normals directly into it at `baseVertex+vi`. */
  getBvhNormalBuffer(): GPUBuffer | null {
    return this.initialized ? this._sceneArenaBinding('bvhNormals').buffer : null;
  }

  getBvhNormalBinding(): GPUBufferBinding | null {
    return this.initialized ? this._sceneArenaBinding('bvhNormals') : null;
  }

  /** Shared BVH geometry ranges for RC; eliminates RC's duplicate geometry upload. */
  sceneGeometryBufferBindings(): SceneGeometryBufferBindings | null {
    if (!this.initialized) return null;
    return {
      bvhNodes: this._sceneArenaBinding('bvhNodes'),
      bvhIndices: this._sceneArenaBinding('bvhIndex'),
      bvhPositions: this._sceneArenaBinding('bvhPositions'),
      bvhNormals: this._sceneArenaBinding('bvhNormals'),
    };
  }

  refreshTlasRefit(
    device: GPUDevice,
    mutation: import('./BvhUpdateSink.js').TlasRefitMutation,
  ): void {
    if (!this.initialized) return;
    for (const nodes of mutation.nodes) {
      this._writeSceneArenaSegment(
        device,
        'tlasNodes',
        nodes.byteOffset,
        nodes.data,
      );
    }
    for (const matrix of mutation.worldToLocal) {
      this._writeSceneArenaSegment(
        device,
        'tlasInstanceWorldToLocal',
        matrix.byteOffset,
        matrix.data,
      );
    }
    for (const matrix of mutation.localToWorld) {
      this._writeSceneArenaSegment(
        device,
        'tlasInstanceLocalToWorld',
        matrix.byteOffset,
        matrix.data,
      );
    }
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
    this._writeSceneArenaSegment(
      device,
      'bvhIndex',
      indexSlice.byteOffset,
      indexSlice.data,
    );
    refreshBeerTexture(device, this._bvhBeerTexture!, beerFull.data, beerFull.triCount);
    refreshEmissiveTexture(
      device, this._bvhEmissiveTexture!, new Float32Array(emissiveFull.data), emissiveFull.triCount);
    if (roughMetalFull && this._bvhRoughMetalTexture) {
      refreshBeerTexture(
        device, this._bvhRoughMetalTexture, roughMetalFull.data, roughMetalFull.triCount);
    }
  }

  refreshMaterialTextureAtlas(
    device: GPUDevice,
    materialTextureAtlas: MaterialTextureAtlasPayload,
  ): void {
    if (!this.initialized) return;
    const next = uploadMaterialTextureAtlas(device, materialTextureAtlas);
    const previous = this._materialTextureAtlas;
    this._materialTextureAtlas = next;
    previous?.atlasTexture.destroy();
    previous?.baseColorMetaTexture.destroy();
  }

  private _uploadGeometryReplacementUnsafe(
    device: GPUDevice,
    bvhBuffers: Pick<
      SceneBVHBuffers,
      'bvhNodes' | 'bvhIndex' | 'bvhBeerColors' | 'bvhEmissiveLe' | 'materialTextureAtlas' | 'bvhRoughMetal' | 'bvhNormals' | 'bvhTangents' | 'bvhColors' | 'bvhPositions' | 'bvhMode' | 'tlas'
  >,
  ): void {
    this._uploadSceneStorageArena(
      device,
      sceneGeometryStorageSources(bvhBuffers),
      [0, 1],
    );
    this._bvhBeerTriCount = bvhBuffers.bvhBeerColors.count;
    this._bvhBeerTexture = uploadBeerTexture(
      device, bvhBuffers.bvhBeerColors.cpuData, this._bvhBeerTriCount);
    this._bvhRoughMetalTriCount = bvhBuffers.bvhRoughMetal.count;
    this._bvhRoughMetalTexture = uploadBeerTexture(
      device, bvhBuffers.bvhRoughMetal.cpuData, this._bvhRoughMetalTriCount);
    this._materialTextureAtlas = uploadMaterialTextureAtlas(device, bvhBuffers.materialTextureAtlas);
    this._bvhEmissiveTriCount = bvhBuffers.bvhEmissiveLe.count;
    this._bvhEmissiveTexture = uploadEmissiveTexture(
      device, new Float32Array(bvhBuffers.bvhEmissiveLe.cpuData), this._bvhEmissiveTriCount);
    this._bvhTangentTexture = uploadTangentTexture(
      device,
      new Float32Array(bvhBuffers.bvhTangents.cpuData),
      bvhBuffers.bvhTangents.count,
    );
    this._bvhVertexColorTexture = uploadVertexColorTexture(
      device,
      new Float32Array(bvhBuffers.bvhColors.cpuData),
      bvhBuffers.bvhColors.count,
    );
  }

  dispose(): void {
    for (const buffer of this._sceneStorageArenaBuffers) destroyResource(buffer);
    destroyResource(this._bvhBeerTexture?.texture);
    destroyResource(this._bvhEmissiveTexture?.texture);
    destroyResource(this._bvhRoughMetalTexture?.texture);
    destroyResource(this._materialTextureAtlas?.atlasTexture);
    destroyResource(this._materialTextureAtlas?.baseColorMetaTexture);
    destroyResource(this._bvhTangentTexture?.texture);
    destroyResource(this._bvhVertexColorTexture?.texture);
    destroyResource(this._lightTreeBuffer);
    destroyResource(this._analyticLightsTexture?.texture);
    if (this._env != null) {
      destroyResource(this._env.map);
      destroyResource(this._env.marginal);
      destroyResource(this._env.conditional);
      destroyResource(this._env.paramsBuffer);
      this._env = null;
    }
    this._sceneStorageArenaBuffers = [null, null, null];
    this._sceneStorageArenaPayload = null;
    this._bvhBeerTexture = null;
    this._bvhEmissiveTexture = null;
    this._bvhRoughMetalTexture = null;
    this._materialTextureAtlas = null;
    this._bvhTangentTexture = null;
    this._bvhVertexColorTexture = null;
    this._lightTreeBuffer = null;
    this._analyticLightsTexture = null;
  }

  private _swapGeometryResources(
    other: BvhBufferHost,
    includeArena = true,
    device?: GPUDevice,
  ): void {
    if (includeArena) this._swapSceneStorageArenaShards(other, [0, 1], device);
    [this._bvhBeerTexture, other._bvhBeerTexture] = [other._bvhBeerTexture, this._bvhBeerTexture];
    [this._bvhBeerTriCount, other._bvhBeerTriCount] = [other._bvhBeerTriCount, this._bvhBeerTriCount];
    [this._bvhEmissiveTexture, other._bvhEmissiveTexture] = [other._bvhEmissiveTexture, this._bvhEmissiveTexture];
    [this._bvhEmissiveTriCount, other._bvhEmissiveTriCount] = [other._bvhEmissiveTriCount, this._bvhEmissiveTriCount];
    [this._bvhRoughMetalTexture, other._bvhRoughMetalTexture] = [other._bvhRoughMetalTexture, this._bvhRoughMetalTexture];
    [this._bvhRoughMetalTriCount, other._bvhRoughMetalTriCount] = [other._bvhRoughMetalTriCount, this._bvhRoughMetalTriCount];
    [this._materialTextureAtlas, other._materialTextureAtlas] = [other._materialTextureAtlas, this._materialTextureAtlas];
    [this._bvhTangentTexture, other._bvhTangentTexture] = [other._bvhTangentTexture, this._bvhTangentTexture];
    [this._bvhVertexColorTexture, other._bvhVertexColorTexture] = [other._bvhVertexColorTexture, this._bvhVertexColorTexture];
  }

  private _swapAllResources(other: BvhBufferHost): void {
    this._swapGeometryResources(other, false);
    this._swapSceneStorageArenaShards(other, [0, 1, 2]);
    [this._emitterCount, other._emitterCount] = [other._emitterCount, this._emitterCount];
    [this._lightTreeBuffer, other._lightTreeBuffer] = [other._lightTreeBuffer, this._lightTreeBuffer];
    [this._analyticLightsTexture, other._analyticLightsTexture] = [other._analyticLightsTexture, this._analyticLightsTexture];
    [this._env, other._env] = [other._env, this._env];
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
