/**
 * Atlas + border-scratch GPUTexture cache for probe updates (W4c).
 */
import type { AtlasTextureSlot } from './probeGrid.js';
import type { ProbeUpdateGpuState } from './probeUpdateGpuState.js';

export interface AtlasCohortReplacementTransaction {
  commit(): void;
  rollback(): void;
  finalize(): void;
}

export class ProbeUpdateAtlasTextureCache {
  private _textureCache = new WeakMap<AtlasTextureSlot, GPUTexture>();
  private _trackedCacheTextures = new Set<GPUTexture>();
  private _visScratchSize = '';

  getOrCreateAtlasTexture(
    device: GPUDevice,
    slot: AtlasTextureSlot,
    format: GPUTextureFormat,
  ): GPUTexture {
    const cached = this._textureCache.get(slot);
    if (cached) return cached;

    const gpuTex = device.createTexture({
      size: [slot.width, slot.height, 1],
      format,
      usage:
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_SRC |
        GPUTextureUsage.COPY_DST,
    });

    try {
      this._textureCache.set(slot, gpuTex);
      this._trackedCacheTextures.add(gpuTex);
      return gpuTex;
    } catch (error) {
      this._textureCache.delete(slot);
      this._trackedCacheTextures.delete(gpuTex);
      try { gpuTex.destroy(); } catch { /* preserve publication failure */ }
      throw error;
    }
  }

  getOrCreateScratchTexture(
    device: GPUDevice,
    gpu: ProbeUpdateGpuState,
    atlas: GPUTexture,
  ): GPUTexture {
    const sizeTag = `${atlas.width}|${atlas.height}`;
    if (gpu.visScratchTex && this._visScratchSize === sizeTag) return gpu.visScratchTex;
    const previous = gpu.visScratchTex;
    const candidate = device.createTexture({
      size: [atlas.width, atlas.height, 1],
      format: 'rgba16float',
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_SRC |
        GPUTextureUsage.COPY_DST,
    });
    if (candidate === previous) {
      throw new Error('DDGI visibility scratch candidate aliases the live texture.');
    }
    gpu.visScratchTex = candidate;
    this._visScratchSize = sizeTag;
    try { previous?.destroy(); } catch { /* candidate is already published */ }
    return candidate;
  }

  getCachedAtlas(slot: AtlasTextureSlot): GPUTexture | undefined {
    return this._textureCache.get(slot);
  }

  /**
   * Retire the GPU textures owned by atlas slots that the ProbeGrid is
   * replacing. WeakMap keys disappearing does not release their GPUTexture
   * values while `_trackedCacheTextures` still owns them, so grid resize must
   * explicitly close that ownership edge.
   */
  retireAtlasSlots(slots: readonly AtlasTextureSlot[]): void {
    const retired = new Set<GPUTexture>();
    for (const slot of slots) {
      const texture = this._textureCache.get(slot);
      this._textureCache.delete(slot);
      if (texture == null || retired.has(texture)) continue;
      retired.add(texture);
      this._trackedCacheTextures.delete(texture);
      try { texture.destroy(); } catch { /* continue retiring the old cohort */ }
    }
  }

  /** Publish a prepared atlas cohort as one recoverable cache transaction. */
  replaceCachedAtlasCohort(
    replacements: readonly {
      readonly slot: AtlasTextureSlot;
      readonly texture: GPUTexture;
    }[],
  ): void {
    const transaction = this.prepareCachedAtlasCohort(replacements);
    try {
      transaction.commit();
    } catch (error) {
      transaction.rollback();
      throw error;
    }
    transaction.finalize();
  }

  /**
   * Stage a reversible atlas-cache publication. Candidate textures become
   * cache-owned immediately; rollback/finalize retire the losing generation.
   */
  prepareCachedAtlasCohort(
    replacements: readonly {
      readonly slot: AtlasTextureSlot;
      readonly texture: GPUTexture;
    }[],
  ): AtlasCohortReplacementTransaction {
    const previous = replacements.map(({ slot }) => this._textureCache.get(slot));
    const live = new Set(previous.filter((texture): texture is GPUTexture => texture != null));
    const candidates = new Set<GPUTexture>();
    const slots = new Set<AtlasTextureSlot>();
    for (const { slot, texture } of replacements) {
      if (
        slots.has(slot) ||
        live.has(texture) ||
        candidates.has(texture)
      ) {
        throw new Error(
          'DDGI atlas replacement duplicates a slot or aliases a live/sibling texture.',
        );
      }
      slots.add(slot);
      candidates.add(texture);
    }
    for (const candidate of candidates) {
      this._trackedCacheTextures.add(candidate);
    }

    let state: 'prepared' | 'committed' | 'closed' = 'prepared';
    const restorePrevious = (): void => {
      for (let index = 0; index < replacements.length; index += 1) {
        const replacement = replacements[index]!;
        const prior = previous[index];
        if (prior != null) this._textureCache.set(replacement.slot, prior);
        else this._textureCache.delete(replacement.slot);
      }
    };
    const retire = (textures: Iterable<GPUTexture>): void => {
      for (const texture of new Set(textures)) {
        this._trackedCacheTextures.delete(texture);
        try { texture.destroy(); } catch { /* continue cohort retirement */ }
      }
    };
    return {
      commit: () => {
        if (state !== 'prepared') return;
        let published = 0;
        try {
          for (let index = 0; index < replacements.length; index += 1) {
            const replacement = replacements[index]!;
            this._textureCache.set(replacement.slot, replacement.texture);
            published += 1;
          }
          state = 'committed';
        } catch (error) {
          for (let index = 0; index < published; index += 1) {
            const replacement = replacements[index]!;
            const prior = previous[index];
            if (prior != null) this._textureCache.set(replacement.slot, prior);
            else this._textureCache.delete(replacement.slot);
          }
          retire(candidates);
          state = 'closed';
          throw error;
        }
      },
      rollback: () => {
        if (state === 'closed') return;
        if (state === 'committed') restorePrevious();
        retire(candidates);
        state = 'closed';
      },
      finalize: () => {
        if (state === 'closed') return;
        if (state !== 'committed') {
          retire(candidates);
          state = 'closed';
          return;
        }
        retire(live);
        state = 'closed';
      },
    };
  }

  dispose(): void {
    try {
      for (const tex of this._trackedCacheTextures) {
        try { tex.destroy(); } catch { /* continue retiring the cache cohort */ }
      }
    } finally {
      this._trackedCacheTextures.clear();
      this._textureCache = new WeakMap();
      this._visScratchSize = '';
    }
  }
}
