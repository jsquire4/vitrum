/**
 * Atlas + border-scratch GPUTexture cache for probe updates (W4c).
 */
import type { AtlasTextureSlot } from './probeGrid.js';
import type { ProbeUpdateGpuState } from './probeUpdateGpuState.js';

function assertAtlasDimensionsSupported(
  device: GPUDevice,
  width: number,
  height: number,
  label: string,
): void {
  if (
    !Number.isSafeInteger(width) || width < 1 ||
    !Number.isSafeInteger(height) || height < 1
  ) {
    throw new RangeError(`${label} dimensions must be positive safe integers.`);
  }
  const limit = device.limits?.maxTextureDimension2D;
  if (typeof limit === 'number' && (width > limit || height > limit)) {
    throw new RangeError(
      `${label} ${width}x${height} exceeds ` +
      `device.limits.maxTextureDimension2D=${limit}.`,
    );
  }
}

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
    assertAtlasDimensionsSupported(
      device,
      slot.width,
      slot.height,
      'DDGI atlas',
    );
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
    assertAtlasDimensionsSupported(
      device,
      atlas.width,
      atlas.height,
      'DDGI visibility scratch atlas',
    );
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
    this.prepareAtlasSlotRetirement(slots)();
  }

  /**
   * Complete all fallible snapshot/dedup allocation before a grid publication.
   * The returned retirement step only deletes known identities and performs
   * best-effort destruction, so it can safely follow an irreversible slot swap.
   */
  prepareAtlasSlotRetirement(
    slots: readonly AtlasTextureSlot[],
  ): () => void {
    const textures: GPUTexture[] = [];
    for (let slotIndex = 0; slotIndex < slots.length; slotIndex += 1) {
      const texture = this._textureCache.get(slots[slotIndex]!);
      if (texture == null) continue;
      let duplicate = false;
      for (let index = 0; index < textures.length; index += 1) {
        if (textures[index] === texture) {
          duplicate = true;
          break;
        }
      }
      if (!duplicate) textures.push(texture);
    }
    return () => {
      for (let slotIndex = 0; slotIndex < slots.length; slotIndex += 1) {
        this._textureCache.delete(slots[slotIndex]!);
      }
      for (let index = 0; index < textures.length; index += 1) {
        const texture = textures[index]!;
        this._trackedCacheTextures.delete(texture);
        try { texture.destroy(); } catch { /* continue retiring the old cohort */ }
      }
    };
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
