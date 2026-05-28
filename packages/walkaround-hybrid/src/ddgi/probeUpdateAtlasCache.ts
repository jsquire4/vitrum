/**
 * Atlas + border-scratch GPUTexture cache for probe updates (W4c).
 */
import type { AtlasTextureSlot } from './probeGrid.js';
import type { ProbeUpdateGpuState } from './probeUpdateGpuState.js';

export class ProbeUpdateAtlasTextureCache {
  private _textureCache = new WeakMap<AtlasTextureSlot, GPUTexture>();
  private _trackedCacheTextures = new Set<GPUTexture>();
  private _irrScratchSize = '';
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

    this._textureCache.set(slot, gpuTex);
    this._trackedCacheTextures.add(gpuTex);
    return gpuTex;
  }

  getOrCreateScratchTexture(
    device: GPUDevice,
    gpu: ProbeUpdateGpuState,
    atlas: GPUTexture,
    which: 'irr' | 'vis',
  ): GPUTexture {
    const sizeTag = `${atlas.width}|${atlas.height}`;
    if (which === 'irr') {
      if (gpu.irrScratchTex && this._irrScratchSize === sizeTag) return gpu.irrScratchTex;
      gpu.irrScratchTex?.destroy();
      gpu.irrScratchTex = device.createTexture({
        size: [atlas.width, atlas.height, 1],
        format: 'rgba16float',
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_SRC |
          GPUTextureUsage.COPY_DST,
      });
      this._irrScratchSize = sizeTag;
      return gpu.irrScratchTex;
    }
    if (gpu.visScratchTex && this._visScratchSize === sizeTag) return gpu.visScratchTex;
    gpu.visScratchTex?.destroy();
    gpu.visScratchTex = device.createTexture({
      size: [atlas.width, atlas.height, 1],
      format: 'rgba16float',
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_SRC |
        GPUTextureUsage.COPY_DST,
    });
    this._visScratchSize = sizeTag;
    return gpu.visScratchTex;
  }

  getCachedAtlas(slot: AtlasTextureSlot): GPUTexture | undefined {
    return this._textureCache.get(slot);
  }

  dispose(): void {
    for (const tex of this._trackedCacheTextures) tex.destroy();
    this._trackedCacheTextures.clear();
    this._textureCache = new WeakMap();
    this._irrScratchSize = '';
    this._visScratchSize = '';
  }
}
