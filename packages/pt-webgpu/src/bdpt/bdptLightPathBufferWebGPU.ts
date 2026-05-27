/**
 * WebGPU RGBA32F light-path cache (width = maxLightBounces, height = 3).
 * Layout matches fork BDPT / {@link BdptLightPathBuffer} in `@vitrum/pt-webgl`.
 */

export interface BdptLightPathBufferWebGPUOptions {
  readonly maxLightBounces?: number;
}

const LIGHT_PATH_HEIGHT = 3;

/** WebGPU usage flags (tests use stub devices without global GPUTextureUsage). */
const TEX_BINDING = 0x04;
const COPY_DST = 0x02;
const STORAGE_BINDING = 0x08;

export class BdptLightPathBufferWebGPU {
  readonly maxLightBounces: number;
  readonly texture: GPUTexture;
  readonly view: GPUTextureView;

  #disposed = false;

  constructor(device: GPUDevice, options: BdptLightPathBufferWebGPUOptions = {}) {
    const max = options.maxLightBounces ?? 3;
    if (!Number.isFinite(max) || max < 1 || max > 3) {
      throw new RangeError(`[BdptLightPathBufferWebGPU] maxLightBounces must be 1..3 (got ${max})`);
    }
    this.maxLightBounces = max;
    this.texture = device.createTexture({
      label: 'vitrum.pt-webgpu.bdpt.lightPath',
      size: { width: max, height: LIGHT_PATH_HEIGHT, depthOrArrayLayers: 1 },
      format: 'rgba32float',
      usage: TEX_BINDING | COPY_DST | STORAGE_BINDING,
    });
    this.view = this.texture.createView();
  }

  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    this.texture.destroy();
  }
}

/** 1×1 invalid light-path placeholder for bind-group layout when BDPT is off. */
export function createBdptLightPathPlaceholder(device: GPUDevice): GPUTexture {
  return device.createTexture({
    label: 'vitrum.pt-webgpu.bdpt.placeholder',
    size: { width: 1, height: 3, depthOrArrayLayers: 1 },
    format: 'rgba32float',
    usage: TEX_BINDING | COPY_DST,
  });
}
