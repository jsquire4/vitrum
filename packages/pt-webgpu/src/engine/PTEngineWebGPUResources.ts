/**
 * Per-viewport GPU resources owned by `PTEngineWebGPU`.
 *
 * Five rgba16float storage textures (accum, normalDepth, albedo, variance,
 * motionVectors) and two storage buffers (accum, varianceMoments) sized to
 * `width * height * 16` bytes. The engine class delegates lifecycle (alloc,
 * resize, destroy, clear) here so it can stay an orchestrator.
 *
 * Behaviour is preserved verbatim from the pre-W4-A9 inline implementation:
 * - `ensure(width, height)` rebuilds the entire set when size changes or any
 *   resource is missing. It clears the accum/variance buffers on resize.
 * - `destroy()` releases every GPU handle and resets the cached size to 0.
 * - `clear()` resets the accumulator-only buffers without touching textures.
 *
 * A boolean `resizedSinceLastEnsure()` lets the orchestrator know to drop the
 * cached path-trace bind group when the views change identity.
 */
export interface AccumResources {
  readonly accumTexture: GPUTexture;
  readonly accumView: GPUTextureView;
  readonly normalDepthTexture: GPUTexture;
  readonly normalDepthView: GPUTextureView;
  readonly albedoTexture: GPUTexture;
  readonly albedoView: GPUTextureView;
  readonly varianceTexture: GPUTexture;
  readonly varianceView: GPUTextureView;
  readonly motionVectorsTexture: GPUTexture;
  readonly motionVectorsView: GPUTextureView;
  readonly accumBuffer: GPUBuffer;
  readonly varianceMomentsBuffer: GPUBuffer;
}

export class PTEngineWebGPUResources {
  readonly #device: GPUDevice;

  #accumTexture: GPUTexture | null = null;
  #accumView: GPUTextureView | null = null;
  #normalDepthTexture: GPUTexture | null = null;
  #normalDepthView: GPUTextureView | null = null;
  #albedoTexture: GPUTexture | null = null;
  #albedoView: GPUTextureView | null = null;
  #varianceTexture: GPUTexture | null = null;
  #varianceView: GPUTextureView | null = null;
  #motionVectorsTexture: GPUTexture | null = null;
  #motionVectorsView: GPUTextureView | null = null;
  #accumBuffer: GPUBuffer | null = null;
  #varianceMomentsBuffer: GPUBuffer | null = null;

  #accumBufferByteSize = 0;
  #accumWidth = 0;
  #accumHeight = 0;
  #didResizeLastEnsure = false;

  constructor(device: GPUDevice) {
    this.#device = device;
  }

  get width(): number {
    return this.#accumWidth;
  }

  get height(): number {
    return this.#accumHeight;
  }

  /** Texture exposed for FrameOutput.primaryRadiance (may be null pre-render). */
  get accumTexture(): GPUTexture | null {
    return this.#accumTexture;
  }

  get normalDepthTexture(): GPUTexture | null {
    return this.#normalDepthTexture;
  }

  get albedoTexture(): GPUTexture | null {
    return this.#albedoTexture;
  }

  get varianceTexture(): GPUTexture | null {
    return this.#varianceTexture;
  }

  get motionVectorsTexture(): GPUTexture | null {
    return this.#motionVectorsTexture;
  }

  /**
   * After ensure(), true if the resources were (re)built (size change or
   * first allocation). Used by the orchestrator to invalidate the cached
   * path-trace bind group.
   */
  resizedSinceLastEnsure(): boolean {
    return this.#didResizeLastEnsure;
  }

  /**
   * Snapshot views/buffers as non-null. Throws if `ensure` was not called or
   * resources have been destroyed.
   */
  snapshot(): AccumResources {
    if (
      this.#accumTexture == null ||
      this.#accumView == null ||
      this.#normalDepthTexture == null ||
      this.#normalDepthView == null ||
      this.#albedoTexture == null ||
      this.#albedoView == null ||
      this.#varianceTexture == null ||
      this.#varianceView == null ||
      this.#motionVectorsTexture == null ||
      this.#motionVectorsView == null ||
      this.#accumBuffer == null ||
      this.#varianceMomentsBuffer == null
    ) {
      throw new Error('PTEngineWebGPUResources.snapshot: resources not allocated');
    }
    return {
      accumTexture: this.#accumTexture,
      accumView: this.#accumView,
      normalDepthTexture: this.#normalDepthTexture,
      normalDepthView: this.#normalDepthView,
      albedoTexture: this.#albedoTexture,
      albedoView: this.#albedoView,
      varianceTexture: this.#varianceTexture,
      varianceView: this.#varianceView,
      motionVectorsTexture: this.#motionVectorsTexture,
      motionVectorsView: this.#motionVectorsView,
      accumBuffer: this.#accumBuffer,
      varianceMomentsBuffer: this.#varianceMomentsBuffer,
    };
  }

  /**
   * Allocate (or reuse) the accum textures and storage buffers for `width
   * x height`. If sizes already match, returns without touching the GPU.
   * On resize, all old resources are destroyed first and the accum / variance
   * storage buffers are zero-cleared.
   */
  ensure(width: number, height: number): void {
    const targetByteSize = width * height * 16;
    const textureReady =
      this.#accumTexture != null && this.#accumWidth === width && this.#accumHeight === height;
    const bufferReady = this.#accumBuffer != null && this.#accumBufferByteSize === targetByteSize;
    if (textureReady && bufferReady) {
      this.#didResizeLastEnsure = false;
      return;
    }
    this.destroy();
    this.#accumTexture = this.#device.createTexture({
      label: 'vitrum.pt-webgpu.accum',
      size: { width, height, depthOrArrayLayers: 1 },
      format: 'rgba16float',
      usage:
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_SRC,
    });
    this.#accumView = this.#accumTexture.createView();
    this.#normalDepthTexture = this.#device.createTexture({
      label: 'vitrum.pt-webgpu.normalDepth',
      size: { width, height, depthOrArrayLayers: 1 },
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.#normalDepthView = this.#normalDepthTexture.createView();
    this.#albedoTexture = this.#device.createTexture({
      label: 'vitrum.pt-webgpu.albedo',
      size: { width, height, depthOrArrayLayers: 1 },
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.#albedoView = this.#albedoTexture.createView();
    this.#varianceTexture = this.#device.createTexture({
      label: 'vitrum.pt-webgpu.variance',
      size: { width, height, depthOrArrayLayers: 1 },
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.#varianceView = this.#varianceTexture.createView();
    this.#motionVectorsTexture = this.#device.createTexture({
      label: 'vitrum.pt-webgpu.motionVectors',
      size: { width, height, depthOrArrayLayers: 1 },
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.#motionVectorsView = this.#motionVectorsTexture.createView();
    this.#accumBuffer = this.#device.createBuffer({
      label: 'vitrum.pt-webgpu.accum.buffer',
      size: Math.max(16, targetByteSize),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.#varianceMomentsBuffer = this.#device.createBuffer({
      label: 'vitrum.pt-webgpu.varianceMoments.buffer',
      size: Math.max(16, targetByteSize),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.#accumBufferByteSize = targetByteSize;
    this.#accumWidth = width;
    this.#accumHeight = height;
    this.#didResizeLastEnsure = true;
    this.clear();
  }

  /**
   * Zero-clear the accum and variance-moments storage buffers without
   * touching textures (used on reset and at the end of every (re)alloc).
   */
  clear(): void {
    if (this.#accumBuffer == null) return;
    const encoder = this.#device.createCommandEncoder({
      label: 'vitrum.pt-webgpu.clearAccum',
    });
    encoder.clearBuffer(this.#accumBuffer);
    if (this.#varianceMomentsBuffer != null) {
      encoder.clearBuffer(this.#varianceMomentsBuffer);
    }
    this.#device.queue.submit([encoder.finish()]);
  }

  /** Destroy every GPU handle and reset cached dimensions. Idempotent. */
  destroy(): void {
    this.#accumTexture?.destroy();
    this.#accumTexture = null;
    this.#accumView = null;
    this.#normalDepthTexture?.destroy();
    this.#normalDepthTexture = null;
    this.#normalDepthView = null;
    this.#albedoTexture?.destroy();
    this.#albedoTexture = null;
    this.#albedoView = null;
    this.#varianceTexture?.destroy();
    this.#varianceTexture = null;
    this.#varianceView = null;
    this.#motionVectorsTexture?.destroy();
    this.#motionVectorsTexture = null;
    this.#motionVectorsView = null;
    this.#accumBuffer?.destroy();
    this.#accumBuffer = null;
    this.#varianceMomentsBuffer?.destroy();
    this.#varianceMomentsBuffer = null;
    this.#accumBufferByteSize = 0;
    this.#accumWidth = 0;
    this.#accumHeight = 0;
    this.#didResizeLastEnsure = false;
  }
}
