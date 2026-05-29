/**
 * WebGPU BDPT light-path scratch cache.
 *
 * Was an `rgba32float` read_write storage TEXTURE (width = maxLightBounces,
 * height = 3), but core WebGPU only permits `read_write` storage-texture access
 * for `r32float/uint/sint` (gpuweb #4651) — an `rgba32float` read_write storage
 * texture is rejected at bind-group creation on every conformant implementation
 * (Dawn + wgpu-native lavapipe/dzn). The cache is therefore a read_write storage
 * BUFFER, mirroring the BDPT eye-stack (`bdptEyeStack`, group 2 binding 6).
 *
 * Layout: `maxLightBounces` columns × 3 rows of vec4f, flattened row-minor as
 * `idx = col * 3 + row` (matches WGSL `bdptLightPathIndex`). Per light-vertex:
 * row 0 = pos (+ kind sentinel in .w), row 1 = normal + pdfFwd, row 2 =
 * throughput + pdfRev. Layout matches fork BDPT / {@link BdptLightPathBuffer}
 * in `@vitrum/pt-webgl`.
 */

export interface BdptLightPathBufferWebGPUOptions {
  readonly maxLightBounces?: number;
}

/** vec4f rows per light-vertex column (was the former texture height). */
const LIGHT_PATH_ROWS = 3;
/** Bytes per vec4f. */
const VEC4F_BYTES = 16;

/** WebGPU usage flags (tests use stub devices without global GPUBufferUsage). */
const STORAGE = 0x080;
const COPY_DST = 0x008;

export class BdptLightPathBufferWebGPU {
  readonly maxLightBounces: number;
  readonly buffer: GPUBuffer;

  #disposed = false;

  constructor(device: GPUDevice, options: BdptLightPathBufferWebGPUOptions = {}) {
    const max = options.maxLightBounces ?? 3;
    if (!Number.isFinite(max) || max < 1 || max > 3) {
      throw new RangeError(`[BdptLightPathBufferWebGPU] maxLightBounces must be 1..3 (got ${max})`);
    }
    this.maxLightBounces = max;
    this.buffer = device.createBuffer({
      label: 'vitrum.pt-webgpu.bdpt.lightPath',
      size: max * LIGHT_PATH_ROWS * VEC4F_BYTES,
      usage: STORAGE | COPY_DST,
    });
  }

  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    this.buffer.destroy();
  }
}

/** Minimal valid light-path placeholder buffer for bind-group layout when BDPT is off. */
export function createBdptLightPathPlaceholder(device: GPUDevice): GPUBuffer {
  return device.createBuffer({
    label: 'vitrum.pt-webgpu.bdpt.placeholder',
    size: LIGHT_PATH_ROWS * VEC4F_BYTES,
    usage: STORAGE | COPY_DST,
  });
}
