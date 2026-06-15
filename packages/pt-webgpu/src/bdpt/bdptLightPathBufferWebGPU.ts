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
 * Layout: `maxLightBounces` columns × 5 rows of vec4f, flattened row-minor as
 * `idx = col * 5 + row` (matches WGSL `bdptLightPathIndex`). Per light-vertex:
 * row 0 = pos (+ kind sentinel in .w), row 1 = normal + pdfFwd, row 2 =
 * throughput + pdfRev, row 3 = (A9) matId (.w) + wo-toward-prev (.xyz) for the REAL
 * light-vertex BSDF in the §10.3 connection (matId < 0 ⇒ emitter, Lambertian),
 * row 4 = hit-local material payload (triIndex plus front-face bit, baryVW, instanceIndex).
 */

export interface BdptLightPathBufferWebGPUOptions {
  readonly maxLightBounces?: number;
}

/** vec4f rows per light-vertex column. Row 3 carries the light-vertex matId +
 *  wo-toward-prev; row 4 carries tri/bary/instance+side payload for mapped materials. */
const LIGHT_PATH_ROWS = 5;
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
    // A9 — cap raised 3 → 8 (matches the eye-subpath depth; the connection sweep's
    // merged-pdf array BDPT_MAX_MERGED=19 accommodates c≤8 + e≤8 + 3 headroom).
    if (!Number.isFinite(max) || max < 1 || max > 8) {
      throw new RangeError(`[BdptLightPathBufferWebGPU] maxLightBounces must be 1..8 (got ${max})`);
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
