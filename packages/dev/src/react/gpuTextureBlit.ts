/**
 * gpuTextureBlit — shared GPU readback + canvas blit helper for the
 * dev-overlay React components (DDGIAtlasViewer, BVHVisualizer,
 * GISignalSplit). A3 (2026-05-19).
 *
 * Each overlay calls `useGpuTextureBlit(canvas, device, texture)` inside
 * a React effect; the hook owns a single staging GPUBuffer + RAF-throttled
 * readback loop, tonemaps the half-float / float texels to LDR, and
 * paints the result onto a 2D canvas overlay.
 *
 * Cost: one `copyTextureToBuffer` + one `mapAsync(READ)` per readback
 * tick (~1–5 ms wall-clock for a 256×256 RGBA16F atlas + the GPU→CPU
 * fence). The hook throttles to ~10 Hz by default so the cost stays
 * out of the engine's render path; hosts that want continuous live
 * updates can pass `throttleMs: 0`.
 *
 * Tonemapping: Reinhard `L / (1 + L)` per channel, gamma 2.2 encoded.
 * Cheap, monotonic, suitable for diagnostic visualization (not WCAG-
 * correct; not for content production).
 */

import { float16BitsToFloat32 } from '@vitrum/shared-denoisers';

const DEFAULT_THROTTLE_MS = 100; // ~10 Hz
const DDGI_F32_MAX = 3.4028234663852886e38;
const DDGI_SAFE_MANTISSA = 16384;
const DDGI_INVALID_EXPONENT = 1_000;

export type GpuTextureBlitDecodeMode =
  | 'linear-rgb'
  | 'ddgi-irradiance'
  | 'ddgi-visibility';

export interface GpuTextureBlitOptions {
  throttleMs?: number;
  label?: string;
  /** Interpretation applied after the texture's storage format is decoded. */
  decodeMode?: GpuTextureBlitDecodeMode;
}

/** Round up to the WebGPU 256-byte row alignment required by
 *  `copyTextureToBuffer`. */
function alignBytesPerRow(rowBytes: number): number {
  return Math.ceil(rowBytes / 256) * 256;
}

/** Reinhard tonemap a linear scalar to [0, 1], gamma-encode to sRGB. */
function tonemapByte(linear: number): number {
  if (!Number.isFinite(linear) || linear < 0) return 0;
  const ldr = linear / (1 + linear);
  return Math.round(255 * Math.pow(ldr, 1 / 2.2));
}

function ddgiDecodeExponent(lane: number): number {
  if (!Number.isFinite(lane) || lane !== Math.round(lane)) {
    return DDGI_INVALID_EXPONENT;
  }
  if (lane === 0 || lane === 1) return 0;
  if (lane >= -149 && lane <= -1) return lane;
  if (lane >= 2 && lane <= 115) return lane - 1;
  return DDGI_INVALID_EXPONENT;
}

function ddgiDecodeScalar(mantissa: number, exponentLane: number): number | null {
  const exponent = ddgiDecodeExponent(exponentLane);
  if (
    exponent === DDGI_INVALID_EXPONENT ||
    !Number.isFinite(mantissa) ||
    (exponent !== 0 && Math.abs(mantissa) > DDGI_SAFE_MANTISSA) ||
    (exponent === 114 && Math.abs(mantissa) >= DDGI_SAFE_MANTISSA)
  ) {
    return null;
  }
  const decoded = Math.fround(mantissa * (2 ** exponent));
  return Number.isFinite(decoded) && Math.abs(decoded) <= DDGI_F32_MAX
    ? decoded
    : null;
}

function ddgiSaturatingSquare(value: number): number {
  if (!(value > 0)) return 0;
  if (value > DDGI_F32_MAX / value) return DDGI_F32_MAX;
  const squared = Math.fround(value * value);
  return Number.isFinite(squared) ? squared : DDGI_F32_MAX;
}

/**
 * Convert one decoded RGBA texel into the three linear channels shown by the
 * canvas. Exported for deterministic codec tests; it is not re-exported from
 * the package surface.
 */
export function decodeGpuTexturePixelForDisplay(
  rgba: Float32Array,
  mode: GpuTextureBlitDecodeMode,
  outRGB: Float32Array,
): void {
  if (mode === 'linear-rgb') {
    outRGB[0] = rgba[0] ?? 0;
    outRGB[1] = rgba[1] ?? 0;
    outRGB[2] = rgba[2] ?? 0;
    return;
  }

  if (mode === 'ddgi-irradiance') {
    const r = ddgiDecodeScalar(rgba[0] ?? 0, rgba[3] ?? 0);
    const g = ddgiDecodeScalar(rgba[1] ?? 0, rgba[3] ?? 0);
    const b = ddgiDecodeScalar(rgba[2] ?? 0, rgba[3] ?? 0);
    if (r == null || g == null || b == null) {
      outRGB.fill(0);
      return;
    }
    outRGB[0] = r;
    outRGB[1] = g;
    outRGB[2] = b;
    return;
  }

  const mean = ddgiDecodeScalar(rgba[0] ?? 0, rgba[2] ?? 0);
  const rawSecondMoment = ddgiDecodeScalar(rgba[1] ?? 0, rgba[3] ?? 0);
  if (mean == null || rawSecondMoment == null || mean < 0 || rawSecondMoment < 0) {
    outRGB.fill(0);
    return;
  }
  const meanSquared = ddgiSaturatingSquare(mean);
  const secondMoment = Math.max(rawSecondMoment, meanSquared);
  // Visibility diagnostic mapping: R=mean, G=ordered second moment,
  // B=variance. Each channel is independently tonemapped below.
  outRGB[0] = mean;
  outRGB[1] = secondMoment;
  outRGB[2] = Math.max(0, secondMoment - meanSquared);
}

/**
 * Read back a GPUTexture (rgba16float or rgba32float) and paint it onto
 * a 2D canvas. Returns a teardown function the caller invokes from its
 * React effect cleanup.
 *
 * When throttleMs > 0, readback runs on an interval; when 0, readback
 * runs on every RAF tick (expensive — caller responsible for not
 * spinning the whole engine to a halt).
 */
export function startGpuTextureBlit(
  canvas: HTMLCanvasElement,
  device: GPUDevice,
  texture: GPUTexture,
  options: GpuTextureBlitOptions = {},
): () => void {
  const throttle = options.throttleMs ?? DEFAULT_THROTTLE_MS;
  const label = options.label ?? 'dev-overlay';
  const decodeMode = options.decodeMode ?? 'linear-rgb';

  const ctxNullable = canvas.getContext('2d');
  if (ctxNullable == null) {
    return () => { /* no 2D context — caller-side fallback */ };
  }
  const ctx: CanvasRenderingContext2D = ctxNullable;

  const { width, height, format } = texture;
  // Match canvas dimensions to texture dimensions so we get 1:1 pixels.
  canvas.width = width;
  canvas.height = height;

  // Allocate one staging buffer + reuse across readbacks. The decoder uses
  // fixed RGBA/RGB scratch arrays so we avoid per-pixel object / DataView
  // allocations (~650k/s at 10 Hz × 256² texture before this).
  let bytesPerPixel: number;
  let decode: (dv: DataView, byteOffset: number, outRGBA: Float32Array) => void;
  switch (format) {
    case 'rgba16float':
      bytesPerPixel = 8;
      decode = (dv, b, outRGBA) => {
        outRGBA[0] = float16BitsToFloat32(dv.getUint16(b + 0, true));
        outRGBA[1] = float16BitsToFloat32(dv.getUint16(b + 2, true));
        outRGBA[2] = float16BitsToFloat32(dv.getUint16(b + 4, true));
        outRGBA[3] = float16BitsToFloat32(dv.getUint16(b + 6, true));
      };
      break;
    case 'rgba32float':
      bytesPerPixel = 16;
      decode = (dv, b, outRGBA) => {
        outRGBA[0] = dv.getFloat32(b + 0, true);
        outRGBA[1] = dv.getFloat32(b + 4, true);
        outRGBA[2] = dv.getFloat32(b + 8, true);
        outRGBA[3] = dv.getFloat32(b + 12, true);
      };
      break;
    default:
      // Unsupported format — paint a single-frame warning and bail.
      ctx.fillStyle = 'rgba(0,0,0,0.65)';
      ctx.fillRect(0, 0, width, height);
      ctx.fillStyle = '#ffb347';
      ctx.font = '11px monospace';
      ctx.fillText(`Unsupported texture format: ${format}`, 8, 16);
      return () => { /* noop */ };
  }

  const bytesPerRow = alignBytesPerRow(width * bytesPerPixel);
  const stagingBuffer = device.createBuffer({
    label: `dev-${label}-staging`,
    size: bytesPerRow * height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const imageData = ctx.createImageData(width, height);
  // Scratch RGB target — reused across every pixel of every readback.
  const rgbaScratch = new Float32Array(4);
  const rgbScratch = new Float32Array(3);

  let cancelled = false;
  let inFlight = false;
  // Per-message dedup so a sustained failure doesn't fill the console.
  // Stringify by error name + message; structured stack traces stay in
  // the first occurrence. Capped so a pathological producer that varies
  // its error message every call (e.g. embeds a timestamp) can't grow
  // the set unboundedly across the blit's lifetime.
  const SEEN_FAILURES_MAX = 32;
  const seenFailures = new Set<string>();

  async function tick(): Promise<void> {
    if (cancelled || inFlight) return;
    inFlight = true;
    try {
      const encoder = device.createCommandEncoder({ label: `dev-${label}-copy` });
      encoder.copyTextureToBuffer(
        { texture },
        { buffer: stagingBuffer, bytesPerRow, rowsPerImage: height },
        { width, height, depthOrArrayLayers: 1 },
      );
      device.queue.submit([encoder.finish()]);
      await stagingBuffer.mapAsync(GPUMapMode.READ);
      if (cancelled) {
        stagingBuffer.unmap();
        return;
      }
      const mapped = stagingBuffer.getMappedRange();
      // One DataView per readback (was: one per pixel).
      const dv = new DataView(mapped);
      const rowBytes = bytesPerRow;
      const out = imageData.data;
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const srcOff = y * rowBytes + x * bytesPerPixel;
          decode(dv, srcOff, rgbaScratch);
          decodeGpuTexturePixelForDisplay(rgbaScratch, decodeMode, rgbScratch);
          const dstOff = (y * width + x) * 4;
          out[dstOff + 0] = tonemapByte(rgbScratch[0]!);
          out[dstOff + 1] = tonemapByte(rgbScratch[1]!);
          out[dstOff + 2] = tonemapByte(rgbScratch[2]!);
          out[dstOff + 3] = 255;
        }
      }
      stagingBuffer.unmap();
      ctx.putImageData(imageData, 0, 0);
    } catch (err) {
      // Painting is best-effort; never let a readback failure tear the
      // overlay down. Log once per failure key (name + message) to keep
      // a sustained failure (texture state lost, device-lost mid-blit,
      // etc.) from filling the console at the readback rate.
      const e = err instanceof Error ? err : new Error(String(err));
      const key = `${e.name}:${e.message}`;
      if (!seenFailures.has(key) && seenFailures.size < SEEN_FAILURES_MAX) {
        seenFailures.add(key);
         
        console.warn(`[dev/${label}] readback failed:`, err);
      }
    } finally {
      inFlight = false;
    }
  }

  let interval: ReturnType<typeof setInterval> | null = null;
  let raf: number | null = null;

  if (throttle > 0) {
    interval = setInterval(() => {
      void tick();
    }, throttle);
  } else {
    const loop = (): void => {
      if (cancelled) return;
      raf = requestAnimationFrame(loop);
      void tick();
    };
    raf = requestAnimationFrame(loop);
  }

  // Kick off an immediate first readback so the canvas doesn't sit blank
  // for the full throttle interval.
  void tick();

  return () => {
    cancelled = true;
    if (interval != null) clearInterval(interval);
    if (raf != null) cancelAnimationFrame(raf);
    try {
      stagingBuffer.destroy();
    } catch {
      // already destroyed
    }
  };
}
