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

const DEFAULT_THROTTLE_MS = 100; // ~10 Hz

/** Round up to the WebGPU 256-byte row alignment required by
 *  `copyTextureToBuffer`. */
function alignBytesPerRow(rowBytes: number): number {
  return Math.ceil(rowBytes / 256) * 256;
}

/** Decode an IEEE 754 binary16 (half-float) bit pattern to a 32-bit float.
 *  Mirrors `shared-denoisers/halfFloat` but inlined to avoid a cross-
 *  package dep just for this dev-only helper. */
function halfBitsToFloat(bits: number): number {
  const sign = (bits >> 15) & 0x1;
  const exponent = (bits >> 10) & 0x1f;
  const mantissa = bits & 0x3ff;
  let value: number;
  if (exponent === 0) {
    value = (mantissa / 1024) * Math.pow(2, -14);
  } else if (exponent === 31) {
    value = mantissa === 0 ? Infinity : NaN;
  } else {
    value = (1 + mantissa / 1024) * Math.pow(2, exponent - 15);
  }
  return sign ? -value : value;
}

/** Reinhard tonemap a linear scalar to [0, 1], gamma-encode to sRGB. */
function tonemapByte(linear: number): number {
  if (!Number.isFinite(linear) || linear < 0) return 0;
  const ldr = linear / (1 + linear);
  return Math.round(255 * Math.pow(ldr, 1 / 2.2));
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
  options: { throttleMs?: number; label?: string } = {},
): () => void {
  const throttle = options.throttleMs ?? DEFAULT_THROTTLE_MS;
  const label = options.label ?? 'dev-overlay';

  const ctxNullable = canvas.getContext('2d');
  if (ctxNullable == null) {
    return () => { /* no 2D context — caller-side fallback */ };
  }
  const ctx: CanvasRenderingContext2D = ctxNullable;

  const { width, height, format } = texture;
  // Match canvas dimensions to texture dimensions so we get 1:1 pixels.
  canvas.width = width;
  canvas.height = height;

  // Allocate one staging buffer + reuse across readbacks. The decoder
  // writes into a 3-component scratch (rgb) so we avoid per-pixel object
  // / DataView allocations (~650k/s at 10 Hz × 256² texture before this).
  let bytesPerPixel: number;
  let decode: (dv: DataView, byteOffset: number, outRGB: Float32Array) => void;
  switch (format) {
    case 'rgba16float':
      bytesPerPixel = 8;
      decode = (dv, b, outRGB) => {
        outRGB[0] = halfBitsToFloat(dv.getUint16(b + 0, true));
        outRGB[1] = halfBitsToFloat(dv.getUint16(b + 2, true));
        outRGB[2] = halfBitsToFloat(dv.getUint16(b + 4, true));
      };
      break;
    case 'rgba32float':
      bytesPerPixel = 16;
      decode = (dv, b, outRGB) => {
        outRGB[0] = dv.getFloat32(b + 0, true);
        outRGB[1] = dv.getFloat32(b + 4, true);
        outRGB[2] = dv.getFloat32(b + 8, true);
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
  const rgbScratch = new Float32Array(3);

  let cancelled = false;
  let inFlight = false;

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
          decode(dv, srcOff, rgbScratch);
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
      // overlay down. Log once per failure type.
      // eslint-disable-next-line no-console
      console.warn(`[dev/${label}] readback failed:`, err);
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
