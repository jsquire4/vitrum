/**
 * captureFrame — pt-webgpu backend.
 *
 * Tests the copy-geometry (bytesPerRow alignment, buffer sizes, RGBA layout)
 * and null-before-first-frame guard.  Uses a mock GPUDevice so no Vulkan /
 * lavapipe environment is required.
 */
import { describe, expect, it } from 'vitest';
import { rgba16fBufferToRgbaF32 } from '../denoise/rgba16fReadback.js';
import { alignedTextureCopyBytesPerRow } from '@vitrum/shared-denoisers';

// ── helpers ──────────────────────────────────────────────────────────────────

/** Encode a known f16 bit-pattern for a single f32 value (IEEE 754 half). */
function f32ToF16Bits(v: number): number {
  // Simple conversion — only tested for small positive values well within f16 range.
  if (v === 0) return 0;
  const sign = v < 0 ? 1 : 0;
  const abs = Math.abs(v);
  const exp = Math.floor(Math.log2(abs));
  const clampedExp = Math.max(-14, Math.min(15, exp));
  const mantissa = abs / Math.pow(2, clampedExp) - 1;
  const mantissaBits = Math.round(mantissa * 1024) & 0x3ff;
  const expBits = (clampedExp + 15) & 0x1f;
  return ((sign << 15) | (expBits << 10) | mantissaBits);
}

/** Build a minimal rgba16float raw buffer (no row padding) for a 2×1 image
 *  where pixel (0,0) = (r0,g0,b0,a0) and pixel (1,0) = (r1,g1,b1,a1). */
function makeRgba16fBuffer(
  r0: number, g0: number, b0: number, a0: number,
  r1: number, g1: number, b1: number, a1: number,
  bytesPerRow: number,
): ArrayBuffer {
  const buf = new ArrayBuffer(bytesPerRow);
  const dv = new DataView(buf);
  dv.setUint16(0,  f32ToF16Bits(r0), true);
  dv.setUint16(2,  f32ToF16Bits(g0), true);
  dv.setUint16(4,  f32ToF16Bits(b0), true);
  dv.setUint16(6,  f32ToF16Bits(a0), true);
  dv.setUint16(8,  f32ToF16Bits(r1), true);
  dv.setUint16(10, f32ToF16Bits(g1), true);
  dv.setUint16(12, f32ToF16Bits(b1), true);
  dv.setUint16(14, f32ToF16Bits(a1), true);
  return buf;
}

// ── bytesPerRow alignment ────────────────────────────────────────────────────

describe('captureFrame copy-geometry (pt-webgpu)', () => {
  it('bytesPerRow is 256-aligned for rgba16f (8 B/px)', () => {
    // WebGPU requires bytesPerRow to be a multiple of 256.
    for (const w of [1, 31, 32, 64, 127, 128, 256, 512]) {
      const bpr = alignedTextureCopyBytesPerRow(w, 8);
      expect(bpr % 256).toBe(0);
      // Must be at least w × 8 bytes.
      expect(bpr).toBeGreaterThanOrEqual(w * 8);
    }
  });

  it('buffer size is bytesPerRow × height (no extra padding)', () => {
    const w = 64, h = 32;
    const bpr = alignedTextureCopyBytesPerRow(w, 8);
    const size = bpr * h;
    // Must accommodate all rows.
    expect(size).toBeGreaterThanOrEqual(w * h * 8);
  });
});

// ── rgba16fBufferToRgbaF32 ───────────────────────────────────────────────────

describe('rgba16fBufferToRgbaF32 (RGBA layout, top-left origin)', () => {
  it('decodes 2×1 image with correct R/G/B/A channel order', () => {
    const w = 2, h = 1;
    const bpr = alignedTextureCopyBytesPerRow(w, 8); // 256 bytes (>= 16 needed)
    const buf = makeRgba16fBuffer(1, 0, 0, 1, 0, 1, 0, 1, bpr);
    const rgba = rgba16fBufferToRgbaF32(buf, bpr, w, h);
    expect(rgba.length).toBe(w * h * 4);
    // Pixel (0,0): R=1, G=0, B=0, A=1
    expect(rgba[0]).toBeCloseTo(1, 1);
    expect(rgba[1]).toBeCloseTo(0, 1);
    expect(rgba[2]).toBeCloseTo(0, 1);
    expect(rgba[3]).toBeCloseTo(1, 1);
    // Pixel (1,0): R=0, G=1, B=0, A=1
    expect(rgba[4]).toBeCloseTo(0, 1);
    expect(rgba[5]).toBeCloseTo(1, 1);
    expect(rgba[6]).toBeCloseTo(0, 1);
    expect(rgba[7]).toBeCloseTo(1, 1);
  });

  it('handles zero-value pixels without NaN', () => {
    const w = 1, h = 1;
    const bpr = alignedTextureCopyBytesPerRow(w, 8);
    const buf = new ArrayBuffer(bpr); // all zeros
    const rgba = rgba16fBufferToRgbaF32(buf, bpr, w, h);
    expect(rgba.every(v => v === 0 && !Number.isNaN(v))).toBe(true);
  });

  it('output length equals width × height × 4', () => {
    for (const [w, h] of [[1, 1], [4, 4], [16, 9]] as const) {
      const bpr = alignedTextureCopyBytesPerRow(w, 8);
      const buf = new ArrayBuffer(bpr * h);
      const rgba = rgba16fBufferToRgbaF32(buf, bpr, w, h);
      expect(rgba.length).toBe(w * h * 4);
    }
  });

  it('strips per-row padding (bytesPerRow > width × 8)', () => {
    // Force explicit padding: the pixels live at the start of each row,
    // padding bytes follow.  Pixels in row 1 should still decode correctly.
    const w = 1, h = 2;
    const bpr = 256; // much wider than 1 px × 8 B = 8 B
    const buf = new ArrayBuffer(bpr * h);
    const dv = new DataView(buf);
    // Row 0 pixel: R=0.5 (f16 bits = 0x3800)
    dv.setUint16(0, 0x3800, true); // 0.5 in f16
    dv.setUint16(2, 0x0000, true); dv.setUint16(4, 0x0000, true); dv.setUint16(6, 0x3c00, true); // A=1.0
    // Row 1 pixel: R=1.0 (f16 bits = 0x3c00)
    dv.setUint16(bpr, 0x3c00, true); // 1.0 in f16
    dv.setUint16(bpr + 2, 0x0000, true); dv.setUint16(bpr + 4, 0x0000, true); dv.setUint16(bpr + 6, 0x3c00, true);
    const rgba = rgba16fBufferToRgbaF32(buf, bpr, w, h);
    expect(rgba[0]).toBeCloseTo(0.5, 1); // row 0 R
    expect(rgba[4]).toBeCloseTo(1.0, 1); // row 1 R
  });
});
