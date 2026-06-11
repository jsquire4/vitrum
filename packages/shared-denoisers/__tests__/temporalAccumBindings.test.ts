/**
 * temporalAccumBindings.test.ts — D12.11 temporal-accum UBO helper contract.
 *
 * Verifies:
 *   1. TEMPORAL_ACCUM_UBO_SIZE_BYTES is 16 (WebGPU minimum uniform-binding
 *      size; the single active f32 padded to 16 bytes).
 *   2. packTemporalAccumUniforms writes `alpha` at byte offset 0 and
 *      zero-fills the three trailing pad slots (bytes 4..15).
 *   3. Byte-identity with the inline ACCUM_UBO in walkaround-hybrid's
 *      bindGroupBuilders.ts — both use `defineUbo([{name:'alpha',type:'f32'}])`
 *      which produces the same 16-byte layout.
 *   4. The helper is exported from the package index.
 */

import { describe, expect, it } from 'vitest';
import {
  TEMPORAL_ACCUM_UBO_SIZE_BYTES,
  packTemporalAccumUniforms,
} from '../src/temporalAccumBindings.js';
// Also verify the index re-export path.
import {
  TEMPORAL_ACCUM_UBO_SIZE_BYTES as SIZE_FROM_INDEX,
  packTemporalAccumUniforms as packFromIndex,
} from '../src/index.js';
// Used for the byte-identity test against the inline consumer pattern.
import { defineUbo } from '@vitrum/shared-samplers';

describe('TemporalAccumUBO — size constant', () => {
  it('TEMPORAL_ACCUM_UBO_SIZE_BYTES is 16 (WebGPU 16-byte uniform minimum)', () => {
    expect(TEMPORAL_ACCUM_UBO_SIZE_BYTES).toBe(16);
  });

  it('re-exported from package index', () => {
    expect(SIZE_FROM_INDEX).toBe(16);
  });
});

describe('packTemporalAccumUniforms — packing contract', () => {
  it('writes alpha at byte offset 0 and zero-fills bytes 4..15', () => {
    const buf = new ArrayBuffer(16);
    packTemporalAccumUniforms({ alpha: 0.05 }, buf);
    const f32v = new Float32Array(buf);
    const u32v = new Uint32Array(buf);
    // alpha at offset 0
    expect(f32v[0]).toBeCloseTo(0.05);
    // pad slots zero-filled
    expect(u32v[1]).toBe(0);
    expect(u32v[2]).toBe(0);
    expect(u32v[3]).toBe(0);
  });

  it('alpha=1.0 (history discard) packs correctly', () => {
    const buf = new ArrayBuffer(16);
    packTemporalAccumUniforms({ alpha: 1.0 }, buf);
    expect(new Float32Array(buf)[0]).toBe(1.0);
  });

  it('alpha=0.0 (pure history) packs correctly', () => {
    const buf = new ArrayBuffer(16);
    packTemporalAccumUniforms({ alpha: 0.0 }, buf);
    // All bytes zero: alpha=0 and pads=0.
    expect(new Uint32Array(buf).every(v => v === 0)).toBe(true);
  });

  it('respects byte offset parameter', () => {
    // Pack into a larger buffer at offset 32; bytes before/after must be untouched.
    const buf = new ArrayBuffer(64);
    const u8 = new Uint8Array(buf).fill(0xff); // pre-fill with non-zero sentinel
    packTemporalAccumUniforms({ alpha: 0.5 }, buf, 32);
    const f32v = new Float32Array(buf);
    // Before: should still be 0xff (as float the pattern is NaN/garbage, but
    // let's just check the uint8 bytes we care about).
    const before = new Uint8Array(buf, 0, 32);
    for (const b of before) expect(b).toBe(0xff);
    // At offset 32: alpha=0.5 (IEEE 754 = 0x3f000000) and pads = 0.
    expect(f32v[32 / 4]).toBeCloseTo(0.5);
    const u32v = new Uint32Array(buf);
    expect(u32v[36 / 4]).toBe(0);
    expect(u32v[40 / 4]).toBe(0);
    expect(u32v[44 / 4]).toBe(0);
    // After: should still be 0xff.
    const after = new Uint8Array(buf, 48);
    for (const b of after) expect(b).toBe(0xff);
  });

  it('index-re-exported pack function is byte-identical to direct import', () => {
    const buf1 = new ArrayBuffer(16);
    const buf2 = new ArrayBuffer(16);
    packTemporalAccumUniforms({ alpha: 0.123 }, buf1);
    packFromIndex({ alpha: 0.123 }, buf2);
    expect(new Uint8Array(buf1)).toEqual(new Uint8Array(buf2));
  });
});

describe('byte-identity with walkaround-hybrid ACCUM_UBO inline pattern', () => {
  it('produces identical 16 bytes as the inline defineUbo pattern in bindGroupBuilders', () => {
    // Replicate the inline ACCUM_UBO consumer pattern:
    //   const ACCUM_UBO = defineUbo([{ name: 'alpha', type: 'f32' }] as const);
    //   ACCUM_UBO.pack(new DataView(accumUboData), 0, { alpha });
    // Both must produce the same bytes for the wiring migration to be safe.
    const INLINE_UBO = defineUbo([{ name: 'alpha', type: 'f32' }] as const);
    const inlineBuf = new ArrayBuffer(INLINE_UBO.sizeBytes);
    INLINE_UBO.pack(new DataView(inlineBuf), 0, { alpha: 0.07 });

    const helperBuf = new ArrayBuffer(TEMPORAL_ACCUM_UBO_SIZE_BYTES);
    packTemporalAccumUniforms({ alpha: 0.07 }, helperBuf);

    expect(new Uint8Array(helperBuf)).toEqual(new Uint8Array(inlineBuf));
    // Sizes must match.
    expect(TEMPORAL_ACCUM_UBO_SIZE_BYTES).toBe(INLINE_UBO.sizeBytes);
  });
});
