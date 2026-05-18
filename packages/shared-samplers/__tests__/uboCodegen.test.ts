/**
 * uboCodegen.test.ts — W2-C13 unit tests for the `defineUbo` codegen helper.
 *
 * Coverage focuses on the WGSL uniform-layout edge cases that historically
 * caused silent runtime corruption when hand-written:
 *   - vec3 alignment / trailing pad
 *   - mixed scalar / vector ordering
 *   - struct size rounding up to max member alignment
 *   - WebGPU 16-byte minimum uniform-binding size
 *   - pack/unpack round-trip identity
 *   - WGSL struct emission matching the byte layout
 */

import { describe, it, expect } from 'vitest';
import { defineUbo } from '../src/uboCodegen.js';

describe('defineUbo — basic scalar UBOs', () => {
  it('pads a lone u32 up to the 16-byte WebGPU minimum', () => {
    const ubo = defineUbo([
      { name: 'frameCount', type: 'u32' },
    ] as const);
    expect(ubo.sizeBytes).toBe(16);
    expect(ubo.fieldOffsets.frameCount).toBe(0);
  });

  it('packs a single u32 little-endian at offset 0', () => {
    const ubo = defineUbo([{ name: 'n', type: 'u32' }] as const);
    const buf = new ArrayBuffer(16);
    new Uint8Array(buf).fill(0xFF); // pre-dirty
    ubo.pack(new DataView(buf), 0, { n: 0x1234ABCD });
    const v = new DataView(buf);
    expect(v.getUint32(0, true)).toBe(0x1234ABCD);
    // Padding bytes 4..15 must be zeroed.
    for (let i = 4; i < 16; i++) expect(v.getUint8(i)).toBe(0);
  });

  it('round-trips a 4-scalar struct (u32 + 3×f32) byte-identically with the legacy hand-rolled pattern', () => {
    const ubo = defineUbo([
      { name: 'iteration',   type: 'u32' },
      { name: 'sigmaColor',  type: 'f32' },
      { name: 'sigmaNormal', type: 'f32' },
      { name: 'sigmaDepth',  type: 'f32' },
    ] as const);
    expect(ubo.sizeBytes).toBe(16);
    expect(ubo.fieldOffsets.iteration).toBe(0);
    expect(ubo.fieldOffsets.sigmaColor).toBe(4);
    expect(ubo.fieldOffsets.sigmaNormal).toBe(8);
    expect(ubo.fieldOffsets.sigmaDepth).toBe(12);

    const buf = new ArrayBuffer(16);
    ubo.pack(new DataView(buf), 0, {
      iteration: 3, sigmaColor: 4.0, sigmaNormal: 128.0, sigmaDepth: 1.0,
    });

    // Compare against the legacy hand-rolled byte pattern.
    const legacy = new ArrayBuffer(16);
    const lv = new DataView(legacy);
    lv.setUint32(0,  3,     true);
    lv.setFloat32(4,  4.0,   true);
    lv.setFloat32(8,  128.0, true);
    lv.setFloat32(12, 1.0,   true);
    expect(new Uint8Array(buf)).toEqual(new Uint8Array(legacy));

    const out = ubo.unpack(new DataView(buf), 0);
    expect(out.iteration).toBe(3);
    expect(out.sigmaColor).toBeCloseTo(4.0);
    expect(out.sigmaNormal).toBeCloseTo(128.0);
    expect(out.sigmaDepth).toBeCloseTo(1.0);
  });

  it('supports non-zero byte offsets without bleeding into neighbouring regions', () => {
    const ubo = defineUbo([{ name: 'x', type: 'f32' }] as const);
    const buf = new ArrayBuffer(64);
    new Uint8Array(buf).fill(0xCC);
    ubo.pack(new DataView(buf), 32, { x: 2.5 });
    const v = new DataView(buf);
    // Region [0, 32) untouched.
    for (let i = 0; i < 32; i++) expect(v.getUint8(i)).toBe(0xCC);
    // Region [32, 48) is the UBO — first 4 bytes hold f32 2.5, rest zero pad.
    expect(v.getFloat32(32, true)).toBeCloseTo(2.5);
    for (let i = 36; i < 48; i++) expect(v.getUint8(i)).toBe(0);
    // Region [48, 64) untouched.
    for (let i = 48; i < 64; i++) expect(v.getUint8(i)).toBe(0xCC);
  });
});

describe('defineUbo — std140 vec3 alignment edge cases', () => {
  it('vec3<f32> takes 16 bytes (12 data + 4 pad) in a struct that ends after it', () => {
    const ubo = defineUbo([{ name: 'p', type: 'vec3f' }] as const);
    expect(ubo.sizeBytes).toBe(16);
    expect(ubo.fieldOffsets.p).toBe(0);
  });

  it('vec3<f32> followed by a u32 — the u32 must land in the vec3 trailing pad slot (offset 12)', () => {
    const ubo = defineUbo([
      { name: 'origin', type: 'vec3f' },
      { name: 'flag',   type: 'u32'   },
    ] as const);
    expect(ubo.fieldOffsets.origin).toBe(0);
    expect(ubo.fieldOffsets.flag).toBe(12);
    expect(ubo.sizeBytes).toBe(16);

    const buf = new ArrayBuffer(16);
    ubo.pack(new DataView(buf), 0, {
      origin: [1, 2, 3] as const,
      flag: 7,
    });
    const v = new DataView(buf);
    expect(v.getFloat32(0, true)).toBeCloseTo(1);
    expect(v.getFloat32(4, true)).toBeCloseTo(2);
    expect(v.getFloat32(8, true)).toBeCloseTo(3);
    expect(v.getUint32(12, true)).toBe(7);
  });

  it('vec3<f32> followed by another vec3<f32> — second vec3 must align to 16, leaving 4 pad bytes between', () => {
    const ubo = defineUbo([
      { name: 'a', type: 'vec3f' },
      { name: 'b', type: 'vec3f' },
    ] as const);
    expect(ubo.fieldOffsets.a).toBe(0);
    expect(ubo.fieldOffsets.b).toBe(16);
    expect(ubo.sizeBytes).toBe(32);

    const buf = new ArrayBuffer(32);
    new Uint8Array(buf).fill(0xAA);
    ubo.pack(new DataView(buf), 0, {
      a: [1, 2, 3] as const,
      b: [4, 5, 6] as const,
    });
    const v = new DataView(buf);
    // bytes 12..15 are pad after `a` — must be zero (pack() zero-fills).
    for (let i = 12; i < 16; i++) expect(v.getUint8(i)).toBe(0);
    // bytes 28..31 are pad after `b`.
    for (let i = 28; i < 32; i++) expect(v.getUint8(i)).toBe(0);
    expect(v.getFloat32(16, true)).toBeCloseTo(4);
    expect(v.getFloat32(20, true)).toBeCloseTo(5);
    expect(v.getFloat32(24, true)).toBeCloseTo(6);
  });

  it('f32 followed by vec3<f32> — the vec3 starts at offset 16 (next 16-aligned slot)', () => {
    const ubo = defineUbo([
      { name: 'scalar', type: 'f32'   },
      { name: 'vec',    type: 'vec3f' },
    ] as const);
    expect(ubo.fieldOffsets.scalar).toBe(0);
    expect(ubo.fieldOffsets.vec).toBe(16);
    expect(ubo.sizeBytes).toBe(32);
  });

  it('vec2<f32> aligns to 8 (NOT 16) and does not get padded up unless followed by a 16-aligned member', () => {
    const ubo = defineUbo([
      { name: 'uv',  type: 'vec2f' },
      { name: 'n',   type: 'u32'   },
    ] as const);
    expect(ubo.fieldOffsets.uv).toBe(0);
    expect(ubo.fieldOffsets.n).toBe(8);
    // max-align is 8, so total = ceil(12/8)*8 = 16, also satisfies min uniform binding.
    expect(ubo.sizeBytes).toBe(16);
  });
});

describe('defineUbo — vec4 and mat4 layouts', () => {
  it('vec4<f32> takes 16 bytes with no internal pad', () => {
    const ubo = defineUbo([{ name: 'rgba', type: 'vec4f' }] as const);
    expect(ubo.sizeBytes).toBe(16);
    const buf = new ArrayBuffer(16);
    ubo.pack(new DataView(buf), 0, { rgba: [0.1, 0.2, 0.3, 0.4] as const });
    const v = new DataView(buf);
    expect(v.getFloat32(0,  true)).toBeCloseTo(0.1);
    expect(v.getFloat32(4,  true)).toBeCloseTo(0.2);
    expect(v.getFloat32(8,  true)).toBeCloseTo(0.3);
    expect(v.getFloat32(12, true)).toBeCloseTo(0.4);
  });

  it('mat4x4<f32> takes 64 bytes in column-major order', () => {
    const ubo = defineUbo([{ name: 'M', type: 'mat4x4f' }] as const);
    expect(ubo.sizeBytes).toBe(64);
    const identity = [
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ] as const;
    const buf = new ArrayBuffer(64);
    ubo.pack(new DataView(buf), 0, { M: identity });
    const v = new DataView(buf);
    for (let i = 0; i < 16; i++) {
      expect(v.getFloat32(i * 4, true)).toBeCloseTo(identity[i]!);
    }
  });

  it('mat4x4 followed by a scalar rounds up to a 16-aligned boundary after the scalar', () => {
    const ubo = defineUbo([
      { name: 'M',     type: 'mat4x4f' },
      { name: 'scale', type: 'f32'     },
    ] as const);
    expect(ubo.fieldOffsets.M).toBe(0);
    expect(ubo.fieldOffsets.scale).toBe(64);
    // max align 16 → 68 rounds up to 80
    expect(ubo.sizeBytes).toBe(80);
  });
});

describe('defineUbo — error handling', () => {
  it('rejects empty field lists', () => {
    expect(() => defineUbo([] as const)).toThrow(/empty/);
  });

  it('rejects duplicate field names', () => {
    expect(() => defineUbo([
      { name: 'x', type: 'f32' },
      { name: 'x', type: 'u32' },
    ] as const)).toThrow(/duplicate field name "x"/);
  });

  it('throws if pack offset + sizeBytes exceeds the DataView', () => {
    const ubo = defineUbo([{ name: 'x', type: 'f32' }] as const);
    const buf = new ArrayBuffer(20);
    expect(() => ubo.pack(new DataView(buf), 8, { x: 1 })).toThrow(RangeError);
  });

  it('throws if unpack offset is negative', () => {
    const ubo = defineUbo([{ name: 'x', type: 'f32' }] as const);
    const buf = new ArrayBuffer(16);
    expect(() => ubo.unpack(new DataView(buf), -1)).toThrow(RangeError);
  });
});

describe('defineUbo — WGSL emission', () => {
  it('emits a struct declaration that lists every field with its WGSL type', () => {
    const ubo = defineUbo([
      { name: 'iteration',   type: 'u32' },
      { name: 'sigmaColor',  type: 'f32' },
      { name: 'sigmaNormal', type: 'f32' },
      { name: 'sigmaDepth',  type: 'f32' },
    ] as const);
    const wgsl = ubo.wgsl('AtrousVarianceAtrousUBO');
    expect(wgsl).toContain('struct AtrousVarianceAtrousUBO {');
    expect(wgsl).toContain('iteration: u32,');
    expect(wgsl).toContain('sigmaColor: f32,');
    expect(wgsl).toContain('sigmaNormal: f32,');
    expect(wgsl).toContain('sigmaDepth: f32,');
    expect(wgsl).toContain('};');
  });

  it('emits the correct WGSL type names for vec/mat fields', () => {
    const ubo = defineUbo([
      { name: 'p',   type: 'vec3f'   },
      { name: 'uv',  type: 'vec2f'   },
      { name: 'rgba', type: 'vec4f'  },
      { name: 'M',   type: 'mat4x4f' },
    ] as const);
    const wgsl = ubo.wgsl('Mixed');
    expect(wgsl).toContain('p: vec3<f32>,');
    expect(wgsl).toContain('uv: vec2<f32>,');
    expect(wgsl).toContain('rgba: vec4<f32>,');
    expect(wgsl).toContain('M: mat4x4<f32>,');
  });

  it('with explicitPadding=true, emits _padN: u32 entries that match the byte gaps', () => {
    // vec3 followed by another vec3 leaves a 4-byte gap between them.
    const ubo = defineUbo([
      { name: 'a', type: 'vec3f' },
      { name: 'b', type: 'vec3f' },
    ] as const);
    const wgsl = ubo.wgsl('PaddedVec3Pair', { explicitPadding: true });
    expect(wgsl).toContain('a: vec3<f32>,');
    expect(wgsl).toContain('_pad0: u32,');
    expect(wgsl).toContain('b: vec3<f32>,');
    expect(wgsl).toContain('_pad1: u32,');
  });

  it('respects a custom indent string', () => {
    const ubo = defineUbo([{ name: 'n', type: 'u32' }] as const);
    const wgsl = ubo.wgsl('Foo', { indent: '\t' });
    expect(wgsl).toContain('\tn: u32,');
  });
});

describe('defineUbo — type inference smoke test', () => {
  // This test exercises the type-level inference at compile time. If
  // UboValue<F> doesn't pick up the right field names/types, this file
  // will fail to typecheck (tsc --noEmit at the workspace root).
  it('infers field names and types into pack/unpack signatures', () => {
    const ubo = defineUbo([
      { name: 'count',  type: 'u32'   },
      { name: 'point',  type: 'vec3f' },
      { name: 'matrix', type: 'mat4x4f' },
    ] as const);
    const buf = new ArrayBuffer(ubo.sizeBytes);
    ubo.pack(new DataView(buf), 0, {
      count: 1,
      point: [0, 0, 0] as const,
      matrix: [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
      ] as const,
    });
    const v = ubo.unpack(new DataView(buf), 0);
    expect(v.count).toBe(1);
    expect(v.point[0]).toBe(0);
    expect(v.matrix[0]).toBe(1);
  });
});
