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

// ─── W2-C13 vocabulary extension: vec2u / vec3u / vec4u ──────────────────────

describe('defineUbo — unsigned-integer vector layouts', () => {
  it('vec2<u32> aligns to 8 and stores two little-endian u32 values', () => {
    const ubo = defineUbo([
      { name: 'screenSize', type: 'vec2u' },
    ] as const);
    expect(ubo.sizeBytes).toBe(16); // 8 bytes rounded up to min uniform binding
    expect(ubo.fieldOffsets.screenSize).toBe(0);

    const buf = new ArrayBuffer(16);
    ubo.pack(new DataView(buf), 0, { screenSize: [1920, 1080] as const });
    const v = new DataView(buf);
    expect(v.getUint32(0, true)).toBe(1920);
    expect(v.getUint32(4, true)).toBe(1080);
  });

  it('vec2u sandwiched between u32 fields takes the next 8-aligned slot', () => {
    // u32 frameSeed at offset 0 → next slot is 8-aligned for vec2u.
    // After vec2u (8 bytes), cursor is 16 → next u32 at 16.
    const ubo = defineUbo([
      { name: 'frameSeed',    type: 'u32'   },
      { name: 'screenSize',   type: 'vec2u' },
      { name: 'emitterCount', type: 'u32'   },
    ] as const);
    expect(ubo.fieldOffsets.frameSeed).toBe(0);
    expect(ubo.fieldOffsets.screenSize).toBe(8);
    expect(ubo.fieldOffsets.emitterCount).toBe(16);
    // max align is 8 → struct rounded to 24 then bumped to MIN 16 (already > 16)
    expect(ubo.sizeBytes).toBe(24);
  });

  it('vec3<u32> consumes 12 written bytes + 4 trailing pad (std140 vec3 rule)', () => {
    const ubo = defineUbo([
      { name: 'probeCount',  type: 'vec3u' },
      { name: 'raysPerProbe', type: 'u32' },
    ] as const);
    // vec3u at 0..12; the u32 must land in the trailing pad slot at offset 12.
    expect(ubo.fieldOffsets.probeCount).toBe(0);
    expect(ubo.fieldOffsets.raysPerProbe).toBe(12);
    expect(ubo.sizeBytes).toBe(16);

    const buf = new ArrayBuffer(16);
    new Uint8Array(buf).fill(0xCC);
    ubo.pack(new DataView(buf), 0, {
      probeCount: [4, 5, 6] as const,
      raysPerProbe: 64,
    });
    const v = new DataView(buf);
    expect(v.getUint32(0, true)).toBe(4);
    expect(v.getUint32(4, true)).toBe(5);
    expect(v.getUint32(8, true)).toBe(6);
    expect(v.getUint32(12, true)).toBe(64);
  });

  it('vec3u followed by another vec3u leaves 4 pad bytes between them', () => {
    const ubo = defineUbo([
      { name: 'a', type: 'vec3u' },
      { name: 'b', type: 'vec3u' },
    ] as const);
    expect(ubo.fieldOffsets.a).toBe(0);
    expect(ubo.fieldOffsets.b).toBe(16);
    expect(ubo.sizeBytes).toBe(32);

    const buf = new ArrayBuffer(32);
    new Uint8Array(buf).fill(0xAA);
    ubo.pack(new DataView(buf), 0, {
      a: [10, 20, 30] as const,
      b: [40, 50, 60] as const,
    });
    const v = new DataView(buf);
    // bytes 12..15 are std140 pad after a — must be zero (pack() zero-fills).
    for (let i = 12; i < 16; i++) expect(v.getUint8(i)).toBe(0);
    // bytes 28..31 are std140 pad after b.
    for (let i = 28; i < 32; i++) expect(v.getUint8(i)).toBe(0);
    expect(v.getUint32(16, true)).toBe(40);
    expect(v.getUint32(20, true)).toBe(50);
    expect(v.getUint32(24, true)).toBe(60);
  });

  it('vec4<u32> aligns to 16 and stores four little-endian u32 values', () => {
    const ubo = defineUbo([
      { name: 'flags', type: 'vec4u' },
    ] as const);
    expect(ubo.sizeBytes).toBe(16);

    const buf = new ArrayBuffer(16);
    ubo.pack(new DataView(buf), 0, {
      flags: [0xDEADBEEF, 0xCAFEBABE, 0x12345678, 0xABCDEF01] as const,
    });
    const v = new DataView(buf);
    expect(v.getUint32(0,  true)).toBe(0xDEADBEEF);
    expect(v.getUint32(4,  true)).toBe(0xCAFEBABE);
    expect(v.getUint32(8,  true)).toBe(0x12345678);
    expect(v.getUint32(12, true)).toBe(0xABCDEF01);
  });

  it('vecNu types coerce non-uint number inputs via >>> 0 (matches u32 scalar behaviour)', () => {
    const ubo = defineUbo([{ name: 'v', type: 'vec3u' }] as const);
    const buf = new ArrayBuffer(16);
    ubo.pack(new DataView(buf), 0, { v: [-1, 1.7, 2 ** 33] as const });
    const v = new DataView(buf);
    // -1 → 0xFFFFFFFF, 1.7 → 1, 2^33 → 0 (>>> 0 mod 2^32)
    expect(v.getUint32(0, true)).toBe(0xFFFFFFFF);
    expect(v.getUint32(4, true)).toBe(1);
    expect(v.getUint32(8, true)).toBe(0);
  });

  it('emits the correct WGSL type names for vec2u / vec3u / vec4u', () => {
    const ubo = defineUbo([
      { name: 'a', type: 'vec2u' },
      { name: 'b', type: 'vec3u' },
      { name: 'c', type: 'vec4u' },
    ] as const);
    const wgsl = ubo.wgsl('UVec');
    expect(wgsl).toContain('a: vec2<u32>,');
    expect(wgsl).toContain('b: vec3<u32>,');
    expect(wgsl).toContain('c: vec4<u32>,');
  });

  it('round-trips vecNu values byte-identically via pack→unpack', () => {
    const ubo = defineUbo([
      { name: 'a', type: 'vec2u' },
      { name: 'b', type: 'vec3u' },
      { name: 'c', type: 'vec4u' },
    ] as const);
    const buf = new ArrayBuffer(ubo.sizeBytes);
    const original = {
      a: [11, 22] as const,
      b: [33, 44, 55] as const,
      c: [66, 77, 88, 99] as const,
    };
    ubo.pack(new DataView(buf), 0, original);
    const read = ubo.unpack(new DataView(buf), 0);
    expect(read.a).toEqual([11, 22]);
    expect(read.b).toEqual([33, 44, 55]);
    expect(read.c).toEqual([66, 77, 88, 99]);
  });
});

// ─── W2-C13 vocabulary extension: array<struct, N> ───────────────────────────

describe('defineUbo — array-of-struct fields', () => {
  it('lays out a 16-element array of a 16-byte struct contiguously (stride == element size)', () => {
    const Inner = defineUbo([
      { name: 'kind', type: 'u32'   },
      { name: 'pad0', type: 'u32'   },
      { name: 'pad1', type: 'u32'   },
      { name: 'pad2', type: 'u32'   },
    ] as const);
    expect(Inner.sizeBytes).toBe(16);

    const Outer = defineUbo([
      { name: 'count', type: 'u32' },
      { name: 'items', type: 'array', element: Inner, count: 16 },
    ] as const);
    // u32 at 0; array (align 16) starts at 16; 16 elements × 16 stride = 256.
    expect(Outer.fieldOffsets.count).toBe(0);
    expect(Outer.fieldOffsets.items).toBe(16);
    expect(Outer.sizeBytes).toBe(16 + 16 * 16);

    const buf = new ArrayBuffer(Outer.sizeBytes);
    Outer.pack(new DataView(buf), 0, {
      count: 3,
      items: [
        { kind: 1, pad0: 0, pad1: 0, pad2: 0 },
        { kind: 2, pad0: 0, pad1: 0, pad2: 0 },
        { kind: 3, pad0: 0, pad1: 0, pad2: 0 },
      ],
    });
    const v = new DataView(buf);
    expect(v.getUint32(0,  true)).toBe(3);
    expect(v.getUint32(16, true)).toBe(1); // items[0].kind
    expect(v.getUint32(32, true)).toBe(2); // items[1].kind
    expect(v.getUint32(48, true)).toBe(3); // items[2].kind
    // items[3..15] left zero by the trailing zero-fill.
    for (let i = 4 * 16; i < Outer.sizeBytes; i++) expect(v.getUint8(i)).toBe(0);
  });

  it('rounds up an element struct size that is NOT a multiple of 16 to the next 16-byte stride', () => {
    // An inner struct of 12 bytes worth of fields. defineUbo will round it
    // up to 16 (max align 4 → struct cursor 12, then min uniform binding
    // bumps it to 16). When this is used as an array element, the std140
    // array stride is 16. We assert this by inspecting the outer size.
    const Inner = defineUbo([
      { name: 'a', type: 'u32' },
      { name: 'b', type: 'u32' },
      { name: 'c', type: 'u32' },
    ] as const);
    // Inner is bumped to 16 (min uniform binding) — stride becomes 16.
    expect(Inner.sizeBytes).toBe(16);

    const Outer = defineUbo([
      { name: 'items', type: 'array', element: Inner, count: 4 },
    ] as const);
    expect(Outer.sizeBytes).toBe(64);
  });

  it('reads array elements back via unpack as an array of the inner value type', () => {
    // Inner = { u32 kind @0; vec3f position @16 (align 16) } → 28 → 32 (max align 16).
    const Inner = defineUbo([
      { name: 'kind',     type: 'u32'   },
      { name: 'position', type: 'vec3f' },
    ] as const);
    expect(Inner.sizeBytes).toBe(32);

    const Outer = defineUbo([
      { name: 'lights', type: 'array', element: Inner, count: 2 },
    ] as const);
    // stride = 32 (already a multiple of 16); 2 × 32 = 64.
    expect(Outer.sizeBytes).toBe(64);

    const buf = new ArrayBuffer(Outer.sizeBytes);
    Outer.pack(new DataView(buf), 0, {
      lights: [
        { kind: 7, position: [1, 2, 3] as const },
        { kind: 9, position: [4, 5, 6] as const },
      ],
    });
    const v = Outer.unpack(new DataView(buf), 0);
    expect(v.lights.length).toBe(2);
    expect(v.lights[0]!.kind).toBe(7);
    expect(v.lights[0]!.position).toEqual([1, 2, 3]);
    expect(v.lights[1]!.kind).toBe(9);
    expect(v.lights[1]!.position).toEqual([4, 5, 6]);
  });

  it('emits a WGSL array<…> field with the conventional element type name', () => {
    const Inner = defineUbo([{ name: 'x', type: 'u32' }] as const);
    const Outer = defineUbo([
      { name: 'lights', type: 'array', element: Inner, count: 16 },
    ] as const);
    const wgsl = Outer.wgsl('Lights');
    // Convention: trim trailing 's', PascalCase → "Light".
    expect(wgsl).toContain('lights: array<Light, 16>,');
  });

  it('packs fewer values than count without overwriting unused trailing slots', () => {
    const Inner = defineUbo([{ name: 'v', type: 'u32' }] as const);
    const Outer = defineUbo([
      { name: 'items', type: 'array', element: Inner, count: 4 },
    ] as const);
    const buf = new ArrayBuffer(Outer.sizeBytes);
    new Uint8Array(buf).fill(0xFF); // pre-dirty
    Outer.pack(new DataView(buf), 0, {
      items: [{ v: 100 }, { v: 200 }],
    });
    const v = new DataView(buf);
    expect(v.getUint32(0, true)).toBe(100);
    expect(v.getUint32(16, true)).toBe(200);
    // items[2..3] slots must be zero (pack zero-fills the whole UBO region).
    for (let i = 32; i < Outer.sizeBytes; i++) expect(v.getUint8(i)).toBe(0);
  });

  it('rejects an array spec with count <= 0 or non-integer count', () => {
    const Inner = defineUbo([{ name: 'x', type: 'u32' }] as const);
    expect(() => defineUbo([
      { name: 'items', type: 'array', element: Inner, count: 0 },
    ] as const)).toThrow(/count > 0/);
    expect(() => defineUbo([
      { name: 'items', type: 'array', element: Inner, count: 2.5 },
    ] as const)).toThrow(/count > 0/);
    expect(() => defineUbo([
      { name: 'items', type: 'array', element: Inner, count: -1 },
    ] as const)).toThrow(/count > 0/);
  });
});
