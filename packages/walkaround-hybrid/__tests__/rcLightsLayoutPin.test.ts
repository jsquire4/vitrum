/**
 * I5.4 — Layout pin: `packRCLights` byte layout vs. WGSL `RCLightBuffer` struct.
 *
 * `RCLightBuffer` is declared in
 * `packages/walkaround-rc/src/wgsl/probeRayCast.wgsl.ts`.  The host packer
 * (`packRCLights`, `HybridEngineRC.ts`) and the shader must agree on the
 * exact byte positions of every field or the GPU reads garbage.  These tests
 * pin the layout constants exported from `HybridEngineRC` against both the
 * WGSL struct comment-documentation AND the live packer output so any future
 * drift trips here before reaching the GPU (mirrors the `rcParamsCodegen` pattern).
 *
 * WGSL struct reference (probeRayCast.wgsl.ts):
 *   struct RCLight {
 *     kind:      u32,        // word 0  = byte 0
 *     _pad0/1/2: f32×3,      // words 1-3 = bytes 4-12
 *     position:  vec3f,      // words 4-6 = bytes 16-28
 *     intensity: f32,        // word 7  = byte 28
 *     direction: vec3f,      // words 8-10 = bytes 32-44
 *     innerCone: f32,        // word 11 = byte 44
 *     color:     vec3f,      // words 12-14 = bytes 48-60
 *     outerCone: f32,        // word 15 = byte 60
 *   };                       // total: 64 bytes
 *   struct RCLightBuffer {
 *     count: u32,            // byte 0
 *     _h0/h1/h2: u32×3,     // bytes 4-12
 *     items: array<RCLight, 16>,  // bytes 16..
 *   };                       // total: 16 + 16 × 64 = 1040 bytes
 */

import { describe, expect, it } from 'vitest';
import {
  packRCLights,
  RC_LIGHTS_BUFFER_BYTES,
  RC_LIGHTS_HEADER_BYTES,
  RC_LIGHT_ENTRY_BYTES,
  RCLightBufferHeaderOffset,
  RCLightEntryOffset,
} from '../src/HybridEngineRC.js';
import { PROBE_RAY_CAST_WGSL } from '@vitrum/walkaround-rc';
import type { DDGILight } from '../src/ddgi/types.js';  // used for typed fixtures below

// ── WGSL-derived layout (anti-tautology) ─────────────────────────────────────
// Parse the ACTUAL RCLight struct out of the live shader source and compute the
// std430-style byte offsets from the declaration order, so a WGSL field reorder
// trips this test even if the TS constants are internally consistent.
function parseRCLightOffsets(wgsl: string): Record<string, number> {
  const m = wgsl.match(/struct RCLight \{([\s\S]*?)\};/);
  if (m == null) throw new Error('RCLight struct not found in PROBE_RAY_CAST_WGSL');
  const offsets: Record<string, number> = {};
  let byte = 0;
  // Fields appear as `name: type,` — possibly several per line (the pad row).
  for (const field of m[1]!.matchAll(/([A-Za-z_][\w]*)\s*:\s*(vec3f|f32|u32)/g)) {
    const [, name, type] = field;
    const size = type === 'vec3f' ? 12 : 4;
    // RCLight is hand-padded so every field is 4-byte aligned and the vec3fs
    // land on 16-byte boundaries via explicit pads — no implicit padding to add.
    offsets[name!] = byte;
    byte += size;
  }
  offsets['__totalBytes'] = byte;
  return offsets;
}
const wgslOffsets = parseRCLightOffsets(PROBE_RAY_CAST_WGSL);

// ── Constant correctness ─────────────────────────────────────────────────────

describe('RCLightBuffer layout constants vs. WGSL struct', () => {
  it('RC_LIGHTS_BUFFER_BYTES = 1040 (header 16 + 16 × 64 entries)', () => {
    expect(RC_LIGHTS_BUFFER_BYTES).toBe(1040);
  });

  it('RC_LIGHTS_HEADER_BYTES = 16 (4 × u32)', () => {
    expect(RC_LIGHTS_HEADER_BYTES).toBe(16);
  });

  it('RC_LIGHT_ENTRY_BYTES = 64 (16 × 4)', () => {
    expect(RC_LIGHT_ENTRY_BYTES).toBe(64);
  });

  it('header count field is at byte 0', () => {
    expect(RCLightBufferHeaderOffset.count).toBe(0);
  });

  it('RCLight field byte offsets match WGSL struct layout', () => {
    // Each field offset is relative to the start of one RCLight (64 bytes = 16 words).
    expect(RCLightEntryOffset.kind).toBe(0);       // word 0
    expect(RCLightEntryOffset.position).toBe(16);  // word 4
    expect(RCLightEntryOffset.intensity).toBe(28); // word 7
    expect(RCLightEntryOffset.direction).toBe(32); // word 8
    expect(RCLightEntryOffset.innerCone).toBe(44); // word 11
    expect(RCLightEntryOffset.color).toBe(48);     // word 12
    expect(RCLightEntryOffset.outerCone).toBe(60); // word 15
  });

  it('TS offsets match offsets DERIVED from the live WGSL struct text (anti-tautology)', () => {
    // Parsed from PROBE_RAY_CAST_WGSL — a field reorder in the shader trips this
    // even when the TS constants above are internally consistent.
    expect(wgslOffsets['kind']).toBe(RCLightEntryOffset.kind);
    expect(wgslOffsets['position']).toBe(RCLightEntryOffset.position);
    expect(wgslOffsets['intensity']).toBe(RCLightEntryOffset.intensity);
    expect(wgslOffsets['direction']).toBe(RCLightEntryOffset.direction);
    expect(wgslOffsets['innerCone']).toBe(RCLightEntryOffset.innerCone);
    expect(wgslOffsets['color']).toBe(RCLightEntryOffset.color);
    expect(wgslOffsets['outerCone']).toBe(RCLightEntryOffset.outerCone);
    expect(wgslOffsets['__totalBytes']).toBe(RC_LIGHT_ENTRY_BYTES);
  });

  it('item 0 absolute byte offset = header (16)', () => {
    expect(RC_LIGHTS_HEADER_BYTES + 0 * RC_LIGHT_ENTRY_BYTES).toBe(16);
  });

  it('item 1 absolute byte offset = 80', () => {
    expect(RC_LIGHTS_HEADER_BYTES + 1 * RC_LIGHT_ENTRY_BYTES).toBe(80);
  });
});

// ── Live packer output ────────────────────────────────────────────────────────

const POINT_LIGHT: DDGILight = {
  kind: 'fixture',
  on: true,
  position: { x: 1, y: 2, z: 3 },
  color: { r: 0.5, g: 0.75, b: 1.0 },
  intensity: 4.0,
};

const SPOT_LIGHT: DDGILight = {
  kind: 'fixture',
  on: true,
  position: { x: 10, y: 20, z: 30 },
  color: { r: 1.0, g: 0.5, b: 0.25 },
  intensity: 8.0,
  spotAxis: { x: 0, y: -1, z: 0 },
  spotCosInner: 0.9,
  spotCosOuter: 0.7,
};

function readU32(buf: ArrayBuffer, byteOffset: number): number {
  return new DataView(buf).getUint32(byteOffset, /* little-endian */ true);
}

function readF32(buf: ArrayBuffer, byteOffset: number): number {
  return new DataView(buf).getFloat32(byteOffset, /* little-endian */ true);
}

describe('packRCLights output byte alignment', () => {
  it('output byteLength equals RC_LIGHTS_BUFFER_BYTES', () => {
    const buf = packRCLights([POINT_LIGHT]);
    expect(buf.byteLength).toBe(RC_LIGHTS_BUFFER_BYTES);
  });

  it('count = 1 at header offset 0', () => {
    const buf = packRCLights([POINT_LIGHT]);
    expect(readU32(buf, RCLightBufferHeaderOffset.count)).toBe(1);
  });

  it('count = 0 when no active fixtures', () => {
    const buf = packRCLights([{ kind: 'sun', on: true, intensity: 1 }]);
    expect(readU32(buf, RCLightBufferHeaderOffset.count)).toBe(0);
  });

  it('point light kind = 1 at item[0].kind', () => {
    const buf = packRCLights([POINT_LIGHT]);
    const item0 = RC_LIGHTS_HEADER_BYTES + 0 * RC_LIGHT_ENTRY_BYTES;
    expect(readU32(buf, item0 + RCLightEntryOffset.kind)).toBe(1);
  });

  it('spot light kind = 2 at item[0].kind', () => {
    const buf = packRCLights([SPOT_LIGHT]);
    const item0 = RC_LIGHTS_HEADER_BYTES + 0 * RC_LIGHT_ENTRY_BYTES;
    expect(readU32(buf, item0 + RCLightEntryOffset.kind)).toBe(2);
  });

  it('point light position packed at correct bytes', () => {
    const buf = packRCLights([POINT_LIGHT]);
    const item0 = RC_LIGHTS_HEADER_BYTES + 0 * RC_LIGHT_ENTRY_BYTES;
    expect(readF32(buf, item0 + RCLightEntryOffset.position + 0)).toBeCloseTo(1);
    expect(readF32(buf, item0 + RCLightEntryOffset.position + 4)).toBeCloseTo(2);
    expect(readF32(buf, item0 + RCLightEntryOffset.position + 8)).toBeCloseTo(3);
  });

  it('point light intensity at correct byte', () => {
    const buf = packRCLights([POINT_LIGHT]);
    const item0 = RC_LIGHTS_HEADER_BYTES + 0 * RC_LIGHT_ENTRY_BYTES;
    expect(readF32(buf, item0 + RCLightEntryOffset.intensity)).toBeCloseTo(4.0);
  });

  it('point light color at correct bytes', () => {
    const buf = packRCLights([POINT_LIGHT]);
    const item0 = RC_LIGHTS_HEADER_BYTES + 0 * RC_LIGHT_ENTRY_BYTES;
    expect(readF32(buf, item0 + RCLightEntryOffset.color + 0)).toBeCloseTo(0.5);
    expect(readF32(buf, item0 + RCLightEntryOffset.color + 4)).toBeCloseTo(0.75);
    expect(readF32(buf, item0 + RCLightEntryOffset.color + 8)).toBeCloseTo(1.0);
  });

  it('spot light innerCone and outerCone at correct bytes', () => {
    const buf = packRCLights([SPOT_LIGHT]);
    const item0 = RC_LIGHTS_HEADER_BYTES + 0 * RC_LIGHT_ENTRY_BYTES;
    expect(readF32(buf, item0 + RCLightEntryOffset.innerCone)).toBeCloseTo(0.9);
    expect(readF32(buf, item0 + RCLightEntryOffset.outerCone)).toBeCloseTo(0.7);
  });

  it('spot light direction at correct bytes', () => {
    const buf = packRCLights([SPOT_LIGHT]);
    const item0 = RC_LIGHTS_HEADER_BYTES + 0 * RC_LIGHT_ENTRY_BYTES;
    expect(readF32(buf, item0 + RCLightEntryOffset.direction + 0)).toBeCloseTo(0);
    expect(readF32(buf, item0 + RCLightEntryOffset.direction + 4)).toBeCloseTo(-1);
    expect(readF32(buf, item0 + RCLightEntryOffset.direction + 8)).toBeCloseTo(0);
  });

  it('second light packed at item[1]', () => {
    const buf = packRCLights([POINT_LIGHT, SPOT_LIGHT]);
    const item1 = RC_LIGHTS_HEADER_BYTES + 1 * RC_LIGHT_ENTRY_BYTES;
    // count = 2
    expect(readU32(buf, RCLightBufferHeaderOffset.count)).toBe(2);
    // item[1] is the spot light
    expect(readU32(buf, item1 + RCLightEntryOffset.kind)).toBe(2);
    expect(readF32(buf, item1 + RCLightEntryOffset.intensity)).toBeCloseTo(8.0);
  });

  it('off-lights are excluded from the buffer', () => {
    const offLight: DDGILight = { ...POINT_LIGHT, on: false };
    const buf = packRCLights([offLight]);
    expect(readU32(buf, RCLightBufferHeaderOffset.count)).toBe(0);
  });

  it('sun-kind lights are excluded', () => {
    const buf = packRCLights([{ kind: 'sun', on: true, intensity: 10 }]);
    expect(readU32(buf, RCLightBufferHeaderOffset.count)).toBe(0);
  });
});
