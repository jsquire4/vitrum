/**
 * packDDGIProbeLights — exact lane assertions (item 26).
 *
 * Node layout (DDGI light UBO; see probeUpdateRays.wgsl.ts DDGILight struct):
 *
 *   Header (4 u32 = 16 bytes):
 *     udata[0]  count (u32)
 *     udata[1..3] padding
 *
 *   Per-light entry, LIGHT_STRIDE_FLOATS=16 floats = 64 bytes, starting at
 *   float offset (4 + i*16):
 *     +0  kind        u32   (0=sun, 1=fixture/teaLight)
 *     +1  _pad0       f32
 *     +2  _pad1       f32
 *     +3  _pad2       f32
 *     +4  position.x  f32   (fixture only; 0 for sun)
 *     +5  position.y  f32
 *     +6  position.z  f32
 *     +7  intensity   f32
 *     +8  direction.x f32   (sun travel dir / spot cone axis)
 *     +9  direction.y f32
 *     +10 direction.z f32
 *     +11 innerCone   f32   (spotCosInner; 0 for sun / point fixture)
 *     +12 color.r     f32
 *     +13 color.g     f32
 *     +14 color.b     f32
 *     +15 outerCone   f32   (spotCosOuter; 0 for sun)
 *
 * These are the EXACT lanes read by the WGSL consumer evalPointLight /
 * evalDirectLighting (probeUpdateRays.wgsl.ts). Any lane mismatch between
 * this packer and the consumer is flagged as a FINDING in the assertion.
 */

import { describe, expect, it, vi } from 'vitest';
import { packDDGIProbeLights } from '../probeUpdateLights.js';

const HEADER_FLOATS = 4;
const LIGHT_STRIDE_FLOATS = 16;

/**
 * Decode the packed buffer into a structured view for readable assertions.
 * Returns the float32 view and uint32 view as parallel arrays.
 */
function decode(buf: ArrayBuffer) {
  return {
    f32: new Float32Array(buf),
    u32: new Uint32Array(buf),
  };
}

/** Float offset of the first float in light i's slot. */
function lightBase(i: number): number {
  return HEADER_FLOATS + i * LIGHT_STRIDE_FLOATS;
}

// ── sun light ─────────────────────────────────────────────────────────────────

describe('packDDGIProbeLights — sun light', () => {
  it('packs kind=0 into slot 0', () => {
    const buf = packDDGIProbeLights([
      { kind: 'sun', on: true, intensity: 1, direction: { x: 0, y: -1, z: 0 } },
    ], 1);
    const { u32 } = decode(buf);
    // count = 1 in header.
    expect(u32[0]).toBe(1);
    // kind = 0 (LIGHT_SUN).
    const base = lightBase(0);
    expect(u32[base]).toBe(0); // kind = LIGHT_SUN
  });

  it('packs sun direction (travel dir) into direction lanes [+8,+9,+10]', () => {
    const buf = packDDGIProbeLights([
      {
        kind: 'sun', on: true, intensity: 5,
        direction: { x: 0.1, y: -0.9, z: 0.2 },
        color: { r: 1, g: 0.95, b: 0.85 },
      },
    ], 1);
    const { f32 } = decode(buf);
    const base = lightBase(0);
    // intensity (multiplied by sunIntensityMul=1 here) → slot +7.
    expect(f32[base + 7]).toBeCloseTo(5, 5);
    // direction → slots +8,+9,+10 (WGSL reads light.direction.xyz, negates for toward-light).
    expect(f32[base + 8]).toBeCloseTo(0.1, 5);
    expect(f32[base + 9]).toBeCloseTo(-0.9, 5);
    expect(f32[base + 10]).toBeCloseTo(0.2, 5);
    // innerCone / outerCone must be 0 for sun (no spot logic).
    expect(f32[base + 11]).toBe(0);
    expect(f32[base + 15]).toBe(0);
  });

  it('applies sunIntensityMul to the sun intensity lane', () => {
    const buf = packDDGIProbeLights([
      { kind: 'sun', on: true, intensity: 4, direction: { x: 0, y: -1, z: 0 } },
    ], 2.5);
    const { f32 } = decode(buf);
    const base = lightBase(0);
    // intensity lane = 4 * 2.5 = 10.
    expect(f32[base + 7]).toBeCloseTo(10, 5);
  });

  it('packs sun color into color lanes [+12,+13,+14]', () => {
    const buf = packDDGIProbeLights([
      { kind: 'sun', on: true, intensity: 1, color: { r: 0.8, g: 0.7, b: 0.6 } },
    ], 1);
    const { f32 } = decode(buf);
    const base = lightBase(0);
    expect(f32[base + 12]).toBeCloseTo(0.8, 5);
    expect(f32[base + 13]).toBeCloseTo(0.7, 5);
    expect(f32[base + 14]).toBeCloseTo(0.6, 5);
  });

  it('uses legacy warm-white default color when sun color is absent', () => {
    const buf = packDDGIProbeLights([
      { kind: 'sun', on: true, intensity: 1 },
    ], 1);
    const { f32 } = decode(buf);
    const base = lightBase(0);
    expect(f32[base + 12]).toBeCloseTo(1, 5);
    expect(f32[base + 13]).toBeCloseTo(0.95, 5);
    expect(f32[base + 14]).toBeCloseTo(0.85, 5);
  });

  it('uses legacy straight-down direction (0,-1,0) when sun direction is absent', () => {
    const buf = packDDGIProbeLights([
      { kind: 'sun', on: true, intensity: 1 },
    ], 1);
    const { f32 } = decode(buf);
    const base = lightBase(0);
    expect(f32[base + 8]).toBeCloseTo(0, 5);
    expect(f32[base + 9]).toBeCloseTo(-1, 5);
    expect(f32[base + 10]).toBeCloseTo(0, 5);
  });

  it('position lanes [+4,+5,+6] are 0 for sun', () => {
    const buf = packDDGIProbeLights([
      { kind: 'sun', on: true, intensity: 1, direction: { x: 0.1, y: -1, z: 0 } },
    ], 1);
    const { f32 } = decode(buf);
    const base = lightBase(0);
    expect(f32[base + 4]).toBe(0);
    expect(f32[base + 5]).toBe(0);
    expect(f32[base + 6]).toBe(0);
  });
});

// ── point fixture ─────────────────────────────────────────────────────────────

describe('packDDGIProbeLights — point fixture', () => {
  it('packs kind=1 into slot 0', () => {
    const buf = packDDGIProbeLights([
      { kind: 'fixture', on: true, intensity: 2, position: { x: 1, y: 2, z: 3 } },
    ], 1);
    const { u32 } = decode(buf);
    const base = lightBase(0);
    expect(u32[base]).toBe(1); // kind = LIGHT_POINT
  });

  it('packs position into position lanes [+4,+5,+6]', () => {
    const buf = packDDGIProbeLights([
      { kind: 'fixture', on: true, intensity: 3, position: { x: 4, y: 5, z: 6 } },
    ], 1);
    const { f32 } = decode(buf);
    const base = lightBase(0);
    expect(f32[base + 4]).toBeCloseTo(4, 5);
    expect(f32[base + 5]).toBeCloseTo(5, 5);
    expect(f32[base + 6]).toBeCloseTo(6, 5);
    expect(f32[base + 7]).toBeCloseTo(3, 5); // intensity
  });

  it('packs color into color lanes [+12,+13,+14]; defaults to white when absent', () => {
    const buf1 = packDDGIProbeLights([
      { kind: 'fixture', on: true, intensity: 1, color: { r: 0.5, g: 0.6, b: 0.7 } },
    ], 1);
    const { f32: f1 } = decode(buf1);
    const base = lightBase(0);
    expect(f1[base + 12]).toBeCloseTo(0.5, 5);
    expect(f1[base + 13]).toBeCloseTo(0.6, 5);
    expect(f1[base + 14]).toBeCloseTo(0.7, 5);

    // Absent color → white.
    const buf2 = packDDGIProbeLights([
      { kind: 'fixture', on: true, intensity: 1 },
    ], 1);
    const { f32: f2 } = decode(buf2);
    expect(f2[base + 12]).toBeCloseTo(1, 5);
    expect(f2[base + 13]).toBeCloseTo(1, 5);
    expect(f2[base + 14]).toBeCloseTo(1, 5);
  });

  it('WGSL-contract: cone axis [+8,+9,+10] and innerCone [+11] are zero for a point fixture (no spot)', () => {
    // WGSL evalPointLight reads light.direction and checks axisLen² > 0.25 to
    // detect a spot.  A point fixture has spotAxis=undefined → all zeros → no
    // cone falloff.  This is the exact check the GPU shader applies.
    const buf = packDDGIProbeLights([
      { kind: 'fixture', on: true, intensity: 1, position: { x: 0, y: 0, z: 0 } },
    ], 1);
    const { f32 } = decode(buf);
    const base = lightBase(0);
    // direction (cone axis) must be zero so axisLen² = 0 < 0.25 → point light.
    expect(f32[base + 8]).toBe(0);
    expect(f32[base + 9]).toBe(0);
    expect(f32[base + 10]).toBe(0);
    // innerCone and outerCone are unused for points but should be 0 per the
    // packer's explicit zero fallback.
    expect(f32[base + 11]).toBe(0); // spotCosInner → innerCone
    expect(f32[base + 15]).toBe(0); // spotCosOuter → outerCone
  });
});

// ── spot fixture ──────────────────────────────────────────────────────────────

describe('packDDGIProbeLights — spot fixture', () => {
  it('packs spotAxis into direction lanes [+8,+9,+10] which the WGSL reads as light.direction', () => {
    // WGSL evalPointLight: `let axisLen2 = dot(light.direction, light.direction);`
    // then `let cosToP = dot(lightDir, light.direction * inverseSqrt(axisLen2));`
    // so light.direction must be the spot axis (toward-light unit vector).
    const buf = packDDGIProbeLights([
      {
        kind: 'fixture', on: true, intensity: 10,
        position: { x: 1, y: 2, z: 3 },
        spotAxis: { x: 0.0, y: -1.0, z: 0.0 }, // pointing straight down
        spotCosInner: 0.866,  // ~30° half-angle
        spotCosOuter: 0.707,  // ~45° half-angle
      },
    ], 1);
    const { f32, u32 } = decode(buf);
    const base = lightBase(0);

    // kind = 1 (LIGHT_POINT — spot uses the same kind, distinguished by direction length).
    expect(u32[base]).toBe(1);

    // Cone axis → WGSL light.direction.
    expect(f32[base + 8]).toBeCloseTo(0.0, 5);
    expect(f32[base + 9]).toBeCloseTo(-1.0, 5);
    expect(f32[base + 10]).toBeCloseTo(0.0, 5);

    // innerCone (cosInner) → WGSL light.innerCone, used in smoothstep(outerCone, innerCone, cosToP).
    expect(f32[base + 11]).toBeCloseTo(0.866, 4);

    // outerCone (cosOuter) → WGSL light.outerCone, lower bound of smoothstep.
    expect(f32[base + 15]).toBeCloseTo(0.707, 4);
  });

  it('teaLight packs identically to fixture (same branch in packer)', () => {
    const args: Parameters<typeof packDDGIProbeLights>[0] = [
      {
        kind: 'teaLight', on: true, intensity: 2,
        position: { x: 7, y: 8, z: 9 },
        color: { r: 1, g: 0.5, b: 0 },
      },
    ];
    const bufFixture = packDDGIProbeLights([{ ...args[0]!, kind: 'fixture' }], 1);
    const bufTeaLight = packDDGIProbeLights(args, 1);
    // Byte-identical treatment (same branch).
    expect(Array.from(new Uint8Array(bufFixture))).toEqual(Array.from(new Uint8Array(bufTeaLight)));
  });
});

// ── off / inactive lights ────────────────────────────────────────────────────

describe('packDDGIProbeLights — inactive lights', () => {
  it('lights with on=false are excluded from the count', () => {
    const buf = packDDGIProbeLights([
      { kind: 'sun', on: false, intensity: 100 },
      { kind: 'fixture', on: true, intensity: 5, position: { x: 0, y: 0, z: 0 } },
    ], 1);
    const { u32, f32 } = decode(buf);
    // Only the fixture survives.
    expect(u32[0]).toBe(1);
    // First (and only) active entry is the fixture: kind=1.
    expect(u32[lightBase(0)]).toBe(1);
    // Its intensity should be 5 (NOT 100 from the sun).
    expect(f32[lightBase(0) + 7]).toBeCloseTo(5, 5);
  });
});

// ── truncation cap ────────────────────────────────────────────────────────────

describe('packDDGIProbeLights — MAX_DDGI_PROBE_LIGHTS truncation', () => {
  it('count is clamped to 16 and a console.warn is emitted when > 16 active lights', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);

    // Build 17 active fixture lights.
    const lights = Array.from({ length: 17 }, (_, i) => ({
      kind: 'fixture' as const,
      on: true,
      intensity: i + 1,
      position: { x: i, y: 0, z: 0 },
    }));

    const buf = packDDGIProbeLights(lights, 1);
    const { u32, f32 } = decode(buf);

    // Header count must be clamped to 16.
    expect(u32[0]).toBe(16);

    // The 17th light (index 16, intensity=17) must NOT appear in the buffer.
    // Slot 15 (last allowed) should have intensity=16.
    expect(f32[lightBase(15) + 7]).toBeCloseTo(16, 5);

    // console.warn must have been called exactly once with a message mentioning
    // the truncation so hosts know the cap was hit.
    expect(warnSpy).toHaveBeenCalledTimes(1);
    expect(warnSpy.mock.calls[0]![0]).toMatch(/DDGI/);
    expect(warnSpy.mock.calls[0]![0]).toMatch(/16/);

    warnSpy.mockRestore();
  });

  it('exactly 16 active lights do NOT trigger the warn', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);

    const lights = Array.from({ length: 16 }, (_, i) => ({
      kind: 'fixture' as const,
      on: true,
      intensity: 1,
      position: { x: i, y: 0, z: 0 },
    }));

    packDDGIProbeLights(lights, 1);
    expect(warnSpy).not.toHaveBeenCalled();

    warnSpy.mockRestore();
  });
});

// ── multi-light ordering ──────────────────────────────────────────────────────

describe('packDDGIProbeLights — multi-light ordering', () => {
  it('preserves original order: sun first, fixture second', () => {
    const buf = packDDGIProbeLights([
      { kind: 'sun', on: true, intensity: 3, direction: { x: 0, y: -1, z: 0 } },
      { kind: 'fixture', on: true, intensity: 7, position: { x: 1, y: 2, z: 3 } },
    ], 1);
    const { u32, f32 } = decode(buf);

    expect(u32[0]).toBe(2); // count
    // Entry 0 → sun (kind=0), intensity=3.
    expect(u32[lightBase(0)]).toBe(0);
    expect(f32[lightBase(0) + 7]).toBeCloseTo(3, 5);
    // Entry 1 → fixture (kind=1), intensity=7.
    expect(u32[lightBase(1)]).toBe(1);
    expect(f32[lightBase(1) + 7]).toBeCloseTo(7, 5);
  });

  it('buffer has the correct total byte length regardless of active count', () => {
    // Buffer must always be DDGI_PROBE_LIGHTS_BUFFER_BYTES = (4 + 16*16)*4 = 1040 bytes.
    const expectedBytes = (4 + 16 * 16) * 4;
    for (const count of [0, 1, 8, 16]) {
      const lights = Array.from({ length: count }, () => ({
        kind: 'fixture' as const, on: true, intensity: 1,
      }));
      expect(packDDGIProbeLights(lights, 1).byteLength).toBe(expectedBytes);
    }
  });
});
