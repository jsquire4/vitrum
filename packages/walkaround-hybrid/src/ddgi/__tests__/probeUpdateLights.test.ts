/**
 * packDDGIProbeLights — exact lane assertions (item 26).
 *
 * Node layout (DDGI runtime light storage buffer; see probeUpdateRays.wgsl.ts):
 *
 *   Header (4 u32 = 16 bytes):
 *     udata[0]  count (u32)
 *     udata[1]  light-record word offset (=4)
 *     udata[2]  alias-table word offset (=4 + count*16)
 *     udata[3]  ABI magic
 *
 *   Per-light entry, LIGHT_STRIDE_FLOATS=16 floats = 64 bytes, starting at
 *   float offset (4 + i*16):
 *     +0  kind        u32   (low bits: 0=sun, 1=point, 2=spot;
 *                            high bit: castShadowDisabled)
 *     +1  distance    f32   (fixture only; 0 = no cutoff)
 *     +2  decay       f32   (fixture only; 0 = no falloff, 2 = inverse-square)
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
 * Records are followed by `count` Walker/Vose alias entries (4 words each:
 * q, alias index, represented pmf, pad). These are the EXACT lanes read by the WGSL consumer evalPointLight /
 * evalDirectLighting (probeUpdateRays.wgsl.ts). Any lane mismatch between
 * this packer and the consumer is flagged as a FINDING in the assertion.
 */

import { describe, expect, it } from 'vitest';
import {
  DDGI_LIGHT_CAST_SHADOW_DISABLED,
  DDGI_LIGHT_KIND_MASK,
  DDGI_LIGHT_KIND_SPOT,
  DDGI_PROBE_LIGHTS_ABI_MAGIC,
  ddgiProbeLightSolidAngle,
  ddgiProbeLightsBufferByteLength,
  packDDGIProbeLights,
} from '../probeUpdateLights.js';
import type { DDGILight } from '../types.js';
import { makeProbeUpdateRaysWGSL } from '../wgsl/probeUpdateRays.wgsl.js';
import { coreEmitterToDDGILight } from '../../coreEmittersToDDGILights.js';

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

function functionBody(source: string, name: string): string {
  const marker = `fn ${name}(`;
  const start = source.indexOf(marker);
  expect(start, `${name} should be present`).toBeGreaterThanOrEqual(0);
  const brace = source.indexOf('{', start);
  expect(brace, `${name} should have a body`).toBeGreaterThanOrEqual(0);

  let depth = 0;
  for (let i = brace; i < source.length; i += 1) {
    const ch = source[i];
    if (ch === '{') depth += 1;
    if (ch === '}') {
      depth -= 1;
      if (depth === 0) return source.slice(brace + 1, i);
    }
  }
  throw new Error(`Could not find end of ${name}`);
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

  it('packs castShadow:false into the high kind bit without changing the sun kind', () => {
    const buf = packDDGIProbeLights([
      {
        kind: 'sun',
        on: true,
        intensity: 1,
        direction: { x: 0, y: -1, z: 0 },
        castShadow: false,
      },
    ], 1);
    const { u32 } = decode(buf);
    const kindWord = u32[lightBase(0)]!;
    expect(kindWord & DDGI_LIGHT_KIND_MASK).toBe(0);
    expect((kindWord & DDGI_LIGHT_CAST_SHADOW_DISABLED) >>> 0).toBe(DDGI_LIGHT_CAST_SHADOW_DISABLED);
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
    const directionLength = Math.hypot(0.1, -0.9, 0.2);
    expect(f32[base + 8]).toBeCloseTo(0.1 / directionLength, 5);
    expect(f32[base + 9]).toBeCloseTo(-0.9 / directionLength, 5);
    expect(f32[base + 10]).toBeCloseTo(0.2 / directionLength, 5);
    // innerCone defaults to sun angular radius = 0; outerCone is unused for sun.
    expect(f32[base + 11]).toBe(0);
    expect(f32[base + 15]).toBe(0);
  });

  it('packs directional angular radius into the sun innerCone lane', () => {
    const buf = packDDGIProbeLights([
      {
        kind: 'sun',
        on: true,
        intensity: 1,
        direction: { x: 0, y: -1, z: 0 },
        angularRadius: 0.04,
      },
    ], 1);
    const { f32 } = decode(buf);
    const base = lightBase(0);
    expect(f32[base + 11]).toBeCloseTo(0.04, 6);
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

describe('core directional emitter → DDGI sun', () => {
  it('derives soft-sun angular radius from authored angularDiameter', () => {
    const light = coreEmitterToDDGILight({
      id: 'soft-sun',
      kind: 'directional',
      direction: [0, 1, 0],
      color: [1, 0.9, 0.8],
      intensity: 2,
      angularDiameter: 0.08,
    });
    expect(light?.kind).toBe('sun');
    expect(light?.angularRadius).toBeCloseTo(0.04, 6);
  });
});

describe('probeUpdateRays soft-sun shader plumbing', () => {
  it('samples the DDGI sun cone from the packed sun angular-radius lane', () => {
    const shader = makeProbeUpdateRaysWGSL(4);
    const evalDirectLighting = functionBody(shader, 'evalDirectLighting');
    expect(shader).toContain('fn ddgiSoftSunDirection');
    expect(shader).toContain('let radius = max(angularRadius, 0.0);');
    expect(evalDirectLighting).toMatch(
      /ddgiSoftSunDirection\(\s*-ddgiNormalizeOr\(light\.direction, vec3f\(0\.0, -1\.0, 0\.0\)\),\s*light\.innerCone,\s*hitPos,\s*\)/,
    );
    expect(evalDirectLighting).toContain('evalSunLight(');
  });
});

describe('probeUpdateRays runtime-light header validation', () => {
  it('requires the exact record/alias partition after proving count arithmetic safe', () => {
    const shader = makeProbeUpdateRaysWGSL(4);
    const runtimeCount = functionBody(shader, 'ddgiRuntimeLightCount');

    expect(runtimeCount).toContain('dataOffset != 4u');
    expect(runtimeCount).toContain(
      'let expectedAliasOffset = dataOffset + count * 16u;',
    );
    expect(runtimeCount).toContain('aliasOffset != expectedAliasOffset');
    expect(runtimeCount.indexOf('count > (words - dataOffset) / 16u'))
      .toBeLessThan(runtimeCount.indexOf('dataOffset + count * 16u'));
  });
});

describe('probeUpdateRays material transport ownership', () => {
  it('passes split surface-source ownership into dielectric transport and preserves unlit output', () => {
    const shader = makeProbeUpdateRaysWGSL(4);
    const surface = shader.indexOf(
      'let surfaceSource = ddgiEvaluateProbeSurfaceRadiance(',
    );
    const suffix = shader.indexOf('radiance = vec3f(', surface);
    expect(surface).toBeGreaterThanOrEqual(0);
    expect(suffix).toBeGreaterThan(surface);
    expect(shader.slice(suffix, suffix + 800)).toContain(
      'surfaceSource, accepted.sourceFeature, containingMedia,',
    );
    expect(shader).toContain('1.0 - mappedTransmission,');
    expect(shader).toContain(
      'currentSurfaceSource.opaqueLo * opaqueWeight +',
    );
    expect(shader).toContain(
      'currentSurfaceSource.persistentReflectionLo,',
    );
    expect(shader).toContain(
      'if ((mat.flags & MATERIAL_FLAG_UNLIT) != 0u)',
    );
    expect(shader).toContain(
      'out.emissionLo = probeMat.albedo * probeMat.layerTransmission;',
    );
    expect(shader).toContain('out.terminalLo = out.emissionLo;');
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

  it('packs castShadow:false into the high kind bit without changing point kind', () => {
    const buf = packDDGIProbeLights([
      {
        kind: 'fixture',
        on: true,
        intensity: 2,
        position: { x: 1, y: 2, z: 3 },
        castShadow: false,
      },
    ], 1);
    const { u32 } = decode(buf);
    const kindWord = u32[lightBase(0)]!;
    expect(kindWord & DDGI_LIGHT_KIND_MASK).toBe(1);
    expect((kindWord & DDGI_LIGHT_CAST_SHADOW_DISABLED) >>> 0).toBe(DDGI_LIGHT_CAST_SHADOW_DISABLED);
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

  it('packs point fixture distance/decay into lanes [+1,+2] with physical default decay', () => {
    const buf = packDDGIProbeLights([
      {
        kind: 'fixture',
        on: true,
        intensity: 1,
        position: { x: 0, y: 0, z: 0 },
        distance: 12,
        decay: 0,
      },
    ], 1);
    const { f32 } = decode(buf);
    const base = lightBase(0);
    expect(f32[base + 1]).toBeCloseTo(12);
    expect(f32[base + 2]).toBeCloseTo(0);

    const defaultBuf = packDDGIProbeLights([
      { kind: 'fixture', on: true, intensity: 1, position: { x: 0, y: 0, z: 0 } },
    ], 1);
    const { f32: defaultF32 } = decode(defaultBuf);
    expect(defaultF32[base + 1]).toBe(0);
    expect(defaultF32[base + 2]).toBe(2);
  });
});

// ── spot fixture ──────────────────────────────────────────────────────────────

describe('packDDGIProbeLights — spot fixture', () => {
  it('packs spotAxis into direction lanes [+8,+9,+10] which the WGSL reads as light.direction', () => {
    // WGSL evalPointLight: `let axisLen2 = dot(light.direction, light.direction);`
    // then `let cosToP = dot(-light.direction * inverseSqrt(axisLen2), lightDir);`
    // so light.direction must be the spot's forward beam/travel axis.
    const buf = packDDGIProbeLights([
      {
        kind: 'fixture', on: true, intensity: 10,
        position: { x: 1, y: 2, z: 3 },
        spotAxis: { x: 0.0, y: -1.0, z: 0.0 }, // pointing straight down
        spotCosInner: 0.866,  // ~30° half-angle
        spotCosOuter: 0.707,  // ~45° half-angle
        distance: 9,
        decay: 1.5,
      },
    ], 1);
    const { f32, u32 } = decode(buf);
    const base = lightBase(0);

    // Cone evaluation in WGSL is gated on LIGHT_SPOT, so the kind is not
    // interchangeable with LIGHT_POINT even though both share this record ABI.
    expect(u32[base]).toBe(DDGI_LIGHT_KIND_SPOT);

    // Distance/decay are real fixture fields, not padding.
    expect(f32[base + 1]).toBeCloseTo(9, 5);
    expect(f32[base + 2]).toBeCloseTo(1.5, 5);

    // Cone axis → WGSL light.direction.
    expect(f32[base + 8]).toBeCloseTo(0.0, 5);
    expect(f32[base + 9]).toBeCloseTo(-1.0, 5);
    expect(f32[base + 10]).toBeCloseTo(0.0, 5);

    // innerCone (cosInner) → WGSL light.innerCone, used in smoothstep(outerCone, innerCone, cosToP).
    expect(f32[base + 11]).toBeCloseTo(0.866, 4);

    // outerCone (cosOuter) → WGSL light.outerCone, lower bound of smoothstep.
    expect(f32[base + 15]).toBeCloseTo(0.707, 4);
  });

  it('preserves spot kind under the castShadow-disabled flag', () => {
    const buf = packDDGIProbeLights([{
      kind: 'fixture',
      on: true,
      intensity: 1,
      spotAxis: { x: 0, y: -1, z: 0 },
      spotCosInner: 0.9,
      spotCosOuter: 0.8,
      castShadow: false,
    }], 1);
    const kindWord = decode(buf).u32[lightBase(0)]!;
    expect(kindWord & DDGI_LIGHT_KIND_MASK).toBe(DDGI_LIGHT_KIND_SPOT);
    expect((kindWord & DDGI_LIGHT_CAST_SHADOW_DISABLED) >>> 0)
      .toBe(DDGI_LIGHT_CAST_SHADOW_DISABLED);
  });

  it('uses the packed kind, not direction length, for alias-table solid angle', () => {
    expect(ddgiProbeLightSolidAngle(1, 0.9)).toBeCloseTo(4 * Math.PI, 6);
    expect(ddgiProbeLightSolidAngle(DDGI_LIGHT_KIND_SPOT, 0.9))
      .toBeCloseTo(2 * Math.PI * 0.1, 6);
    expect(ddgiProbeLightSolidAngle(
      DDGI_LIGHT_KIND_SPOT | DDGI_LIGHT_CAST_SHADOW_DISABLED,
      0.9,
    )).toBeCloseTo(2 * Math.PI * 0.1, 6);

    const buf = packDDGIProbeLights([
      {
        kind: 'fixture',
        on: true,
        intensity: 1,
      },
      {
        kind: 'fixture',
        on: true,
        intensity: 1,
        spotAxis: { x: 0, y: -1, z: 0 },
        spotCosInner: 0.95,
        spotCosOuter: 0.9,
      },
    ], 1);
    const { f32, u32 } = decode(buf);
    expect(u32[lightBase(0)]! & DDGI_LIGHT_KIND_MASK).toBe(1);
    expect(u32[lightBase(1)]! & DDGI_LIGHT_KIND_MASK).toBe(DDGI_LIGHT_KIND_SPOT);
    const aliasBase = u32[2]!;
    const pointPmf = f32[aliasBase + 2]!;
    const spotPmf = f32[aliasBase + 6]!;
    expect(pointPmf).toBeGreaterThan(spotPmf);
    expect(pointPmf + spotPmf).toBeCloseTo(1, 6);
  });

  it('WGSL evaluates packed spotAxis as a forward beam axis and guards hard-edge cones', () => {
    const shader = makeProbeUpdateRaysWGSL(4);
    const body = functionBody(shader, 'evalPointLight');
    const coneHelper = functionBody(shader, 'ddgiSpotConeFalloff');
    const attenuationHelper = functionBody(shader, 'ddgiPointSpotAttenuation');

    expect(shader).toContain('const LIGHT_SPOT:  u32 = 2u;');
    expect(body).toContain('ddgiLightKind(light) == LIGHT_SPOT');
    expect(body).toContain('coneFalloff = ddgiSpotConeFalloff(');
    expect(body).toContain('let distanceAttenuation = ddgiPointSpotAttenuation(');
    expect(coneHelper).toContain('let cosTheta = dot(-axis, wi);');
    expect(coneHelper).toContain('cosInner == cosOuter');
    expect(attenuationHelper).toContain('if (decay > 0.0)');
    expect(attenuationHelper).toContain('let regularizedDist2 = max(dist * dist, dist2Floor);');
    expect(attenuationHelper).toContain('if (cutoffDistance > 0.0)');
    expect(attenuationHelper).toContain('attenuation = attenuation * x;');
    expect(attenuationHelper).not.toContain('attenuation = attenuation * x * x;');
    expect(body).toContain('let atten = light.intensity * distanceAttenuation;');
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

  it.each([true, false])('rejects an unknown kind explicitly even when on=%s', (on) => {
    const unknown = { kind: 'future-kind', on, intensity: 99 } as unknown as DDGILight;
    expect(() => packDDGIProbeLights([unknown], 1)).toThrowError(
      /kind must be 'sun', 'fixture', or 'teaLight'/,
    );
  });
});

describe('packDDGIProbeLights — Float32 publication envelope', () => {
  const f32Max = Math.fround(3.4028234663852886e38);
  const f32MinSubnormal = Math.fround(1.401298464324817e-45);

  it('normalizes Number-range and subnormal directions without overflow/collapse', () => {
    const huge = decode(packDDGIProbeLights([{
      kind: 'sun',
      on: true,
      intensity: 1,
      direction: {
        x: Number.MAX_VALUE,
        y: Number.MAX_VALUE,
        z: Number.MAX_VALUE,
      },
    }], 1)).f32;
    const hugeBase = lightBase(0);
    expect(Math.hypot(
      huge[hugeBase + 8]!,
      huge[hugeBase + 9]!,
      huge[hugeBase + 10]!,
    )).toBeCloseTo(1, 6);

    const tiny = decode(packDDGIProbeLights([{
      kind: 'fixture',
      on: true,
      intensity: 1,
      spotAxis: { x: Number.MIN_VALUE, y: 0, z: 0 },
      spotCosInner: 0.9,
      spotCosOuter: 0.8,
    }], 1)).f32;
    expect(Array.from(tiny.slice(hugeBase + 8, hugeBase + 11)))
      .toEqual([1, 0, 0]);
  });

  it('rejects scalar overflow and positive-underflow before writing a record', () => {
    expect(() => packDDGIProbeLights([{
      kind: 'sun',
      on: true,
      intensity: f32Max,
    }], 2)).toThrow(/sun intensity×multiplier.*finite/);
    expect(() => packDDGIProbeLights([{
      kind: 'fixture',
      on: true,
      intensity: Number.MIN_VALUE,
    }], 1)).toThrow(/intensity.*remain positive/);
    expect(() => packDDGIProbeLights([{
      kind: 'fixture',
      on: true,
      intensity: 1,
      distance: Number.MIN_VALUE,
    }], 1)).toThrow(/distance.*remain positive/);
  });

  it('rejects overflowing coordinates but permits harmless per-lane underflow', () => {
    expect(() => packDDGIProbeLights([{
      kind: 'fixture',
      on: true,
      intensity: 1,
      position: { x: Number.MAX_VALUE, y: 0, z: 0 },
    }], 1)).toThrow(/position\.x.*remain finite/);

    const packed = decode(packDDGIProbeLights([{
      kind: 'fixture',
      on: true,
      intensity: 1,
      position: { x: Number.MIN_VALUE, y: 2, z: 0 },
      color: { r: Number.MIN_VALUE, g: 1, b: 0 },
    }], 1)).f32;
    const base = lightBase(0);
    expect(packed[base + 4]).toBe(0);
    expect(packed[base + 5]).toBe(2);
    expect(packed[base + 12]).toBe(0);
    expect(packed[base + 13]).toBe(1);
  });

  it('rejects complete color and emitted-signal collapse/overflow', () => {
    expect(() => packDDGIProbeLights([{
      kind: 'fixture',
      on: true,
      intensity: 1,
      color: {
        r: Number.MIN_VALUE,
        g: Number.MIN_VALUE,
        b: Number.MIN_VALUE,
      },
    }], 1)).toThrow(/color.*collapse completely/);
    expect(() => packDDGIProbeLights([{
      kind: 'fixture',
      on: true,
      intensity: f32MinSubnormal,
      color: { r: f32MinSubnormal, g: 0, b: 0 },
    }], 1)).toThrow(/color×intensity.*underflow completely/);
    expect(() => packDDGIProbeLights([{
      kind: 'fixture',
      on: true,
      intensity: f32Max,
      color: { r: 2, g: 0, b: 0 },
    }], 1)).toThrow(/color×intensity.*finite/);
  });

  it('keeps host-only alias weights in f64 for maximum finite f32 radiance', () => {
    const packed = decode(packDDGIProbeLights([{
      kind: 'fixture',
      on: true,
      intensity: f32Max,
      color: { r: 1, g: 1, b: 1 },
    }], 1));
    const alias = packed.u32[2]!;
    expect(packed.f32[lightBase(0) + 7]).toBe(f32Max);
    expect(packed.f32[alias]).toBe(1);
    expect(packed.f32[alias + 2]).toBe(1);
  });

  it('does not reject a positive proposal whose luminance is below f32 range', () => {
    const packed = decode(packDDGIProbeLights([{
      kind: 'fixture',
      on: true,
      intensity: f32MinSubnormal,
      color: { r: 1, g: 0, b: 0 },
    }], 1));
    const alias = packed.u32[2]!;
    expect(packed.f32[lightBase(0) + 7]).toBe(f32MinSubnormal);
    expect(packed.f32[alias]).toBe(1);
    expect(packed.f32[alias + 2]).toBe(1);
  });
});

// ── runtime-sized storage ─────────────────────────────────────────────────────

describe('packDDGIProbeLights — runtime-sized storage', () => {
  it('packs every light beyond the former 16-light cap', () => {
    const lights = Array.from({ length: 17 }, (_, i) => ({
      kind: 'fixture' as const,
      on: true,
      intensity: i + 1,
      position: { x: i, y: 0, z: 0 },
    }));

    const buf = packDDGIProbeLights(lights, 1);
    const { u32, f32 } = decode(buf);

    expect(u32[0]).toBe(17);
    expect(f32[lightBase(15) + 7]).toBeCloseTo(16, 5);
    expect(f32[lightBase(16) + 7]).toBeCloseTo(17, 5);
    expect(buf.byteLength).toBe(ddgiProbeLightsBufferByteLength(17));
  });

  it('scales to hundreds of lights without a renderer-authored bound', () => {
    const lights = Array.from({ length: 257 }, (_, i) => ({
      kind: 'fixture' as const,
      on: true,
      intensity: i + 1,
      position: { x: i, y: 0, z: 0 },
    }));

    const buf = packDDGIProbeLights(lights, 1);
    const { u32, f32 } = decode(buf);
    expect(u32[0]).toBe(257);
    expect(f32[lightBase(256) + 7]).toBeCloseTo(257, 5);
    expect(buf.byteLength).toBe(ddgiProbeLightsBufferByteLength(257));
  });

  it('declares a runtime raw-word array, validates its ABI, and alias-samples one light', () => {
    const shader = makeProbeUpdateRaysWGSL(4);
    expect(shader).toMatch(/var<storage, read> lights:\s+array<u32>/);
    expect(shader).toContain('lights[3u] != DDGI_LIGHTS_ABI_MAGIC');
    expect(shader).toContain('let words = arrayLength(&lights);');
    expect(shader).toContain('let draw = ddgiLightAliasDraw(lightCount');
    expect(shader).toContain('let light = ddgiLoadLight(draw.index);');
    expect(shader).toContain('return result / draw.pmf;');
    expect(shader).not.toContain('MAX_LIGHTS');
    expect(shader).not.toContain('array<DDGILight, 16>');
  });

  it('publishes explicit record/alias offsets, ABI magic, and represented PMFs', () => {
    const buf = packDDGIProbeLights([
      { kind: 'fixture', on: true, intensity: 1, color: { r: 1, g: 1, b: 1 } },
      { kind: 'fixture', on: true, intensity: 9, color: { r: 1, g: 1, b: 1 } },
    ], 1);
    const { u32, f32 } = decode(buf);
    expect(Array.from(u32.slice(0, 4))).toEqual([
      2,
      4,
      4 + 2 * LIGHT_STRIDE_FLOATS,
      DDGI_PROBE_LIGHTS_ABI_MAGIC,
    ]);
    const alias = u32[2]!;
    const pmf0 = f32[alias + 2]!;
    const pmf1 = f32[alias + 6]!;
    expect(pmf0 + pmf1).toBeCloseTo(1, 6);
    expect(pmf1).toBeGreaterThan(pmf0);
    expect(pmf0).toBeGreaterThan(0);
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

  it('buffer byte length is exactly header plus active records', () => {
    for (const count of [0, 1, 8, 16, 17, 257]) {
      const lights = Array.from({ length: count }, () => ({
        kind: 'fixture' as const, on: true, intensity: 1,
      }));
      expect(packDDGIProbeLights(lights, 1).byteLength)
        .toBe(ddgiProbeLightsBufferByteLength(count));
    }
  });
});
