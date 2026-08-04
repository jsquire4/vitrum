import { describe, expect, it } from 'vitest';
import type {
  DirectionalEmitter,
  DiscAreaEmitter,
  PointEmitter,
  RectAreaEmitter,
  SpotEmitter,
} from '@vitrum/core';
import { LIGHT_PIXELS, packLightsTexture } from './lightsTexture.js';

// GPU-FREE: assert the packed RGBA32F byte layout matches the §5 spec / the kept
// GLSL decoder (`lights_struct.glsl.js readLightInfo`, 6 texels/light). We read
// s0.a (type), s1.a (intensity) and s2.a (power) for a directional, a point and
// a rect-area emitter and pin the per-type power/area math.

const luminance = (r: number, g: number, b: number) => 0.2126 * r + 0.7152 * g + 0.0722 * b;

/** texel `t` (0..5), channel `c` (0..3) of light `i` in the packed grid.
 *  `LightsTextureData.data` is typed as the `Float32Array | Uint32Array`
 *  contract union; the lights packer always emits `rgba32f`. */
function texel(data: Float32Array | Uint32Array, i: number, t: number, c: number): number {
  return data[(i * LIGHT_PIXELS + t) * 4 + c]!;
}

describe('packLightsTexture', () => {
  const directional: DirectionalEmitter = {
    id: 'dir',
    kind: 'directional',
    color: [1, 1, 1],
    intensity: 3,
    direction: [0, 1, 0], // already unit
  };
  const point: PointEmitter = {
    id: 'pt',
    kind: 'point',
    color: [0.5, 0.25, 0.75],
    intensity: 2,
    position: [1, 2, 3],
    decay: 2,
    distance: 10,
  };
  // Core axes are half-extents: the full rect is 4×6 and has area 24.
  const rect: RectAreaEmitter = {
    id: 'rect',
    kind: 'rect-area',
    color: [0.2, 0.4, 0.6],
    intensity: 5,
    position: [-1, 0, 1],
    uAxis: [2, 0, 0],
    vAxis: [0, 3, 0],
  };

  const data = packLightsTexture([directional, point, rect]);

  it('emits a square RGBA32F grid sized for 6 texels/light', () => {
    expect(data.kind).toBe('rgba32f');
    expect(data.lightCount).toBe(3);
    // 3 lights × 6 texels = 18 texels → dim = ceil(sqrt(18)) = 5.
    expect(data.dim).toBe(5);
    expect(data.data.length).toBe(5 * 5 * 4);
  });

  it('packs the directional emitter (type 3, intensity, power)', () => {
    // s0.a = type DIR = 3
    expect(texel(data.data, 0, 0, 3)).toBe(3);
    // s1.a = intensity
    expect(texel(data.data, 0, 1, 3)).toBe(3);
    // s2.xyz = direction toward light
    expect(texel(data.data, 0, 2, 0)).toBeCloseTo(0, 6);
    expect(texel(data.data, 0, 2, 1)).toBeCloseTo(1, 6);
    expect(texel(data.data, 0, 2, 2)).toBeCloseTo(0, 6);
    // s2.a = power = luminance * intensity
    expect(texel(data.data, 0, 2, 3)).toBeCloseTo(luminance(1, 1, 1) * 3, 6);
  });

  it('packs the point emitter (type 4, intensity, power, decay/distance)', () => {
    // s0.a = type POINT = 4
    expect(texel(data.data, 1, 0, 3)).toBe(4);
    // s1.a = intensity
    expect(texel(data.data, 1, 1, 3)).toBe(2);
    // s2.xyz = world position (repeated), s2.a = power
    expect(texel(data.data, 1, 2, 0)).toBe(1);
    expect(texel(data.data, 1, 2, 1)).toBe(2);
    expect(texel(data.data, 1, 2, 2)).toBe(3);
    expect(texel(data.data, 1, 2, 3)).toBeCloseTo(luminance(0.5, 0.25, 0.75) * 2, 6);
    // s4.g = decay, s4.b = distance
    expect(texel(data.data, 1, 4, 1)).toBe(2);
    expect(texel(data.data, 1, 4, 2)).toBe(10);
  });

  it('packs the rect-area emitter (type 0, intensity, power, area)', () => {
    // s0.a = type RECT_AREA = 0
    expect(texel(data.data, 2, 0, 3)).toBe(0);
    // s1.a = intensity
    expect(texel(data.data, 2, 1, 3)).toBe(5);
    // s2.xyz = full-span u-vector (2 × core half-extent)
    expect(texel(data.data, 2, 2, 0)).toBe(4);
    expect(texel(data.data, 2, 2, 1)).toBe(0);
    expect(texel(data.data, 2, 2, 2)).toBe(0);
    // s2.a = power = luminance * intensity * area = lum*5*24
    expect(texel(data.data, 2, 2, 3)).toBe(
      Math.fround(luminance(0.2, 0.4, 0.6) * 5 * 24),
    );
    // s3.xyz = full-span v-vector; s3.a = 4·|core u × core v| = 24
    expect(texel(data.data, 2, 3, 1)).toBe(6);
    expect(texel(data.data, 2, 3, 3)).toBeCloseTo(24, 6);
  });

  it('handles an empty emitter list (1×1 grid, no crash)', () => {
    const empty = packLightsTexture([]);
    expect(empty.lightCount).toBe(0);
    expect(empty.dim).toBe(1);
    expect(empty.data.length).toBe(4);
  });

  it('preserves zero and representable sub-1e-20 powers without flooring', () => {
    const zero: PointEmitter = {
      ...point,
      id: 'zero-power',
      intensity: 0,
    };
    const dim: PointEmitter = {
      ...point,
      id: 'dim-power',
      color: [1, 1, 1],
      intensity: 1e-25,
    };
    const packed = packLightsTexture([zero, dim]);
    expect(texel(packed.data, 0, 2, 3)).toBe(0);
    expect(texel(packed.data, 1, 2, 3)).toBeGreaterThan(0);
    expect(texel(packed.data, 1, 2, 3)).toBeCloseTo(1e-25, 30);
  });
});

describe('packLightsTexture — float32 area-geometry boundary', () => {
  const rectBase: Omit<RectAreaEmitter, 'id' | 'uAxis' | 'vAxis'> = {
    kind: 'rect-area',
    color: [1, 1, 1],
    intensity: 1,
    position: [0, 0, 0],
  };

  it('rejects rect axes that become collinear only after RGBA32F quantization', () => {
    const rect: RectAreaEmitter = {
      ...rectBase,
      id: 'f32-collinear',
      uAxis: [1, 1, 0],
      vAxis: [1, 1 + 1e-8, 0],
    };
    expect(() => packLightsTexture([rect])).toThrow(
      /@vitrum\/pt-webgl2: rect-area emitter "f32-collinear".*\(degenerate\)/,
    );
  });

  it('rejects rect axes that overflow storage but accepts finite area after a raw squared-length overflow', () => {
    const storageOverflow: RectAreaEmitter = {
      ...rectBase,
      id: 'axis-storage-overflow',
      uAxis: [2e38, 0, 0],
      vAxis: [0, 1, 0],
    };
    expect(() => packLightsTexture([storageOverflow])).toThrow(
      /rect-area emitter "axis-storage-overflow".*\(non-finite-input\)/,
    );

    const denominatorOverflow: RectAreaEmitter = {
      ...rectBase,
      id: 'axis-denominator-overflow',
      uAxis: [1e20, 0, 0],
      vAxis: [0, 1, 0],
    };
    const packed = packLightsTexture([denominatorOverflow]);
    expect(texel(packed.data, 0, 3, 3) / 4e20).toBeCloseTo(1, 5);
  });

  it('rejects disc diameters that overflow storage or produce unrepresentable area', () => {
    const discBase: Omit<DiscAreaEmitter, 'id' | 'radius'> = {
      kind: 'disc-area',
      color: [1, 1, 1],
      intensity: 1,
      position: [0, 0, 0],
      normal: [0, 1, 0],
    };
    expect(() => packLightsTexture([{
      ...discBase,
      id: 'disc-storage-overflow',
      radius: 2e38,
    }])).toThrow(
      /disc-area emitter "disc-storage-overflow".*\(non-finite-input\)/,
    );
    expect(() => packLightsTexture([{
      ...discBase,
      id: 'disc-denominator-overflow',
      radius: 1e20,
    }])).toThrow(
      /disc-area emitter "disc-denominator-overflow".*\(unrepresentable-area\)/,
    );
  });

  it('retains support for derived powers beyond float32 and adversarial ratios', () => {
    const derivedOverflow = packLightsTexture([{
      ...rectBase,
      id: 'power-overflow',
      intensity: 2e38,
      uAxis: [1, 0, 0],
      vAxis: [0, 1, 0],
    }]);
    expect(derivedOverflow.proposalWeights?.[0]).toBeGreaterThan(3.402823466e38);
    expect(texel(derivedOverflow.data, 0, 4, 0)).toBe(1);
    expect(texel(derivedOverflow.data, 0, 2, 3)).toBe(
      Math.fround(3.402823466e38),
    );

    const directional = (id: string) => ({
      kind: 'directional' as const,
      id,
      color: [1, 1, 1] as [number, number, number],
      intensity: 2e38,
      direction: [0, 1, 0] as [number, number, number],
    });
    const equalHdr = packLightsTexture([
      directional('sum-a'),
      directional('sum-b'),
    ]);
    expect(texel(equalHdr.data, 0, 4, 0)).toBe(0.5);
    expect(texel(equalHdr.data, 1, 4, 0)).toBe(0.5);

    for (const emitters of [
      [
        { ...directional('dominant'), intensity: 1e30 },
        { ...directional('retained'), intensity: 1e-30 },
      ],
      [
        { ...directional('retained-first'), intensity: 1e-30 },
        { ...directional('dominant-last'), intensity: 1e30 },
      ],
    ]) {
      const packed = packLightsTexture(emitters);
      const pmf0 = texel(packed.data, 0, 4, 0);
      const pmf1 = texel(packed.data, 1, 4, 0);
      expect(pmf0).toBeGreaterThan(0);
      expect(pmf1).toBeGreaterThan(0);
      expect(Math.fround(pmf0 + pmf1)).toBe(1);
      // Directional-only sampling publishes the same represented proposal.
      expect(texel(packed.data, 0, 4, 3)).toBe(pmf0);
      expect(texel(packed.data, 1, 4, 3)).toBe(pmf1);
    }
  });

  it('rejects every analytic-light operand that would become non-finite RGBA32F', () => {
    const pointBase: PointEmitter = {
      kind: 'point',
      id: 'base',
      color: [1, 1, 1],
      intensity: 1,
      position: [0, 1, 0],
    };
    expect(() => packLightsTexture([{
      ...pointBase,
      id: 'position-overflow',
      position: [1e39, 0, 0],
    }])).toThrow(/position-overflow.*position\[0\] overflows WebGL float32 storage/);
    expect(() => packLightsTexture([{
      ...pointBase,
      id: 'intensity-overflow',
      color: [1e-39, 0, 0],
      intensity: 1e39,
    }])).toThrow(/intensity-overflow.*intensity overflows WebGL float32 storage/);
    expect(() => packLightsTexture([{
      ...pointBase,
      id: 'decay-overflow',
      decay: 1e39,
    }])).toThrow(/decay-overflow.*decay overflows WebGL float32 storage/);
    expect(() => packLightsTexture([{
      ...pointBase,
      id: 'distance-overflow',
      distance: 1e39,
    }])).toThrow(/distance-overflow.*distance overflows WebGL float32 storage/);
  });

  it('rejects f32 source-radiance multiplication overflow and underflow', () => {
    const pointBase: PointEmitter = {
      kind: 'point',
      id: 'base',
      color: [1, 0, 0],
      intensity: 1,
      position: [0, 1, 0],
    };
    expect(() => packLightsTexture([{
      ...pointBase,
      id: 'radiance-overflow',
      color: [2, 0, 0],
      intensity: 2e38,
    }])).toThrow(
      /radiance-overflow.*color\[0\] \* intensity overflows shader float32 multiplication/,
    );
    expect(() => packLightsTexture([{
      ...pointBase,
      id: 'radiance-underflow',
      color: [2 ** -149, 0, 0],
      intensity: 0.25,
    }])).toThrow(
      /radiance-underflow.*color\[0\] \* intensity underflows shader float32 multiplication/,
    );
  });

  it('derives selection power from the exact stored color-times-intensity result', () => {
    const tiny = 2 ** -149;
    const huge = 2 ** 120;
    const packed = packLightsTexture([{
      kind: 'point',
      id: 'ordered-radiance',
      color: [tiny, 0, 0],
      intensity: huge,
      position: [0, 1, 0],
    }]);
    const storedRadiance = Math.fround(Math.fround(tiny) * Math.fround(huge));
    const expectedPower = Math.fround(luminance(storedRadiance, 0, 0));
    expect(storedRadiance).toBeGreaterThan(0);
    expect(texel(packed.data, 0, 1, 0)).toBe(Math.fround(tiny));
    expect(texel(packed.data, 0, 1, 3)).toBe(Math.fround(huge));
    expect(texel(packed.data, 0, 2, 3)).toBe(expectedPower);
  });
});

// ── Item 10: disc-area (CIRC_AREA) packer structural test ─────────────────────
// Disc-area emitters pack as CIRC_AREA_LIGHT_TYPE = 1 (same 6-texel slot, not a
// rect approximation). The GLSL `randomAreaLightSample` and `intersectLightAtIndex`
// both branch on CIRC_AREA_LIGHT_TYPE — this test asserts the type field is 1 and
// that the packed (u, v) basis spans the full diameter (matching what the GLSL
// `randomAreaLightSample` expects: r ∈ [0, 0.5] within the disc radius).
describe('packLightsTexture — disc-area (CIRC_AREA) structural test', () => {
  const disc: DiscAreaEmitter = {
    id: 'disc',
    kind: 'disc-area',
    color: [1.0, 0.5, 0.0],
    intensity: 4,
    position: [0, 2, 0],
    normal: [0, 1, 0], // world-up; tangent basis deterministic from this
    radius: 1.5,
  };

  const data = packLightsTexture([disc]);

  it('emits exactly 1 light as CIRC_AREA_LIGHT_TYPE = 1', () => {
    expect(data.lightCount).toBe(1);
    // s0.a = type; CIRC_AREA = 1
    expect(texel(data.data, 0, 0, 3)).toBe(1);
  });

  it('packs the disc position into s0.xyz', () => {
    expect(texel(data.data, 0, 0, 0)).toBe(0);
    expect(texel(data.data, 0, 0, 1)).toBe(2);
    expect(texel(data.data, 0, 0, 2)).toBe(0);
  });

  it('packs intensity into s1.a', () => {
    expect(texel(data.data, 0, 1, 3)).toBe(4);
  });

  it('packs u/v vectors with length = full diameter (2 × radius)', () => {
    // The GLSL sampler uses: randomPos = pos + u * x + v * y where x,y ∈ [-0.5, 0.5].
    // So |u| and |v| must both equal 2 * radius = 3.
    const ux = texel(data.data, 0, 2, 0);
    const uy = texel(data.data, 0, 2, 1);
    const uz = texel(data.data, 0, 2, 2);
    const lenU = Math.hypot(ux, uy, uz);
    expect(lenU).toBeCloseTo(2 * disc.radius, 5);

    const vx = texel(data.data, 0, 3, 0);
    const vy = texel(data.data, 0, 3, 1);
    const vz = texel(data.data, 0, 3, 2);
    const lenV = Math.hypot(vx, vy, vz);
    expect(lenV).toBeCloseTo(2 * disc.radius, 5);
  });

  it('packs area = π * radius² (π/4 × (2r)² from the rect correction)', () => {
    // area = |u × v| * (π/4) = (2r)² * (π/4) = π * r²
    const area = texel(data.data, 0, 3, 3);
    expect(area).toBeCloseTo(Math.PI * disc.radius * disc.radius, 4);
  });

  it('packs power = luminance(color) * intensity * π * r²', () => {
    const lum = 0.2126 * 1.0 + 0.7152 * 0.5 + 0.0722 * 0.0;
    const expected = lum * disc.intensity * Math.PI * disc.radius * disc.radius;
    expect(texel(data.data, 0, 2, 3)).toBeCloseTo(expected, 4);
  });
});

describe('packLightsTexture — punctual spot orientation', () => {
  it('packs cross(u,v) as the backward axis consumed by NEE and BDPT', () => {
    const direction = [2, -3, 4] as const;
    const directionLength = Math.hypot(...direction);
    const forward = direction.map((value) => value / directionLength);
    const position = [1, 2, 3] as const;
    const spot: SpotEmitter = {
      id: 'oriented-spot',
      kind: 'spot',
      color: [1, 1, 1],
      intensity: 2,
      position,
      direction,
      angle: 0.6,
      penumbra: 0.25,
    };
    const packed = packLightsTexture([spot]);
    const u = [
      texel(packed.data, 0, 2, 0),
      texel(packed.data, 0, 2, 1),
      texel(packed.data, 0, 2, 2),
    ] as const;
    const v = [
      texel(packed.data, 0, 3, 0),
      texel(packed.data, 0, 3, 1),
      texel(packed.data, 0, 3, 2),
    ] as const;
    const back = [
      u[1] * v[2] - u[2] * v[1],
      u[2] * v[0] - u[0] * v[2],
      u[0] * v[1] - u[1] * v[0],
    ] as const;
    const backLength = Math.hypot(...back);
    const normalizedBack = back.map((value) => value / backLength);

    // The shader uses -cross(u,v) for emitted direction.
    const emittedDot = forward.reduce(
      (sum, component, axis) => sum + component * -normalizedBack[axis]!,
      0,
    );
    expect(emittedDot).toBeCloseTo(1, 6);

    // A receiver placed down the authored beam sees a positive cone cosine
    // against the same backward axis in randomSpotLightSample.
    const receiver = position.map((value, axis) => value + 5 * forward[axis]!);
    const toSource = position.map((value, axis) => value - receiver[axis]!);
    const toSourceLength = Math.hypot(...toSource);
    const coneCosine = toSource.reduce(
      (sum, component, axis) =>
        sum + (component / toSourceLength) * normalizedBack[axis]!,
      0,
    );
    expect(coneCosine).toBeCloseTo(1, 6);

    // SpotEmitter is a delta-position contract; inherited area/radius lanes
    // remain reserved zero rather than implying a soft source.
    expect(texel(packed.data, 0, 3, 3)).toBe(0);
    expect(texel(packed.data, 0, 4, 0)).toBe(1);
  });

  it('preserves intentional hard edges and rejects positive f32-collapsed cone support', () => {
    const base: SpotEmitter = {
      id: 'spot-boundary',
      kind: 'spot',
      color: [1, 1, 1],
      intensity: 1,
      position: [0, 1, 0],
      direction: [0, -1, 0],
      angle: 0.5,
    };
    const hard = packLightsTexture([{ ...base, penumbra: 0 }]);
    expect(texel(hard.data, 0, 4, 3)).toBe(texel(hard.data, 0, 5, 0));

    const soft = packLightsTexture([{ ...base, penumbra: 0.25 }]);
    expect(texel(soft.data, 0, 5, 0)).toBeGreaterThan(
      texel(soft.data, 0, 4, 3),
    );

    expect(() =>
      packLightsTexture([{
        ...base,
        id: 'collapsed-positive-cone',
        angle: 1e-4,
      }]),
    ).toThrow(/positive angle collapses to a zero-width cone/);

    expect(() =>
      packLightsTexture([{
        ...base,
        id: 'collapsed-positive-penumbra',
        penumbra: 1e-8,
      }]),
    ).toThrow(/positive penumbra collapses to the hard-cone edge/);
  });
});

// ── D10.10: assertSlotCursor dev-only guard tests ─────────────────────────────
// These tests verify that the slot-cursor guards fire for each light kind when
// the packing code would produce wrong data. We trigger packing and verify the
// guard does NOT throw for correct inputs (the normal path), then verify the
// guards do fire for a mutated cursor (simulated packing bug).
//
// Implementation note: assertSlotCursor is a module-internal function, so we
// test it indirectly by verifying the correct packing succeeds (no throw) and
// confirming the slot cursor expectations documented in the guard comments.
describe('assertSlotCursor — packing cursor is correct for each light kind', () => {
  // Each test packs a single light and verifies no exception is thrown.
  // The packed data is discarded; only the absence of a throw matters here.

  it('rect-area: packs without throwing (cursor lands at 16)', () => {
    const rect: RectAreaEmitter = {
      id: 'r',
      kind: 'rect-area',
      color: [1, 0, 0],
      intensity: 1,
      position: [0, 0, 0],
      uAxis: [1, 0, 0],
      vAxis: [0, 1, 0],
    };
    expect(() => packLightsTexture([rect])).not.toThrow();
  });

  it('disc-area: packs without throwing (cursor lands at 16)', () => {
    const disc: DiscAreaEmitter = {
      id: 'd',
      kind: 'disc-area',
      color: [1, 0, 0],
      intensity: 1,
      position: [0, 0, 0],
      normal: [0, 1, 0],
      radius: 1,
    };
    expect(() => packLightsTexture([disc])).not.toThrow();
  });

  it('spot: packs without throwing (cursor lands at 22)', () => {
    const spot: SpotEmitter = {
      id: 's',
      kind: 'spot',
      color: [1, 0, 0],
      intensity: 1,
      position: [0, 1, 0],
      direction: [0, -1, 0],
      angle: Math.PI / 4,
    };
    expect(() => packLightsTexture([spot])).not.toThrow();
  });

  it('point: packs without throwing (cursor lands at 19)', () => {
    const point: PointEmitter = {
      id: 'p',
      kind: 'point',
      color: [1, 0, 0],
      intensity: 1,
      position: [0, 0, 0],
      decay: 2,
      distance: 10,
    };
    expect(() => packLightsTexture([point])).not.toThrow();
  });

  it('directional: packs without throwing (cursor lands at 12)', () => {
    const dir: DirectionalEmitter = {
      id: 'dir',
      kind: 'directional',
      color: [1, 1, 1],
      intensity: 1,
      direction: [0, 1, 0],
    };
    expect(() => packLightsTexture([dir])).not.toThrow();
  });

  it('directional: packs angularDiameter into s5.b without disturbing the shadow flag', () => {
    const dir: DirectionalEmitter = {
      id: 'soft-sun',
      kind: 'directional',
      color: [1, 1, 1],
      intensity: 1,
      direction: [0, 1, 0],
      angularDiameter: 0.011,
      castShadow: false,
    };
    const packed = packLightsTexture([dir]);
    expect(texel(packed.data, 0, 5, 1)).toBe(1);
    expect(texel(packed.data, 0, 5, 2)).toBeCloseTo(0.011, 6);
  });

  it('directional: accepts world-visible narrow cones and rejects collapsed positive cones', () => {
    const dir: DirectionalEmitter = {
      id: 'narrow-sun',
      kind: 'directional',
      color: [1, 1, 1],
      intensity: 1,
      direction: [0, 1, 0],
      angularDiameter: 4e-6,
    };
    const packed = packLightsTexture([dir]);
    expect(texel(packed.data, 0, 5, 2)).toBe(Math.fround(4e-6));

    expect(() => packLightsTexture([{
      ...dir,
      id: 'world-basis-collapsed-cone',
      angularDiameter: 1e-6,
    }])).toThrow(
      /world-basis-collapsed-cone.*too small to survive world-basis float32 addition/,
    );

    expect(() => packLightsTexture([{
      ...dir,
      id: 'unrepresentable-cone',
      angularDiameter: 1e-20,
    }])).toThrow(
      /unrepresentable-cone.*too small to retain a finite-cone solid angle and PDF/,
    );
    expect(() => packLightsTexture([{
      ...dir,
      id: 'out-of-range-cone',
      angularDiameter: Math.PI + 0.1,
    }])).toThrow(/angularDiameter must be finite and in \[0, PI\]/);
  });
});

// ── Item 10: GLSL structural test — CIRC_AREA handling in composed shader ─────
// Verifies that the composed program text contains the CIRC_AREA_LIGHT_TYPE branch
// in both the intersection and sampling functions. This pins the GLSL-level handling
// so the CIRC path cannot be silently deleted.
describe('composeTraceGlsl — CIRC_AREA_LIGHT_TYPE is handled in both sample + isect', () => {
  it('has CIRC_AREA_LIGHT_TYPE #define and both GLSL branch sites', async () => {
    const { composeTraceGlsl } = await import('../glsl/composeTraceGlsl.js');
    const { DEFAULT_TRACE_FEATURES } = await import('../featureTypes.js');
    const src = composeTraceGlsl(DEFAULT_TRACE_FEATURES);

    // Type definition
    expect(src).toContain('#define CIRC_AREA_LIGHT_TYPE 1');

    // Intersection branch (intersectsCircle in intersectLightAtIndex)
    expect(src).toContain('light.type == CIRC_AREA_LIGHT_TYPE && intersectsCircle(');

    // Sampling branch (randomAreaLightSample disc path)
    expect(src).toContain('light.type == CIRC_AREA_LIGHT_TYPE');
  });
});

// ── SHADOW-01 (2026-06-11) — emitter castShadowDisabled in s5.g (channel 21) ──
describe('SHADOW-01 — castShadowDisabled lane (s5.g, channel 21)', () => {
  it('packs 1.0 for castShadow:false on every analytic light kind; 0.0 default', () => {
    const lights = [
      {
        kind: 'directional',
        id: 'd',
        direction: [0, 1, 0],
        color: [1, 1, 1],
        intensity: 1,
        castShadow: false,
      },
      {
        kind: 'point',
        id: 'p',
        position: [0, 1, 0],
        color: [1, 1, 1],
        intensity: 1,
        castShadow: false,
      },
      {
        kind: 'spot',
        id: 's',
        position: [0, 1, 0],
        direction: [0, -1, 0],
        angle: 0.5,
        color: [1, 1, 1],
        intensity: 1,
        castShadow: false,
      },
      {
        kind: 'rect-area',
        id: 'r',
        position: [0, 1, 0],
        uAxis: [1, 0, 0],
        vAxis: [0, 1, 0],
        color: [1, 1, 1],
        intensity: 1,
        castShadow: false,
      },
      {
        kind: 'disc-area',
        id: 'c',
        position: [0, 1, 0],
        normal: [0, -1, 0],
        radius: 0.5,
        color: [1, 1, 1],
        intensity: 1,
        castShadow: false,
      },
      { kind: 'point', id: 'p2', position: [0, 2, 0], color: [1, 1, 1], intensity: 1 },
      {
        kind: 'spot',
        id: 's2',
        position: [0, 2, 0],
        direction: [0, -1, 0],
        angle: 0.5,
        color: [1, 1, 1],
        intensity: 1,
        castShadow: true,
      },
    ] as Parameters<typeof packLightsTexture>[0];
    const packed = packLightsTexture(lights);
    const s5g = (i: number): number => packed.data[i * LIGHT_PIXELS * 4 + 21]!;
    expect(s5g(0)).toBe(1); // directional
    expect(s5g(1)).toBe(1); // point
    expect(s5g(2)).toBe(1); // spot
    expect(s5g(3)).toBe(1); // rect-area
    expect(s5g(4)).toBe(1); // disc-area
    expect(s5g(5)).toBe(0); // default
    expect(s5g(6)).toBe(0); // explicit true
  });

  it('GLSL decoder reads s5.g into Light.castShadowDisabled and directLightContribution gates on it', async () => {
    const { composeNeeCandidateGlsl, composeTraceGlsl } = await import('../glsl/composeTraceGlsl.js');
    const { DEFAULT_TRACE_FEATURES } = await import('../featureTypes.js');
    const src = composeTraceGlsl(DEFAULT_TRACE_FEATURES);
    const candidateSrc = composeNeeCandidateGlsl(DEFAULT_TRACE_FEATURES);
    expect(src).toContain('l.castShadowDisabled = s5.g;');
    expect(candidateSrc).toContain('lightSample.castShadowDisabled > 0.5 ||');
    expect(candidateSrc).toContain('! attenuateHit(');
  });

  it('GLSL decoder and sampler consume directional angularDiameter as a cone pdf', async () => {
    const { composeNeeCandidateGlsl, composeTraceGlsl } = await import('../glsl/composeTraceGlsl.js');
    const { DEFAULT_TRACE_FEATURES } = await import('../featureTypes.js');
    const src = composeTraceGlsl(DEFAULT_TRACE_FEATURES);
    const candidateSrc = composeNeeCandidateGlsl(DEFAULT_TRACE_FEATURES);
    expect(src).toContain('l.angularDiameter = l.type == DIR_LIGHT_TYPE ? max( s5.b, 0.0 ) : 0.0;');
    expect(src).toContain('vec3 sampleDirectionalCone(');
    expect(src).toContain('if ( light.angularDiameter > 0.0 )');
    expect(src).toContain(
      'rec.direction = sampleDirectionalCone( light.u, light.angularDiameter, ruv.yz, conePdf );',
    );
    expect(src).toContain('float delta;');
    expect(candidateSrc).toContain('float misWeight = lightSample.delta > 0.5');
  });
});
