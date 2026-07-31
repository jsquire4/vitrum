import { describe, expect, it } from 'vitest';
import type { HdriEnvironment, NoneEnvironment, ProceduralSkyEnvironment } from '@vitrum/core';
import { buildEquirectInfo } from './equirectHdrInfo.js';
import * as SamplingEquirect from '../glsl/shader/sampling/equirect_sampling_functions.glsl.js';

const equirectFunctions = (SamplingEquirect as unknown as Record<string, string>)
  .equirect_functions;

const luminance = (r: number, g: number, b: number) => 0.2126 * r + 0.7152 * g + 0.0722 * b;
const texelSolidAngle = (y: number, width: number, height: number) =>
  ((2 * Math.PI) / width) *
  (Math.cos((y / height) * Math.PI) - Math.cos(((y + 1) / height) * Math.PI));

// A tiny 4×2 synthetic equirect HDRI. Row 0 is dim, row 1 has a bright hot spot
// in the second column. Row-major RGB float pixels (3 floats/pixel), the shape
// the opaque core EnvironmentMapRef carries for the 'hdri' kind.
function tinyHdri(): HdriEnvironment {
  const width = 4;
  const height = 2;
  const px: [number, number, number][] = [
    // row 0 (dim, near-uniform)
    [0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1],
    // row 1 (one very bright pixel)
    [0.2, 0.2, 0.2],
    [9.0, 9.0, 9.0],
    [0.2, 0.2, 0.2],
    [0.2, 0.2, 0.2],
  ];
  const data = new Float32Array(width * height * 3);
  for (let i = 0; i < px.length; i += 1) {
    data[i * 3 + 0] = px[i]![0];
    data[i * 3 + 1] = px[i]![1];
    data[i * 3 + 2] = px[i]![2];
  }
  return { kind: 'hdri', hdri: { width, height, data } };
}

describe('buildEquirectInfo', () => {
  it('returns all-null grids for none environments', () => {
    const none: NoneEnvironment = { kind: 'none' };
    const out = buildEquirectInfo(none);
    expect(out.map).toBeNull();
    expect(out.marginal).toBeNull();
    expect(out.conditional).toBeNull();
    expect(out.totalSum).toBe(0);
  });

  it('rejects an hdri lacking CPU pixel data', () => {
    expect(() => buildEquirectInfo({ kind: 'hdri', hdri: { mock: true } })).toThrow(
      /authored HDRI is not CPU-readable: no raw or DataTexture-shaped pixel data was supplied/,
    );
  });

  it('rejects a short DataTexture-shaped HDRI payload', () => {
    expect(() =>
      buildEquirectInfo({
        kind: 'hdri',
        hdri: {
          image: {
            width: 2,
            height: 1,
            data: new Float32Array([1]),
          },
        },
      }),
    ).toThrow(/data length 1 must exactly equal 6 \(RGB\) or 8 \(RGBA\)/);
  });

  it.each([
    [
      'fractional dimensions',
      { width: 1.5, height: 1, data: new Float32Array([1, 1, 1]) },
      /width and height must be positive safe integers/,
    ],
    [
      'non-finite dimensions',
      { width: Number.POSITIVE_INFINITY, height: 1, data: new Float32Array([1, 1, 1]) },
      /width and height must be positive safe integers/,
    ],
    [
      'trailing pixel values',
      { width: 1, height: 1, data: new Float32Array([1, 1, 1, 1, 99]) },
      /data length 5 must exactly equal 3 \(RGB\) or 4 \(RGBA\)/,
    ],
    [
      'non-finite radiance',
      { width: 1, height: 1, data: new Float32Array([Number.NaN, 1, 1]) },
      /every RGB radiance value and optional alpha value must decode to a finite float/,
    ],
    [
      'negative radiance',
      { width: 1, height: 1, data: new Float32Array([-0.25, 1, 1]) },
      /radiance must be finite and nonnegative/,
    ],
  ])('rejects %s instead of degrading to EMPTY_ENV', (_label, hdri, message) => {
    expect(() => buildEquirectInfo({ kind: 'hdri', hdri })).toThrow(message);
  });

  it('rejects incompatible dataType backing and non-RGB channel hints', () => {
    expect(() =>
      buildEquirectInfo({
        kind: 'hdri',
        hdri: {
          width: 1,
          height: 1,
          data: new Uint16Array([1, 1, 1]),
          __vitrum_hint__: { channels: 3, dataType: 'uint8' },
        },
      }),
    ).toThrow(/dataType "uint8" requires Uint8Array or Uint8ClampedArray/);
    expect(() =>
      buildEquirectInfo({
        kind: 'hdri',
        hdri: {
          width: 1,
          height: 1,
          data: new Float32Array([1, 1]),
          __vitrum_hint__: { channels: 2, dataType: 'float32' },
        },
      }),
    ).toThrow(/channels must be 3 \(RGB\) or 4 \(RGBA\)/);
  });

  it('accepts DataTexture-shaped HDRI handles with explicit channel hints', () => {
    const out = buildEquirectInfo({
      kind: 'hdri',
      hdri: {
        image: {
          width: 2,
          height: 1,
          data: new Float32Array([1, 0, 0, 1, 0, 1, 0, 1]),
        },
        __vitrum_hint__: { channels: 4, dataType: 'float32', colorSpace: 'linear' },
      },
    });

    expect(out.map).not.toBeNull();
    expect(out.map!.width).toBe(2);
    expect(out.map!.height).toBe(1);
    expect(out.map!.data[0]).toBeCloseTo(1);
    expect(out.map!.data[4]).toBeCloseTo(0);
    expect(out.map!.data[5]).toBeCloseTo(1);
    expect(out.totalSum).toBeGreaterThan(0);
  });

  it('normalizes hinted uint8 sRGB HDRI handles before building CDFs', () => {
    const out = buildEquirectInfo({
      kind: 'hdri',
      hdri: {
        width: 1,
        height: 1,
        data: new Uint8Array([128, 255, 0, 255]),
        __vitrum_hint__: { channels: 4, dataType: 'uint8', colorSpace: 'srgb' },
      },
    });

    expect(out.map).not.toBeNull();
    expect(out.map!.data[0]).toBeCloseTo(0.21586, 4);
    expect(out.map!.data[1]).toBeCloseTo(1);
    expect(out.map!.data[2]).toBeCloseTo(0);
    expect(out.totalSum).toBeGreaterThan(0);
  });

  it('rejects HDRI intensity that cannot cross the WebGL float32 boundary', () => {
    expect(() =>
      buildEquirectInfo({
        kind: 'hdri',
        intensity: 1e300,
        hdri: {
          width: 1,
          height: 1,
          data: new Float32Array([1, 1, 1]),
        },
      }),
    ).toThrow(/HDRI environment intensity overflows WebGL float32 storage/);
  });

  it('rejects a finite HDRI texel and intensity whose realized radiance overflows', () => {
    expect(() =>
      buildEquirectInfo({
        kind: 'hdri',
        intensity: 100,
        hdri: {
          width: 1,
          height: 1,
          data: new Float32Array([1e37, 0, 0]),
        },
      }),
    ).toThrow(
      /realized radiance pixel 0 color\[0\] \* intensity overflows shader float32 multiplication/,
    );
  });

  it('rejects a positive HDRI texel and intensity whose realized support underflows', () => {
    expect(() =>
      buildEquirectInfo({
        kind: 'hdri',
        intensity: 0.01,
        hdri: {
          width: 1,
          height: 1,
          data: new Float32Array([1e-44, 0, 0]),
        },
      }),
    ).toThrow(
      /realized radiance pixel 0 color\[0\] \* intensity underflows shader float32 multiplication/,
    );
  });

  it('rejects an environment importance mass that cannot cross a float uniform boundary', () => {
    expect(() =>
      buildEquirectInfo({
        kind: 'hdri',
        hdri: {
          width: 1,
          height: 1,
          data: new Float32Array([3e38, 3e38, 3e38]),
        },
      }),
    ).toThrow(/environment importance mass overflows WebGL float32 storage/);
  });

  const env = tinyHdri();
  const out = buildEquirectInfo(env);

  it('produces map + marginal + conditional grids with the right dims', () => {
    expect(out.map).not.toBeNull();
    expect(out.map!.width).toBe(4);
    expect(out.map!.height).toBe(2);
    // marginal is 1×height (height columns, 1 row)
    expect(out.marginal!.width).toBe(2);
    expect(out.marginal!.height).toBe(1);
    // conditional is width×height
    expect(out.conditional!.width).toBe(4);
    expect(out.conditional!.height).toBe(2);
  });

  it('packs the marginal inverse CDF into the conditional texture green lane', () => {
    const marginal = out.marginal!.data;
    const distribution = out.conditional!;
    for (let y = 0; y < distribution.height; y += 1) {
      for (let x = 0; x < distribution.width; x += 1) {
        const green = distribution.data[(y * distribution.width + x) * 4 + 1]!;
        expect(green).toBe(marginal[y * 4]!);
      }
    }
  });

  it('totalSum equals the solid-angle-weighted luminance integral', () => {
    let expected = 0;
    const { width, height, data } = env.hdri as {
      width: number;
      height: number;
      data: Float32Array;
    };
    for (let i = 0; i < width * height; i += 1) {
      const y = Math.floor(i / width);
      expected +=
        luminance(data[i * 3]!, data[i * 3 + 1]!, data[i * 3 + 2]!) *
        texelSolidAngle(y, width, height);
    }
    expect(out.totalSum).toBeCloseTo(expected, 5);
  });

  it('the marginal forward-CDF table is monotonic and ends exactly at one', () => {
    const m = out.marginal!.data;
    const n = out.marginal!.width; // = height
    let prev = 0;
    for (let i = 0; i < n; i += 1) {
      const v = m[i * 4 + 0]!;
      expect(v).toBeGreaterThanOrEqual(prev);
      expect(v).toBeGreaterThan(0);
      expect(v).toBeLessThanOrEqual(1);
      prev = v;
    }
    expect(prev).toBe(1);
  });

  it('the brighter row dominates the marginal distribution', () => {
    const m = out.marginal!.data;
    const row0Pmf = m[0]!;
    const row1Pmf = m[4]! - m[0]!;
    expect(row0Pmf).toBeLessThan(0.1);
    expect(row1Pmf).toBeGreaterThan(0.9);
  });

  it('weights uniform equirect rows by solid angle so equator rows dominate poles', () => {
    const data = new Float32Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
    const uniform = buildEquirectInfo({
      kind: 'hdri',
      hdri: {
        width: 1,
        height: 3,
        data,
        __vitrum_hint__: { channels: 4, dataType: 'float32', colorSpace: 'linear' },
      },
    });

    expect(uniform.totalSum).toBeCloseTo(4 * Math.PI, 6);
    const m = uniform.marginal!.data;
    expect(m[0]!).toBeCloseTo(0.25, 6);
    expect(m[4]!).toBeCloseTo(0.75, 6);
    expect(m[8]!).toBe(1);
  });

  it('packs a normalized realized solid-angle PDF into map alpha', () => {
    const map = out.map!;
    let integrated = 0;
    for (let y = 0; y < map.height; y += 1) {
      const theta0 = (y / map.height) * Math.PI;
      const theta1 = ((y + 1) / map.height) * Math.PI;
      const cellSolidAngle = ((2 * Math.PI) / map.width) * (Math.cos(theta0) - Math.cos(theta1));
      for (let x = 0; x < map.width; x += 1) {
        integrated += map.data[(y * map.width + x) * 4 + 3]! * cellSolidAngle;
      }
    }
    expect(integrated).toBeCloseTo(1, 6);
  });

  it('does not lose a representable source bin to an inverse-table quantizer', () => {
    const candidate = buildEquirectInfo({
      kind: 'hdri',
      hdri: {
        width: 4,
        height: 1,
        data: new Float32Array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0, 0, 0, 0, 0, 0]),
      },
    });
    expect(candidate.conditional!.data[0]).toBeCloseTo(0.1, 6);
    expect(candidate.map!.data[3]).toBeGreaterThan(0);
    expect(candidate.map!.data[7]).toBeGreaterThan(0);
  });

  it('reports zero density for a bin flattened by the uploaded Float32 CDF', () => {
    const candidate = buildEquirectInfo({
      kind: 'hdri',
      hdri: {
        width: 2,
        height: 1,
        data: new Float32Array([1, 1, 1, 1e-8, 1e-8, 1e-8]),
      },
    });
    expect(candidate.conditional!.data[0]).toBe(1);
    expect(candidate.conditional!.data[4]).toBe(1);
    expect(candidate.map!.data[7]).toBe(0);
  });

  it('bakes procedural-sky environments into the equirect HDRI path', () => {
    const sky: ProceduralSkyEnvironment = {
      kind: 'procedural-sky',
      sunDirection: [0, 1, 0],
      turbidity: 2,
      rayleigh: 1,
      mieCoefficient: 0.005,
      mieDirectionalG: 0.8,
      intensity: 1,
    };

    const skyOut = buildEquirectInfo(sky);

    expect(skyOut.map).not.toBeNull();
    expect(skyOut.map!.width).toBe(256);
    expect(skyOut.map!.height).toBe(128);
    expect(skyOut.marginal!.width).toBe(128);
    expect(skyOut.conditional!.width).toBe(256);
    expect(skyOut.conditional!.height).toBe(128);
    expect(skyOut.totalSum).toBeGreaterThan(0);
  });

  it('honors zero procedural-sky intensity as a black environment', () => {
    const sky: ProceduralSkyEnvironment = {
      kind: 'procedural-sky',
      sunDirection: [0, 1, 0],
      turbidity: 2,
      rayleigh: 1,
      mieCoefficient: 0.005,
      mieDirectionalG: 0.8,
      intensity: 0,
    };

    const skyOut = buildEquirectInfo(sky);

    expect(skyOut.map).not.toBeNull();
    expect(skyOut.totalSum).toBe(0);
    expect(Array.from(skyOut.map!.data).every((v) => v === 0)).toBe(true);
  });

  it('places the procedural-sky maximum near the authored sun direction', () => {
    const sky: ProceduralSkyEnvironment = {
      kind: 'procedural-sky',
      sunDirection: [1, 0.05, 0],
      turbidity: 2,
      rayleigh: 1,
      mieCoefficient: 0.005,
      mieDirectionalG: 0.8,
      intensity: 1,
    };
    const out = buildEquirectInfo(sky);
    const map = out.map!;
    let maxLum = -Infinity;
    let maxIdx = 0;
    for (let i = 0; i < map.width * map.height; i += 1) {
      const lum = luminance(map.data[i * 4]!, map.data[i * 4 + 1]!, map.data[i * 4 + 2]!);
      if (lum > maxLum) {
        maxLum = lum;
        maxIdx = i;
      }
    }

    const pyMax = (maxIdx / map.width) | 0;
    const pxMax = maxIdx % map.width;
    const thetaMax = ((pyMax + 0.5) / map.height) * Math.PI;
    const phiMax = ((pxMax + 0.5) / map.width - 0.5) * (2 * Math.PI);

    expect(thetaMax).toBeGreaterThan(Math.PI / 2 - 0.3);
    expect(thetaMax).toBeLessThan(Math.PI / 2 + 0.3);
    expect(Math.abs(phiMax)).toBeLessThan(0.3);
  });

  it.each([
    [
      [0, 1, 0],
      [0, -1, 0],
    ],
    [
      [0.3, 0.8, 0.4],
      [-0.3, -0.8, -0.4],
    ],
  ] as const)(
    'keeps canonical row-zero-north lookup aligned with procedural sun %j',
    (sun, opposite) => {
      const length = Math.hypot(...sun);
      const normalizedSun: [number, number, number] = [
        sun[0] / length,
        sun[1] / length,
        sun[2] / length,
      ];
      const candidate = buildEquirectInfo({
        kind: 'procedural-sky',
        sunDirection: normalizedSun,
        turbidity: 2,
        rayleigh: 1,
        mieCoefficient: 0.005,
        mieDirectionalG: 0.8,
        intensity: 1,
      });
      const map = candidate.map!;
      const lookup = (direction: readonly [number, number, number]): number => {
        const directionLength = Math.hypot(...direction);
        const phi = Math.atan2(direction[2] / directionLength, direction[0] / directionLength);
        const theta = Math.acos(direction[1] / directionLength);
        const u = (((phi / (2 * Math.PI) + 0.5) % 1) + 1) % 1;
        const v = Math.min(theta / Math.PI, 0.99999994);
        const x = Math.min(Math.floor(u * map.width), map.width - 1);
        const y = Math.min(Math.floor(v * map.height), map.height - 1);
        const offset = (y * map.width + x) * 4;
        return luminance(map.data[offset]!, map.data[offset + 1]!, map.data[offset + 2]!);
      };

      expect(lookup(normalizedSun)).toBeGreaterThan(lookup(opposite) * 2);
    },
  );

  it('uses forward-CDF residuals for continuous solid-angle GLSL sampling', () => {
    expect(equirectFunctions).toContain(
      'float rowResidual = clamp( ( rowXi - rowPrior ) / rowWidth',
    );
    expect(equirectFunctions).toContain('float columnResidual = clamp(');
    expect(equirectFunctions).toContain(
      'float cosTheta = mix( cos( theta0 ), cos( theta1 ), rowResidual );',
    );
    expect(equirectFunctions).toContain(
      'bool equirectScaleIsRepresentable( vec3 value, float scale )',
    );
    expect(equirectFunctions).toContain(
      'vec3 scaled = value * scale;',
    );
    expect(equirectFunctions).toContain(
      '! any( isnan( scaled ) ) && ! any( isinf( scaled ) )',
    );
    expect(equirectFunctions).toContain(
      'finiteEquirectScaledColor( color, environmentIntensity )',
    );
    expect(
      equirectFunctions?.match(/finiteEquirectEnvironmentColor\( packed\.rgb \)/g) ?? [],
    ).toHaveLength(2);
    expect(
      equirectFunctions?.match(/return finiteEquirectPdf\( packed\.a \);/g) ?? [],
    ).toHaveLength(2);
    expect(equirectFunctions).not.toContain('color = packed.rgb;');
    expect(equirectFunctions).not.toContain('equirectDirectionToUv( direction )');
  });
});
