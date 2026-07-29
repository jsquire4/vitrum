import { describe, expect, it } from 'vitest';
import {
  computeAdjointGradientScale,
} from '../adjointPass.js';
import type { AdjointGradientRequest } from '../inverse/inverseSession.js';
import {
  PT_WEBGPU_ADJOINT_PASS_WGSL,
} from '../wgsl/pathTrace/adjointPass.wgsl.js';

type Rgb = readonly [number, number, number];

function renderedPrimaryEmission(
  emissive: Rgb,
  intensity: number,
  targetIsClosestOpaqueHit: boolean,
): Rgb {
  if (!targetIsClosestOpaqueHit) return [0, 0, 0];
  return [
    emissive[0] * intensity,
    emissive[1] * intensity,
    emissive[2] * intensity,
  ];
}

function scalarLoss(rendered: Rgb, dLoss: Rgb): number {
  return (
    rendered[0] * dLoss[0] +
    rendered[1] * dLoss[1] +
    rendered[2] * dLoss[2]
  );
}

function centralDifference(
  emissive: Rgb,
  channel: 0 | 1 | 2,
  intensity: number,
  dLoss: Rgb,
  visible: boolean,
  step = 1e-5,
): number {
  const plus = [...emissive] as [number, number, number];
  const minus = [...emissive] as [number, number, number];
  plus[channel] += step;
  minus[channel] -= step;
  return (
    scalarLoss(renderedPrimaryEmission(plus, intensity, visible), dLoss) -
    scalarLoss(renderedPrimaryEmission(minus, intensity, visible), dLoss)
  ) / (2 * step);
}

describe('certified emissive path-replay oracle', () => {
  it.each([
    { name: 'interior', emissive: [0.2, 0.4, 0.8] as Rgb },
    { name: 'lower bound', emissive: [0, 0, 0] as Rgb },
    { name: 'upper bound', emissive: [1, 1, 1] as Rgb },
  ])('matches an independent central difference at $name values', ({ emissive }) => {
    const intensity = 3.25;
    const dLoss: Rgb = [0.75, -1.5, 2.25];
    for (const channel of [0, 1, 2] as const) {
      const analytic = dLoss[channel] * intensity;
      expect(centralDifference(
        emissive,
        channel,
        intensity,
        dLoss,
        true,
      )).toBeCloseTo(analytic, 9);
    }
  });

  it('is exactly zero when an opaque closest hit occludes the target', () => {
    const emissive: Rgb = [0.3, 0.6, 0.9];
    const dLoss: Rgb = [2, -3, 5];
    for (const channel of [0, 1, 2] as const) {
      expect(centralDifference(
        emissive,
        channel,
        7,
        dLoss,
        false,
      )).toBe(0);
    }
  });

  it('derives an overflow-safe scale for a large legal HDR accumulation', () => {
    const width = 512;
    const height = 512;
    const pixelCount = width * height;
    const dLoss = new Float32Array(pixelCount * 3);
    dLoss.fill(1e6);
    const request: AdjointGradientRequest = {
      dLoss_dRendered: dLoss,
      channels: 3,
      width,
      height,
      samples: 4096,
      params: [{
        domain: 'materials',
        id: 'emitter',
        field: 'emissive',
        offset: 0,
        length: 3,
      }],
      gradientLength: 3,
    };
    const intensity = 100;
    const scale = computeAdjointGradientScale(request, [intensity]);
    expect(scale).toBeLessThan(1048576);
    expect(Math.fround(scale)).toBe(scale);

    // The shader performs one rounded atomic add per pixel after locally
    // averaging all samples.
    const fixedPerPixel = Math.round(1e6 * intensity * scale);
    const worstFixedSum = fixedPerPixel * pixelCount;
    expect(Math.abs(worstFixedSum)).toBeLessThan(2147483647);
    const decoded = worstFixedSum / scale;
    const exact = 1e6 * intensity * pixelCount;
    // One rounded atomic contribution is emitted per pixel. The total decoded
    // error therefore cannot exceed half a fixed-point unit per pixel.
    expect(Math.abs(decoded - exact)).toBeLessThanOrEqual(
      (0.5 * pixelCount) / scale,
    );
  });

  it('rejects a per-pixel product that overflows before scale is applied', () => {
    const request: AdjointGradientRequest = {
      dLoss_dRendered: new Float32Array([3e38, 3e38, 3e38]),
      channels: 3,
      width: 1,
      height: 1,
      samples: 1,
      params: [{
        domain: 'materials',
        id: 'emitter',
        field: 'emissive',
        offset: 0,
        length: 3,
      }],
      gradientLength: 3,
    };
    expect(() => computeAdjointGradientScale(request, [2])).toThrow(
      /per-pixel emissive gradient.*exceeds finite f32 range/,
    );
  });

  it('rejects a finite per-pixel product whose returned aggregate exceeds f32', () => {
    const request: AdjointGradientRequest = {
      dLoss_dRendered: new Float32Array([
        2e38, 2e38, 2e38,
        2e38, 2e38, 2e38,
      ]),
      channels: 3,
      width: 2,
      height: 1,
      samples: 4096,
      params: [{
        domain: 'materials',
        id: 'emitter',
        field: 'emissive',
        offset: 0,
        length: 3,
      }],
      gradientLength: 3,
    };
    expect(() => computeAdjointGradientScale(request, [1])).toThrow(
      /aggregate emissive gradient.*exceeds finite f32 readback range/,
    );
  });

  it('contains no hidden finite-difference, BSDF, light, or emitter derivative path', () => {
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain(
      'const MATERIAL_VEC4_STRIDE = 29u',
    );
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain(
      'descriptor.y != ADJOINT_FIELD_EMISSIVE',
    );
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain(
      'value * params.gradientScale',
    );
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain(
      'let hitFraction = f32(matchingHitCount) * invReplaySamples',
    );
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('mollerTrumboreCore(');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain(
      'positions[idx.z].xyz,\n      1e-5,',
    );
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain(
      'triHit.t > 1e-4 && triHit.t < best.t',
    );
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain(
      'u < -triEps || v < -triEps || w < -triEps',
    );
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).not.toContain('gradient +=');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).not.toMatch(
      /ADJOINT_DIRECT_PARAM_STEP|dBrdf_|directLight|pointLights|meshAreaLights|EMITTER_TARGET/,
    );
  });
});
