import { describe, expect, it } from 'vitest';
import { PT_WEBGPU_ADJOINT_PASS_WGSL } from '../wgsl/pathTrace/adjointPass.wgsl.js';

function powerHeuristic(lightPdf: number, brdfPdf: number): number {
  const light2 = lightPdf * lightPdf;
  const brdf2 = brdfPdf * brdfPdf;
  return light2 / (light2 + brdf2);
}

describe('complete direct-light adjoint estimator', () => {
  it('includes the derivative of the BRDF/light MIS weight', () => {
    const lightPdf = 0.37;
    const liCos = 2.4;
    const theta = 0.41;
    const brdf = (value: number) => 0.18 + 0.73 * value;
    const brdfPdf = (value: number) => 0.09 + 0.52 * value;
    const contribution = (value: number) =>
      liCos * brdf(value) * powerHeuristic(lightPdf, brdfPdf(value));

    const epsilon = 1e-5;
    const finiteDifference =
      (contribution(theta + epsilon) - contribution(theta - epsilon)) /
      (2 * epsilon);
    const pdf = brdfPdf(theta);
    const denominator = lightPdf * lightPdf + pdf * pdf;
    const mis = powerHeuristic(lightPdf, pdf);
    const dMis = -2 * lightPdf * lightPdf * pdf * 0.52 /
      (denominator * denominator);
    const analytic = liCos * (0.73 * mis + brdf(theta) * dMis);
    const incorrectlyFrozenMis = liCos * 0.73 * mis;

    expect(finiteDifference).toBeCloseTo(analytic, 8);
    expect(Math.abs(finiteDifference - incorrectlyFrozenMis)).toBeGreaterThan(0.1);
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain(
      'adjointDirectContributionForMaterial(plus',
    );
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain(
      'material.anisotropyRotation,\n    Li,\n    lightPdf,',
    );
  });

  it('hard-gates every unlit field without a direct colour/emission response', () => {
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('if (isUnlit &&');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('d.y != 0u &&'); // baseColor
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('d.y != 19u &&'); // AO
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('d.y != 20u &&'); // light map
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('d.y != 2u &&'); // emissive
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('d.y != 6u)'); // emissive intensity
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain(
      'their gradients are bit-exact zero',
    );
  });
});
