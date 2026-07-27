import { describe, expect, it } from 'vitest';

import {
  ROUGH_DIELECTRIC_SMOOTH_THRESHOLD,
  roughDielectricAdd,
  roughDielectricCross,
  roughDielectricDot,
  roughDielectricEventProbabilities,
  roughDielectricFresnel,
  roughDielectricHalfVector,
  roughDielectricJacobian,
  roughDielectricNormalize,
  roughDielectricReflectionEval,
  roughDielectricScale,
  roughDielectricSmithG1,
  roughDielectricSmithG1RoughnessDerivative,
  roughDielectricSmithG1Wgsl,
  roughDielectricTransmissionEval,
  roughDielectricTransmissionPdf,
  roughDielectricVisibleNormalPdf,
  sampleRoughDielectric,
  type RoughDielectricConfig,
  type RoughDielectricVec3,
} from '../math/roughDielectric.js';
import { PT_WEBGPU_ADJOINT_PASS_WGSL } from '../wgsl/pathTrace/adjointPass.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_BSDF_WGSL } from '../wgsl/pathTrace/bsdf.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_MATERIAL_FUNCS_WGSL } from '../wgsl/pathTrace/material.wgsl.js';
import { RESTIR_PT_HYBRID_SHIFT_WGSL } from '../wgsl/pathTrace/restirPtHybridShift.wgsl.js';

const N: RoughDielectricVec3 = [0, 0, 1];
const TWO_PI = 2 * Math.PI;

function radicalInverse(index: number, base: number): number {
  let value = 0;
  let factor = 1 / base;
  let n = index;
  while (n > 0) {
    value += (n % base) * factor;
    n = Math.floor(n / base);
    factor /= base;
  }
  return value;
}

function direction(cosTheta: number, phi = 0, hemisphere = 1): RoughDielectricVec3 {
  const sinTheta = Math.sqrt(Math.max(0, 1 - cosTheta * cosTheta));
  return [sinTheta * Math.cos(phi), sinTheta * Math.sin(phi), hemisphere * cosTheta];
}

function integrateHemisphere(
  hemisphere: 1 | -1,
  integrand: (wi: RoughDielectricVec3) => number,
  thetaSteps = 512,
  phiSteps = 512,
): number {
  let sum = 0;
  const dTheta = 0.5 * Math.PI / thetaSteps;
  const dPhi = TWO_PI / phiSteps;
  for (let it = 0; it < thetaSteps; it += 1) {
    const theta = (it + 0.5) * dTheta;
    const z = Math.cos(theta);
    const radial = Math.sin(theta);
    for (let ip = 0; ip < phiSteps; ip += 1) {
      const phi = (ip + 0.5) * dPhi;
      sum += integrand([radial * Math.cos(phi), radial * Math.sin(phi), hemisphere * z]) * radial;
    }
  }
  return sum * dTheta * dPhi;
}

function integrateVisibleNormalSlopes(
  config: RoughDielectricConfig,
  wo: RoughDielectricVec3,
  steps: number,
): number {
  let sum = 0;
  const du = 2 / steps;
  for (let iy = 0; iy < steps; iy += 1) {
    const uy = -1 + (iy + 0.5) * du;
    const slopeY = Math.tan(0.5 * Math.PI * uy);
    const jacobianY = 0.5 * Math.PI * (1 + slopeY * slopeY);
    for (let ix = 0; ix < steps; ix += 1) {
      const ux = -1 + (ix + 0.5) * du;
      const slopeX = Math.tan(0.5 * Math.PI * ux);
      const jacobianX = 0.5 * Math.PI * (1 + slopeX * slopeX);
      const wm = roughDielectricNormalize([slopeX, slopeY, 1]);
      const slopeSolidAngle = 1 / (1 + slopeX * slopeX + slopeY * slopeY) ** 1.5;
      sum += roughDielectricVisibleNormalPdf(config, N, wo, wm) *
        slopeSolidAngle * jacobianX * jacobianY;
    }
  }
  return sum * du * du;
}

function transmissionMixturePdf(
  config: RoughDielectricConfig,
  wo: RoughDielectricVec3,
  wi: RoughDielectricVec3,
): number {
  const wm = roughDielectricHalfVector(N, wo, wi, config.etaTOverI);
  return roughDielectricEventProbabilities(config, N, wo, wm, 1).transmission *
    roughDielectricTransmissionPdf(config, N, wo, wi);
}

describe('rough dielectric numerical production oracle', () => {
  it('normalizes the visible-normal density across roughness and angle', () => {
    for (const roughness of [0.2, 0.5, 0.9]) {
      for (const cosO of [1, 0.7, 0.25]) {
        const config = { roughness, etaTOverI: 1.5 };
        const wo = direction(cosO);
        const coarse = integrateVisibleNormalSlopes(config, wo, 256);
        const fine = integrateVisibleNormalSlopes(config, wo, 512);
        const extrapolated = fine + (fine - coarse) / 3;
        expect(
          Math.abs(fine - coarse), `error estimate r=${roughness} cosO=${cosO}`,
        ).toBeLessThan(2e-3);
        expect(
          Math.abs(extrapolated - fine), `extrapolation r=${roughness} cosO=${cosO}`,
        ).toBeLessThan(7e-4);
        expect(extrapolated, `roughness=${roughness} cosO=${cosO}`).toBeCloseTo(1, 3);
      }
    }
  });

  it('pins visible-normal support to the incident-facing hemisphere', () => {
    const config = { roughness: 0.2, etaTOverI: 1.5 };
    const wo = direction(0.25);
    const tangentWm = roughDielectricNormalize([-wo[2], 0, wo[0]]);
    const invisibleWm = roughDielectricNormalize([-1, 0, 0.01]);
    expect(Math.abs(roughDielectricDot(wo, tangentWm))).toBeLessThan(1e-12);
    expect(roughDielectricVisibleNormalPdf(config, N, wo, tangentWm)).toBe(0);
    expect(roughDielectricDot(wo, invisibleWm)).toBeLessThan(0);
    expect(roughDielectricVisibleNormalPdf(config, N, wo, invisibleWm)).toBe(0);

    for (let i = 0; i <= 128; i += 1) {
      const wm = direction(i / 128, TWO_PI * radicalInverse(i + 1, 2));
      const pdf = roughDielectricVisibleNormalPdf(config, N, wo, wm);
      expect(Number.isFinite(pdf)).toBe(true);
      expect(pdf).toBeGreaterThanOrEqual(0);
      if (roughDielectricDot(wo, wm) <= 0) expect(pdf).toBe(0);
    }
  });

  it('matches exact Smith values and roughness derivatives across angle/roughness grids', () => {
    for (const roughness of [0.04, 0.1, 0.2, 0.5, 0.9]) {
      for (const cosTheta of [0.999999, 0.95, 0.5, 0.1, 0.001]) {
        const alpha = Math.max(roughness * roughness, 1e-3);
        const tanTheta2 = (1 - cosTheta * cosTheta) /
          (cosTheta * cosTheta);
        const independentValue = 2 /
          (1 + Math.sqrt(1 + alpha * alpha * tanTheta2));
        const value = roughDielectricSmithG1(cosTheta, roughness);
        expect(value).toBeCloseTo(independentValue, 14);
        expect(Number.isFinite(value)).toBe(true);
        expect(value).toBeGreaterThanOrEqual(0);
        expect(value).toBeLessThanOrEqual(1);

        const eps = 1e-6;
        const finiteDifference = (
          roughDielectricSmithG1(cosTheta, roughness + eps) -
          roughDielectricSmithG1(cosTheta, roughness - eps)
        ) / (2 * eps);
        const analytic = roughDielectricSmithG1RoughnessDerivative(
          cosTheta,
          roughness,
        );
        const error = Math.abs(analytic - finiteDifference);
        expect(error).toBeLessThan(2e-8 + 2e-6 * Math.abs(finiteDifference));
      }
    }
  });

  it('integrates transmitted PDFs and matches VNDF/event sampling frequencies', () => {
    const sampleCount = 32_768;
    for (const roughness of [0.2, 0.4, 0.75]) {
      for (const etaTOverI of [1 / 1.5, 1.5, 2]) {
        for (const cosO of [1, 0.5]) {
          const config = { roughness, etaTOverI };
          const wo = direction(cosO);
          const conditionalIntegral = integrateHemisphere(
            -1, (wi) => roughDielectricTransmissionPdf(config, N, wo, wi),
          );
          const mixtureIntegral = integrateHemisphere(
            -1, (wi) => transmissionMixturePdf(config, wo, wi),
          );
          let validRefractions = 0;
          let sampledEvents = 0;
          let expectedEvents = 0;
          for (let i = 1; i <= sampleCount; i += 1) {
            const sample = sampleRoughDielectric(
              config, N, wo, 1, radicalInverse(i, 2), radicalInverse(i, 3),
            );
            const valid = sample.wi != null && roughDielectricDot(N, sample.wi) < -1e-8;
            if (valid) {
              validRefractions += 1;
              expectedEvents += sample.probabilities.transmission;
              if (radicalInverse(i, 5) < sample.probabilities.transmission) sampledEvents += 1;
            }
            expect(
              sample.probabilities.reflection + sample.probabilities.diffuse +
                sample.probabilities.transmission,
            ).toBeCloseTo(1, 12);
          }
          const label = `r=${roughness} eta=${etaTOverI} cosO=${cosO}`;
          expect(conditionalIntegral, label).toBeCloseTo(validRefractions / sampleCount, 2);
          expect(mixtureIntegral, label).toBeCloseTo(expectedEvents / sampleCount, 2);
          expect(sampledEvents / sampleCount, label).toBeCloseTo(mixtureIntegral, 2);
        }
      }
    }
  }, 20_000);

  it('matches an independent finite-difference Walter refraction Jacobian', () => {
    const independentRefract = (
      wo: RoughDielectricVec3,
      wm: RoughDielectricVec3,
      etaTOverI: number,
    ): RoughDielectricVec3 => {
      const incoming = roughDielectricScale(wo, -1);
      const etaIOverT = 1 / etaTOverI;
      const nDotI = roughDielectricDot(wm, incoming);
      const k = 1 - etaIOverT * etaIOverT * (1 - nDotI * nDotI);
      if (!(k > 0)) throw new Error('Jacobian fixture unexpectedly reached TIR');
      return roughDielectricNormalize(roughDielectricAdd(
        roughDielectricScale(incoming, etaIOverT),
        roughDielectricScale(wm, -(etaIOverT * nDotI + Math.sqrt(k))),
      ));
    };

    const config = { roughness: 0.42, etaTOverI: 1.5 };
    const wo = direction(0.73, 0.31);
    for (const [u1, u2] of [[0.11, 0.23], [0.43, 0.77], [0.81, 0.36]] as const) {
      const sample = sampleRoughDielectric(config, N, wo, 1, u1, u2);
      expect(sample.wi).not.toBeNull();
      const wi = sample.wi!;
      expect(roughDielectricDot(N, wi)).toBeLessThan(0);
      const reconstructed = roughDielectricHalfVector(N, wo, wi, config.etaTOverI);
      expect(Math.abs(roughDielectricDot(reconstructed, sample.wm))).toBeCloseTo(1, 10);

      const referenceAxis: RoughDielectricVec3 = Math.abs(sample.wm[1]) < 0.9
        ? [0, 1, 0]
        : [1, 0, 0];
      const tangent = roughDielectricNormalize(roughDielectricCross(sample.wm, referenceAxis));
      const bitangent = roughDielectricCross(sample.wm, tangent);
      const eps = 1e-5;
      const perturb = (axis: RoughDielectricVec3, amount: number) =>
        roughDielectricNormalize(roughDielectricAdd(sample.wm, roughDielectricScale(axis, amount)));
      const wiUPlus = independentRefract(wo, perturb(tangent, eps), config.etaTOverI);
      const wiUMinus = independentRefract(wo, perturb(tangent, -eps), config.etaTOverI);
      const wiVPlus = independentRefract(wo, perturb(bitangent, eps), config.etaTOverI);
      const wiVMinus = independentRefract(wo, perturb(bitangent, -eps), config.etaTOverI);
      const derivative = (
        plus: RoughDielectricVec3,
        minus: RoughDielectricVec3,
      ) => roughDielectricScale(
        roughDielectricAdd(plus, roughDielectricScale(minus, -1)),
        1 / (2 * eps),
      );
      const dWiDu = derivative(wiUPlus, wiUMinus);
      const dWiDv = derivative(wiVPlus, wiVMinus);
      const dOmegaWiPerDomegaM = Math.abs(roughDielectricDot(
        roughDielectricCross(dWiDu, dWiDv),
        wi,
      ));
      const numericalDomegaMPerDomegaWi = 1 / dOmegaWiPerDomegaM;
      expect(
        roughDielectricJacobian(N, wo, wi, config.etaTOverI),
      ).toBeCloseTo(numericalDomegaMPerDomegaWi, 5);
    }
  });

  it('keeps integrated dielectric reflection plus radiance transmission at or below one', () => {
    for (const roughness of [0.4, 0.7, 1]) {
      for (const etaTOverI of [1.25, 1.5, 2]) {
        for (const cosO of [1, 0.7, 0.3]) {
          const config = { roughness, etaTOverI };
          const wo = direction(cosO, 0.17);
          const integrateEnergy = (steps: number) => {
            const reflected = integrateHemisphere(
              1,
              (wi) => roughDielectricReflectionEval(config, N, wo, wi) *
                roughDielectricDot(N, wi),
              steps,
              steps,
            );
            const transmitted = integrateHemisphere(
              -1,
              (wi) => roughDielectricTransmissionEval(config, N, wo, wi, false) *
                Math.abs(roughDielectricDot(N, wi)),
              steps,
              steps,
            );
            return reflected + transmitted;
          };
          const coarse = integrateEnergy(256);
          const fine = integrateEnergy(512);
          const extrapolated = fine + (fine - coarse) / 3;
          const label = `r=${roughness} eta=${etaTOverI} cosO=${cosO}`;
          expect(Math.abs(fine - coarse), `energy convergence ${label}`).toBeLessThan(3e-3);
          expect(extrapolated, label).toBeGreaterThanOrEqual(0);
          expect(extrapolated, label).toBeLessThanOrEqual(1 + 5e-4);
        }
      }
    }
  }, 20_000);

  it('obeys eta-squared transport reciprocity and finite forward/reverse PDFs', () => {
    for (const etaTOverI of [1 / 1.5, 1.1, 1.5, 2]) {
      const config = { roughness: 0.42, etaTOverI };
      const wo = direction(0.76, 0.2);
      const sample = sampleRoughDielectric(config, N, wo, 1, 0.37, 0.61);
      if (sample.wi == null || roughDielectricDot(N, sample.wi) >= 0) continue;
      const importance = roughDielectricTransmissionEval(config, N, wo, sample.wi, true);
      const radiance = roughDielectricTransmissionEval(config, N, wo, sample.wi, false);
      expect(importance / radiance).toBeCloseTo(etaTOverI ** 2, 10);
      expect(radiance).toBeCloseTo(
        roughDielectricTransmissionEval(config, N, sample.wi, wo, true), 9,
      );
      const forwardPdf = roughDielectricTransmissionPdf(config, N, wo, sample.wi);
      const reversePdf = roughDielectricTransmissionPdf(config, N, sample.wi, wo);
      expect(forwardPdf).toBeGreaterThan(0);
      expect(reversePdf).toBeGreaterThan(0);

      const forwardWm = roughDielectricHalfVector(N, wo, sample.wi, etaTOverI);
      const reverseWm = roughDielectricHalfVector(N, sample.wi, wo, etaTOverI);
      const forwardFromHalfVectorMeasure = roughDielectricVisibleNormalPdf(
        config, N, wo, forwardWm,
      ) * roughDielectricJacobian(N, wo, sample.wi, etaTOverI);
      const reverseFromHalfVectorMeasure = roughDielectricVisibleNormalPdf(
        config, roughDielectricScale(N, -1), sample.wi, reverseWm,
      ) * roughDielectricJacobian(N, sample.wi, wo, etaTOverI);
      expect(forwardPdf).toBeCloseTo(forwardFromHalfVectorMeasure, 11);
      expect(reversePdf).toBeCloseTo(reverseFromHalfVectorMeasure, 11);

      // BDPT stores area densities. Converting each direction to area measure
      // and back must reproduce the solid-angle density on both interface sides.
      const distanceSquared = 2.3 ** 2;
      const forwardDestinationCosine = Math.abs(roughDielectricDot(N, sample.wi));
      const reverseDestinationCosine = Math.abs(roughDielectricDot(N, wo));
      const forwardAreaPdf = forwardPdf * forwardDestinationCosine / distanceSquared;
      const reverseAreaPdf = reversePdf * reverseDestinationCosine / distanceSquared;
      expect(
        forwardAreaPdf * distanceSquared / forwardDestinationCosine,
      ).toBeCloseTo(forwardPdf, 12);
      expect(
        reverseAreaPdf * distanceSquared / reverseDestinationCosine,
      ).toBeCloseTo(reversePdf, 12);
    }
  });

  it('keeps the exact Smith implementation shared by every production consumer', () => {
    const forwardSmith = roughDielectricSmithG1Wgsl('smithG1');
    const replaySmith = roughDielectricSmithG1Wgsl('rptHybrid_smithG1');
    expect(PT_WEBGPU_PATH_TRACE_MATERIAL_FUNCS_WGSL).toContain(forwardSmith);
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain(forwardSmith);
    expect(RESTIR_PT_HYBRID_SHIFT_WGSL).toContain(replaySmith);
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain('smithG1(nDotO, roughness)');
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain('let oDotM = dot(wo, wm);');
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain('oDotM <= 1e-6');
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).not.toContain('let oDotM = abs(dot(wo, wm));');

    for (const source of [
      PT_WEBGPU_PATH_TRACE_MATERIAL_FUNCS_WGSL,
      PT_WEBGPU_ADJOINT_PASS_WGSL,
      RESTIR_PT_HYBRID_SHIFT_WGSL,
    ]) {
      expect(source).not.toContain('(roughness + 1.0) * (roughness + 1.0) * 0.125');
    }
  });

  it('stays finite at normal, grazing, TIR, and the delta boundary', () => {
    for (const entry of [
      { roughness: 1, etaTOverI: 1.5, cosO: 1 },
      { roughness: 0.2, etaTOverI: 1.5, cosO: 1e-4 },
      { roughness: 0.5, etaTOverI: 1 / 1.5, cosO: 0.2 },
      { roughness: ROUGH_DIELECTRIC_SMOOTH_THRESHOLD + 1e-7, etaTOverI: 2, cosO: 0.999 },
    ]) {
      const config = { roughness: entry.roughness, etaTOverI: entry.etaTOverI };
      const wo = direction(entry.cosO);
      for (let i = 1; i <= 2048; i += 1) {
        const sample = sampleRoughDielectric(config, N, wo, 1, radicalInverse(i, 2), radicalInverse(i, 3));
        expect(Number.isFinite(sample.fresnel)).toBe(true);
        if (sample.wi == null || roughDielectricDot(N, sample.wi) >= -1e-8) continue;
        expect(Number.isFinite(roughDielectricTransmissionPdf(config, N, wo, sample.wi))).toBe(true);
        expect(Number.isFinite(roughDielectricTransmissionEval(config, N, wo, sample.wi, false))).toBe(true);
      }
    }
    const smooth = { roughness: ROUGH_DIELECTRIC_SMOOTH_THRESHOLD, etaTOverI: 1.5 };
    expect(roughDielectricTransmissionPdf(smooth, N, N, [0, 0, -1])).toBe(0);
    expect(roughDielectricFresnel(0.2, 1 / 1.5)).toBe(1);
  });
});
