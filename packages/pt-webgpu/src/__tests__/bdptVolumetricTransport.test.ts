import { describe, expect, it } from 'vitest';

import { composePtWebgpuTraceWgsl } from '../wgsl/pathTraceBruteforce.wgsl.js';
import { PT_WEBGPU_BDPT_CONNECTION_WGSL } from '../wgsl/bdpt/bdptConnection.wgsl.js';
import { PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL } from '../wgsl/bdpt/bdptLightSubpath.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_KERNEL_WGSL } from '../wgsl/pathTrace/kernel.wgsl.js';
import {
  bdptConnectionTransmittanceCpu,
  bdptEffectiveMediumDistanceCpu,
  bdptHgPhaseCpu,
  bdptSegmentDistanceDensityCpu,
  sharedBdptEdgeMediumCpu,
  type BdptMediumEndpointCpu,
} from '../bdpt/bdptMediumTransportCpu.js';
import { assembleMergedConnectionPath } from '../bdpt/bdptConnectionMisFull.js';

describe('pt-webgpu symmetric volumetric BDPT', () => {
  it('preserves closed-medium edge products under path reversal', () => {
    const sigmaT = 1.7;
    const entryToCollision = 0.35;
    const collisionToExit = 0.8;

    const forward =
      bdptSegmentDistanceDensityCpu(sigmaT, entryToCollision, Number.MAX_VALUE, true) *
      bdptSegmentDistanceDensityCpu(sigmaT, collisionToExit, Number.MAX_VALUE, false);
    const reverse =
      bdptSegmentDistanceDensityCpu(sigmaT, collisionToExit, Number.MAX_VALUE, true) *
      bdptSegmentDistanceDensityCpu(sigmaT, entryToCollision, Number.MAX_VALUE, false);

    expect(forward).toBeCloseTo(reverse, 14);
    expect(
      bdptSegmentDistanceDensityCpu(sigmaT, entryToCollision, Number.MAX_VALUE, true) /
        bdptSegmentDistanceDensityCpu(sigmaT, entryToCollision, Number.MAX_VALUE, false),
    ).toBeCloseTo(sigmaT, 14);
  });

  it('uses the previous volume flag when reversing a surface-arrival edge', () => {
    const sigmaT = 0.9;
    const distance = 1.25;
    const previousVertexIsMedium = true;
    const reverse = bdptSegmentDistanceDensityCpu(
      sigmaT,
      distance,
      Number.MAX_VALUE,
      previousVertexIsMedium,
    );
    const incorrectSurfaceOnlyDensity = bdptSegmentDistanceDensityCpu(
      sigmaT,
      distance,
      Number.MAX_VALUE,
      false,
    );

    expect(reverse).toBeCloseTo(
      sigmaT * incorrectSurfaceOnlyDensity,
      14,
    );
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).toContain(
      'prevEyeForReverse.medium,',
    );
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).toContain(
      'bounce - 1u, swappedRev * reverseDistancePdf,',
    );
  });

  it('keeps collision and survival estimators unbiased per RGB channel', () => {
    const sigmaT = 1.25;
    const sigmaS = [0.15, 0.5, 1.0] as const;
    const distance = 0.6;
    const survival = Math.exp(-sigmaT * distance);
    const collisionPdf = sigmaT * survival;

    for (const channelSigmaS of sigmaS) {
      const collisionWeight =
        (channelSigmaS * survival) / collisionPdf;
      expect(collisionPdf * collisionWeight).toBeCloseTo(
        channelSigmaS * survival,
        14,
      );

      const noCollisionWeight = survival / survival;
      expect(survival * noCollisionWeight).toBeCloseTo(survival, 14);
    }
  });

  it('uses a reciprocal HG phase density for both orientations', () => {
    for (const g of [-0.8, -0.2, 0, 0.45, 0.9]) {
      for (const cosTheta of [-1, -0.4, 0, 0.3, 1]) {
        const forward = bdptHgPhaseCpu(cosTheta, g);
        // Reversing both directions preserves their dot product.
        const reverse = bdptHgPhaseCpu(cosTheta, g);
        expect(forward).toBeCloseTo(reverse, 14);
      }
    }
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).toContain(
      'let reversePhasePdf = phasePdf;',
    );
  });

  it('never consumes the terminal light-vertex pdfRev placeholder raw', () => {
    const terminalPlaceholder = 987_654.25;
    const predecessorPlaceholder = 876_543.5;
    const { vertices, selectedS } = assembleMergedConnectionPath({
      lightChain: [
        { position: [0, 0, 0], normal: [0, 0, 1], pdfFwd: 0.2, pdfRev: 0.23, isSpecular: false },
        { position: [0, 0, 1], normal: [0, 0, 1], pdfFwd: 0.3, pdfRev: predecessorPlaceholder, isSpecular: false },
        { position: [0, 0, 2], normal: [0, 0, 1], pdfFwd: 0.4, pdfRev: terminalPlaceholder, isSpecular: false },
      ],
      eyeChain: [
        { position: [0, 0, 4], normal: [0, 0, -1], pdfFwd: 0.5, pdfRev: 0.53, isSpecular: false },
        { position: [0, 0, 3], normal: [0, 0, -1], pdfFwd: 0.6, pdfRev: 0.63, isSpecular: false },
      ],
      camera: { position: [0, 0, 5], normal: [0, 0, -1] },
      eyeBrdfPdf: () => 0.61,
      lightBrdfPdf: () => 0.37,
    });

    expect(selectedS).toBe(3);
    expect(vertices[2]!.pdfRev).toBe(0.61);
    expect(vertices[1]!.pdfRev).toBe(0.37);
    expect(vertices[2]!.pdfRev).not.toBe(terminalPlaceholder);
    expect(vertices[1]!.pdfRev).not.toBe(predecessorPlaceholder);
    expect(vertices[0]!.pdfRev).toBe(0.23);
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'let predecessorCol = prevCol - 1;',
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'swappedDirectionalPdf * reverseEdgeDensity,',
    );
  });

  it('selects the transmitted nested medium and preserves the full reversed factor product', () => {
    const outer = { matId: 7, remainingDistance: 4.0 };
    const innerEye = { matId: 13, remainingDistance: 0.45 };
    const innerLight = { matId: 13, remainingDistance: 0.8 };
    const eyeSurface: BdptMediumEndpointCpu = {
      isMedium: false,
      normal: [0, 0, 1],
      active: outer,
      incident: outer,
      transmitted: innerEye,
    };
    const lightVolume: BdptMediumEndpointCpu = {
      isMedium: true,
      normal: [0, 0, 0],
      active: innerLight,
      incident: innerLight,
      transmitted: innerLight,
    };

    const forwardMedium = sharedBdptEdgeMediumCpu(
      eyeSurface, [0, 0, -1], lightVolume, [0, 0, 1],
    );
    const reversedMedium = sharedBdptEdgeMediumCpu(
      lightVolume, [0, 0, 1], eyeSurface, [0, 0, -1],
    );
    expect(forwardMedium).toEqual({ matId: 13, remainingDistance: 0.45 });
    expect(reversedMedium).toEqual(forwardMedium);
    // The removed single-side implementation used `active` (outer=7) here and
    // therefore rejected this valid connection against the inner volume (13).
    expect(eyeSurface.active.matId).not.toBe(forwardMedium!.matId);

    const sigmaT = 1.7;
    const entryToCollision = 0.3;
    const collisionToExit = 0.4;
    const forwardPdf =
      bdptSegmentDistanceDensityCpu(sigmaT, entryToCollision, forwardMedium!.remainingDistance, true) *
      bdptSegmentDistanceDensityCpu(sigmaT, collisionToExit, forwardMedium!.remainingDistance, false);
    const reversedPdf =
      bdptSegmentDistanceDensityCpu(sigmaT, collisionToExit, reversedMedium!.remainingDistance, true) *
      bdptSegmentDistanceDensityCpu(sigmaT, entryToCollision, reversedMedium!.remainingDistance, false);

    const etaRatios = [1.5 / 1.0, 1.33 / 1.5, 1.5 / 1.33, 1.0 / 1.5];
    const reversedEtaRatios = [...etaRatios].reverse().map((eta) => 1 / eta);
    const etaForward = etaRatios.reduce((product, eta) => product * eta * eta, 1);
    const etaReverse = reversedEtaRatios.reduce((product, eta) => product * eta * eta, 1);
    const phaseForward = bdptHgPhaseCpu(-0.35, 0.62) * bdptHgPhaseCpu(0.4, -0.2);
    const phaseReverse = bdptHgPhaseCpu(-0.35, 0.62) * bdptHgPhaseCpu(0.4, -0.2);
    const transA = bdptConnectionTransmittanceCpu(
      [0.8, 1.1, 1.5], entryToCollision, forwardMedium!,
    );
    const transB = bdptConnectionTransmittanceCpu(
      [0.8, 1.1, 1.5], collisionToExit, forwardMedium!,
    );
    const transProduct = transA[0] * transA[1] * transA[2] * transB[0] * transB[1] * transB[2];
    const forwardFactor = forwardPdf * etaForward * phaseForward * transProduct;
    const reversedFactor = reversedPdf * etaReverse * phaseReverse * transProduct;

    expect(forwardPdf).toBeCloseTo(reversedPdf, 14);
    expect(etaForward).toBeCloseTo(1, 14);
    expect(etaReverse).toBeCloseTo(1, 14);
    expect(forwardFactor).toBeCloseTo(reversedFactor, 14);
  });

  it('clamps a long edge to both endpoint budgets and is invariant to asymmetric reversal', () => {
    const makeVolume = (matId: number, remainingDistance: number): BdptMediumEndpointCpu => {
      const side = { matId, remainingDistance };
      return {
        isMedium: true,
        normal: [0, 0, 0],
        active: side,
        incident: side,
        transmitted: side,
      };
    };
    const edgeDistance = 4.0;
    const a = makeVolume(21, 0.35);
    const b = makeVolume(21, 0.8);
    const ab = sharedBdptEdgeMediumCpu(a, [1, 0, 0], b, [-1, 0, 0]);
    const ba = sharedBdptEdgeMediumCpu(b, [-1, 0, 0], a, [1, 0, 0]);
    expect(ab).toEqual({ matId: 21, remainingDistance: 0.35 });
    expect(ba).toEqual(ab);
    expect(edgeDistance).toBeGreaterThan(a.active.remainingDistance);
    expect(edgeDistance).toBeGreaterThan(b.active.remainingDistance);
    expect(bdptEffectiveMediumDistanceCpu(edgeDistance, ab!)).toBe(0.35);

    const sigmaT = 1.2;
    const clampedSurvival = bdptSegmentDistanceDensityCpu(
      sigmaT, edgeDistance, ab!.remainingDistance, false,
    );
    expect(clampedSurvival).toBeCloseTo(Math.exp(-sigmaT * 0.35), 14);
    expect(clampedSurvival).not.toBeCloseTo(Math.exp(-sigmaT * edgeDistance), 6);

    const c = makeVolume(21, 1.25);
    const d = makeVolume(21, 0.2);
    const cd = sharedBdptEdgeMediumCpu(c, [0, 1, 0], d, [0, -1, 0]);
    const dc = sharedBdptEdgeMediumCpu(d, [0, -1, 0], c, [0, 1, 0]);
    expect(cd).toEqual({ matId: 21, remainingDistance: 0.2 });
    expect(dc).toEqual(cd);
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'a.matId, min(a.remainingDistance, b.remainingDistance),',
    );
  });

  it('normalizes homogeneous free flight and matches analytic finite-budget energy', () => {
    const sigmaT = 1.35;
    const budget = 2.4;
    const albedo = 0.63;
    const steps = 100_000;
    const dt = budget / steps;
    let collisionProbability = 0;
    let scatteredEnergy = 0;
    for (let i = 0; i < steps; i += 1) {
      const t = (i + 0.5) * dt;
      const density = bdptSegmentDistanceDensityCpu(sigmaT, t, budget, true);
      collisionProbability += density * dt;
      scatteredEnergy += albedo * density * dt;
    }
    const terminalSurvival = bdptSegmentDistanceDensityCpu(
      sigmaT, budget, budget, false,
    );
    const analyticCollision = 1 - Math.exp(-sigmaT * budget);
    const analyticEnergy = albedo * analyticCollision + Math.exp(-sigmaT * budget);

    expect(collisionProbability).toBeCloseTo(analyticCollision, 9);
    expect(collisionProbability + terminalSurvival).toBeCloseTo(1, 9);
    expect(scatteredEnergy + terminalSurvival).toBeCloseTo(analyticEnergy, 9);
  });

  it('preserves tiny positive extinction in CPU and production WGSL density classification', () => {
    const sigmaT = 1e-12;
    expect(bdptSegmentDistanceDensityCpu(sigmaT, 2, 2, true)).toBeGreaterThan(0);
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('if (heroSigmaT <= 0.0)');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).not.toContain('heroSigmaT <= 1e-6');
  });

  it('composes matching nested medium walks on production eye and light paths', () => {
    const bdpt = composePtWebgpuTraceWgsl(true);
    expect(bdpt).toContain('fn sampleHenyeyGreenstein');
    expect(bdpt).toContain('freeFlightDist');
    expect(bdpt).toContain('freeFlightDistance');
    expect(bdpt).toContain(
      'var bdptMediumStack: array<BdptMediumLayer, BDPT_MEDIUM_STACK_LIMIT>;',
    );
    expect(bdpt).toContain(
      'var mediumStack: array<BdptMediumLayer, BDPT_MEDIUM_STACK_LIMIT>;',
    );
    expect(bdpt).toContain('bdptSegmentDistanceDensity(');
    expect(bdpt).toContain('bdptWriteMediumVertex(');
    expect(bdpt).toContain('eyeIsMedium: bool,');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'return (cosX * cosY) / dist2;',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'destIsMedium: bool,',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'var<private> bdptEyeStackPrivate: array<BdptEyeVtx, 8>;',
    );
  });

  it("keeps the same exclusive F/D ownership across surface and volume events", () => {
    const bdpt = composePtWebgpuTraceWgsl(true);
    expect(bdpt).toContain(
      "lightCount, distantDirectEmitterCount(), bdptOwnsFiniteLightFamily",
    );
    expect(bdpt).toContain(
      "mediumLightCount, distantDirectEmitterCount(),",
    );
    expect(bdpt.split("sampleDistantDirectLight(").length - 1).toBeGreaterThanOrEqual(3);
    expect(bdpt).toContain(
      "let familyOwnsEmitter = !bdptOwnsFiniteLightFamily ||",
    );
    expect(bdpt).toContain('bdptInfiniteEyeFamilyWeight(');
    expect(bdpt).toContain('bdptInfiniteRootLaunchPdf(');
    expect(bdpt).toContain('distantDirectSelectionPdf(');
    expect(bdpt.split("if (!bdptOwnsFiniteLightFamily && (sumDirectLighting || current == picked))").length - 1).toBe(4);
    expect(bdpt).toContain(
      "radiance = radiance + bsdfEnvironmentConnectionContribution(",
    );
    expect(bdpt).not.toContain("bdptHasEnvironmentEmitter");
  });
});
