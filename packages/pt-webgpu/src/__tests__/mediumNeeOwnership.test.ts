import { describe, expect, it } from 'vitest';
import { PT_WEBGPU_PATH_TRACE_KERNEL_WGSL } from '../wgsl/pathTrace/kernel.wgsl.js';
import { PT_WEBGPU_MEDIUM_NEE_WGSL } from '../wgsl/pathTrace/mediumNee.wgsl.js';

function powerHeuristic(pdfA: number, pdfB: number): number {
  if (
    !(pdfA >= 0) || !(pdfB >= 0) ||
    pdfA > 3.402823466e38 || pdfB > 3.402823466e38
  ) return 0;
  const scale = Math.max(pdfA, pdfB);
  if (!(scale > 0)) return 0;
  const a = pdfA / scale;
  const b = pdfB / scale;
  return (a * a) / (a * a + b * b);
}

function localMediumNeeWeight(options: {
  readonly lightPdf: number;
  readonly phasePdf: number;
  readonly delta: boolean;
  readonly softDirectional: boolean;
  readonly bdptCrossFamilyEnabled: boolean;
}): number | 'bdpt-family' {
  if (options.bdptCrossFamilyEnabled) return 'bdpt-family';
  if (options.delta || options.softDirectional) return 1;
  return powerHeuristic(options.lightPdf, options.phasePdf);
}

function straightBoundaryIsNull(incidentIor: number, transmittedIor: number): boolean {
  const scale = Math.max(Math.abs(incidentIor), Math.abs(transmittedIor), 1);
  return Math.abs(incidentIor - transmittedIor) <= 1e-4 * scale;
}

function straightBoundaryTransitionAllowed(
  stackIors: readonly number[],
  materialIor: number,
  frontFace: boolean,
): boolean {
  const incidentIor = stackIors.at(-1) ?? 1;
  const transmittedIor = frontFace ? materialIor : (stackIors.at(-2) ?? 1);
  return straightBoundaryIsNull(incidentIor, transmittedIor);
}

interface MediumLayerOracle {
  readonly ior: number;
  readonly sigmaT: number;
  readonly remainingDistance: number;
}

function traceNestedExitOracle(
  sourceLayers: readonly MediumLayerOracle[],
  distanceToInnerExit: number,
  distanceAfterInnerExit: number,
): { readonly visible: boolean; readonly transmittance: number } {
  const stack = sourceLayers.map((layer) => ({ ...layer }));
  if (stack.length === 0) return { visible: false, transmittance: 1 };
  let transmittance = 1;
  const inner = stack.at(-1)!;
  transmittance *= Math.exp(-inner.sigmaT * Math.min(distanceToInnerExit, inner.remainingDistance));
  const transmittedIor = stack.at(-2)?.ior ?? 1;
  if (!straightBoundaryIsNull(inner.ior, transmittedIor)) {
    return { visible: false, transmittance };
  }
  stack.pop();
  if (stack.length > 0) {
    const enclosing = stack.at(-1)!;
    transmittance *= Math.exp(
      -enclosing.sigmaT * Math.min(distanceAfterInnerExit, enclosing.remainingDistance),
    );
  }
  return { visible: true, transmittance };
}

describe('pt-webgpu in-medium estimator ownership', () => {
  it('assigns an unpaired soft directional cone wholly to local NEE', () => {
    const lightPdf = 0.2;
    const phasePdf = 0.6;
    expect(
      localMediumNeeWeight({
        lightPdf,
        phasePdf,
        delta: false,
        softDirectional: true,
        bdptCrossFamilyEnabled: false,
      }),
    ).toBe(1);
    expect(powerHeuristic(lightPdf, phasePdf)).toBeCloseTo(0.1, 12);
  });

  it('keeps power-MIS partition of unity where both local estimators exist', () => {
    const lightPdf = 0.2;
    const phasePdf = 0.6;
    const neeWeight = localMediumNeeWeight({
      lightPdf,
      phasePdf,
      delta: false,
      softDirectional: false,
      bdptCrossFamilyEnabled: false,
    });
    expect(neeWeight).toBeTypeOf('number');
    const phaseWeight = powerHeuristic(phasePdf, lightPdf);
    expect((neeWeight as number) + phaseWeight).toBeCloseTo(1, 12);
  });

  it('preserves delta ownership and delegates enabled cross-family MIS to BDPT', () => {
    expect(
      localMediumNeeWeight({
        lightPdf: 0.2,
        phasePdf: 0.6,
        delta: true,
        softDirectional: false,
        bdptCrossFamilyEnabled: false,
      }),
    ).toBe(1);
    expect(
      localMediumNeeWeight({
        lightPdf: 0.2,
        phasePdf: 0.6,
        delta: false,
        softDirectional: true,
        bdptCrossFamilyEnabled: true,
      }),
    ).toBe('bdpt-family');
  });

  it('pins the endpoint-family partition in the production shader', () => {
    const phaseConnection = PT_WEBGPU_MEDIUM_NEE_WGSL.slice(
      PT_WEBGPU_MEDIUM_NEE_WGSL.indexOf('fn mediumPhaseEmitterConnection('),
    );
    expect(phaseConnection).toContain('intersectRectAreaLightRay(');
    expect(phaseConnection).toContain('intersectMeshAreaLightRay(');
    expect(phaseConnection).toContain('environmentPdf(sampledDirection)');
    expect(phaseConnection).not.toContain('directionalLights[');
    expect(phaseConnection).not.toContain('sampleDirectionalCone(');

    expect(PT_WEBGPU_MEDIUM_NEE_WGSL).toContain('let softDirectionalWithoutComplement =');
    expect(PT_WEBGPU_MEDIUM_NEE_WGSL).toContain(
      'flat < params.directionalLightCount && !light.delta &&',
    );
    expect(PT_WEBGPU_MEDIUM_NEE_WGSL).toContain('light.delta || softDirectionalWithoutComplement');
    expect(PT_WEBGPU_MEDIUM_NEE_WGSL).toContain('if (bdptCrossFamilyEnabled) {');
    expect(PT_WEBGPU_MEDIUM_NEE_WGSL).toContain('mis = bdptInfiniteEyeFamilyWeight(');
  });
});

describe('pt-webgpu straight medium visibility boundary partition', () => {
  it('allows optical-identity boundaries but rejects unsupported IOR transitions', () => {
    expect(straightBoundaryTransitionAllowed([1], 1, true)).toBe(true);
    expect(straightBoundaryTransitionAllowed([1.5, 1.5], 1.5, false)).toBe(true);
    expect(straightBoundaryTransitionAllowed([1], 1.5, true)).toBe(false);
    expect(straightBoundaryTransitionAllowed([1.5], 1.5, false)).toBe(false);
    expect(straightBoundaryTransitionAllowed([1.33, 1.5], 1.5, false)).toBe(false);
  });

  it('pins fail-closed IOR handling before any medium-stack transition', () => {
    expect(PT_WEBGPU_MEDIUM_NEE_WGSL).toContain(
      'fn mediumStraightBoundaryIsNull(incidentIor: f32, transmittedIor: f32) -> bool',
    );
    expect(PT_WEBGPU_MEDIUM_NEE_WGSL).toContain('incidentIor = stack[depth - 1u].ior;');
    expect(PT_WEBGPU_MEDIUM_NEE_WGSL).toContain('transmittedIor = stack[depth - 2u].ior;');

    const rejectIndex = PT_WEBGPU_MEDIUM_NEE_WGSL.indexOf(
      'if (!mediumStraightBoundaryIsNull(incidentIor, transmittedIor)) {',
    );
    const enterIndex = PT_WEBGPU_MEDIUM_NEE_WGSL.indexOf(
      'stack[depth] = bdptMediumLayer(matId, mat, heroLambda, boundary);',
    );
    expect(rejectIndex).toBeGreaterThan(-1);
    expect(enterIndex).toBeGreaterThan(rejectIndex);
  });

  it('preserves enclosing same-IOR media and attenuates each active segment', () => {
    const result = traceNestedExitOracle(
      [
        { ior: 1.33, sigmaT: 0.2, remainingDistance: 10 },
        { ior: 1.33, sigmaT: 1.1, remainingDistance: 10 },
      ],
      0.4,
      0.6,
    );
    expect(result.visible).toBe(true);
    expect(result.transmittance).toBeCloseTo(Math.exp(-1.1 * 0.4) * Math.exp(-0.2 * 0.6), 14);
    // The retired top-only reconstruction treated the inner exit as 1.33→1.0.
    expect(straightBoundaryTransitionAllowed([1.33], 1.33, false)).toBe(false);
  });

  it('rejects a nested non-identity exit and honors finite top-layer distance', () => {
    const rejected = traceNestedExitOracle(
      [
        { ior: 1.33, sigmaT: 0.2, remainingDistance: 10 },
        { ior: 1.5, sigmaT: 2, remainingDistance: 0.1 },
      ],
      0.4,
      0.6,
    );
    expect(rejected.visible).toBe(false);
    expect(rejected.transmittance).toBeCloseTo(Math.exp(-2 * 0.1), 14);
  });

  it('clones the complete eye stack and threads it through both medium estimators', () => {
    expect(PT_WEBGPU_MEDIUM_NEE_WGSL).toContain('for (var i = 0u; i < sourceDepth; i = i + 1u)');
    expect(PT_WEBGPU_MEDIUM_NEE_WGSL).toContain('stack[sourceDepth - 1u] = sourceLayer;');
    expect(PT_WEBGPU_MEDIUM_NEE_WGSL).toContain('var depth = sourceDepth;');
    expect(PT_WEBGPU_MEDIUM_NEE_WGSL).not.toContain('stack[0u] = sourceLayer;');
    expect(
      PT_WEBGPU_PATH_TRACE_KERNEL_WGSL.match(/&bdptMediumStack, bdptMediumDepth,/g),
    ).toHaveLength(2);
  });
});
