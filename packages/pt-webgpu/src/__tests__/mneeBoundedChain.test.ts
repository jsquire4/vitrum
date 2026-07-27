import { describe, expect, it } from 'vitest';
import {
  MNEE_CHAIN_MAX_VERTICES,
  MNEE_CHAIN_WGSL,
} from '../wgsl/pathTrace/mneeNewton.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL } from '../wgsl/pathTrace/caustic.wgsl.js';

type Vec2 = readonly [number, number];
type Mat2 = readonly [Vec2, Vec2];

function add(a: Vec2, b: Vec2): Vec2 {
  return [a[0] + b[0], a[1] + b[1]];
}

function sub(a: Vec2, b: Vec2): Vec2 {
  return [a[0] - b[0], a[1] - b[1]];
}

function mulMatVec(a: Mat2, x: Vec2): Vec2 {
  return [a[0][0] * x[0] + a[0][1] * x[1], a[1][0] * x[0] + a[1][1] * x[1]];
}

function mulMat(a: Mat2, b: Mat2): Mat2 {
  return [
    [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
    [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
  ];
}

function subMat(a: Mat2, b: Mat2): Mat2 {
  return [
    [a[0][0] - b[0][0], a[0][1] - b[0][1]],
    [a[1][0] - b[1][0], a[1][1] - b[1][1]],
  ];
}

function inverse(a: Mat2): Mat2 {
  const det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
  if (Math.abs(det) < 1e-12) throw new Error('singular test block');
  return [
    [a[1][1] / det, -a[0][1] / det],
    [-a[1][0] / det, a[0][0] / det],
  ];
}

function solveBlockTridiagonal(
  lower: readonly Mat2[],
  diagonal: readonly Mat2[],
  upper: readonly Mat2[],
  rhs: readonly Vec2[],
): Vec2[] {
  const count = diagonal.length;
  const cPrime: Mat2[] = new Array(count);
  const dPrime: Vec2[] = new Array(count);
  for (let i = 0; i < count; i += 1) {
    let denom = diagonal[i]!;
    let reducedRhs = rhs[i]!;
    if (i > 0) {
      denom = subMat(denom, mulMat(lower[i]!, cPrime[i - 1]!));
      reducedRhs = sub(reducedRhs, mulMatVec(lower[i]!, dPrime[i - 1]!));
    }
    const inv = inverse(denom);
    cPrime[i] = i + 1 < count ? mulMat(inv, upper[i]!) : [[0, 0], [0, 0]];
    dPrime[i] = mulMatVec(inv, reducedRhs);
  }
  const result: Vec2[] = new Array(count);
  for (let i = count - 1; i >= 0; i -= 1) {
    result[i] = i + 1 < count
      ? sub(dPrime[i]!, mulMatVec(cPrime[i]!, result[i + 1]!))
      : dPrime[i]!;
  }
  return result;
}

describe('bounded 3–8 vertex MNEE chain', () => {
  it('uses the public eight-vertex capacity and an O(N) coupled Newton solve', () => {
    expect(MNEE_CHAIN_MAX_VERTICES).toBe(8);
    expect(MNEE_CHAIN_WGSL).toContain('struct MneeBoundedChainGeometry');
    expect(MNEE_CHAIN_WGSL).toContain('fn mneeBoundedChainResidualAt(');
    expect(MNEE_CHAIN_WGSL).toContain('fn mneeNewtonSolveChainBounded(');
    expect(MNEE_CHAIN_WGSL).toContain('denom = denom - lower[fi] * cPrime[fi - 1u];');
    expect(MNEE_CHAIN_WGSL).toContain('delta[reverseIndex] = delta[reverseIndex] -');
    expect(MNEE_CHAIN_WGSL).toContain('fn mneeBoundedChainFocusingDet(');
  });

  it('block-Thomas recurrence recovers known coupled systems for every supported long length', () => {
    for (let count = 3; count <= MNEE_CHAIN_MAX_VERTICES; count += 1) {
      const zero: Mat2 = [[0, 0], [0, 0]];
      const lower: Mat2[] = [];
      const diagonal: Mat2[] = [];
      const upper: Mat2[] = [];
      const expected: Vec2[] = [];
      for (let i = 0; i < count; i += 1) {
        lower.push(i === 0 ? zero : [[-0.18, 0.03], [0.02, -0.14]]);
        diagonal.push([[4.2 + i * 0.11, 0.16], [-0.09, 3.8 + i * 0.07]]);
        upper.push(i + 1 === count ? zero : [[-0.12, -0.02], [0.04, -0.17]]);
        expected.push([Math.sin(i + 0.25), Math.cos(i * 0.7 + 0.1)]);
      }
      const rhs = expected.map((value, i) => {
        let row = mulMatVec(diagonal[i]!, value);
        if (i > 0) row = add(row, mulMatVec(lower[i]!, expected[i - 1]!));
        if (i + 1 < count) row = add(row, mulMatVec(upper[i]!, expected[i + 1]!));
        return row;
      });
      const solved = solveBlockTridiagonal(lower, diagonal, upper, rhs);
      for (let i = 0; i < count; i += 1) {
        expect(solved[i]![0]).toBeCloseTo(expected[i]![0], 11);
        expect(solved[i]![1]).toBeCloseTo(expected[i]![1], 11);
      }
    }
  });

  it('samples complete unified length/event/facet PMFs and revalidates solved transport', () => {
    const code = PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL;
    expect(code).toContain('fn boundedManifoldCaustic(');
    expect(code).toContain('let maximumLength = min(params.mneeMaxChainLength, 8u);');
    expect(code).toContain('let chainLength = 1u + min(');
    expect(code).toContain('let lengthSelectionPdf = 1.0 / f32(maximumLength);');
    expect(code).toContain('let facet = mneeProposeConditionalFacet(');
    expect(code).toContain('let conditionalPdf = facets[pi].pdf * 0.5;');
    expect(code).toContain('logFacetEventPdf = logFacetEventPdf + log(conditionalPdf);');
    expect(code).toContain('logInterfaceNumerator - vec3f(logFacetEventPdf)');
    expect(code).toContain('let solved = mneeNewtonSolveChainBounded(');
    expect(code).toContain('pathMeasure = mneeBoundedChainFocusingDet(');
    expect(code).toContain('let optics = mneeFacetOpticsAt(');
    expect(code).toContain('let segment = mneeChainAttenuateSegment(solvedStack, segmentDistance);');
    expect(code).toContain('abs(media.etaT[pi] - boundary.etaT) > 1e-3');
    expect(code).toContain('return boundedManifoldCaustic(');
    expect(code).not.toContain('fn pointLightBoundedChainCaustic(');
  });

  it('evaluates glossy and metallic receiver BRDFs without the retired D9 rejection', () => {
    const start = PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL.indexOf(
      'fn boundedManifoldCaustic(',
    );
    const end = PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL.indexOf(
      'fn manifoldNeeContribution(',
      start,
    );
    const unified = PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL.slice(start, end);
    expect(unified).toContain('let fr = evaluateBrdfFullWithClearcoatNormal(');
    expect(unified).not.toContain('causticReceiverRejected');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).not.toContain(
      'fn causticReceiverRejected(',
    );
  });
});
