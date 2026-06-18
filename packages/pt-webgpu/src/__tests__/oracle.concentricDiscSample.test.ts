/**
 * H55 proof oracle — Shirley-Chiu concentric-disc mapping.
 *
 * The WGSL contract tests already pin that the full/lite kernels call the
 * shared helper. This file is the independent behavior oracle: it checks the
 * signed-quadrant geometry and radial area law of the mapping, then verifies
 * the production helpers keep the sign-preserving divisions the oracle needs.
 */
import { describe, expect, it } from 'vitest';
import { PT_WEBGPU_PATH_TRACE_KERNEL_CORE_WGSL } from '../wgsl/pathTrace/kernelCore.wgsl.js';
import { PT_WEBGPU_ADJOINT_PASS_WGSL } from '../wgsl/pathTrace/adjointPass.wgsl.js';

type Vec2 = readonly [number, number];

function concentricDiscSampleReference([a, b]: Vec2): Vec2 {
  if (a === 0 && b === 0) {
    return [0, 0];
  }
  let r: number;
  let phi: number;
  if (Math.abs(a) >= Math.abs(b)) {
    r = a;
    phi = (Math.PI / 4) * (b / a);
  } else {
    r = b;
    phi = (Math.PI / 2) - (Math.PI / 4) * (a / b);
  }
  return [r * Math.cos(phi), r * Math.sin(phi)];
}

function expectVecClose(actual: Vec2, expected: Vec2, precision = 12): void {
  expect(actual[0]).toBeCloseTo(expected[0], precision);
  expect(actual[1]).toBeCloseTo(expected[1], precision);
}

function extractFunctionBody(src: string, name: string): string {
  const start = src.indexOf(`fn ${name}`);
  expect(start, `${name} exists`).toBeGreaterThanOrEqual(0);
  const open = src.indexOf('{', start);
  expect(open, `${name} body opens`).toBeGreaterThanOrEqual(0);
  let depth = 0;
  for (let i = open; i < src.length; i += 1) {
    const ch = src[i];
    if (ch === '{') depth += 1;
    if (ch === '}') {
      depth -= 1;
      if (depth === 0) {
        return src.slice(open + 1, i);
      }
    }
  }
  throw new Error(`${name} body did not close`);
}

describe('concentricDiscSample — independent Shirley-Chiu oracle', () => {
  it('maps hand-derived quadrant anchors with signed denominators', () => {
    const s = Math.SQRT1_2;
    const cases: ReadonlyArray<{ readonly xi: Vec2; readonly expected: Vec2 }> = [
      { xi: [0, 0], expected: [0, 0] },
      { xi: [1, 0], expected: [1, 0] },
      { xi: [-1, 0], expected: [-1, 0] },
      { xi: [0, 1], expected: [0, 1] },
      { xi: [0, -1], expected: [0, -1] },
      { xi: [1, 1], expected: [s, s] },
      { xi: [1, -1], expected: [s, -s] },
      { xi: [-1, 1], expected: [-s, s] },
      { xi: [-1, -1], expected: [-s, -s] },
    ];

    for (const { xi, expected } of cases) {
      expectVecClose(concentricDiscSampleReference(xi), expected);
    }
  });

  it('keeps every stratum in the unit disc and preserves the radial area law', () => {
    const n = 129;
    let sumX = 0;
    let sumY = 0;
    let sumR2 = 0;
    let maxR2 = 0;

    for (let y = 0; y < n; y += 1) {
      for (let x = 0; x < n; x += 1) {
        const a = ((x + 0.5) / n) * 2 - 1;
        const b = ((y + 0.5) / n) * 2 - 1;
        const [dx, dy] = concentricDiscSampleReference([a, b]);
        const r2 = dx * dx + dy * dy;
        maxR2 = Math.max(maxR2, r2);
        sumR2 += r2;
        sumX += dx;
        sumY += dy;
      }
    }

    const count = n * n;
    expect(maxR2).toBeLessThanOrEqual(1 + 1e-12);
    expect(Math.abs(sumX / count)).toBeLessThan(1e-15);
    expect(Math.abs(sumY / count)).toBeLessThan(1e-15);
    expect(sumR2 / count).toBeCloseTo(0.5, 3);
  });

  it('keeps production and adjoint WGSL on the sign-preserving Shirley-Chiu form', () => {
    const production = extractFunctionBody(PT_WEBGPU_PATH_TRACE_KERNEL_CORE_WGSL, 'concentricDiscSample');
    const adjoint = extractFunctionBody(PT_WEBGPU_ADJOINT_PASS_WGSL, 'adjointConcentricDiscSample');

    for (const body of [production, adjoint]) {
      expect(body).toContain('if (a == 0.0 && b == 0.0)');
      expect(body).toContain('(b / a)');
      expect(body).toContain('(a / b)');
      expect(body).not.toContain('max(abs(a)');
      expect(body).not.toContain('max(abs(b)');
    }
  });
});
