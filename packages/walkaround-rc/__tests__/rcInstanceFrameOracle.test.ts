import { describe, expect, it } from 'vitest';
import { PROBE_RAY_CAST_WGSL } from '../src/index.js';

type V3 = readonly [number, number, number];
type M3 = readonly [V3, V3, V3];

const dot = (a: V3, b: V3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: V3, b: V3): V3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const normalize = (v: V3): V3 => {
  const inv = 1 / Math.hypot(v[0], v[1], v[2]);
  return [v[0] * inv, v[1] * inv, v[2] * inv];
};
const transformDirection = (m: M3, v: V3): V3 => [
  m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
  m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
  m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
];
const determinant = (m: M3): number => dot(m[0], cross(m[1], m[2]));
const transformNormalInverseTranspose = (m: M3, n: V3): V3 => {
  const invDet = 1 / determinant(m);
  const c0 = cross(m[1], m[2]);
  const c1 = cross(m[2], m[0]);
  const c2 = cross(m[0], m[1]);
  return normalize([
    (c0[0] * n[0] + c1[0] * n[1] + c2[0] * n[2]) * invDet,
    (c0[1] * n[0] + c1[1] * n[1] + c2[1] * n[2]) * invDet,
    (c0[2] * n[0] + c1[2] * n[1] + c2[2] * n[2]) * invDet,
  ]);
};

function functionBody(source: string, name: string): string {
  const start = source.indexOf(`fn ${name}(`);
  expect(start).toBeGreaterThanOrEqual(0);
  const brace = source.indexOf('{', start);
  let depth = 0;
  for (let i = brace; i < source.length; i += 1) {
    if (source[i] === '{') depth += 1;
    if (source[i] === '}' && --depth === 0) return source.slice(brace + 1, i);
  }
  throw new Error(`unterminated WGSL function ${name}`);
}

describe('RC TLAS normal/tangent frame oracle', () => {
  // Rotation-Z(90°) × reflected non-uniform scale(-2, 3, 0.5), column-major.
  const localToWorld: M3 = [
    [0, -2, 0],
    [-3, 0, 0],
    [0, 0, 0.5],
  ];

  it('inverse-transpose normal remains perpendicular where direct L2W multiplication fails', () => {
    const localNormal = normalize([1, 1, 1]);
    const localTangent = normalize([1, -1, 0]);
    expect(Math.abs(dot(localNormal, localTangent))).toBeLessThan(1e-12);

    const worldTangent = normalize(transformDirection(localToWorld, localTangent));
    const correctNormal = transformNormalInverseTranspose(localToWorld, localNormal);
    const naiveNormal = normalize(transformDirection(localToWorld, localNormal));
    expect(Math.abs(dot(correctNormal, worldTangent))).toBeLessThan(1e-12);
    expect(Math.abs(dot(naiveNormal, worldTangent))).toBeGreaterThan(0.1);
  });

  it('transformed UV derivatives preserve reflected-instance determinant handedness', () => {
    const tangent = normalize(transformDirection(localToWorld, [1, 0, 0]));
    const bitangent = normalize(transformDirection(localToWorld, [0, 1, 0]));
    const normal = transformNormalInverseTranspose(localToWorld, [0, 0, 1]);
    expect(Math.sign(dot(cross(tangent, bitangent), normal)))
      .toBe(Math.sign(determinant(localToWorld)));
    expect(determinant(localToWorld)).toBeLessThan(0);
  });

  it('assembles the CPU-oracle transforms and finite normalization guards into production WGSL', () => {
    const smooth = functionBody(PROBE_RAY_CAST_WGSL, 'rcSmoothNormalForHit');
    const tangentFrame = functionBody(PROBE_RAY_CAST_WGSL, 'rcMaterialTangentFrameForHit');
    const normalMap = functionBody(PROBE_RAY_CAST_WGSL, 'rcApplyNormalMapAtOffsetForHit');
    expect(smooth).toContain('tlasTransformNormalFromLocalCols(');
    expect(smooth).toContain('tlasLoadWorldToLocalColumn(base)');
    expect(tangentFrame).toContain(
      'let positionScale = max(rcMaxAbsVec3(dp1), rcMaxAbsVec3(dp2));',
    );
    expect(tangentFrame).toContain('dp1 = dp1 / positionScale;');
    expect(tangentFrame).toContain('dp2 = dp2 / positionScale;');
    expect(tangentFrame).toContain(
      'tangent = rcTransformDirectionCols(l2w0, l2w1, l2w2, tangent);',
    );
    expect(tangentFrame).toContain(
      'bitangent = rcTransformDirectionCols(l2w0, l2w1, l2w2, bitangent);',
    );
    expect(tangentFrame).toContain('rcTangentHandednessForLocalToWorld(l2w0, l2w1, l2w2)');
    expect(normalMap).toContain('if (!rcCanNormalize(tangentSampleRaw))');
    expect(normalMap).toContain('rcSafeNormalizeOr(\n    tangentSampleRaw,');
    expect(normalMap).toContain('rcSafeNormalizeOr(perturbedRaw, fallbackNormal)');
  });
});
