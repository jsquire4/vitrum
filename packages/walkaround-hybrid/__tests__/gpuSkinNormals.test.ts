/**
 * Theme 2 — GPU normal-skinning WGSL structural pins.
 *
 * Runtime GPU execution is deferred (no device in CI — see report doc entry),
 * so these tests pin the structural properties the with-normals LBS kernel
 * must have:
 *   - it binds a skinned-normal output buffer (binding 7),
 *   - it reads the rest normals (binding 2) — closing the "binds restNormals
 *     but never skins normals" gap the old kernel had,
 *   - it transforms the normal via an inverse-transpose of the blended skin
 *     linear part (not the plain matrix), and composes the world matrix's
 *     inverse-transpose when world-applying,
 *   - the position math is unchanged from the position-only kernel.
 */

import { describe, expect, it } from 'vitest';

import { mat3InverseTranspose } from '@vitrum/core';
import {
  GPU_SKIN_BVH_WITH_NORMALS_WGSL,
} from '../src/skin/gpuSkinBvh.wgsl.js';

type Vec3 = readonly [number, number, number];

function cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/** Faithful scalar mirror of the WGSL function, returned as matrix columns. */
function gpuInverseTransposeMirror(c0: Vec3, c1: Vec3, c2: Vec3): readonly Vec3[] {
  const matrixScale = Math.max(
    ...c0.map(Math.abs),
    ...c1.map(Math.abs),
    ...c2.map(Math.abs),
  );
  if (matrixScale <= 0) return [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  const scale = (v: Vec3, factor: number): Vec3 => [
    v[0] * factor,
    v[1] * factor,
    v[2] * factor,
  ];
  const n0 = scale(c0, 1 / matrixScale);
  const n1 = scale(c1, 1 / matrixScale);
  const n2 = scale(c2, 1 / matrixScale);
  const cofactors = [cross(n1, n2), cross(n2, n0), cross(n0, n1)] as const;
  const determinant = dot(n0, cofactors[0]);
  if (Math.abs(determinant) < 1e-20) {
    const magnitude = Math.hypot(...cofactors.flat());
    return magnitude < 1e-20
      ? [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
      : cofactors;
  }
  return cofactors.map((column) => scale(column, 1 / determinant / matrixScale));
}

function transformColumns(columns: readonly Vec3[], vector: Vec3): Vec3 {
  return [
    columns[0]![0] * vector[0] + columns[1]![0] * vector[1] + columns[2]![0] * vector[2],
    columns[0]![1] * vector[0] + columns[1]![1] * vector[1] + columns[2]![1] * vector[2],
    columns[0]![2] * vector[0] + columns[1]![2] * vector[1] + columns[2]![2] * vector[2],
  ];
}

function normalize(vector: Vec3): Vec3 {
  const magnitude = Math.hypot(...vector);
  return [
    vector[0] / magnitude,
    vector[1] / magnitude,
    vector[2] / magnitude,
  ];
}

describe('GPU normal-skinning WGSL', () => {
  it('with-normals kernel binds a skinned-normal output at binding 7', () => {
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).toContain(
      '@group(0) @binding(7) var<storage, read_write> skinnedNormals: array<vec4f>',
    );
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).not.toContain('skinnedNormals[vi]');
  });

  it('with-normals kernel actually reads the rest normals (binding 2)', () => {
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).toContain('restNormals[vi]');
  });

  it('with-normals kernel transforms the normal by an inverse-transpose', () => {
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).toContain('fn mat3InverseTranspose');
    // Normal is produced by applying the inverse-transpose matrix to the rest
    // normal, not the plain blended matrix.
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).toContain('mat3InverseTranspose(col0, col1, col2)');
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).toMatch(/var outN = nt \* rn/);
  });

  it('with-normals kernel composes the world inverse-transpose when applyWorld', () => {
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).toContain('mat3InverseTranspose(w0, w1, w2)');
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).toMatch(/outN = wnt \* outN/);
  });

  it('with-normals kernel normalizes the output normal and guards degeneracy', () => {
    // Scale first so both subnormal and near-max-f32 normals can be
    // normalized without length() underflow/overflow.
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).toContain(
      'let normalScale = max(abs(outN.x), max(abs(outN.y), abs(outN.z)))',
    );
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).toContain(
      'let scaledNormal = outN / normalScale',
    );
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).toContain(
      'let scaledLength = length(scaledNormal)',
    );
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).toContain(
      'safeN = scaledNormal / scaledLength',
    );
  });

  it('mirrors CPU rank-2 cofactor handling instead of applying the raw singular matrix', () => {
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).toContain(
      'let componentScale = max(max(abs(c0), abs(c1)), abs(c2))',
    );
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).toContain(
      'let n0 = c0 / matrixScale',
    );
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).toContain(
      'let det = dot(n0, cofactor0)',
    );
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).toContain(
      'let invDetAndScale = (1.0 / det) / matrixScale',
    );
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).toContain(
      'let cofactorMagnitude = sqrt(',
    );
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).toContain(
      'return mat3x3f(cofactor0, cofactor1, cofactor2)',
    );
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).toContain(
      'return mat3x3f(vec3f(0.0), vec3f(0.0), vec3f(0.0))',
    );
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).not.toContain(
      'return mat3x3f(c0, c1, c2)',
    );
  });

  it.each([
    {
      label: 'tiny full-rank uniform scale',
      columns: [
        [1e-11, 0, 0],
        [0, 1e-11, 0],
        [0, 0, 1e-11],
      ] as const,
      normal: [0.25, -0.5, 0.75] as const,
    },
    {
      label: 'tiny rank-2 scale',
      columns: [
        [2e-11, 0, 0],
        [0, 3e-11, 0],
        [0, 0, 0],
      ] as const,
      normal: [0, 0, 1] as const,
    },
  ])('keeps finite CPU-parity inverse and normal direction for $label', ({ columns, normal }) => {
    const [c0, c1, c2] = columns;
    const gpuColumns = gpuInverseTransposeMirror(c0, c1, c2);
    const gpuDirection = normalize(
      transformColumns(gpuColumns, normal),
    );
    const cpuMatrix = mat3InverseTranspose([
      c0[0], c1[0], c2[0],
      c0[1], c1[1], c2[1],
      c0[2], c1[2], c2[2],
    ]);
    const cpuDirection = normalize([
      cpuMatrix[0]! * normal[0] + cpuMatrix[1]! * normal[1] + cpuMatrix[2]! * normal[2],
      cpuMatrix[3]! * normal[0] + cpuMatrix[4]! * normal[1] + cpuMatrix[5]! * normal[2],
      cpuMatrix[6]! * normal[0] + cpuMatrix[7]! * normal[1] + cpuMatrix[8]! * normal[2],
    ]);

    const gpuRowMajor = [
      gpuColumns[0]![0], gpuColumns[1]![0], gpuColumns[2]![0],
      gpuColumns[0]![1], gpuColumns[1]![1], gpuColumns[2]![1],
      gpuColumns[0]![2], gpuColumns[1]![2], gpuColumns[2]![2],
    ];
    const comparisonScale = Math.max(1, ...Array.from(cpuMatrix, Math.abs));
    for (let i = 0; i < 9; i += 1) {
      expect((gpuRowMajor[i]! - cpuMatrix[i]!) / comparisonScale).toBeCloseTo(0, 6);
    }
    expect(gpuDirection.every(Number.isFinite)).toBe(true);
    expect(cpuDirection.every(Number.isFinite)).toBe(true);
    for (let i = 0; i < 3; i += 1) {
      expect(gpuDirection[i]).toBeCloseTo(cpuDirection[i]!, 6);
    }
  });

  it('WS1 — skins BOTH positions AND normals into the shared merged buffers at outIdx', () => {
    // Positions go to the shared merged buffer at baseVertex + vi.
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).toContain('bvhPositions[outIdx] = vec4f(outPos, uvPack)');
    // WS1 (2026-05-29) — normals now write into the SHARED merged bvh_normal
    // buffer at the SAME world-space slot (outIdx), so the smooth-shading-
    // normal blend consumes the skinned normal. The old mesh-local `vi` write
    // (dropped by applyGpuSkinnedRefit) is gone.
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).toContain(
      'let uv1Pack = skinnedNormals[outIdx].w',
    );
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).toContain(
      'skinnedNormals[outIdx] = vec4f(safeN, uv1Pack)',
    );
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).not.toContain(
      'skinnedNormals[vi] = vec4f(safeN, uv1Pack)',
    );
  });

  it('with-normals kernel still skins positions into bvhPositions', () => {
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).toContain('sp = sp + wi * p4');
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).toContain('outPos = (skinParams.matrixWorld * sp).xyz');
  });
});
