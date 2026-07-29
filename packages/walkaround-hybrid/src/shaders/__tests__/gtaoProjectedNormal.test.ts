import { describe, expect, it } from 'vitest';
import { BIND_GROUP_TABLE } from '../../pipeline/bindGroupDescriptors.js';
import { GTAO_UBO } from '../../pipeline/passes/uboLayouts.js';
import { GTAO_WGSL } from '../gtao.wgsl.js';
import { GTAO_COMMON_WGSL } from '../gtaoCommon.wgsl.js';

type Vec3 = readonly [number, number, number];

function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function subtract(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function scale(v: Vec3, amount: number): Vec3 {
  return [v[0] * amount, v[1] * amount, v[2] * amount];
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function length(v: Vec3): number {
  return Math.sqrt(dot(v, v));
}

function normalize(v: Vec3): Vec3 {
  const magnitude = length(v);
  return [
    v[0] / magnitude,
    v[1] / magnitude,
    v[2] / magnitude,
  ];
}

function projectNormalForSlice(
  normal: Vec3,
  direction: Vec3,
  view: Vec3,
): {
  readonly axis: Vec3;
  readonly orthoDirection: Vec3;
  readonly projected: Vec3;
  readonly angle: number;
} {
  const orthoDirection = subtract(direction, scale(view, dot(direction, view)));
  const axis = normalize(cross(orthoDirection, view));
  const projected = subtract(normal, scale(axis, dot(normal, axis)));
  const projectedLength = Math.max(length(projected), 1e-6);
  const sign = Math.sign(dot(orthoDirection, projected));
  const cosine = Math.min(
    1,
    Math.max(0, dot(projected, view) / projectedLength),
  );
  return {
    axis,
    orthoDirection,
    projected,
    angle: sign * Math.acos(cosine),
  };
}

describe('GTAO projected-normal slice geometry', () => {
  it('uses the in-plane direction for a non-zero signed projected-normal angle', () => {
    const view: Vec3 = [0, 0, 1];
    const direction: Vec3 = [1, 0, 0];
    const normal = normalize([0.6, 0, 0.8]);
    const result = projectNormalForSlice(normal, direction, view);

    expect(result.angle).toBeCloseTo(Math.atan2(0.6, 0.8), 12);
    expect(dot(result.axis, result.projected)).toBeCloseTo(0, 12);
    expect(dot(result.axis, result.orthoDirection)).toBeCloseTo(0, 12);
    expect(dot(result.axis, view)).toBeCloseTo(0, 12);

    // The removed implementation signed with dot(axis, projected), which is
    // identically zero because projected is constructed perpendicular to axis.
    expect(Math.sign(dot(result.axis, result.projected))).toBe(0);
  });

  it('pins the XeGTAO plane construction and world-to-view normal transform', () => {
    expect(GTAO_WGSL).toContain(
      'directionVec - viewAxis * dot(directionVec, viewAxis)',
    );
    expect(GTAO_WGSL).toContain(
      'normalize(cross(orthoDirection, viewAxis))',
    );
    expect(GTAO_WGSL).toContain(
      'sign(dot(orthoDirection, projNormal))',
    );
    expect(GTAO_WGSL).not.toContain(
      'sign(dot(axisVec, projNormal))',
    );
    expect(GTAO_WGSL).toContain(
      '(gtao_ubo.viewMatrix * vec4f(surfNormalWorld, 0.0)).xyz',
    );
    expect(GTAO_COMMON_WGSL).toContain('viewMatrix: mat4x4f');
  });

  it('keeps the host UBO, WGSL struct, resource binding, and matrix offset coherent', () => {
    expect(GTAO_UBO.sizeBytes).toBe(96);
    expect(GTAO_UBO.fieldOffsets.viewMatrix).toBe(32);

    for (const id of ['gtao', 'gtaoUpsample'] as const) {
      const family = BIND_GROUP_TABLE.find((entry) => entry.id === id);
      const uniform = family?.entries.find((entry) => entry.kind === 'uniform');
      expect(uniform?.minSizeBytes).toBe(96);
    }
  });
});
