import { describe, expect, it } from 'vitest';
import * as BvhRayFunctionsNS from './bvh/bvh_ray_functions.glsl.js';
import * as BdptConnectionNS from './render/bdpt_connection.glsl.js';
import * as BdptLightSubpathNS from './render/bdpt_light_subpath.glsl.js';

const bvhRayFunctions =
  (BvhRayFunctionsNS as unknown as Record<string, string>)[
    'bvh_ray_functions'
  ] ?? '';
const bdptLightSubpath =
  (BdptLightSubpathNS as unknown as Record<string, string>)[
    'bdpt_light_subpath'
  ] ?? '';
const bdptConnection =
  (BdptConnectionNS as unknown as Record<string, string>)[
    'bdpt_connection'
  ] ?? '';

type Vec3 = readonly [number, number, number];
type Vec4 = readonly [number, number, number, number];

function stableBoundsIntersection(
  origin: Vec3,
  direction: Vec3,
  boundsMin: Vec3,
  boundsMax: Vec3,
): { hit: boolean; distance: number } {
  if (
    [...origin, ...direction, ...boundsMin, ...boundsMax]
      .some((value) => !Number.isFinite(value)) ||
    boundsMin.some((value, axis) => value > boundsMax[axis]!)
  ) {
    return { hit: false, distance: 0 };
  }

  let nearDistance = 0;
  let farDistance = Number.POSITIVE_INFINITY;
  for (let axis = 0; axis < 3; axis += 1) {
    const axisOrigin = origin[axis]!;
    const axisDirection = direction[axis]!;
    const slabMin = boundsMin[axis]!;
    const slabMax = boundsMax[axis]!;
    if (axisDirection === 0) {
      if (axisOrigin < slabMin || axisOrigin > slabMax) {
        return { hit: false, distance: 0 };
      }
      continue;
    }
    const first = (slabMin - axisOrigin) / axisDirection;
    const second = (slabMax - axisOrigin) / axisDirection;
    nearDistance = Math.max(nearDistance, Math.min(first, second));
    farDistance = Math.min(farDistance, Math.max(first, second));
    if (farDistance < nearDistance) return { hit: false, distance: 0 };
  }
  return { hit: true, distance: nearDistance };
}

function storedRowsValid(rows: readonly Vec4[]): boolean {
  if (rows.length !== 14 || rows.some((row) => row.some((v) => !Number.isFinite(v)))) {
    return false;
  }
  const kind = rows[0]![3];
  const forwardDensity = rows[1]![3];
  const throughput = rows[2]!;
  const stackCount = rows[5]![0];
  return (
    (kind === 0 || kind === 1 || kind === 2) &&
    forwardDensity > 0 &&
    throughput[3] >= 0 &&
    throughput.slice(0, 3).every((value) => value >= 0) &&
    Number.isInteger(stackCount) &&
    stackCount >= 0 &&
    stackCount <= 8
  );
}

describe('pt-webgl2 BDPT/BVH scale safety', () => {
  it('classifies axis-parallel slab boundary, inside, and outside rays explicitly', () => {
    const boundsMin: Vec3 = [-1, -1, -1];
    const boundsMax: Vec3 = [1, 1, 1];

    expect(stableBoundsIntersection(
      [-1, 0, -5],
      [0, 0, 1],
      boundsMin,
      boundsMax,
    )).toEqual({ hit: true, distance: 4 });
    expect(stableBoundsIntersection(
      [0, 0, 0],
      [-0, 1, 0],
      boundsMin,
      boundsMax,
    )).toEqual({ hit: true, distance: 0 });
    expect(stableBoundsIntersection(
      [1.01, 0, -5],
      [0, 0, 1],
      boundsMin,
      boundsMax,
    )).toEqual({ hit: false, distance: 0 });

    // This is the exact indeterminate product retired from the shader.
    expect((boundsMin[0] - -1) * (1 / 0)).toBeNaN();
    expect(bvhRayFunctions).toContain('if ( direction == 0.0 )');
    expect(bvhRayFunctions).toContain(
      'if ( origin < slabMin || origin > slabMax )',
    );
    expect(bvhRayFunctions).toContain(
      'float first = ( slabMin - origin ) / direction;',
    );
    expect(bvhRayFunctions).not.toContain('vec3 invDir = 1.0 / rayDirection;');
    expect(bvhRayFunctions).not.toContain(
      'invDir * ( boundsMin - rayOrigin )',
    );
  });

  it('uses exact launches only for accepted transmission and containment replay', () => {
    const compact = bdptLightSubpath.replace(/\s+/g, ' ');
    expect(compact).toContain(
      'vec3 rayOrigin = exactPreviousTransmission ? p0.xyz : stepRayOrigin( p0.xyz, vec3( 0.0 ), unitScatterDirection, 0.0 )',
    );
    expect(compact).toContain(
      'vec3 endpointLaunchOrigin = p0.xyz;',
    );
    expect(compact).toContain(
      'rayOrigin = stepRayOrigin( p0.xyz, vec3( 0.0 ), offsetSurf.faceNormal * side, 0.0 )',
    );
    expect(bdptLightSubpath).not.toContain(
      'p0.xyz + scatterDir * RAY_OFFSET',
    );
    expect(bdptLightSubpath).not.toContain(
      'p0.xyz + endpointLaunchDirection * RAY_OFFSET',
    );
    expect(bdptLightSubpath).not.toContain(
      'p0.xyz + offsetSurf.faceNormal * ( side * RAY_OFFSET )',
    );
  });

  it('uses scale-safe edge length and normalization at the stored-vertex boundary', () => {
    const compactConnection = bdptConnection.replace(/\s+/g, ' ');
    expect(bdptLightSubpath).toContain(
      'float edgeDistance = vitrumLengthVec3( newPos - p0.xyz );',
    );
    expect(bdptLightSubpath).toContain(
      'vitrumNormalizeVec3( edgeToPredecessor, vec3( 0.0 ) )',
    );
    expect(bdptLightSubpath).toContain(
      'vitrumNormalizeVec3( scatterDir, vec3( 0.0 ) )',
    );
    expect(bdptLightSubpath).not.toContain(
      'float edgeDistance = distance( p0.xyz, newPos );',
    );
    expect(bdptLightSubpath).not.toMatch(/\bnormalize\s*\(/);
    expect(bdptConnection).toContain(
      'float distance = vitrumLengthVec3( d );',
    );
    expect(compactConnection).toContain(
      'vec3 connDir = vitrumNormalizeVec3( toLight, vec3( 0.0 ) )',
    );
    expect(bdptConnection).toContain(
      'if ( all( equal( d, vec3( 0.0 ) ) ) )',
    );
    expect(bdptConnection).not.toMatch(/\bnormalize\s*\(/);
    expect(bdptConnection).not.toMatch(/\bdistance\s*\(/);
    expect(bdptConnection).not.toMatch(/\blength\s*\(/);
  });

  it('fails endpoint and bounce records closed before any nonfinite row is published', () => {
    const validRows: Vec4[] = [
      [0, 0, 0, 0],
      [0, 1, 0, 0.25],
      [4, 2, 1, 0],
      [0, 1, 0, -4],
      [0, 0, 0, 0],
      [0, -1, -1, -1],
      [-1, -1, -1, -1],
      [-1, -1, 0, 0],
      [-1, -1, -1, -1],
      [-1, -1, -1, -1],
      [-1, -1, -1, -1],
      [-1, -1, -1, -1],
      [-1, -1, -1, -1],
      [-1, -1, -1, -1],
    ];
    expect(storedRowsValid(validRows)).toBe(true);
    expect(storedRowsValid(validRows.map((row, index) =>
      index === 2 ? [Number.POSITIVE_INFINITY, row[1], row[2], row[3]] : row,
    ))).toBe(false);
    expect(storedRowsValid(validRows.map((row, index) =>
      index === 1 ? [row[0], row[1], row[2], Number.NaN] : row,
    ))).toBe(false);
    expect(storedRowsValid(validRows.map((row, index) =>
      index === 5 ? [8.5, row[1], row[2], row[3]] : row,
    ))).toBe(false);

    const overflowedThroughput = Math.fround(
      Math.fround(3e38) / Math.fround(1e-38),
    );
    expect(overflowedThroughput).toBe(Number.POSITIVE_INFINITY);
    expect(bdptLightSubpath).toContain('bool bdptStoredVertexRowsValid(');
    const compactLightSubpath = bdptLightSubpath.replace(/\s+/g, ' ');
    expect(compactLightSubpath).toContain(
      'v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13',
    );
    expect(compactLightSubpath).toContain(
      'p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13',
    );
    expect(bdptConnection.replace(/\s+/g, ' ')).toContain(
      'bdptStoredVertexRowsValid( lv0, lv1, lv2, lv3, lv4, lv5, lv6, lv7, lv8, lv9, lv10, lv11, lv12, lv13 )',
    );
    expect(bdptLightSubpath).toContain(
      'float predecessorReverseDensity = reverseScatterPdf * p2.w;',
    );
    expect(compactLightSubpath.indexOf(
      'v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13',
    )).toBeLessThan(compactLightSubpath.lastIndexOf(
      'predecessor2.w = predecessorReverseDensity;',
    ));
  });
});
