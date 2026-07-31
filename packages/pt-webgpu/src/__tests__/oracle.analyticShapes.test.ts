/**
 * H55 proof oracle — pt-webgpu analytic-shape local intersectors.
 *
 * The production WGSL dispatches sphere/box/capsule/cylinder/H-channel came
 * shapes from analyticHeaders. This file independently pins the local-frame ray
 * math so support is not just a capability/string-contract claim.
 */
import { describe, expect, it } from 'vitest';

import { PT_WEBGPU_ANALYTIC_SHAPES } from '../scene/uploadSceneBuffers.js';
import { PT_WEBGPU_PATH_TRACE_INTERSECTION_WGSL } from '../wgsl/pathTrace/intersection.wgsl.js';
import { PT_WEBGPU_INTERSECTION_CORE_WGSL } from '../wgsl/pathTrace/intersectionCore.wgsl.js';

type Vec3 = readonly [number, number, number];

interface Ray {
  readonly origin: Vec3;
  readonly direction: Vec3;
}

interface Hit {
  readonly t: number;
  readonly normal: Vec3;
}

const INF = Number.POSITIVE_INFINITY;
const EPS = 0;

function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function add(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function scale(a: Vec3, s: number): Vec3 {
  return [a[0] * s, a[1] * s, a[2] * s];
}

function normalize(v: Vec3): Vec3 {
  const len = Math.hypot(v[0], v[1], v[2]);
  if (len <= 1e-12) return [0, 0, 0];
  return [v[0] / len, v[1] / len, v[2] / len];
}

function safeInvDir(d: Vec3): Vec3 {
  const maxFiniteF32 = 3.402823e38;
  function inv(v: number): number {
    if (v === 0) return v >= 0 ? maxFiniteF32 : -maxFiniteF32;
    return Math.max(-maxFiniteF32, Math.min(maxFiniteF32, 1 / v));
  }
  return [inv(d[0]), inv(d[1]), inv(d[2])];
}

function expectHit(hit: Hit, expectedT: number, expectedNormal: Vec3): void {
  expect(hit.t).toBeCloseTo(expectedT, 10);
  expect(hit.normal[0]).toBeCloseTo(expectedNormal[0], 10);
  expect(hit.normal[1]).toBeCloseTo(expectedNormal[1], 10);
  expect(hit.normal[2]).toBeCloseTo(expectedNormal[2], 10);
}

function hitPoint(ray: Ray, t: number): Vec3 {
  return add(ray.origin, scale(ray.direction, t));
}

function intersectAabbDetailed(
  ray: Ray,
  bmin: Vec3,
  bmax: Vec3,
  tMin = 0,
  tMax = INF,
): Hit | null {
  const inv = safeInvDir(ray.direction);
  const t0: Vec3 = [
    (bmin[0] - ray.origin[0]) * inv[0],
    (bmin[1] - ray.origin[1]) * inv[1],
    (bmin[2] - ray.origin[2]) * inv[2],
  ];
  const t1: Vec3 = [
    (bmax[0] - ray.origin[0]) * inv[0],
    (bmax[1] - ray.origin[1]) * inv[1],
    (bmax[2] - ray.origin[2]) * inv[2],
  ];
  const tsm: Vec3 = [Math.min(t0[0], t1[0]), Math.min(t0[1], t1[1]), Math.min(t0[2], t1[2])];
  const tbg: Vec3 = [Math.max(t0[0], t1[0]), Math.max(t0[1], t1[1]), Math.max(t0[2], t1[2])];
  const tNear = Math.max(tsm[0], tsm[1], tsm[2]);
  const tFar = Math.min(tbg[0], tbg[1], tbg[2]);
  if (tNear > tFar || tFar < tMin || tNear > tMax) return null;

  let t = tNear;
  let fromFar = false;
  if (t < tMin) {
    t = tFar;
    fromFar = true;
  }

  const close = (a: number, b: number) => Math.abs(a - b) < 1e-4;
  let normal: Vec3;
  if (!fromFar) {
    if (close(t, tsm[0])) normal = [ray.direction[0] > 0 ? -1 : 1, 0, 0];
    else if (close(t, tsm[1])) normal = [0, ray.direction[1] > 0 ? -1 : 1, 0];
    else normal = [0, 0, ray.direction[2] > 0 ? -1 : 1];
  } else if (close(t, tbg[0])) normal = [ray.direction[0] > 0 ? 1 : -1, 0, 0];
  else if (close(t, tbg[1])) normal = [0, ray.direction[1] > 0 ? 1 : -1, 0];
  else normal = [0, 0, ray.direction[2] > 0 ? 1 : -1];
  return { t, normal };
}

function intersectSphereLocal(ray: Ray, center: Vec3, radius: number): Hit | null {
  const oc = sub(ray.origin, center);
  const a = dot(ray.direction, ray.direction);
  const b = 2 * dot(oc, ray.direction);
  const c = dot(oc, oc) - radius * radius;
  const disc = b * b - 4 * a * c;
  if (disc < 0) return null;
  const s = Math.sqrt(disc);
  let t = (-b - s) / (2 * a);
  if (t < EPS) t = (-b + s) / (2 * a);
  if (t < EPS) return null;
  return { t, normal: normalize(sub(hitPoint(ray, t), center)) };
}

function intersectCylinderLocal(ray: Ray, center: Vec3, radius: number, halfHeight: number): Hit | null {
  const ro = sub(ray.origin, center);
  const rd = ray.direction;
  const a = rd[0] * rd[0] + rd[2] * rd[2];
  const b = 2 * (ro[0] * rd[0] + ro[2] * rd[2]);
  const c = ro[0] * ro[0] + ro[2] * ro[2] - radius * radius;
  let bestT = INF;
  let bestN: Vec3 = [0, 1, 0];

  const disc = b * b - 4 * a * c;
  if (disc >= 0 && Math.abs(a) > 1e-8) {
    const s = Math.sqrt(disc);
    for (const t of [(-b - s) / (2 * a), (-b + s) / (2 * a)]) {
      const y = ro[1] + rd[1] * t;
      if (t > EPS && t < bestT && Math.abs(y) <= halfHeight) {
        bestT = t;
        bestN = normalize([ro[0] + rd[0] * t, 0, ro[2] + rd[2] * t]);
      }
    }
  }

  if (Math.abs(rd[1]) > 1e-8) {
    for (const [t, n] of [
      [(halfHeight - ro[1]) / rd[1], [0, 1, 0] as Vec3],
      [(-halfHeight - ro[1]) / rd[1], [0, -1, 0] as Vec3],
    ] as const) {
      const p = add(ro, scale(rd, t));
      if (t > EPS && t < bestT && p[0] * p[0] + p[2] * p[2] <= radius * radius) {
        bestT = t;
        bestN = n;
      }
    }
  }

  return bestT < INF ? { t: bestT, normal: bestN } : null;
}

function intersectCapsuleLocal(ray: Ray, pa: Vec3, pb: Vec3, radius: number): Hit | null {
  const ba = sub(pb, pa);
  const oa = sub(ray.origin, pa);
  const baba = dot(ba, ba);
  if (!(baba > 0)) return intersectSphereLocal(ray, pa, radius);
  const bard = dot(ba, ray.direction);
  const baoa = dot(ba, oa);
  const rdrd = dot(ray.direction, ray.direction);
  if (!(rdrd > 0)) return null;
  const rdoa = dot(ray.direction, oa);
  const oaoa = dot(oa, oa);
  const a = baba * rdrd - bard * bard;
  const b = baba * rdoa - baoa * bard;
  const c = baba * oaoa - baoa * baoa - radius * radius * baba;
  const h = b * b - a * c;
  let bestT = INF;
  let bestN: Vec3 = [0, 1, 0];

  if (h >= 0 && a > 0) {
    const sqrtH = Math.sqrt(h);
    for (const tBody of [(-b - sqrtH) / a, (-b + sqrtH) / a]) {
      const y = baoa + tBody * bard;
      if (tBody > EPS && tBody < bestT && y > 0 && y < baba) {
        const p = sub(add(oa, scale(ray.direction, tBody)), scale(ba, y / baba));
        bestT = tBody;
        bestN = normalize(p);
      }
    }
  }

  for (const [center, acceptHemisphere] of [
    [pa, (p: Vec3) => dot(sub(p, pa), ba) <= 0],
    [pb, (p: Vec3) => dot(sub(p, pb), ba) >= 0],
  ] as const) {
    const oc = sub(ray.origin, center);
    const bSphere = dot(oc, ray.direction);
    const cSphere = dot(oc, oc) - radius * radius;
    const hSphere = bSphere * bSphere - rdrd * cSphere;
    if (hSphere >= 0) {
      const sqrtH = Math.sqrt(hSphere);
      for (const tCap of [
        (-bSphere - sqrtH) / rdrd,
        (-bSphere + sqrtH) / rdrd,
      ]) {
        const p = hitPoint(ray, tCap);
        if (tCap > EPS && tCap < bestT && acceptHemisphere(p)) {
          bestT = tCap;
          bestN = normalize(sub(p, center));
        }
      }
    }
  }

  return bestT < INF ? { t: bestT, normal: bestN } : null;
}

function intersectHChannelLocal(
  ray: Ray,
  lengthX: number,
  railWidth: number,
  blockHeight: number,
  webThickness: number,
): Hit | null {
  const hx = lengthX * 0.5;
  const hy = blockHeight * 0.5;
  const hz = railWidth * 0.5;
  const t = webThickness * 0.5;
  const boundaryProbe = Math.min(hx, hy, hz, t) * 1e-5;
  const boxes: readonly [Vec3, Vec3][] = [
    [[-hx, hy - t, -hz], [hx, hy, hz]],
    [[-hx, -hy, -hz], [hx, -hy + t, hz]],
    [[-hx, -hy + t, -t], [hx, hy - t, t]],
  ];

  const contains = (point: Vec3): boolean => boxes.some(([bmin, bmax]) =>
    point[0] >= bmin[0] && point[0] <= bmax[0] &&
    point[1] >= bmin[1] && point[1] <= bmax[1] &&
    point[2] >= bmin[2] && point[2] <= bmax[2]);

  let advancedT = 0;
  for (let boundary = 0; boundary < 6; boundary += 1) {
    const scanRay: Ray = {
      origin: hitPoint(ray, advancedT),
      direction: ray.direction,
    };
    let nearest: Hit | null = null;
    for (const [bmin, bmax] of boxes) {
      const hit = intersectAabbDetailed(scanRay, bmin, bmax);
      if (hit && (!nearest || hit.t < nearest.t)) nearest = hit;
    }
    if (!nearest) return null;
    const absoluteT = advancedT + nearest.t;
    const before = contains(hitPoint(ray, Math.max(0, absoluteT - boundaryProbe)));
    const after = contains(hitPoint(ray, absoluteT + boundaryProbe));
    if (before !== after) return { t: absoluteT, normal: nearest.normal };
    advancedT = absoluteT + 2 * boundaryProbe;
  }
  return null;
}

describe('pt-webgpu analytic-shape oracle', () => {
  it('keeps public analytic shape ids aligned with the WGSL discriminants and dispatch', () => {
    expect(PT_WEBGPU_ANALYTIC_SHAPES).toEqual([
      'unknown',
      'sphere',
      'box',
      'capsule',
      'cylinder',
      'h-channel-came',
    ]);
    for (const symbol of [
      'const SHAPE_SPHERE = 1u;',
      'const SHAPE_BOX = 2u;',
      'const SHAPE_CAPSULE = 3u;',
      'const SHAPE_CYLINDER = 4u;',
      'const SHAPE_H_CHANNEL_CAME = 5u;',
      'fn intersectSphereLocal(',
      'fn intersectCylinderLocal(',
      'fn intersectCapsuleLocal(',
      'fn intersectHChannelLocal(',
    ]) {
      expect(PT_WEBGPU_INTERSECTION_CORE_WGSL).toContain(symbol);
    }
    expect(PT_WEBGPU_PATH_TRACE_INTERSECTION_WGSL).toContain('if (shapeId == SHAPE_SPHERE)');
    expect(PT_WEBGPU_PATH_TRACE_INTERSECTION_WGSL).toContain('} else if (shapeId == SHAPE_H_CHANNEL_CAME)');
    expect(PT_WEBGPU_PATH_TRACE_INTERSECTION_WGSL).toContain('(*hit).triIndex = params.triangleCount + ai;');
  });

  it('keeps capsule far roots and H-channel union-boundary traversal in production WGSL', () => {
    expect(PT_WEBGPU_INTERSECTION_CORE_WGSL).toContain(
      'let tBodyFarScaled = (-b + sqrtH) / a;',
    );
    expect(PT_WEBGPU_INTERSECTION_CORE_WGSL).toContain(
      'dot(capPointA, ba) <= 0.0',
    );
    expect(PT_WEBGPU_INTERSECTION_CORE_WGSL).toContain(
      'dot(capPointB, ba) >= 0.0',
    );
    expect(PT_WEBGPU_INTERSECTION_CORE_WGSL).toContain(
      'fn hChannelContainsLocal(',
    );
    expect(PT_WEBGPU_INTERSECTION_CORE_WGSL).toContain(
      'for (var boundary = 0u; boundary < 6u; boundary = boundary + 1u)',
    );
  });

  it('keeps analytic dimensions exact and separates local from world ray thresholds', () => {
    expect(PT_WEBGPU_PATH_TRACE_INTERSECTION_WGSL).toContain(
      'localT = intersectSphereLocal(localRay, p0.xyz, p0.w, &localN);',
    );
    expect(PT_WEBGPU_PATH_TRACE_INTERSECTION_WGSL).toContain(
      'localRay, p0.xyz - p1.xyz, p0.xyz + p1.xyz, 0.0, INFINITY, &localN,',
    );
    expect(PT_WEBGPU_PATH_TRACE_INTERSECTION_WGSL).toContain(
      'if (!(localT > 0.0) || localT >= INFINITY)',
    );
    expect(PT_WEBGPU_PATH_TRACE_INTERSECTION_WGSL).not.toContain('max(p0.w, 1e-4)');
    expect(PT_WEBGPU_PATH_TRACE_INTERSECTION_WGSL).not.toContain('max(p1.w, 1e-4)');
    expect(PT_WEBGPU_INTERSECTION_CORE_WGSL).not.toContain('baba <= 1e-12');
    expect(PT_WEBGPU_INTERSECTION_CORE_WGSL).toContain(
      'let boundaryProbe = featureScale * 1e-5;',
    );
  });

  it('pins sphere front-face, inside-exit, and miss behavior', () => {
    expectHit(
      intersectSphereLocal({ origin: [-3, 0, 0], direction: [1, 0, 0] }, [0, 0, 0], 1)!,
      2,
      [-1, 0, 0],
    );
    expectHit(
      intersectSphereLocal({ origin: [0, 0, 0], direction: [0, 1, 0] }, [0, 0, 0], 1)!,
      1,
      [0, 1, 0],
    );
    expect(intersectSphereLocal({ origin: [-3, 0, 0], direction: [0, 1, 0] }, [0, 0, 0], 1)).toBeNull();
  });

  it('pins box slab entry normals and inside-exit normals', () => {
    expectHit(
      intersectAabbDetailed({ origin: [3, 0.25, 0], direction: [-1, 0, 0] }, [-1, -2, -0.5], [1, 2, 0.5])!,
      2,
      [1, 0, 0],
    );
    expectHit(
      intersectAabbDetailed({ origin: [0, 0, 0], direction: [0, 1, 0] }, [-1, -2, -0.5], [1, 2, 0.5])!,
      2,
      [0, 1, 0],
    );
    expect(intersectAabbDetailed({ origin: [3, 3, 0], direction: [-1, 0, 0] }, [-1, -2, -0.5], [1, 2, 0.5])).toBeNull();
  });

  it('pins finite cylinder side and cap selection', () => {
    expectHit(
      intersectCylinderLocal({ origin: [3, 0.5, 0], direction: [-1, 0, 0] }, [0, 0, 0], 1, 2)!,
      2,
      [1, 0, 0],
    );
    expectHit(
      intersectCylinderLocal({ origin: [0, 3, 0], direction: [0, -1, 0] }, [0, 0, 0], 1, 2)!,
      1,
      [0, 1, 0],
    );
    expect(intersectCylinderLocal({ origin: [2, 3, 0], direction: [0, -1, 0] }, [0, 0, 0], 1, 2)).toBeNull();
  });

  it('pins capsule body and spherical cap selection', () => {
    expectHit(
      intersectCapsuleLocal({ origin: [2, 0, 0], direction: [-1, 0, 0] }, [0, -1, 0], [0, 1, 0], 0.5)!,
      1.5,
      [1, 0, 0],
    );
    expectHit(
      intersectCapsuleLocal({ origin: [0, 2, 0], direction: [0, -1, 0] }, [0, -1, 0], [0, 1, 0], 0.5)!,
      0.5,
      [0, 1, 0],
    );
    expect(intersectCapsuleLocal({ origin: [2, 2, 0], direction: [0, 1, 0] }, [0, -1, 0], [0, 1, 0], 0.5)).toBeNull();

    expectHit(
      intersectCapsuleLocal({ origin: [0, 0, 0], direction: [1, 0, 0] }, [0, -1, 0], [0, 1, 0], 0.5)!,
      0.5,
      [1, 0, 0],
    );
    expectHit(
      intersectCapsuleLocal({ origin: [0, 0, 0], direction: [2, 0, 0] }, [0, -1, 0], [0, 1, 0], 0.5)!,
      0.25,
      [1, 0, 0],
    );
    expectHit(
      intersectCapsuleLocal({ origin: [0, 1.25, 0], direction: [0, 1, 0] }, [0, -1, 0], [0, 1, 0], 0.5)!,
      0.25,
      [0, 1, 0],
    );
    expectHit(
      intersectCapsuleLocal({ origin: [0, 0, 0], direction: [0, 1, 0] }, [0, 0, 0], [0, 0, 0], 0.5)!,
      0.5,
      [0, 1, 0],
    );
  });

  it('pins H-channel as nearest hit across top rail, bottom rail, and web boxes', () => {
    expectHit(
      intersectHChannelLocal({ origin: [0, 3, 0], direction: [0, -1, 0] }, 4, 1, 2, 0.4)!,
      2,
      [0, 1, 0],
    );
    expectHit(
      intersectHChannelLocal({ origin: [0, -3, 0], direction: [0, 1, 0] }, 4, 1, 2, 0.4)!,
      2,
      [0, -1, 0],
    );
    expectHit(
      intersectHChannelLocal({ origin: [0, 0, 3], direction: [0, 0, -1] }, 4, 1, 2, 0.4)!,
      2.8,
      [0, 0, 1],
    );
    expect(intersectHChannelLocal({ origin: [0, 0, 3], direction: [0, 1, 0] }, 4, 1, 2, 0.4)).toBeNull();
    expectHit(
      intersectHChannelLocal({ origin: [0, 0, 0], direction: [0, 1, 0] }, 4, 1, 2, 0.4)!,
      1,
      [0, 1, 0],
    );
  });

  it('preserves valid analytic geometry far below the former absolute floors', () => {
    const s = 1e-9;
    expectHit(
      intersectSphereLocal(
        { origin: [-3 * s, 0, 0], direction: [1, 0, 0] },
        [0, 0, 0],
        s,
      )!,
      2 * s,
      [-1, 0, 0],
    );
    expectHit(
      intersectAabbDetailed(
        { origin: [3 * s, 0, 0], direction: [-1, 0, 0] },
        [-s, -2 * s, -0.5 * s],
        [s, 2 * s, 0.5 * s],
      )!,
      2 * s,
      [1, 0, 0],
    );
    expectHit(
      intersectCylinderLocal(
        { origin: [3 * s, 0, 0], direction: [-1, 0, 0] },
        [0, 0, 0],
        s,
        2 * s,
      )!,
      2 * s,
      [1, 0, 0],
    );
    expectHit(
      intersectCapsuleLocal(
        { origin: [2 * s, 0, 0], direction: [-1, 0, 0] },
        [0, -s, 0],
        [0, s, 0],
        0.5 * s,
      )!,
      1.5 * s,
      [1, 0, 0],
    );
    expectHit(
      intersectHChannelLocal(
        { origin: [0, 3 * s, 0], direction: [0, -1, 0] },
        4 * s,
        s,
        2 * s,
        0.4 * s,
      )!,
      2 * s,
      [0, 1, 0],
    );
  });
});
