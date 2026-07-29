import { describe, expect, it } from 'vitest';
import { PROBE_RAY_CAST_WGSL } from '../src/wgsl/probeRayCast.wgsl.js';
import { CASCADE_MERGE_WGSL } from '../src/wgsl/cascadeMerge.wgsl.js';

type Vec3 = readonly [number, number, number];
type Rgb = readonly [number, number, number];

function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function normalize(v: Vec3): Vec3 {
  const length = Math.sqrt(dot(v, v));
  return [v[0] / length, v[1] / length, v[2] / length];
}

function refract(i: Vec3, n: Vec3, eta: number): Vec3 | null {
  const ni = dot(n, i);
  const k = 1 - eta * eta * (1 - ni * ni);
  if (k < 0) return null;
  const c = eta * ni + Math.sqrt(k);
  return normalize([
    eta * i[0] - c * n[0],
    eta * i[1] - c * n[1],
    eta * i[2] - c * n[2],
  ]);
}

function distance(a: Vec3, b: Vec3): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

function acceptsTerminal(
  events: readonly ('bulk' | 'thin' | 'opaque' | 'miss')[],
  budget: number,
): boolean {
  let interfaces = 0;
  for (const event of events) {
    if (event === 'opaque' || event === 'miss') return interfaces <= budget;
    const cost = event === 'thin' ? 2 : 1;
    if (interfaces + cost > budget) return false;
    interfaces += cost;
  }
  return false;
}

function dielectricFresnel(cosTheta: number, etaI: number, etaT: number): number {
  const c = Math.min(1, Math.max(0, cosTheta));
  const sin2T = (etaI / etaT) ** 2 * Math.max(0, 1 - c * c);
  if (sin2T >= 1) return 1;
  const cosT = Math.sqrt(Math.max(0, 1 - sin2T));
  const rs = (etaI * c - etaT * cosT) / (etaI * c + etaT * cosT);
  const rp = (etaT * c - etaI * cosT) / (etaT * c + etaI * cosT);
  return 0.5 * (rs * rs + rp * rp);
}

function thinSheetTransmittance(cosTheta: number, enclosingIor: number, sheetIor: number): number {
  const entryT = 1 - dielectricFresnel(cosTheta, enclosingIor, sheetIor);
  const sin2Inside = (enclosingIor / sheetIor) ** 2 * Math.max(0, 1 - cosTheta * cosTheta);
  if (sin2Inside >= 1) return 0;
  const cosInside = Math.sqrt(Math.max(0, 1 - sin2Inside));
  const exitT = 1 - dielectricFresnel(cosInside, sheetIor, enclosingIor);
  return entryT * exitT;
}

function nestedExitMatchesTop(
  bvhMode: number,
  topMaterial: number,
  hitMaterial: number,
  topInstance: number,
  hitInstance: number,
): boolean {
  return topMaterial === hitMaterial && (bvhMode !== 1 || topInstance === hitInstance);
}

function completeSuffixDistanceCap(
  initialHitDistance: number,
  roomSize: Vec3,
  interfaceBudget: number,
): number {
  return initialHitDistance +
    Math.hypot(roomSize[0], roomSize[1], roomSize[2]) * (interfaceBudget + 1);
}

function scaleRgb(a: Rgb, b: Rgb | number): Rgb {
  return typeof b === 'number'
    ? [a[0] * b, a[1] * b, a[2] * b]
    : [a[0] * b[0], a[1] * b[1], a[2] * b[2]];
}

function addRgb(a: Rgb, b: Rgb): Rgb {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function beerRgb(
  attenuationColor: Rgb,
  segmentDistance: number,
  attenuationDistance: number,
): Rgb {
  return attenuationColor.map(
    channel => channel ** (segmentDistance / attenuationDistance),
  ) as [number, number, number];
}

function mergeResolvedCascade(
  local: readonly [number, number, number, number],
  upper: Rgb,
): Rgb {
  return local[3] > 0.5
    ? [local[0], local[1], local[2]]
    : addRgb([local[0], local[1], local[2]], upper);
}

describe('RC bounded dielectric transport closure', () => {
  it('uses channel-separated exact transport and no legacy one-hit continuation', () => {
    expect(PROBE_RAY_CAST_WGSL).toContain('fn rcTraceTransmittedChannel(');
    expect(PROBE_RAY_CAST_WGSL).toContain('fn rcDielectricInterfaceTransmissionRgb(');
    expect(PROBE_RAY_CAST_WGSL).toContain('materialSpectralAttenuation(');
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'fn rcCompleteDielectricSuffixMaxDistance(initialHitDistance: f32) -> f32',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'let boundedSegments = u.transmittedInterfaceBudget + 1u;',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'initialHitDistance + length(u.roomSize) * f32(boundedSegments)',
    );
    expect(PROBE_RAY_CAST_WGSL).toMatch(
      /rcTraceTransmittedChannel\(\s+ray, hit, dielectricMaxDistance, 0u/,
    );
    expect(PROBE_RAY_CAST_WGSL).toMatch(
      /rcTraceTransmittedChannel\(\s+ray, hit, dielectricMaxDistance, 1u/,
    );
    expect(PROBE_RAY_CAST_WGSL).toMatch(
      /rcTraceTransmittedChannel\(\s+ray, hit, dielectricMaxDistance, 2u/,
    );
    expect(PROBE_RAY_CAST_WGSL).not.toMatch(
      /rcTraceTransmittedChannel\(\s+ray, hit, maxT,/,
    );
    expect(PROBE_RAY_CAST_WGSL).not.toContain('let secondHit');
  });

  it('completes a refracted RGB bulk suffix beyond a non-last cascade interval', () => {
    const lowerIntervalFar = 1;
    const entryDistance = 0.75;
    const bulkDistance = 0.5;
    const receiverDistanceAfterExit = 0.75;
    const completeDistance =
      entryDistance + bulkDistance + receiverDistanceAfterExit;
    expect(completeDistance).toBeGreaterThan(lowerIntervalFar);

    const cap = completeSuffixDistanceCap(entryDistance, [3, 3, 3], 2);
    expect(cap).toBeGreaterThan(completeDistance);

    const incident = normalize([0.45, 0, -0.893]);
    const inside = refract(incident, [0, 0, 1], 1 / 1.5);
    expect(inside).not.toBeNull();
    const exit = refract(inside!, [0, 0, 1], 1.5);
    expect(exit).not.toBeNull();

    const straightReceiverX =
      incident[0] / -incident[2] * (bulkDistance + receiverDistanceAfterExit);
    const refractedReceiverX =
      inside![0] / -inside![2] * bulkDistance +
      exit![0] / -exit![2] * receiverDistanceAfterExit;
    expect(Math.abs(refractedReceiverX - straightReceiverX)).toBeGreaterThan(0.05);
    const targetSplit = (refractedReceiverX + straightReceiverX) * 0.5;
    expect(refractedReceiverX < targetSplit).not.toBe(straightReceiverX < targetSplit);

    const entryT = 1 - dielectricFresnel(-dot(incident, [0, 0, 1]), 1, 1.5);
    const exitT = 1 - dielectricFresnel(-dot(inside!, [0, 0, 1]), 1.5, 1);
    const beer = beerRgb([0.81, 0.49, 0.25], bulkDistance, 1);
    const receiver: Rgb = [6, 3, 1.5];
    const completed = scaleRgb(scaleRgb(receiver, beer), entryT * exitT);

    expect(completed[0]).toBeGreaterThan(0);
    expect(completed[1]).toBeGreaterThan(0);
    expect(completed[2]).toBeGreaterThan(0);
    expect(completed[0]).toBeGreaterThan(completed[1]);
    expect(completed[1]).toBeGreaterThan(completed[2]);
  });

  it('allows completed environment suffixes from last and non-last cascades only outside bulk', () => {
    const suffix = PROBE_RAY_CAST_WGSL.slice(
      PROBE_RAY_CAST_WGSL.indexOf('fn rcTraceTransmittedChannel('),
      PROBE_RAY_CAST_WGSL.indexOf('// ─── Entry point'),
    );
    expect(suffix).toContain('if (mediumDepth != 0u) { return 0.0; }');
    expect(suffix).not.toContain('u.cascadeIndex != u.lastCascade');
    expect(suffix).toContain('return throughput * rcRgbChannel(env, channel);');

    const environment: Rgb = [2, 1, 0.5];
    const thinT = thinSheetTransmittance(0.72, 1, 1.5);
    const nonLast = scaleRgb(environment, thinT);
    const last = scaleRgb(environment, thinT);
    expect(nonLast).toEqual(last);
    expect(nonLast.every(channel => channel > 0)).toBe(true);
  });

  it('keeps completed glass and opaque hits terminal while empty intervals merge upper once', () => {
    expect(CASCADE_MERGE_WGSL).toContain('if (local.a > 0.5) { return; }');
    expect(CASCADE_MERGE_WGSL).toContain(
      'rc_lowerCascade[lowerOutIdx] = vec4f(local.rgb + merged, 1.0);',
    );
    const completedGlass: readonly [number, number, number, number] =
      [0.9, 0.5, 0.2, 1];
    const opaque: readonly [number, number, number, number] =
      [0.3, 0.4, 0.5, 1];
    const empty: readonly [number, number, number, number] =
      [0, 0, 0, 0];
    const upper: Rgb = [3, 2, 1];
    expect(mergeResolvedCascade(completedGlass, upper)).toEqual(
      completedGlass.slice(0, 3),
    );
    expect(mergeResolvedCascade(opaque, upper)).toEqual(opaque.slice(0, 3));
    expect(mergeResolvedCascade(empty, upper)).toEqual(upper);
  });

  it('samples rough mapped dielectric interfaces and fails invalid orientation closed', () => {
    expect(PROBE_RAY_CAST_WGSL).toContain('let interfaceBtdf = rcSampleGgxDielectricTransmission(');
    expect(PROBE_RAY_CAST_WGSL).toMatch(
      /rcSampleGgxDielectricTransmission\([\s\S]*?probeMat\.roughness/,
    );
    expect(PROBE_RAY_CAST_WGSL).not.toContain('RC_GLASS_MAX_ROUGHNESS');
    expect(PROBE_RAY_CAST_WGSL).toContain('dot(shadingNormal, hit.normal) >= 0.0');
    expect(PROBE_RAY_CAST_WGSL).toContain('dot(ray.direction, faceNormal) >= 0.0');
  });

  it('uses a runtime 1..8 budget with eight-entry static medium stacks', () => {
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'transmittedInterfaceBudget: u32',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'const RC_GLASS_STATIC_MAX_INTERFACES: u32 = 8u;',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'let interfaceBudget = u.transmittedInterfaceBudget;',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'step <= RC_GLASS_STATIC_MAX_INTERFACES',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'interfaceCount + interfaceCost > interfaceBudget',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain('var mediumIor: array<vec3f, 8>;');
    expect(PROBE_RAY_CAST_WGSL).toContain('var mediumTri: array<u32, 8>;');
    expect(PROBE_RAY_CAST_WGSL).toContain('var mediumInstance: array<u32, 8>;');
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'var mediumAttenuationColor: array<vec3f, 8>;',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'var mediumAttenuationDistance: array<f32, 8>;',
    );
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'select(1u, 2u, thinSheet)',
    );
    expect(acceptsTerminal(
      ['bulk', 'bulk', 'bulk', 'bulk', 'bulk', 'bulk', 'bulk', 'bulk', 'opaque'],
      8,
    )).toBe(true);
    expect(acceptsTerminal(
      ['bulk', 'bulk', 'bulk', 'bulk', 'bulk', 'bulk', 'bulk', 'bulk', 'miss'],
      8,
    )).toBe(true);
    expect(acceptsTerminal(
      ['bulk', 'bulk', 'bulk', 'bulk', 'bulk', 'bulk', 'bulk', 'bulk', 'bulk'],
      8,
    )).toBe(false);
    expect(acceptsTerminal(['thin', 'thin', 'thin', 'thin', 'miss'], 8)).toBe(true);
    expect(acceptsTerminal(['thin', 'thin', 'thin', 'thin', 'thin'], 8)).toBe(false);
    expect(acceptsTerminal(['bulk', 'opaque'], 1)).toBe(true);
    expect(acceptsTerminal(['thin', 'opaque'], 1)).toBe(false);
  });

  it('treats front- and back-wound thin sheets as the same enclosing-medium to sheet round trip', () => {
    expect(PROBE_RAY_CAST_WGSL).toContain('let thinSheet = thickness <= 0.0;');
    expect(PROBE_RAY_CAST_WGSL).toContain('dot(ray.direction, alignedInterfaceNormal) < 0.0');
    expect(PROBE_RAY_CAST_WGSL).not.toMatch(/else if \(!entering\) \{\s*incidentIor = probeMat\.opticalIor/);
    expect(PROBE_RAY_CAST_WGSL).toContain('var targetIor = probeMat.opticalIor;');

    const front = thinSheetTransmittance(0.63, 1.33, 1.52);
    const back = thinSheetTransmittance(0.63, 1.33, 1.52);
    expect(front).toBeGreaterThan(0);
    expect(back).toBeCloseTo(front, 15);
  });

  it('supports a reciprocal thin sheet inside an active bulk without mutating the bulk stack', () => {
    expect(PROBE_RAY_CAST_WGSL).toMatch(
      /if \(mediumDepth > 0u\) \{\s*incidentIor = mediumIor\[mediumDepth - 1u\];\s*\}/,
    );
    expect(PROBE_RAY_CAST_WGSL).toMatch(
      /if \(thinSheet\)[\s\S]*?etaTargetChannel,[\s\S]*?etaIncidentChannel,[\s\S]*?\} else \{\s*if \(entering\)/,
    );
    expect(PROBE_RAY_CAST_WGSL).not.toContain('thinSheet && mediumDepth > 0u');

    const waterToSheetToWater = thinSheetTransmittance(0.71, 1.33, 1.52);
    expect(waterToSheetToWater).toBeGreaterThan(0);
    expect(waterToSheetToWater).toBeLessThanOrEqual(1);
  });

  it('fails closed instead of popping an out-of-order nested bulk medium', () => {
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'if (rcLoadTriMaterialId(mediumTri[top]) != matId) { return 0.0; }',
    );
    expect(PROBE_RAY_CAST_WGSL).toMatch(
      /if \(!thinSheet && !entering\)[\s\S]*?rcLoadTriMaterialId\(mediumTri\[top\]\) != matId[\s\S]*?targetIor/,
    );
  });

  it('pairs TLAS exits with the exact pushed instance while merged BVH remains material-based', () => {
    expect(PROBE_RAY_CAST_WGSL).toContain(
      'mediumInstance[mediumDepth] = hit.instanceIndex;',
    );
    expect(PROBE_RAY_CAST_WGSL).toMatch(
      /rcLoadTriMaterialId\(mediumTri\[top\]\) != matId[\s\S]*?u\.bvhMode == 1u && mediumInstance\[top\] != hit\.instanceIndex/,
    );

    expect(nestedExitMatchesTop(1, 7, 7, 3, 3)).toBe(true);
    expect(nestedExitMatchesTop(1, 7, 7, 3, 4)).toBe(false);
    expect(nestedExitMatchesTop(1, 7, 8, 3, 3)).toBe(false);
    expect(nestedExitMatchesTop(0, 7, 7, 3, 4)).toBe(true);
  });

  it('closes a tilted parallel slab and fails total internal reflection', () => {
    const incident = normalize([0, 0, -1]);
    const normal = normalize([0.342, 0, 0.94]);
    const inside = refract(incident, normal, 1 / 1.52);
    expect(inside).not.toBeNull();
    const exit = refract(inside!, normal, 1.52);
    expect(exit).not.toBeNull();
    expect(distance(inside!, incident)).toBeGreaterThan(0.05);
    expect(distance(exit!, incident)).toBeLessThan(1e-10);

    const sixtyDegreesInside = normalize([
      Math.sin(Math.PI / 3), 0, -Math.cos(Math.PI / 3),
    ]);
    expect(refract(sixtyDegreesInside, [0, 0, 1], 1.52)).toBeNull();
  });
});
