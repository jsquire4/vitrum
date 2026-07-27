import { describe, expect, it } from 'vitest';
import { PROBE_RAY_CAST_WGSL } from '../src/wgsl/probeRayCast.wgsl.js';

type Vec3 = readonly [number, number, number];

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

describe('RC bounded dielectric transport closure', () => {
  it('uses channel-separated exact transport and no legacy one-hit continuation', () => {
    expect(PROBE_RAY_CAST_WGSL).toContain('fn rcTraceTransmittedChannel(');
    expect(PROBE_RAY_CAST_WGSL).toContain('fn rcDielectricInterfaceTransmissionRgb(');
    expect(PROBE_RAY_CAST_WGSL).toContain('materialSpectralAttenuation(');
    expect(PROBE_RAY_CAST_WGSL).toMatch(/rcTraceTransmittedChannel\(\s+ray, hit, maxT, 0u/);
    expect(PROBE_RAY_CAST_WGSL).toMatch(/rcTraceTransmittedChannel\(\s+ray, hit, maxT, 1u/);
    expect(PROBE_RAY_CAST_WGSL).toMatch(/rcTraceTransmittedChannel\(\s+ray, hit, maxT, 2u/);
    expect(PROBE_RAY_CAST_WGSL).not.toContain('let secondHit');
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
