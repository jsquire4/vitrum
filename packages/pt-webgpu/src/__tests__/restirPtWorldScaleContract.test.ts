import { describe, expect, it } from 'vitest';

import { RESERVOIR_PT_HERO_WGSL } from '../wgsl/pathTrace/reservoirPtHero.wgsl.js';
import { RESTIR_PT_RESOLVE_WGSL } from '../wgsl/pathTrace/restirPtResolve.wgsl.js';
import { RESTIR_PT_SPATIAL_WGSL } from '../wgsl/pathTrace/restirPtSpatial.wgsl.js';
import { RESTIR_PT_TEMPORAL_WGSL } from '../wgsl/pathTrace/restirPtTemporal.wgsl.js';

function executableSource(source: string): string {
  return source
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/\/\/.*$/gm, '');
}

function relativeCoordinate(value: number, center: number, extent: number): number {
  const scale = Math.max(Math.abs(value), Math.abs(center), Math.abs(extent));
  if (!(scale > 0) || !Number.isFinite(scale) || !(extent > 0)) return 0;
  const denominator = extent / scale;
  if (!(denominator > 0)) return 0;
  return Math.max(-1, Math.min(1, (value / scale - center / scale) / denominator));
}

function rayEpsilon(
  origin: readonly [number, number, number],
  segmentLength: number,
): number {
  const scale = Math.max(
    Math.abs(origin[0]),
    Math.abs(origin[1]),
    Math.abs(origin[2]),
    segmentLength,
  );
  return scale * 2 ** -18;
}

describe('ReSTIR-PT world-scale contract', () => {
  it('normalizes analytic identity from exact authored extents without a world-unit floor', () => {
    const source = executableSource(RESERVOIR_PT_HERO_WGSL);
    expect(source).toContain('extent = vec3f(p0.w)');
    expect(source).toContain('extent = abs(p1.xyz)');
    expect(source).toContain('extent = vec3f(p0.w, p1.x, p0.w)');
    expect(source).toContain('rptRelativeCoordinate(localPoint.x, center.x, extent.x)');
    expect(source).not.toContain('max(p0.w, 1e-4)');
    expect(source).not.toContain('vec3f(1e-4)');

    for (const scale of [1e-30, 1e30]) {
      expect(relativeCoordinate(2 * scale, scale, 2 * scale)).toBeCloseTo(0.5, 14);
      expect(relativeCoordinate(-scale, scale, 2 * scale)).toBeCloseTo(-1, 14);
    }
  });

  it('derives reconnection ray bounds from f32-relevant magnitudes', () => {
    const reservoir = executableSource(RESERVOIR_PT_HERO_WGSL);
    expect(reservoir).toContain('!rptFinitePositive(reconScale)');
    expect(reservoir).not.toContain('dRecon <= 1e-6');
    const temporal = executableSource(RESTIR_PT_TEMPORAL_WGSL);
    const spatial = executableSource(RESTIR_PT_SPATIAL_WGSL);
    for (const source of [temporal, spatial]) {
      expect(source).toContain('rptWorldRayEpsilon(xv, dist)');
      expect(source).toContain('rptWorldRayEpsilon(xs, remainingDistance)');
      expect(source).toContain('safe_normalize(remaining)');
      expect(source).not.toContain('1e-3');
      expect(source).not.toContain('1e-4');
      expect(source).not.toContain('2e-3');
    }

    const tiny = rayEpsilon([1e-30, -2e-30, 0], 4e-30);
    const ordinary = rayEpsilon([1, -2, 0], 4);
    const huge = rayEpsilon([1e30, -2e30, 0], 4e30);
    expect(tiny / ordinary).toBeCloseTo(1e-30, 12);
    expect(huge / ordinary).toBeCloseTo(1e30, 12);
  });

  it('uses a dimensionless local-path coplanarity threshold', () => {
    const source = executableSource(RESTIR_PT_SPATIAL_WGSL);
    expect(source).toContain('RPT_SPATIAL_COPLANAR_REL_TOL');
    expect(source).toContain('planeDist > localPathScale * RPT_SPATIAL_COPLANAR_REL_TOL');
    expect(source).not.toContain('RPT_SPATIAL_COPLANAR_TOL');
  });

  it('accepts every finite positive geometric and target contribution', () => {
    const temporal = executableSource(RESTIR_PT_TEMPORAL_WGSL);
    const spatial = executableSource(RESTIR_PT_SPATIAL_WGSL);
    const resolve = executableSource(RESTIR_PT_RESOLVE_WGSL);
    expect(temporal).not.toContain('clip.w <= 1e-6');
    expect(temporal).not.toContain('pHatPrev_atCur >= 1e-9');
    expect(spatial).not.toContain('pHatQ_native >= 1e-9');
    expect(resolve).not.toContain('dist2 < 1e-8');
    expect(resolve).not.toContain('cosTheta <= 1e-6');
    expect(resolve).toContain('let distance = rptScaledLength(toS)');
  });
});
