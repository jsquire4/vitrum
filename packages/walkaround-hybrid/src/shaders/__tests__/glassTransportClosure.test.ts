import { describe, expect, it } from 'vitest';
import { REFRACTIVE_CAUSTICS_WGSL } from '../refractiveCaustics.wgsl.js';
import {
  RIS_GI_GLASS_RESERVOIR_LOOP_WGSL,
  RIS_GI_GLASS_TRANSPORT_PREFIX_WGSL,
} from '../risGiGlassWalk.wgsl.js';
import { RIS_GI_WGSL } from '../risGi.wgsl.js';
import { RIS_GI_NRC_BODY } from '../risGiNrc.wgsl.js';
import { SHADE_WGSL } from '../shade.wgsl.js';
import { SHADING_TERMS_WGSL } from '../shadingTerms.wgsl.js';

type Vec3 = readonly [number, number, number];

function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function normalize(v: Vec3): Vec3 {
  const length = Math.sqrt(dot(v, v));
  return [v[0] / length, v[1] / length, v[2] / length];
}

/** CPU oracle for WGSL refract(I, N, eta). Null is total internal reflection. */
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

function composeDirectWithProducerOwnedPrefix(
  localSurfaceRadiance: Vec3,
  transmittedPrefixRadiance: Vec3,
  faceLayerTransmission: Vec3,
): Vec3 {
  return [
    localSurfaceRadiance[0] * faceLayerTransmission[0] + transmittedPrefixRadiance[0],
    localSurfaceRadiance[1] * faceLayerTransmission[1] + transmittedPrefixRadiance[1],
    localSurfaceRadiance[2] * faceLayerTransmission[2] + transmittedPrefixRadiance[2],
  ];
}

type InterfaceEvent = 'bulk' | 'thin' | 'opaque' | 'miss';

/** Mirrors the four-interface budget plus terminal-query contract. */
function escapesWithinBudget(events: readonly InterfaceEvent[]): boolean {
  let interfaces = 0;
  for (const event of events) {
    if (event === 'miss') return interfaces > 0;
    if (event === 'opaque') return false;
    const cost = event === 'thin' ? 2 : 1;
    if (interfaces + cost > 4) return false;
    interfaces += cost;
  }
  return false;
}

describe('bounded hybrid dielectric transport closure', () => {
  it('shares the same real per-interface walk between regular and NRC GI', () => {
    for (const source of [RIS_GI_WGSL, RIS_GI_NRC_BODY]) {
      expect(source).toContain('const GLASS_WALK_MAX_INTERFACES: u32 = 4u;');
      expect(source).toContain('var mediumDepth: u32 = 0u;');
      expect(source).toContain('dielectricInterfaceTransmissionRgb(');
      expect(source).toContain('materialSpectralAttenuation(');
      expect(source).toContain('ggxSampleDielectricTransmission(');
      expect(source).toContain('let nextDir = interfaceBtdf.direction;');
      expect(source).toContain('applyNormalMapForHit(walkHit, walkSmoothNormal)');
      expect(source).not.toContain('GLASS_GI_MAX_ROUGHNESS');
      expect(source).not.toContain('straight-through approximation');
      expect(source).not.toContain('no secondary refraction');
    }
    expect(RIS_GI_GLASS_TRANSPORT_PREFIX_WGSL).toContain(
      'walkHit.dist / mediumThickness[top]',
    );
    expect(RIS_GI_GLASS_RESERVOIR_LOOP_WGSL).toContain(
      'Lo_g = Lo_g * glassPathThroughput;',
    );
  });

  it('does not apply the camera-prefix Fresnel/Beer throughput twice in shade', () => {
    expect(SHADING_TERMS_WGSL).toContain(
      'camera-side dielectric throughput is already present in g.Lo',
    );
    expect(SHADING_TERMS_WGSL).not.toContain(
      'grisVisibility * fresnelT * beerAlbedo',
    );
  });

  it('does not apply the primary face layer twice to transmitted GI', () => {
    expect(RIS_GI_GLASS_TRANSPORT_PREFIX_WGSL).toContain(
      'faceLayerTransmission(primaryLayer)',
    );
    expect(RIS_GI_GLASS_RESERVOIR_LOOP_WGSL).toContain(
      'Lo_g = Lo_g * glassPathThroughput;',
    );
    expect(SHADE_WGSL).toContain(
      ') * layerTransmission + Lo_transmittedGI,',
    );
    expect(SHADE_WGSL).not.toMatch(
      /Lo_transmittedGI[\s\S]*?\)\s*\*\s*layerTransmission,/,
    );

    const prefixComplete: Vec3 = [0.8, 0.6, 0.4];
    const composed = composeDirectWithProducerOwnedPrefix(
      [0, 0, 0],
      prefixComplete,
      [0.25, 0.5, 0.75],
    );
    expect(composed).toEqual(prefixComplete);
  });

  it('pays exact interface Fresnel in the independent caustic walk', () => {
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain(
      'dielectricInterfaceTransmissionRgb(',
    );
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain(
      'interfaceWeight * exitWeight * transmission',
    );
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain(
      'applyNormalMapForHit(hit, interfaceSmoothNormal)',
    );
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain('ggxSampleDielectricTransmission(');
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain('faceLayerTransmission(layerControls)');
    expect(REFRACTIVE_CAUSTICS_WGSL).not.toContain('materialRm.x > 0.0001');
  });

  it('routes no-glass escapes through the direct-sun baseline residual', () => {
    expect(REFRACTIVE_CAUSTICS_WGSL).toMatch(
      /if \(out\.sawGlass == 0u\) \{\s*out\.eligible = 0u;/,
    );
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain(
      'if (path.eligible == 0u)',
    );
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain(
      'let fallback = refractiveCausticChannel(baseline, channel) * 0.5;',
    );
  });

  it('reserves a terminal query after exactly four dielectric interfaces', () => {
    expect(RIS_GI_GLASS_TRANSPORT_PREFIX_WGSL).toContain(
      'gi <= GLASS_WALK_MAX_INTERFACES',
    );
    expect(RIS_GI_GLASS_TRANSPORT_PREFIX_WGSL).toContain(
      'interfaceCount + interfaceCost > GLASS_WALK_MAX_INTERFACES',
    );
    expect(RIS_GI_GLASS_TRANSPORT_PREFIX_WGSL).toContain(
      'select(1u, 2u, walkThickness <= 0.0)',
    );
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain('depth <= 4u');
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain(
      'interfaceCount + interfaceCost > 4u',
    );
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain(
      'select(1u, 2u, materialThickness <= 0.0)',
    );
    expect(escapesWithinBudget(['bulk', 'bulk', 'bulk', 'bulk', 'miss'])).toBe(true);
    expect(escapesWithinBudget(['bulk', 'bulk', 'bulk', 'bulk', 'bulk', 'miss'])).toBe(false);
    expect(escapesWithinBudget(['thin', 'thin', 'miss'])).toBe(true);
    expect(escapesWithinBudget(['thin', 'thin', 'thin', 'miss'])).toBe(false);
  });

  it('uses the exact solid-angle proposal weight without a hidden firefly clamp', () => {
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain(
      'ubo.sunAngular.x <= 1.5707963268',
    );
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain(
      'let jacobianWeight = omegaSearch / omegaSun;',
    );
    expect(REFRACTIVE_CAUSTICS_WGSL).not.toContain(
      'min(32.0, omegaSearch / omegaSun)',
    );
  });

  it('refracts back out of a tilted parallel slab instead of keeping the entry direction', () => {
    const incident = normalize([0, 0, -1]);
    const orientedBoundaryNormal = normalize([0.342, 0, 0.94]);
    const ior = 1.52;
    const inside = refract(incident, orientedBoundaryNormal, 1 / ior);
    expect(inside).not.toBeNull();
    const exit = refract(inside!, orientedBoundaryNormal, ior);
    expect(exit).not.toBeNull();

    // The entry solve bends materially; a straight-through exit would retain
    // this wrong direction. The reciprocal exit solve restores the incident
    // direction for a parallel slab (with a lateral position offset only).
    expect(distance(inside!, incident)).toBeGreaterThan(0.05);
    expect(distance(exit!, incident)).toBeLessThan(1e-10);
  });

  it('fails the transmitted branch on total internal reflection', () => {
    const sixtyDegreesInside = normalize([
      Math.sin(Math.PI / 3),
      0,
      -Math.cos(Math.PI / 3),
    ]);
    expect(refract(sixtyDegreesInside, [0, 0, 1], 1.52)).toBeNull();
  });
});
