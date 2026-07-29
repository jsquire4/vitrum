import { describe, expect, it } from 'vitest';
import { makeProbeUpdateRaysWGSL } from '../../ddgi/wgsl/probeUpdateRays.wgsl.js';
import { GGX_BRDF_WGSL } from '../ggxBrdf.wgsl.js';
import { MATERIAL_ATLAS_WGSL } from '../materialAtlas.wgsl.js';

type Vec3 = readonly [number, number, number];

const PI = Math.PI;
const MIN_CONTINUOUS_GGX_ROUGHNESS = 0.01;

function materialF0(
  albedo: Vec3,
  metal: number,
  specularColor: Vec3,
  specularIntensity: number,
): Vec3 {
  const dielectric = specularColor.map((component) =>
    Math.max(0, component) *
    Math.min(1, Math.max(0, specularIntensity)),
  );
  const metallic = Math.min(1, Math.max(0, metal));
  return dielectric.map(
    (component, index) =>
      component * (1 - metallic) + albedo[index]! * metallic,
  ) as unknown as Vec3;
}

function boundedRoughness(roughness: number): number {
  return Math.max(
    MIN_CONTINUOUS_GGX_ROUGHNESS,
    Math.min(1, Math.max(0, roughness)),
  );
}

function distributionGgx(nDotH: number, roughness: number): number {
  if (!(nDotH > 0)) return 0;
  const bounded = boundedRoughness(roughness);
  const alpha = bounded * bounded;
  const alpha2 = alpha * alpha;
  const nDotH2 = nDotH * nDotH;
  const denominator = Math.max(0, 1 - nDotH2) + nDotH2 * alpha2;
  return alpha2 / (PI * denominator * denominator);
}

function smithG1(nDotV: number, alpha2: number): number {
  if (!(nDotV > 0)) return 0;
  return (2 * nDotV) /
    (nDotV + Math.sqrt(alpha2 + (1 - alpha2) * nDotV * nDotV));
}

function alignedGgxContribution(
  roughness: number,
  f0: number,
): number {
  const bounded = boundedRoughness(roughness);
  const alpha = bounded * bounded;
  const alpha2 = alpha * alpha;
  const d = distributionGgx(1, bounded);
  const g = smithG1(1, alpha2) ** 2;
  return d * g * f0 / 4;
}

function charlieD(nDotH: number, alpha: number): number {
  const invAlpha = 1 / alpha;
  const sinThetaH = Math.sqrt(Math.max(0, 1 - nDotH * nDotH));
  return (2 + invAlpha) * sinThetaH ** invAlpha / (2 * PI);
}

describe('default material lobe energy', () => {
  it('consumes the atlas absolute-F0 representation', () => {
    expect(materialF0([0.8, 0.7, 0.6], 0, [0.04, 0.04, 0.04], 1)).toEqual([
      0.04,
      0.04,
      0.04,
    ]);

    expect(GGX_BRDF_WGSL).toContain('max(specularColor, vec3f(0.0))');
    expect(GGX_BRDF_WGSL).not.toContain('absoluteThinFilmF0');
  });

  it('represents thin-film reflectance directly without a numeric sentinel', () => {
    const nearZero: Vec3 = [1e-8, 0, 0.25];
    const decodedNearZero = materialF0([0, 0, 0], 0, nearZero, 1);
    decodedNearZero.forEach((component, index) => {
      expect(component).toBeCloseTo(nearZero[index]!, 12);
    });

    const unit: Vec3 = [1, 1, 1];
    expect(materialF0([0, 0, 0], 0, unit, 1)).toEqual(unit);

    expect(MATERIAL_ATLAS_WGSL).toContain(
      'payload.specular = vec4f(film.reflectance, 1.0)',
    );
    const ddgi = makeProbeUpdateRaysWGSL(8);
    expect(
      ddgi.match(/vec4f\(film\.reflectance, 1\.0\)/g),
    ).toHaveLength(2);
    expect(ddgi).not.toContain('mat.specular.rgb >= vec3f(2.0)');
    expect(ddgi).toContain(
      'let dielectricF0 = ddgiProbeDielectricF0(mat);',
    );
  });

  it('gives authored roughness zero a finite non-black reflection lobe', () => {
    const contribution = alignedGgxContribution(0, 0.04);
    expect(Number.isFinite(contribution)).toBe(true);
    expect(contribution).toBeGreaterThan(0);

    expect(GGX_BRDF_WGSL).toContain(
      'const MIN_CONTINUOUS_GGX_ROUGHNESS: f32 = 0.01;',
    );
    expect(GGX_BRDF_WGSL).toContain(
      'let d = max(0.0, 1.0 - nDotH2) + nDotH2 * alpha2;',
    );
  });

  it('keeps glTF-default clearcoat roughness zero active when clearcoat is enabled', () => {
    const contribution = alignedGgxContribution(0, 0.04);
    expect(Number.isFinite(contribution)).toBe(true);
    expect(contribution).toBeGreaterThan(0);

    const start = GGX_BRDF_WGSL.indexOf('fn evalClearcoatLobe(');
    const end = GGX_BRDF_WGSL.indexOf('\n// KHR_materials_sheen', start);
    const body = GGX_BRDF_WGSL.slice(start, end);
    expect(body).toContain(
      'let rough = boundedContinuousGgxRoughness(clearcoatRoughness);',
    );
    expect(body).not.toContain('rough <= 0.0');
  });

  it('keeps glTF-default sheen roughness zero active when sheen is enabled', () => {
    const alpha = Math.max(0 * 0, 1e-3);
    const nDotL = Math.SQRT1_2;
    const nDotV = Math.SQRT1_2;
    const visibility = 1 / (4 * (nDotL + nDotV - nDotL * nDotV));
    const contribution = charlieD(0, alpha) * visibility * nDotL;
    expect(Number.isFinite(contribution)).toBe(true);
    expect(contribution).toBeGreaterThan(0);

    const start = GGX_BRDF_WGSL.indexOf('fn evalSheenLobe(');
    const end = GGX_BRDF_WGSL.indexOf(
      '\nfn evalGGXWithSpecularClearcoatSheen(',
      start,
    );
    const body = GGX_BRDF_WGSL.slice(start, end);
    expect(body).toContain(
      'let alpha = max(sheenRough * sheenRough, 1.0e-3);',
    );
    expect(body).not.toContain('sheenRough <= 0.0');
  });
});
