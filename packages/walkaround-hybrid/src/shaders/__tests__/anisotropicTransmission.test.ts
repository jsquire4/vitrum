import { describe, expect, it } from 'vitest';
import { GGX_BRDF_WGSL } from '../ggxBrdf.wgsl.js';
import { REFRACTIVE_CAUSTICS_WGSL } from '../refractiveCaustics.wgsl.js';

type Vec3 = readonly [number, number, number];

function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function normalize(v: Vec3): Vec3 {
  const length = Math.sqrt(dot(v, v));
  return [v[0] / length, v[1] / length, v[2] / length];
}

function refract(i: Vec3, n: Vec3, eta: number): Vec3 | null {
  const nDotI = dot(n, i);
  const k = 1 - eta * eta * (1 - nDotI * nDotI);
  if (k < 0) return null;
  const scale = eta * nDotI + Math.sqrt(k);
  return normalize([
    eta * i[0] - scale * n[0],
    eta * i[1] - scale * n[1],
    eta * i[2] - scale * n[2],
  ]);
}

function rotateFrame(rotation: number): readonly [Vec3, Vec3] {
  const c = Math.cos(rotation);
  const s = Math.sin(rotation);
  return [[c, s, 0], [-s, c, 0]];
}

function anisotropyAxes(roughness: number, anisotropy: number): readonly [number, number] {
  const alpha = Math.max(roughness * roughness, 1e-6);
  const aspect = Math.sqrt(1 - 0.9 * Math.min(1, Math.max(0, anisotropy)));
  return [alpha / aspect, alpha * aspect];
}

function anisotropicDistribution(
  h: Vec3,
  tangent: Vec3,
  bitangent: Vec3,
  alphaX: number,
  alphaY: number,
): number {
  const nDotH = h[2];
  const denom = (dot(tangent, h) / alphaX) ** 2
    + (dot(bitangent, h) / alphaY) ** 2
    + nDotH * nDotH;
  return 1 / (Math.PI * alphaX * alphaY * denom * denom);
}

function transmissionPdfForMicrofacet(
  wm: Vec3,
  rotation: number,
  roughness: number,
  anisotropy: number,
): number {
  const wo: Vec3 = [0, 0, 1];
  const wi = refract([0, 0, -1], wm, 1 / 1.5);
  if (wi === null) return 0;
  const [tangent, bitangent] = rotateFrame(rotation);
  const [alphaX, alphaY] = anisotropyAxes(roughness, anisotropy);
  const d = anisotropicDistribution(wm, tangent, bitangent, alphaX, alphaY);
  const woDotM = dot(wo, wm);
  const wiDotM = dot(wi, wm);
  const denom = wiDotM + woDotM / 1.5;
  const microfacetPdf = d * woDotM; // G1(wo)==NdotWo==1 at normal incidence.
  return microfacetPdf * Math.abs(wiDotM) / (denom * denom);
}

describe('anisotropic rough-dielectric transmission', () => {
  it('exposes one authored-frame VNDF sample/eval/pdf family', () => {
    expect(GGX_BRDF_WGSL).toContain('fn ggxSampleVndfTangentAnisotropic(');
    expect(GGX_BRDF_WGSL).toContain(
      'fn ggxDielectricTransmissionPdfAnisotropyFrame(',
    );
    expect(GGX_BRDF_WGSL).toContain(
      'fn evalGgxDielectricTransmissionAnisotropyFrame(',
    );
    expect(GGX_BRDF_WGSL).toContain(
      'fn ggxSampleDielectricTransmissionAnisotropyFrame(',
    );
    expect(GGX_BRDF_WGSL).toContain(
      'let dWmDWi = abs(wiDotM) / (denom * denom);',
    );
  });

  it('delegates the isotropic and exact-delta endpoints to the established helper', () => {
    const start = GGX_BRDF_WGSL.indexOf(
      'fn ggxSampleDielectricTransmissionAnisotropyFrame(',
    );
    const end = GGX_BRDF_WGSL.indexOf('\n// VNDF reflection PDF', start);
    const body = GGX_BRDF_WGSL.slice(start, end);
    expect(body).toContain('if (aniso <= 0.0 || rough <= 0.0)');
    expect(body).toContain('return ggxSampleDielectricTransmission(');
  });

  it('rotates the anisotropic transmission density with the authored frame', () => {
    const tilt = 0.18;
    const tangentTilted = normalize([Math.sin(tilt), 0, Math.cos(tilt)]);
    const bitangentTilted = normalize([0, Math.sin(tilt), Math.cos(tilt)]);
    const tangentDensity = transmissionPdfForMicrofacet(
      tangentTilted, 0, 0.45, 0.8,
    );
    const bitangentDensity = transmissionPdfForMicrofacet(
      bitangentTilted, 0, 0.45, 0.8,
    );
    const rotatedTangentDensity = transmissionPdfForMicrofacet(
      tangentTilted, Math.PI / 2, 0.45, 0.8,
    );

    expect(tangentDensity).toBeGreaterThan(bitangentDensity);
    expect(rotatedTangentDensity).toBeCloseTo(bitangentDensity, 10);
    expect(Number.isFinite(tangentDensity)).toBe(true);
    expect(tangentDensity).toBeGreaterThan(0);
  });

  it('routes both caustic interfaces through sampled controls and the authored frame', () => {
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain(
      'let interfaceAnisotropy = sampleAnisotropyControls(hit);',
    );
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain(
      'MATERIAL_MAP_ANISOTROPY_TEXEL_OFFSET',
    );
    expect(
      REFRACTIVE_CAUSTICS_WGSL.match(
        /ggxSampleDielectricTransmissionAnisotropyFrame\(/g,
      ),
    ).toHaveLength(2);
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain('interfaceAnisotropy.x');
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain('interfaceAnisotropy.y');
  });
});
