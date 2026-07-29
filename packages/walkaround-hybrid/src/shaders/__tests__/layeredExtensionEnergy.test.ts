import { describe, expect, it } from 'vitest';
import { GGX_BRDF_WGSL } from '../ggxBrdf.wgsl.js';

type Vec3 = readonly [number, number, number];

function schlick(cosTheta: number, f0: number): number {
  const m = Math.min(1, Math.max(0, 1 - cosTheta));
  return f0 + (1 - f0) * m ** 5;
}

function clearcoatBaseAttenuation(
  clearcoat: number,
  viewDotClearcoatNormal: number,
): number {
  const layerWeight =
    Math.min(1, Math.max(0, clearcoat)) *
    schlick(Math.abs(viewDotClearcoatNormal), 0.04);
  return Math.min(1, Math.max(0, 1 - layerWeight));
}

function sheenDirectionalAlbedo(cosThetaRaw: number, alpha: number): number {
  const cosTheta = Math.min(1, Math.max(0, cosThetaRaw));
  const c = 1 - cosTheta;
  return (
    0.65584461 * c ** 3 +
    1 /
      (4.16526551 +
        Math.exp(-7.97291361 * Math.sqrt(alpha) + 6.33516894))
  );
}

function sheenBaseAttenuation(
  sheen: number,
  sheenRoughness: number,
  maxSheenColor: number,
  nDotV: number,
  nDotL: number,
): number {
  const rough = Math.max(sheenRoughness, 0.07);
  const alpha = rough * rough;
  const fullSheenScale = Math.min(
    1 -
      Math.min(1, Math.max(0, maxSheenColor)) *
        sheenDirectionalAlbedo(Math.abs(nDotV), alpha),
    1 -
      Math.min(1, Math.max(0, maxSheenColor)) *
        sheenDirectionalAlbedo(Math.abs(nDotL), alpha),
  );
  const factor = Math.min(1, Math.max(0, sheen));
  return Math.min(
    1,
    Math.max(0, 1 * (1 - factor) + fullSheenScale * factor),
  );
}

function orderedLayerResponse(
  base: Vec3,
  sheenLobe: Vec3,
  clearcoatLobe: Vec3,
  sheenAttenuation: number,
  clearcoatAttenuation: number,
): Vec3 {
  return base.map(
    (channel, index) =>
      (channel * sheenAttenuation + sheenLobe[index]!) *
        clearcoatAttenuation +
      clearcoatLobe[index]!,
  ) as unknown as Vec3;
}

function functionBody(source: string, name: string): string {
  const start = source.indexOf(`fn ${name}(`);
  if (start < 0) throw new Error(`Missing WGSL function ${name}`);
  const bodyStart = source.indexOf('{', start);
  let depth = 0;
  for (let index = bodyStart; index < source.length; index += 1) {
    if (source[index] === '{') depth += 1;
    if (source[index] === '}') {
      depth -= 1;
      if (depth === 0) return source.slice(bodyStart, index + 1);
    }
  }
  throw new Error(`Unterminated WGSL function ${name}`);
}

describe('walkaround KHR clearcoat/sheen layer energy', () => {
  it('uses the authored clearcoat normal for the shared coat reflection/loss weight', () => {
    expect(clearcoatBaseAttenuation(1, 1)).toBeCloseTo(0.96, 12);
    expect(clearcoatBaseAttenuation(1, 0.2)).toBeCloseTo(0.6454272, 12);
    expect(clearcoatBaseAttenuation(0, 0.2)).toBe(1);

    const clearcoat = functionBody(GGX_BRDF_WGSL, 'evalClearcoatLobe');
    const attenuation = functionBody(
      GGX_BRDF_WGSL,
      'clearcoatBaseAttenuation',
    );
    expect(clearcoat).toContain(
      'clearcoatLayerWeight(cc, n, wo)',
    );
    expect(attenuation).toContain(
      'clearcoatLayerWeight(clearcoat, clearcoatNormal, wo)',
    );
  });

  it('matches the Estevez-Kulla sheen directional-albedo attenuation numerically', () => {
    const attenuation = sheenBaseAttenuation(0.75, 0.5, 0.8, 0.6, 0.4);
    expect(attenuation).toBeCloseTo(0.8740114839579394, 12);
    expect(sheenBaseAttenuation(0, 0.5, 0.8, 0.6, 0.4)).toBe(1);
  });

  it('attenuates base then sheen below clearcoat in both direct and specular-GI evaluators', () => {
    const sheenAttenuation = sheenBaseAttenuation(
      0.75,
      0.5,
      0.8,
      0.6,
      0.4,
    );
    const clearcoatAttenuation = clearcoatBaseAttenuation(1, 0.2);
    const base: Vec3 = [0.8, 0.5, 0.2];
    const sheenLobe: Vec3 = [0.15, 0.08, 0.04];
    const clearcoatLobe: Vec3 = [0.09, 0.09, 0.09];
    const layered = orderedLayerResponse(
      base,
      sheenLobe,
      clearcoatLobe,
      sheenAttenuation,
      clearcoatAttenuation,
    );
    const naive = base.map(
      (channel, index) =>
        channel + sheenLobe[index]! + clearcoatLobe[index]!,
    );
    expect(layered).toEqual([
      expect.closeTo(0.6381027078870541, 12),
      expect.closeTo(0.42368956842940886, 12),
      expect.closeTo(0.22863924497176355, 12),
    ]);
    layered.forEach((channel, index) => {
      expect(channel).toBeLessThan(naive[index]!);
    });

    for (const name of [
      'evalGGXWithSpecularClearcoatSheenWithAnisotropyFrame',
      'evalGGXSpecularOnlyWithSpecularClearcoatSheenWithAnisotropyFrame',
    ]) {
      const body = functionBody(GGX_BRDF_WGSL, name);
      expect(body).toContain('let sheenAttenuation = sheenBaseAttenuation(');
      expect(body).toContain(
        'let clearcoatAttenuation = clearcoatBaseAttenuation(',
      );
      expect(body).toContain(
        'sheenLobe) *\n    clearcoatAttenuation + clearcoatLobe',
      );
    }
  });
});
