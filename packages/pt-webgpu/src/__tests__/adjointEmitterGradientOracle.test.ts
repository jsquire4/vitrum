import { describe, expect, it } from 'vitest';
import {
  evaluateBrdf,
  evaluateBrdfWithAnisotropy,
  evaluateBrdfWithClearcoat,
  evaluateBrdfWithSheen,
  type Vec3,
} from '../inverse/brdfAdjoint.js';
import { PT_WEBGPU_ADJOINT_PASS_WGSL } from '../wgsl/pathTrace/adjointPass.wgsl.js';

const H = 1e-4;

const dot = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mul = (a: Vec3, b: Vec3): Vec3 => [a[0] * b[0], a[1] * b[1], a[2] * b[2]];
const scale = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
const perturb = (a: Vec3, channel: number, delta: number): Vec3 => [
  a[0] + (channel === 0 ? delta : 0),
  a[1] + (channel === 1 ? delta : 0),
  a[2] + (channel === 2 ? delta : 0),
];

function norm(v: Vec3): Vec3 {
  const l = Math.hypot(v[0], v[1], v[2]);
  return [v[0] / l, v[1] / l, v[2] / l];
}

const material = {
  baseColor: [0.55, 0.28, 0.14] as Vec3,
  roughness: 0.42,
  metallic: 0.18,
  specularColor: [0.75, 0.9, 1.0] as Vec3,
  specularIntensity: 0.82,
  clearcoat: 0.31,
  clearcoatRoughness: 0.22,
  sheen: 0.27,
  sheenRoughness: 0.48,
  sheenColor: [0.35, 0.18, 0.08] as Vec3,
  anisotropy: 0.36,
  anisotropyRotation: 0.4,
};

const normal: Vec3 = [0, 0, 1];
const wo = norm([0.18, -0.12, 1.0]);
const wi = norm([-0.32, 0.24, 1.0]);
const dLoss: Vec3 = [1.4, -0.25, 0.7];
const color: Vec3 = [0.8, 0.45, 0.2];
const intensity = 2.25;

function directLightBrdfValue(): Vec3 {
  const baseIso = evaluateBrdf(
    material.baseColor,
    material.roughness,
    material.metallic,
    normal,
    wo,
    wi,
    material.specularColor,
    material.specularIntensity,
  );
  const anisotropicBase = evaluateBrdfWithAnisotropy(
    material.baseColor,
    material.roughness,
    material.metallic,
    normal,
    wo,
    wi,
    material.anisotropy,
    material.anisotropyRotation,
    material.specularColor,
    material.specularIntensity,
  );
  const clearcoatOnly = sub(
    evaluateBrdfWithClearcoat(
      material.baseColor,
      material.roughness,
      material.metallic,
      normal,
      wo,
      wi,
      material.clearcoat,
      material.clearcoatRoughness,
      material.specularColor,
      material.specularIntensity,
    ),
    baseIso,
  );
  const sheenOnly = sub(
    evaluateBrdfWithSheen(
      material.baseColor,
      material.roughness,
      material.metallic,
      normal,
      wo,
      wi,
      material.sheen,
      material.sheenRoughness,
      material.sheenColor,
      material.specularColor,
      material.specularIntensity,
    ),
    baseIso,
  );
  return add(add(anisotropicBase, clearcoatOnly), sheenOnly);
}

function lossFromRadiance(radiance: Vec3, factor: number): number {
  const rendered = mul(directLightBrdfValue(), scale(radiance, factor));
  return dot(dLoss, rendered);
}

function deltaEmitterLoss(c: Vec3, i: number, factor: number): number {
  return lossFromRadiance(scale(c, i), factor);
}

function meshEmitterLoss(c: Vec3, i: number, mapScale: Vec3, factor: number): number {
  return lossFromRadiance(mul(scale(c, i), mapScale), factor);
}

describe('path-replay adjoint emitter gradients — independent CPU oracle', () => {
  const brdf = directLightBrdfValue();

  it.each([
    ['directional', Math.max(dot(normal, wi), 0)],
    ['point', Math.max(dot(normal, wi), 0) * 0.42],
    ['spot', Math.max(dot(normal, wi), 0) * 0.65 * 0.31],
    ['finite area', Math.max(dot(normal, wi), 0) / 3.7],
  ])('%s color/intensity partials match finite differences of L = f * G * color * intensity', (_label, factor) => {
    const expectedColor: Vec3 = [
      dLoss[0] * brdf[0] * factor * intensity,
      dLoss[1] * brdf[1] * factor * intensity,
      dLoss[2] * brdf[2] * factor * intensity,
    ];
    const expectedIntensity = dot(dLoss, mul(brdf, scale(color, factor)));

    for (let c = 0; c < 3; c++) {
      const plus = perturb(color, c, H);
      const minus = perturb(color, c, -H);
      const fd = (deltaEmitterLoss(plus, intensity, factor) - deltaEmitterLoss(minus, intensity, factor)) / (2 * H);
      expect(fd).toBeCloseTo(expectedColor[c]!, 6);
    }

    const fdIntensity = (
      deltaEmitterLoss(color, intensity + H, factor) -
      deltaEmitterLoss(color, intensity - H, factor)
    ) / (2 * H);
    expect(fdIntensity).toBeCloseTo(expectedIntensity, 6);
  });

  it('mapped mesh-area gradients use packedRadiance/color and packedRadiance/intensity quotients when nonzero', () => {
    const mapScale: Vec3 = [0.35, 0.8, 1.6];
    const factor = Math.max(dot(normal, wi), 0) / 2.9;
    const packedRadiance = mul(scale(color, intensity), mapScale);
    const dLossDPacked = mul(dLoss, scale(brdf, factor));
    const expectedColor: Vec3 = [
      dLossDPacked[0] * (packedRadiance[0] / color[0]),
      dLossDPacked[1] * (packedRadiance[1] / color[1]),
      dLossDPacked[2] * (packedRadiance[2] / color[2]),
    ];
    const expectedIntensity = dot(dLossDPacked, scale(packedRadiance, 1 / intensity));

    for (let c = 0; c < 3; c++) {
      const plus = perturb(color, c, H);
      const minus = perturb(color, c, -H);
      const fd = (meshEmitterLoss(plus, intensity, mapScale, factor) - meshEmitterLoss(minus, intensity, mapScale, factor)) / (2 * H);
      expect(fd).toBeCloseTo(expectedColor[c]!, 6);
    }

    const fdIntensity = (
      meshEmitterLoss(color, intensity + H, mapScale, factor) -
      meshEmitterLoss(color, intensity - H, mapScale, factor)
    ) / (2 * H);
    expect(fdIntensity).toBeCloseTo(expectedIntensity, 6);
  });

  it('the production shader keeps emitter gradients on the closed-form radiance chain', () => {
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('fn scatterEmitterRadianceGradient');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('dPackedRadiance_dColor = packedRadiance / emitterColor;');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('dPackedRadiance_dIntensity = packedRadiance / emitterIntensity;');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('dLoss_dR * brdfValue * (nDotL * attenuation)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('dLoss_dR * brdfValue * (nDotL * areaFactor)');
  });
});
