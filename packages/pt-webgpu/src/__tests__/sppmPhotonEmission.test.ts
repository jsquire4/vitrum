import { describe, expect, it } from 'vitest';
import { SPPM_PHOTON_PASS_WGSL } from '../wgsl/pathTrace/sppmBindings.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_KERNEL_WGSL } from '../wgsl/pathTrace/kernel.wgsl.js';
import { composeSppmPhotonPassWgsl } from '../wgsl/pathTraceBruteforce.wgsl.js';

function expectedSelectedSourcePower(sourcePower: number, sourceCount: number, photonCount: number): number {
  const pSelect = 1 / sourceCount;
  const perPhotonFlux = sourcePower / (photonCount * pSelect);
  return photonCount * pSelect * perPhotonFlux;
}

describe('SPPM photon emission source normalization (PTWG-03)', () => {
  it('keeps one-light and two-light expected flux equal for an unchanged source', () => {
    const sourcePower = 7.25;
    const photonCount = 65536;

    expect(expectedSelectedSourcePower(sourcePower, 1, photonCount)).toBeCloseTo(sourcePower, 12);
    expect(expectedSelectedSourcePower(sourcePower, 2, photonCount)).toBeCloseTo(sourcePower, 12);
    expect(expectedSelectedSourcePower(sourcePower, 8, photonCount)).toBeCloseTo(sourcePower, 12);
  });

  it('normalizes delta photon flux by 1 / p_select', () => {
    expect(SPPM_PHOTON_PASS_WGSL).toContain('let lightSelectInvPdf = f32(availableLightCount);');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('dIrrMean.rgb * diskArea * lightSelectInvPdf');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('pointLights[pointBase + 1u].rgb * (4.0 * PI) * lightSelectInvPdf');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('sradW.rgb * solidAngle * softness * lightSelectInvPdf');
  });

  it('excludes castShadow:false emitters from photon source selection', () => {
    // SPPM launches forward photon paths from light sources. For no-shadow
    // emitters, parity with the rest of pt-webgpu means keeping direct/camera
    // visibility but not seeding caustic/shadow transport photons.
    expect(SPPM_PHOTON_PASS_WGSL).toContain('fn sppmDirectionalCastsShadow(dirIdx: u32) -> bool');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('return directionalLights[dBase].w >= 0.0;');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('return pointLights[pointBase + 2u].z <= 0.5;');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('return spotLights[spotBase + 3u].z <= 0.5;');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('return rectAreaLights[rectBase].w <= 0.5;');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('return meshAreaLights[meshBase + 3u].w <= 0.5;');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('if (!sppmPointCastsShadow(pointIdx)) { continue; }');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('if (!sppmMeshAreaCastsShadow(meshIdx)) { continue; }');
  });

  it('uses the packed N-directional RGB records instead of the legacy scalar lightDir lane', () => {
    expect(SPPM_PHOTON_PASS_WGSL).toContain('var availableLightCount = 0u;');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('for (var dirIdx = 0u; dirIdx < params.directionalLightCount; dirIdx = dirIdx + 1u)');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('if (sppmDirectionalCastsShadow(dirIdx))');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('let dDirAD = directionalLights[dBase]');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('let dIrrMean = directionalLights[dBase + 1u]');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('photonDir    = -towardLightDir;');
    expect(SPPM_PHOTON_PASS_WGSL).not.toContain('vec3f(params.lightDir.w) * diskArea');
  });

  it('applies spot smooth-penumbra weighting to sampled spot photons', () => {
    expect(SPPM_PHOTON_PASS_WGSL).toContain('let cosInner = sradW.w;');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('let softness = smoothstep(cosMin, max(cosInner, cosMin + 1e-6), cosTheta);');
  });

  it('includes rect/disc, mesh-area, and environment sources in the same flat selection order as NEE', () => {
    expect(SPPM_PHOTON_PASS_WGSL).toContain('if (sppmRectAreaCastsShadow(rectIdx))');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('if (sppmMeshAreaCastsShadow(meshIdx))');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('if (hasEnvironmentMap() || params.environmentSun.w > 1e-6) {');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('availableLightCount = availableLightCount + 1u;');
  });

  it('emits rect/disc photons from the NEE packed source data and area conventions', () => {
    expect(SPPM_PHOTON_PASS_WGSL).toContain('let rectBase = rectIdx * 4u;');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('let rpos = rectAreaLights[rectBase].xyz;');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('let ru = rectAreaLights[rectBase + 1u].xyz;');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('let rv = rectAreaLights[rectBase + 2u].xyz;');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('let rshape = rectAreaLights[rectBase + 3u];');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('let isDisc = abs(rshape.w - 1.0) < 0.5;');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('sppmConcentricDiscSample(vec2f(xi1 * 2.0 - 1.0, xi2 * 2.0 - 1.0))');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('area = max(PI * dot(ru, ru), 1e-6);');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('area = max(4.0 * length(cross(ru, rv)), 1e-6);');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('let hemi = cosineHemisphereSample(&rng, lightNormal);');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('photonFlux   = rr * area * PI * lightSelectInvPdf');
  });

  it('emits mesh-area photons from the NEE packed triangle data and area convention', () => {
    expect(SPPM_PHOTON_PASS_WGSL).toContain('let meshBase = meshIdx * 4u;');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('let a = meshAreaLights[meshBase].xyz;');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('let b = meshAreaLights[meshBase + 1u].xyz;');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('let c = meshAreaLights[meshBase + 2u].xyz;');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('let mr = meshAreaLights[meshBase + 3u].rgb;');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('let area = max(0.5 * length(cross(b - a, c - a)), 1e-6);');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('photonFlux   = mr * area * PI * lightSelectInvPdf');
  });

  it('emits environment photons through the same NEE importance and PDF helpers', () => {
    expect(SPPM_PHOTON_PASS_WGSL).toContain('let envSample = sampleEnvironmentImportance(&rng);');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('envColor = sampleEnvironmentColor(envDir);');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('envPdf = max(environmentPdf(envDir), 1e-8);');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('photonFlux   = envColor * diskArea * lightSelectInvPdf /');
    expect(SPPM_PHOTON_PASS_WGSL).not.toContain('TODO(PTWG-03 area/env)');
  });

  it('composes the full connect helpers before the SPPM photon entry point', () => {
    const wgsl = composeSppmPhotonPassWgsl();
    const envSampler = wgsl.indexOf('fn sampleEnvironmentImportance(');
    const photonEntry = wgsl.indexOf('fn sppmEmitPhotons(');

    expect(envSampler).toBeGreaterThanOrEqual(0);
    expect(photonEntry).toBeGreaterThan(envSampler);
  });
});

describe('SPPM per-pixel progressive stats update site (PTWG-04)', () => {
  it('guards photonMapContribution so one pixel updates SPPM stats once per frame', () => {
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).toContain('var sppmGatherUpdated = false;');
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).toContain('let sppmReceiverEligible = transmission <= 0.3 &&');
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).toContain('if (!sppmGatherUpdated && sppmReceiverEligible)');
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).toContain('sppmGatherUpdated = true;');

    const calls = PT_WEBGPU_PATH_TRACE_KERNEL_WGSL.match(/photonMapContribution\(/g) ?? [];
    expect(calls).toHaveLength(1);
  });
});
