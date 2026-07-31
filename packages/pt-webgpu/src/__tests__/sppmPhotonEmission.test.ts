import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { SPPM_PHOTON_PASS_WGSL } from '../wgsl/pathTrace/sppmBindings.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_KERNEL_WGSL } from '../wgsl/pathTrace/kernel.wgsl.js';
import { composeSppmPhotonPassWgsl } from '../wgsl/pathTraceBruteforce.wgsl.js';
import { sppmSceneBoundsFromPackedPositions } from '../sppmParams.js';

const PT_WEBGPU_INDEX_SOURCE = readFileSync(new URL('../index.ts', import.meta.url), 'utf8');
const SCENE_MUTATION_ROUTER_SOURCE = readFileSync(
  new URL('../sceneMutationRouter.ts', import.meta.url),
  'utf8',
);
const SHADE_PROLOGUE_SOURCE = readFileSync(
  new URL('../wgsl/pathTrace/shadePrologue.wgsl.ts', import.meta.url),
  'utf8',
);

function expectedSelectedSourcePower(
  sourcePower: number,
  sourceCount: number,
  photonCount: number,
): number {
  const pSelect = 1 / sourceCount;
  const unnormalizedPhotonPower = sourcePower / pSelect;
  const accumulatedPower = photonCount * pSelect * unnormalizedPhotonPower;
  return accumulatedPower / photonCount;
}

function sampleDirectionalConeZ(
  angularDiameter: number,
  u: number,
  v: number,
): readonly [number, number, number] {
  if (angularDiameter <= 0) return [0, 0, 1];
  const cosHalfAngle = Math.cos(angularDiameter * 0.5);
  const cosTheta = cosHalfAngle + (1 - cosHalfAngle) * u;
  const sinTheta = Math.sqrt(Math.max(0, 1 - cosTheta * cosTheta));
  const phi = 2 * Math.PI * v;
  return [sinTheta * Math.cos(phi), sinTheta * Math.sin(phi), cosTheta];
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
    expect(SPPM_PHOTON_PASS_WGSL).toContain(
      'pointLights[pointBase + 1u].rgb * (4.0 * PI) * lightSelectInvPdf',
    );
    expect(SPPM_PHOTON_PASS_WGSL).toContain(
      'sradW.rgb * solidAngle * softness * lightSelectInvPdf',
    );
  });

  it('stores unnormalized photon power and leaves the sole 1/N to gather', () => {
    const photonCountRefs = SPPM_PHOTON_PASS_WGSL.match(/sppmStats\.photonCount/g) ?? [];
    expect(photonCountRefs).toHaveLength(1); // dispatch bounds check only
    expect(SPPM_PHOTON_PASS_WGSL).not.toMatch(/photonFlux[\s\S]{0,120}photonCount/);
  });

  it('uses production alpha/material/BSDF transport and deposits only after a prior delta event', () => {
    expect(SPPM_PHOTON_PASS_WGSL).toContain('alphaTestPassThrough(');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('sampleNextBounceDirectionWithClearcoatNormal(');
    expect(SPPM_PHOTON_PASS_WGSL).toContain(
      'if (!bs.sampledIsDelta || bs.sampledEventPdf <= 0.0) { break; }',
    );
    expect(SPPM_PHOTON_PASS_WGSL).toContain(
      'if (hadDeltaChainEvent && diffuseReceiverWeight > 0.0)',
    );
    expect(SPPM_PHOTON_PASS_WGSL.indexOf('sppmInsertPhoton(photonIdx')).toBeLessThan(
      SPPM_PHOTON_PASS_WGSL.indexOf('hadDeltaChainEvent = true;'),
    );
  });

  it('keeps photon-walk depth independent from the MNEE chain-length option', () => {
    const composed = composeSppmPhotonPassWgsl();
    expect(composed).toContain('const SPPM_PHOTON_MAX_BOUNCES = 8u;');
    expect(composed).toContain('let maxBounces = SPPM_PHOTON_MAX_BOUNCES;');
    expect(composed).not.toContain('clamp(params.mneeMaxChainLength');
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
    expect(SPPM_PHOTON_PASS_WGSL).toContain(
      'for (var dirIdx = 0u; dirIdx < params.directionalLightCount; dirIdx = dirIdx + 1u)',
    );
    expect(SPPM_PHOTON_PASS_WGSL).toContain('if (sppmDirectionalCastsShadow(dirIdx))');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('let dDirAD = directionalLights[dBase]');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('let dIrrMean = directionalLights[dBase + 1u]');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('photonDir    = -sampledTowardLightDir;');
    expect(SPPM_PHOTON_PASS_WGSL).not.toContain('vec3f(params.lightDir.w) * diskArea');
  });

  it('samples zero-width and soft directional photons over the authored cone', () => {
    expect(sampleDirectionalConeZ(0, 0.27, 0.81)).toEqual([0, 0, 1]);
    const angularDiameter = 0.24;
    const cosHalfAngle = Math.cos(angularDiameter * 0.5);
    const sampleCount = 20_000;
    let meanCosine = 0;
    let minimumCosine = 1;
    let maximumLengthError = 0;
    for (let i = 0; i < sampleCount; i += 1) {
      const direction = sampleDirectionalConeZ(
        angularDiameter,
        (i + 0.5) / sampleCount,
        (i * 0.6180339887498949) % 1,
      );
      const length = Math.hypot(...direction);
      maximumLengthError = Math.max(maximumLengthError, Math.abs(length - 1));
      meanCosine += direction[2];
      minimumCosine = Math.min(minimumCosine, direction[2]);
    }
    meanCosine /= sampleCount;
    expect(maximumLengthError).toBeLessThan(1e-13);
    expect(minimumCosine).toBeGreaterThanOrEqual(cosHalfAngle);
    expect(meanCosine).toBeCloseTo((1 + cosHalfAngle) * 0.5, 12);

    expect(SPPM_PHOTON_PASS_WGSL).toContain('fn sppmSampleDirectionalCone(');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('let angularDiameterRaw = dDirAD.w;');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('-1.0 - angularDiameterRaw,');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('buildOnb(sampledTowardLightDir, &lt, &lb);');
    expect(SPPM_PHOTON_PASS_WGSL).not.toContain('dIrrMean.rgb * diskArea * solidAngle');
  });

  it('applies spot smooth-penumbra weighting to sampled spot photons', () => {
    expect(SPPM_PHOTON_PASS_WGSL).toContain('let cosInner = sradW.w;');
    expect(SPPM_PHOTON_PASS_WGSL).toContain(
      'let softness = smoothstep(cosMin, max(cosInner, cosMin + 1e-6), cosTheta);',
    );
  });

  it('includes rect/disc, mesh-area, and environment sources in the same flat selection order as NEE', () => {
    expect(SPPM_PHOTON_PASS_WGSL).toContain('if (sppmRectAreaCastsShadow(rectIdx))');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('if (sppmMeshAreaCastsShadow(meshIdx))');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('if (hasEnvironmentMap()) {');
    expect(SPPM_PHOTON_PASS_WGSL).not.toContain('params.environmentSun.w');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('availableLightCount = availableLightCount + 1u;');
  });

  it('emits rect/disc photons from the NEE packed source data and area conventions', () => {
    expect(SPPM_PHOTON_PASS_WGSL).toContain('let rectBase = rectIdx * 4u;');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('let rpos = rectAreaLights[rectBase].xyz;');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('let ru = rectAreaLights[rectBase + 1u].xyz;');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('let rv = rectAreaLights[rectBase + 2u].xyz;');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('let rshape = rectAreaLights[rectBase + 3u];');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('let isDisc = abs(rshape.w - 1.0) < 0.5;');
    expect(SPPM_PHOTON_PASS_WGSL).toContain(
      'sppmConcentricDiscSample(vec2f(xi1 * 2.0 - 1.0, xi2 * 2.0 - 1.0))',
    );
    expect(SPPM_PHOTON_PASS_WGSL).toContain(
      'ru, rv, select(4.0, PI, isDisc),',
    );
    expect(SPPM_PHOTON_PASS_WGSL).toContain('if (areaMeasure.valid != 0u)');
    expect(SPPM_PHOTON_PASS_WGSL).toContain(
      'let hemi = cosineHemisphereSample(&rng, lightNormal);',
    );
    expect(SPPM_PHOTON_PASS_WGSL).toContain(
      'photonFlux   = rr * areaMeasure.area * PI * lightSelectInvPdf',
    );
  });

  it('emits mesh-area photons from the NEE packed triangle data and area convention', () => {
    expect(SPPM_PHOTON_PASS_WGSL).toContain('let meshBase = meshAreaLightBase(meshIdx);');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('let a = meshAreaLights[meshBase].xyz;');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('let b = meshAreaLights[meshBase + 1u].xyz;');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('let c = meshAreaLights[meshBase + 2u].xyz;');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('let mr = sampleMeshAreaLightRadiance(');
    expect(SPPM_PHOTON_PASS_WGSL).toContain(
      'let areaMeasure = measureAreaVector(e1, e2, 0.5);',
    );
    expect(SPPM_PHOTON_PASS_WGSL).toContain(
      'mr * areaMeasure.area * PI * sidedPowerScale * lightSelectInvPdf',
    );
  });

  it('emits environment photons through the same NEE importance and PDF helpers', () => {
    expect(SPPM_PHOTON_PASS_WGSL).toContain('let envSample = sampleEnvironmentImportance(&rng);');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('envColor = sampleEnvironmentColor(envDir);');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('envPdf = 0.25 * INV_PI;');
    expect(SPPM_PHOTON_PASS_WGSL).toContain(
      'photonFlux = ptScaleEnvironmentRadiance(envColor, diskArea);',
    );
    expect(SPPM_PHOTON_PASS_WGSL).toContain(
      'photonFlux = ptScaleEnvironmentRadiance(\n' +
      '        photonFlux,\n' +
      '        lightSelectInvPdf,\n' +
      '      );',
    );
    expect(SPPM_PHOTON_PASS_WGSL).toContain(
      'photonFlux = ptScaleEnvironmentRadiance(photonFlux, 1.0 / envPdf);',
    );
    expect(SPPM_PHOTON_PASS_WGSL).not.toContain('TODO(PTWG-03 area/env)');
  });

  it('samples extinction/scattering free flight while photon chains travel inside media', () => {
    expect(SPPM_PHOTON_PASS_WGSL).toContain('var photonMediumDepth = 0u;');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('var photonMediumSigmaT: array<vec3f, 8>;');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('let freeFlightDist =');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('if (freeFlightDist < mediumSegmentDistance)');
    expect(SPPM_PHOTON_PASS_WGSL).toContain(
      'flux = flux * photonSigmaS * transmittance / max(pdfHero, 1e-9);',
    );
    expect(SPPM_PHOTON_PASS_WGSL).toContain(
      'let sigmaA = sppmMaterialSigmaA(matId, mat, photonHeroLambda);',
    );
    expect(SPPM_PHOTON_PASS_WGSL).toContain(
      'var sigmaA = select(vec3f(0.0), max(mat.sigmaA, vec3f(0.0)), mat.hasSigmaA);',
    );
    expect(SPPM_PHOTON_PASS_WGSL).toContain(
      'let sigmaS = sppmMaterialSigmaS(mat, photonHeroLambda);',
    );
    expect(SPPM_PHOTON_PASS_WGSL).toContain('sigmaA + sigmaS, vec3f(0.0)');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('SPPM_PHOTON_KIND_VOLUME');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('if (bs.enteredMedium)');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('bs.exitedMedium');
    expect(SPPM_PHOTON_PASS_WGSL).not.toContain('Beer-Lambert medium extinction omitted');
  });

  it('composes the full connect helpers before the SPPM photon entry point', () => {
    const wgsl = composeSppmPhotonPassWgsl();
    const envSampler = wgsl.indexOf('fn sampleEnvironmentImportance(');
    const photonEntry = wgsl.indexOf('fn sppmEmitPhotons(');

    expect(envSampler).toBeGreaterThanOrEqual(0);
    expect(photonEntry).toBeGreaterThan(envSampler);
  });
});
describe('SPPM source-measure and launch-volume oracles', () => {
  it('cancels source PMF, position PDF, and direction PDF for every source class', () => {
    const pSelect = 1 / 7;

    const directionalIrradiance = 3;
    const launchArea = 11;
    const directionalWeight = (directionalIrradiance * launchArea) / pSelect;
    const directionalIntegral = pSelect * (1 / launchArea) * directionalWeight * launchArea;
    expect(directionalIntegral).toBeCloseTo(directionalIrradiance * launchArea, 12);

    const pointIntensity = 2.5;
    const sphereMeasure = 4 * Math.PI;
    const pointWeight = (pointIntensity * sphereMeasure) / pSelect;
    const pointIntegral = pSelect * (1 / sphereMeasure) * pointWeight * sphereMeasure;
    expect(pointIntegral).toBeCloseTo(pointIntensity * sphereMeasure, 12);

    const spotIntensity = 4;
    const cosOuter = 0.5;
    const cosInner = 0.8;
    const coneMeasure = 2 * Math.PI * (1 - cosOuter);
    let spotEstimator = 0;
    let spotReference = 0;
    const steps = 20_000;
    for (let i = 0; i < steps; i++) {
      const cosTheta = cosOuter + ((i + 0.5) / steps) * (1 - cosOuter);
      const t = Math.min(1, Math.max(0, (cosTheta - cosOuter) / (cosInner - cosOuter)));
      const softness = t * t * (3 - 2 * t);
      const storedWeight = (spotIntensity * coneMeasure * softness) / pSelect;
      spotEstimator += (pSelect * storedWeight) / steps;
      spotReference += (spotIntensity * 2 * Math.PI * softness * (1 - cosOuter)) / steps;
    }
    expect(spotEstimator).toBeCloseTo(spotReference, 10);

    for (const [kind, area, radiance] of [
      ['rectangle', 6, 1.5],
      ['disc', Math.PI * 4, 2],
      ['mesh', 3.25, 0.75],
    ] as const) {
      const storedWeight = (radiance * area * Math.PI) / pSelect;
      const integrated = pSelect * storedWeight;
      expect(integrated, kind).toBeCloseTo(radiance * area * Math.PI, 12);
    }

    const envArea = 9;
    const envBins = [
      { omega: 2 * Math.PI, pdf: 0.1 / (2 * Math.PI), radiance: 1 },
      { omega: 2 * Math.PI, pdf: 0.9 / (2 * Math.PI), radiance: 5 },
    ];
    const envIntegral = envBins.reduce((sum, bin) => {
      const sampleMass = bin.pdf * bin.omega;
      const storedWeight = (bin.radiance * envArea) / (pSelect * bin.pdf);
      return sum + pSelect * sampleMass * storedWeight;
    }, 0);
    const envReference = envBins.reduce((sum, bin) => sum + bin.radiance * envArea * bin.omega, 0);
    expect(envIntegral).toBeCloseTo(envReference, 12);
  });

  it('scene-centered launch disks cover extreme-aspect AABBs for any source direction', () => {
    const min = [-500_000, -0.5, -0.005] as const;
    const max = [500_000, 0.5, 0.005] as const;
    const positions = new Float32Array([
      min[0],
      min[1],
      min[2],
      0,
      max[0],
      min[1],
      min[2],
      0,
      min[0],
      max[1],
      min[2],
      0,
      min[0],
      min[1],
      max[2],
      0,
      max[0],
      max[1],
      max[2],
      0,
    ]);
    const bounds = sppmSceneBoundsFromPackedPositions(positions);
    expect(bounds).not.toBeNull();
    expect(bounds!.center).toEqual([0, 0, 0]);
    expect(bounds!.extent).toBeGreaterThanOrEqual(500_000);

    const directions = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
      [1 / Math.sqrt(3), 1 / Math.sqrt(3), 1 / Math.sqrt(3)],
    ] as const;
    for (let i = 0; i < positions.length; i += 4) {
      const rel = [
        positions[i]! - bounds!.center[0],
        positions[i + 1]! - bounds!.center[1],
        positions[i + 2]! - bounds!.center[2],
      ] as const;
      const relLength2 = rel[0] ** 2 + rel[1] ** 2 + rel[2] ** 2;
      for (const direction of directions) {
        const axial = rel[0] * direction[0] + rel[1] * direction[1] + rel[2] * direction[2];
        const projectedRadius = Math.sqrt(Math.max(0, relLength2 - axial * axial));
        expect(projectedRadius).toBeLessThanOrEqual(bounds!.extent + 1e-6);
      }
    }
  });

  it('refreshes scene center and radius on incremental geometry publication', () => {
    const first = sppmSceneBoundsFromPackedPositions(new Float32Array([0, 0, 0, 0, 2, 2, 2, 0]));
    const moved = sppmSceneBoundsFromPackedPositions(
      new Float32Array([100, -4, 7, 0, 140, 6, 9, 0]),
    );
    expect(first?.center).toEqual([1, 1, 1]);
    expect(moved?.center).toEqual([120, 1, 8]);
    expect(moved!.extent).not.toBe(first!.extent);

    expect(PT_WEBGPU_INDEX_SOURCE).toContain('refreshSceneGeometryStats: () =>');
    expect(PT_WEBGPU_INDEX_SOURCE).toContain('this.#computeSppmSceneStats();');
    expect(SCENE_MUTATION_ROUTER_SOURCE).toContain(
      'rollbackSceneGeometryStats = host.refreshSceneGeometryStats?.() ?? null;',
    );
  });

  it('pins environment direction and spot falloff conventions', () => {
    expect(SPPM_PHOTON_PASS_WGSL).toContain('envDir = envSample.wi;');
    expect(SPPM_PHOTON_PASS_WGSL).toContain(
      'photonOrigin = sceneCenter + diskPos + envDir * extent * 2.0;',
    );
    expect(SPPM_PHOTON_PASS_WGSL).toContain('photonDir    = -envDir;');
    expect(SPPM_PHOTON_PASS_WGSL).toContain(
      'let softness = smoothstep(cosMin, max(cosInner, cosMin + 1e-6), cosTheta);',
    );
    expect(SPPM_PHOTON_PASS_WGSL).toContain(
      'photonFlux   = sradW.rgb * solidAngle * softness * lightSelectInvPdf;',
    );
  });

  it('uses one ownership gate for the complete supported emitter matrix', () => {
    for (const countField of [
      'directionalLightCount',
      'pointLightCount',
      'spotLightCount',
      'rectAreaLightCount',
      'meshAreaLightCount',
    ]) {
      expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).toContain(
        `lightCount = lightCount + params.${countField};`,
      );
    }
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).toContain('let directFamilyCount = select(');
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).toContain(
      'lightCount, distantDirectEmitterCount(), bdptOwnsFiniteLightFamily,',
    );
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).toContain(
      'if (directFamilyCount > 0u && !sppmOwnsCurrentDirect)',
    );
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).toContain(
      'if (!prevSampleAllowsAreaMis && !sppmOwnsCurrentEmission)',
    );
    expect(SHADE_PROLOGUE_SOURCE).toContain(
      '!sppmOwnsCurrentEmission${emissiveOwnershipGuard} &&\n' +
        '      (isFrontFace || mat.doubleSided)',
    );
  });
});

describe('SPPM per-pixel progressive stats update site (PTWG-04)', () => {
  it('guards photonMapUpdateProgressive so one pixel mutates stats once per frame', () => {
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).toContain('var sppmGatherUpdated = false;');
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).toContain(
      'let sppmReceiverEligible =\n' + '        (1.0 - metallic) * (1.0 - transmission) > 0.0;',
    );
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).toContain(
      'if (!sppmGatherUpdated && sppmReceiverEligible)',
    );
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).toContain('sppmGatherUpdated = true;');

    const calls = PT_WEBGPU_PATH_TRACE_KERNEL_WGSL.match(/photonMapUpdateProgressive\(/g) ?? [];
    expect(calls).toHaveLength(1);
  });
});
