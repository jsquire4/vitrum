/** Contract pins for opt-in diffuse DDGI-proxy GRIS reuse. */

import { describe, expect, it } from 'vitest';
import { WALKAROUND_UBO_SIZE_BYTES } from '../src/pipeline/constants.js';
import { packWalkaroundUBO, updateUBO } from '../src/pipeline/uboUpdater.js';
import { WALKAROUND_UBO_WGSL } from '../src/shaders/walkaroundUbo.wgsl.js';
import { SPATIAL_GI_WGSL, SPATIAL_GI_GRIS_WGSL } from '../src/shaders/spatialGi.wgsl.js';
import { TEMPORAL_GI_WGSL, TEMPORAL_GI_GRIS_WGSL } from '../src/shaders/temporalGi.wgsl.js';
import { GRIS_REUSE_WGSL } from '../src/shaders/grisReuse.wgsl.js';
import { RIS_GI_WGSL } from '../src/shaders/risGi.wgsl.js';
import { RIS_GI_NRC_BODY } from '../src/shaders/risGiNrc.wgsl.js';
import { SHADING_TERMS_WGSL } from '../src/shaders/shadingTerms.wgsl.js';
import { buildReservoirGiWgsl, RESERVOIR_GI_WGSL } from '../src/shaders/reservoirGi.wgsl.js';
import {
  RESERVOIR_GI_BASE_STRIDE_U32,
  RESERVOIR_GI_GRIS_STRIDE_U32,
  reservoirGiStrideU32ForGrisReuse,
} from '../src/gi/giLayout.js';
import type { PipelineFrameInputs } from '../src/pipeline/WalkaroundGPUPipeline.js';

function fakeInputs(grisReuseOverride?: number): PipelineFrameInputs {
  const matrix = new Float32Array(16);
  return {
    camera: {
      viewMatrix: matrix,
      projMatrix: matrix,
      prevViewProjMatrix: matrix,
      cameraPos: [0, 0, 0],
    },
    screen: {
      screenWidth: 64,
      screenHeight: 64,
      frameSeed: 7,
      swapChainView: {} as GPUTextureView,
      swapChainFormat: 'bgra8unorm',
    },
    lighting: {
      totalEmissivePower: 1,
      emitterCount: 4,
      primaryLightDir: [0, 1, 0],
      primaryLightIntensity: 1,
      skyTint: [0, 0, 0],
      skyIrradiance: 0,
      emitterDist2Floor: 0.01,
      directFireflyClamp: 4,
      causticBoost: 1,
      causticVisClamp: 1,
      lightTreeEnabled: 1,
      lightTreeNodeCount: 7,
    },
    restirDI: {
      temporalMClampDI: 20,
      spatialReuseRadiusPx: 30,
      spatialDepthTolFloor: 0.05,
    },
    restirGI: {
      restirGiWCap: 16,
      restirGiIrrClamp: 5,
      restirGiMClamp: 50,
      restirGiSpatialRadiusPx: 12,
      restirGiSpatialNormalDotMin: 0.9,
      restirGiSpatialCoplanarTol: 0.05,
      ...(grisReuseOverride === undefined ? {} : { grisReuse: grisReuseOverride }),
    },
    gtao: {
      gtaoRadiusPx: 32,
      gtaoIntensity: 2,
      gtaoDepthThreshold: 2,
      gtaoBilateralDepthSigma: 0.25,
      adaptiveSamplingThresholdLow: 0.01,
      adaptiveSamplingThresholdHigh: 0.1,
    },
    filter: {
      triIntersectEpsilon: 1e-5,
      glassMixScale: 0.7,
      indirectFireflyClamp: [1, 1, 1],
      atrousDirectSigmas: [128, 5, 0.05],
      atrousIndirectSigmas: [32, 20, 0.5],
      stainedGlassFlags: 0,
    },
    bvh: { bvhMode: 0, tlasNodeCount: 0 },
    nrc: {},
    composite: { tonemapMode: 0, exposure: 1, outputColorSpace: 0 },
  } as PipelineFrameInputs;
}

function capturingDevice(backing: Uint8Array): GPUDevice {
  return {
    queue: {
      writeBuffer: (_buffer: GPUBuffer, offset: number, data: ArrayBuffer) => {
        backing.set(new Uint8Array(data), offset);
      },
    },
  } as unknown as GPUDevice;
}

describe('WalkaroundUBO GRIS fields', () => {
  it('keeps the gate at u32[103] and epoch bits at u32[105]', () => {
    expect(WALKAROUND_UBO_SIZE_BYTES).toBe(432);
    expect(WALKAROUND_UBO_WGSL).toContain('grisReuse:              u32,');
    expect(WALKAROUND_UBO_WGSL).toContain('offset 412 — GRIS reuse gate');
    expect(WALKAROUND_UBO_WGSL).toContain('y bits = GRIS history epoch');

    const bytes = new Uint8Array(WALKAROUND_UBO_SIZE_BYTES);
    updateUBO(capturingDevice(bytes), {} as GPUBuffer, fakeInputs(1), undefined, undefined, undefined, 0x89abcdef);
    const words = new Uint32Array(bytes.buffer);
    expect(words[103]).toBe(1);
    expect(words[105]).toBe(0x89abcdef);
  });

  it('defaults the opt-in gate and epoch to zero', () => {
    const words = new Uint32Array(packWalkaroundUBO(fakeInputs()));
    expect(words[103]).toBe(0);
    expect(words[105]).toBe(0);
  });

  it('flipping the gate changes only u32[103]', () => {
    const off = new Uint32Array(packWalkaroundUBO(fakeInputs(0)));
    const on = new Uint32Array(packWalkaroundUBO(fakeInputs(1)));
    for (let index = 0; index < off.length; index += 1) {
      expect(on[index]).toBe(index === 103 ? 1 : off[index]);
    }
  });

  it('advancing history changes only u32[105]', () => {
    const epoch0 = new Uint32Array(packWalkaroundUBO(fakeInputs(1)));
    const epoch9 = new Uint32Array(packWalkaroundUBO(
      fakeInputs(1),
      undefined,
      undefined,
      undefined,
      9,
    ));
    for (let index = 0; index < epoch0.length; index += 1) {
      expect(epoch9[index]).toBe(index === 105 ? 9 : epoch0[index]);
    }
  });
});

describe('GRIS shader variants', () => {
  it('keeps default spatial and temporal reuse free of GRIS machinery', () => {
    for (const source of [SPATIAL_GI_WGSL, TEMPORAL_GI_WGSL]) {
      expect(source).toContain('jacobianReconnectionShift(');
      expect(source).not.toContain('grisDomainToCanonicalJacobian(');
      expect(source).not.toContain('grisTransformedDensity(');
      expect(source).not.toContain('ubo.grisReuse == 1u');
    }
  });

  it('spatial reuse evaluates the bounded all-candidate/all-technique matrix', () => {
    expect(SPATIAL_GI_GRIS_WGSL).toContain('var domains: array<ReservoirPT, 6>;');
    expect(SPATIAL_GI_GRIS_WGSL).toContain('j < domainCount');
    expect(SPATIAL_GI_GRIS_WGSL).toContain('grisDomainToCanonicalJacobian(');
    expect(SPATIAL_GI_GRIS_WGSL).toContain('grisTransformedDensity(');
    expect(SPATIAL_GI_GRIS_WGSL).toContain(
      'grisWeightedDensity(domainM[j], transformed)',
    );
    expect(SPATIAL_GI_GRIS_WGSL).toContain('candidate.W * sourceJ');
    expect(SPATIAL_GI_GRIS_WGSL).toContain('foldInvalidReservoirGICandidates(');
    expect(SPATIAL_GI_GRIS_WGSL).toContain('candidate.sampleVisibility > 0.0');
  });

  it('temporal reuse evaluates both current and native previous receivers', () => {
    expect(TEMPORAL_GI_GRIS_WGSL).toContain('var domains: array<ReservoirPT, 2>;');
    expect(TEMPORAL_GI_GRIS_WGSL).toContain('domains[0] = current;');
    expect(TEMPORAL_GI_GRIS_WGSL).toContain('domains[1] = previous;');
    expect(TEMPORAL_GI_GRIS_WGSL).toContain('j < 2u');
    expect(TEMPORAL_GI_GRIS_WGSL).toContain('grisTransformedDensity(');
    expect(TEMPORAL_GI_GRIS_WGSL).toContain(
      'grisWeightedDensity(domainM[j], transformed)',
    );
    expect(TEMPORAL_GI_GRIS_WGSL).toContain('candidate.W * sourceJ');
    expect(TEMPORAL_GI_GRIS_WGSL).toContain('foldInvalidReservoirGICandidates(');
  });

  it('rejects stale or invisible reservoir metadata in both reuse passes', () => {
    for (const source of [SPATIAL_GI_GRIS_WGSL, TEMPORAL_GI_GRIS_WGSL]) {
      expect(source).toContain('historyEpoch != epoch');
      expect(source).toContain('!reservoirGiFinite(');
      expect(source).toContain('nativePHat > 0.0');
      expect(source).toContain('sampleVisibility > 0.0');
      expect(source).toContain('grisProxyVisibilityAt(');
      expect(source).not.toContain('grisShiftJacobian(');
      expect(source).not.toContain('grisPairwiseDenom');
    }
  });

  it('pins inverse-J densities and the environment identity map', () => {
    expect(GRIS_REUSE_WGSL).toContain('fn grisDomainToCanonicalJacobian(');
    expect(GRIS_REUSE_WGSL).toContain('if (sampleKind == GI_SAMPLE_ENVIRONMENT) { return 1.0; }');
    expect(GRIS_REUSE_WGSL).toContain('fn grisTransformedDensity(');
    expect(GRIS_REUSE_WGSL).toContain(
      'let result = pHatDomain / domainToCanonicalJacobian;',
    );
    expect(GRIS_REUSE_WGSL).toContain(
      'reservoirGiFinite(result) && result > 0.0',
    );
    expect(GRIS_REUSE_WGSL).toContain('Invalid inverse mappings and occluded techniques contribute exactly zero.');
  });
});

describe('GRIS reservoir physical layout', () => {
  it('pins the compact and widened stride authority', () => {
    expect(RESERVOIR_GI_BASE_STRIDE_U32).toBe(20);
    expect(RESERVOIR_GI_GRIS_STRIDE_U32).toBe(28);
    expect(reservoirGiStrideU32ForGrisReuse(false)).toBe(20);
    expect(reservoirGiStrideU32ForGrisReuse(true)).toBe(28);
    expect(buildReservoirGiWgsl({ grisCache: false })).toContain(
      'const RESERVOIR_GI_STRIDE: u32 = 20u;',
    );
    expect(RESERVOIR_GI_WGSL).toContain('const RESERVOIR_GI_STRIDE: u32 = 28u;');
  });

  it('pins every appended metadata word in the canonical pack/unpack path', () => {
    const fields = [
      ['wi_recon', 20],
      ['sampleVisibility', 23],
      ['prefixVertexCount', 24],
      ['sampleKind', 25],
      ['nativePHat', 26],
      ['historyEpoch', 27],
    ] as const;
    for (const [field, index] of fields) {
      expect(RESERVOIR_GI_WGSL).toContain(`r.${field}`);
      expect(RESERVOIR_GI_WGSL).toContain(`words[${index}u]`);
    }
    expect(RESERVOIR_GI_WGSL.match(/r\.historyEpoch\s*=\s*words\[27u\]/g)).toHaveLength(1);
    expect(RESERVOIR_GI_WGSL).toContain('words[27u] = r.historyEpoch;');
    expect(RESERVOIR_GI_WGSL).toContain('words[26u] = bitcast<u32>(r.nativePHat);');
    expect(RESERVOIR_GI_WGSL).toContain('words[23u] = bitcast<u32>(r.sampleVisibility);');
  });

  it('pins metadata-aware replacement and exact invalid-attempt folding', () => {
    expect(RESERVOIR_GI_WGSL).toContain('fn updateReservoirGIWithMetadata(');
    expect(RESERVOIR_GI_WGSL).toContain('(*r).nativePHat = nativePHat;');
    expect(RESERVOIR_GI_WGSL).toContain('(*r).sampleVisibility = sampleVisibility;');
    expect(RESERVOIR_GI_WGSL).toContain('(*r).historyEpoch = historyEpoch;');
    expect(RESERVOIR_GI_WGSL).toContain(
      '(*r).M = reservoirGiSaturatingAddU32((*r).M, attemptCount);',
    );
    expect(RESERVOIR_GI_WGSL).toContain('(*r).nativePHat = 0.0;');
    expect(RESERVOIR_GI_WGSL).toContain('(*r).sampleVisibility = 0.0;');
  });

  it('guards every GRIS density/weight boundary and saturates represented M', () => {
    expect(GRIS_REUSE_WGSL).toContain('fn grisSafeDirection(value: vec3f)');
    expect(GRIS_REUSE_WGSL).toContain('fn grisWeightedDensity(attempts: u32, density: f32)');
    expect(GRIS_REUSE_WGSL).toContain('!reservoirGiFinite(pHatDomain)');
    expect(RESERVOIR_GI_WGSL).toContain('fn reservoirGiSaturatingAddU32');
    for (const source of [TEMPORAL_GI_GRIS_WGSL, SPATIAL_GI_GRIS_WGSL]) {
      expect(source).toContain(
        'denominator = denominator + grisWeightedDensity(domainM[j], transformed);',
      );
      expect(source).toContain('if (!reservoirGiFinite(weight) || !(weight > 0.0))');
      expect(source).toContain('out.M = reservoirGiSaturatingAddU32(oldM, attempts);');
      expect(source).toContain('!grisSampleKindValid(');
    }
  });
});

describe('GRIS target/proposal boundary', () => {
  it('forces cosine proposal sampling even when PPG resources are compiled', () => {
    for (const source of [RIS_GI_WGSL, RIS_GI_NRC_BODY]) {
      expect(source).toContain('let ppgGuidedOn = (ubo.ppgEnabled == 1u) && !grisOn;');
      expect(source).toContain('let alpha = select(0.0, ubo.ppgMixAlpha, ppgGuidedOn);');
    }
  });

  it('prevents NRC prediction substitution from changing the proxy suffix', () => {
    expect(RIS_GI_NRC_BODY).toContain('nrcCanSubstitute && !grisOn,');
    expect(RIS_GI_NRC_BODY).toContain('if (grisOn) {');
    expect(RIS_GI_NRC_BODY).toContain('pHat = luminance(Lo) * cosTheta * INV_PI * candidateVisibility;');
  });

  it('reweights GRIS samples for glossy/metal while reserving glass for transmitted GI', () => {
    const transmittedStart = SHADING_TERMS_WGSL.indexOf('fn lo_transmittedGI(');
    const specularStart = SHADING_TERMS_WGSL.indexOf('fn lo_indirectSpecular(');
    expect(transmittedStart).toBeGreaterThan(-1);
    expect(specularStart).toBeGreaterThan(transmittedStart);

    const transmitted = SHADING_TERMS_WGSL.slice(transmittedStart, specularStart);
    const specular = SHADING_TERMS_WGSL.slice(specularStart);

    expect(SHADING_TERMS_WGSL).toContain('g.historyEpoch != bitcast<u32>(ubo.sunAngular.y)');
    expect(SHADING_TERMS_WGSL).toContain('return clamp(g.sampleVisibility, 0.0, 1.0);');
    expect(specular).toContain('if (isGlass) { return vec3f(0.0); }');
    expect(specular).toContain('let grisVisibility = giReservoirVisibility(g);');
    expect(specular).toContain('if (grisVisibility <= 0.0) { return vec3f(0.0); }');
    expect(specular).toContain('let toS = giReservoirDirectionVector(g, pos);');
    expect(specular).toContain('return g.Lo * specBrdf * g.W * grisVisibility;');
    expect(specular).not.toContain('if (ubo.grisReuse == 1u) { return vec3f(0.0); }');
    expect(transmitted).toContain('if (!isGlass) { return vec3f(0.0); }');
  });
});
