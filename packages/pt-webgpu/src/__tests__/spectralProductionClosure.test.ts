import { readFileSync } from 'node:fs';
import { describe, expect, it, vi } from 'vitest';
import type { Scene } from '@vitrum/core';
import { createPTEngine_WebGPU } from '../index.js';
import { assertSpectralSceneSupported } from '../spectralSceneValidation.js';
import { PT_WEBGPU_PATH_TRACE_KERNEL_WGSL } from '../wgsl/pathTrace/kernel.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_KERNEL_LITE_WGSL } from '../wgsl/pathTrace/kernelLite.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_MATERIAL_FUNCS_WGSL } from '../wgsl/pathTrace/material.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_CONNECT_LITE_WGSL } from '../wgsl/pathTrace/connectLite.wgsl.js';
import { PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL } from '../wgsl/bdpt/bdptLightSubpath.wgsl.js';
import { PT_WEBGPU_BDPT_CONNECTION_WGSL } from '../wgsl/bdpt/bdptConnection.wgsl.js';
import { SPPM_GROUP3_BINDINGS_WGSL, SPPM_PHOTON_PASS_WGSL } from '../wgsl/pathTrace/sppmBindings.wgsl.js';
import { composeShadePrologueWgsl } from '../wgsl/pathTrace/shadePrologue.wgsl.js';
import { RESERVOIR_PT_HERO_WGSL } from '../wgsl/pathTrace/reservoirPtHero.wgsl.js';
import { RESTIR_PT_TEMPORAL_WGSL } from '../wgsl/pathTrace/restirPtTemporal.wgsl.js';
import { RESTIR_PT_SPATIAL_WGSL } from '../wgsl/pathTrace/restirPtSpatial.wgsl.js';

const ENGINE_SOURCE = readFileSync(new URL('../index.ts', import.meta.url), 'utf8');

function validationDevice() {
  const createBuffer = vi.fn();
  return {
    createBuffer,
    device: {
      createCommandEncoder: vi.fn(),
      createBuffer,
      limits: {
        maxStorageBuffersPerShaderStage: 64,
        maxStorageTexturesPerShaderStage: 8,
      },
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      lost: new Promise<never>(() => {}),
    } as unknown as GPUDevice,
  };
}

function sceneWithMaterial(material: Record<string, unknown>): Scene {
  return {
    primitives: [{
      kind: 'mesh',
      id: 'film',
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      material: {
        baseColor: [0.8, 0.7, 0.6],
        roughness: 0.2,
        metallic: 0,
        ...material,
      },
    }],
    emitters: [],
    environment: { kind: 'none' },
  };
}

describe('pt-webgpu spectral production closure', () => {
  it('constructs each wavelength-coherent estimator combination without eager allocation', async () => {
    for (const compatible of [
      { restirPtReuse: true },
      { causticStrategy: 'manifold-nee' as const },
      { causticStrategy: 'photon-map' as const },
    ]) {
      const { device, createBuffer } = validationDevice();
      const engine = await createPTEngine_WebGPU({
        device,
        spectral: true,
        ...compatible,
      });
      expect(createBuffer).not.toHaveBeenCalled();
      engine.dispose();
    }
  });

  it('rejects standalone RGB iridescence but accepts wavelength-resolved TMM overrides', () => {
    expect(() => assertSpectralSceneSupported(sceneWithMaterial({
      iridescence: 1,
      iridescenceIor: 1.3,
      iridescenceThicknessRange: [100, 400],
    }))).toThrow(/RGB-integrated/);

    expect(() => assertSpectralSceneSupported(sceneWithMaterial({
      thinFilmStack: {
        layers: [{ ior: 1.4, thicknessNm: 320 }],
        incidentIor: 1,
        angleDependent: true,
      },
    }))).not.toThrow();

    expect(() => assertSpectralSceneSupported(sceneWithMaterial({
      iridescence: 1,
      thinFilmStack: {
        layers: [{ ior: 1.4, thicknessNm: 320 }],
      },
    }))).not.toThrow();

    const repackStart = ENGINE_SOURCE.indexOf('  #repackScene(');
    const repackEnd = ENGINE_SOURCE.indexOf('  #syncLiteTextures(', repackStart);
    const repack = ENGINE_SOURCE.slice(repackStart, repackEnd);
    const validationAt = repack.indexOf('assertSpectralSceneSupported(scene)');
    const packAt = repack.indexOf('const packed = buildPackedScene(scene');
    const uploadAt = repack.indexOf('uploadedScene = uploadPackedScene(');
    const publishAt = repack.indexOf('this.#scene = scene;');
    expect(validationAt).toBeGreaterThanOrEqual(0);
    expect(validationAt).toBeLessThan(packAt);
    expect(packAt).toBeLessThan(uploadAt);
    expect(uploadAt).toBeLessThan(publishAt);
  });

  it('uses the exact invocation hero in both BDPT eye and light paths', () => {
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).toContain(
      'bdptSetInvocationHeroLambda(heroLambda);',
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'spectralEmissionAtHero(emitRad, bdptInvocationHeroLambdaNm)',
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'bdptSampleMaterialAtPayload',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('bdptInvocationHeroLambdaNm');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).not.toContain('params.heroLambdaNm');
  });

  it('keeps material, texture, layer, media, and environment factors scalar at hero lambda', () => {
    const prologue = composeShadePrologueWgsl('');
    expect(prologue).toContain('spectralCombinedReflectanceAtHero');
    expect(prologue).toContain('spectralRgbFactorAtHero(mat.sheenColor');
    expect(prologue).toContain('spectralRgbFactorAtHero(mat.specularColor');
    expect(PT_WEBGPU_PATH_TRACE_MATERIAL_FUNCS_WGSL).toContain(
      'return vec3f(spectralRgbFactorAtHero(layerRgb, heroLambda))',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'spectralRgbFactorAtHero(sigmaA, heroLambda)',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'spectralRgbFactorAtHero(sigmaS, heroLambda)',
    );
    expect(PT_WEBGPU_PATH_TRACE_CONNECT_LITE_WGSL).toContain(
      'spectralEmissionAtHero(envRgb, heroLambda)',
    );
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_LITE_WGSL).toContain(
      'spectralRgbFactorAtHero(sigmaA, heroLambda)',
    );
  });

  it('builds the BDPT light path invocation-locally after binding the eye hero', () => {
    const setHero = PT_WEBGPU_PATH_TRACE_KERNEL_WGSL.indexOf(
      'bdptSetInvocationHeroLambda(heroLambda);',
    );
    const buildLight = PT_WEBGPU_PATH_TRACE_KERNEL_WGSL.indexOf(
      'bdptBuildInvocationLightSubpath(gid.xy);',
    );
    expect(setHero).toBeGreaterThanOrEqual(0);
    expect(buildLight).toBeGreaterThan(setHero);
    expect(ENGINE_SOURCE).not.toContain('bdptEyePathBuffer');
    expect(ENGINE_SOURCE).not.toContain('bdptLightPathBuffer');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).not.toContain('sampleHeroWavelengthMIS');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).not.toContain('550.0');
  });

  it('performs CMF reconstruction once per independent persistent/output boundary', () => {
    const fullCalls = PT_WEBGPU_PATH_TRACE_KERNEL_WGSL.match(/heroWavelengthToRgb\(/g) ?? [];
    const liteCalls = PT_WEBGPU_PATH_TRACE_KERNEL_LITE_WGSL.match(/heroWavelengthToRgb\(/g) ?? [];
    expect(fullCalls).toHaveLength(1);
    expect(liteCalls).toHaveLength(1);
    expect(SPPM_GROUP3_BINDINGS_WGSL.match(/heroWavelengthToRgb\(/g) ?? []).toHaveLength(1);
    expect(SPPM_PHOTON_PASS_WGSL).not.toContain('heroWavelengthToRgb');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).not.toContain('heroWavelengthToRgb');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).not.toContain('heroWavelengthToRgb');
  });

  it('carries the winning ReSTIR sample wavelength through temporal and spatial reuse', () => {
    expect(RESERVOIR_PT_HERO_WGSL).toContain(
      '(*r).heroLambdaV = heroLambda;',
    );
    expect(RESERVOIR_PT_HERO_WGSL).toContain(
      'rptHydrateVisibleDomain(r);',
    );
    expect(RESERVOIR_PT_HERO_WGSL).toContain(
      'fn restirPtTargetForDomainAtHero(',
    );

    // Cross-domain target evaluations use the candidate sample's wavelength,
    // never the receiving pixel/frame's wavelength.
    expect(RESTIR_PT_TEMPORAL_WGSL).toContain(
      'rCur, rPrev.heroLambdaV, woCur, rPrev.xs, rPrev.Lo,',
    );
    expect(RESTIR_PT_TEMPORAL_WGSL).toContain(
      'rPrev, rCur.heroLambdaV, woPrev, rCur.xs, rCur.Lo,',
    );
    expect(RESTIR_PT_TEMPORAL_WGSL).toContain(
      '&rGris, rPrev.xs, rPrev.ns, rPrev.Lo, rPrev.heroLambdaV,',
    );
    expect(RESTIR_PT_SPATIAL_WGSL).toContain(
      'rCenter, qR[i].heroLambdaV, woCenter, qR[i].xs, qR[i].Lo,',
    );
    expect(RESTIR_PT_SPATIAL_WGSL).toContain(
      '&rOut, qR[i].xs, qR[i].ns, qR[i].Lo, qR[i].heroLambdaV,',
    );
  });
});
