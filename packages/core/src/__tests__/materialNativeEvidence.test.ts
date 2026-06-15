/**
 * GATE-02 — every `native` material promise must carry concrete evidence.
 *
 * This test is deliberately a ledger gate, not a renderer. It prevents the
 * high-risk failure mode where a backend row is promoted to `native` in
 * BACKEND_PROMISE_LEDGER without a named packer/shader/readback proof that an
 * implementer can audit. Backend package tests still own the actual math and
 * shader assertions; this file makes the evidence matrix exhaustive.
 */

import { describe, expect, it } from 'vitest';
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { BACKEND_PROMISE_LEDGER } from '../engine/promiseLedger.js';

type BackendWithMaterialEvidence = 'walkaround-hybrid' | 'pt-webgl2' | 'pt-webgpu';
type EvidenceKind = 'packer+shader' | 'shared-classifier' | 'readback-oracle';

interface MaterialNativeEvidence {
  readonly kind: EvidenceKind;
  readonly tests: readonly string[];
  readonly sources: readonly string[];
}

function evidence(
  kind: EvidenceKind,
  tests: readonly string[],
  sources: readonly string[],
): MaterialNativeEvidence {
  return { kind, tests, sources };
}

function group(
  fields: readonly string[],
  ev: MaterialNativeEvidence,
): Record<string, MaterialNativeEvidence> {
  return Object.fromEntries(fields.map((field) => [field, ev]));
}

const WALKAROUND_EMISSIVE = evidence('shared-classifier', [
  'packages/shared-bvh/src/__tests__/emitterClassify.test.ts',
  'packages/walkaround-hybrid/src/restir/__tests__/emissiveLePacking.test.ts',
  'packages/walkaround-hybrid/src/restir/__tests__/directLightEmitterCore.test.ts',
], [
  'packages/shared-bvh/src/emitterClassify.ts',
  'packages/walkaround-hybrid/src/restir/emitterList.ts',
  'packages/walkaround-hybrid/src/restir/packingHelpers.ts',
]);

const WALKAROUND_EXTENSIONS = evidence('shared-classifier', [
  'packages/walkaround-hybrid/__tests__/surfaceTextureIds.test.ts',
  'packages/walkaround-hybrid/__tests__/consumedMaterialFields.test.ts',
], [
  'packages/shared-bvh/src/emitterClassify.ts',
  'packages/walkaround-hybrid/src/restir/consumedMaterialFields.ts',
  'packages/walkaround-hybrid/src/restir/packingHelpers.ts',
]);

const WALKAROUND_ENV_INTENSITY = evidence('packer+shader', [
  'packages/walkaround-hybrid/src/__tests__/materialTextureAtlas.test.ts',
  'packages/walkaround-hybrid/src/__tests__/mutationMatrix.test.ts',
  'packages/walkaround-hybrid/src/__tests__/consumedMaterialFields.test.ts',
], [
  'packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts',
  'packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts',
  'packages/walkaround-hybrid/src/shaders/ris.wgsl.ts',
  'packages/walkaround-hybrid/src/shaders/restirPHat.wgsl.ts',
  'packages/walkaround-hybrid/src/shaders/shadingTerms.wgsl.ts',
]);

const PT_WEBGL2_SCALARS = evidence('packer+shader', [
  'packages/pt-webgl2/src/scene/materialsTexture.test.ts',
  'packages/pt-webgl2/src/glsl/composeTraceGlsl.test.ts',
  'packages/pt-webgl2/src/scene/materialStrideParity.test.ts',
], [
  'packages/pt-webgl2/src/scene/materialsTexture.ts',
  'packages/pt-webgl2/src/glsl/shader/structs/material_struct.glsl.js',
  'packages/pt-webgl2/src/glsl/render/get_surface_record_function.glsl.js',
  'packages/pt-webgl2/src/glsl/shader/bsdf/bsdf_functions.glsl.js',
]);

const PT_WEBGL2_TEXTURES = evidence('packer+shader', [
  'packages/pt-webgl2/src/scene/untestedMaterialMaps.test.ts',
  'packages/pt-webgl2/src/scene/materialsTexture.test.ts',
  'packages/pt-webgl2/src/scene/materialStrideParity.test.ts',
], [
  'packages/pt-webgl2/src/scene/texturesArray.ts',
  'packages/pt-webgl2/src/scene/materialsTexture.ts',
  'packages/pt-webgl2/src/glsl/shader/structs/material_struct.glsl.js',
  'packages/pt-webgl2/src/glsl/render/get_surface_record_function.glsl.js',
  'packages/pt-webgl2/src/glsl/render/attenuate_hit_function.glsl.js',
]);

const PT_WEBGPU_SCALARS = evidence('packer+shader', [
  'packages/pt-webgpu/src/__tests__/scenePack.test.ts',
  'packages/pt-webgpu/src/__tests__/implicitMeshEmitter.test.ts',
  'packages/pt-webgpu/src/__tests__/wgslContract.test.ts',
], [
  'packages/pt-webgpu/src/scene/materialPacking.ts',
  'packages/pt-webgpu/src/scene/emitterPacking.ts',
  'packages/pt-webgpu/src/wgsl/pathTrace/material.wgsl.ts',
  'packages/pt-webgpu/src/wgsl/pathTrace/bsdf.wgsl.ts',
]);

const PT_WEBGPU_TEXTURES = evidence('packer+shader', [
  'packages/pt-webgpu/src/__tests__/materialTextures.test.ts',
  'packages/pt-webgpu/src/__tests__/materialTextureArray.test.ts',
  'packages/pt-webgpu/src/__tests__/wgslContract.test.ts',
], [
  'packages/pt-webgpu/src/scene/materialTextures.ts',
  'packages/pt-webgpu/src/scene/materialTextureArray.ts',
  'packages/pt-webgpu/src/wgsl/pathTrace/material.wgsl.ts',
]);

const MATERIAL_NATIVE_EVIDENCE: Record<BackendWithMaterialEvidence, Record<string, MaterialNativeEvidence>> = {
  'walkaround-hybrid': {
    ...group(['emissive', 'emissiveIntensity'], WALKAROUND_EMISSIVE),
    envMapIntensity: WALKAROUND_ENV_INTENSITY,
    extensions: WALKAROUND_EXTENSIONS,
  },
  'pt-webgl2': {
    ...group([
      'baseColor', 'roughness', 'metallic', 'emissive', 'emissiveIntensity',
      'alphaMode', 'alphaCutoff', 'opacity', 'transmission', 'ior',
      'attenuationColor', 'attenuationDistance', 'normalScale', 'aoMapIntensity',
      'clearcoatNormalScale', 'bumpScale', 'lightMapIntensity', 'sheen',
      'sheenColor', 'sheenRoughness', 'clearcoat', 'clearcoatRoughness',
      'iridescence', 'iridescenceIor', 'iridescenceThicknessRange',
      'specularIntensity', 'specularColor', 'envMapIntensity',
      'spectralAttenuation', 'dispersionAbbeNumber', 'scatteringCoefficient',
      'scatteringAnisotropy', 'thinFilmStack', 'anisotropy', 'anisotropyRotation',
    ], PT_WEBGL2_SCALARS),
    ...group([
      'baseColorMap', 'normalMap', 'roughnessMap', 'metallicMap',
      'transmissionMap', 'emissiveMap', 'alphaMap', 'aoMap', 'clearcoatMap',
      'clearcoatRoughnessMap', 'clearcoatNormalMap', 'sheenColorMap',
      'sheenRoughnessMap', 'iridescenceMap', 'iridescenceThicknessMap',
      'anisotropyMap', 'specularColorMap', 'specularIntensityMap', 'bumpMap',
      'lightMap',
    ], PT_WEBGL2_TEXTURES),
  },
  'pt-webgpu': {
    ...group([
      'baseColor', 'roughness', 'metallic', 'emissive', 'emissiveIntensity',
      'alphaMode', 'alphaCutoff', 'opacity', 'transmission', 'ior',
      'attenuationColor', 'attenuationDistance', 'normalScale', 'aoMapIntensity',
      'clearcoatNormalScale', 'bumpScale', 'lightMapIntensity', 'sheen',
      'sheenColor', 'sheenRoughness', 'clearcoat', 'clearcoatRoughness',
      'iridescence', 'iridescenceIor', 'iridescenceThicknessRange',
      'envMapIntensity', 'spectralAttenuation', 'dispersionAbbeNumber',
      'scatteringCoefficient', 'scatteringAnisotropy', 'scatteringCoefficientRGB',
      'thinFilmStack', 'anisotropy', 'anisotropyRotation',
    ], PT_WEBGPU_SCALARS),
    ...group([
      'baseColorMap', 'normalMap', 'roughnessMap', 'metallicMap',
      'transmissionMap', 'emissiveMap', 'alphaMap', 'aoMap', 'anisotropyMap',
      'bumpMap', 'lightMap',
    ], PT_WEBGPU_TEXTURES),
  },
};

const REPO_ROOT = resolve(fileURLToPath(new URL('../../../../', import.meta.url)));

function nativeMaterialRows(backend: BackendWithMaterialEvidence): string[] {
  return Object.entries(BACKEND_PROMISE_LEDGER[backend].supportDetails.materials)
    .filter(([, grade]) => grade === 'native')
    .map(([field]) => field)
    .sort();
}

describe('GATE-02 native material evidence', () => {
  it('has explicit packer/shader/readback evidence for every native material row', () => {
    for (const backend of Object.keys(MATERIAL_NATIVE_EVIDENCE) as BackendWithMaterialEvidence[]) {
      expect(
        Object.keys(MATERIAL_NATIVE_EVIDENCE[backend]).sort(),
        `${backend} native material evidence rows`,
      ).toEqual(nativeMaterialRows(backend));
    }
  });

  it('does not accept empty evidence records', () => {
    for (const [backend, rows] of Object.entries(MATERIAL_NATIVE_EVIDENCE)) {
      for (const [field, ev] of Object.entries(rows)) {
        expect(ev.tests.length, `${backend}.${field} tests`).toBeGreaterThan(0);
        expect(ev.sources.length, `${backend}.${field} sources`).toBeGreaterThan(0);
      }
    }
  });

  it('points every evidence record at files that exist', () => {
    for (const [backend, rows] of Object.entries(MATERIAL_NATIVE_EVIDENCE)) {
      for (const [field, ev] of Object.entries(rows)) {
        for (const path of [...ev.tests, ...ev.sources]) {
          expect(existsSync(resolve(REPO_ROOT, path)), `${backend}.${field}: ${path}`).toBe(true);
        }
      }
    }
  });
});
