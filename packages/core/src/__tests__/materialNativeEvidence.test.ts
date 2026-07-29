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
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { BACKEND_PROMISE_LEDGER } from '../engine/promiseLedger.js';

type BackendWithMaterialEvidence = 'walkaround-hybrid' | 'pt-webgl2' | 'pt-webgpu';
type EvidenceKind = 'packer+shader' | 'shared-classifier' | 'readback-oracle';

interface EvidenceSnippet {
  readonly path: string;
  readonly includes: readonly string[];
  readonly excludes?: readonly string[];
}

interface MaterialNativeEvidence {
  readonly kind: EvidenceKind;
  readonly tests: readonly string[];
  readonly sources: readonly string[];
  readonly snippets?: readonly EvidenceSnippet[];
}

function evidence(
  kind: EvidenceKind,
  tests: readonly string[],
  sources: readonly string[],
  snippets?: readonly EvidenceSnippet[],
): MaterialNativeEvidence {
  return { kind, tests, sources, ...(snippets != null ? { snippets } : {}) };
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

const WALKAROUND_EMISSIVE_MAP = evidence('packer+shader', [
  'packages/walkaround-hybrid/src/__tests__/materialTextureAtlas.test.ts',
  'packages/walkaround-hybrid/src/__tests__/consumedMaterialFields.test.ts',
], [
  'packages/walkaround-hybrid/src/bvh/materialTextureAtlasPack.ts',
  'packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts',
  'packages/walkaround-hybrid/src/shaders/shade.wgsl.ts',
], [
  {
    path: 'packages/walkaround-hybrid/src/__tests__/materialTextureAtlas.test.ts',
    includes: [
      'packs emissiveMap as an sRGB-decoded atlas slot for visible emitter glow',
    ],
  },
  {
    path: 'packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts',
    includes: ['fn sampleEmissiveMap('],
  },
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

const WALKAROUND_DOUBLE_SIDED = evidence('packer+shader', [
  'packages/shared-bvh/src/__tests__/materialEntry.test.ts',
  'packages/walkaround-hybrid/src/__tests__/doubleSidedTransport.test.ts',
  'packages/walkaround-rc/__tests__/doubleSidedTransport.test.ts',
], [
  'packages/shared-bvh/src/materialEntry.ts',
  'packages/shared-bvh/src/wgsl/tlasTraversal.wgsl.ts',
  'packages/walkaround-hybrid/src/bvh/materialTextureAtlasPack.ts',
  'packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts',
  'packages/walkaround-hybrid/src/ddgi/wgsl/probeUpdateRays.wgsl.ts',
  'packages/walkaround-rc/src/wgsl/probeRayCast.wgsl.ts',
]);

const WALKAROUND_ALPHA = evidence('packer+shader', [
  'packages/walkaround-hybrid/src/__tests__/materialTextureAtlas.test.ts',
  'packages/walkaround-hybrid/src/__tests__/transparentAlphaTransportContract.test.ts',
  'packages/walkaround-hybrid/src/__tests__/transparentOitMaterialParity.test.ts',
], [
  'packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts',
  'packages/walkaround-hybrid/src/shaders/transparentOit.wgsl.ts',
  'packages/walkaround-hybrid/src/ddgi/wgsl/probeUpdateRays.wgsl.ts',
  'packages/walkaround-rc/src/wgsl/probeRayCast.wgsl.ts',
]);

const WALKAROUND_LIGHT_MAP = evidence('packer+shader', [
  'packages/walkaround-hybrid/src/__tests__/materialTextureAtlas.test.ts',
  'packages/walkaround-hybrid/src/__tests__/transparentAlphaTransportContract.test.ts',
  'packages/walkaround-hybrid/src/ddgi/__tests__/ddgiMaterialMapSemantics.test.ts',
  'packages/walkaround-rc/__tests__/probeRayCastWgsl.test.ts',
], [
  'packages/walkaround-hybrid/src/shaders/shade.wgsl.ts',
  'packages/walkaround-hybrid/src/shaders/restirGiMaterial.wgsl.ts',
  'packages/walkaround-hybrid/src/ddgi/wgsl/probeUpdateRays.wgsl.ts',
  'packages/walkaround-rc/src/wgsl/probeRayCast.wgsl.ts',
]);

const WALKAROUND_RICH_LOBES = evidence('packer+shader', [
  'packages/walkaround-hybrid/src/__tests__/materialTextureAtlas.test.ts',
  'packages/walkaround-hybrid/src/__tests__/restirDiMaterialParity.test.ts',
  'packages/walkaround-hybrid/src/__tests__/restirGiMaterialParity.test.ts',
  'packages/walkaround-hybrid/src/ddgi/__tests__/ddgiGlossyProbeBounce.test.ts',
], [
  'packages/walkaround-hybrid/src/shaders/ggxBrdf.wgsl.ts',
  'packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts',
  'packages/walkaround-hybrid/src/shaders/shadingTerms.wgsl.ts',
  'packages/walkaround-hybrid/src/shaders/restirPHat.wgsl.ts',
  'packages/walkaround-hybrid/src/shaders/restirGiMaterial.wgsl.ts',
  'packages/walkaround-rc/src/wgsl/rcBrdf.wgsl.ts',
]);

const PT_WEBGL2_SCALARS = evidence('packer+shader', [
  'packages/pt-webgl2/src/scene/materialsTexture.test.ts',
  'packages/pt-webgl2/src/glsl/composeTraceGlsl.test.ts',
  'packages/pt-webgl2/src/scene/materialStrideParity.test.ts',
], [
  'packages/pt-webgl2/src/scene/materialsTexture.ts',
  'packages/pt-webgl2/src/glsl/shader/structs/material_mapped_rich.glsl.ts',
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
  'packages/pt-webgl2/src/glsl/shader/structs/material_mapped_rich.glsl.ts',
  'packages/pt-webgl2/src/glsl/render/get_surface_record_function.glsl.js',
  'packages/pt-webgl2/src/glsl/render/attenuate_hit_function.glsl.js',
]);

const PT_WEBGL2_DOUBLE_SIDED = evidence('packer+shader', [
  'packages/pt-webgl2/src/scene/materialsTexture.test.ts',
  'packages/pt-webgl2/src/glsl/doubleSidedTierEvidence.test.ts',
], [
  'packages/pt-webgl2/src/scene/materialsTexture.ts',
  'packages/pt-webgl2/src/glsl/shader/structs/material_basic.glsl.ts',
  'packages/pt-webgl2/src/glsl/render/get_surface_record_basic.glsl.ts',
  'packages/pt-webgl2/src/glsl/render/attenuate_hit_basic.glsl.ts',
  'packages/pt-webgl2/src/glsl/shader/structs/material_scalar_rich.glsl.ts',
  'packages/pt-webgl2/src/glsl/render/get_surface_record_scalar_rich.glsl.ts',
  'packages/pt-webgl2/src/glsl/render/attenuate_hit_scalar_rich.glsl.ts',
  'packages/pt-webgl2/src/glsl/shader/structs/material_mapped_pbr.glsl.ts',
  'packages/pt-webgl2/src/glsl/render/get_surface_record_mapped_pbr.glsl.ts',
  'packages/pt-webgl2/src/glsl/render/attenuate_hit_mapped_pbr.glsl.ts',
  'packages/pt-webgl2/src/glsl/shader/structs/material_mapped_rich.glsl.ts',
  'packages/pt-webgl2/src/glsl/render/get_surface_record_function.glsl.js',
  'packages/pt-webgl2/src/glsl/render/attenuate_hit_function.glsl.js',
], [
  {
    path: 'packages/pt-webgl2/src/scene/materialsTexture.test.ts',
    includes: [
      'packs authored double-sided opaque surfaces while preserving closed-volume exit traversal',
    ],
  },
  {
    path: 'packages/pt-webgl2/src/glsl/doubleSidedTierEvidence.test.ts',
    includes: [
      'consumes the packed side lane for radiance and visibility',
      "'basic'",
      "'scalar-rich'",
      "'mapped-pbr'",
      "'mapped-rich'",
    ],
  },
  {
    path: 'packages/pt-webgl2/src/scene/materialsTexture.ts',
    includes: ["if (m.doubleSided === true || (!isThinFilm && transmission > 0.0))"],
  },
]);

const PT_WEBGPU_SCALARS = evidence('packer+shader', [
  'packages/pt-webgpu/src/__tests__/scenePack.materials.test.ts',
  'packages/pt-webgpu/src/__tests__/implicitMeshEmitter.test.ts',
  'packages/pt-webgpu/src/__tests__/volumetricSss.test.ts',
  'packages/pt-webgpu/src/__tests__/wgslContract.test.ts',
], [
  'packages/pt-webgpu/src/scene/materialPacking.ts',
  'packages/pt-webgpu/src/scene/emitterPacking.ts',
  'packages/pt-webgpu/src/wgsl/pathTrace/material.wgsl.ts',
  'packages/pt-webgpu/src/wgsl/pathTrace/kernel.wgsl.ts',
  'packages/pt-webgpu/src/wgsl/pathTrace/bsdf.wgsl.ts',
], [
  {
    path: 'packages/pt-webgpu/src/__tests__/scenePack.materials.test.ts',
    includes: [
      'packs layered/spectral/thin-film summaries',
      'packs authored dispersionAbbe independently of spectral attenuation',
      'SPEC-01 specular scalar packing',
      'VOL-THICKNESS KHR_materials_volume scalar packing',
    ],
  },
  {
    path: 'packages/pt-webgpu/src/__tests__/volumetricSss.test.ts',
    includes: [
      'σ_a packing from attenuationColor / attenuationDistance',
      'volume thickness packing and attenuation-distance clamp',
    ],
  },
  {
    path: 'packages/pt-webgpu/src/wgsl/pathTrace/kernel.wgsl.ts',
	    includes: [
	      'WS4 volumetric random walk',
	      'legacy Beer-Lambert + forward-scatter-radiance fallback',
	    ],
	  },
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

const PT_WEBGPU_RICH_LOBES = evidence('packer+shader', [
  'packages/pt-webgpu/src/__tests__/scenePack.materials.test.ts',
  'packages/pt-webgpu/src/__tests__/materialTextures.test.ts',
  'packages/pt-webgpu/src/__tests__/extensionLobeReference.test.ts',
  'packages/pt-webgpu/src/__tests__/wgslContract.test.ts',
], [
  'packages/pt-webgpu/src/scene/materialPacking.ts',
  'packages/pt-webgpu/src/scene/materialTextures.ts',
  'packages/pt-webgpu/src/wgsl/pathTrace/material.wgsl.ts',
  'packages/pt-webgpu/src/wgsl/pathTrace/bsdf.wgsl.ts',
  'packages/pt-webgpu/src/wgsl/pathTrace/kernel.wgsl.ts',
], [
  {
    path: 'packages/pt-webgpu/src/__tests__/scenePack.materials.test.ts',
    includes: ['SPEC-01 specular scalar packing'],
  },
  {
    path: 'packages/pt-webgpu/src/__tests__/materialTextures.test.ts',
    includes: [
      'collects extension-lobe maps into the correct color-space source arrays',
      'packs extension-lobe wrap modes and UV metadata',
    ],
  },
]);

const PT_WEBGPU_DOUBLE_SIDED = evidence('packer+shader', [
  'packages/pt-webgpu/src/__tests__/scenePack.materials.test.ts',
  'packages/pt-webgpu/src/__tests__/doubleSidedTraversal.test.ts',
], [
  'packages/pt-webgpu/src/scene/materialPacking.ts',
  'packages/pt-webgpu/src/wgsl/pathTrace/material.wgsl.ts',
], [
  {
    path: 'packages/pt-webgpu/src/__tests__/scenePack.materials.test.ts',
    includes: ['packs MaterialSpec.doubleSided as bit2 and defaults it to false'],
  },
  {
    path: 'packages/pt-webgpu/src/wgsl/pathTrace/material.wgsl.ts',
    includes: ['return mat.doubleSided || mat.transmission > 0.0;'],
    excludes: ['return mat.doubleSided || mat.transmission > 1e-6;'],
  },
]);

const MATERIAL_NATIVE_EVIDENCE: Record<BackendWithMaterialEvidence, Record<string, MaterialNativeEvidence>> = {
  'walkaround-hybrid': {
    ...group(['emissive', 'emissiveIntensity'], WALKAROUND_EMISSIVE),
    emissiveMap: WALKAROUND_EMISSIVE_MAP,
    ...group(['alphaMode', 'alphaCutoff', 'opacity', 'alphaMap'], WALKAROUND_ALPHA),
    doubleSided: WALKAROUND_DOUBLE_SIDED,
    ...group(['lightMap', 'lightMapIntensity'], WALKAROUND_LIGHT_MAP),
    ...group([
      'clearcoatMap', 'clearcoatRoughnessMap', 'clearcoatNormalMap',
      'clearcoatNormalScale', 'sheenColorMap', 'sheenRoughnessMap',
      'iridescenceMap', 'iridescenceThicknessMap', 'anisotropyMap',
      'specularColorMap', 'specularIntensityMap', 'sheen', 'sheenColor',
      'sheenRoughness', 'clearcoat', 'clearcoatRoughness', 'iridescence',
      'iridescenceIor', 'iridescenceThicknessRange', 'specularIntensity',
      'specularColor', 'anisotropy', 'anisotropyRotation',
    ], WALKAROUND_RICH_LOBES),
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
      'lightMap', 'frontLayer', 'backLayer',
    ], PT_WEBGL2_TEXTURES),
    doubleSided: PT_WEBGL2_DOUBLE_SIDED,
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
      'thinFilmStack',
    ], PT_WEBGPU_SCALARS),
    ...group([
      'baseColorMap', 'normalMap', 'roughnessMap', 'metallicMap',
      'transmissionMap', 'emissiveMap', 'alphaMap', 'aoMap',
      'bumpMap', 'lightMap', 'frontLayer', 'backLayer',
    ], PT_WEBGPU_TEXTURES),
    ...group([
      'clearcoatMap', 'clearcoatRoughnessMap', 'clearcoatNormalMap',
      'sheenColorMap', 'sheenRoughnessMap', 'iridescenceMap',
      'iridescenceThicknessMap', 'specularColorMap', 'specularIntensityMap',
      'specularIntensity', 'specularColor',
    ], PT_WEBGPU_RICH_LOBES),
    doubleSided: PT_WEBGPU_DOUBLE_SIDED,
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

  it('pins explicit proof snippets where a broad evidence file would be ambiguous', () => {
    for (const [backend, rows] of Object.entries(MATERIAL_NATIVE_EVIDENCE)) {
      for (const [field, ev] of Object.entries(rows)) {
        for (const snippet of ev.snippets ?? []) {
          const text = readFileSync(resolve(REPO_ROOT, snippet.path), 'utf8');
          for (const needle of snippet.includes) {
            expect(
              text.includes(needle),
              `${backend}.${field}: ${snippet.path} must contain ${needle}`,
            ).toBe(true);
          }
          for (const needle of snippet.excludes ?? []) {
            expect(
              text.includes(needle),
              `${backend}.${field}: ${snippet.path} must not contain ${needle}`,
            ).toBe(false);
          }
        }
      }
    }
  });
});
