/**
 * GATE-02B — every `approximate` material promise must carry concrete evidence.
 *
 * Native rows already have GATE-02. Approximate rows are just as risky: without
 * an exhaustive evidence matrix, a backend can quietly promise "handled enough"
 * while the actual renderer path, warning surface, or promotion tail drifts.
 *
 * This test stays deliberately static. Backend package tests own the math and
 * shader assertions; this file makes the approximate ledger rows auditably
 * tied to source files and residual-truthfulness surfaces.
 */

import { describe, expect, it } from 'vitest';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { BACKEND_PROMISE_LEDGER } from '../engine/promiseLedger.js';

type BackendWithApproximateEvidence = 'walkaround-hybrid' | 'pt-webgl2' | 'pt-webgpu';
type EvidenceKind =
  | 'quantized-packer+shader'
  | 'atlas+truthfulness-surface'
  | 'shared-vertex-displacement'
  | 'pathtracer-feature-with-proof-tail'
  | 'volume-approximation';

interface EvidenceSnippet {
  readonly path: string;
  readonly includes: readonly string[];
}

interface MaterialApproximateEvidence {
  readonly kind: EvidenceKind;
  readonly reason: string;
  readonly tests: readonly string[];
  readonly sources: readonly string[];
  readonly snippets?: readonly EvidenceSnippet[];
}

function evidence(
  kind: EvidenceKind,
  reason: string,
  tests: readonly string[],
  sources: readonly string[],
  snippets?: readonly EvidenceSnippet[],
): MaterialApproximateEvidence {
  return { kind, reason, tests, sources, ...(snippets != null ? { snippets } : {}) };
}

function group(
  fields: readonly string[],
  ev: MaterialApproximateEvidence,
): Record<string, MaterialApproximateEvidence> {
  return Object.fromEntries(fields.map((field) => [field, ev]));
}

const WALKAROUND_COMPACT_TRIANGLE = evidence(
  'quantized-packer+shader',
  'packed into compact per-triangle lanes and consumed by the realtime GI shaders',
  [
    'packages/walkaround-hybrid/src/restir/__tests__/roughMetalPacking.test.ts',
    'packages/walkaround-hybrid/src/__tests__/materialTextureAtlas.test.ts',
    'packages/walkaround-hybrid/src/__tests__/consumedMaterialFields.test.ts',
  ],
  [
    'packages/walkaround-hybrid/src/restir/packingHelpers.ts',
    'packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts',
    'packages/walkaround-hybrid/src/shaders/shade.wgsl.ts',
    'packages/walkaround-hybrid/src/restir/consumedMaterialFields.ts',
  ],
  [
    {
      path: 'packages/walkaround-hybrid/src/restir/packingHelpers.ts',
      includes: [
        'export function packBVHIndexWFromCore',
        'export function packBVHRoughMetalFromCore',
        'export function packBVHBeerColorsFromCore',
      ],
    },
    {
      path: 'packages/walkaround-hybrid/src/restir/__tests__/roughMetalPacking.test.ts',
      includes: [
        'GLTF-unlit',
        'Scalar alpha cutout',
        'AO map strength',
      ],
    },
  ],
);

const WALKAROUND_ALPHA = evidence(
  'atlas+truthfulness-surface',
  'camera/OIT alpha is rendered, while fractional transport remains explicitly approximate',
  [
    'packages/walkaround-hybrid/src/restir/__tests__/roughMetalPacking.test.ts',
    'packages/walkaround-hybrid/src/__tests__/materialTextureAtlas.test.ts',
    'packages/walkaround-hybrid/src/__tests__/transparentAlphaTransportContract.test.ts',
    'packages/walkaround-hybrid/src/__tests__/transparentOitMaterialParity.test.ts',
    'packages/walkaround-hybrid/src/__tests__/consumedMaterialFields.test.ts',
  ],
  [
    'packages/walkaround-hybrid/src/restir/packingHelpers.ts',
    'packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts',
    'packages/walkaround-hybrid/src/shaders/transparentOit.wgsl.ts',
    'packages/walkaround-hybrid/src/restir/consumedMaterialFields.ts',
  ],
  [
    {
      path: 'packages/walkaround-hybrid/src/restir/consumedMaterialFields.ts',
      includes: [
        'collectApproximateAlphaBlendPrimitiveIds',
        'ReSTIR/GI participation remains approximate',
      ],
    },
  ],
);

const WALKAROUND_ATLAS_MAPS = evidence(
  'atlas+truthfulness-surface',
  'readable material maps are atlas-backed and sampled by shade/traversal, with compact-GI limits',
  [
    'packages/walkaround-hybrid/src/__tests__/materialTextureAtlas.test.ts',
    'packages/walkaround-hybrid/src/__tests__/restirDiMaterialParity.test.ts',
    'packages/walkaround-hybrid/src/__tests__/restirGiMaterialParity.test.ts',
    'packages/walkaround-hybrid/src/__tests__/consumedMaterialFields.test.ts',
  ],
  [
    'packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts',
    'packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts',
    'packages/walkaround-hybrid/src/shaders/shade.wgsl.ts',
    'packages/walkaround-hybrid/src/shaders/restirGiMaterial.wgsl.ts',
    'packages/walkaround-hybrid/src/restir/consumedMaterialFields.ts',
  ],
  [
    {
      path: 'packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts',
      includes: [
        "field: 'baseColorMap'",
        "field: 'normalMap'",
        "field: 'thicknessMap'",
      ],
    },
    {
      path: 'packages/walkaround-hybrid/src/__tests__/materialTextureAtlas.test.ts',
      includes: [
        'shade and traversal sample material maps from the shared atlas module',
        'packs thicknessMap as a linear G-channel atlas slot for Beer-Lambert tinting',
      ],
    },
  ],
);

const WALKAROUND_EMISSIVE_MAP = evidence(
  'atlas+truthfulness-surface',
  'visible/direct mapped emission is consumed, while all-path texel-PDF exactness is still a promotion tail',
  [
    'packages/walkaround-hybrid/src/__tests__/materialTextureAtlas.test.ts',
    'packages/walkaround-hybrid/src/restir/__tests__/emissiveLePacking.test.ts',
    'packages/walkaround-hybrid/src/__tests__/consumedMaterialFields.test.ts',
    'packages/walkaround-hybrid/src/__tests__/restirGiMaterialParity.test.ts',
  ],
  [
    'packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts',
    'packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts',
    'packages/walkaround-hybrid/src/shaders/restirGiMaterial.wgsl.ts',
    'packages/walkaround-hybrid/src/restir/consumedMaterialFields.ts',
  ],
  [
    {
      path: 'packages/walkaround-hybrid/src/restir/consumedMaterialFields.ts',
      includes: [
        'EMISSIVE_MAP_TEXEL_PDF_APPROXIMATION_DETAILS',
        'global-exact-texel-alias-pdf',
      ],
    },
  ],
);

const WALKAROUND_LIGHT_MAP = evidence(
  'atlas+truthfulness-surface',
  'light maps are camera-visible baked radiance, not global transport participants',
  [
    'packages/walkaround-hybrid/src/__tests__/materialTextureAtlas.test.ts',
    'packages/walkaround-hybrid/src/__tests__/transparentOitMaterialParity.test.ts',
    'packages/walkaround-hybrid/src/__tests__/consumedMaterialFields.test.ts',
  ],
  [
    'packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts',
    'packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts',
    'packages/walkaround-hybrid/src/shaders/transparentOit.wgsl.ts',
    'packages/walkaround-hybrid/src/restir/consumedMaterialFields.ts',
  ],
  [
    {
      path: 'packages/walkaround-hybrid/src/restir/consumedMaterialFields.ts',
      includes: [
        'LIGHT_MAP_CAMERA_VISIBLE_APPROXIMATION_DETAILS',
        'global-transport-light-map-participation',
      ],
    },
  ],
);

const WALKAROUND_RICH_LOBES = evidence(
  'atlas+truthfulness-surface',
  'rich lobes are consumed by shade/DI/GI payloads but still await furnace/reference promotion',
  [
    'packages/walkaround-hybrid/src/__tests__/materialTextureAtlas.test.ts',
    'packages/walkaround-hybrid/src/__tests__/restirDiMaterialParity.test.ts',
    'packages/walkaround-hybrid/src/__tests__/restirGiMaterialParity.test.ts',
    'packages/walkaround-hybrid/src/__tests__/consumedMaterialFields.test.ts',
  ],
  [
    'packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts',
    'packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts',
    'packages/walkaround-hybrid/src/shaders/shade.wgsl.ts',
    'packages/walkaround-hybrid/src/shaders/restirGiMaterial.wgsl.ts',
    'packages/walkaround-hybrid/src/restir/consumedMaterialFields.ts',
  ],
  [
    {
      path: 'packages/walkaround-hybrid/src/restir/consumedMaterialFields.ts',
      includes: [
        'RICH_MATERIAL_GI_APPROXIMATION_DETAILS',
        'collectApproximateRichMaterialPrimitiveFields',
      ],
    },
    {
      path: 'packages/walkaround-hybrid/src/__tests__/materialTextureAtlas.test.ts',
      includes: [
        'fn sampleSpecularControls(',
        'fn sampleClearcoatControls(',
        'fn sampleSheenControls(',
        'fn sampleAnisotropyControls(',
        'fn sampleIridescenceControls(',
        'fn applyBumpMapForHit(',
      ],
    },
  ],
);

const WALKAROUND_FACE_LAYERS = evidence(
  'atlas+truthfulness-surface',
  'per-face absorption layers are scalar-atlas metadata with explicit layer-normal residual warnings',
  [
    'packages/walkaround-hybrid/src/__tests__/materialTextureAtlas.test.ts',
    'packages/walkaround-hybrid/src/__tests__/restirDiMaterialParity.test.ts',
    'packages/walkaround-hybrid/src/__tests__/restirGiMaterialParity.test.ts',
    'packages/walkaround-hybrid/src/__tests__/transparentOitMaterialParity.test.ts',
    'packages/walkaround-hybrid/src/__tests__/consumedMaterialFields.test.ts',
  ],
  [
    'packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts',
    'packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts',
    'packages/walkaround-hybrid/src/shaders/shade.wgsl.ts',
    'packages/walkaround-hybrid/src/shaders/restirGiMaterial.wgsl.ts',
    'packages/walkaround-hybrid/src/ddgi/wgsl/probeUpdateRays.wgsl.ts',
    'packages/walkaround-hybrid/src/restir/consumedMaterialFields.ts',
  ],
  [
    {
      path: 'packages/walkaround-hybrid/src/__tests__/materialTextureAtlas.test.ts',
      includes: [
        'packs frontLayer/backLayer transmission, roughness, and layer-local normal maps',
      ],
    },
    {
      path: 'packages/walkaround-hybrid/src/restir/consumedMaterialFields.ts',
      includes: [
        'frontLayer.normalMap',
        'backLayer.normalScale',
      ],
    },
  ],
);

const WALKAROUND_VOLUME_SCATTERING = evidence(
  'volume-approximation',
  'sigma_s and anisotropy are scalar-atlas metadata consumed by bounded single-scatter tinting, not full volumetric path tracing',
  [
    'packages/walkaround-hybrid/src/__tests__/materialTextureAtlas.test.ts',
    'packages/walkaround-hybrid/src/__tests__/restirDiMaterialParity.test.ts',
    'packages/walkaround-hybrid/src/__tests__/restirGiMaterialParity.test.ts',
    'packages/walkaround-hybrid/src/ddgi/__tests__/ddgiGlossyProbeBounce.test.ts',
  ],
  [
    'packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts',
    'packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts',
    'packages/walkaround-hybrid/src/shaders/shade.wgsl.ts',
    'packages/walkaround-hybrid/src/shaders/restirPHat.wgsl.ts',
    'packages/walkaround-hybrid/src/shaders/restirGiMaterial.wgsl.ts',
    'packages/walkaround-hybrid/src/ddgi/wgsl/probeUpdateRays.wgsl.ts',
  ],
  [
    {
      path: 'packages/walkaround-hybrid/src/__tests__/materialTextureAtlas.test.ts',
      includes: [
        'packs volume scattering sigmaS and anisotropy metadata',
      ],
    },
    {
      path: 'packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts',
      includes: [
        'fn applyVolumeScatteringApproximation(',
      ],
    },
  ],
);

const SHARED_VERTEX_DISPLACEMENT = evidence(
  'shared-vertex-displacement',
  'CPU-readable displacement maps become shared-BVH geometry before BVH build; bounded uniform microdisplacement is opt-in, adaptive/error-bounded microgeometry is not contracted',
  [
    'packages/shared-bvh/src/__tests__/scenePack.test.ts',
    'packages/walkaround-hybrid/src/restir/__tests__/bvhCoreMaterialResolver.test.ts',
    'packages/walkaround-hybrid/src/__tests__/mutationMatrix.test.ts',
    'packages/pt-webgl2/src/__tests__/engineContract.test.ts',
    'packages/pt-webgpu/src/__tests__/liteTierCapabilities.test.ts',
  ],
  [
    'packages/shared-bvh/src/vertexDisplacement.ts',
    'packages/shared-bvh/src/scenePack.ts',
    'packages/shared-bvh/src/worldSpaceMerge.ts',
  ],
  [
    {
      path: 'packages/shared-bvh/src/vertexDisplacement.ts',
      includes: [
        'maybeDisplaceMeshPositions',
        'maybeMicrodisplaceMeshGeometry',
      ],
    },
  ],
);

const PT_WEBGL2_UNLIT = evidence(
  'pathtracer-feature-with-proof-tail',
  'unlit is terminal base-color visibility, not emissive light transport',
  [
    'packages/pt-webgl2/src/scene/materialsTexture.test.ts',
    'packages/pt-webgl2/src/glsl/composeTraceGlsl.test.ts',
  ],
  [
    'packages/pt-webgl2/src/scene/materialsTexture.ts',
    'packages/pt-webgl2/src/glsl/composeTraceGlsl.ts',
    'packages/pt-webgl2/src/glsl/shader/structs/material_struct.glsl.js',
  ],
  [
    {
      path: 'packages/pt-webgl2/src/glsl/composeTraceGlsl.test.ts',
      includes: [
        'bool unlit;',
        'if ( material.unlit )',
      ],
    },
  ],
);

const PT_WEBGL2_THICKNESS = evidence(
  'volume-approximation',
  'thickness/thicknessMap clamp Beer-Lambert attenuation but remain thin-shell approximations',
  [
    'packages/pt-webgl2/src/scene/materialsTexture.test.ts',
    'packages/pt-webgl2/src/scene/materialStrideParity.test.ts',
    'packages/pt-webgl2/src/__tests__/engineContract.test.ts',
  ],
  [
    'packages/pt-webgl2/src/scene/materialsTexture.ts',
    'packages/pt-webgl2/src/glsl/render/get_surface_record_function.glsl.js',
    'packages/pt-webgl2/src/glsl/render/attenuate_hit_function.glsl.js',
  ],
  [
    {
      path: 'packages/pt-webgl2/src/scene/materialsTexture.test.ts',
      includes: [
        'packs thicknessMap layer, thickness scalar, UV1 bit, transform, and wrap mode',
      ],
    },
  ],
);

const PT_WEBGL2_SCATTER_RGB = evidence(
  'volume-approximation',
  'per-channel sigma_s is packed, but the WebGL2 SSS model remains a simplified single-scatter path',
  [
    'packages/pt-webgl2/src/scene/materialsTexture.test.ts',
    'packages/pt-webgl2/src/glsl/composeTraceGlsl.test.ts',
  ],
  [
    'packages/pt-webgl2/src/scene/materialsTexture.ts',
    'packages/pt-webgl2/src/glsl/shader/structs/material_struct.glsl.js',
    'packages/pt-webgl2/src/glsl/shader/bsdf/bsdf_functions.glsl.js',
  ],
  [
    {
      path: 'packages/pt-webgl2/src/scene/materialsTexture.test.ts',
      includes: [
        'scatteringCoefficientRGB packs per-channel sigmaS override and majorant sigmaT',
        'scatteringCoefficientRGB alone activates translucent SSS and packs sigmaS',
      ],
    },
  ],
);

const PT_WEBGPU_UNLIT = evidence(
  'pathtracer-feature-with-proof-tail',
  'unlit is terminal base-color visibility, not emissive light transport',
  [
    'packages/pt-webgpu/src/__tests__/scenePack.materials.test.ts',
    'packages/pt-webgpu/src/__tests__/inverseSession.test.ts',
    'packages/pt-webgpu/src/__tests__/wgslContract.test.ts',
  ],
  [
    'packages/pt-webgpu/src/scene/materialPacking.ts',
    'packages/pt-webgpu/src/wgsl/pathTrace/material.wgsl.ts',
    'packages/pt-webgpu/src/wgsl/pathTrace/kernel.wgsl.ts',
  ],
  [
    {
      path: 'packages/pt-webgpu/src/__tests__/scenePack.materials.test.ts',
      includes: [
        'packs MaterialSpec.shadingModel=unlit as bit1 in vec4 #26 .w',
      ],
    },
  ],
);

const PT_WEBGPU_THICKNESS = evidence(
  'volume-approximation',
  'volume thickness is packed and sampled, but closed-surface thin-shell integration is still approximate',
  [
    'packages/pt-webgpu/src/__tests__/scenePack.materials.test.ts',
    'packages/pt-webgpu/src/__tests__/materialTextures.test.ts',
    'packages/pt-webgpu/src/__tests__/volumetricSss.test.ts',
  ],
  [
    'packages/pt-webgpu/src/scene/materialPacking.ts',
    'packages/pt-webgpu/src/scene/materialTextures.ts',
    'packages/pt-webgpu/src/wgsl/pathTrace/material.wgsl.ts',
    'packages/pt-webgpu/src/wgsl/pathTrace/kernel.wgsl.ts',
  ],
  [
    {
      path: 'packages/pt-webgpu/src/__tests__/scenePack.materials.test.ts',
      includes: [
        'VOL-THICKNESS KHR_materials_volume scalar packing',
        'marks presence when only thicknessMap is authored',
      ],
    },
    {
      path: 'packages/pt-webgpu/src/__tests__/materialTextures.test.ts',
      includes: [
        'collects thicknessMap as LINEAR KHR volume data and packs layer, UV, wrap, and transform',
      ],
    },
  ],
);

const PT_WEBGPU_RICH_LOBES = evidence(
  'pathtracer-feature-with-proof-tail',
  'full-tier rich lobes are packed/sampled, while specialty-integrator and furnace proof tails keep rows approximate',
  [
    'packages/pt-webgpu/src/__tests__/scenePack.materials.test.ts',
    'packages/pt-webgpu/src/__tests__/materialTextures.test.ts',
    'packages/pt-webgpu/src/__tests__/untestedMaterialMaps.test.ts',
    'packages/pt-webgpu/src/__tests__/extensionLobeReference.test.ts',
    'packages/pt-webgpu/src/__tests__/ggxAnisotropicBrdf.test.ts',
    'packages/pt-webgpu/src/__tests__/adjointHarness.test.ts',
  ],
  [
    'packages/pt-webgpu/src/scene/materialPacking.ts',
    'packages/pt-webgpu/src/scene/materialTextures.ts',
    'packages/pt-webgpu/src/wgsl/pathTrace/material.wgsl.ts',
    'packages/pt-webgpu/src/wgsl/pathTrace/bsdf.wgsl.ts',
    'packages/pt-webgpu/src/wgsl/pathTrace/kernel.wgsl.ts',
  ],
  [
    {
      path: 'packages/pt-webgpu/src/__tests__/scenePack.materials.test.ts',
      includes: [
        'SPEC-01 specular scalar packing',
      ],
    },
    {
      path: 'packages/pt-webgpu/src/__tests__/materialTextures.test.ts',
      includes: [
        'collects extension-lobe maps into the correct color-space source arrays',
        'packs extension-lobe wrap modes and UV metadata',
      ],
    },
    {
      path: 'packages/pt-webgpu/src/__tests__/untestedMaterialMaps.test.ts',
      includes: [
        'anisotropyMap: layer id packed at descriptor float offset b+22',
        'WGSL materialAnisotropy reads vec4[5].z',
      ],
    },
  ],
);

const MATERIAL_APPROXIMATE_EVIDENCE: Record<BackendWithApproximateEvidence, Record<string, MaterialApproximateEvidence>> = {
  'walkaround-hybrid': {
    ...group([
      'baseColor', 'roughness', 'metallic', 'shadingModel',
      'transmission', 'ior', 'attenuationColor', 'attenuationDistance',
      'thickness', 'aoMapIntensity',
    ], WALKAROUND_COMPACT_TRIANGLE),
    ...group(['alphaMode', 'alphaCutoff', 'opacity', 'alphaMap'], WALKAROUND_ALPHA),
    ...group([
      'baseColorMap', 'roughnessMap', 'metallicMap', 'normalMap', 'normalScale',
      'transmissionMap', 'thicknessMap', 'aoMap',
    ], WALKAROUND_ATLAS_MAPS),
    emissiveMap: WALKAROUND_EMISSIVE_MAP,
    ...group(['lightMap', 'lightMapIntensity'], WALKAROUND_LIGHT_MAP),
    ...group([
      'clearcoatMap', 'clearcoatRoughnessMap', 'clearcoatNormalMap',
      'clearcoatNormalScale', 'sheenColorMap', 'sheenRoughnessMap',
      'iridescenceMap', 'iridescenceThicknessMap', 'anisotropyMap',
      'specularColorMap', 'specularIntensityMap', 'bumpMap', 'bumpScale',
      'sheen', 'sheenColor', 'sheenRoughness', 'clearcoat',
      'clearcoatRoughness', 'iridescence', 'iridescenceIor',
      'iridescenceThicknessRange', 'specularIntensity', 'specularColor',
      'anisotropy', 'anisotropyRotation',
    ], WALKAROUND_RICH_LOBES),
    ...group(['frontLayer', 'backLayer'], WALKAROUND_FACE_LAYERS),
    ...group([
      'scatteringCoefficient',
      'scatteringAnisotropy',
      'scatteringCoefficientRGB',
    ], WALKAROUND_VOLUME_SCATTERING),
    ...group(['displacementMap', 'displacementScale', 'displacementBias', 'displacementSubdivisions'], SHARED_VERTEX_DISPLACEMENT),
  },
  'pt-webgl2': {
    shadingModel: PT_WEBGL2_UNLIT,
    ...group(['thickness', 'thicknessMap'], PT_WEBGL2_THICKNESS),
    ...group(['displacementMap', 'displacementScale', 'displacementBias', 'displacementSubdivisions'], SHARED_VERTEX_DISPLACEMENT),
    scatteringCoefficientRGB: PT_WEBGL2_SCATTER_RGB,
  },
  'pt-webgpu': {
    shadingModel: PT_WEBGPU_UNLIT,
    ...group(['thickness', 'thicknessMap'], PT_WEBGPU_THICKNESS),
    ...group([
      'clearcoatMap', 'clearcoatRoughnessMap', 'clearcoatNormalMap',
      'sheenColorMap', 'sheenRoughnessMap', 'iridescenceMap',
      'iridescenceThicknessMap', 'anisotropyMap', 'specularColorMap',
      'specularIntensityMap', 'specularIntensity', 'specularColor',
      'anisotropy', 'anisotropyRotation',
    ], PT_WEBGPU_RICH_LOBES),
    ...group(['displacementMap', 'displacementScale', 'displacementBias', 'displacementSubdivisions'], SHARED_VERTEX_DISPLACEMENT),
  },
};

const REPO_ROOT = resolve(fileURLToPath(new URL('../../../../', import.meta.url)));

function approximateMaterialRows(backend: BackendWithApproximateEvidence): string[] {
  return Object.entries(BACKEND_PROMISE_LEDGER[backend].supportDetails.materials)
    .filter(([, grade]) => grade === 'approximate')
    .map(([field]) => field)
    .sort();
}

describe('GATE-02B approximate material evidence', () => {
  it('has explicit renderer/proof-tail evidence for every approximate material row', () => {
    for (const backend of Object.keys(MATERIAL_APPROXIMATE_EVIDENCE) as BackendWithApproximateEvidence[]) {
      expect(
        Object.keys(MATERIAL_APPROXIMATE_EVIDENCE[backend]).sort(),
        `${backend} approximate material evidence rows`,
      ).toEqual(approximateMaterialRows(backend));
    }
  });

  it('does not accept empty evidence records', () => {
    for (const [backend, rows] of Object.entries(MATERIAL_APPROXIMATE_EVIDENCE)) {
      for (const [field, ev] of Object.entries(rows)) {
        expect(ev.reason.trim().length, `${backend}.${field} reason`).toBeGreaterThan(0);
        expect(ev.tests.length, `${backend}.${field} tests`).toBeGreaterThan(0);
        expect(ev.sources.length, `${backend}.${field} sources`).toBeGreaterThan(0);
      }
    }
  });

  it('points every evidence record at files that exist', () => {
    for (const [backend, rows] of Object.entries(MATERIAL_APPROXIMATE_EVIDENCE)) {
      for (const [field, ev] of Object.entries(rows)) {
        for (const path of [...ev.tests, ...ev.sources]) {
          expect(existsSync(resolve(REPO_ROOT, path)), `${backend}.${field}: ${path}`).toBe(true);
        }
      }
    }
  });

  it('pins explicit proof snippets where broad evidence files would be ambiguous', () => {
    for (const [backend, rows] of Object.entries(MATERIAL_APPROXIMATE_EVIDENCE)) {
      for (const [field, ev] of Object.entries(rows)) {
        for (const snippet of ev.snippets ?? []) {
          const text = readFileSync(resolve(REPO_ROOT, snippet.path), 'utf8');
          for (const needle of snippet.includes) {
            expect(
              text.includes(needle),
              `${backend}.${field}: ${snippet.path} must contain ${needle}`,
            ).toBe(true);
          }
        }
      }
    }
  });
});
