// materials.ts — glTF pbrMetallicRoughness + KHR extensions → core MaterialSpec.
//
// ORM texture note:
//   glTF stores roughness+metallic in a SINGLE texture:
//     G channel = roughnessTexture, B channel = metallicTexture
//   The same texture ref is supplied as both `roughnessMap` and `metallicMap`.
//   Backends sample G for roughness and B for metallic; atlas packers dedupe by
//   handle so this does not duplicate texture storage.
//
// doubleSided → alphaMode 'blend' is incorrect; doubleSided controls backface culling,
// not blending. It maps directly to MaterialSpec.doubleSided.
//
// KHR_materials_emissive_strength: scales emissiveIntensity (default 1).
//
// D13.5: per-extension parsers extracted to private helpers (_parse*Ext).
// Each returns a Partial<MaterialSpec> merged into the final spec at the end.
// This keeps the 215-line convertMaterial down to a flat base-PBR + merge.

import type { GltfJson, GltfMaterial } from './gltfTypes.js';
import type { MaterialSpec, TextureRef, Vec3 } from '@vitrum/core';
import { resolveTextureRef, type GltfTextureSourceExtension } from './textures.js';

export type GltfMaterialDiagnosticCode =
  | 'invalid-material-dispersion'
  | 'invalid-material-alpha-mode'
  | 'material-texture-not-found'
  | 'material-texture-unresolved'
  | 'material-texture-sampler-not-found'
  | 'invalid-material-texture-sampler'
  | 'spec-gloss-approximation'
  | 'spec-gloss-texture-alpha-approximation'
  | 'unknown-material-extension';

export interface GltfMaterialDiagnostic {
  readonly severity: 'warning' | 'error';
  readonly code: GltfMaterialDiagnosticCode;
  readonly path: string;
  readonly message: string;
  readonly materialIndex?: number;
  readonly textureIndex?: number;
  readonly samplerIndex?: number;
  readonly samplerProperty?: 'wrapS' | 'wrapT' | 'magFilter' | 'minFilter';
  readonly samplerValue?: number;
  readonly extensionName?: string;
}

export type GltfMaterialDiagnosticSink = (diagnostic: GltfMaterialDiagnostic) => void;

export class GltfMaterialImportError extends Error {
  readonly diagnostic: GltfMaterialDiagnostic;

  constructor(diagnostic: GltfMaterialDiagnostic) {
    super(diagnostic.message);
    this.name = 'GltfMaterialImportError';
    this.diagnostic = diagnostic;
  }
}

type MaterialTextureInfo = {
  readonly index: number;
  readonly texCoord?: number;
  readonly extensions?: Record<string, unknown>;
};

const KNOWN_KHR_EXTENSIONS = new Set([
  'KHR_materials_unlit',
  'KHR_materials_transmission',
  'KHR_materials_ior',
  'KHR_materials_volume',
  'KHR_materials_specular',
  'KHR_materials_sheen',
  'KHR_materials_clearcoat',
  'KHR_materials_iridescence',
  'KHR_materials_anisotropy',
  'KHR_materials_dispersion',
  'KHR_materials_emissive_strength',
  'KHR_texture_transform', // per-texture, handled at TextureRef level
  'KHR_materials_variants',
  'KHR_materials_pbrSpecularGlossiness',
]);

const PRESERVE_RAW_EXTENSION_KEYS = new Set([
  'KHR_materials_pbrSpecularGlossiness',
]);

// ── Per-extension parsers (D13.5) ──────────────────────────────────────────────

function _parseTransmissionExt(
  ext: Record<string, unknown>,
  handleMap: Map<number, unknown>,
  gltf: GltfJson | undefined,
  warnings: string[],
  materialName: string,
  materialIndex: number | undefined,
  textureSourceExtensions: readonly GltfTextureSourceExtension[],
  onDiagnostic: GltfMaterialDiagnosticSink | undefined,
): Partial<MaterialSpec> {
  const txExt = ext['KHR_materials_transmission'] as
    | { transmissionFactor?: number; transmissionTexture?: { index: number; texCoord?: number } }
    | undefined;
  if (!txExt) return {};
  const transmission = txExt.transmissionFactor ?? 0;
  const transmissionMap = resolveMaterialTextureRef(
    txExt.transmissionTexture,
    handleMap,
    gltf,
    materialTextureSourcePath(materialIndex, 'extensions.KHR_materials_transmission.transmissionTexture'),
    textureSourceExtensions,
    warnings,
    materialName,
    materialIndex,
    onDiagnostic,
  );
  return {
    ...(transmission > 0 ? { transmission } : {}),
    ...(transmissionMap ? { transmissionMap } : {}),
  };
}

function _parseVolumeExt(
  ext: Record<string, unknown>,
  handleMap: Map<number, unknown>,
  gltf: GltfJson | undefined,
  warnings: string[],
  materialName: string,
  materialIndex: number | undefined,
  textureSourceExtensions: readonly GltfTextureSourceExtension[],
  onDiagnostic: GltfMaterialDiagnosticSink | undefined,
): Partial<MaterialSpec> {
  const volExt = ext['KHR_materials_volume'] as
    | {
        thicknessFactor?: number;
        thicknessTexture?: { index: number; texCoord?: number };
        attenuationDistance?: number;
        attenuationColor?: [number, number, number];
      }
    | undefined;
  if (!volExt) return {};
  const thickness = volExt.thicknessFactor ?? 0;
  const thicknessMap = resolveMaterialTextureRef(
    volExt.thicknessTexture,
    handleMap,
    gltf,
    materialTextureSourcePath(materialIndex, 'extensions.KHR_materials_volume.thicknessTexture'),
    textureSourceExtensions,
    warnings,
    materialName,
    materialIndex,
    onDiagnostic,
  );
  const attenuationDistance = volExt.attenuationDistance ?? Infinity;
  const attenuationColor: Vec3 | undefined = volExt.attenuationColor;
  return {
    thickness,
    ...(thicknessMap ? { thicknessMap } : {}),
    attenuationDistance,
    ...(attenuationColor ? { attenuationColor } : {}),
  };
}

function _parseIorExt(ext: Record<string, unknown>): Partial<MaterialSpec> {
  const iorExt = ext['KHR_materials_ior'] as { ior?: number } | undefined;
  if (!iorExt) return {};
  return { ior: iorExt.ior ?? 1.5 };
}

function _parseSpecularExt(
  ext: Record<string, unknown>,
  handleMap: Map<number, unknown>,
  gltf: GltfJson | undefined,
  warnings: string[],
  materialName: string,
  materialIndex: number | undefined,
  textureSourceExtensions: readonly GltfTextureSourceExtension[],
  onDiagnostic: GltfMaterialDiagnosticSink | undefined,
): Partial<MaterialSpec> {
  const specExt = ext['KHR_materials_specular'] as
    | {
        specularFactor?: number;
        specularTexture?: { index: number; texCoord?: number };
        specularColorFactor?: [number, number, number];
        specularColorTexture?: { index: number; texCoord?: number };
      }
    | undefined;
  if (!specExt) return {};
  const specularIntensity = specExt.specularFactor ?? 1;
  const specularColor: Vec3 | undefined = specExt.specularColorFactor;
  const specularIntensityMap = resolveMaterialTextureRef(
    specExt.specularTexture,
    handleMap,
    gltf,
    materialTextureSourcePath(materialIndex, 'extensions.KHR_materials_specular.specularTexture'),
    textureSourceExtensions,
    warnings,
    materialName,
    materialIndex,
    onDiagnostic,
  );
  const specularColorMap = resolveMaterialTextureRef(
    specExt.specularColorTexture,
    handleMap,
    gltf,
    materialTextureSourcePath(materialIndex, 'extensions.KHR_materials_specular.specularColorTexture'),
    textureSourceExtensions,
    warnings,
    materialName,
    materialIndex,
    onDiagnostic,
  );
  return {
    ...(specularIntensity !== 1 ? { specularIntensity } : {}),
    ...(specularColor ? { specularColor } : {}),
    ...(specularIntensityMap ? { specularIntensityMap } : {}),
    ...(specularColorMap ? { specularColorMap } : {}),
  };
}

function _parseSheenExt(
  ext: Record<string, unknown>,
  handleMap: Map<number, unknown>,
  gltf: GltfJson | undefined,
  warnings: string[],
  materialName: string,
  materialIndex: number | undefined,
  textureSourceExtensions: readonly GltfTextureSourceExtension[],
  onDiagnostic: GltfMaterialDiagnosticSink | undefined,
): Partial<MaterialSpec> {
  const sheenExt = ext['KHR_materials_sheen'] as
    | {
        sheenColorFactor?: [number, number, number];
        sheenColorTexture?: { index: number; texCoord?: number };
        sheenRoughnessFactor?: number;
        sheenRoughnessTexture?: { index: number; texCoord?: number };
      }
    | undefined;
  if (!sheenExt) return {};
  const sheenColor: Vec3 | undefined = sheenExt.sheenColorFactor;
  const sheenRoughness = sheenExt.sheenRoughnessFactor ?? 0;
  const sheenColorMap = resolveMaterialTextureRef(
    sheenExt.sheenColorTexture,
    handleMap,
    gltf,
    materialTextureSourcePath(materialIndex, 'extensions.KHR_materials_sheen.sheenColorTexture'),
    textureSourceExtensions,
    warnings,
    materialName,
    materialIndex,
    onDiagnostic,
  );
  const sheenRoughnessMap = resolveMaterialTextureRef(
    sheenExt.sheenRoughnessTexture,
    handleMap,
    gltf,
    materialTextureSourcePath(materialIndex, 'extensions.KHR_materials_sheen.sheenRoughnessTexture'),
    textureSourceExtensions,
    warnings,
    materialName,
    materialIndex,
    onDiagnostic,
  );
  return {
    sheen: 1 as const,
    sheenRoughness,
    ...(sheenColor ? { sheenColor } : {}),
    ...(sheenColorMap ? { sheenColorMap } : {}),
    ...(sheenRoughnessMap ? { sheenRoughnessMap } : {}),
  };
}

function _parseClearcoatExt(
  ext: Record<string, unknown>,
  handleMap: Map<number, unknown>,
  gltf: GltfJson | undefined,
  warnings: string[],
  materialName: string,
  materialIndex: number | undefined,
  textureSourceExtensions: readonly GltfTextureSourceExtension[],
  onDiagnostic: GltfMaterialDiagnosticSink | undefined,
): Partial<MaterialSpec> {
  const ccExt = ext['KHR_materials_clearcoat'] as
    | {
        clearcoatFactor?: number;
        clearcoatTexture?: { index: number; texCoord?: number };
        clearcoatRoughnessFactor?: number;
        clearcoatRoughnessTexture?: { index: number; texCoord?: number };
        clearcoatNormalTexture?: { index: number; texCoord?: number; scale?: number };
      }
    | undefined;
  if (!ccExt) return {};
  const clearcoat = ccExt.clearcoatFactor ?? 0;
  const clearcoatRoughness = ccExt.clearcoatRoughnessFactor ?? 0;
  const clearcoatMap = resolveMaterialTextureRef(
    ccExt.clearcoatTexture,
    handleMap,
    gltf,
    materialTextureSourcePath(materialIndex, 'extensions.KHR_materials_clearcoat.clearcoatTexture'),
    textureSourceExtensions,
    warnings,
    materialName,
    materialIndex,
    onDiagnostic,
  );
  const clearcoatRoughnessMap = resolveMaterialTextureRef(
    ccExt.clearcoatRoughnessTexture,
    handleMap,
    gltf,
    materialTextureSourcePath(materialIndex, 'extensions.KHR_materials_clearcoat.clearcoatRoughnessTexture'),
    textureSourceExtensions,
    warnings,
    materialName,
    materialIndex,
    onDiagnostic,
  );
  const clearcoatNormalMap = resolveMaterialTextureRef(
    ccExt.clearcoatNormalTexture,
    handleMap,
    gltf,
    materialTextureSourcePath(materialIndex, 'extensions.KHR_materials_clearcoat.clearcoatNormalTexture'),
    textureSourceExtensions,
    warnings,
    materialName,
    materialIndex,
    onDiagnostic,
  );
  const clearcoatNormalScale = (ccExt.clearcoatNormalTexture as { scale?: number } | undefined)?.scale ?? 1;
  return {
    clearcoat,
    clearcoatRoughness,
    ...(clearcoatMap ? { clearcoatMap } : {}),
    ...(clearcoatRoughnessMap ? { clearcoatRoughnessMap } : {}),
    ...(clearcoatNormalMap ? { clearcoatNormalMap } : {}),
    ...(clearcoatNormalScale !== 1 ? { clearcoatNormalScale } : {}),
  };
}

function _parseIridescenceExt(
  ext: Record<string, unknown>,
  handleMap: Map<number, unknown>,
  gltf: GltfJson | undefined,
  warnings: string[],
  materialName: string,
  materialIndex: number | undefined,
  textureSourceExtensions: readonly GltfTextureSourceExtension[],
  onDiagnostic: GltfMaterialDiagnosticSink | undefined,
): Partial<MaterialSpec> {
  const iridExt = ext['KHR_materials_iridescence'] as
    | {
        iridescenceFactor?: number;
        iridescenceTexture?: { index: number; texCoord?: number };
        iridescenceIor?: number;
        iridescenceThicknessMinimum?: number;
        iridescenceThicknessMaximum?: number;
        iridescenceThicknessTexture?: { index: number; texCoord?: number };
      }
    | undefined;
  if (!iridExt) return {};
  const iridescence = iridExt.iridescenceFactor ?? 0;
  const iridescenceIor = iridExt.iridescenceIor ?? 1.3;
  const iridescenceThicknessRange: readonly [number, number] = [
    iridExt.iridescenceThicknessMinimum ?? 100,
    iridExt.iridescenceThicknessMaximum ?? 400,
  ];
  const iridescenceMap = resolveMaterialTextureRef(
    iridExt.iridescenceTexture,
    handleMap,
    gltf,
    materialTextureSourcePath(materialIndex, 'extensions.KHR_materials_iridescence.iridescenceTexture'),
    textureSourceExtensions,
    warnings,
    materialName,
    materialIndex,
    onDiagnostic,
  );
  const iridescenceThicknessMap = resolveMaterialTextureRef(
    iridExt.iridescenceThicknessTexture,
    handleMap,
    gltf,
    materialTextureSourcePath(materialIndex, 'extensions.KHR_materials_iridescence.iridescenceThicknessTexture'),
    textureSourceExtensions,
    warnings,
    materialName,
    materialIndex,
    onDiagnostic,
  );
  return {
    iridescence,
    iridescenceIor,
    iridescenceThicknessRange,
    ...(iridescenceMap ? { iridescenceMap } : {}),
    ...(iridescenceThicknessMap ? { iridescenceThicknessMap } : {}),
  };
}

function _parseAnisotropyExt(
  ext: Record<string, unknown>,
  handleMap: Map<number, unknown>,
  gltf: GltfJson | undefined,
  warnings: string[],
  materialName: string,
  materialIndex: number | undefined,
  textureSourceExtensions: readonly GltfTextureSourceExtension[],
  onDiagnostic: GltfMaterialDiagnosticSink | undefined,
): Partial<MaterialSpec> {
  const anisoExt = ext['KHR_materials_anisotropy'] as
    | {
        anisotropyStrength?: number;
        anisotropyRotation?: number;
        anisotropyTexture?: { index: number; texCoord?: number };
      }
    | undefined;
  if (!anisoExt) return {};
  const anisotropy = anisoExt.anisotropyStrength ?? 0;
  const anisotropyRotation = anisoExt.anisotropyRotation ?? 0;
  const anisotropyMap = resolveMaterialTextureRef(
    anisoExt.anisotropyTexture,
    handleMap,
    gltf,
    materialTextureSourcePath(materialIndex, 'extensions.KHR_materials_anisotropy.anisotropyTexture'),
    textureSourceExtensions,
    warnings,
    materialName,
    materialIndex,
    onDiagnostic,
  );
  return {
    anisotropy,
    anisotropyRotation,
    ...(anisotropyMap ? { anisotropyMap } : {}),
  };
}

function _parseDispersionExt(
  ext: Record<string, unknown>,
  warnings: string[],
  materialName: string,
  materialIndex: number | undefined,
  onDiagnostic: GltfMaterialDiagnosticSink | undefined,
): Partial<MaterialSpec> {
  const dispersionExt = ext['KHR_materials_dispersion'] as { dispersion?: number } | undefined;
  if (!dispersionExt || dispersionExt.dispersion === undefined) return {};
  const dispersion = dispersionExt.dispersion;
  if (!Number.isFinite(dispersion) || dispersion < 0) {
    throwMaterialImportError(warnings, onDiagnostic, {
      severity: 'error',
      code: 'invalid-material-dispersion',
      path: materialSourcePath(materialIndex, 'extensions.KHR_materials_dispersion.dispersion'),
      ...materialDiagnosticIndex(materialIndex),
      extensionName: 'KHR_materials_dispersion',
      message:
        `[vitrum/gltf-adapter] Material "${materialName}" uses KHR_materials_dispersion ` +
        `with invalid dispersion=${String(dispersion)}.`,
    });
  }
  if (dispersion === 0) return {};
  return {
    // KHR_materials_dispersion defines `dispersion = 20 / Abbe number`.
    dispersionAbbeNumber: 20 / dispersion,
  };
}

function _parseSpecularGlossinessExt(
  ext: Record<string, unknown>,
  handleMap: Map<number, unknown>,
  gltf: GltfJson | undefined,
  warnings: string[],
  materialName: string,
  alphaMode: MaterialSpec['alphaMode'],
  materialIndex: number | undefined,
  textureSourceExtensions: readonly GltfTextureSourceExtension[],
  onDiagnostic: GltfMaterialDiagnosticSink | undefined,
): Partial<MaterialSpec> {
  const sgExt = ext['KHR_materials_pbrSpecularGlossiness'] as
    | {
        diffuseFactor?: [number, number, number, number];
        diffuseTexture?: { index: number; texCoord?: number };
        specularFactor?: [number, number, number];
        glossinessFactor?: number;
        specularGlossinessTexture?: { index: number; texCoord?: number };
      }
    | undefined;
  if (!sgExt) return {};

  emitMaterialDiagnostic(warnings, onDiagnostic, {
    severity: 'warning',
    code: 'spec-gloss-approximation',
    path: materialSourcePath(materialIndex, 'extensions.KHR_materials_pbrSpecularGlossiness'),
    ...materialDiagnosticIndex(materialIndex),
    extensionName: 'KHR_materials_pbrSpecularGlossiness',
    message:
      `[vitrum/gltf-adapter] Material "${materialName}" uses archived ` +
      'KHR_materials_pbrSpecularGlossiness. Scalar diffuse/specular/glossiness ' +
      'values are converted approximately to metallic-roughness + specular fields.',
  });

  const diffuseFactor = sgExt.diffuseFactor ?? [1, 1, 1, 1];
  const specularFactor = sgExt.specularFactor ?? [1, 1, 1];
  const glossinessFactor = sgExt.glossinessFactor ?? 1;
  const diffuseTexture = resolveMaterialTextureRef(
    sgExt.diffuseTexture,
    handleMap,
    gltf,
    materialTextureSourcePath(
      materialIndex,
      'extensions.KHR_materials_pbrSpecularGlossiness.diffuseTexture',
    ),
    textureSourceExtensions,
    warnings,
    materialName,
    materialIndex,
    onDiagnostic,
  );
  const specularGlossinessTexture = resolveMaterialTextureRef(
    sgExt.specularGlossinessTexture,
    handleMap,
    gltf,
    materialTextureSourcePath(
      materialIndex,
      'extensions.KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture',
    ),
    textureSourceExtensions,
    warnings,
    materialName,
    materialIndex,
    onDiagnostic,
  );

  if (specularGlossinessTexture) {
    emitMaterialDiagnostic(warnings, onDiagnostic, {
      severity: 'warning',
      code: 'spec-gloss-texture-alpha-approximation',
      path: materialSourcePath(
        materialIndex,
        'extensions.KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture',
      ),
      ...materialDiagnosticIndex(materialIndex),
      extensionName: 'KHR_materials_pbrSpecularGlossiness',
      message:
        `[vitrum/gltf-adapter] Material "${materialName}" uses ` +
        'KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture. The RGB ' +
        'specular map is imported as specularColorMap. Raw import uses scalar ' +
        'glossinessFactor for roughness until loadGltfAndDecodeTextures() or ' +
        'decodeSceneTextures() can bake glossiness-in-alpha ' +
        'into a roughnessMap.',
    });
  }

  const opacity = diffuseFactor[3] < 1 && alphaMode !== 'opaque'
    ? { opacity: diffuseFactor[3] }
    : {};
  return {
    baseColor: [diffuseFactor[0], diffuseFactor[1], diffuseFactor[2]],
    roughness: 1 - clamp01(glossinessFactor),
    metallic: 0,
    specularColor: [specularFactor[0], specularFactor[1], specularFactor[2]],
    ...(diffuseTexture ? { baseColorMap: diffuseTexture } : {}),
    ...(specularGlossinessTexture ? { specularColorMap: specularGlossinessTexture } : {}),
    ...opacity,
  };
}

// ── Public API ─────────────────────────────────────────────────────────────────

/**
 * Convert one glTF material to a core MaterialSpec.
 *
 * @param gltfMat     - The raw glTF material object.
 * @param handleMap   - Map from glTF texture index → decoded image handle.
 * @param warnings    - Mutable array; unknown extensions are appended here.
 */
export function convertMaterial(
  gltfMat: GltfMaterial,
  handleMap: Map<number, unknown>,
  warnings: string[],
  gltf?: GltfJson,
  materialIndex?: number,
  textureSourceExtensions: readonly GltfTextureSourceExtension[] = [],
  onDiagnostic?: GltfMaterialDiagnosticSink,
): MaterialSpec {
  const pbr = gltfMat.pbrMetallicRoughness ?? {};
  const ext = gltfMat.extensions ?? {};

  // ── Base color ─────────────────────────────────────────────────────────────
  const baseColorFactor = pbr.baseColorFactor ?? [1, 1, 1, 1];
  const baseColor: Vec3 = [baseColorFactor[0], baseColorFactor[1], baseColorFactor[2]];
  const baseColorAlpha = baseColorFactor[3] ?? 1;
  const materialName = gltfMat.name ?? '(unnamed)';
  const baseColorMap = resolveMaterialTextureRef(
    pbr.baseColorTexture,
    handleMap,
    gltf,
    materialTextureSourcePath(materialIndex, 'pbrMetallicRoughness.baseColorTexture'),
    textureSourceExtensions,
    warnings,
    materialName,
    materialIndex,
    onDiagnostic,
  );

  // ── Metallic / roughness ───────────────────────────────────────────────────
  const metallic = pbr.metallicFactor ?? 1.0;
  const roughness = pbr.roughnessFactor ?? 1.0;
  // glTF combined metallic-roughness texture: G=roughness, B=metallic.
  const roughnessMap = resolveMaterialTextureRef(
    pbr.metallicRoughnessTexture,
    handleMap,
    gltf,
    materialTextureSourcePath(materialIndex, 'pbrMetallicRoughness.metallicRoughnessTexture'),
    textureSourceExtensions,
    warnings,
    materialName,
    materialIndex,
    onDiagnostic,
  );
  const metallicMap = roughnessMap;

  // ── Normal map ────────────────────────────────────────────────────────────
  const normalMap = resolveMaterialTextureRef(
    gltfMat.normalTexture,
    handleMap,
    gltf,
    materialTextureSourcePath(materialIndex, 'normalTexture'),
    textureSourceExtensions,
    warnings,
    materialName,
    materialIndex,
    onDiagnostic,
  );
  const normalScale = gltfMat.normalTexture?.scale ?? 1;

  // ── Occlusion ─────────────────────────────────────────────────────────────
  const aoMap = resolveMaterialTextureRef(
    gltfMat.occlusionTexture,
    handleMap,
    gltf,
    materialTextureSourcePath(materialIndex, 'occlusionTexture'),
    textureSourceExtensions,
    warnings,
    materialName,
    materialIndex,
    onDiagnostic,
  );
  const aoMapIntensity = gltfMat.occlusionTexture?.strength ?? 1;

  // ── Emissive ──────────────────────────────────────────────────────────────
  const emissiveFactor = gltfMat.emissiveFactor ?? [0, 0, 0];
  const emissive: Vec3 = [emissiveFactor[0], emissiveFactor[1], emissiveFactor[2]];
  const emissiveMap = resolveMaterialTextureRef(
    gltfMat.emissiveTexture,
    handleMap,
    gltf,
    materialTextureSourcePath(materialIndex, 'emissiveTexture'),
    textureSourceExtensions,
    warnings,
    materialName,
    materialIndex,
    onDiagnostic,
  );
  const emissiveStrengthExt = ext['KHR_materials_emissive_strength'] as
    | { emissiveStrength?: number }
    | undefined;
  const emissiveIntensity = emissiveStrengthExt?.emissiveStrength ?? 1;

  // ── Alpha mode ────────────────────────────────────────────────────────────
  const rawAlphaMode: unknown = gltfMat.alphaMode;
  if (
    rawAlphaMode !== undefined &&
    rawAlphaMode !== 'OPAQUE' &&
    rawAlphaMode !== 'MASK' &&
    rawAlphaMode !== 'BLEND'
  ) {
    throwMaterialImportError(warnings, onDiagnostic, {
      severity: 'error',
      code: 'invalid-material-alpha-mode',
      path: materialSourcePath(materialIndex, 'alphaMode'),
      ...materialDiagnosticIndex(materialIndex),
      message:
        `[vitrum/gltf-adapter] Material "${materialName}" alphaMode must be ` +
        `"OPAQUE", "MASK", or "BLEND"; received ${JSON.stringify(rawAlphaMode)}.`,
    });
  }
  const alphaMode =
    rawAlphaMode === 'MASK'
      ? ('mask' as const)
      : rawAlphaMode === 'BLEND'
        ? ('blend' as const)
        : ('opaque' as const);
  const alphaCutoff = gltfMat.alphaCutoff;
  // Combine baseColor alpha factor into opacity for blend/mask materials.
  const opacity = baseColorAlpha < 1 && alphaMode !== 'opaque' ? baseColorAlpha : undefined;
  const shadingModel = ext.KHR_materials_unlit ? 'unlit' as const : undefined;

  // ── doubleSided ────────────────────────────────────────────────────────────
  const doubleSided = gltfMat.doubleSided ?? false;

  // ── Extension policy warnings ─────────────────────────────────────────────
  for (const key of Object.keys(ext)) {
    if (!KNOWN_KHR_EXTENSIONS.has(key)) {
      emitMaterialDiagnostic(warnings, onDiagnostic, {
        severity: 'warning',
        code: 'unknown-material-extension',
        path: materialSourcePath(materialIndex, `extensions.${key}`),
        ...materialDiagnosticIndex(materialIndex),
        extensionName: key,
        message:
          `[vitrum/gltf-adapter] Material "${gltfMat.name ?? '(unnamed)'}" uses unknown extension ` +
          `"${key}" which is not mapped to a core MaterialSpec field. It is stored in extensions.`,
      });
    }
  }

  // ── Per-extension partial specs (D13.5) ────────────────────────────────────
  const transmissionPartial = _parseTransmissionExt(
    ext,
    handleMap,
    gltf,
    warnings,
    materialName,
    materialIndex,
    textureSourceExtensions,
    onDiagnostic,
  );
  const volumePartial       = _parseVolumeExt(
    ext,
    handleMap,
    gltf,
    warnings,
    materialName,
    materialIndex,
    textureSourceExtensions,
    onDiagnostic,
  );
  const iorPartial          = _parseIorExt(ext);
  const specularPartial     = _parseSpecularExt(
    ext,
    handleMap,
    gltf,
    warnings,
    materialName,
    materialIndex,
    textureSourceExtensions,
    onDiagnostic,
  );
  const sheenPartial        = _parseSheenExt(
    ext,
    handleMap,
    gltf,
    warnings,
    materialName,
    materialIndex,
    textureSourceExtensions,
    onDiagnostic,
  );
  const clearcoatPartial    = _parseClearcoatExt(
    ext,
    handleMap,
    gltf,
    warnings,
    materialName,
    materialIndex,
    textureSourceExtensions,
    onDiagnostic,
  );
  const iridescencePartial  = _parseIridescenceExt(
    ext,
    handleMap,
    gltf,
    warnings,
    materialName,
    materialIndex,
    textureSourceExtensions,
    onDiagnostic,
  );
  const anisotropyPartial   = _parseAnisotropyExt(
    ext,
    handleMap,
    gltf,
    warnings,
    materialName,
    materialIndex,
    textureSourceExtensions,
    onDiagnostic,
  );
  const dispersionPartial   = _parseDispersionExt(
    ext,
    warnings,
    materialName,
    materialIndex,
    onDiagnostic,
  );
  const specGlossPartial    = _parseSpecularGlossinessExt(
    ext,
    handleMap,
    gltf,
    warnings,
    materialName,
    alphaMode,
    materialIndex,
    textureSourceExtensions,
    onDiagnostic,
  );

  // ── Assemble MaterialSpec ─────────────────────────────────────────────────
  const mat: MaterialSpec = {
    baseColor,
    roughness,
    metallic,
    ...(shadingModel ? { shadingModel } : {}),
    ...(emissive[0] !== 0 || emissive[1] !== 0 || emissive[2] !== 0 ? { emissive } : {}),
    ...(emissiveIntensity !== 1 ? { emissiveIntensity } : {}),
    ...(emissiveMap ? { emissiveMap } : {}),
    alphaMode,
    ...(alphaCutoff !== undefined ? { alphaCutoff } : {}),
    ...(opacity !== undefined ? { opacity } : {}),
    doubleSided,
    ...(baseColorMap ? { baseColorMap } : {}),
    ...(normalMap ? { normalMap } : {}),
    ...(normalScale !== 1 ? { normalScale } : {}),
    ...(metallicMap ? { metallicMap } : {}),
    ...(roughnessMap ? { roughnessMap } : {}),
    ...(aoMap ? { aoMap } : {}),
    ...(aoMapIntensity !== 1 ? { aoMapIntensity } : {}),
    ...transmissionPartial,
    ...iorPartial,
    ...volumePartial,
    ...specularPartial,
    ...sheenPartial,
    ...clearcoatPartial,
    ...iridescencePartial,
    ...anisotropyPartial,
    ...dispersionPartial,
    ...specGlossPartial,
    extensions: {
      // Preserve unknown/unsupported extensions as raw data under their original key.
      ...Object.fromEntries(
        Object.entries(ext).filter(([key]) =>
          !KNOWN_KHR_EXTENSIONS.has(key) ||
          PRESERVE_RAW_EXTENSION_KEYS.has(key),
        ),
      ),
    },
  };

  return mat;
}

function resolveMaterialTextureRef(
  info: MaterialTextureInfo | undefined,
  handleMap: Map<number, unknown>,
  gltf: GltfJson | undefined,
  sourcePath: string | undefined,
  textureSourceExtensions: readonly GltfTextureSourceExtension[],
  warnings: string[],
  materialName: string,
  materialIndex: number | undefined,
  onDiagnostic: GltfMaterialDiagnosticSink | undefined,
): TextureRef | undefined {
  if (!info) return undefined;
  const path = sourcePath ?? materialSourcePath(materialIndex, `texture:${String(info.index)}`);
  if (!Number.isSafeInteger(info.index) || info.index < 0) {
    throwMaterialImportError(warnings, onDiagnostic, {
      severity: 'error',
      code: 'material-texture-not-found',
      path: `${path}.index`,
      ...materialDiagnosticIndex(materialIndex),
      textureIndex: info.index,
      message:
        `[vitrum/gltf-adapter] Material "${materialName}" texture index must be a non-negative safe integer; ` +
        `received ${String(info.index)} at ${path}.`,
    });
  }
  if (info.texCoord !== undefined && (!Number.isSafeInteger(info.texCoord) || info.texCoord < 0)) {
    throwMaterialImportError(warnings, onDiagnostic, {
      severity: 'error',
      code: 'material-texture-unresolved',
      path: `${path}.texCoord`,
      ...materialDiagnosticIndex(materialIndex),
      textureIndex: info.index,
      message:
        `[vitrum/gltf-adapter] Material "${materialName}" texture texCoord must be a non-negative safe integer.`,
    });
  }
  validateTextureTransform(info, path, warnings, materialName, materialIndex, onDiagnostic);
  const ref = resolveTextureRef(info, handleMap, gltf, sourcePath, textureSourceExtensions);
  if (ref) {
    emitMaterialTextureSamplerDiagnostics(
      info,
      gltf,
      sourcePath,
      warnings,
      materialName,
      materialIndex,
      onDiagnostic,
    );
    return ref;
  }

  const texture = gltf?.textures?.[info.index];
  if (gltf?.textures && texture === undefined) {
    throwMaterialImportError(warnings, onDiagnostic, {
      severity: 'error',
      code: 'material-texture-not-found',
      path: `${path}.index`,
      ...materialDiagnosticIndex(materialIndex),
      textureIndex: info.index,
      message:
        `[vitrum/gltf-adapter] Material "${materialName}" references missing texture index ` +
        `${info.index} at ${path}.`,
    });
  }

  throwMaterialImportError(warnings, onDiagnostic, {
    severity: 'error',
    code: 'material-texture-unresolved',
    path,
    ...materialDiagnosticIndex(materialIndex),
    textureIndex: info.index,
    message:
      `[vitrum/gltf-adapter] Material "${materialName}" references texture index ${info.index} ` +
      'but no decoded or raw image handle is available.',
  });
}

function validateTextureTransform(
  info: MaterialTextureInfo,
  path: string,
  warnings: string[],
  materialName: string,
  materialIndex: number | undefined,
  onDiagnostic: GltfMaterialDiagnosticSink | undefined,
): void {
  const raw = info.extensions?.KHR_texture_transform;
  if (raw === undefined) return;
  if (raw === null || typeof raw !== 'object' || Array.isArray(raw)) {
    throwMaterialImportError(warnings, onDiagnostic, {
      severity: 'error', code: 'material-texture-unresolved', path: `${path}.extensions.KHR_texture_transform`,
      ...materialDiagnosticIndex(materialIndex), textureIndex: info.index,
      message: `[vitrum/gltf-adapter] Material "${materialName}" KHR_texture_transform must be an object.`,
    });
  }
  const transform = raw as Record<string, unknown>;
  const vec2IsFinite = (value: unknown): value is readonly [number, number] =>
    Array.isArray(value) && value.length === 2 && value.every(Number.isFinite);
  if (
    (transform.offset !== undefined && !vec2IsFinite(transform.offset)) ||
    (transform.scale !== undefined && !vec2IsFinite(transform.scale)) ||
    (transform.rotation !== undefined && !Number.isFinite(transform.rotation)) ||
    (transform.texCoord !== undefined &&
      (!Number.isSafeInteger(transform.texCoord) || (transform.texCoord as number) < 0))
  ) {
    throwMaterialImportError(warnings, onDiagnostic, {
      severity: 'error', code: 'material-texture-unresolved', path: `${path}.extensions.KHR_texture_transform`,
      ...materialDiagnosticIndex(materialIndex), textureIndex: info.index,
      message:
        `[vitrum/gltf-adapter] Material "${materialName}" KHR_texture_transform requires finite VEC2 offset/scale, ` +
        'a finite rotation, and a non-negative safe-integer texCoord.',
    });
  }
}

function emitMaterialTextureSamplerDiagnostics(
  info: MaterialTextureInfo,
  gltf: GltfJson | undefined,
  sourcePath: string | undefined,
  warnings: string[],
  materialName: string,
  materialIndex: number | undefined,
  onDiagnostic: GltfMaterialDiagnosticSink | undefined,
): void {
  const texture = gltf?.textures?.[info.index];
  const samplerIndex = texture?.sampler;
  if (samplerIndex === undefined) return;
  const sampler = gltf?.samplers?.[samplerIndex];
  const materialPath = sourcePath ?? materialSourcePath(materialIndex, `texture:${info.index}`);
  if (sampler == null) {
    throwMaterialImportError(warnings, onDiagnostic, {
      severity: 'error',
      code: 'material-texture-sampler-not-found',
      path: `textures[${info.index}].sampler`,
      ...materialDiagnosticIndex(materialIndex),
      textureIndex: info.index,
      samplerIndex,
      message:
        `[vitrum/gltf-adapter] Material "${materialName}" references texture ${info.index} at ${materialPath}, ` +
        `whose sampler index ${samplerIndex} does not exist.`,
    });
  }

  emitInvalidSamplerDiagnosticIfNeeded(
    sampler.wrapS,
    'wrapS',
    isGltfWrapMode,
    info,
    samplerIndex,
    materialPath,
    warnings,
    materialName,
    materialIndex,
    onDiagnostic,
  );
  emitInvalidSamplerDiagnosticIfNeeded(
    sampler.wrapT,
    'wrapT',
    isGltfWrapMode,
    info,
    samplerIndex,
    materialPath,
    warnings,
    materialName,
    materialIndex,
    onDiagnostic,
  );
  emitInvalidSamplerDiagnosticIfNeeded(
    sampler.magFilter,
    'magFilter',
    (value) => value === 9728 || value === 9729,
    info,
    samplerIndex,
    materialPath,
    warnings,
    materialName,
    materialIndex,
    onDiagnostic,
  );
  emitInvalidSamplerDiagnosticIfNeeded(
    sampler.minFilter,
    'minFilter',
    (value) => value === 9728 || value === 9729 || (value >= 9984 && value <= 9987),
    info,
    samplerIndex,
    materialPath,
    warnings,
    materialName,
    materialIndex,
    onDiagnostic,
  );
}

function emitInvalidSamplerDiagnosticIfNeeded(
  value: number | undefined,
  property: 'wrapS' | 'wrapT' | 'magFilter' | 'minFilter',
  isValid: (value: number) => boolean,
  info: MaterialTextureInfo,
  samplerIndex: number,
  materialPath: string,
  warnings: string[],
  materialName: string,
  materialIndex: number | undefined,
  onDiagnostic: GltfMaterialDiagnosticSink | undefined,
): void {
  if (value === undefined || isValid(value)) return;
  const path = `samplers[${samplerIndex}].${property}`;
  throwMaterialImportError(warnings, onDiagnostic, {
    severity: 'error',
    code: 'invalid-material-texture-sampler',
    path,
    ...materialDiagnosticIndex(materialIndex),
    textureIndex: info.index,
    samplerIndex,
    samplerProperty: property,
    samplerValue: value,
    message:
      `[vitrum/gltf-adapter] Material "${materialName}" references texture ${info.index} at ${materialPath}, ` +
      `but ${path} has invalid value ${value}.`,
  });
}

function isGltfWrapMode(value: number): boolean {
  return value === 33071 || value === 33648 || value === 10497;
}

function materialTextureSourcePath(materialIndex: number | undefined, suffix: string): string | undefined {
  return materialIndex === undefined ? undefined : `materials[${materialIndex}].${suffix}`;
}

function materialSourcePath(materialIndex: number | undefined, suffix: string): string {
  return materialIndex === undefined ? `materials[?].${suffix}` : `materials[${materialIndex}].${suffix}`;
}

function materialDiagnosticIndex(
  materialIndex: number | undefined,
): Partial<Pick<GltfMaterialDiagnostic, 'materialIndex'>> {
  return materialIndex === undefined ? {} : { materialIndex };
}

function emitMaterialDiagnostic(
  warnings: string[],
  onDiagnostic: GltfMaterialDiagnosticSink | undefined,
  diagnostic: GltfMaterialDiagnostic,
): void {
  if (onDiagnostic) {
    try {
      onDiagnostic(diagnostic);
      return;
    } catch {
      // Host diagnostic callbacks must not abort material conversion.
    }
  }
  warnings.push(diagnostic.message);
}

function throwMaterialImportError(
  warnings: string[],
  onDiagnostic: GltfMaterialDiagnosticSink | undefined,
  diagnostic: GltfMaterialDiagnostic,
): never {
  if (onDiagnostic) {
    try {
      onDiagnostic(diagnostic);
    } catch {
      // A diagnostic observer cannot suppress a strict import failure.
    }
  } else {
    warnings.push(diagnostic.message);
  }
  throw new GltfMaterialImportError(diagnostic);
}

/**
 * Default material when a primitive has no material reference.
 * Per glTF 2.0 spec §5.21: metallic=1, roughness=1, baseColor=[1,1,1,1].
 */
export const GLTF_DEFAULT_MATERIAL: MaterialSpec = {
  baseColor: [1, 1, 1],
  roughness: 1,
  metallic: 1,
};

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 1;
  return Math.min(1, Math.max(0, value));
}
