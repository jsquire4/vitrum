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
// not blending.  It has no MaterialSpec field, so we attach it to extensions.
//
// KHR_materials_emissive_strength: scales emissiveIntensity (default 1).
//
// D13.5: per-extension parsers extracted to private helpers (_parse*Ext).
// Each returns a Partial<MaterialSpec> merged into the final spec at the end.
// This keeps the 215-line convertMaterial down to a flat base-PBR + merge.

import type { GltfMaterial } from './gltfTypes.js';
import type { MaterialSpec, Vec3 } from '@vitrum/core';
import { resolveTextureRef } from './textures.js';

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

const KNOWN_UNSUPPORTED_EXTENSION_MESSAGES: Readonly<Partial<Record<string, string>>> = {};

const PRESERVE_RAW_EXTENSION_KEYS = new Set([
  'KHR_materials_pbrSpecularGlossiness',
]);

// ── Per-extension parsers (D13.5) ──────────────────────────────────────────────

function _parseTransmissionExt(
  ext: Record<string, unknown>,
  handleMap: Map<number, unknown>,
): Partial<MaterialSpec> {
  const txExt = ext['KHR_materials_transmission'] as
    | { transmissionFactor?: number; transmissionTexture?: { index: number; texCoord?: number } }
    | undefined;
  if (!txExt) return {};
  const transmission = txExt.transmissionFactor ?? 0;
  const transmissionMap = resolveTextureRef(txExt.transmissionTexture, handleMap);
  return {
    ...(transmission > 0 ? { transmission } : {}),
    ...(transmissionMap ? { transmissionMap } : {}),
  };
}

function _parseVolumeExt(
  ext: Record<string, unknown>,
  handleMap: Map<number, unknown>,
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
  const thicknessMap = resolveTextureRef(volExt.thicknessTexture, handleMap);
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
  const specularIntensityMap = resolveTextureRef(specExt.specularTexture, handleMap);
  const specularColorMap = resolveTextureRef(specExt.specularColorTexture, handleMap);
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
  const sheenColorMap = resolveTextureRef(sheenExt.sheenColorTexture, handleMap);
  const sheenRoughnessMap = resolveTextureRef(sheenExt.sheenRoughnessTexture, handleMap);
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
  if (clearcoat <= 0) return {};
  const clearcoatRoughness = ccExt.clearcoatRoughnessFactor ?? 0;
  const clearcoatMap = resolveTextureRef(ccExt.clearcoatTexture, handleMap);
  const clearcoatRoughnessMap = resolveTextureRef(ccExt.clearcoatRoughnessTexture, handleMap);
  const clearcoatNormalMap = resolveTextureRef(ccExt.clearcoatNormalTexture, handleMap);
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
  if (iridescence <= 0) return {};
  const iridescenceIor = iridExt.iridescenceIor ?? 1.3;
  const iridescenceThicknessRange: readonly [number, number] = [
    iridExt.iridescenceThicknessMinimum ?? 100,
    iridExt.iridescenceThicknessMaximum ?? 400,
  ];
  const iridescenceMap = resolveTextureRef(iridExt.iridescenceTexture, handleMap);
  const iridescenceThicknessMap = resolveTextureRef(iridExt.iridescenceThicknessTexture, handleMap);
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
  const anisotropyMap = resolveTextureRef(anisoExt.anisotropyTexture, handleMap);
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
): Partial<MaterialSpec> {
  const dispersionExt = ext['KHR_materials_dispersion'] as { dispersion?: number } | undefined;
  if (!dispersionExt || dispersionExt.dispersion === undefined) return {};
  const dispersion = dispersionExt.dispersion;
  if (!Number.isFinite(dispersion) || dispersion < 0) {
    warnings.push(
      `[vitrum/gltf-adapter] Material "${materialName}" uses KHR_materials_dispersion ` +
        `with invalid dispersion=${String(dispersion)}. Dispersion is ignored.`,
    );
    return {};
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
  warnings: string[],
  materialName: string,
  alphaMode: MaterialSpec['alphaMode'],
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

  warnings.push(
    `[vitrum/gltf-adapter] Material "${materialName}" uses archived ` +
      'KHR_materials_pbrSpecularGlossiness. Scalar diffuse/specular/glossiness ' +
      'values are converted approximately to metallic-roughness + specular fields.',
  );

  const diffuseFactor = sgExt.diffuseFactor ?? [1, 1, 1, 1];
  const specularFactor = sgExt.specularFactor ?? [1, 1, 1];
  const glossinessFactor = sgExt.glossinessFactor ?? 1;
  const diffuseTexture = resolveTextureRef(sgExt.diffuseTexture, handleMap);
  const specularGlossinessTexture = resolveTextureRef(sgExt.specularGlossinessTexture, handleMap);

  if (specularGlossinessTexture) {
    warnings.push(
      `[vitrum/gltf-adapter] Material "${materialName}" uses ` +
        'KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture. The RGB ' +
        'specular map is imported as specularColorMap, but glossiness-in-alpha ' +
        'cannot be inverted into a roughnessMap without a texture bake, so the ' +
        'scalar glossinessFactor is used for roughness.',
    );
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
): MaterialSpec {
  const pbr = gltfMat.pbrMetallicRoughness ?? {};
  const ext = gltfMat.extensions ?? {};

  // ── Base color ─────────────────────────────────────────────────────────────
  const baseColorFactor = pbr.baseColorFactor ?? [1, 1, 1, 1];
  const baseColor: Vec3 = [baseColorFactor[0], baseColorFactor[1], baseColorFactor[2]];
  const baseColorAlpha = baseColorFactor[3] ?? 1;
  const baseColorMap = resolveTextureRef(pbr.baseColorTexture, handleMap);

  // ── Metallic / roughness ───────────────────────────────────────────────────
  const metallic = pbr.metallicFactor ?? 1.0;
  const roughness = pbr.roughnessFactor ?? 1.0;
  // glTF combined metallic-roughness texture: G=roughness, B=metallic.
  const roughnessMap = resolveTextureRef(pbr.metallicRoughnessTexture, handleMap);
  const metallicMap = roughnessMap;

  // ── Normal map ────────────────────────────────────────────────────────────
  const normalMap = resolveTextureRef(gltfMat.normalTexture, handleMap);
  const normalScale = gltfMat.normalTexture?.scale ?? 1;

  // ── Occlusion ─────────────────────────────────────────────────────────────
  const aoMap = resolveTextureRef(gltfMat.occlusionTexture, handleMap);
  const aoMapIntensity = gltfMat.occlusionTexture?.strength ?? 1;

  // ── Emissive ──────────────────────────────────────────────────────────────
  const emissiveFactor = gltfMat.emissiveFactor ?? [0, 0, 0];
  const emissive: Vec3 = [emissiveFactor[0], emissiveFactor[1], emissiveFactor[2]];
  const emissiveMap = resolveTextureRef(gltfMat.emissiveTexture, handleMap);
  const emissiveStrengthExt = ext['KHR_materials_emissive_strength'] as
    | { emissiveStrength?: number }
    | undefined;
  const emissiveIntensity = emissiveStrengthExt?.emissiveStrength ?? 1;

  // ── Alpha mode ────────────────────────────────────────────────────────────
  const alphaMode =
    gltfMat.alphaMode === 'MASK'
      ? ('mask' as const)
      : gltfMat.alphaMode === 'BLEND'
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
    const unsupportedMessage = KNOWN_UNSUPPORTED_EXTENSION_MESSAGES[key];
    if (unsupportedMessage) {
      warnings.push(
        `[vitrum/gltf-adapter] Material "${gltfMat.name ?? '(unnamed)'}" uses ${key}. ` +
          unsupportedMessage,
      );
    } else if (!KNOWN_KHR_EXTENSIONS.has(key)) {
      warnings.push(
        `[vitrum/gltf-adapter] Material "${gltfMat.name ?? '(unnamed)'}" uses unknown extension ` +
          `"${key}" which is not mapped to a core MaterialSpec field. It is stored in extensions.`,
      );
    }
  }

  // ── Per-extension partial specs (D13.5) ────────────────────────────────────
  const transmissionPartial = _parseTransmissionExt(ext, handleMap);
  const volumePartial       = _parseVolumeExt(ext, handleMap);
  const iorPartial          = _parseIorExt(ext);
  const specularPartial     = _parseSpecularExt(ext, handleMap);
  const sheenPartial        = _parseSheenExt(ext, handleMap);
  const clearcoatPartial    = _parseClearcoatExt(ext, handleMap);
  const iridescencePartial  = _parseIridescenceExt(ext, handleMap);
  const anisotropyPartial   = _parseAnisotropyExt(ext, handleMap);
  const dispersionPartial   = _parseDispersionExt(ext, warnings, gltfMat.name ?? '(unnamed)');
  const specGlossPartial    = _parseSpecularGlossinessExt(
    ext,
    handleMap,
    warnings,
    gltfMat.name ?? '(unnamed)',
    alphaMode,
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
      ...(doubleSided ? { doubleSided: true } : {}),
      // Preserve unknown/unsupported extensions as raw data under their original key.
      ...Object.fromEntries(
        Object.entries(ext).filter(([key]) =>
          !KNOWN_KHR_EXTENSIONS.has(key) ||
          KNOWN_UNSUPPORTED_EXTENSION_MESSAGES[key] ||
          PRESERVE_RAW_EXTENSION_KEYS.has(key),
        ),
      ),
    },
  };

  return mat;
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
