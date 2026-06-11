// materials.ts — glTF pbrMetallicRoughness + KHR extensions → core MaterialSpec.
//
// ORM texture note:
//   glTF stores roughness+metallic in a SINGLE texture:
//     G channel = roughnessTexture, B channel = metallicTexture
//   Per pt-webgpu materialTextures.ts convention, the combined texture is
//   supplied as `roughnessMap` (which the backend treats as the ORM slot).
//   `metallicMap` is NOT set when the same handle is used for both channels —
//   it would trigger the "distinct handles" warning in collectMaterialTextures.
//   When a material supplies ONLY metallicRoughnessTexture, it maps to roughnessMap.
//
// doubleSided → alphaMode 'blend' is incorrect; doubleSided controls backface culling,
// not blending.  It has no MaterialSpec field, so we attach it to extensions.
//
// KHR_materials_emissive_strength: scales emissiveIntensity (default 1).

import type { GltfMaterial } from './gltfTypes.js';
import type { MaterialSpec, Vec3 } from '@vitrum/core';
import { resolveTextureRef } from './textures.js';

const KNOWN_KHR_EXTENSIONS = new Set([
  'KHR_materials_transmission',
  'KHR_materials_ior',
  'KHR_materials_volume',
  'KHR_materials_specular',
  'KHR_materials_sheen',
  'KHR_materials_clearcoat',
  'KHR_materials_iridescence',
  'KHR_materials_anisotropy',
  'KHR_materials_emissive_strength',
  'KHR_texture_transform', // per-texture, handled at TextureRef level
]);

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
  // glTF combined ORM texture → roughnessMap (B=metallic, G=roughness per spec).
  const roughnessMap = resolveTextureRef(pbr.metallicRoughnessTexture, handleMap);
  // metallicMap is intentionally omitted when it shares the same handle (see module note).

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

  // ── KHR_materials_transmission ────────────────────────────────────────────
  const txExt = ext['KHR_materials_transmission'] as
    | { transmissionFactor?: number; transmissionTexture?: { index: number; texCoord?: number } }
    | undefined;
  const transmission = txExt?.transmissionFactor ?? 0;
  const transmissionMap = resolveTextureRef(txExt?.transmissionTexture, handleMap);

  // ── KHR_materials_ior ─────────────────────────────────────────────────────
  const iorExt = ext['KHR_materials_ior'] as { ior?: number } | undefined;
  const ior = iorExt?.ior ?? 1.5;

  // ── KHR_materials_volume ──────────────────────────────────────────────────
  const volExt = ext['KHR_materials_volume'] as
    | {
        thicknessFactor?: number;
        attenuationDistance?: number;
        attenuationColor?: [number, number, number];
      }
    | undefined;
  const thickness = volExt?.thicknessFactor ?? 0;
  const attenuationDistance = volExt?.attenuationDistance ?? Infinity;
  const attenuationColor: Vec3 | undefined = volExt?.attenuationColor;

  // ── KHR_materials_specular ────────────────────────────────────────────────
  const specExt = ext['KHR_materials_specular'] as
    | {
        specularFactor?: number;
        specularTexture?: { index: number; texCoord?: number };
        specularColorFactor?: [number, number, number];
        specularColorTexture?: { index: number; texCoord?: number };
      }
    | undefined;
  const specularIntensity = specExt?.specularFactor ?? 1;
  const specularColor: Vec3 | undefined = specExt?.specularColorFactor;
  const specularIntensityMap = resolveTextureRef(specExt?.specularTexture, handleMap);
  const specularColorMap = resolveTextureRef(specExt?.specularColorTexture, handleMap);

  // ── KHR_materials_sheen ───────────────────────────────────────────────────
  const sheenExt = ext['KHR_materials_sheen'] as
    | {
        sheenColorFactor?: [number, number, number];
        sheenColorTexture?: { index: number; texCoord?: number };
        sheenRoughnessFactor?: number;
        sheenRoughnessTexture?: { index: number; texCoord?: number };
      }
    | undefined;
  const _sheen = sheenExt ? 1 : 0; // unused — kept for documentation of the computed value
  const sheenColor: Vec3 | undefined = sheenExt?.sheenColorFactor;
  const sheenRoughness = sheenExt?.sheenRoughnessFactor ?? 0;
  const sheenColorMap = resolveTextureRef(sheenExt?.sheenColorTexture, handleMap);
  const sheenRoughnessMap = resolveTextureRef(sheenExt?.sheenRoughnessTexture, handleMap);

  // ── KHR_materials_clearcoat ───────────────────────────────────────────────
  const ccExt = ext['KHR_materials_clearcoat'] as
    | {
        clearcoatFactor?: number;
        clearcoatTexture?: { index: number; texCoord?: number };
        clearcoatRoughnessFactor?: number;
        clearcoatRoughnessTexture?: { index: number; texCoord?: number };
        clearcoatNormalTexture?: { index: number; texCoord?: number; scale?: number };
      }
    | undefined;
  const clearcoat = ccExt?.clearcoatFactor ?? 0;
  const clearcoatRoughness = ccExt?.clearcoatRoughnessFactor ?? 0;
  const clearcoatMap = resolveTextureRef(ccExt?.clearcoatTexture, handleMap);
  const clearcoatRoughnessMap = resolveTextureRef(ccExt?.clearcoatRoughnessTexture, handleMap);
  const clearcoatNormalMap = resolveTextureRef(ccExt?.clearcoatNormalTexture, handleMap);
  const clearcoatNormalScale = (ccExt?.clearcoatNormalTexture as { scale?: number } | undefined)?.scale ?? 1;

  // ── KHR_materials_iridescence ─────────────────────────────────────────────
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
  const iridescence = iridExt?.iridescenceFactor ?? 0;
  const iridescenceIor = iridExt?.iridescenceIor ?? 1.3;
  const iridescenceThicknessRange =
    iridExt
      ? ([iridExt.iridescenceThicknessMinimum ?? 100, iridExt.iridescenceThicknessMaximum ?? 400] as const)
      : undefined;
  const iridescenceMap = resolveTextureRef(iridExt?.iridescenceTexture, handleMap);
  const iridescenceThicknessMap = resolveTextureRef(
    iridExt?.iridescenceThicknessTexture,
    handleMap,
  );

  // ── KHR_materials_anisotropy ──────────────────────────────────────────────
  const anisoExt = ext['KHR_materials_anisotropy'] as
    | {
        anisotropyStrength?: number;
        anisotropyRotation?: number;
        anisotropyTexture?: { index: number; texCoord?: number };
      }
    | undefined;
  const anisotropy = anisoExt?.anisotropyStrength ?? 0;
  const anisotropyRotation = anisoExt?.anisotropyRotation ?? 0;
  const anisotropyMap = resolveTextureRef(anisoExt?.anisotropyTexture, handleMap);

  // ── doubleSided (no MaterialSpec field — kept in extensions) ──────────────
  const doubleSided = gltfMat.doubleSided ?? false;

  // ── Unknown extension warnings ────────────────────────────────────────────
  for (const key of Object.keys(ext)) {
    if (!KNOWN_KHR_EXTENSIONS.has(key)) {
      warnings.push(
        `[vitrum/gltf-adapter] Material "${gltfMat.name ?? '(unnamed)'}" uses unknown extension ` +
          `"${key}" which is not mapped to a core MaterialSpec field. It is stored in extensions.`,
      );
    }
  }

  // ── Assemble MaterialSpec ─────────────────────────────────────────────────
  const mat: MaterialSpec = {
    baseColor,
    roughness,
    metallic,
    ...(emissive[0] !== 0 || emissive[1] !== 0 || emissive[2] !== 0 ? { emissive } : {}),
    ...(emissiveIntensity !== 1 ? { emissiveIntensity } : {}),
    ...(emissiveMap ? { emissiveMap } : {}),
    alphaMode,
    ...(alphaCutoff !== undefined ? { alphaCutoff } : {}),
    ...(opacity !== undefined ? { opacity } : {}),
    ...(baseColorMap ? { baseColorMap } : {}),
    ...(normalMap ? { normalMap } : {}),
    ...(normalScale !== 1 ? { normalScale } : {}),
    ...(roughnessMap ? { roughnessMap } : {}),
    ...(aoMap ? { aoMap } : {}),
    ...(aoMapIntensity !== 1 ? { aoMapIntensity } : {}),
    ...(transmission > 0 ? { transmission } : {}),
    ...(transmissionMap ? { transmissionMap } : {}),
    ...(iorExt ? { ior } : {}),
    ...(volExt ? { thickness, attenuationDistance } : {}),
    ...(attenuationColor ? { attenuationColor } : {}),
    ...(specExt && specularIntensity !== 1 ? { specularIntensity } : {}),
    ...(specularColor ? { specularColor } : {}),
    ...(specularIntensityMap ? { specularIntensityMap } : {}),
    ...(specularColorMap ? { specularColorMap } : {}),
    ...(sheenExt ? { sheen: 1 as const, sheenRoughness } : {}),
    ...(sheenColor ? { sheenColor } : {}),
    ...(sheenColorMap ? { sheenColorMap } : {}),
    ...(sheenRoughnessMap ? { sheenRoughnessMap } : {}),
    ...(clearcoat > 0 ? { clearcoat, clearcoatRoughness } : {}),
    ...(clearcoatMap ? { clearcoatMap } : {}),
    ...(clearcoatRoughnessMap ? { clearcoatRoughnessMap } : {}),
    ...(clearcoatNormalMap ? { clearcoatNormalMap } : {}),
    ...(clearcoatNormalScale !== 1 ? { clearcoatNormalScale } : {}),
    ...(iridescence > 0 ? { iridescence, iridescenceIor } : {}),
    ...(iridescenceThicknessRange ? { iridescenceThicknessRange } : {}),
    ...(iridescenceMap ? { iridescenceMap } : {}),
    ...(iridescenceThicknessMap ? { iridescenceThicknessMap } : {}),
    ...(anisoExt ? { anisotropy, anisotropyRotation } : {}),
    ...(anisotropyMap ? { anisotropyMap } : {}),
    extensions: {
      ...(doubleSided ? { doubleSided: true } : {}),
      // Preserve unknown extensions as raw data under their original key.
      ...Object.fromEntries(
        Object.entries(ext).filter(([key]) => !KNOWN_KHR_EXTENSIONS.has(key)),
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
