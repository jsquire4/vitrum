import type { MaterialSpec, Vec3 } from '@vitrum/core';

export type GltfMaterialPointerField =
  | 'baseColorFactor'
  | 'metallicFactor'
  | 'roughnessFactor'
  | 'emissiveFactor'
  | 'alphaCutoff'
  | 'emissiveStrength'
  | 'transmissionFactor'
  | 'thicknessFactor'
  | 'attenuationDistance'
  | 'ior'
  | 'specularFactor'
  | 'specularColorFactor'
  | 'clearcoatFactor'
  | 'clearcoatRoughnessFactor'
  | 'sheenColorFactor'
  | 'sheenRoughnessFactor'
  | 'iridescenceFactor'
  | 'iridescenceIor'
  | 'iridescenceThicknessMinimum'
  | 'iridescenceThicknessMaximum'
  | 'anisotropyStrength'
  | 'anisotropyRotation';

export interface GltfMaterialPointerTarget {
  readonly pointer: string;
  readonly materialIndex: number;
  readonly field: GltfMaterialPointerField;
  readonly components: 1 | 3 | 4;
}

type PointerSpec = Omit<GltfMaterialPointerTarget, 'pointer' | 'materialIndex'>;

const MATERIAL_POINTER_SPECS: Readonly<Record<string, PointerSpec>> = Object.freeze({
  'pbrMetallicRoughness/baseColorFactor': { field: 'baseColorFactor', components: 4 },
  'pbrMetallicRoughness/metallicFactor': { field: 'metallicFactor', components: 1 },
  'pbrMetallicRoughness/roughnessFactor': { field: 'roughnessFactor', components: 1 },
  emissiveFactor: { field: 'emissiveFactor', components: 3 },
  alphaCutoff: { field: 'alphaCutoff', components: 1 },
  'extensions/KHR_materials_emissive_strength/emissiveStrength': { field: 'emissiveStrength', components: 1 },
  'extensions/KHR_materials_transmission/transmissionFactor': { field: 'transmissionFactor', components: 1 },
  'extensions/KHR_materials_volume/thicknessFactor': { field: 'thicknessFactor', components: 1 },
  'extensions/KHR_materials_volume/attenuationDistance': { field: 'attenuationDistance', components: 1 },
  'extensions/KHR_materials_ior/ior': { field: 'ior', components: 1 },
  'extensions/KHR_materials_specular/specularFactor': { field: 'specularFactor', components: 1 },
  'extensions/KHR_materials_specular/specularColorFactor': { field: 'specularColorFactor', components: 3 },
  'extensions/KHR_materials_clearcoat/clearcoatFactor': { field: 'clearcoatFactor', components: 1 },
  'extensions/KHR_materials_clearcoat/clearcoatRoughnessFactor': { field: 'clearcoatRoughnessFactor', components: 1 },
  'extensions/KHR_materials_sheen/sheenColorFactor': { field: 'sheenColorFactor', components: 3 },
  'extensions/KHR_materials_sheen/sheenRoughnessFactor': { field: 'sheenRoughnessFactor', components: 1 },
  'extensions/KHR_materials_iridescence/iridescenceFactor': { field: 'iridescenceFactor', components: 1 },
  'extensions/KHR_materials_iridescence/iridescenceIor': { field: 'iridescenceIor', components: 1 },
  'extensions/KHR_materials_iridescence/iridescenceThicknessMinimum': { field: 'iridescenceThicknessMinimum', components: 1 },
  'extensions/KHR_materials_iridescence/iridescenceThicknessMaximum': { field: 'iridescenceThicknessMaximum', components: 1 },
  'extensions/KHR_materials_anisotropy/anisotropyStrength': { field: 'anisotropyStrength', components: 1 },
  'extensions/KHR_materials_anisotropy/anisotropyRotation': { field: 'anisotropyRotation', components: 1 },
});

export function resolveGltfMaterialAnimationPointer(pointer: string | undefined): GltfMaterialPointerTarget | undefined {
  if (typeof pointer !== 'string') return undefined;
  const match = /^\/materials\/(\d+)\/(.+)$/.exec(pointer);
  if (!match) return undefined;
  const materialIndex = Number(match[1]);
  const path = decodePointerPath(match[2] ?? '');
  const spec = MATERIAL_POINTER_SPECS[path];
  if (!Number.isSafeInteger(materialIndex) || materialIndex < 0 || spec === undefined) return undefined;
  return {
    pointer,
    materialIndex,
    ...spec,
  };
}

export function supportedGltfMaterialAnimationPointers(): readonly string[] {
  return Object.keys(MATERIAL_POINTER_SPECS).map((path) => `/materials/{index}/${path}`);
}

export function applyGltfMaterialPointerValue(
  material: MaterialSpec,
  target: GltfMaterialPointerTarget,
  value: Float32Array,
): MaterialSpec {
  const out: Record<string, unknown> = { ...material };
  const scalar = finiteOr(value[0], 0);
  switch (target.field) {
    case 'baseColorFactor': {
      out.baseColor = [
        finiteOr(value[0], 1),
        finiteOr(value[1], 1),
        finiteOr(value[2], 1),
      ] satisfies Vec3;
      const alpha = finiteOr(value[3], 1);
      out.opacity = material.alphaMode !== 'opaque' && alpha < 1 ? clamp01(alpha) : undefined;
      break;
    }
    case 'metallicFactor':
      out.metallic = clamp01(scalar);
      break;
    case 'roughnessFactor':
      out.roughness = clamp01(scalar);
      break;
    case 'emissiveFactor':
      out.emissive = [
        Math.max(0, finiteOr(value[0], 0)),
        Math.max(0, finiteOr(value[1], 0)),
        Math.max(0, finiteOr(value[2], 0)),
      ] satisfies Vec3;
      break;
    case 'alphaCutoff':
      out.alphaCutoff = clamp01(scalar);
      break;
    case 'emissiveStrength':
      out.emissiveIntensity = Math.max(0, scalar);
      break;
    case 'transmissionFactor':
      out.transmission = clamp01(scalar);
      break;
    case 'thicknessFactor':
      out.thickness = Math.max(0, scalar);
      break;
    case 'attenuationDistance':
      out.attenuationDistance = scalar > 0 ? scalar : Infinity;
      break;
    case 'ior':
      out.ior = Math.max(1, scalar);
      break;
    case 'specularFactor':
      out.specularIntensity = clamp01(scalar);
      break;
    case 'specularColorFactor':
      out.specularColor = [
        clamp01(finiteOr(value[0], 1)),
        clamp01(finiteOr(value[1], 1)),
        clamp01(finiteOr(value[2], 1)),
      ] satisfies Vec3;
      break;
    case 'clearcoatFactor':
      out.clearcoat = clamp01(scalar);
      break;
    case 'clearcoatRoughnessFactor':
      out.clearcoatRoughness = clamp01(scalar);
      break;
    case 'sheenColorFactor':
      out.sheen = 1;
      out.sheenColor = [
        clamp01(finiteOr(value[0], 0)),
        clamp01(finiteOr(value[1], 0)),
        clamp01(finiteOr(value[2], 0)),
      ] satisfies Vec3;
      break;
    case 'sheenRoughnessFactor':
      out.sheen = 1;
      out.sheenRoughness = clamp01(scalar);
      break;
    case 'iridescenceFactor':
      out.iridescence = clamp01(scalar);
      break;
    case 'iridescenceIor':
      out.iridescenceIor = Math.max(1, scalar);
      break;
    case 'iridescenceThicknessMinimum': {
      const [, max = 400] = material.iridescenceThicknessRange ?? [100, 400];
      out.iridescenceThicknessRange = [Math.max(0, scalar), max] as const;
      break;
    }
    case 'iridescenceThicknessMaximum': {
      const [min = 100] = material.iridescenceThicknessRange ?? [100, 400];
      out.iridescenceThicknessRange = [min, Math.max(0, scalar)] as const;
      break;
    }
    case 'anisotropyStrength':
      out.anisotropy = clamp01(scalar);
      break;
    case 'anisotropyRotation':
      out.anisotropyRotation = finiteOr(value[0], 0);
      break;
  }
  return out as unknown as MaterialSpec;
}

function decodePointerPath(path: string): string {
  return path
    .split('/')
    .map((segment) => segment.replace(/~1/g, '/').replace(/~0/g, '~'))
    .join('/');
}

function finiteOr(value: number | undefined, fallback: number): number {
  return value !== undefined && Number.isFinite(value) ? value : fallback;
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.min(1, Math.max(0, value));
}
