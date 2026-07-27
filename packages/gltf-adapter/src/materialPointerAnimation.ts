import type { MaterialSpec, Vec3 } from '@vitrum/core';

export type GltfMaterialPointerField =
  | 'baseColorFactor'
  | 'metallicFactor'
  | 'roughnessFactor'
  | 'emissiveFactor'
  | 'alphaCutoff'
  | 'normalScale'
  | 'aoMapIntensity'
  | 'emissiveStrength'
  | 'transmissionFactor'
  | 'thicknessFactor'
  | 'attenuationColor'
  | 'attenuationDistance'
  | 'ior'
  | 'specularFactor'
  | 'specularColorFactor'
  | 'clearcoatFactor'
  | 'clearcoatRoughnessFactor'
  | 'clearcoatNormalScale'
  | 'sheenColorFactor'
  | 'sheenRoughnessFactor'
  | 'iridescenceFactor'
  | 'iridescenceIor'
  | 'iridescenceThicknessMinimum'
  | 'iridescenceThicknessMaximum'
  | 'anisotropyStrength'
  | 'anisotropyRotation'
  | 'dispersion';

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
  'normalTexture/scale': { field: 'normalScale', components: 1 },
  'occlusionTexture/strength': { field: 'aoMapIntensity', components: 1 },
  'extensions/KHR_materials_emissive_strength/emissiveStrength': { field: 'emissiveStrength', components: 1 },
  'extensions/KHR_materials_transmission/transmissionFactor': { field: 'transmissionFactor', components: 1 },
  'extensions/KHR_materials_volume/thicknessFactor': { field: 'thicknessFactor', components: 1 },
  'extensions/KHR_materials_volume/attenuationColor': { field: 'attenuationColor', components: 3 },
  'extensions/KHR_materials_volume/attenuationDistance': { field: 'attenuationDistance', components: 1 },
  'extensions/KHR_materials_ior/ior': { field: 'ior', components: 1 },
  'extensions/KHR_materials_specular/specularFactor': { field: 'specularFactor', components: 1 },
  'extensions/KHR_materials_specular/specularColorFactor': { field: 'specularColorFactor', components: 3 },
  'extensions/KHR_materials_clearcoat/clearcoatFactor': { field: 'clearcoatFactor', components: 1 },
  'extensions/KHR_materials_clearcoat/clearcoatRoughnessFactor': { field: 'clearcoatRoughnessFactor', components: 1 },
  'extensions/KHR_materials_clearcoat/clearcoatNormalTexture/scale': {
    field: 'clearcoatNormalScale',
    components: 1,
  },
  'extensions/KHR_materials_sheen/sheenColorFactor': { field: 'sheenColorFactor', components: 3 },
  'extensions/KHR_materials_sheen/sheenRoughnessFactor': { field: 'sheenRoughnessFactor', components: 1 },
  'extensions/KHR_materials_iridescence/iridescenceFactor': { field: 'iridescenceFactor', components: 1 },
  'extensions/KHR_materials_iridescence/iridescenceIor': { field: 'iridescenceIor', components: 1 },
  'extensions/KHR_materials_iridescence/iridescenceThicknessMinimum': { field: 'iridescenceThicknessMinimum', components: 1 },
  'extensions/KHR_materials_iridescence/iridescenceThicknessMaximum': { field: 'iridescenceThicknessMaximum', components: 1 },
  'extensions/KHR_materials_anisotropy/anisotropyStrength': { field: 'anisotropyStrength', components: 1 },
  'extensions/KHR_materials_anisotropy/anisotropyRotation': { field: 'anisotropyRotation', components: 1 },
  'extensions/KHR_materials_dispersion/dispersion': { field: 'dispersion', components: 1 },
});

export function resolveGltfMaterialAnimationPointer(pointer: string | undefined): GltfMaterialPointerTarget | undefined {
  if (typeof pointer !== 'string') return undefined;
  const match = /^\/materials\/(0|[1-9]\d*)\/(.+)$/.exec(pointer);
  if (!match) return undefined;
  const materialIndex = Number(match[1]);
  const path = decodePointerPath(match[2] ?? '');
  if (path === undefined) return undefined;
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
  // KHR_animation_pointer requires output values to be valid for the target
  // property. Validation happens during import and again after runtime
  // sampling; this application layer therefore preserves accepted values
  // exactly instead of hiding malformed data through clamping/fallbacks.
  const scalar = value[0]!;
  switch (target.field) {
    case 'baseColorFactor': {
      out.baseColor = [
        value[0]!,
        value[1]!,
        value[2]!,
      ] satisfies Vec3;
      const alpha = value[3]!;
      out.opacity = material.alphaMode !== 'opaque' && alpha < 1 ? alpha : undefined;
      break;
    }
    case 'metallicFactor':
      out.metallic = scalar;
      break;
    case 'roughnessFactor':
      out.roughness = scalar;
      break;
    case 'emissiveFactor':
      out.emissive = [
        value[0]!,
        value[1]!,
        value[2]!,
      ] satisfies Vec3;
      break;
    case 'alphaCutoff':
      out.alphaCutoff = scalar;
      break;
    case 'normalScale':
      out.normalScale = scalar;
      break;
    case 'aoMapIntensity':
      out.aoMapIntensity = scalar;
      break;
    case 'emissiveStrength':
      out.emissiveIntensity = scalar;
      break;
    case 'transmissionFactor':
      out.transmission = scalar;
      break;
    case 'thicknessFactor':
      out.thickness = scalar;
      break;
    case 'attenuationColor':
      out.attenuationColor = [
        value[0]!,
        value[1]!,
        value[2]!,
      ] satisfies Vec3;
      break;
    case 'attenuationDistance':
      out.attenuationDistance = scalar;
      break;
    case 'ior':
      out.ior = scalar;
      break;
    case 'specularFactor':
      out.specularIntensity = scalar;
      break;
    case 'specularColorFactor':
      out.specularColor = [
        value[0]!,
        value[1]!,
        value[2]!,
      ] satisfies Vec3;
      break;
    case 'clearcoatFactor':
      out.clearcoat = scalar;
      break;
    case 'clearcoatRoughnessFactor':
      out.clearcoatRoughness = scalar;
      break;
    case 'clearcoatNormalScale':
      out.clearcoatNormalScale = scalar;
      break;
    case 'sheenColorFactor':
      out.sheen = 1;
      out.sheenColor = [
        value[0]!,
        value[1]!,
        value[2]!,
      ] satisfies Vec3;
      break;
    case 'sheenRoughnessFactor':
      out.sheen = 1;
      out.sheenRoughness = scalar;
      break;
    case 'iridescenceFactor':
      out.iridescence = scalar;
      break;
    case 'iridescenceIor':
      out.iridescenceIor = scalar;
      break;
    case 'iridescenceThicknessMinimum': {
      const [, max = 400] = material.iridescenceThicknessRange ?? [100, 400];
      out.iridescenceThicknessRange = [scalar, max] as const;
      break;
    }
    case 'iridescenceThicknessMaximum': {
      const [min = 100] = material.iridescenceThicknessRange ?? [100, 400];
      out.iridescenceThicknessRange = [min, scalar] as const;
      break;
    }
    case 'anisotropyStrength':
      out.anisotropy = scalar;
      break;
    case 'anisotropyRotation':
      out.anisotropyRotation = scalar;
      break;
    case 'dispersion':
      out.dispersionAbbeNumber = scalar === 0 ? undefined : 20 / scalar;
      break;
  }
  return out as unknown as MaterialSpec;
}

function decodePointerPath(path: string): string | undefined {
  const decoded: string[] = [];
  for (const segment of path.split('/')) {
    if (/~(?:[^01]|$)/.test(segment)) return undefined;
    decoded.push(segment.replace(/~1/g, '/').replace(/~0/g, '~'));
  }
  return decoded.join('/');
}
