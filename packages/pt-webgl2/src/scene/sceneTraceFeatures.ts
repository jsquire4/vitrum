import type { MaterialSpec, Scene } from '@vitrum/core';
import { assertThinFilmLayerLimit } from './materialsTexture.js';

export interface SceneTraceFeatures {
  readonly basicMaterials: boolean;
  readonly scalarRichMaterials: boolean;
  readonly mappedPbrMaterials: boolean;
  readonly mappedRichMaterials: boolean;
  /** Scene contains the single supported participating-medium boundary. */
  readonly fog: boolean;
}

const BASIC_MATERIAL_KEYS = new Set<keyof MaterialSpec>([
  'baseColor',
  'roughness',
  'metallic',
  'emissive',
  'emissiveIntensity',
  'shadingModel',
  'doubleSided',
]);

const SCALAR_RICH_MATERIAL_KEYS = new Set<keyof MaterialSpec>([
  'baseColor',
  'roughness',
  'metallic',
  'emissive',
  'emissiveIntensity',
  'shadingModel',
  'alphaMode',
  'alphaCutoff',
  'opacity',
  'doubleSided',
  'transmission',
  'ior',
  'attenuationColor',
  'attenuationDistance',
  'thickness',
  // Map-specific scalar controls are inert when their corresponding map is
  // absent. Keeping them in this tier is safe and avoids promoting adapter
  // defaults to the much larger texture graph.
  'normalScale',
  'aoMapIntensity',
  'clearcoatNormalScale',
  'bumpScale',
  'displacementScale',
  'displacementBias',
  'displacementSubdivisions',
  'lightMapIntensity',
  'sheen',
  'sheenColor',
  'sheenRoughness',
  'clearcoat',
  'clearcoatRoughness',
  'iridescence',
  'iridescenceIor',
  'iridescenceThicknessRange',
  'specularIntensity',
  'specularColor',
  'envMapIntensity',
  'spectralAttenuation',
  'dispersionAbbeNumber',
  'scatteringCoefficient',
  'scatteringAnisotropy',
  'scatteringCoefficientRGB',
  'frontLayer',
  'backLayer',
  'thinFilmStack',
  'anisotropy',
  'anisotropyRotation',
]);

const MAPPED_PBR_MATERIAL_KEYS = new Set<keyof MaterialSpec>([
  'baseColor',
  'roughness',
  'metallic',
  'emissive',
  'emissiveIntensity',
  'shadingModel',
  'alphaMode',
  'alphaCutoff',
  'opacity',
  'doubleSided',
  'ior',
  'baseColorMap',
  'normalMap',
  'normalScale',
  'roughnessMap',
  'metallicMap',
  'emissiveMap',
  'alphaMap',
  'aoMap',
  'aoMapIntensity',
  'bumpMap',
  'bumpScale',
  'displacementMap',
  'displacementScale',
  'displacementBias',
  'displacementSubdivisions',
  'lightMap',
  'lightMapIntensity',
  'envMapIntensity',
]);

const MAPPED_RICH_MATERIAL_KEYS = new Set<keyof MaterialSpec>([
  ...SCALAR_RICH_MATERIAL_KEYS,
  'baseColorMap',
  'normalMap',
  'roughnessMap',
  'metallicMap',
  'transmissionMap',
  'thicknessMap',
  'emissiveMap',
  'alphaMap',
  'aoMap',
  'clearcoatMap',
  'clearcoatRoughnessMap',
  'clearcoatNormalMap',
  'sheenColorMap',
  'sheenRoughnessMap',
  'iridescenceMap',
  'iridescenceThicknessMap',
  'anisotropyMap',
  'specularColorMap',
  'specularIntensityMap',
  'bumpMap',
  'displacementMap',
  'lightMap',
  // pt-webgl2 has no backend-specific Material.extensions contract. An empty
  // bag is inert and accepted; any key is rejected before scene allocation.
  'extensions',
]);

const SPECTRAL_CURVE_KEYS = new Set(['wavelengthStart', 'wavelengthEnd', 'values']);
const SURFACE_LAYER_KEYS = new Set(['transmission', 'roughness', 'normalMap', 'normalScale']);
const THIN_FILM_STACK_KEYS = new Set(['layers', 'incidentIor', 'angleDependent']);
const THIN_FILM_LAYER_KEYS = new Set(['ior', 'extinctionCoefficient', 'thicknessNm']);
const EMPTY_KEYS = new Set<string>();

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  if (value == null || typeof value !== 'object' || Array.isArray(value)) return false;
  const prototype: unknown = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function hasOnlyOwnKeys(
  value: unknown,
  allowedKeys: ReadonlySet<string>,
): value is Record<string, unknown> {
  if (!isPlainRecord(value) || Object.getOwnPropertySymbols(value).length !== 0) return false;
  return Object.getOwnPropertyNames(value).every((key) => allowedKeys.has(key));
}

function isScalarSurfaceLayer(value: unknown): boolean {
  if (!hasOnlyOwnKeys(value, SURFACE_LAYER_KEYS)) return false;
  // A present per-face normal map needs the texture-rich material graph. Even
  // an undefined own property stays conservative: future adapters may attach
  // lazy handles through accessors.
  return !Object.prototype.hasOwnProperty.call(value, 'normalMap');
}

function isMappedSurfaceLayer(value: unknown): boolean {
  return hasOnlyOwnKeys(value, SURFACE_LAYER_KEYS);
}

function isParticipatingMedium(material: MaterialSpec): boolean {
  if (!((material.transmission ?? 0) > 0)) return false;
  const scalar = material.scatteringCoefficient ?? 0;
  const rgb = material.scatteringCoefficientRGB;
  return scalar > 0 || (rgb != null && (rgb[0] > 0 || rgb[1] > 0 || rgb[2] > 0));
}

function isScalarThinFilmStack(value: unknown): boolean {
  if (!hasOnlyOwnKeys(value, THIN_FILM_STACK_KEYS)) return false;
  const layers = value.layers;
  return (
    Array.isArray(layers) && layers.every((layer) => hasOnlyOwnKeys(layer, THIN_FILM_LAYER_KEYS))
  );
}

/**
 * A deliberately conservative compiler tier. Any authored field outside the
 * compact opaque PBR subset selects the full shader, including unknown runtime
 * fields and the extension escape hatch. This makes false negatives cost only
 * compile time; false positives (which could drop authored behavior) are not
 * permitted.
 */
export function materialUsesBasicWebGl2Shader(material: MaterialSpec): boolean {
  if (!isPlainRecord(material) || Object.getOwnPropertySymbols(material).length !== 0) return false;
  for (const key of Object.getOwnPropertyNames(material)) {
    if (!BASIC_MATERIAL_KEYS.has(key as keyof MaterialSpec)) return false;
  }
  return true;
}

/**
 * Texture-free full-transport tier. It preserves all scalar/vector material
 * behavior while rejecting every texture handle, extension, unknown own key,
 * and future nested field. Rejection only selects the full graph; it never
 * drops authored behavior.
 */
export function materialUsesScalarRichWebGl2Shader(material: MaterialSpec): boolean {
  if (!hasOnlyOwnKeys(material, SCALAR_RICH_MATERIAL_KEYS)) {
    return false;
  }

  if (
    material.spectralAttenuation !== undefined &&
    !hasOnlyOwnKeys(material.spectralAttenuation, SPECTRAL_CURVE_KEYS)
  ) {
    return false;
  }
  if (material.frontLayer !== undefined && !isScalarSurfaceLayer(material.frontLayer)) {
    return false;
  }
  if (material.backLayer !== undefined && !isScalarSurfaceLayer(material.backLayer)) {
    return false;
  }
  if (material.thinFilmStack !== undefined && !isScalarThinFilmStack(material.thinFilmStack)) {
    return false;
  }
  return true;
}

/**
 * Texture-capable opaque base PBR. Advanced lobes, physical transmission,
 * material extensions, and unknown fields remain on the full graph.
 */
export function materialUsesMappedPbrWebGl2Shader(material: MaterialSpec): boolean {
  return hasOnlyOwnKeys(material, MAPPED_PBR_MATERIAL_KEYS);
}

/**
 * Complete supported public material graph. This is the production superset:
 * mixed maps + physical transmission/volume + Disney lobes + spectral/layers
 * all remain authored behavior. Unknown keys and non-empty extension bags are
 * rejected rather than routed to the retired legacy monolith.
 */
export function materialUsesMappedRichWebGl2Shader(material: MaterialSpec): boolean {
  if (!hasOnlyOwnKeys(material, MAPPED_RICH_MATERIAL_KEYS)) {
    return false;
  }
  if (material.extensions !== undefined && !hasOnlyOwnKeys(material.extensions, EMPTY_KEYS)) {
    return false;
  }
  if (
    material.spectralAttenuation !== undefined &&
    !hasOnlyOwnKeys(material.spectralAttenuation, SPECTRAL_CURVE_KEYS)
  ) {
    return false;
  }
  if (material.frontLayer !== undefined && !isMappedSurfaceLayer(material.frontLayer)) {
    return false;
  }
  if (material.backLayer !== undefined && !isMappedSurfaceLayer(material.backLayer)) {
    return false;
  }
  if (material.thinFilmStack !== undefined && !isScalarThinFilmStack(material.thinFilmStack)) {
    return false;
  }
  return true;
}

/**
 * Fail before any upload/allocation when a scene asks pt-webgl2 to interpret
 * material data outside its public contract. This makes the legacy full shader
 * unreachable through every accepted scene/mutation entry point.
 */
export function validateWebGl2SceneMaterials(scene: Scene, method = 'setScene'): void {
  for (const primitive of scene.primitives) {
    const material = primitive.material;
    assertThinFilmLayerLimit(
      material,
      `${method}: primitive ${JSON.stringify(String(primitive.id))} material`,
    );
    if (materialUsesMappedRichWebGl2Shader(material)) continue;
    const prefix = `[vitrum/pt-webgl2] ${method}: primitive "${String(primitive.id)}" material`;
    if (!isPlainRecord(material)) {
      throw new TypeError(`${prefix} must be a plain record with own public fields`);
    }
    const symbols = Object.getOwnPropertySymbols(material);
    if (symbols.length > 0) {
      throw new TypeError(
        `${prefix} has unsupported symbol key(s): ${symbols.map(String).join(', ')}`,
      );
    }
    const unknown = Object.getOwnPropertyNames(material)
      .filter((key) => !MAPPED_RICH_MATERIAL_KEYS.has(key as keyof MaterialSpec))
      .sort();
    if (unknown.length > 0) {
      throw new TypeError(`${prefix} has unsupported field(s): ${unknown.join(', ')}`);
    }
    if (material.extensions !== undefined && !hasOnlyOwnKeys(material.extensions, EMPTY_KEYS)) {
      const extensionKeys =
        material.extensions != null && typeof material.extensions === 'object'
          ? [
              ...Object.getOwnPropertyNames(material.extensions),
              ...Object.getOwnPropertySymbols(material.extensions).map(String),
            ].sort()
          : [`<${typeof material.extensions}>`];
      throw new TypeError(
        `${prefix}.extensions has unsupported key(s): ${extensionKeys.join(', ')}`,
      );
    }
    const malformed: string[] = [];
    if (
      material.spectralAttenuation !== undefined &&
      !hasOnlyOwnKeys(material.spectralAttenuation, SPECTRAL_CURVE_KEYS)
    )
      malformed.push('spectralAttenuation');
    if (material.frontLayer !== undefined && !isMappedSurfaceLayer(material.frontLayer)) {
      malformed.push('frontLayer');
    }
    if (material.backLayer !== undefined && !isMappedSurfaceLayer(material.backLayer)) {
      malformed.push('backLayer');
    }
    if (material.thinFilmStack !== undefined && !isScalarThinFilmStack(material.thinFilmStack))
      malformed.push('thinFilmStack');
    throw new TypeError(
      `${prefix} has malformed supported field(s): ${malformed.join(', ') || '<unknown>'}`,
    );
  }
}

export function deriveSceneTraceFeatures(scene: Scene | null): SceneTraceFeatures {
  if (scene == null) {
    return {
      basicMaterials: false,
      scalarRichMaterials: false,
      mappedPbrMaterials: false,
      mappedRichMaterials: true,
      fog: false,
    };
  }
  const basicMaterials = scene.primitives.every((primitive) =>
    materialUsesBasicWebGl2Shader(primitive.material),
  );
  const scalarRichMaterials =
    !basicMaterials &&
    scene.primitives.every((primitive) => materialUsesScalarRichWebGl2Shader(primitive.material));
  const mappedPbrMaterials =
    !basicMaterials &&
    !scalarRichMaterials &&
    scene.primitives.every((primitive) => materialUsesMappedPbrWebGl2Shader(primitive.material));
  return {
    basicMaterials,
    scalarRichMaterials,
    mappedPbrMaterials,
    mappedRichMaterials:
      !basicMaterials &&
      !scalarRichMaterials &&
      !mappedPbrMaterials &&
      scene.primitives.every((primitive) => materialUsesMappedRichWebGl2Shader(primitive.material)),
    fog: scene.primitives.some((primitive) => isParticipatingMedium(primitive.material)),
  };
}
