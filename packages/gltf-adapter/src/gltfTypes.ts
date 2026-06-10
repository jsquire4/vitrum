// gltfTypes.ts — Minimal glTF 2.0 JSON schema types needed for parsing.
//
// Reference: glTF 2.0 specification (Khronos Group)
// https://www.khronos.org/registry/glTF/specs/2.0/glTF-2.0.html
// CREDITS.md attributes this reference.

export interface GltfJson {
  asset: { version: string; generator?: string };
  scene?: number;
  scenes?: GltfScene[];
  nodes?: GltfNode[];
  meshes?: GltfMesh[];
  materials?: GltfMaterial[];
  accessors?: GltfAccessor[];
  bufferViews?: GltfBufferView[];
  buffers?: GltfBuffer[];
  textures?: GltfTexture[];
  images?: GltfImage[];
  samplers?: GltfSampler[];
  skins?: GltfSkin[];
  animations?: unknown[];
  cameras?: unknown[];
  extensions?: Record<string, unknown>;
  extensionsUsed?: string[];
  extensionsRequired?: string[];
}

export interface GltfScene {
  name?: string;
  nodes?: number[];
}

export interface GltfNode {
  name?: string;
  mesh?: number;
  skin?: number;
  camera?: number;
  children?: number[];
  /** Column-major 4×4 matrix. Mutually exclusive with TRS. */
  matrix?: number[];
  translation?: [number, number, number];
  rotation?: [number, number, number, number]; // xyzw quaternion
  scale?: [number, number, number];
  extensions?: Record<string, unknown>;
}

export interface GltfMesh {
  name?: string;
  primitives: GltfPrimitive[];
}

export interface GltfPrimitive {
  attributes: {
    POSITION?: number;
    NORMAL?: number;
    TEXCOORD_0?: number;
    TEXCOORD_1?: number;
    TANGENT?: number;
    COLOR_0?: number;
    JOINTS_0?: number;
    WEIGHTS_0?: number;
    [key: string]: number | undefined;
  };
  indices?: number;
  material?: number;
  /** 0=POINTS, 1=LINES, 2=LINE_LOOP, 3=LINE_STRIP, 4=TRIANGLES (default),
   *  5=TRIANGLE_STRIP, 6=TRIANGLE_FAN */
  mode?: number;
  targets?: Array<Record<string, number>>;
  extensions?: Record<string, unknown>;
}

export interface GltfAccessor {
  bufferView?: number;
  byteOffset?: number;
  componentType: GltfComponentType;
  normalized?: boolean;
  count: number;
  type: 'SCALAR' | 'VEC2' | 'VEC3' | 'VEC4' | 'MAT2' | 'MAT3' | 'MAT4';
  min?: number[];
  max?: number[];
  sparse?: GltfAccessorSparse;
}

export const enum GltfComponentType {
  BYTE = 5120,
  UNSIGNED_BYTE = 5121,
  SHORT = 5122,
  UNSIGNED_SHORT = 5123,
  UNSIGNED_INT = 5125,
  FLOAT = 5126,
}

export interface GltfAccessorSparse {
  count: number;
  indices: { bufferView: number; byteOffset?: number; componentType: GltfComponentType };
  values: { bufferView: number; byteOffset?: number };
}

export interface GltfBufferView {
  buffer: number;
  byteOffset?: number;
  byteLength: number;
  byteStride?: number;
  target?: number;
}

export interface GltfBuffer {
  uri?: string;
  byteLength: number;
}

export interface GltfTexture {
  sampler?: number;
  source?: number;
  extensions?: Record<string, unknown>;
}

export interface GltfImage {
  uri?: string;
  mimeType?: string;
  bufferView?: number;
  name?: string;
}

export interface GltfSampler {
  magFilter?: number;
  minFilter?: number;
  wrapS?: number;
  wrapT?: number;
}

export interface GltfSkin {
  inverseBindMatrices?: number;
  skeleton?: number;
  joints: number[];
  name?: string;
}

// ── Material types ──────────────────────────────────────────────────────────

export interface GltfMaterial {
  name?: string;
  pbrMetallicRoughness?: GltfPbrMetallicRoughness;
  normalTexture?: GltfNormalTextureInfo;
  occlusionTexture?: GltfOcclusionTextureInfo;
  emissiveFactor?: [number, number, number];
  emissiveTexture?: GltfTextureInfo;
  alphaMode?: 'OPAQUE' | 'MASK' | 'BLEND';
  alphaCutoff?: number;
  doubleSided?: boolean;
  extensions?: {
    KHR_materials_transmission?: { transmissionFactor?: number; transmissionTexture?: GltfTextureInfo };
    KHR_materials_ior?: { ior?: number };
    KHR_materials_volume?: {
      thicknessFactor?: number;
      thicknessTexture?: GltfTextureInfo;
      attenuationDistance?: number;
      attenuationColor?: [number, number, number];
    };
    KHR_materials_specular?: {
      specularFactor?: number;
      specularTexture?: GltfTextureInfo;
      specularColorFactor?: [number, number, number];
      specularColorTexture?: GltfTextureInfo;
    };
    KHR_materials_sheen?: {
      sheenColorFactor?: [number, number, number];
      sheenColorTexture?: GltfTextureInfo;
      sheenRoughnessFactor?: number;
      sheenRoughnessTexture?: GltfTextureInfo;
    };
    KHR_materials_clearcoat?: {
      clearcoatFactor?: number;
      clearcoatTexture?: GltfTextureInfo;
      clearcoatRoughnessFactor?: number;
      clearcoatRoughnessTexture?: GltfTextureInfo;
      clearcoatNormalTexture?: GltfNormalTextureInfo;
    };
    KHR_materials_iridescence?: {
      iridescenceFactor?: number;
      iridescenceTexture?: GltfTextureInfo;
      iridescenceIor?: number;
      iridescenceThicknessMinimum?: number;
      iridescenceThicknessMaximum?: number;
      iridescenceThicknessTexture?: GltfTextureInfo;
    };
    KHR_materials_anisotropy?: {
      anisotropyStrength?: number;
      anisotropyRotation?: number;
      anisotropyTexture?: GltfTextureInfo;
    };
    KHR_materials_emissive_strength?: { emissiveStrength?: number };
    KHR_lights_punctual?: unknown;
    [key: string]: unknown;
  };
}

export interface GltfTextureInfo {
  index: number;
  texCoord?: number;
  extensions?: {
    KHR_texture_transform?: GltfTextureTransform;
    [key: string]: unknown;
  };
}

export interface GltfNormalTextureInfo extends GltfTextureInfo {
  scale?: number;
}

export interface GltfOcclusionTextureInfo extends GltfTextureInfo {
  strength?: number;
}

export interface GltfPbrMetallicRoughness {
  baseColorFactor?: [number, number, number, number];
  baseColorTexture?: GltfTextureInfo;
  metallicFactor?: number;
  roughnessFactor?: number;
  metallicRoughnessTexture?: GltfTextureInfo;
}

export interface GltfTextureTransform {
  offset?: [number, number];
  rotation?: number;
  scale?: [number, number];
  texCoord?: number;
}
