// gltfTypes.ts — Minimal glTF 2.0 JSON schema types needed for parsing.
//
// Reference: glTF 2.0 specification (Khronos Group)
// https://www.khronos.org/registry/glTF/specs/2.0/glTF-2.0.html
// CREDITS.md attributes this reference.

export interface GltfJson {
  asset: { version: string; minVersion?: string; generator?: string };
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
  animations?: GltfAnimation[];
  cameras?: GltfCamera[];
  extensions?: {
    KHR_materials_variants?: {
      variants?: Array<{ name?: string }>;
    };
    [key: string]: unknown;
  };
  extensionsUsed?: string[];
  extensionsRequired?: string[];
}

export interface GltfPerspectiveCamera {
  type: 'perspective';
  name?: string;
  perspective?: {
    yfov?: number;
    znear?: number;
    zfar?: number;
    aspectRatio?: number;
    [key: string]: unknown;
  };
  extensions?: Record<string, unknown>;
  extras?: unknown;
  [key: string]: unknown;
}

export interface GltfOrthographicCamera {
  type: 'orthographic';
  name?: string;
  orthographic?: {
    xmag?: number;
    ymag?: number;
    znear?: number;
    zfar?: number;
    [key: string]: unknown;
  };
  extensions?: Record<string, unknown>;
  extras?: unknown;
  [key: string]: unknown;
}

export interface GltfUnknownCamera {
  type?: string;
  name?: string;
  perspective?: Record<string, unknown>;
  orthographic?: Record<string, unknown>;
  extensions?: Record<string, unknown>;
  extras?: unknown;
  [key: string]: unknown;
}

export type GltfCamera = GltfPerspectiveCamera | GltfOrthographicCamera | GltfUnknownCamera;

/** @public — glTF schema lattice — contract; consumed by gltfAdapter callers via typed parse results. */
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
  /** Instance morph-target weights; overrides the mesh-level `weights`. */
  weights?: number[];
  extensions?: {
    EXT_mesh_gpu_instancing?: GltfMeshGpuInstancingExtension;
    KHR_node_visibility?: {
      /** Whether this node's visual subtree is displayed. Default: true. */
      visible?: boolean;
    };
    [key: string]: unknown;
  };
}

export interface GltfMeshGpuInstancingExtension {
  attributes?: {
    TRANSLATION?: number;
    ROTATION?: number;
    SCALE?: number;
    [key: string]: number | undefined;
  };
}

/** @public — glTF schema lattice — contract; consumed by gltfAdapter callers via typed parse results. */
export interface GltfMesh {
  name?: string;
  primitives: GltfPrimitive[];
  /** Default morph-target weights (overridden by node-level `weights`). */
  weights?: number[];
}

/** @public — glTF schema lattice — contract; consumed by gltfAdapter callers via typed parse results. */
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
  extensions?: {
    KHR_draco_mesh_compression?: unknown;
    KHR_materials_variants?: {
      mappings?: Array<{
        material: number;
        variants: number[];
      }>;
    };
    [key: string]: unknown;
  };
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

/** @public — glTF schema lattice — contract; consumed by gltfAdapter callers via typed parse results. */
export interface GltfAccessorSparse {
  count: number;
  indices: { bufferView: number; byteOffset?: number; componentType: GltfComponentType };
  values: { bufferView: number; byteOffset?: number };
}

/** @public — glTF schema lattice — contract; consumed by gltfAdapter callers via typed parse results. */
export interface GltfBufferView {
  buffer: number;
  byteOffset?: number;
  byteLength: number;
  byteStride?: number;
  target?: number;
  /** Per-bufferView extensions (EXT_meshopt_compression lives here). */
  extensions?: Record<string, unknown>;
}

/** @public — glTF schema lattice — contract; consumed by gltfAdapter callers via typed parse results. */
export interface GltfBuffer {
  uri?: string;
  byteLength: number;
  /**
   * Per-buffer extensions. EXT/KHR_meshopt_compression may mark a buffer
   * `fallback: true` when it is reserved for uncompressed fallback storage.
   */
  extensions?: Record<string, unknown>;
}

/** @public — glTF schema lattice — contract; consumed by gltfAdapter callers via typed parse results. */
export interface GltfTexture {
  sampler?: number;
  source?: number;
  extensions?: {
    KHR_texture_basisu?: { source: number };
    EXT_texture_webp?: { source: number };
    MSFT_texture_dds?: { source: number };
    [key: string]: unknown;
  };
}

/** @public — glTF schema lattice — contract; consumed by gltfAdapter callers via typed parse results. */
export interface GltfImage {
  uri?: string;
  mimeType?: string;
  bufferView?: number;
  name?: string;
}

/** @public — glTF schema lattice — contract; consumed by gltfAdapter callers via typed parse results. */
export interface GltfSampler {
  magFilter?: number;
  minFilter?: number;
  wrapS?: number;
  wrapT?: number;
}

/** @public — glTF schema lattice — contract; consumed by gltfAdapter callers via typed parse results. */
export interface GltfSkin {
  inverseBindMatrices?: number;
  skeleton?: number;
  joints: number[];
  name?: string;
}

// ── Animation types ─────────────────────────────────────────────────────────
// Reference: glTF 2.0 spec §3.11 (Animations)
// https://www.khronos.org/registry/glTF/specs/2.0/glTF-2.0.html#animations

/** @public — glTF schema lattice — contract; consumed by gltfAdapter callers via typed parse results. */
export interface GltfAnimation {
  name?: string;
  channels?: GltfAnimationChannel[];
  samplers?: GltfAnimationSampler[];
}

/** @public — glTF schema lattice — contract; consumed by gltfAdapter callers via typed parse results. */
export interface GltfAnimationChannel {
  sampler: number;
  target: {
    node?: number;
    /** 'translation' | 'rotation' | 'scale' | 'weights' (unknown paths warn + skip). */
    path: string;
    extensions?: {
      KHR_animation_pointer?: {
        pointer?: string;
      };
      [key: string]: unknown;
    };
  };
}

/** @public — glTF schema lattice — contract; consumed by gltfAdapter callers via typed parse results. */
export interface GltfAnimationSampler {
  /** Accessor index for keyframe times (SCALAR float seconds). */
  input: number;
  /** Accessor index for keyframe values. */
  output: number;
  /** 'LINEAR' (default) | 'STEP' | 'CUBICSPLINE' (unknown values warn + degrade to LINEAR). */
  interpolation?: string;
}

// ── KHR_lights_punctual types ───────────────────────────────────────────────
// Reference: KHR_lights_punctual extension specification
// https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_lights_punctual/README.md

/** @public — glTF schema lattice — contract; consumed by gltfAdapter callers via typed parse results. */
export interface KhrLightsPunctualLight {
  type: 'point' | 'spot' | 'directional';
  name?: string;
  /** Linear RGB color of the light; default [1, 1, 1]. */
  color?: [number, number, number];
  /**
   * Photometric intensity of the light.
   * - point / spot: candela (cd = lm/sr)
   * - directional: lux (lx = lm/m²)
   */
  intensity?: number;
  /** Maximum range of the light (point / spot only). 0 = infinite. */
  range?: number;
  spot?: {
    innerConeAngle?: number; // radians; default 0
    outerConeAngle?: number; // radians; default π/4
  };
}

export interface KhrLightsPunctualRoot {
  lights: KhrLightsPunctualLight[];
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
    KHR_materials_unlit?: Record<string, never>;
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
    KHR_materials_dispersion?: { dispersion?: number };
    KHR_materials_emissive_strength?: { emissiveStrength?: number };
    KHR_materials_pbrSpecularGlossiness?: {
      diffuseFactor?: [number, number, number, number];
      diffuseTexture?: GltfTextureInfo;
      specularFactor?: [number, number, number];
      glossinessFactor?: number;
      specularGlossinessTexture?: GltfTextureInfo;
    };
    KHR_lights_punctual?: unknown;
    [key: string]: unknown;
  };
}

/** @public — glTF schema lattice — contract; consumed by gltfAdapter callers via typed parse results. */
export interface GltfTextureInfo {
  index: number;
  texCoord?: number;
  extensions?: {
    KHR_texture_transform?: GltfTextureTransform;
    [key: string]: unknown;
  };
}

/** @public — glTF schema lattice — contract; consumed by gltfAdapter callers via typed parse results. */
export interface GltfNormalTextureInfo extends GltfTextureInfo {
  scale?: number;
}

/** @public — glTF schema lattice — contract; consumed by gltfAdapter callers via typed parse results. */
export interface GltfOcclusionTextureInfo extends GltfTextureInfo {
  strength?: number;
}

/** @public — glTF schema lattice — contract; consumed by gltfAdapter callers via typed parse results. */
export interface GltfPbrMetallicRoughness {
  baseColorFactor?: [number, number, number, number];
  baseColorTexture?: GltfTextureInfo;
  metallicFactor?: number;
  roughnessFactor?: number;
  metallicRoughnessTexture?: GltfTextureInfo;
}

/** @public — glTF schema lattice — contract; consumed by gltfAdapter callers via typed parse results. */
export interface GltfTextureTransform {
  offset?: [number, number];
  rotation?: number;
  scale?: [number, number];
  texCoord?: number;
}
