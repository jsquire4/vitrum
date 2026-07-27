// gltfToScene.ts — Top-level glTF 2.0 → @vitrum/core Scene converter.
//
// Entry point: gltfToScene(input, opts) → Promise<{ scene, warnings }>
//
// Design constraints:
//   - Browser + Node compatible, with compressed-geometry codecs loaded lazily.
//   - Browser + Node compatible (no DOM-only APIs in core path; image decode is
//     pluggable via opts.decodeImage).
//   - Honest scope: v1 supports the core profile; exclusions are documented in
//     README.md and emitted as per-file warnings.
//
// Supported:
//   - Skins → SkinnedMeshPrimitive when a mesh node binds `skin` and the
//     primitive provides both JOINTS_0 + WEIGHTS_0 (JOINTS_N u8/u16,
//     WEIGHTS_N float/u8/u16, inverseBindMatrices, rest-pose joint transforms
//     converted into the skinned mesh node's local space). Secondary influence
//     sets are collapsed into the core contract's strongest four unique joints
//     per vertex with an explicit import diagnostic.
//   - Morph targets → SkinnedMeshPrimitive.morphTargets / morphTargetNormals /
//     morphTargetTangents / morphTargetUvs / morphTargetUv1s / morphWeights
//     (POSITION + NORMAL + TANGENT + TEXCOORD_0 plus the glTF UV semantic mapped
//     to core uv1 deltas; node/mesh weights; unskinned morphed meshes are promoted
//     with a synthesized identity skeleton).
//   - Animations → core AnimationClip[] on the result (LINEAR / STEP /
//     CUBICSPLINE; translation / rotation / scale / weights channels; channel
//     node ids are `gltf-node-<i>`, resolved to primitives via
//     result.animationTargets). Geometry is still imported at rest pose; the
//     host samples clips (sampleAnimationClip) and pushes updates.
//   - KHR_lights_punctual → SceneEmitter[] (point, spot, directional;
//     world-transform applied to position/direction).
//   - EXT_mesh_gpu_instancing → InstancedMeshPrimitive for mesh nodes
//     (TRANSLATION/ROTATION/SCALE accessors; nodeWorld baked into instances;
//     local instance matrices are returned for GltfSceneController animation).
//
//   - KHR_draco_mesh_compression / EXT/KHR_meshopt_compression → resolved by
//     lazy built-in decoders, with optional host overrides. `host-only` policy
//     keeps explicit fallback/error behavior for controlled runtimes.
//
// Cameras are reported as metadata for host inspection, but are not injected
// into @vitrum/core Scene because Vitrum render cameras are frame-owned.
//
// Primitive modes: TRIANGLES (4) is converted directly; TRIANGLE_STRIP (5) and
// TRIANGLE_FAN (6) are triangulated into indexed triangle lists (winding per
// glTF §3.7.2.1, degenerates dropped). POINTS/LINES/LINE_LOOP/LINE_STRIP are
// imported as deterministic fallback-generated meshes (tiny cubes / thin line
// prisms) and reported as approximate topology fidelity.
//
// References:
//   - glTF 2.0 specification (Khronos Group)
//     https://www.khronos.org/registry/glTF/specs/2.0/glTF-2.0.html
//   - KHR_lights_punctual extension
//     https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_lights_punctual/README.md

import {
  asMat4,
  cloneSparseArray,
  solveSkin,
  sparseArrayHasDefinedEntry,
  sparseArrayOwnIndices,
  validateScene as validateCoreScene,
  type AnimationClip,
  type MaterialSpec,
  type Mat4,
  type Scene,
  type SceneEmitter,
  type ScenePrimitive,
  type SkinnedMeshPrimitive,
  type TextureRef,
} from '@vitrum/core';
import type {
  GltfAccessor,
  GltfJson,
  GltfNode,
  GltfPrimitive,
  KhrLightsPunctualRoot,
} from './gltfTypes.js';
import { GltfComponentType } from './gltfTypes.js';
import {
  buildTextureHandleMap,
  GLTF_TEXTURE_SOURCE_EXTENSIONS,
  gltfTextureRefSource,
  type DecodeImageFn,
  type GltfImageBytesMap,
  type GltfTextureAcquisitionDiagnosticCode,
  type GltfTextureSourceExtension,
} from './textures.js';
import { decodeGltfUtf8, parseGlb } from './glbParser.js';
import {
  accessorBufferViewRange,
  unpackAccessorFloat,
  unpackAccessorUint32,
  validateAccessorNormalization,
  validateBufferViewAccess,
  type GltfAccessorDiagnostic,
  type GltfAccessorDiagnosticCode,
} from './accessors.js';
import { buildWorldTransforms, composeTrsMat4, mat4Invert, mat4Mul } from './transforms.js';
import {
  convertMaterial,
  GltfMaterialImportError,
  GLTF_DEFAULT_MATERIAL,
  type GltfMaterialDiagnostic,
  type GltfMaterialDiagnosticCode,
} from './materials.js';
import { generateVertexNormals } from './normals.js';
import { generateTangents } from './tangents.js';
import {
  animationNodeId,
  convertAnimations,
  GltfAnimationImportError,
  type GltfAnimationImportDiagnosticCode,
} from './animations.js';
import { resolveCompression } from './compression.js';
import type {
  DracoDecodeFn,
  GltfCompressionDiagnostic,
  GltfCompressionDiagnosticCode,
  MeshoptDecodeFn,
} from './compression.js';
import { createImportBuiltinCompressionDecoders } from './builtinCompressionDecoders.js';
import {
  GLTF_MODE_TRIANGLE_FAN,
  GLTF_MODE_TRIANGLE_STRIP,
  sequentialIndices,
  triangulateTopology,
} from './triangulation.js';
import { GltfParseFailed } from './errors.js';
import {
  buildPointLineFallbackGeometry,
  isPointLineMode,
  pointLineModeName,
} from './primitiveModeFallback.js';
import {
  collectGltfSceneReachability,
  gltfPrimitiveKey,
  type GltfSceneReachability,
} from './sceneScope.js';
import { analyzeGltfAsset } from './featureReport.js';
import {
  collectSceneCameras,
  validateGltfCameraMetadata,
  type GltfSceneCamera,
} from './cameraMetadata.js';
import {
  attachGltfResourceOwner,
  createAsyncResourceLimiter,
  DecodedImageHandleOwner,
  GLTF_INPUT_RESOURCE_KEY,
  gltfArrayBufferByteLength,
  gltfBufferResourceKey,
  GltfResourceLimitError,
  ImportResourceLedger,
  type GltfImportResourceContext,
  type GltfImportResourceLimits,
} from './importResourceBudget.js';
export type { GltfImportResourceLimits } from './importResourceBudget.js';
export type {
  GltfOrthographicCameraProjection,
  GltfPerspectiveCameraProjection,
  GltfSceneCamera,
} from './cameraMetadata.js';

function mapSparseArray<T, U>(
  values: ReadonlyArray<T | undefined>,
  mapEntry: (value: T | undefined, index: number) => U | undefined,
): Array<U | undefined> {
  const mapped: Array<U | undefined> = [];
  for (const index of sparseArrayOwnIndices(values)) {
    mapped[index] = mapEntry(values[index], index);
  }
  return mapped;
}

const GLTF_PRIMITIVE_MODE_TRIANGLES = 4;

const MATERIAL_TEXTURE_REF_FIELDS = [
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
] as const satisfies readonly (keyof MaterialSpec)[];

export type MaterialTextureRefField = (typeof MATERIAL_TEXTURE_REF_FIELDS)[number];

/** Compressed-geometry decoder selection. Built-ins are lazy and are the default. */
export type GltfCompressionDecoderPolicy = 'builtin' | 'host-only';

const SUPPORTED_REQUIRED_EXTENSIONS = new Set([
  'KHR_draco_mesh_compression',
  'EXT_meshopt_compression',
  'KHR_meshopt_compression',
  'KHR_lights_punctual',
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
  'KHR_materials_variants',
  'KHR_materials_pbrSpecularGlossiness',
  'KHR_animation_pointer',
  'KHR_node_visibility',
  // Accessor unpacking already converts BYTE/SHORT normalized attributes to
  // float32, which is the representation contract KHR_mesh_quantization needs.
  'KHR_mesh_quantization',
  'KHR_texture_transform',
  'EXT_mesh_gpu_instancing',
]);

export interface GltfToSceneOptions {
  /**
   * Import-wide memory and concurrency ceilings. A value of 0 disables an
   * individual byte/pixel ceiling; operation concurrency must remain positive.
   */
  readonly resourceLimits?: GltfImportResourceLimits;

  /**
   * Pre-loaded buffer data for .gltf files with external buffer URIs.
   * Key = buffer index (0-based), value = the loaded ArrayBuffer.
   *
   * The adapter does NOT fetch buffer URIs. If a glTF file references external
   * buffers, the host must fetch them and supply them here. GLB files carry
   * their binary chunk inline and do not need this.
   *
   * For GLB files, buffer 0 is automatically populated from the binary chunk.
   */
  readonly buffers?: ReadonlyMap<number, ArrayBuffer> | Record<number, ArrayBuffer>;

  /**
   * Optional async image decode callback.
   *
   * Called once per unique image (deduplicated by image index). Receives the
   * raw image bytes and mimeType. Return value becomes the `handle` in each
   * TextureRef that references this image.
   *
   * When omitted:
   *   - Browser: `createImageBitmap(new Blob([bytes], { type }))` is used.
   *     Result is an `ImageBitmap`, accepted by pt-webgpu. CPU-readable
   *     backends such as pt-webgl2 and walkaround-hybrid still need
   *     decodeSceneTextures() or a host-supplied pixel handle.
   *   - Non-browser: returns `{ kind: 'raw-image', mimeType, data: Uint8Array }`.
   *     Backends require decodeSceneTextures() or a host decodeImage/decodePixels
   *     hook before those raw image bytes are backend-ready.
   *
   * sRGB ownership: the adapter passes bytes as-is. The callback and/or backend
   * are responsible for colorspace-correct upload (sRGB for baseColor/emissive;
   * linear for normal/ORM/ao/lightMap/bumpMap). See README for details.
   */
  readonly decodeImage?: DecodeImageFn;

  /**
   * Pre-loaded image bytes for .gltf files with external image URIs.
   * Key = glTF image index. `loadGltfAsset()` fills this map automatically
   * for URL/base-URI inputs; low-level `gltfToScene()` callers can provide it
   * directly when they own resource resolution.
   */
  readonly imageBytes?: GltfImageBytesMap;

  /**
   * Texture-source extensions the host can decode and wants the adapter to
   * select over a texture's base `source`.
   *
   * glTF texture-source extensions (`KHR_texture_basisu`, `EXT_texture_webp`,
   * `MSFT_texture_dds`) point a texture at an alternate image index. The adapter
   * can fetch/read those bytes, but the host is responsible for decode support
   * through `decodeImage` or browser-native image decoding. When omitted, the
   * adapter uses the base `texture.source` fallback and treats required
   * texture-source extensions as unsupported.
   */
  readonly textureSourceExtensions?: readonly GltfTextureSourceExtension[];

  /**
   * Active `KHR_materials_variants` selection. Pass a root variant name or
   * variant index. When omitted, the adapter uses each primitive's vanilla
   * glTF `material` fallback.
   */
  readonly materialVariant?: string | number;

  /**
   * Index of the glTF scene to import (0-based). Defaults to gltf.scene or 0.
   */
  readonly sceneIndex?: number;

  /**
   * Optional host override for the lazy built-in
   * `KHR_draco_mesh_compression` decoder. Receives the
   * compressed blob plus the extension's semantic → Draco-attribute-unique-id
   * map. Its optional third context argument supplies the declared schema for
   * each Draco-owned accessor. The hook must return decoded typed arrays whose
   * counts/types match those accessors (which per spec describe the decoded
   * data — the adapter then applies its standard accessor conversion,
   * including `normalized` handling). May be sync or async.
   *
   * With `compressionDecoderPolicy: 'host-only'` and no hook, optional
   * primitives use fully validated uncompressed fallbacks when available;
   * required or fallback-free compressed geometry fails closed.
   */
  readonly dracoDecode?: DracoDecodeFn;

  /**
   * Optional host override for the lazy built-in
   * `EXT/KHR_meshopt_compression` decoder.
   *
   * Mirrors `MeshoptDecoder.decodeGltfBuffer` from `meshoptimizer` (see README
   * "Compressed geometry"): receives the compressed bytes plus the extension's
   * `count` / `byteStride` / `mode` / `filter` and must return exactly
   * `count × byteStride` decoded bytes. Decoding happens at bufferView
   * resolution, so geometry, animation and image consumers all transparently
   * see decompressed data. May be sync or async.
   *
   * With `compressionDecoderPolicy: 'host-only'` and no hook, optional
   * bufferViews use validated fallback bytes when available;
   * required or fallback-free compressed data fails closed.
   */
  readonly meshoptDecode?: MeshoptDecodeFn;

  /**
   * Selects whether missing compression hooks fall back to the adapter's lazy
   * built-in Draco/meshoptimizer decoders. Defaults to `builtin`. Use
   * `host-only` only when the host deliberately controls codec availability;
   * explicit `dracoDecode` / `meshoptDecode` hooks always take precedence.
   */
  readonly compressionDecoderPolicy?: GltfCompressionDecoderPolicy;

  /**
   * Half-width, in asset units, for generated mesh fallback geometry used when
   * importing glTF POINTS/LINES/LINE_LOOP/LINE_STRIP primitive modes. When
   * omitted, the adapter derives a small deterministic size from the primitive
   * bounding-box diagonal.
   */
  readonly pointLineFallbackRadius?: number;
}

export interface GltfToSceneResult {
  readonly scene: Scene;
  /**
   * glTF cameras reachable from the selected scene, one entry per camera node.
   * These are metadata only: @vitrum/core Scene has no render-camera field, so
   * hosts that want to honor an authored camera should translate one of these
   * records into per-frame `FrameInput` view/projection matrices.
   */
  readonly cameras: ReadonlyArray<GltfSceneCamera>;
  /**
   * glTF animations converted to core `AnimationClip`s (empty when the file
   * has none).
   *
   * CHANNEL-TARGET MAPPING: each channel's `target.node` is the stable id
   * `gltf-node-<index>` (the glTF node index — see `animationNodeId()`), NOT a
   * `ScenePrimitive.id`. Use `animationTargets` to resolve a channel node id
   * to the primitive ids created from that node's mesh:
   *
   *   - `translation` / `rotation` / `scale` channels: re-compose the node's
   *     local TRS from the sampled values, then push either
   *     `{ transform }` for ordinary mesh-like primitives or `{ instances }`
   *     for `EXT_mesh_gpu_instancing` primitives (using the returned
   *     `instancingBindings` local matrices) through `engine.updatePrimitive`.
   *     Channels animating an ANCESTOR of a mesh node are handled by
   *     `GltfSceneController`, which retains the glTF hierarchy and recomputes
   *     world transforms.
   *   - `weights` channels: write the sampled vector into the skinned
   *     primitive's `morphWeights`, re-run `solveSkin` (@vitrum/core), and
   *     push the deformed `positions`/`normals` through `updatePrimitive`.
   *   - Joint-node channels (skeletal animation): the channel id names the
   *     joint's glTF node; after sampling the skeleton pose, hosts rebuild
   *     `SkinnedMeshPrimitive.bones` (joint world matrices) and re-run
   *     `solveSkin`. The adapter does not retarget skeletal clips.
   *
   * Evaluate clips with `sampleAnimationClip(clip, time)` from `@vitrum/core`.
   */
  readonly animations: ReadonlyArray<AnimationClip>;
  /**
   * Maps an animation channel node id (`gltf-node-<index>`) to the
   * `ScenePrimitive.id`s created from that node's mesh. Nodes that produced no
   * primitives (joints, empties, camera/light nodes) are absent.
   */
  readonly animationTargets: Readonly<Record<string, ReadonlyArray<string>>>;
  /**
   * Materials converted during import, indexed by the original glTF material
   * index. `gltfToScene()` always provides this; it is optional on the public
   * type so tests/hosts can still construct controller input manually.
   */
  readonly convertedMaterials?: ReadonlyArray<MaterialSpec>;
  /**
   * Primitive provenance needed by `GltfSceneController.setVariant()` to patch
   * only primitives affected by KHR_materials_variants.
   */
  readonly materialVariantBindings?: ReadonlyArray<GltfMaterialVariantBinding>;
  /**
   * Primitive-to-glTF-material provenance. Used by KHR_animation_pointer material
   * playback to route `/materials/N/...` channels to every imported primitive
   * that references that material.
   */
  readonly materialBindings?: ReadonlyArray<GltfMaterialBinding>;
  /**
   * Primitive provenance for `EXT_mesh_gpu_instancing`. Each local instance
   * matrix is the accessor-authored TRS before the node world transform is
   * applied. `GltfSceneController` uses these to update `InstancedMeshPrimitive`
   * instance matrices when a node or ancestor animates.
   */
  readonly instancingBindings?: ReadonlyArray<GltfInstancingBinding>;
  /**
   * Punctual-emitter provenance. The scene controller uses this to recompute
   * point/spot positions and spot/directional directions when a light node or
   * any ancestor is animated.
   */
  readonly punctualEmitterBindings?: ReadonlyArray<GltfPunctualEmitterBinding>;
  /**
   * Complete imported visual-object inventory before KHR_node_visibility is
   * applied. Hidden objects are absent from `scene`, but the controller retains
   * these snapshots so an animated visibility pointer can restore them.
   */
  readonly nodeVisibilityPrimitives?: ReadonlyArray<ScenePrimitive>;
  readonly nodeVisibilityEmitters?: ReadonlyArray<SceneEmitter>;
  /** Non-fatal issues encountered during conversion. Inspect these for skipped
   *  primitives, unsupported extensions, missing buffers, sparse patches, etc. */
  readonly warnings: string[];
  /** Structured form of converter-owned warnings. Existing string warnings are
   *  preserved for compatibility; diagnostics give hosts stable codes and
   *  source paths for filtering, UI, and compatibility reports. */
  readonly diagnostics: readonly GltfImportDiagnostic[];
}

export interface GltfMaterialVariantBinding {
  readonly primitiveId: string;
  readonly meshIndex: number;
  readonly primitiveIndex: number;
  readonly baseMaterialIndex?: number;
  readonly basePatch?: GltfMaterialVariantPrimitivePatch;
  readonly variantPatches?: ReadonlyArray<GltfMaterialVariantPatch>;
}

export interface GltfMaterialBinding {
  readonly primitiveId: string;
  readonly materialIndex: number;
}

export interface GltfMaterialVariantPrimitivePatch {
  readonly materialIndex?: number;
  readonly materialRouting?: MaterialSpec;
  readonly droppedTextureFields?: readonly MaterialTextureRefField[];
  readonly uvs?: Float32Array | undefined;
  readonly uv1?: Float32Array | undefined;
  readonly uvSets?: ReadonlyArray<Float32Array | undefined> | undefined;
  readonly tangents?: Float32Array | undefined;
}

export interface GltfMaterialVariantPatch {
  readonly variantIndex: number;
  readonly patch: GltfMaterialVariantPrimitivePatch;
}

export interface GltfInstancingBinding {
  readonly primitiveId: string;
  readonly nodeIndex: number;
  readonly localInstanceTransforms: ReadonlyArray<Mat4>;
  /**
   * Rest-pose deformation source for an instanced skinned or morphed mesh.
   * The rendered scene primitive remains one real `InstancedMeshPrimitive`;
   * `GltfSceneController` CPU-solves this shared vertex stream once per pose
   * and patches that stream together with the complete instance array.
   */
  readonly deformationSource?: SkinnedMeshPrimitive;
}

export interface GltfPunctualEmitterBinding {
  readonly emitterId: string;
  readonly nodeIndex: number;
  readonly lightIndex: number;
}

export type GltfImportDiagnosticCode =
  | GltfTextureAcquisitionDiagnosticCode
  | GltfAnimationImportDiagnosticCode
  | GltfAccessorDiagnosticCode
  | GltfMaterialDiagnosticCode
  | GltfCompressionDiagnosticCode
  | 'unsupported-version'
  | 'unsupported-required-extension'
  | 'invalid-camera'
  | 'ignored-camera'
  | 'generated-tangents'
  | 'generated-flat-normals'
  | 'unreadable-normal'
  | 'unreadable-optional-attribute'
  | 'missing-tangent-texcoord'
  | 'missing-material-texcoord'
  | 'tangent-generation-failed'
  | 'skin-rest-pose'
  | 'singular-skin-transform'
  | 'unreadable-inverse-bind-matrices'
  | 'unreadable-skin-joints'
  | 'unreadable-skin-weights'
  | 'ignored-skin-attributes'
  | 'incomplete-skin-attributes'
  | 'collapsed-skin-influence-sets'
  | 'scene-not-found'
  | 'ignored-gpu-instancing'
  | 'ignored-gpu-instancing-attribute'
  | 'unsupported-primitive-mode'
  | 'fallback-generated-primitive-mode'
  | 'missing-position'
  | 'unreadable-position'
  | 'unreadable-indices'
  | 'invalid-primitive-attribute'
  | 'ignored-vertex-color-set'
  | 'ignored-primitive-attribute'
  | 'empty-triangulated-primitive'
  | 'material-variant-not-found'
  | 'material-variant-list-malformed'
  | 'material-not-found'
  | 'material-variant-material-missing'
  | 'material-variant-mapping-malformed'
  | 'ignored-material-texcoord'
  | 'ignored-morph-target-texcoord'
  | 'ignored-morph-target-attribute'
  | 'invalid-morph-target-delta-length'
  | 'morph-weight-count-mismatch'
  | 'missing-punctual-light'
  | 'unsupported-punctual-light-type';

export interface GltfImportDiagnostic {
  readonly severity: 'warning' | 'error';
  readonly code: GltfImportDiagnosticCode;
  readonly path: string;
  readonly message: string;
}

export class GltfImportError extends Error {
  readonly diagnostics: readonly GltfImportDiagnostic[];

  constructor(message: string, diagnostics: readonly GltfImportDiagnostic[]) {
    super(message);
    this.name = 'GltfImportError';
    this.diagnostics = diagnostics;
  }
}

function diagnosticChainWithTerminal(
  diagnostics: readonly GltfImportDiagnostic[],
  terminal: GltfImportDiagnostic,
): readonly GltfImportDiagnostic[] {
  const alreadyRecorded = diagnostics.some(
    (diagnostic) =>
      diagnostic === terminal ||
      (diagnostic.code === terminal.code &&
        diagnostic.path === terminal.path &&
        diagnostic.message === terminal.message),
  );
  return alreadyRecorded ? [...diagnostics] : [...diagnostics, terminal];
}

function emitImportDiagnostic(
  warnings: string[],
  diagnostics: GltfImportDiagnostic[],
  diagnostic: GltfImportDiagnostic,
): void {
  diagnostics.push(diagnostic);
  warnings.push(diagnostic.message);
}

function throwImportBoundaryError(
  code: GltfImportDiagnosticCode,
  path: string,
  message: string,
): never {
  throw new GltfImportError(message, [{ severity: 'error', code, path, message }]);
}

function parseCanonicalSetSemantic(
  semantic: string,
  prefix: 'TEXCOORD' | 'COLOR' | 'JOINTS' | 'WEIGHTS',
  path: string,
  code: 'ignored-primitive-attribute' | 'ignored-morph-target-attribute',
): number | undefined {
  if (semantic !== prefix && !semantic.startsWith(`${prefix}_`)) return undefined;
  const match = new RegExp(`^${prefix}_(\\d+)$`, 'u').exec(semantic);
  if (match == null) {
    throwImportBoundaryError(
      code,
      path,
      `[vitrum/gltf-adapter] Reserved semantic "${semantic}" must use the ` +
        `${prefix}_<non-negative canonical integer> form.`,
    );
  }
  const suffix = match[1]!;
  const index = Number(suffix);
  if (!Number.isSafeInteger(index)) {
    throwImportBoundaryError(
      code,
      path,
      `[vitrum/gltf-adapter] Reserved semantic "${semantic}" exceeds the supported non-negative safe-integer range.`,
    );
  }
  const canonical = `${prefix}_${index}`;
  if (semantic !== canonical) {
    throwImportBoundaryError(
      code,
      path,
      `[vitrum/gltf-adapter] Reserved semantic "${semantic}" is not canonical; use "${canonical}".`,
    );
  }
  return index;
}

const PRIMITIVE_ATTRIBUTE_SEMANTICS = new Set(['POSITION', 'NORMAL', 'TANGENT']);
const MORPH_TARGET_ATTRIBUTE_SEMANTICS = new Set(['POSITION', 'NORMAL', 'TANGENT']);
const INDEXED_ATTRIBUTE_SEMANTIC_PREFIXES = [
  'TEXCOORD',
  'COLOR',
  'JOINTS',
  'WEIGHTS',
] as const;

function validatePrimitiveAttributeSemantic(semantic: string, path: string): void {
  if (PRIMITIVE_ATTRIBUTE_SEMANTICS.has(semantic) || semantic.startsWith('_')) return;
  for (const prefix of INDEXED_ATTRIBUTE_SEMANTIC_PREFIXES) {
    if (
      parseCanonicalSetSemantic(
        semantic,
        prefix,
        path,
        'ignored-primitive-attribute',
      ) !== undefined
    ) {
      return;
    }
  }
  throwImportBoundaryError(
    'ignored-primitive-attribute',
    path,
    `[vitrum/gltf-adapter] Unknown primitive attribute "${semantic}" cannot be represented exactly.`,
  );
}

function validateMorphTargetAttributeSemantic(semantic: string, path: string): void {
  if (MORPH_TARGET_ATTRIBUTE_SEMANTICS.has(semantic) || semantic.startsWith('_')) return;
  for (const prefix of INDEXED_ATTRIBUTE_SEMANTIC_PREFIXES) {
    const setIndex = parseCanonicalSetSemantic(
      semantic,
      prefix,
      path,
      'ignored-morph-target-attribute',
    );
    if (setIndex === undefined) continue;
    if (prefix === 'TEXCOORD' || prefix === 'COLOR') return;
    break;
  }
  throwImportBoundaryError(
    'ignored-morph-target-attribute',
    path,
    `[vitrum/gltf-adapter] Unknown morph-target attribute "${semantic}" cannot be represented exactly.`,
  );
}

function validateReachablePrimitiveSemantics(
  gltf: GltfJson,
  reachability: GltfSceneReachability,
): void {
  for (const [meshIndex, mesh] of (gltf.meshes ?? []).entries()) {
    for (const [primitiveIndex, primitive] of mesh.primitives.entries()) {
      if (!reachability.primitiveKeys.has(gltfPrimitiveKey(meshIndex, primitiveIndex))) {
        continue;
      }
      const primitivePath = `meshes[${meshIndex}].primitives[${primitiveIndex}]`;
      for (const semantic of Object.keys(primitive.attributes ?? {})) {
        validatePrimitiveAttributeSemantic(
          semantic,
          `${primitivePath}.attributes.${semantic}`,
        );
      }
      for (const [targetIndex, target] of (primitive.targets ?? []).entries()) {
        for (const semantic of Object.keys(target)) {
          validateMorphTargetAttributeSemantic(
            semantic,
            `${primitivePath}.targets[${targetIndex}].${semantic}`,
          );
        }
      }
    }
  }
}

function rethrowResourceLimitError(error: unknown): void {
  if (error instanceof GltfResourceLimitError) throw error;
}

function describeImportBoundaryValue(value: unknown): string {
  return value !== null && typeof value === 'object'
    ? Object.prototype.toString.call(value)
    : String(value);
}

interface ParsedGltfVersion {
  readonly major: number;
  readonly minor: number;
  readonly source: string;
}

function parseGltfVersion(value: unknown, path: 'asset.version' | 'asset.minVersion'): ParsedGltfVersion {
  if (typeof value !== 'string') {
    throwImportBoundaryError(
      'unsupported-version',
      path,
      `[vitrum/gltf-adapter] glTF ${path} must use the "<major>.<minor>" string form; received ${JSON.stringify(value)}.`,
    );
  }
  const match = /^(\d+)\.(\d+)$/u.exec(value);
  const major = match === null ? Number.NaN : Number(match[1]);
  const minor = match === null ? Number.NaN : Number(match[2]);
  if (!Number.isSafeInteger(major) || !Number.isSafeInteger(minor)) {
    throwImportBoundaryError(
      'unsupported-version',
      path,
      `[vitrum/gltf-adapter] glTF ${path} must use allocation-safe integer major/minor components; received ${JSON.stringify(value)}.`,
    );
  }
  return { major, minor, source: value };
}

function compareGltfVersions(a: ParsedGltfVersion, b: ParsedGltfVersion): number {
  if (a.major !== b.major) return a.major < b.major ? -1 : 1;
  if (a.minor !== b.minor) return a.minor < b.minor ? -1 : 1;
  return 0;
}

/** @internal Shared by the low-level converter and the high-level resource loader. */
export function assertSupportedGltfVersion(gltf: GltfJson): void {
  const asset = gltf.asset as
    | { readonly version?: unknown; readonly minVersion?: unknown }
    | undefined;
  const implemented: ParsedGltfVersion = { major: 2, minor: 0, source: '2.0' };
  const version = parseGltfVersion(asset?.version, 'asset.version');
  if (version.major !== implemented.major) {
    throwImportBoundaryError(
      'unsupported-version',
      'asset.version',
      `[vitrum/gltf-adapter] glTF asset.version ${JSON.stringify(version.source)} targets unsupported major version ${version.major}; this adapter implements glTF 2.x.`,
    );
  }
  if (asset?.minVersion === undefined) return;

  const minVersion = parseGltfVersion(asset.minVersion, 'asset.minVersion');
  if (compareGltfVersions(minVersion, version) > 0) {
    throwImportBoundaryError(
      'unsupported-version',
      'asset.minVersion',
      `[vitrum/gltf-adapter] glTF asset.minVersion ${JSON.stringify(minVersion.source)} exceeds asset.version ${JSON.stringify(version.source)}.`,
    );
  }
  if (compareGltfVersions(minVersion, implemented) > 0) {
    throwImportBoundaryError(
      'unsupported-version',
      'asset.minVersion',
      `[vitrum/gltf-adapter] glTF asset.minVersion ${JSON.stringify(minVersion.source)} requires features newer than this adapter's glTF 2.0 implementation.`,
    );
  }
}

function resolveValidatedSceneIndex(
  gltf: GltfJson,
  requestedSceneIndex: number | undefined,
): number {
  const scenesValue: unknown = gltf.scenes;
  if (scenesValue !== undefined && !Array.isArray(scenesValue)) {
    throwImportBoundaryError(
      'scene-not-found',
      'scenes',
      '[vitrum/gltf-adapter] glTF scenes must be an array when supplied.',
    );
  }
  const scenes = scenesValue as GltfJson['scenes'];
  const declaredDefault: unknown = gltf.scene;
  if (declaredDefault !== undefined) {
    if (!Number.isSafeInteger(declaredDefault) || (declaredDefault as number) < 0) {
      throwImportBoundaryError(
        'scene-not-found',
        'scene',
        `[vitrum/gltf-adapter] glTF scene must be a non-negative safe integer; received ${describeImportBoundaryValue(declaredDefault)}.`,
      );
    }
    const defaultSceneIndex = declaredDefault as number;
    if (scenes == null || scenes[defaultSceneIndex] == null) {
      throwImportBoundaryError(
        'scene-not-found',
        'scene',
        `[vitrum/gltf-adapter] glTF scene points at nonexistent scenes[${defaultSceneIndex}].`,
      );
    }
  }
  if (requestedSceneIndex !== undefined) {
    if (!Number.isSafeInteger(requestedSceneIndex) || requestedSceneIndex < 0) {
      throwImportBoundaryError(
        'scene-not-found',
        'options.sceneIndex',
        `[vitrum/gltf-adapter] opts.sceneIndex must be a non-negative safe integer; received ${String(requestedSceneIndex)}.`,
      );
    }
    if (scenes == null || scenes[requestedSceneIndex] == null) {
      throwImportBoundaryError(
        'scene-not-found',
        'options.sceneIndex',
        `[vitrum/gltf-adapter] opts.sceneIndex ${requestedSceneIndex} does not identify an existing glTF scene.`,
      );
    }
    return requestedSceneIndex;
  }
  if (declaredDefault !== undefined) return declaredDefault as number;
  if (scenes !== undefined) {
    if (scenes[0] == null) {
      throwImportBoundaryError(
        'scene-not-found',
        'scenes[0]',
        '[vitrum/gltf-adapter] glTF declares a scenes array but has no valid implicit scene 0.',
      );
    }
    return 0;
  }
  // glTF permits assets without scenes. With no explicit/default selection,
  // this legitimately maps to an empty core Scene.
  return 0;
}

/**
 * Convert a glTF 2.0 file to a `@vitrum/core` Scene.
 *
 * @param input - Either a `GltfJson` object (parsed JSON for .gltf files) or
 *   an `ArrayBuffer` (raw GLB or JSON bytes). When an `ArrayBuffer` is provided
 *   whose first 4 bytes are the GLB magic, it is parsed as GLB; otherwise it is
 *   decoded as UTF-8 JSON.
 * @param opts  - Optional conversion settings (see `GltfToSceneOptions`).
 */
interface BuildPrimitiveContext {
  readonly gltf: GltfJson;
  readonly buffers: Map<number, ArrayBuffer>;
  readonly warnings: string[];
  readonly diagnostics: GltfImportDiagnostic[];
  readonly onAccessorDiagnostic: (diagnostic: GltfAccessorDiagnostic) => void;
  readonly coreMaterials: MaterialSpec[];
  readonly opts: GltfToSceneOptions;
  readonly selectedMaterialVariant: number | undefined;
  readonly primitives: ScenePrimitive[];
  readonly animationTargets: Record<string, string[]>;
  readonly materialVariantBindings: GltfMaterialVariantBinding[];
  readonly materialBindings: GltfMaterialBinding[];
  readonly instancingBindings: GltfInstancingBinding[];
  readonly primId: { value: number };
  readonly resourceLedger: ImportResourceLedger;
}

interface BuildPrimitiveNodeContext {
  readonly node: GltfNode;
  readonly nodeIdx: number;
  /** The node's mesh index, already narrowed to a defined number by the caller. */
  readonly meshIndex: number;
  readonly mesh: NonNullable<GltfJson['meshes']>[number];
  readonly worldMat: Mat4;
  readonly skinData: SkinData | undefined;
  readonly bones: Float32Array | undefined;
  readonly boneInverses: Float32Array | undefined;
  readonly instanceTransforms: MeshGpuInstancingTransforms | undefined;
}

// Section-8 inner loop body (D15-4): builds/pushes the ScenePrimitive(s) for one
// glTF mesh primitive. Extracted verbatim from the node→mesh→primitive flatten
// loop; the primitive-loop-level `continue;` statements became `return;` (nested
// skin-influence-set-loop `continue;` are preserved). All previously outer-scope
// state is threaded via ctx (function-scope constants + accumulators) and nodeCtx
// (per-node state); `primIdCounter` is a mutable holder.
function buildPrimitiveFromMeshPrimitive(
  ctx: BuildPrimitiveContext,
  nodeCtx: BuildPrimitiveNodeContext,
  prim: GltfPrimitive,
  primitiveIndex: number,
): void {
  const {
    gltf,
    buffers,
    warnings,
    diagnostics,
    onAccessorDiagnostic,
    coreMaterials,
    opts,
    selectedMaterialVariant,
    primitives,
    animationTargets,
    materialVariantBindings,
    materialBindings,
    instancingBindings,
    resourceLedger,
  } = ctx;
  const {
    node,
    nodeIdx,
    meshIndex,
    mesh,
    worldMat,
    skinData,
    bones,
    boneInverses,
    instanceTransforms,
  } = nodeCtx;
  const primitivePath = `meshes[${node.mesh}].primitives[${primitiveIndex}]`;
  // Mode check — native triangle modes plus deterministic generated-mesh
  // fallback for point/line modes. Unknown modes are still skipped.
  const mode = prim.mode ?? GLTF_PRIMITIVE_MODE_TRIANGLES;
  if (
    mode !== GLTF_PRIMITIVE_MODE_TRIANGLES &&
    mode !== GLTF_MODE_TRIANGLE_STRIP &&
    mode !== GLTF_MODE_TRIANGLE_FAN &&
    !isPointLineMode(mode)
  ) {
    throwImportBoundaryError(
      'unsupported-primitive-mode',
      `${primitivePath}.mode`,
      `[vitrum/gltf-adapter] Mesh "${mesh.name ?? node.mesh}" primitive has unsupported mode ${String(mode)}.`,
    );
  }

  // ── Unpack attributes ──────────────────────────────────────────────────
  const posIdx = prim.attributes['POSITION'];
  if (posIdx === undefined) {
    throwImportBoundaryError(
      'missing-position',
      `${primitivePath}.attributes.POSITION`,
      `[vitrum/gltf-adapter] Mesh "${mesh.name ?? node.mesh}" primitive has no POSITION attribute.`,
    );
  }

  const positionAccessor = gltf.accessors?.[posIdx];
  const meshQuantization = _usesMeshQuantization(gltf);
  if (
    positionAccessor === undefined ||
    positionAccessor.type !== 'VEC3' ||
    positionAccessor.count <= 0 ||
    !_validPositionEncoding(positionAccessor, meshQuantization)
  ) {
    throwImportBoundaryError(
      'invalid-primitive-attribute',
      `${primitivePath}.attributes.POSITION`,
      `[vitrum/gltf-adapter] Mesh "${mesh.name ?? node.mesh}" POSITION accessor ${posIdx} must be a non-empty VEC3/FLOAT accessor, or a BYTE/UNSIGNED_BYTE/SHORT/UNSIGNED_SHORT accessor when KHR_mesh_quantization is declared.`,
    );
  }

  let positions: Float32Array;
  try {
    positions = unpackAccessorFloat(
      gltf,
      buffers,
      posIdx,
      warnings,
      onAccessorDiagnostic,
      resourceLedger,
    );
  } catch (e) {
    rethrowResourceLimitError(e);
    throwImportBoundaryError(
      'unreadable-position',
      `${primitivePath}.attributes.POSITION`,
      `[vitrum/gltf-adapter] Failed to read POSITION for mesh "${mesh.name ?? node.mesh}": ${String(e)}`,
    );
  }

  if (positions.some((value) => !Number.isFinite(value))) {
    throwImportBoundaryError(
      'invalid-primitive-attribute',
      `${primitivePath}.attributes.POSITION`,
      `[vitrum/gltf-adapter] Mesh "${mesh.name ?? node.mesh}" POSITION accessor ${posIdx} contains non-finite values.`,
    );
  }

  const vertexCount = Math.floor(positions.length / 3);

  // Indices (optional).
  let indices: Uint32Array | undefined;
  if (prim.indices !== undefined) {
    try {
      indices = unpackAccessorUint32(
        gltf,
        buffers,
        prim.indices,
        null,
        onAccessorDiagnostic,
        resourceLedger,
      );
    } catch (e) {
      rethrowResourceLimitError(e);
      throwImportBoundaryError(
        'unreadable-indices',
        `${primitivePath}.indices`,
        `[vitrum/gltf-adapter] Failed to read indices for mesh "${mesh.name ?? node.mesh}": ${String(e)}`,
      );
    }
  }

  if (indices !== undefined) {
    for (let i = 0; i < indices.length; i++) {
      const vertexIndex = indices[i]!;
      if (vertexIndex >= vertexCount) {
        throwImportBoundaryError(
          'invalid-primitive-attribute',
          `${primitivePath}.indices`,
          `[vitrum/gltf-adapter] Mesh "${mesh.name ?? node.mesh}" index accessor ${prim.indices} contains vertex index ${vertexIndex} at element ${i}, outside POSITION count ${vertexCount}.`,
        );
      }
    }
  }

  // TRIANGLE_STRIP / TRIANGLE_FAN → indexed triangle list (GLTF-05).
  // Works for indexed and non-indexed inputs; degenerate (repeated-index)
  // triangles are dropped per glTF §3.7.2.1 winding rules.
  if (mode === GLTF_MODE_TRIANGLE_STRIP || mode === GLTF_MODE_TRIANGLE_FAN) {
    const src =
      indices ??
      sequentialIndices(
        positions.length / 3,
        resourceLedger,
        `${primitivePath} sequential topology`,
      );
    const tris = triangulateTopology(
      src,
      mode,
      resourceLedger,
      `${primitivePath} triangulated topology`,
    );
    if (tris.length === 0) {
      emitImportDiagnostic(warnings, diagnostics, {
        severity: 'warning',
        code: 'empty-triangulated-primitive',
        path: `meshes[${node.mesh}].primitives[${primitiveIndex}]`,
        message:
          `[vitrum/gltf-adapter] Mesh "${mesh.name ?? node.mesh}" ` +
          `${mode === GLTF_MODE_TRIANGLE_STRIP ? 'TRIANGLE_STRIP' : 'TRIANGLE_FAN'} primitive ` +
          'yields no non-degenerate triangles. Primitive SKIPPED.',
      });
      return;
    }
    indices = tris;
  }

  if (mode === GLTF_PRIMITIVE_MODE_TRIANGLES && (indices?.length ?? vertexCount) % 3 !== 0) {
    throwImportBoundaryError(
      'invalid-primitive-attribute',
      primitivePath,
      `[vitrum/gltf-adapter] Mesh "${mesh.name ?? node.mesh}" TRIANGLES primitive has ${indices?.length ?? vertexCount} vertices/indices; the count must be divisible by 3.`,
    );
  }

  const usesPointLineFallback = isPointLineMode(mode);

  // Normals — generate flat normals if absent or unreadable.
  // Point/line fallback generates its own mesh normals below, so reporting
  // normal generation for the discarded source topology would be misleading.
  const normIdx = prim.attributes['NORMAL'];
  const normalAccessor = normIdx === undefined ? undefined : gltf.accessors?.[normIdx];
  const validNormalAccessor =
    normIdx === undefined ||
    (normalAccessor !== undefined &&
      normalAccessor.type === 'VEC3' &&
      normalAccessor.count === vertexCount &&
      _validNormalEncoding(normalAccessor, meshQuantization));
  if (!usesPointLineFallback && !validNormalAccessor) {
    throwImportBoundaryError(
      'invalid-primitive-attribute',
      `${primitivePath}.attributes.NORMAL`,
      `[vitrum/gltf-adapter] Mesh "${mesh.name ?? node.mesh}" NORMAL accessor ${normIdx} must be VEC3 with count ${vertexCount} and a valid FLOAT or KHR_mesh_quantization encoding.`,
    );
  }
  const normAttempt =
    usesPointLineFallback || !validNormalAccessor
      ? undefined
      : _tryUnpackFloat(
          gltf,
          buffers,
          normIdx,
          `NORMAL for mesh "${mesh.name ?? node.mesh}"`,
          warnings,
          onAccessorDiagnostic,
          diagnostics,
          'unreadable-normal',
          `${primitivePath}.attributes.NORMAL`,
          resourceLedger,
        );
  if (!usesPointLineFallback && normAttempt === undefined && normIdx === undefined) {
    emitImportDiagnostic(warnings, diagnostics, {
      severity: 'warning',
      code: 'generated-flat-normals',
      path: `${primitivePath}.attributes.NORMAL`,
      message:
        `[vitrum/gltf-adapter] Mesh "${mesh.name ?? node.mesh}" has no NORMAL attribute. ` +
        'Generating vertex normals (area-weighted for indexed geometry).',
    });
  }
  let normals: Float32Array = usesPointLineFallback
    ? new Float32Array(0)
    : (normAttempt ??
      generateVertexNormals(
        positions,
        indices,
        resourceLedger,
        `${primitivePath} generated normals`,
      ));

  // UVs — optional, but glTF TEXCOORD_N accessors must be VEC2.
  const uv0Idx = prim.attributes['TEXCOORD_0'];
  let uvs = _validatePrimitiveAttributeAccessor(
    gltf,
    uv0Idx,
    ['VEC2'],
    vertexCount,
    'TEXCOORD_0',
    `${mesh.name ?? node.mesh}`,
    `${primitivePath}.attributes.TEXCOORD_0`,
    warnings,
    diagnostics,
  )
    ? _tryUnpackFloat(
        gltf,
        buffers,
        uv0Idx,
        `TEXCOORD_0 for "${mesh.name ?? node.mesh}"`,
        warnings,
        onAccessorDiagnostic,
        diagnostics,
        'unreadable-optional-attribute',
        `${primitivePath}.attributes.TEXCOORD_0`,
        resourceLedger,
      )
    : undefined;
  const uv1Idx = prim.attributes['TEXCOORD_1'];
  let uv1 = _validatePrimitiveAttributeAccessor(
    gltf,
    uv1Idx,
    ['VEC2'],
    vertexCount,
    'TEXCOORD_1',
    `${mesh.name ?? node.mesh}`,
    `${primitivePath}.attributes.TEXCOORD_1`,
    warnings,
    diagnostics,
  )
    ? _tryUnpackFloat(
        gltf,
        buffers,
        uv1Idx,
        `TEXCOORD_1 for "${mesh.name ?? node.mesh}"`,
        warnings,
        onAccessorDiagnostic,
        diagnostics,
        'unreadable-optional-attribute',
        `${primitivePath}.attributes.TEXCOORD_1`,
        resourceLedger,
      )
    : undefined;
  let primitiveUvs = uvs;
  let primitiveUv1 = uv1;
  let uvSets: Array<Float32Array | undefined> = [];
  if (uvs !== undefined) uvSets[0] = uvs;
  if (uv1 !== undefined) uvSets[1] = uv1;
  for (const attrName of Object.keys(prim.attributes).sort()) {
    const texCoord = parseCanonicalSetSemantic(
      attrName,
      'TEXCOORD',
      `${primitivePath}.attributes.${attrName}`,
      'ignored-primitive-attribute',
    );
    if (texCoord === undefined) continue;
    if (texCoord < 2) continue;
    const accessorIndex = prim.attributes[attrName];
    uvSets[texCoord] = _validatePrimitiveAttributeAccessor(
      gltf,
      accessorIndex,
      ['VEC2'],
      vertexCount,
      attrName,
      `${mesh.name ?? node.mesh}`,
      `${primitivePath}.attributes.${attrName}`,
      warnings,
      diagnostics,
    )
      ? _tryUnpackFloat(
          gltf,
          buffers,
          accessorIndex,
          `${attrName} for "${mesh.name ?? node.mesh}"`,
          warnings,
          onAccessorDiagnostic,
          diagnostics,
          'unreadable-optional-attribute',
          `${primitivePath}.attributes.${attrName}`,
          resourceLedger,
        )
      : undefined;
  }
  let primitiveUvSets = uvSets;

  // Tangents — optional (xyzw per vertex).
  const tangentIdx = prim.attributes['TANGENT'];
  let tangents = _validatePrimitiveAttributeAccessor(
    gltf,
    tangentIdx,
    ['VEC4'],
    vertexCount,
    'TANGENT',
    `${mesh.name ?? node.mesh}`,
    `${primitivePath}.attributes.TANGENT`,
    warnings,
    diagnostics,
  )
    ? _tryUnpackFloat(
        gltf,
        buffers,
        tangentIdx,
        `TANGENT for "${mesh.name ?? node.mesh}"`,
        warnings,
        onAccessorDiagnostic,
        diagnostics,
        'unreadable-optional-attribute',
        `${primitivePath}.attributes.TANGENT`,
        resourceLedger,
      )
    : undefined;

  // Vertex colors — preserve every COLOR_n lane. COLOR_0 is also exposed
  // through the source-compatible `colors` alias in the core Scene contract.
  const color0Idx = prim.attributes['COLOR_0'];
  let colors = _validatePrimitiveAttributeAccessor(
    gltf,
    color0Idx,
    ['VEC3', 'VEC4'],
    vertexCount,
    'COLOR_0',
    `${mesh.name ?? node.mesh}`,
    `${primitivePath}.attributes.COLOR_0`,
    warnings,
    diagnostics,
  )
    ? _tryUnpackFloat(
        gltf,
        buffers,
        color0Idx,
        `COLOR_0 for "${mesh.name ?? node.mesh}"`,
        warnings,
        onAccessorDiagnostic,
        diagnostics,
        'unreadable-optional-attribute',
        `${primitivePath}.attributes.COLOR_0`,
        resourceLedger,
      )
    : undefined;
  let colorSets: Array<Float32Array | undefined> = [];
  if (colors !== undefined) colorSets[0] = colors;
  for (const attrName of Object.keys(prim.attributes).sort()) {
    const colorSet = parseCanonicalSetSemantic(
      attrName,
      'COLOR',
      `${primitivePath}.attributes.${attrName}`,
      'ignored-primitive-attribute',
    );
    if (colorSet === undefined) continue;
    if (colorSet < 1) continue;
    const accessorIndex = prim.attributes[attrName];
    colorSets[colorSet] = _validatePrimitiveAttributeAccessor(
      gltf,
      accessorIndex,
      ['VEC3', 'VEC4'],
      vertexCount,
      attrName,
      `${mesh.name ?? node.mesh}`,
      `${primitivePath}.attributes.${attrName}`,
      warnings,
      diagnostics,
    )
      ? _tryUnpackFloat(
          gltf,
          buffers,
          accessorIndex,
          `${attrName} for "${mesh.name ?? node.mesh}"`,
          warnings,
          onAccessorDiagnostic,
          diagnostics,
          'unreadable-optional-attribute',
          `${primitivePath}.attributes.${attrName}`,
          resourceLedger,
        )
      : undefined;
  }

  // ── Skinning attributes ────────────────────────────────────────────────
  // Only unpacked when this node has a skin. JOINTS_N / WEIGHTS_N without
  // node.skin carry no glTF skinning semantics, but report the ignored data
  // so strict one-call loading can reject the degradation before rendering.
  let skinIndices: Uint32Array | undefined;
  let skinWeights: Float32Array | undefined;
  let skinInfluencesPerVertex: number | undefined;
  const skinInfluenceSetIndices = collectSkinInfluenceSetIndices(
    prim.attributes,
    primitivePath,
  );
  const jointsIdx = prim.attributes['JOINTS_0'];
  const weightsIdx = prim.attributes['WEIGHTS_0'];
  if (!skinData && skinInfluenceSetIndices.length > 0) {
    const firstSet = skinInfluenceSetIndices[0] ?? 0;
    const firstJoints = prim.attributes[`JOINTS_${firstSet}`];
    emitImportDiagnostic(warnings, diagnostics, {
      severity: 'warning',
      code: 'ignored-skin-attributes',
      path:
        firstJoints !== undefined
          ? `${primitivePath}.attributes.JOINTS_${firstSet}`
          : `${primitivePath}.attributes.WEIGHTS_${firstSet}`,
      message:
        `[vitrum/gltf-adapter] Mesh "${mesh.name ?? node.mesh}" primitive includes ` +
        'JOINTS_N/WEIGHTS_N data, but the node does not bind a skin. ' +
        'Skin attributes are ignored and the primitive is imported as a static mesh.',
    });
  }
  if (skinData && bones && boneInverses) {
    if (jointsIdx !== undefined && weightsIdx !== undefined) {
      const decodedSets: SkinInfluenceSet[] = [];
      for (const setIndex of skinInfluenceSetIndices) {
        const setJointsIdx = prim.attributes[`JOINTS_${setIndex}`];
        const setWeightsIdx = prim.attributes[`WEIGHTS_${setIndex}`];
        if (setJointsIdx === undefined || setWeightsIdx === undefined) {
          throwImportBoundaryError(
            'incomplete-skin-attributes',
            `${primitivePath}.attributes.${setJointsIdx === undefined ? `JOINTS_${setIndex}` : `WEIGHTS_${setIndex}`}`,
            `[vitrum/gltf-adapter] Mesh "${mesh.name ?? node.mesh}" node binds a skin, but influence set ${setIndex} does not provide both JOINTS_${setIndex} and WEIGHTS_${setIndex}.`,
          );
        }

        let decodedJoints: Uint32Array;
        let decodedWeights: Float32Array;
        try {
          decodedJoints = _unpackJoints(
            gltf,
            buffers,
            setJointsIdx,
            onAccessorDiagnostic,
            `JOINTS_${setIndex}`,
            resourceLedger,
          );
        } catch (e) {
          rethrowResourceLimitError(e);
          const message =
            `[vitrum/gltf-adapter] Failed to read JOINTS_${setIndex} for ` +
            `"${mesh.name ?? node.mesh}": ${String(e)}`;
          throw new GltfImportError(
            message,
            diagnosticChainWithTerminal(diagnostics, {
              severity: 'error',
              code: 'unreadable-skin-joints',
              path: `${primitivePath}.attributes.JOINTS_${setIndex}`,
              message,
            }),
          );
        }
        try {
          decodedWeights = unpackAccessorFloat(
            gltf,
            buffers,
            setWeightsIdx,
            warnings,
            onAccessorDiagnostic,
            resourceLedger,
          );
        } catch (e) {
          rethrowResourceLimitError(e);
          throwImportBoundaryError(
            'unreadable-skin-weights',
            `${primitivePath}.attributes.WEIGHTS_${setIndex}`,
            `[vitrum/gltf-adapter] Failed to read WEIGHTS_${setIndex} for "${mesh.name ?? node.mesh}": ${String(e)}`,
          );
        }

        const expectedLength = Math.floor(positions.length / 3) * 4;
        if (decodedJoints.length !== expectedLength || decodedWeights.length !== expectedLength) {
          throwImportBoundaryError(
            'invalid-primitive-attribute',
            `${primitivePath}.attributes.JOINTS_${setIndex}`,
            `[vitrum/gltf-adapter] Mesh "${mesh.name ?? node.mesh}" influence set ${setIndex} has JOINTS_${setIndex}/WEIGHTS_${setIndex} length mismatch for ${expectedLength / 4} vertices.`,
          );
        }
        const jointCount = Math.floor(bones.length / 16);
        for (let lane = 0; lane < expectedLength; lane++) {
          const joint = decodedJoints[lane]!;
          const weight = decodedWeights[lane]!;
          if (!Number.isSafeInteger(joint) || joint < 0 || joint >= jointCount) {
            throwImportBoundaryError(
              'unreadable-skin-joints',
              `${primitivePath}.attributes.JOINTS_${setIndex}`,
              `[vitrum/gltf-adapter] JOINTS_${setIndex} lane ${lane} references joint ${joint}; skin has ${jointCount} joints.`,
            );
          }
          if (!Number.isFinite(weight) || weight < 0) {
            throwImportBoundaryError(
              'unreadable-skin-weights',
              `${primitivePath}.attributes.WEIGHTS_${setIndex}`,
              `[vitrum/gltf-adapter] WEIGHTS_${setIndex} lane ${lane} must be finite and non-negative; received ${String(weight)}.`,
            );
          }
        }
        decodedSets.push({ setIndex, joints: decodedJoints, weights: decodedWeights });
      }

      if (decodedSets.length > 0) {
        const vertexCount = Math.floor(positions.length / 3);
        for (let vertex = 0; vertex < vertexCount; vertex += 1) {
          const weightSum = mergedSkinInfluencesForVertex(decodedSets, vertex)
            .reduce((sum, [, weight]) => sum + weight, 0);
          if (!(weightSum > 0) || !Number.isFinite(weightSum)) {
            throwImportBoundaryError(
              'unreadable-skin-weights',
              `${primitivePath}.attributes.WEIGHTS_0`,
              `[vitrum/gltf-adapter] Mesh "${mesh.name ?? node.mesh}" skin weights for ` +
                `vertex ${vertex} have no positive finite influence.`,
            );
          }
        }

        const hasSecondarySets = decodedSets.some((set) => set.setIndex !== 0);
        if (hasSecondarySets) {
          const packed = packSkinInfluenceSets(
            decodedSets,
            vertexCount,
            resourceLedger,
            `${primitivePath} collapsed skin influences`,
          );
          skinIndices = packed.skinIndices;
          skinWeights = packed.skinWeights;
          skinInfluencesPerVertex = packed.skinInfluencesPerVertex;
        } else {
          const onlySet = decodedSets[0]!;
          for (let vertex = 0; vertex < vertexCount; vertex += 1) {
            const base = vertex * 4;
            const sum =
              onlySet.weights[base]! +
              onlySet.weights[base + 1]! +
              onlySet.weights[base + 2]! +
              onlySet.weights[base + 3]!;
            for (let lane = 0; lane < 4; lane += 1) {
              onlySet.weights[base + lane] = onlySet.weights[base + lane]! / sum;
            }
          }
          skinIndices = onlySet.joints;
          skinWeights = onlySet.weights;
        }
      }
    } else {
      throwImportBoundaryError(
        'incomplete-skin-attributes',
        `${primitivePath}.attributes.${jointsIdx === undefined ? 'JOINTS_0' : 'WEIGHTS_0'}`,
        `[vitrum/gltf-adapter] Mesh "${mesh.name ?? node.mesh}" node binds a skin, but the primitive does not provide both JOINTS_0 and WEIGHTS_0.`,
      );
    }
  }

  // Warn on unsupported primitive attributes.
  for (const attrName of Object.keys(prim.attributes)) {
    if (
      ![
        'POSITION',
        'NORMAL',
        'TEXCOORD_0',
        'TEXCOORD_1',
        'TANGENT',
        'COLOR_0',
        'JOINTS_0',
        'WEIGHTS_0',
      ].includes(attrName) &&
      !isSkinInfluenceAttribute(attrName) &&
      !/^TEXCOORD_\d+$/u.test(attrName) &&
      !/^COLOR_\d+$/u.test(attrName)
    ) {
      if (attrName.startsWith('_')) {
        emitImportDiagnostic(warnings, diagnostics, {
          severity: 'warning',
          code: 'ignored-primitive-attribute',
          path: `${primitivePath}.attributes.${attrName}`,
          message:
            `[vitrum/gltf-adapter] Application-specific primitive attribute "${attrName}" ` +
            `in mesh "${mesh.name ?? node.mesh}" is not consumed by the core Scene contract and was ignored.`,
        });
        continue;
      }
      throwImportBoundaryError(
        'ignored-primitive-attribute',
        `${primitivePath}.attributes.${attrName}`,
        `[vitrum/gltf-adapter] Unknown primitive attribute "${attrName}" in mesh "${mesh.name ?? node.mesh}" cannot be represented exactly.`,
      );
    }
  }

  // Material.
  const materialIndex = _resolvePrimitiveMaterialIndex(
    gltf,
    prim,
    prim.material,
    selectedMaterialVariant,
    warnings,
    diagnostics,
    `${mesh.name ?? node.mesh}`,
    primitivePath,
  );
  if (
    materialIndex !== undefined &&
    (!Number.isInteger(materialIndex) || materialIndex < 0 || materialIndex >= coreMaterials.length)
  ) {
    throwImportBoundaryError(
      'material-not-found',
      `${primitivePath}.material`,
      `[vitrum/gltf-adapter] Mesh "${mesh.name ?? node.mesh}" primitive references missing material ${String(materialIndex)}.`,
    );
  }
  const material =
    materialIndex !== undefined && materialIndex < coreMaterials.length
      ? (coreMaterials[materialIndex] ?? GLTF_DEFAULT_MATERIAL)
      : GLTF_DEFAULT_MATERIAL;
  const uvResolvedMaterial = _resolvePrimitiveUvMaterial(
    material,
    uvs,
    uv1,
    uvSets,
    warnings,
    diagnostics,
    primitivePath,
    mesh.name ?? meshIndex,
  );
  uvs = uvResolvedMaterial.uvs;
  uv1 = uvResolvedMaterial.uv1;
  uvSets = cloneSparseArray(uvResolvedMaterial.uvSets ?? []);

  // Morph targets (GLTF-04) — POSITION/NORMAL/TANGENT plus the
  // glTF UV semantics currently carried in core uvs/uv1 + node/mesh weights.
  let morph = _extractMorphTargets(
    gltf,
    buffers,
    prim.targets,
    node.weights ?? mesh.weights,
    positions.length / 3,
    uvSets,
    `${mesh.name ?? node.mesh}`,
    primitivePath,
    warnings,
    diagnostics,
    onAccessorDiagnostic,
    resourceLedger,
  );
  if (usesPointLineFallback) {
    const originalVertexCount = Math.floor(positions.length / 3);
    const fallback = buildPointLineFallbackGeometry(
      positions,
      indices,
      mode,
      opts.pointLineFallbackRadius,
      resourceLedger,
      `${primitivePath} point/line fallback`,
    );
    if (fallback == null) {
      emitImportDiagnostic(warnings, diagnostics, {
        severity: 'warning',
        code: 'empty-triangulated-primitive',
        path: `meshes[${node.mesh}].primitives[${primitiveIndex}]`,
        message:
          `[vitrum/gltf-adapter] Mesh "${mesh.name ?? node.mesh}" ` +
          `${pointLineModeName(mode)} primitive could not produce non-degenerate fallback mesh geometry. ` +
          'Primitive SKIPPED.',
      });
      return;
    }
    emitImportDiagnostic(warnings, diagnostics, {
      severity: 'warning',
      code: 'fallback-generated-primitive-mode',
      path: `meshes[${node.mesh}].primitives[${primitiveIndex}].mode`,
      message:
        `[vitrum/gltf-adapter] Mesh "${mesh.name ?? node.mesh}" primitive mode ${mode} ` +
        `(${pointLineModeName(mode)}) was imported as fallback-generated mesh geometry ` +
        `(radius ${fallback.radius}). Topology fidelity is approximate, but the primitive is renderable.`,
    });
    const remapPath = `${primitivePath} point/line remap`;
    uvs = remapVec2Attribute(uvs, fallback.sourceVertices, resourceLedger, `${remapPath}.uvs`);
    uv1 = remapVec2Attribute(uv1, fallback.sourceVertices, resourceLedger, `${remapPath}.uv1`);
    uvSets = mapSparseArray(uvSets, (stream, texCoord) =>
      remapVec2Attribute(
        stream,
        fallback.sourceVertices,
        resourceLedger,
        `${remapPath}.uvSets[${texCoord}]`,
      ),
    );
    primitiveUvs = remapVec2Attribute(
      primitiveUvs,
      fallback.sourceVertices,
      resourceLedger,
      `${remapPath}.primitiveUvs`,
    );
    primitiveUv1 = remapVec2Attribute(
      primitiveUv1,
      fallback.sourceVertices,
      resourceLedger,
      `${remapPath}.primitiveUv1`,
    );
    primitiveUvSets = mapSparseArray(primitiveUvSets, (stream, texCoord) =>
      remapVec2Attribute(
        stream,
        fallback.sourceVertices,
        resourceLedger,
        `${remapPath}.primitiveUvSets[${texCoord}]`,
      ),
    );
    colors = remapVertexColors(
      colors,
      originalVertexCount,
      fallback.sourceVertices,
      resourceLedger,
      `${remapPath}.colors`,
    );
    colorSets = mapSparseArray(colorSets, (stream, colorSet) =>
      remapVertexColors(
        stream,
        originalVertexCount,
        fallback.sourceVertices,
        resourceLedger,
        `${remapPath}.colorSets[${colorSet}]`,
      ),
    );
    tangents = undefined;
    if (skinIndices && skinWeights) {
      const influenceCount = skinInfluencesPerVertex ?? 4;
      skinIndices = remapPackedUintAttribute(
        skinIndices,
        influenceCount,
        fallback.sourceVertices,
        resourceLedger,
        `${remapPath}.skinIndices`,
      );
      skinWeights = remapPackedFloatAttribute(
        skinWeights,
        influenceCount,
        fallback.sourceVertices,
        resourceLedger,
        `${remapPath}.skinWeights`,
      );
    }
    morph = remapMorphData(morph, fallback.sourceVertices, resourceLedger, `${remapPath}.morph`);
    positions = fallback.positions;
    normals = fallback.normals;
    indices = fallback.indices;
  }
  const finalTangents =
    tangents ??
    _maybeGenerateTangents(
      positions,
      normals,
      uvs,
      uv1,
      uvSets,
      indices,
      uvResolvedMaterial.material,
      `${mesh.name ?? node.mesh}`,
      warnings,
      diagnostics,
      primitivePath,
      resourceLedger,
    );
  if (morph?.morphTargetTangents && !finalTangents) {
    throwImportBoundaryError(
      'missing-tangent-texcoord',
      `${primitivePath}.targets[].TANGENT`,
      `[vitrum/gltf-adapter] Mesh "${mesh.name ?? node.mesh}" authors TANGENT morph deltas but has no representable base tangent stream.`,
    );
  }

  const id = `gltf-prim-${ctx.primId.value++}`;
  const registerPrimitiveProvenance = (primitiveId: string): void => {
    (animationTargets[animationNodeId(nodeIdx)] ??= []).push(primitiveId);
    if (
      materialIndex !== undefined &&
      Number.isInteger(materialIndex) &&
      materialIndex >= 0 &&
      materialIndex < coreMaterials.length
    ) {
      materialBindings.push({ primitiveId, materialIndex });
    }
  };
  const registerPrimitiveVariants = (primitiveId: string): void => {
    if ((prim.extensions?.KHR_materials_variants?.mappings?.length ?? 0) === 0) return;
    materialVariantBindings.push({
      primitiveId,
      meshIndex,
      primitiveIndex,
      ...(prim.material !== undefined ? { baseMaterialIndex: prim.material } : {}),
      basePatch: _buildPrimitiveMaterialVariantPatch(
        coreMaterials,
        prim.material,
        positions,
        normals,
        primitiveUvs,
        primitiveUv1,
        primitiveUvSets,
        indices,
        tangents,
        `${mesh.name ?? node.mesh}`,
        warnings,
        diagnostics,
        primitivePath,
        resourceLedger,
      ),
      variantPatches: _buildPrimitiveMaterialVariantPatches(
        gltf,
        prim,
        coreMaterials,
        prim.material,
        positions,
        normals,
        primitiveUvs,
        primitiveUv1,
        primitiveUvSets,
        indices,
        tangents,
        `${mesh.name ?? node.mesh}`,
        warnings,
        diagnostics,
        primitivePath,
        resourceLedger,
      ),
    });
  };

  let skinArg =
    skinIndices && skinWeights && bones && boneInverses
      ? {
          skinIndices,
          skinWeights,
          ...(skinInfluencesPerVertex != null ? { skinInfluencesPerVertex } : {}),
          bones,
          boneInverses,
        }
      : undefined;

  // Morphed-but-unskinned mesh: core carries morph targets only on
  // SkinnedMeshPrimitive (solveSkin pre-blends morphs before LBS), so we
  // promote the mesh with a synthesized identity skin — one identity bone,
  // every vertex weighted [1,0,0,0]. The LBS pass is then a no-op and
  // solveSkin output equals restPos + Σ w_t · Δ_t in mesh-local space
  // (the primitive `transform` still applies on top, mirroring how
  // bindMatrix is only needed for world-space bone chains).
  if (morph && !skinArg) {
    const vertexCount = positions.length / 3;
    const influenceAllocation = geometryArrayAllocation(
      [vertexCount, 4],
      Float32Array.BYTES_PER_ELEMENT,
      `${primitivePath} identity skin influences`,
    );
    const identityMatrixBytes = 16 * Float32Array.BYTES_PER_ELEMENT;
    resourceLedger.chargeDecodedGeometryBytes(
      checkedGeometrySum(
        [
          influenceAllocation.byteLength,
          influenceAllocation.byteLength,
          identityMatrixBytes,
          identityMatrixBytes,
        ],
        `${primitivePath} identity skin byte length`,
      ),
      `${primitivePath} identity skin`,
    );
    const identitySkinWeights = new Float32Array(influenceAllocation.elementCount);
    for (let v = 0; v < vertexCount; v++) identitySkinWeights[v * 4] = 1;
    const identityBone = new Float32Array(16);
    identityBone[0] = 1;
    identityBone[5] = 1;
    identityBone[10] = 1;
    identityBone[15] = 1;
    skinArg = {
      skinIndices: new Uint32Array(influenceAllocation.elementCount), // all bone 0
      skinWeights: identitySkinWeights,
      bones: identityBone,
      boneInverses: new Float32Array(identityBone),
    };
  }

  const primitiveInstances = instanceTransforms?.worldInstanceTransforms;
  if (primitiveInstances && instanceTransforms) {
    registerPrimitiveProvenance(id);
    registerPrimitiveVariants(id);
    let deformationSource: SkinnedMeshPrimitive | undefined;
    let renderedPositions = positions;
    let renderedNormals = normals;
    let renderedUvs = uvs;
    let renderedUv1 = uv1;
    let renderedUvSets: ReadonlyArray<Float32Array | undefined> = uvSets;
    let renderedTangents = finalTangents;
    if (skinArg) {
      const source = _buildPrimitive(
        id,
        worldMat,
        positions,
        normals,
        indices,
        uvs,
        uv1,
        uvSets,
        finalTangents,
        colors,
        colorSets,
        uvResolvedMaterial.material,
        skinArg,
        morph,
        undefined,
      );
      if (source.kind !== 'skinned-mesh') {
        throw new Error('[vitrum/gltf-adapter] Internal error: deformation source is not skinned.');
      }
      deformationSource = source;
      chargeSolveSkinAllocations(resourceLedger, source, `${primitivePath} instanced skin solve`);
      const solved = solveSkin(source);
      renderedPositions = solved.positions;
      renderedNormals = solved.normals;
      renderedUvs = solved.uvs;
      renderedUv1 = solved.uv1;
      renderedUvSets = solved.uvSets ?? uvSets;
      renderedTangents = solved.tangents;
    }
    instancingBindings.push({
      primitiveId: id,
      nodeIndex: nodeIdx,
      localInstanceTransforms: instanceTransforms.localInstanceTransforms,
      ...(deformationSource ? { deformationSource } : {}),
    });
    primitives.push(
      _buildPrimitive(
        id,
        worldMat,
        renderedPositions,
        renderedNormals,
        indices,
        renderedUvs,
        renderedUv1,
        renderedUvSets,
        renderedTangents,
        colors,
        colorSets,
        uvResolvedMaterial.material,
        undefined,
        undefined,
        primitiveInstances,
      ),
    );
    return;
  }

  if (!primitiveInstances) {
    registerPrimitiveProvenance(id);
    registerPrimitiveVariants(id);
  }
  primitives.push(
    _buildPrimitive(
      id,
      worldMat,
      positions,
      normals,
      indices,
      uvs,
      uv1,
      uvSets,
      finalTangents,
      colors,
      colorSets,
      uvResolvedMaterial.material,
      skinArg,
      morph,
      primitiveInstances,
    ),
  );
}

function assertGltfBufferIndex(value: unknown, path: string): asserts value is number {
  if (!Number.isSafeInteger(value) || (value as number) < 0) {
    throw new TypeError(`[vitrum/gltf-adapter] ${path} must be a non-negative safe integer.`);
  }
}

function gltfBufferMapEntries(value: object): Iterable<unknown> | undefined {
  let entriesMethod: unknown;
  let getMethod: unknown;
  let iteratorMethod: unknown;
  try {
    const candidate = value as {
      readonly entries?: unknown;
      readonly get?: unknown;
      readonly [Symbol.iterator]?: unknown;
    };
    entriesMethod = candidate.entries;
    getMethod = candidate.get;
    iteratorMethod = candidate[Symbol.iterator];
  } catch (cause) {
    throw new TypeError(
      '[vitrum/gltf-adapter] options.buffers map methods could not be inspected.',
      { cause },
    );
  }
  if (
    typeof entriesMethod !== 'function' ||
    typeof getMethod !== 'function' ||
    typeof iteratorMethod !== 'function'
  ) {
    return undefined;
  }
  let entries: unknown;
  try {
    entries = Reflect.apply(entriesMethod, value, []);
  } catch (cause) {
    throw new TypeError('[vitrum/gltf-adapter] options.buffers.entries() failed.', { cause });
  }
  if (
    entries == null ||
    (typeof entries !== 'object' && typeof entries !== 'function') ||
    typeof (entries as { readonly [Symbol.iterator]?: unknown })[Symbol.iterator] !== 'function'
  ) {
    throw new TypeError('[vitrum/gltf-adapter] options.buffers.entries() must return an iterable.');
  }
  return entries as Iterable<unknown>;
}

function seedGltfBufferEntry(
  rawIndex: unknown,
  value: unknown,
  path: string,
  resourceLedger: ImportResourceLedger,
  output: Map<number, ArrayBuffer>,
): void {
  assertGltfBufferIndex(rawIndex, `${path} key`);
  const byteLength = gltfArrayBufferByteLength(value);
  if (byteLength === undefined) {
    throw new TypeError(
      `[vitrum/gltf-adapter] ${path}[${rawIndex}] must be a non-shared ArrayBuffer.`,
    );
  }
  resourceLedger.chargeEncodedBytes(
    gltfBufferResourceKey(rawIndex),
    byteLength,
    `${path}[${rawIndex}]`,
  );
  output.set(rawIndex, value as ArrayBuffer);
}

function seedGltfOptionBuffers(
  value: GltfToSceneOptions['buffers'] | undefined,
  resourceLedger: ImportResourceLedger,
  output: Map<number, ArrayBuffer>,
): void {
  if (value === undefined) return;
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(
      '[vitrum/gltf-adapter] options.buffers must be a map or numeric-keyed object.',
    );
  }
  const mapEntries = gltfBufferMapEntries(value);
  if (mapEntries !== undefined) {
    for (const entry of mapEntries) {
      if (!Array.isArray(entry) || entry.length < 2) {
        throw new TypeError(
          '[vitrum/gltf-adapter] options.buffers map entries must be [index, ArrayBuffer] pairs.',
        );
      }
      seedGltfBufferEntry(entry[0], entry[1], 'options.buffers', resourceLedger, output);
    }
    return;
  }

  for (const [key, buffer] of Object.entries(value)) {
    if (!/^(?:0|[1-9]\d*)$/.test(key)) {
      throw new TypeError(
        `[vitrum/gltf-adapter] options.buffers key "${key}" must be a canonical non-negative integer.`,
      );
    }
    const bufferIndex = Number(key);
    seedGltfBufferEntry(bufferIndex, buffer, 'options.buffers', resourceLedger, output);
  }
}

const SHARED_ARRAY_BUFFER_BYTE_LENGTH_DESCRIPTOR =
  typeof globalThis.SharedArrayBuffer === 'function'
    ? Object.getOwnPropertyDescriptor(globalThis.SharedArrayBuffer.prototype, 'byteLength')
    : undefined;
const SHARED_ARRAY_BUFFER_BYTE_LENGTH_GETTER =
  // eslint-disable-next-line @typescript-eslint/unbound-method -- Invoked only with an explicit candidate receiver.
  SHARED_ARRAY_BUFFER_BYTE_LENGTH_DESCRIPTOR?.get;

function isSharedArrayBufferValue(value: unknown): boolean {
  if (SHARED_ARRAY_BUFFER_BYTE_LENGTH_GETTER === undefined) return false;
  try {
    Reflect.apply(SHARED_ARRAY_BUFFER_BYTE_LENGTH_GETTER, value, []);
    return true;
  } catch {
    return false;
  }
}

export async function gltfToScene(
  input: ArrayBuffer | GltfJson,
  opts: GltfToSceneOptions = {},
): Promise<GltfToSceneResult> {
  const ledger = new ImportResourceLedger(opts.resourceLimits);
  const decodedImageHandles = new DecodedImageHandleOwner();
  const resourceContext: GltfImportResourceContext = {
    ledger,
    limiter: createAsyncResourceLimiter(ledger.limits.maxConcurrentResourceOperations),
    decodedImageHandles,
  };
  try {
    const result = await gltfToSceneWithResourceContext(input, opts, resourceContext);
    return attachGltfResourceOwner(result, decodedImageHandles);
  } catch (error) {
    decodedImageHandles.rollback();
    throw error;
  }
}

/**
 * Internal loader bridge. The caller owns the already-created context so
 * fetch, image decode, accessor decode, and generated geometry share one
 * monotonic budget without resetting between stages.
 *
 * @internal Not re-exported from the package index.
 */
export async function gltfToSceneWithResourceContext(
  input: ArrayBuffer | GltfJson,
  opts: GltfToSceneOptions,
  resourceContext: GltfImportResourceContext,
): Promise<GltfToSceneResult> {
  const { ledger: resourceLedger } = resourceContext;
  const warnings: string[] = [];

  // ── 1. Parse input ─────────────────────────────────────────────────────────
  let gltf: GltfJson;
  const buffers = new Map<number, ArrayBuffer>();

  // Seed from opts.buffers.
  seedGltfOptionBuffers(opts.buffers, resourceLedger, buffers);

  const inputByteLength = gltfArrayBufferByteLength(input);
  if (inputByteLength !== undefined) {
    const inputBuffer = input as ArrayBuffer;
    resourceLedger.chargeEncodedBytes(GLTF_INPUT_RESOURCE_KEY, inputByteLength, 'input');
    // Detect GLB by magic bytes.
    if (inputByteLength >= 4 && new DataView(inputBuffer).getUint32(0, true) === 0x46546c67) {
      const hasOverride = buffers.has(0);
      const binResourceKey = hasOverride
        ? `${gltfBufferResourceKey(0)}:unused-glb-copy`
        : gltfBufferResourceKey(0);
      const glb = parseGlb(inputBuffer, {
        beforeBinChunkCopy: ({ byteLength }) => {
          resourceLedger.ensureEncodedBytes(binResourceKey, byteLength, 'GLB BIN chunk');
        },
      });
      gltf = glb.json;
      if (glb.binChunk !== undefined) {
        const binByteLength = gltfArrayBufferByteLength(glb.binChunk);
        if (binByteLength === undefined) {
          throw new TypeError(
            '[vitrum/gltf-adapter] Parsed GLB BIN chunk is not a genuine ArrayBuffer.',
          );
        }
        resourceLedger.chargeEncodedBytes(binResourceKey, binByteLength, 'GLB BIN chunk');
        if (!hasOverride) buffers.set(0, glb.binChunk);
      }
    } else {
      // Treat as UTF-8 JSON.
      try {
        const text = decodeGltfUtf8(inputBuffer);
        gltf = JSON.parse(text) as GltfJson;
      } catch (cause) {
        throw new GltfParseFailed({
          format: 'gltf-json',
          reason: 'json-parse-failed',
          message: '[vitrum/gltf-adapter] glTF JSON input is not valid JSON.',
          cause,
        });
      }
    }
  } else {
    if (isSharedArrayBufferValue(input)) {
      throw new TypeError('[vitrum/gltf-adapter] input must not be a SharedArrayBuffer.');
    }
    if (ArrayBuffer.isView(input)) {
      throw new TypeError(
        '[vitrum/gltf-adapter] input must be an ArrayBuffer, not an ArrayBuffer view.',
      );
    }
    if (!isObject(input)) {
      throw new TypeError(
        '[vitrum/gltf-adapter] input must be a glTF JSON object or non-shared ArrayBuffer.',
      );
    }
    gltf = input as GltfJson;
  }

  assertSupportedGltfVersion(gltf);
  const sceneIndex = resolveValidatedSceneIndex(gltf, opts.sceneIndex);

  const diagnostics: GltfImportDiagnostic[] = [];
  const onAccessorDiagnostic = (diagnostic: GltfAccessorDiagnostic): void => {
    emitImportDiagnostic(warnings, diagnostics, diagnostic);
  };
  const onMaterialDiagnostic = (diagnostic: GltfMaterialDiagnostic): void => {
    emitImportDiagnostic(warnings, diagnostics, diagnostic);
  };
  const onCompressionDiagnostic = (diagnostic: GltfCompressionDiagnostic): void => {
    diagnostics.push(diagnostic);
  };
  const sceneReachability = collectGltfSceneReachability(
    gltf,
    sceneIndex,
    opts.textureSourceExtensions ?? [],
  );
  validateReachablePrimitiveSemantics(gltf, sceneReachability);
  const cameraMetadataIssues = validateGltfCameraMetadata(
    gltf,
    sceneReachability.cameraIndices,
  );
  if (cameraMetadataIssues.length > 0) {
    throw new GltfImportError(
      cameraMetadataIssues[0]!.message,
      cameraMetadataIssues.map((issue) => ({
        severity: 'error',
        code: 'invalid-camera',
        path: issue.path,
        message: issue.message,
      })),
    );
  }
  const scopedFeatureReport = analyzeGltfAsset(gltf, {
    ...(opts.textureSourceExtensions
      ? { textureSourceExtensions: opts.textureSourceExtensions }
      : {}),
    sceneIndex,
  });

  const requiredExtensions = scopedFeatureReport.extensions.required;
  for (let i = 0; i < requiredExtensions.length; i += 1) {
    const ext = requiredExtensions[i]!;
    if (!isRequiredExtensionSupported(ext, opts.textureSourceExtensions)) {
      const message =
        `[vitrum/gltf-adapter] extensionsRequired includes unsupported extension "${ext}". ` +
        'Required glTF extensions cannot be safely ignored.';
      throw new GltfImportError(message, [
        {
          severity: 'error',
          code: 'unsupported-required-extension',
          path:
            scopedFeatureReport.extensions.sourcePaths[ext]?.find((path) =>
              path.startsWith('extensionsRequired['),
            ) ?? `extensionsRequired[${i}]`,
          message,
        },
      ]);
    }
  }

  // ── 3. Warn on out-of-scope top-level features ─────────────────────────────
  if (sceneReachability.cameraIndices.size > 0) {
    for (const cameraIndex of [...sceneReachability.cameraIndices].sort((a, b) => a - b)) {
      emitImportDiagnostic(warnings, diagnostics, {
        severity: 'warning',
        code: 'ignored-camera',
        path: `cameras[${cameraIndex}]`,
        message:
          '[vitrum/gltf-adapter] Camera nodes are present and reported on result.cameras, but are not ' +
          'injected into the @vitrum/core Scene contract; pass camera data via FrameInput instead.',
      });
    }
  }
  if (sceneReachability.skinIndices.size > 0) {
    emitImportDiagnostic(warnings, diagnostics, {
      severity: 'warning',
      code: 'skin-rest-pose',
      path: [...sceneReachability.skinIndices]
        .sort((a, b) => a - b)
        .map((skinIndex) => `skins[${skinIndex}]`)
        .join(','),
      message:
        `[vitrum/gltf-adapter] Selected scene ${sceneIndex} uses ${sceneReachability.skinIndices.size} skin(s). ` +
        'Skinned nodes are imported as SkinnedMeshPrimitive at rest pose. ' +
        'The engine does not advance clips itself: drive the pose host-side by ' +
        'sampling the imported animations (sampleAnimationClip), rebuilding bone ' +
        'matrices, and re-running solveSkin.',
    });
  }

  const extUsed = gltf.extensionsUsed ?? [];

  // ── 3.5. Resolve compressed geometry (GLTF-02) ─────────────────────────────
  // KHR_draco_mesh_compression + EXT_meshopt_compression via host overrides or
  // the adapter's lazy built-in decoders. Runs BEFORE texture/accessor
  // reads so every downstream consumer sees decompressed bufferViews. Returns
  // a clone when compression is present (the caller's GltfJson is never
  // mutated); required decode failures and optional data without an exact
  // fallback fail closed before accessor conversion.
  const builtinCompressionDecoders =
    opts.compressionDecoderPolicy === 'host-only'
      ? undefined
      : createImportBuiltinCompressionDecoders(resourceLedger);
  const dracoDecode = opts.dracoDecode ?? builtinCompressionDecoders?.dracoDecode;
  const meshoptDecode = opts.meshoptDecode ?? builtinCompressionDecoders?.meshoptDecode;
  gltf = await resolveCompression(
    gltf,
    buffers,
    {
      dracoDecode,
      meshoptDecode,
    },
    warnings,
    onCompressionDiagnostic,
    {
      sceneReachability,
      bufferViewIndices: sceneReachability.bufferViewIndices,
      resourceLedger,
      hookOutputPrecharged: {
        draco:
          dracoDecode !== undefined &&
          dracoDecode === builtinCompressionDecoders?.dracoDecode,
        meshopt:
          meshoptDecode !== undefined &&
          meshoptDecode === builtinCompressionDecoders?.meshoptDecode,
      },
    },
  );

  // ── 4. Resolve textures ────────────────────────────────────────────────────
  const handleMap = await buildTextureHandleMap(
    gltf,
    buffers,
    opts.decodeImage,
    warnings,
    opts.imageBytes,
    opts.textureSourceExtensions,
    (diagnostic) => {
      diagnostics.push(diagnostic);
    },
    sceneReachability.textureIndices,
    resourceContext,
  );

  // ── 5. Pre-convert materials ───────────────────────────────────────────────
  const coreMaterials = (gltf.materials ?? []).map((m, materialIndex) => {
    if (!sceneReachability.materialIndices.has(materialIndex)) return GLTF_DEFAULT_MATERIAL;
    try {
      return convertMaterial(
        m,
        handleMap,
        warnings,
        gltf,
        materialIndex,
        opts.textureSourceExtensions,
        onMaterialDiagnostic,
      );
    } catch (error) {
      if (error instanceof GltfMaterialImportError) {
        // Preserve acquisition/selection diagnostics collected before the
        // terminal material-resolution failure.  Import is atomic, but callers
        // still need the complete structured cause chain to repair the asset.
        throw new GltfImportError(
          error.message,
          diagnosticChainWithTerminal(diagnostics, error.diagnostic),
        );
      }
      throw error;
    }
  });
  const selectedMaterialVariant = _resolveMaterialVariantSelection(
    gltf,
    opts.materialVariant,
    warnings,
    diagnostics,
  );

  // ── 6. Pick the target scene ───────────────────────────────────────────────
  const gltfScene = gltf.scenes?.[sceneIndex];
  const rootNodes = gltfScene?.nodes ?? [];

  // ── 7. Build world transforms for all nodes ────────────────────────────────
  const worldTransforms = buildWorldTransforms(
    gltf,
    rootNodes,
    resourceLedger,
    'scene node transforms',
  );

  const cameras = collectSceneCameras(gltf, worldTransforms, resourceLedger, 'scene cameras');

  // ── 8. Flatten node → mesh → primitives ───────────────────────────────────
  const primitives: ScenePrimitive[] = [];
  const animationTargets: Record<string, string[]> = {};
  const materialVariantBindings: GltfMaterialVariantBinding[] = [];
  const materialBindings: GltfMaterialBinding[] = [];
  const instancingBindings: GltfInstancingBinding[] = [];
  const primId = { value: 0 };

  const buildPrimitiveCtx: BuildPrimitiveContext = {
    gltf,
    buffers,
    warnings,
    diagnostics,
    onAccessorDiagnostic,
    coreMaterials,
    opts,
    selectedMaterialVariant,
    primitives,
    animationTargets,
    materialVariantBindings,
    materialBindings,
    instancingBindings,
    primId,
    resourceLedger,
  };

  const gltfNodes = gltf.nodes ?? [];
  const gltfMeshes = gltf.meshes ?? [];
  const gltfSkins = gltf.skins ?? [];

  for (const [nodeIdx, worldMat] of worldTransforms) {
    const node = gltfNodes[nodeIdx];
    if (!node || node.mesh === undefined) continue;
    const instanceTransforms = _extractMeshGpuInstancing(
      gltf,
      buffers,
      nodeIdx,
      node,
      worldMat,
      warnings,
      diagnostics,
      onAccessorDiagnostic,
      resourceLedger,
    );

    const mesh = gltfMeshes[node.mesh];
    if (!mesh) continue;

    // ── Skin data for this node (if any) ──────────────────────────────────
    // glTF 2.0 §3.8: a node may reference a skin by index. All primitives in
    // the node's mesh share the same skin.
    const skinData = _extractSkinData(
      gltf,
      buffers,
      gltfSkins,
      node.skin,
      worldMat,
      worldTransforms,
      warnings,
      diagnostics,
      onAccessorDiagnostic,
      resourceLedger,
    );
    const { bones, boneInverses } = skinData ?? {};

    const nodeCtx: BuildPrimitiveNodeContext = {
      node,
      nodeIdx,
      meshIndex: node.mesh,
      mesh,
      worldMat,
      skinData,
      bones,
      boneInverses,
      instanceTransforms,
    };
    for (const [primitiveIndex, prim] of mesh.primitives.entries()) {
      buildPrimitiveFromMeshPrimitive(buildPrimitiveCtx, nodeCtx, prim, primitiveIndex);
    }
  }

  // ── 9. Parse KHR_lights_punctual emitters ──────────────────────────────────
  // Reference: KHR_lights_punctual extension specification
  // https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_lights_punctual/README.md
  //
  // Intensity units convention adopted:
  //   - glTF punctual point/spot: candela (cd = lm/sr)
  //   - glTF punctual directional: lux (lx = lm/m²)
  //
  //   vitrum's EmitterBase.intensity is a dimensionless linear scalar (the
  //   backends multiply color × intensity for energy). Rather than introducing a
  //   unit conversion that would require knowing scene scale, we pass the
  //   photometric value directly as the intensity field and document the
  //   convention in the comment. Hosts that need SI-calibrated rendering should
  //   divide by their reference value (e.g., sunny-day ≈ 100,000 lx for the
  //   directional, typical tungsten bulb ≈ 50-100 cd for a point).
  //   This is consistent with how three.js PointLight/DirectionalLight expose
  //   intensity post–three.js r155 (they pass glTF values straight through).
  const emitters: SceneEmitter[] = [];
  const punctualEmitterBindings: GltfPunctualEmitterBinding[] = [];
  let emitterIdCounter = 0;

  if (extUsed.includes('KHR_lights_punctual')) {
    const rawRootExt = gltf.extensions?.['KHR_lights_punctual'];
    if (rawRootExt !== undefined && (!isObject(rawRootExt) || !Array.isArray(rawRootExt.lights))) {
      throwImportBoundaryError(
        'missing-punctual-light',
        'extensions.KHR_lights_punctual',
        '[vitrum/gltf-adapter] KHR_lights_punctual root extension must contain a lights array.',
      );
    }
    const rootExt = rawRootExt as KhrLightsPunctualRoot | undefined;
    const lights = rootExt?.lights ?? [];

    for (const [nodeIdx, worldMat] of worldTransforms) {
      const node = gltfNodes[nodeIdx];
      if (!node?.extensions) continue;

      const rawNodeLightRef = node.extensions['KHR_lights_punctual'];
      if (rawNodeLightRef === undefined) continue;
      if (
        !isObject(rawNodeLightRef) ||
        !Number.isSafeInteger(rawNodeLightRef.light) ||
        (rawNodeLightRef.light as number) < 0
      ) {
        throwImportBoundaryError(
          'missing-punctual-light',
          `nodes[${nodeIdx}].extensions.KHR_lights_punctual.light`,
          `[vitrum/gltf-adapter] Node ${nodeIdx} KHR_lights_punctual light must be a non-negative safe integer.`,
        );
      }
      const nodeLightRef = rawNodeLightRef as { light: number };

      const light = lights[nodeLightRef.light];
      if (!light) {
        throwImportBoundaryError(
          'missing-punctual-light',
          `nodes[${nodeIdx}].extensions.KHR_lights_punctual.light`,
          `[vitrum/gltf-adapter] Node ${nodeIdx} references missing KHR_lights_punctual light ${nodeLightRef.light}.`,
        );
      }

      const id = `gltf-light-${emitterIdCounter++}`;
      const emitter = _convertPunctualLight(
        light,
        worldMat,
        id,
        warnings,
        diagnostics,
        `extensions.KHR_lights_punctual.lights[${nodeLightRef.light}]`,
      );
      emitters.push(emitter);
      punctualEmitterBindings.push({
        emitterId: id,
        nodeIndex: nodeIdx,
        lightIndex: nodeLightRef.light,
      });
    }
  }

  // ── 10. Convert animations (GLTF-03) ───────────────────────────────────────
  let animations: AnimationClip[];
  try {
    animations = convertAnimations(
      gltf,
      buffers,
      warnings,
      (diagnostic) => {
        diagnostics.push(diagnostic);
      },
      onAccessorDiagnostic,
      {
        reachableNodeIndices: sceneReachability.nodeIndices,
        reachableMaterialIndices: sceneReachability.materialIndices,
        reachableCameraIndices: sceneReachability.cameraIndices,
        reachablePunctualLightIndices: sceneReachability.punctualLightIndices,
        resourceLedger,
      },
    );
  } catch (error) {
    if (error instanceof GltfAnimationImportError) {
      throw new GltfImportError(
        error.message,
        diagnosticChainWithTerminal(diagnostics, error.diagnostic),
      );
    }
    throw error;
  }

  // KHR_node_visibility is inherited down the node hierarchy. Hidden visual
  // objects are omitted from the core Scene, while cameras intentionally remain
  // in metadata per the extension specification. The complete object arrays are
  // returned separately for controller-driven visibility animation.
  const effectiveNodeVisibility = buildEffectiveNodeVisibility(gltf, rootNodes);
  const visiblePrimitiveIds = new Set<string>();
  for (const [nodeIndex, visible] of effectiveNodeVisibility) {
    if (!visible) continue;
    for (const primitiveId of animationTargets[animationNodeId(nodeIndex)] ?? []) {
      visiblePrimitiveIds.add(primitiveId);
    }
  }
  const visibleEmitterIds = new Set(
    punctualEmitterBindings
      .filter((binding) => effectiveNodeVisibility.get(binding.nodeIndex) !== false)
      .map((binding) => binding.emitterId),
  );
  const visiblePrimitives = primitives.filter((primitive) =>
    visiblePrimitiveIds.has(String(primitive.id)),
  );
  const visibleEmitters = emitters.filter((emitter) => visibleEmitterIds.has(String(emitter.id)));

  const scene: Scene = {
    primitives: visiblePrimitives,
    emitters: visibleEmitters,
    environment: { kind: 'none' },
  };
  validateCoreScene(scene);

  return {
    scene,
    cameras,
    animations,
    animationTargets,
    convertedMaterials: coreMaterials,
    materialVariantBindings,
    materialBindings,
    instancingBindings,
    punctualEmitterBindings,
    nodeVisibilityPrimitives: primitives,
    nodeVisibilityEmitters: emitters,
    warnings,
    diagnostics,
  };
}

function buildEffectiveNodeVisibility(
  gltf: GltfJson,
  rootNodeIndices: readonly number[],
): Map<number, boolean> {
  const result = new Map<number, boolean>();
  const nodes = gltf.nodes ?? [];
  const stack = rootNodeIndices.map((nodeIndex) => ({ nodeIndex, parentVisible: true }));
  while (stack.length > 0) {
    const { nodeIndex, parentVisible } = stack.pop()!;
    const node = nodes[nodeIndex];
    if (!node || result.has(nodeIndex)) continue;
    const ownVisible = node.extensions?.KHR_node_visibility?.visible !== false;
    const visible = parentVisible && ownVisible;
    result.set(nodeIndex, visible);
    for (const child of node.children ?? []) {
      stack.push({ nodeIndex: child, parentVisible: visible });
    }
  }
  return result;
}

// ── Private helpers ──────────────────────────────────────────────────────────

function isRequiredExtensionSupported(
  ext: string,
  textureSourceExtensions: readonly GltfTextureSourceExtension[] | undefined,
): boolean {
  if (SUPPORTED_REQUIRED_EXTENSIONS.has(ext)) return true;
  if (!GLTF_TEXTURE_SOURCE_EXTENSIONS.includes(ext as GltfTextureSourceExtension)) return false;
  return textureSourceExtensions?.includes(ext as GltfTextureSourceExtension) ?? false;
}

function _resolveMaterialVariantSelection(
  gltf: GltfJson,
  selector: string | number | undefined,
  warnings: string[],
  diagnostics: GltfImportDiagnostic[],
): number | undefined {
  if (selector === undefined) return undefined;
  const variantList = gltf.extensions?.KHR_materials_variants?.variants;
  if (variantList !== undefined && !Array.isArray(variantList)) {
    emitImportDiagnostic(warnings, diagnostics, {
      severity: 'warning',
      code: 'material-variant-list-malformed',
      path: 'extensions.KHR_materials_variants.variants',
      message:
        '[vitrum/gltf-adapter] materialVariant was requested, but ' +
        'extensions.KHR_materials_variants.variants is missing or malformed. Base materials are used.',
    });
    return undefined;
  }
  const variants = variantList ?? [];
  if (typeof selector === 'number') {
    if (Number.isInteger(selector) && selector >= 0 && selector < variants.length) return selector;
    emitImportDiagnostic(warnings, diagnostics, {
      severity: 'warning',
      code: 'material-variant-not-found',
      path: 'extensions.KHR_materials_variants.variants',
      message:
        `[vitrum/gltf-adapter] materialVariant index ${selector} was requested, ` +
        `but this asset declares ${variants.length} variant(s). Base materials are used.`,
    });
    return undefined;
  }
  const index = variants.findIndex((variant) => variant.name === selector);
  if (index >= 0) return index;
  emitImportDiagnostic(warnings, diagnostics, {
    severity: 'warning',
    code: 'material-variant-not-found',
    path: 'extensions.KHR_materials_variants.variants',
    message:
      `[vitrum/gltf-adapter] materialVariant "${selector}" was requested, but no ` +
      'KHR_materials_variants entry with that name exists. Base materials are used.',
  });
  return undefined;
}

function _resolvePrimitiveMaterialIndex(
  gltf: GltfJson,
  primitive: GltfPrimitive,
  baseMaterialIndex: number | undefined,
  selectedVariantIndex: number | undefined,
  warnings: string[],
  diagnostics: GltfImportDiagnostic[],
  meshLabel: string,
  primitivePath: string,
): number | undefined {
  if (selectedVariantIndex === undefined) return baseMaterialIndex;
  const mappings = primitive.extensions?.KHR_materials_variants?.mappings ?? [];
  let matchedMapping:
    | { readonly mapping: (typeof mappings)[number]; readonly index: number }
    | undefined;
  for (const [mappingIndex, candidate] of mappings.entries()) {
    if (Array.isArray(candidate.variants) && candidate.variants.includes(selectedVariantIndex)) {
      matchedMapping = { mapping: candidate, index: mappingIndex };
      break;
    }
  }
  if (!matchedMapping) return baseMaterialIndex;
  const { mapping, index: mappingIndex } = matchedMapping;
  if (mapping.material < 0 || mapping.material >= (gltf.materials?.length ?? 0)) {
    emitImportDiagnostic(warnings, diagnostics, {
      severity: 'warning',
      code: 'material-variant-material-missing',
      path: `${primitivePath}.extensions.KHR_materials_variants.mappings[${mappingIndex}].material`,
      message:
        `[vitrum/gltf-adapter] Mesh "${meshLabel}" KHR_materials_variants mapping for ` +
        `variant ${selectedVariantIndex} references missing material ${mapping.material}. ` +
        'Base material is used.',
    });
    return baseMaterialIndex;
  }
  return mapping.material;
}

function _coreMaterialForIndex(
  coreMaterials: readonly MaterialSpec[],
  materialIndex: number | undefined,
): MaterialSpec {
  return materialIndex !== undefined && materialIndex < coreMaterials.length
    ? (coreMaterials[materialIndex] ?? GLTF_DEFAULT_MATERIAL)
    : GLTF_DEFAULT_MATERIAL;
}

function _buildPrimitiveMaterialVariantPatch(
  coreMaterials: readonly MaterialSpec[],
  materialIndex: number | undefined,
  positions: Float32Array,
  normals: Float32Array,
  uvs: Float32Array | undefined,
  primitiveUv1: Float32Array | undefined,
  primitiveUvSets: ReadonlyArray<Float32Array | undefined>,
  indices: Uint32Array | undefined,
  authoredTangents: Float32Array | undefined,
  meshLabel: string,
  warnings: string[],
  diagnostics: GltfImportDiagnostic[],
  primitivePath: string,
  resourceLedger: ImportResourceLedger,
): GltfMaterialVariantPrimitivePatch {
  const material = _coreMaterialForIndex(coreMaterials, materialIndex);
  const uvResolved = _resolvePrimitiveUvMaterial(
    material,
    uvs,
    primitiveUv1,
    primitiveUvSets,
    warnings,
    diagnostics,
    primitivePath,
    meshLabel,
  );
  const finalTangents =
    authoredTangents ??
    _maybeGenerateTangents(
      positions,
      normals,
      uvResolved.uvs,
      uvResolved.uv1,
      uvResolved.uvSets,
      indices,
      uvResolved.material,
      meshLabel,
      warnings,
      diagnostics,
      primitivePath,
      resourceLedger,
    );
  return {
    ...(materialIndex !== undefined ? { materialIndex } : {}),
    materialRouting: uvResolved.material,
    uvs: uvResolved.uvs,
    ...(uvResolved.droppedTextureFields != null && uvResolved.droppedTextureFields.length > 0
      ? { droppedTextureFields: uvResolved.droppedTextureFields }
      : {}),
    uv1: uvResolved.uv1,
    uvSets: uvResolved.uvSets,
    tangents: finalTangents,
  };
}

function _buildPrimitiveMaterialVariantPatches(
  gltf: GltfJson,
  primitive: GltfPrimitive,
  coreMaterials: readonly MaterialSpec[],
  baseMaterialIndex: number | undefined,
  positions: Float32Array,
  normals: Float32Array,
  uvs: Float32Array | undefined,
  primitiveUv1: Float32Array | undefined,
  primitiveUvSets: ReadonlyArray<Float32Array | undefined>,
  indices: Uint32Array | undefined,
  authoredTangents: Float32Array | undefined,
  meshLabel: string,
  warnings: string[],
  diagnostics: GltfImportDiagnostic[],
  primitivePath: string,
  resourceLedger: ImportResourceLedger,
): GltfMaterialVariantPatch[] {
  const patches = new Map<number, GltfMaterialVariantPatch>();
  const mappings = primitive.extensions?.KHR_materials_variants?.mappings ?? [];
  for (const [mappingIndex, mapping] of mappings.entries()) {
    if (!Array.isArray(mapping.variants)) {
      emitImportDiagnostic(warnings, diagnostics, {
        severity: 'warning',
        code: 'material-variant-mapping-malformed',
        path: `${primitivePath}.extensions.KHR_materials_variants.mappings[${mappingIndex}].variants`,
        message:
          `[vitrum/gltf-adapter] Mesh "${meshLabel}" KHR_materials_variants mapping ` +
          `${mappingIndex} has a missing or malformed variants array. Mapping skipped.`,
      });
      continue;
    }
    for (const variantIndex of mapping.variants) {
      if (patches.has(variantIndex)) continue;
      const materialIndex = _resolvePrimitiveMaterialIndex(
        gltf,
        primitive,
        baseMaterialIndex,
        variantIndex,
        warnings,
        diagnostics,
        meshLabel,
        primitivePath,
      );
      patches.set(variantIndex, {
        variantIndex,
        patch: _buildPrimitiveMaterialVariantPatch(
          coreMaterials,
          materialIndex,
          positions,
          normals,
          uvs,
          primitiveUv1,
          primitiveUvSets,
          indices,
          authoredTangents,
          meshLabel,
          warnings,
          diagnostics,
          primitivePath,
          resourceLedger,
        ),
      });
    }
  }
  return [...patches.values()];
}

function _maybeGenerateTangents(
  positions: Float32Array,
  normals: Float32Array,
  uvs: Float32Array | undefined,
  uv1: Float32Array | undefined,
  uvSets: ReadonlyArray<Float32Array | undefined> | undefined,
  indices: Uint32Array | undefined,
  material: MaterialSpec,
  meshLabel: string,
  warnings: string[],
  diagnostics: GltfImportDiagnostic[],
  primitivePath: string,
  resourceLedger: ImportResourceLedger,
): Float32Array | undefined {
  if (!materialNeedsTangentFrame(material)) return undefined;
  const tangentUvSet = tangentFrameTexCoord(material);
  if (tangentUvSet == null) {
    // A single authored/generated tangent stream cannot represent multiple UV
    // parameterisations. All shipping backends derive each selected texture
    // lane's exact cotangent frame from its own UV derivatives, so omission is
    // the lossless representation and is not a degradation diagnostic.
    return undefined;
  }
  const tangentUvs =
    uvSets?.[tangentUvSet] ?? (tangentUvSet === 1 ? uv1 : tangentUvSet === 0 ? uvs : undefined);
  if (!tangentUvs) {
    throwImportBoundaryError(
      'missing-material-texcoord',
      `${primitivePath}.attributes.TEXCOORD_${tangentUvSet}`,
      `[vitrum/gltf-adapter] Mesh "${meshLabel}" references missing TEXCOORD_${tangentUvSet} for a tangent-space material map.`,
    );
  }
  const generated = generateTangents(
    positions,
    normals,
    tangentUvs,
    indices,
    resourceLedger,
    `${primitivePath} generated tangents`,
  );
  if (!generated) {
    emitImportDiagnostic(warnings, diagnostics, {
      severity: 'warning',
      code: 'tangent-generation-failed',
      path: `${primitivePath}.attributes.TANGENT`,
      message:
        `[vitrum/gltf-adapter] Mesh "${meshLabel}" has a degenerate TEXCOORD_${tangentUvSet} ` +
        'parameterisation, so a stable per-vertex tangent stream cannot be generated.',
    });
    return undefined;
  }
  emitImportDiagnostic(warnings, diagnostics, {
    severity: 'warning',
    code: 'generated-tangents',
    path: `${primitivePath}.attributes.TANGENT`,
    message:
      `[vitrum/gltf-adapter] Mesh "${meshLabel}" uses a tangent-space material map without ` +
      `TANGENT; generated per-vertex tangents from POSITION/NORMAL/TEXCOORD_${tangentUvSet}.`,
  });
  return generated;
}

function materialNeedsTangentFrame(material: MaterialSpec): boolean {
  return (
    material.normalMap !== undefined ||
    material.clearcoatNormalMap !== undefined ||
    material.bumpMap !== undefined ||
    material.anisotropyMap !== undefined
  );
}

function tangentFrameTexCoord(material: MaterialSpec): number | null {
  const candidates = [
    material.normalMap?.texCoord,
    material.clearcoatNormalMap?.texCoord,
    material.bumpMap?.texCoord,
    material.anisotropyMap?.texCoord,
  ].filter((texCoord): texCoord is number => texCoord !== undefined);
  if (candidates.length === 0) return 0;
  const channels = new Set(candidates);
  if (channels.size > 1) return null;
  return channels.values().next().value ?? 0;
}

const GPU_INSTANCE_ATTRIBUTE_SPECS = {
  TRANSLATION: { type: 'VEC3' },
  ROTATION: { type: 'VEC4' },
  SCALE: { type: 'VEC3' },
} as const;

interface MeshGpuInstancingTransforms {
  readonly worldInstanceTransforms: readonly Mat4[];
  readonly localInstanceTransforms: readonly Mat4[];
}

function _extractMeshGpuInstancing(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
  nodeIdx: number,
  node: GltfNode,
  worldMat: Mat4,
  warnings: string[],
  diagnostics: GltfImportDiagnostic[],
  onAccessorDiagnostic: (diagnostic: GltfAccessorDiagnostic) => void,
  resourceLedger: ImportResourceLedger,
): MeshGpuInstancingTransforms | undefined {
  const extension = node.extensions?.EXT_mesh_gpu_instancing;
  if (extension === undefined) return undefined;
  const pathBase = `nodes[${nodeIdx}].extensions.EXT_mesh_gpu_instancing`;
  const attributes = isObject(extension) ? extension.attributes : undefined;
  if (!isObject(extension) || !isObject(attributes)) {
    throwImportBoundaryError(
      'ignored-gpu-instancing',
      pathBase,
      `[vitrum/gltf-adapter] Node "${node.name ?? nodeIdx}" uses EXT_mesh_gpu_instancing without an attributes object.`,
    );
  }

  const attrs = attributes;
  let instanceCount: number | undefined;
  let rotationNormalizationTolerance = 1e-4;

  const fail = (path: string, message: string): never => {
    throwImportBoundaryError('ignored-gpu-instancing', path, message);
  };

  const readAccessor = (
    semantic: keyof typeof GPU_INSTANCE_ATTRIBUTE_SPECS,
  ): Float32Array | undefined => {
    const rawAccessorIndex = attrs[semantic];
    if (rawAccessorIndex === undefined) return undefined;
    const attrPath = `${pathBase}.attributes.${semantic}`;
    if (
      typeof rawAccessorIndex !== 'number' ||
      !Number.isInteger(rawAccessorIndex) ||
      rawAccessorIndex < 0
    ) {
      return fail(
        attrPath,
        `[vitrum/gltf-adapter] Node "${node.name ?? nodeIdx}" EXT_mesh_gpu_instancing ` +
          `${semantic} must reference a non-negative accessor index.`,
      );
    }
    const accessorIndex = rawAccessorIndex;
    const accessor = gltf.accessors?.[accessorIndex];
    const spec = GPU_INSTANCE_ATTRIBUTE_SPECS[semantic];
    if (!accessor) {
      return fail(
        attrPath,
        `[vitrum/gltf-adapter] Node "${node.name ?? nodeIdx}" EXT_mesh_gpu_instancing ` +
          `${semantic} references missing accessor ${accessorIndex}.`,
      );
    }
    const isFloatTransform =
      accessor.componentType === GltfComponentType.FLOAT &&
      accessor.normalized !== true;
    const isQuantizedRotation =
      semantic === 'ROTATION' &&
      (accessor.componentType === GltfComponentType.BYTE ||
        accessor.componentType === GltfComponentType.SHORT) &&
      accessor.normalized === true;
    if (
      accessor.type !== spec.type ||
      (!isFloatTransform && !isQuantizedRotation) ||
      !Number.isSafeInteger(accessor.count) ||
      accessor.count <= 0
    ) {
      const format =
        semantic === 'ROTATION'
          ? 'FLOAT, normalized BYTE, or normalized SHORT VEC4'
          : `non-normalized FLOAT ${spec.type}`;
      return fail(
        attrPath,
        `[vitrum/gltf-adapter] Node "${node.name ?? nodeIdx}" EXT_mesh_gpu_instancing ` +
          `${semantic} accessor must be a non-empty ${format} accessor.`,
      );
    }
    if (semantic === 'ROTATION') {
      rotationNormalizationTolerance =
        accessor.componentType === GltfComponentType.BYTE
          ? 2 / 127
          : accessor.componentType === GltfComponentType.SHORT
            ? 2 / 32767
            : 1e-4;
    }
    if (instanceCount === undefined) {
      instanceCount = accessor.count;
    } else if (accessor.count !== instanceCount) {
      return fail(
        attrPath,
        `[vitrum/gltf-adapter] Node "${node.name ?? nodeIdx}" EXT_mesh_gpu_instancing ` +
          `${semantic} count ${accessor.count} does not match instance count ${instanceCount}. ` +
          'All transform attributes must have identical counts.',
      );
    }
    try {
      const decoded = unpackAccessorFloat(
        gltf,
        buffers,
        accessorIndex,
        warnings,
        onAccessorDiagnostic,
        resourceLedger,
      );
      const components = semantic === 'ROTATION' ? 4 : 3;
      if (decoded.length !== accessor.count * components || !decoded.every(Number.isFinite)) {
        return fail(
          attrPath,
          `[vitrum/gltf-adapter] EXT_mesh_gpu_instancing ${semantic} decoded to an invalid value count or non-finite values.`,
        );
      }
      return decoded;
    } catch (e) {
      rethrowResourceLimitError(e);
      return fail(
        attrPath,
        `[vitrum/gltf-adapter] Failed to read EXT_mesh_gpu_instancing ${semantic} for ` +
          `node "${node.name ?? nodeIdx}": ${String(e)}`,
      );
    }
  };

  const translations = readAccessor('TRANSLATION');
  const rotations = readAccessor('ROTATION');
  const scales = readAccessor('SCALE');

  for (const key of Object.keys(attrs)) {
    if (key in GPU_INSTANCE_ATTRIBUTE_SPECS) continue;
    throwImportBoundaryError(
      'ignored-gpu-instancing-attribute',
      `${pathBase}.attributes.${key}`,
      `[vitrum/gltf-adapter] Node "${node.name ?? nodeIdx}" EXT_mesh_gpu_instancing attribute "${key}" cannot be represented exactly.`,
    );
  }

  if (instanceCount === undefined || instanceCount <= 0) {
    throwImportBoundaryError(
      'ignored-gpu-instancing',
      `${pathBase}.attributes`,
      `[vitrum/gltf-adapter] Node "${node.name ?? nodeIdx}" uses EXT_mesh_gpu_instancing without TRANSLATION, ROTATION, or SCALE accessors.`,
    );
  }

  const worldInstanceTransforms: Mat4[] = [];
  const localInstanceTransforms: Mat4[] = [];
  for (let i = 0; i < instanceCount; i += 1) {
    const t: [number, number, number] = translations
      ? [translations[i * 3 + 0] ?? 0, translations[i * 3 + 1] ?? 0, translations[i * 3 + 2] ?? 0]
      : [0, 0, 0];
    const r: [number, number, number, number] = rotations
      ? [
          rotations[i * 4 + 0] ?? 0,
          rotations[i * 4 + 1] ?? 0,
          rotations[i * 4 + 2] ?? 0,
          rotations[i * 4 + 3] ?? 1,
        ]
      : [0, 0, 0, 1];
    if (rotations) {
      const length = Math.hypot(r[0], r[1], r[2], r[3]);
      if (
        !Number.isFinite(length) ||
        length <= 1e-10 ||
        Math.abs(length - 1) > rotationNormalizationTolerance
      ) {
        throwImportBoundaryError(
          'ignored-gpu-instancing',
          `${pathBase}.attributes.ROTATION`,
          `[vitrum/gltf-adapter] EXT_mesh_gpu_instancing ROTATION instance ${i} must be a normalized quaternion.`,
        );
      }
    }
    const s: [number, number, number] = scales
      ? [scales[i * 3 + 0] ?? 1, scales[i * 3 + 1] ?? 1, scales[i * 3 + 2] ?? 1]
      : [1, 1, 1];
    const local = asMat4(
      composeTrsMat4(t, r, s, resourceLedger, `${pathBase}.instances[${i}].localMatrix`),
    );
    localInstanceTransforms.push(local);
    worldInstanceTransforms.push(
      asMat4(mat4Mul(worldMat, local, resourceLedger, `${pathBase}.instances[${i}].worldMatrix`)),
    );
  }
  return { worldInstanceTransforms, localInstanceTransforms };
}

function isObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/** Read an authored optional accessor. Absence is optional; malformed data is not. */
function _tryUnpackFloat(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
  accessorIndex: number | undefined,
  label: string,
  warnings: string[],
  onAccessorDiagnostic: (diagnostic: GltfAccessorDiagnostic) => void,
  diagnostics?: GltfImportDiagnostic[],
  diagnosticCode: GltfImportDiagnosticCode = 'unreadable-optional-attribute',
  diagnosticPath?: string,
  resourceLedger?: ImportResourceLedger,
): Float32Array | undefined {
  if (accessorIndex === undefined) return undefined;
  try {
    const decoded = unpackAccessorFloat(
      gltf,
      buffers,
      accessorIndex,
      warnings,
      onAccessorDiagnostic,
      resourceLedger,
    );
    if (!decoded.every(Number.isFinite)) {
      throw new Error(`accessor ${accessorIndex} contains non-finite values`);
    }
    return decoded;
  } catch (e) {
    rethrowResourceLimitError(e);
    const message = `[vitrum/gltf-adapter] Failed to read ${label}: ${String(e)}`;
    void diagnostics;
    throwImportBoundaryError(
      diagnosticCode,
      diagnosticPath ?? `accessors[${accessorIndex}]`,
      message,
    );
  }
}

function _usesMeshQuantization(gltf: GltfJson): boolean {
  return (
    gltf.extensionsUsed?.includes('KHR_mesh_quantization') === true ||
    gltf.extensionsRequired?.includes('KHR_mesh_quantization') === true
  );
}

function _validPositionEncoding(accessor: GltfAccessor, meshQuantization: boolean): boolean {
  if (accessor.componentType === GltfComponentType.FLOAT) {
    return accessor.normalized !== true;
  }
  if (!meshQuantization) return false;
  // KHR_mesh_quantization permits both normalized and unnormalized signed or
  // unsigned 8/16-bit POSITION storage; authored node/skin transforms carry
  // any required dequantization scale and offset.
  return (
    accessor.componentType === GltfComponentType.BYTE ||
    accessor.componentType === GltfComponentType.UNSIGNED_BYTE ||
    accessor.componentType === GltfComponentType.SHORT ||
    accessor.componentType === GltfComponentType.UNSIGNED_SHORT
  );
}

function _validNormalEncoding(accessor: GltfAccessor, meshQuantization: boolean): boolean {
  if (accessor.componentType === GltfComponentType.FLOAT) {
    return accessor.normalized !== true;
  }
  if (!meshQuantization || accessor.normalized !== true) return false;
  // KHR_mesh_quantization permits signed normalized 8/16-bit normals. Unsigned
  // encodings cannot represent the required [-1, 1] normal range.
  return (
    accessor.componentType === GltfComponentType.BYTE ||
    accessor.componentType === GltfComponentType.SHORT
  );
}

function _validatePrimitiveAttributeAccessor(
  gltf: GltfJson,
  accessorIndex: number | undefined,
  allowedTypes: readonly GltfAccessor['type'][],
  expectedCount: number,
  attributeName: string,
  meshLabel: string,
  path: string,
  warnings: string[],
  diagnostics: GltfImportDiagnostic[],
): boolean {
  if (accessorIndex === undefined) return true;
  const accessor = gltf.accessors?.[accessorIndex];
  if (!accessor) {
    throwImportBoundaryError(
      'invalid-primitive-attribute',
      path,
      `[vitrum/gltf-adapter] Mesh "${meshLabel}" ${attributeName} references missing accessor ${accessorIndex}.`,
    );
  }
  if (allowedTypes.includes(accessor.type) && accessor.count === expectedCount) return true;
  void warnings;
  void diagnostics;
  throwImportBoundaryError(
    'invalid-primitive-attribute',
    path,
    `[vitrum/gltf-adapter] Mesh "${meshLabel}" ${attributeName} accessor must be ` +
      `${allowedTypes.join(' or ')} with count ${expectedCount}, but accessor ${accessorIndex} ` +
      `is ${accessor.type} with count ${accessor.count}.`,
  );
}

interface SkinData {
  bones: Float32Array;
  boneInverses: Float32Array;
}

function writeMat4Product(
  out: Float32Array,
  outOffset: number,
  a: ArrayLike<number>,
  b: ArrayLike<number>,
): void {
  for (let column = 0; column < 4; column += 1) {
    for (let row = 0; row < 4; row += 1) {
      let sum = 0;
      for (let k = 0; k < 4; k += 1) {
        sum += (a[k * 4 + row] ?? 0) * (b[column * 4 + k] ?? 0);
      }
      out[outOffset + column * 4 + row] = sum;
    }
  }
}

function mat4RangeIsFinite(values: Float32Array, offset: number): boolean {
  for (let component = 0; component < 16; component += 1) {
    if (!Number.isFinite(values[offset + component])) return false;
  }
  return true;
}

/**
 * Build rest-pose bone matrices and inverse bind matrices for a glTF skin.
 * Returns `undefined` if the node has no skin.
 */
function _extractSkinData(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
  gltfSkins: NonNullable<GltfJson['skins']>,
  skinIdx: number | undefined,
  meshWorld: Float32Array,
  worldTransforms: Map<number, Float32Array>,
  warnings: string[],
  diagnostics: GltfImportDiagnostic[],
  onAccessorDiagnostic: (diagnostic: GltfAccessorDiagnostic) => void,
  resourceLedger: ImportResourceLedger,
): SkinData | undefined {
  if (skinIdx === undefined) return undefined;
  const gltfSkin = gltfSkins[skinIdx];
  if (!Number.isSafeInteger(skinIdx) || skinIdx < 0 || !gltfSkin) {
    throwImportBoundaryError(
      'unreadable-skin-joints',
      'nodes[].skin',
      `[vitrum/gltf-adapter] Skinned mesh references missing skin ${String(skinIdx)}.`,
    );
  }

  const jointCount = gltfSkin.joints.length;
  if (jointCount === 0) {
    throwImportBoundaryError(
      'unreadable-skin-joints',
      `skins[${skinIdx}].joints`,
      `[vitrum/gltf-adapter] Skin "${gltfSkin.name ?? skinIdx}" must contain at least one joint.`,
    );
  }
  const bonesAllocation = geometryArrayAllocation(
    [jointCount, 16],
    Float32Array.BYTES_PER_ELEMENT,
    `skins[${skinIdx}] rest-pose bones`,
  );
  resourceLedger.chargeDecodedGeometryBytes(
    checkedGeometrySum(
      [
        bonesAllocation.byteLength,
        ...(gltfSkin.inverseBindMatrices === undefined ? [bonesAllocation.byteLength] : []),
      ],
      `skins[${skinIdx}] rest-pose matrix byte length`,
    ),
    `skins[${skinIdx}] rest-pose matrices`,
  );
  const meshWorldInverse = mat4Invert(
    meshWorld,
    resourceLedger,
    `skins[${skinIdx}] mesh world inverse`,
  );
  if (!meshWorldInverse) {
    throwImportBoundaryError(
      'singular-skin-transform',
      `skins[${skinIdx}]`,
      `[vitrum/gltf-adapter] Skinned mesh node for skin "${gltfSkin.name ?? skinIdx}" has a singular world transform.`,
    );
  }

  const bones = new Float32Array(bonesAllocation.elementCount);
  for (let j = 0; j < jointCount; j++) {
    const jointNodeIdx = gltfSkin.joints[j]!;
    if (!Number.isSafeInteger(jointNodeIdx) || jointNodeIdx < 0 || !gltf.nodes?.[jointNodeIdx]) {
      throwImportBoundaryError(
        'unreadable-skin-joints',
        `skins[${skinIdx}].joints[${j}]`,
        `[vitrum/gltf-adapter] Skin "${gltfSkin.name ?? skinIdx}" references missing joint node ${String(jointNodeIdx)}.`,
      );
    }
    const jointWorld = worldTransforms.get(jointNodeIdx);
    if (!jointWorld) {
      throwImportBoundaryError(
        'unreadable-skin-joints',
        `skins[${skinIdx}].joints[${j}]`,
        `[vitrum/gltf-adapter] Joint node ${jointNodeIdx} for skin "${gltfSkin.name ?? skinIdx}" has no resolved world transform.`,
      );
    }
    const boneOffset = j * 16;
    writeMat4Product(bones, boneOffset, meshWorldInverse, jointWorld);
    if (!mat4RangeIsFinite(bones, boneOffset)) {
      throwImportBoundaryError(
        'unreadable-skin-joints',
        `skins[${skinIdx}].joints[${j}]`,
        `[vitrum/gltf-adapter] Joint node ${jointNodeIdx} produced a non-finite rest-pose transform.`,
      );
    }
  }

  let boneInverses: Float32Array;
  if (gltfSkin.inverseBindMatrices === undefined) {
    // glTF 2.0 defines an omitted inverseBindMatrices accessor as identity.
    boneInverses = new Float32Array(bonesAllocation.elementCount);
    for (let j = 0; j < jointCount; j++) {
      boneInverses[j * 16 + 0] = 1;
      boneInverses[j * 16 + 5] = 1;
      boneInverses[j * 16 + 10] = 1;
      boneInverses[j * 16 + 15] = 1;
    }
  } else {
    const accessorIndex = gltfSkin.inverseBindMatrices;
    const accessor = gltf.accessors?.[accessorIndex];
    if (
      !Number.isSafeInteger(accessorIndex) ||
      accessorIndex < 0 ||
      !accessor ||
      accessor.type !== 'MAT4' ||
      accessor.componentType !== GltfComponentType.FLOAT ||
      accessor.normalized === true ||
      accessor.count !== jointCount
    ) {
      throwImportBoundaryError(
        'unreadable-inverse-bind-matrices',
        `skins[${skinIdx}].inverseBindMatrices`,
        `[vitrum/gltf-adapter] Skin "${gltfSkin.name ?? skinIdx}" inverseBindMatrices must reference a FLOAT MAT4 accessor with exactly ${jointCount} entries.`,
      );
    }
    try {
      boneInverses = unpackAccessorFloat(
        gltf,
        buffers,
        accessorIndex,
        warnings,
        onAccessorDiagnostic,
        resourceLedger,
      );
    } catch (e) {
      rethrowResourceLimitError(e);
      throwImportBoundaryError(
        'unreadable-inverse-bind-matrices',
        `skins[${skinIdx}].inverseBindMatrices`,
        `[vitrum/gltf-adapter] Failed to read inverseBindMatrices for skin "${gltfSkin.name ?? skinIdx}": ${String(e)}`,
      );
    }
    if (boneInverses.length !== jointCount * 16 || !boneInverses.every(Number.isFinite)) {
      throwImportBoundaryError(
        'unreadable-inverse-bind-matrices',
        `skins[${skinIdx}].inverseBindMatrices`,
        `[vitrum/gltf-adapter] Skin "${gltfSkin.name ?? skinIdx}" inverseBindMatrices decoded to ${boneInverses.length} values; expected ${jointCount * 16} finite values.`,
      );
    }
  }

  void diagnostics;
  return { bones, boneInverses };
}

interface PrimitiveUvMaterialResolution {
  readonly material: MaterialSpec;
  readonly droppedTextureFields?: readonly MaterialTextureRefField[];
  readonly uvs?: Float32Array;
  /** glTF TEXCOORD_0 compatibility alias carried by core `uvs`. */
  readonly uvSourceTexCoord?: number;
  readonly uv1?: Float32Array;
  /** glTF TEXCOORD_1 compatibility alias carried by core `uv1`. */
  readonly uv1SourceTexCoord?: number;
  /** Sparse, index-preserving glTF TEXCOORD_N streams. */
  readonly uvSets?: ReadonlyArray<Float32Array | undefined>;
}

function _isTextureRef(value: unknown): value is TextureRef {
  return value !== null && typeof value === 'object' && 'handle' in value;
}

function _cloneMaterialWithoutTextureRefs(
  material: MaterialSpec,
  fields: readonly MaterialTextureRefField[],
): MaterialSpec {
  const next: Record<string, unknown> = { ...material };
  for (const field of fields) delete next[field];
  return next as unknown as MaterialSpec;
}

function uvLaneResult(
  material: MaterialSpec,
  uvs: Float32Array | undefined,
  uv1: Float32Array | undefined,
  uvSets: ReadonlyArray<Float32Array | undefined>,
  droppedTextureFields?: readonly MaterialTextureRefField[],
): PrimitiveUvMaterialResolution {
  const normalizedUvSets = cloneSparseArray(uvSets);
  const uv0 = uvs ?? normalizedUvSets[0];
  const uvOne = uv1 ?? normalizedUvSets[1];
  if (uv0 != null) normalizedUvSets[0] = uv0;
  if (uvOne != null) normalizedUvSets[1] = uvOne;
  return {
    material,
    ...(droppedTextureFields != null && droppedTextureFields.length > 0
      ? { droppedTextureFields }
      : {}),
    ...(uv0 != null ? { uvs: uv0, uvSourceTexCoord: 0 } : {}),
    ...(uvOne != null ? { uv1: uvOne, uv1SourceTexCoord: 1 } : {}),
    ...(sparseArrayHasDefinedEntry(normalizedUvSets) ? { uvSets: normalizedUvSets } : {}),
  };
}

function _resolvePrimitiveUvMaterial(
  material: MaterialSpec,
  uvs: Float32Array | undefined,
  uv1: Float32Array | undefined,
  uvSets: ReadonlyArray<Float32Array | undefined>,
  warnings: string[],
  diagnostics: GltfImportDiagnostic[],
  primitivePath: string,
  meshName: string | number,
): PrimitiveUvMaterialResolution {
  const textureFields: { field: MaterialTextureRefField; ref: TextureRef; texCoord: number }[] = [];

  for (const field of MATERIAL_TEXTURE_REF_FIELDS) {
    const ref = material[field];
    if (!_isTextureRef(ref)) continue;
    const texCoord = ref.texCoord ?? 0;
    if (!Number.isSafeInteger(texCoord) || texCoord < 0) {
      const source = gltfTextureRefSource(ref);
      throwImportBoundaryError(
        'missing-material-texcoord',
        source?.path ?? `${primitivePath}.material.${field}.texCoord`,
        `[vitrum/gltf-adapter] Mesh "${meshName}" material field "${field}" has invalid texCoord ${String(texCoord)}.`,
      );
    }
    textureFields.push({ field, ref, texCoord });
  }

  const streamFor = (texCoord: number): Float32Array | undefined =>
    uvSets[texCoord] ?? (texCoord === 0 ? uvs : texCoord === 1 ? uv1 : undefined);
  const missingFields = textureFields.filter(({ texCoord }) => streamFor(texCoord) == null);
  if (missingFields.length === 0) return uvLaneResult(material, uvs, uv1, uvSets);
  const { field, ref, texCoord } = missingFields[0]!;
  const source = gltfTextureRefSource(ref);
  throwImportBoundaryError(
    'missing-material-texcoord',
    source?.path ?? `${primitivePath}.material.${field}`,
    `[vitrum/gltf-adapter] Mesh "${meshName}" material field "${field}" references TEXCOORD_${texCoord}, but the primitive has no readable TEXCOORD_${texCoord} accessor.`,
  );
}

function checkedGeometryProduct(factors: readonly number[], allocationPath: string): number {
  let product = 1;
  for (const factor of factors) {
    if (!Number.isSafeInteger(factor) || factor < 0) {
      throw new RangeError(
        `[vitrum/gltf-adapter] ${allocationPath} dimensions must be non-negative safe integers.`,
      );
    }
    if (product !== 0 && factor > Math.floor(Number.MAX_SAFE_INTEGER / product)) {
      throw new RangeError(
        `[vitrum/gltf-adapter] ${allocationPath} exceeds the safe integer range.`,
      );
    }
    product *= factor;
  }
  return product;
}

function checkedGeometrySum(values: readonly number[], allocationPath: string): number {
  let total = 0;
  for (const value of values) {
    if (!Number.isSafeInteger(value) || value < 0 || value > Number.MAX_SAFE_INTEGER - total) {
      throw new RangeError(
        `[vitrum/gltf-adapter] ${allocationPath} exceeds the safe integer range.`,
      );
    }
    total += value;
  }
  return total;
}

function geometryArrayAllocation(
  dimensions: readonly number[],
  bytesPerElement: number,
  allocationPath: string,
): { readonly elementCount: number; readonly byteLength: number } {
  const elementCount = checkedGeometryProduct(dimensions, `${allocationPath} element count`);
  const byteLength = checkedGeometryProduct(
    [elementCount, bytesPerElement],
    `${allocationPath} byte length`,
  );
  return { elementCount, byteLength };
}

function allocateGeometryFloat32(
  resourceLedger: ImportResourceLedger,
  dimensions: readonly number[],
  allocationPath: string,
): Float32Array {
  const allocation = geometryArrayAllocation(
    dimensions,
    Float32Array.BYTES_PER_ELEMENT,
    allocationPath,
  );
  resourceLedger.chargeDecodedGeometryBytes(allocation.byteLength, allocationPath);
  return new Float32Array(allocation.elementCount);
}

function remapVec2Attribute(
  attr: Float32Array | undefined,
  sourceVertices: Uint32Array,
  resourceLedger: ImportResourceLedger,
  allocationPath: string,
): Float32Array | undefined {
  if (attr == null) return undefined;
  const allocation = geometryArrayAllocation(
    [sourceVertices.length, 2],
    Float32Array.BYTES_PER_ELEMENT,
    allocationPath,
  );
  resourceLedger.chargeDecodedGeometryBytes(allocation.byteLength, allocationPath);
  const out = new Float32Array(allocation.elementCount);
  for (let i = 0; i < sourceVertices.length; i += 1) {
    const src = sourceVertices[i]! * 2;
    out[i * 2 + 0] = attr[src + 0] ?? 0;
    out[i * 2 + 1] = attr[src + 1] ?? 0;
  }
  return out;
}

function remapVertexColors(
  colors: Float32Array | undefined,
  sourceVertexCount: number,
  sourceVertices: Uint32Array,
  resourceLedger: ImportResourceLedger,
  allocationPath: string,
): Float32Array | undefined {
  if (colors == null || sourceVertexCount <= 0) return undefined;
  const components = Math.max(1, Math.floor(colors.length / sourceVertexCount));
  const allocation = geometryArrayAllocation(
    [sourceVertices.length, components],
    Float32Array.BYTES_PER_ELEMENT,
    allocationPath,
  );
  resourceLedger.chargeDecodedGeometryBytes(allocation.byteLength, allocationPath);
  const out = new Float32Array(allocation.elementCount);
  for (let i = 0; i < sourceVertices.length; i += 1) {
    const src = sourceVertices[i]! * components;
    const dst = i * components;
    for (let c = 0; c < components; c += 1) out[dst + c] = colors[src + c] ?? (c === 3 ? 1 : 0);
  }
  return out;
}

function remapPackedUintAttribute(
  values: Uint32Array,
  components: number,
  sourceVertices: Uint32Array,
  resourceLedger: ImportResourceLedger,
  allocationPath: string,
): Uint32Array {
  const allocation = geometryArrayAllocation(
    [sourceVertices.length, components],
    Uint32Array.BYTES_PER_ELEMENT,
    allocationPath,
  );
  resourceLedger.chargeDecodedGeometryBytes(allocation.byteLength, allocationPath);
  const out = new Uint32Array(allocation.elementCount);
  for (let i = 0; i < sourceVertices.length; i += 1) {
    const src = sourceVertices[i]! * components;
    out.set(values.subarray(src, src + components), i * components);
  }
  return out;
}

function remapPackedFloatAttribute(
  values: Float32Array,
  components: number,
  sourceVertices: Uint32Array,
  resourceLedger: ImportResourceLedger,
  allocationPath: string,
): Float32Array {
  const allocation = geometryArrayAllocation(
    [sourceVertices.length, components],
    Float32Array.BYTES_PER_ELEMENT,
    allocationPath,
  );
  resourceLedger.chargeDecodedGeometryBytes(allocation.byteLength, allocationPath);
  const out = new Float32Array(allocation.elementCount);
  for (let i = 0; i < sourceVertices.length; i += 1) {
    const src = sourceVertices[i]! * components;
    out.set(values.subarray(src, src + components), i * components);
  }
  return out;
}

function remapMorphData(
  morph: MorphData | undefined,
  sourceVertices: Uint32Array,
  resourceLedger: ImportResourceLedger,
  allocationPath: string,
): MorphData | undefined {
  if (morph == null) return undefined;
  resourceLedger.chargeDecodedGeometryBytes(
    morph.morphWeights.byteLength,
    `${allocationPath}.morphWeights`,
  );
  return {
    morphTargets: morph.morphTargets.map((target, index) =>
      remapVec3Attribute(
        target,
        sourceVertices,
        resourceLedger,
        `${allocationPath}.morphTargets[${index}]`,
      ),
    ),
    ...(morph.morphTargetNormals != null
      ? {
          morphTargetNormals: morph.morphTargetNormals.map((target, index) =>
            remapVec3Attribute(
              target,
              sourceVertices,
              resourceLedger,
              `${allocationPath}.morphTargetNormals[${index}]`,
            ),
          ),
        }
      : {}),
    ...(morph.morphTargetTangents != null
      ? {
          morphTargetTangents: morph.morphTargetTangents.map((target, index) =>
            remapVec3Attribute(
              target,
              sourceVertices,
              resourceLedger,
              `${allocationPath}.morphTargetTangents[${index}]`,
            ),
          ),
        }
      : {}),
    ...(morph.morphTargetUvs != null
      ? {
          morphTargetUvs: morph.morphTargetUvs.map(
            (target, index) =>
              remapVec2Attribute(
                target,
                sourceVertices,
                resourceLedger,
                `${allocationPath}.morphTargetUvs[${index}]`,
              )!,
          ),
        }
      : {}),
    ...(morph.morphTargetUv1s != null
      ? {
          morphTargetUv1s: morph.morphTargetUv1s.map(
            (target, index) =>
              remapVec2Attribute(
                target,
                sourceVertices,
                resourceLedger,
                `${allocationPath}.morphTargetUv1s[${index}]`,
              )!,
          ),
        }
      : {}),
    ...(morph.morphTargetUvSets != null
      ? {
          morphTargetUvSets: mapSparseArray(
            morph.morphTargetUvSets,
            (targets, texCoord) => targets?.map(
              (target, index) =>
                remapVec2Attribute(
                  target,
                  sourceVertices,
                  resourceLedger,
                  `${allocationPath}.morphTargetUvSets[${texCoord}][${index}]`,
                )!,
            ),
          ),
        }
      : {}),
    morphWeights: new Float32Array(morph.morphWeights),
  };
}

function remapVec3Attribute(
  values: Float32Array,
  sourceVertices: Uint32Array,
  resourceLedger: ImportResourceLedger,
  allocationPath: string,
): Float32Array {
  const allocation = geometryArrayAllocation(
    [sourceVertices.length, 3],
    Float32Array.BYTES_PER_ELEMENT,
    allocationPath,
  );
  resourceLedger.chargeDecodedGeometryBytes(allocation.byteLength, allocationPath);
  const out = new Float32Array(allocation.elementCount);
  for (let i = 0; i < sourceVertices.length; i += 1) {
    const src = sourceVertices[i]! * 3;
    out[i * 3 + 0] = values[src + 0] ?? 0;
    out[i * 3 + 1] = values[src + 1] ?? 0;
    out[i * 3 + 2] = values[src + 2] ?? 0;
  }
  return out;
}

function skinPrimitiveUvSet(
  primitive: SkinnedMeshPrimitive,
  texCoord: number,
): Float32Array | undefined {
  return (
    primitive.uvSets?.[texCoord] ??
    (texCoord === 0 ? primitive.uvs : texCoord === 1 ? primitive.uv1 : undefined)
  );
}

/**
 * `solveSkin` is a core helper and intentionally has no adapter dependency.
 * Account its complete no-output-buffer allocation path here before invoking
 * it so one import-wide geometry ceiling also governs this derived geometry.
 */
function chargeSolveSkinAllocations(
  resourceLedger: ImportResourceLedger,
  primitive: SkinnedMeshPrimitive,
  allocationPath: string,
): void {
  const byteLengths: number[] = [
    primitive.positions.byteLength,
    primitive.normals.byteLength,
    primitive.bones.byteLength,
    2 * 9 * Float32Array.BYTES_PER_ELEMENT,
  ];
  if (primitive.tangents != null) {
    byteLengths.push(primitive.tangents.byteLength);
  }

  const morphActive =
    primitive.morphTargets != null &&
    primitive.morphWeights != null &&
    primitive.morphTargets.length > 0 &&
    primitive.morphWeights.some((weight) => weight !== 0);
  if (morphActive) {
    byteLengths.push(primitive.positions.byteLength);
    if (primitive.morphTargetNormals != null) {
      byteLengths.push(primitive.normals.byteLength);
    }
    if (primitive.morphTargetTangents != null && primitive.tangents != null) {
      byteLengths.push(
        geometryArrayAllocation(
          [primitive.positions.length / 3, 3],
          Float32Array.BYTES_PER_ELEMENT,
          `${allocationPath} morphed tangents`,
        ).byteLength,
      );
    }

    let copiedUv0 = false;
    let copiedUv1 = false;
    if (primitive.morphTargetUvSets != null) {
      for (const texCoord of sparseArrayOwnIndices(primitive.morphTargetUvSets)) {
        if (primitive.morphTargetUvSets[texCoord] == null) continue;
        const base = skinPrimitiveUvSet(primitive, texCoord);
        if (base != null) byteLengths.push(base.byteLength);
        if (texCoord === 0) copiedUv0 = true;
        if (texCoord === 1) copiedUv1 = true;
      }
    }
    if (!copiedUv0 && primitive.morphTargetUvs != null && primitive.uvs != null) {
      byteLengths.push(primitive.uvs.byteLength);
    }
    if (!copiedUv1 && primitive.morphTargetUv1s != null && primitive.uv1 != null) {
      byteLengths.push(primitive.uv1.byteLength);
    }
  }

  resourceLedger.chargeDecodedGeometryBytes(
    checkedGeometrySum(byteLengths, `${allocationPath} byte length`),
    allocationPath,
  );
}

/**
 * Assemble a ScenePrimitive from already-unpacked attribute buffers.
 * Returns a skinned or static primitive depending on which optional fields
 * are present.
 */
function _buildPrimitive(
  id: string,
  worldMat: Mat4,
  positions: Float32Array,
  normals: Float32Array,
  indices: Uint32Array | undefined,
  uvs: Float32Array | undefined,
  uv1: Float32Array | undefined,
  uvSets: ReadonlyArray<Float32Array | undefined>,
  tangents: Float32Array | undefined,
  colors: Float32Array | undefined,
  colorSets: ReadonlyArray<Float32Array | undefined>,
  material: MaterialSpec,
  skin?: {
    skinIndices: Uint32Array;
    skinWeights: Float32Array;
    skinInfluencesPerVertex?: number;
    bones: Float32Array;
    boneInverses: Float32Array;
    bindMatrix?: Float32Array;
    bindMatrixInverse?: Float32Array;
  },
  morph?: MorphData,
  instances?: readonly Mat4[],
): ScenePrimitive {
  const base = {
    id,
    positions,
    normals,
    ...(uvs ? { uvs } : {}),
    ...(uv1 ? { uv1 } : {}),
    ...(sparseArrayHasDefinedEntry(uvSets) ? { uvSets } : {}),
    ...(tangents ? { tangents } : {}),
    ...(colors ? { colors } : {}),
    ...(sparseArrayHasDefinedEntry(colorSets) ? { colorSets } : {}),
    ...(indices ? { indices } : {}),
    material,
  };
  if (skin) {
    return {
      kind: 'skinned-mesh' as const,
      ...base,
      transform: worldMat,
      skinIndices: skin.skinIndices,
      skinWeights: skin.skinWeights,
      ...(skin.skinInfluencesPerVertex != null
        ? { skinInfluencesPerVertex: skin.skinInfluencesPerVertex }
        : {}),
      bones: skin.bones,
      boneInverses: skin.boneInverses,
      ...(skin.bindMatrix ? { bindMatrix: skin.bindMatrix } : {}),
      ...(skin.bindMatrixInverse ? { bindMatrixInverse: skin.bindMatrixInverse } : {}),
      ...(morph
        ? {
            morphTargets: morph.morphTargets,
            ...(morph.morphTargetNormals ? { morphTargetNormals: morph.morphTargetNormals } : {}),
            ...(morph.morphTargetTangents
              ? { morphTargetTangents: morph.morphTargetTangents }
              : {}),
            ...(morph.morphTargetUvs ? { morphTargetUvs: morph.morphTargetUvs } : {}),
            ...(morph.morphTargetUv1s ? { morphTargetUv1s: morph.morphTargetUv1s } : {}),
            ...(morph.morphTargetUvSets ? { morphTargetUvSets: morph.morphTargetUvSets } : {}),
            morphWeights: morph.morphWeights,
          }
        : {}),
    };
  }
  if (instances) {
    return {
      kind: 'instanced-mesh' as const,
      ...base,
      instances,
    };
  }
  return { kind: 'mesh' as const, ...base, transform: worldMat };
}

// ── Morph-target extraction (GLTF-04) ────────────────────────────────────────

interface MorphData {
  /** Per-target POSITION deltas, each `vertexCount * 3` (zeros when a target
   *  has no POSITION delta — glTF allows NORMAL-only targets). */
  morphTargets: Float32Array[];
  /** Per-target NORMAL deltas — present only when at least one target carries
   *  NORMAL deltas; targets without one get zeros (solveSkin requires the
   *  array to be parallel with morphTargets). */
  morphTargetNormals?: Float32Array[];
  /** Per-target TANGENT direction deltas (glTF target TANGENT is VEC3, not the
   *  base VEC4 tangent handedness). Present only when at least one target
   *  carries TANGENT; targets without one get zeros. */
  morphTargetTangents?: Float32Array[];
  /** Per-target TEXCOORD_0 compatibility alias. */
  morphTargetUvs?: Float32Array[];
  /** Per-target TEXCOORD_1 compatibility alias. */
  morphTargetUv1s?: Float32Array[];
  /** Per-UV-set, per-target TEXCOORD_N deltas. */
  morphTargetUvSets?: Array<Float32Array[] | undefined>;
  /** Initial per-target weights from `node.weights ?? mesh.weights` (zeros
   *  when neither is authored). */
  morphWeights: Float32Array;
}

/**
 * Parse glTF `primitive.targets` into core morph-target delta arrays.
 *
 * glTF §3.7.2.2: each target maps attribute names to accessors carrying
 * DELTAS from the base attribute (sparse accessors are common here and are
 * handled by `unpackAccessorFloat`). POSITION, NORMAL, TANGENT, and every
 * TEXCOORD_N with a matching base stream map onto the corresponding scalable
 * `SkinnedMeshPrimitive` morph arrays.
 *
 * Returns `undefined` when the primitive has no targets.
 */
function _extractMorphTargets(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
  targets: ReadonlyArray<Record<string, number>> | undefined,
  authoredWeights: number[] | undefined,
  vertexCount: number,
  baseUvSets: ReadonlyArray<Float32Array | undefined>,
  meshLabel: string,
  primitivePath: string,
  warnings: string[],
  diagnostics: GltfImportDiagnostic[],
  onAccessorDiagnostic: (diagnostic: GltfAccessorDiagnostic) => void,
  resourceLedger: ImportResourceLedger,
): MorphData | undefined {
  if (!targets || targets.length === 0) return undefined;
  const tCount = targets.length;

  const morphTargets: Float32Array[] = [];
  const normalDeltas: (Float32Array | null)[] = [];
  const tangentDeltas: (Float32Array | null)[] = [];
  const morphUvSetIndexSet = new Set<number>();
  for (let targetIndex = 0; targetIndex < targets.length; targetIndex += 1) {
    for (const semantic of Object.keys(targets[targetIndex]!)) {
      const texCoord = parseCanonicalSetSemantic(
        semantic,
        'TEXCOORD',
        `${primitivePath}.targets[${targetIndex}].${semantic}`,
        'ignored-morph-target-attribute',
      );
      if (texCoord === undefined) continue;
      morphUvSetIndexSet.add(texCoord);
    }
  }
  const morphUvSetIndices = [...morphUvSetIndexSet].sort((a, b) => a - b);
  const uvSetDeltas: Array<Array<Float32Array | null> | undefined> = [];
  for (const texCoord of morphUvSetIndices) uvSetDeltas[texCoord] = [];
  const populatedUvSets = new Set<number>();
  let anyNormals = false;
  let anyTangents = false;

  for (let t = 0; t < tCount; t++) {
    const target = targets[t]!;

    // POSITION delta.
    _validatePrimitiveAttributeAccessor(
      gltf,
      target['POSITION'],
      ['VEC3'],
      vertexCount,
      'POSITION morph delta',
      meshLabel,
      `${primitivePath}.targets[${t}].POSITION`,
      warnings,
      diagnostics,
    );
    const posDelta = _tryUnpackFloat(
      gltf,
      buffers,
      target['POSITION'],
      `morph target ${t} POSITION for "${meshLabel}"`,
      warnings,
      onAccessorDiagnostic,
      diagnostics,
      'unreadable-optional-attribute',
      `${primitivePath}.targets[${t}].POSITION`,
      resourceLedger,
    );
    if (posDelta && posDelta.length !== vertexCount * 3) {
      throwImportBoundaryError(
        'invalid-morph-target-delta-length',
        `${primitivePath}.targets[${t}].POSITION`,
        `[vitrum/gltf-adapter] Morph target ${t} POSITION delta length ${posDelta.length} != ${vertexCount * 3} for "${meshLabel}".`,
      );
    }
    morphTargets.push(
      posDelta ??
        allocateGeometryFloat32(
          resourceLedger,
          [vertexCount, 3],
          `${primitivePath}.targets[${t}].POSITION default delta`,
        ),
    );

    // NORMAL delta.
    _validatePrimitiveAttributeAccessor(
      gltf,
      target['NORMAL'],
      ['VEC3'],
      vertexCount,
      'NORMAL morph delta',
      meshLabel,
      `${primitivePath}.targets[${t}].NORMAL`,
      warnings,
      diagnostics,
    );
    const nrmDelta = _tryUnpackFloat(
      gltf,
      buffers,
      target['NORMAL'],
      `morph target ${t} NORMAL for "${meshLabel}"`,
      warnings,
      onAccessorDiagnostic,
      diagnostics,
      'unreadable-optional-attribute',
      `${primitivePath}.targets[${t}].NORMAL`,
      resourceLedger,
    );
    if (nrmDelta && nrmDelta.length !== vertexCount * 3) {
      throwImportBoundaryError(
        'invalid-morph-target-delta-length',
        `${primitivePath}.targets[${t}].NORMAL`,
        `[vitrum/gltf-adapter] Morph target ${t} NORMAL delta length ${nrmDelta.length} != ${vertexCount * 3} for "${meshLabel}".`,
      );
    }
    if (nrmDelta) anyNormals = true;
    normalDeltas.push(nrmDelta ?? null);

    // TANGENT direction delta (glTF morph target TANGENT is VEC3; base tangent
    // handedness remains in the base TANGENT.w lane).
    _validatePrimitiveAttributeAccessor(
      gltf,
      target['TANGENT'],
      ['VEC3'],
      vertexCount,
      'TANGENT morph delta',
      meshLabel,
      `${primitivePath}.targets[${t}].TANGENT`,
      warnings,
      diagnostics,
    );
    const tanDelta = _tryUnpackFloat(
      gltf,
      buffers,
      target['TANGENT'],
      `morph target ${t} TANGENT for "${meshLabel}"`,
      warnings,
      onAccessorDiagnostic,
      diagnostics,
      'unreadable-optional-attribute',
      `${primitivePath}.targets[${t}].TANGENT`,
      resourceLedger,
    );
    if (tanDelta && tanDelta.length !== vertexCount * 3) {
      throwImportBoundaryError(
        'invalid-morph-target-delta-length',
        `${primitivePath}.targets[${t}].TANGENT`,
        `[vitrum/gltf-adapter] Morph target ${t} TANGENT delta length ${tanDelta.length} != ${vertexCount * 3} for "${meshLabel}".`,
      );
    }
    if (tanDelta) anyTangents = true;
    tangentDeltas.push(tanDelta ?? null);

    for (const texCoord of morphUvSetIndices) {
      const semantic = `TEXCOORD_${texCoord}`;
      _validatePrimitiveAttributeAccessor(
        gltf,
        target[semantic],
        ['VEC2'],
        vertexCount,
        `${semantic} morph delta`,
        meshLabel,
        `${primitivePath}.targets[${t}].${semantic}`,
        warnings,
        diagnostics,
      );
      const delta = _tryUnpackFloat(
        gltf,
        buffers,
        target[semantic],
        `morph target ${t} ${semantic} for "${meshLabel}"`,
        warnings,
        onAccessorDiagnostic,
        diagnostics,
        'unreadable-optional-attribute',
        `${primitivePath}.targets[${t}].${semantic}`,
        resourceLedger,
      );
      if (delta && baseUvSets[texCoord] == null) {
        throwImportBoundaryError(
          'ignored-morph-target-texcoord',
          `${primitivePath}.targets[${t}].${semantic}`,
          `[vitrum/gltf-adapter] Morph target ${t} ${semantic} delta in mesh "${meshLabel}" has no matching base ${semantic} stream.`,
        );
      }
      if (delta && delta.length !== vertexCount * 2) {
        throwImportBoundaryError(
          'invalid-morph-target-delta-length',
          `${primitivePath}.targets[${t}].${semantic}`,
          `[vitrum/gltf-adapter] Morph target ${t} ${semantic} delta length ${delta.length} != ${vertexCount * 2} for "${meshLabel}".`,
        );
      }
      if (delta) populatedUvSets.add(texCoord);
      uvSetDeltas[texCoord]!.push(delta ?? null);
    }

    for (const attr of Object.keys(target)) {
      const colorSet = parseCanonicalSetSemantic(
        attr,
        'COLOR',
        `${primitivePath}.targets[${t}].${attr}`,
        'ignored-morph-target-attribute',
      );
      if (
        attr !== 'POSITION' &&
        attr !== 'NORMAL' &&
        attr !== 'TANGENT' &&
        !/^TEXCOORD_\d+$/.test(attr)
      ) {
        if (colorSet !== undefined || attr.startsWith('_')) {
          emitImportDiagnostic(warnings, diagnostics, {
            severity: 'warning',
            code: 'ignored-morph-target-attribute',
            path: `${primitivePath}.targets[${t}].${attr}`,
            message:
              `[vitrum/gltf-adapter] Morph target ${t} attribute "${attr}" in mesh ` +
              `"${meshLabel}" is valid optional/application-specific glTF data but is not ` +
              'represented by the core Scene morph contract and was ignored.',
          });
          continue;
        }
        throwImportBoundaryError(
          'ignored-morph-target-attribute',
          `${primitivePath}.targets[${t}].${attr}`,
          `[vitrum/gltf-adapter] Morph target ${t} attribute "${attr}" in mesh "${meshLabel}" cannot be represented exactly.`,
        );
      }
    }
  }

  // Initial weights: node-level overrides mesh-level per glTF §3.7.2.2;
  // absent weights default to 0 (rest pose).
  const morphWeights = allocateGeometryFloat32(
    resourceLedger,
    [tCount],
    `${primitivePath}.weights`,
  );
  if (authoredWeights) {
    if (authoredWeights.length !== tCount) {
      throwImportBoundaryError(
        'morph-weight-count-mismatch',
        `${primitivePath}.weights`,
        `[vitrum/gltf-adapter] Mesh "${meshLabel}" morph weights length ${authoredWeights.length} != target count ${tCount}.`,
      );
    }
    for (let t = 0; t < tCount; t++) {
      const weight = authoredWeights[t]!;
      if (!Number.isFinite(weight)) {
        throwImportBoundaryError(
          'morph-weight-count-mismatch',
          `${primitivePath}.weights[${t}]`,
          `[vitrum/gltf-adapter] Mesh "${meshLabel}" morph weight ${t} must be finite; received ${String(weight)}.`,
        );
      }
      morphWeights[t] = weight;
    }
  }

  const morphTargetUvSets: Array<Float32Array[] | undefined> = mapSparseArray(
    uvSetDeltas,
    (deltas, texCoord) =>
      populatedUvSets.has(texCoord)
        ? deltas!.map(
            (delta, targetIndex) =>
              delta ??
              allocateGeometryFloat32(
                resourceLedger,
                [vertexCount, 2],
                `${primitivePath}.targets[${targetIndex}].TEXCOORD_${texCoord} default delta`,
              ),
          )
        : undefined,
  );
  const anyMorphUvSets = sparseArrayHasDefinedEntry(morphTargetUvSets);

  return {
    morphTargets,
    ...(anyNormals
      ? {
          morphTargetNormals: normalDeltas.map(
            (normal, targetIndex) =>
              normal ??
              allocateGeometryFloat32(
                resourceLedger,
                [vertexCount, 3],
                `${primitivePath}.targets[${targetIndex}].NORMAL default delta`,
              ),
          ),
        }
      : {}),
    ...(anyTangents
      ? {
          morphTargetTangents: tangentDeltas.map(
            (tangent, targetIndex) =>
              tangent ??
              allocateGeometryFloat32(
                resourceLedger,
                [vertexCount, 3],
                `${primitivePath}.targets[${targetIndex}].TANGENT default delta`,
              ),
          ),
        }
      : {}),
    ...(morphTargetUvSets[0] ? { morphTargetUvs: morphTargetUvSets[0] } : {}),
    ...(morphTargetUvSets[1] ? { morphTargetUv1s: morphTargetUvSets[1] } : {}),
    ...(anyMorphUvSets ? { morphTargetUvSets } : {}),
    morphWeights,
  };
}

/**
 * Convert one KHR_lights_punctual light (with its node world transform) to a
 * core SceneEmitter. Invalid or unsupported payloads fail at the import boundary.
 */
function _convertPunctualLight(
  light: NonNullable<KhrLightsPunctualRoot['lights']>[number],
  worldMat: Mat4,
  id: string,
  warnings: string[],
  diagnostics: GltfImportDiagnostic[],
  lightPath: string,
): SceneEmitter {
  if (!isObject(light)) {
    throwImportBoundaryError(
      'missing-punctual-light',
      lightPath,
      '[vitrum/gltf-adapter] KHR_lights_punctual light must be an object.',
    );
  }
  if (
    light.color !== undefined &&
    (!Array.isArray(light.color) ||
      light.color.length !== 3 ||
      !light.color.every((value) => Number.isFinite(value) && value >= 0))
  ) {
    throwImportBoundaryError(
      'missing-punctual-light',
      `${lightPath}.color`,
      '[vitrum/gltf-adapter] KHR_lights_punctual color must contain exactly three finite non-negative values.',
    );
  }
  const color: [number, number, number] = light.color
    ? [light.color[0], light.color[1], light.color[2]]
    : [1, 1, 1];
  const intensity = light.intensity ?? 1;
  if (!Number.isFinite(intensity) || intensity < 0) {
    throwImportBoundaryError(
      'missing-punctual-light',
      `${lightPath}.intensity`,
      '[vitrum/gltf-adapter] KHR_lights_punctual intensity must be finite and non-negative.',
    );
  }
  if (light.range !== undefined && (!Number.isFinite(light.range) || light.range <= 0)) {
    throwImportBoundaryError(
      'missing-punctual-light',
      `${lightPath}.range`,
      '[vitrum/gltf-adapter] KHR_lights_punctual range must be finite and greater than zero when present.',
    );
  }

  // Column-major 4×4: translation at indices 12, 13, 14.
  const px = worldMat[12] ?? 0;
  const py = worldMat[13] ?? 0;
  const pz = worldMat[14] ?? 0;
  if (![px, py, pz].every(Number.isFinite)) {
    throwImportBoundaryError(
      'missing-punctual-light',
      lightPath,
      '[vitrum/gltf-adapter] KHR_lights_punctual node transform produced a non-finite position.',
    );
  }

  // glTF lights point along -Z in local space; world -Z column is at indices 8,9,10.
  const lzx = -(worldMat[8] ?? 0);
  const lzy = -(worldMat[9] ?? 0);
  const lzz = -(worldMat[10] ?? 0);
  const lzLen = Math.hypot(lzx, lzy, lzz);
  if (!Number.isFinite(lzLen) || lzLen <= 1e-10) {
    throwImportBoundaryError(
      'missing-punctual-light',
      lightPath,
      '[vitrum/gltf-adapter] KHR_lights_punctual node transform has no finite light direction.',
    );
  }
  const dirX = lzx / lzLen;
  const dirY = lzy / lzLen;
  const dirZ = lzz / lzLen;

  switch (light.type) {
    case 'point':
      return {
        kind: 'point',
        id,
        color,
        intensity,
        position: [px, py, pz],
        ...(light.range != null && light.range > 0 ? { distance: light.range } : {}),
        decay: 2, // glTF punctual always uses physical inverse-square falloff
      };

    case 'spot': {
      const inner = light.spot?.innerConeAngle ?? 0;
      const outer = light.spot?.outerConeAngle ?? Math.PI / 4;
      if (
        !Number.isFinite(inner) ||
        !Number.isFinite(outer) ||
        inner < 0 ||
        outer <= 0 ||
        inner >= outer ||
        outer > Math.PI / 2
      ) {
        throwImportBoundaryError(
          'missing-punctual-light',
          `${lightPath}.spot`,
          '[vitrum/gltf-adapter] Spot cone angles must satisfy 0 <= innerConeAngle < outerConeAngle <= PI/2.',
        );
      }
      const penumbra = outer > 1e-10 ? 1 - inner / outer : 0;
      return {
        kind: 'spot',
        id,
        color,
        intensity,
        position: [px, py, pz],
        direction: [dirX, dirY, dirZ],
        angle: outer,
        penumbra: Math.max(0, Math.min(1, penumbra)),
        ...(light.range != null && light.range > 0 ? { distance: light.range } : {}),
        decay: 2,
      };
    }

    case 'directional':
      // Core contract: direction points AT the light (toward source).
      return {
        kind: 'directional',
        id,
        color,
        intensity,
        direction: [-dirX, -dirY, -dirZ],
      };

    default:
      void warnings;
      void diagnostics;
      throwImportBoundaryError(
        'unsupported-punctual-light-type',
        `${lightPath}.type`,
        `[vitrum/gltf-adapter] KHR_lights_punctual light "${light.name ?? id}" has unsupported type "${String((light as { type?: unknown }).type)}".`,
      );
  }
}

/**
 * Unpack a JOINTS_0 accessor into a Uint32Array (4 indices per vertex).
 *
 * glTF §3.7.2.1: JOINTS_0 may be UNSIGNED_BYTE (5121) or UNSIGNED_SHORT (5123).
 * Normalized flag is NOT applied to joint indices (they are raw integers).
 *
 * Reference: glTF 2.0 spec §3.7.2 (Skinned Mesh Attributes)
 * https://www.khronos.org/registry/glTF/specs/2.0/glTF-2.0.html#skinned-mesh-attributes
 */
interface SkinInfluenceSet {
  readonly setIndex: number;
  readonly joints: Uint32Array;
  readonly weights: Float32Array;
}

interface PackedSkinInfluences {
  readonly skinIndices: Uint32Array;
  readonly skinWeights: Float32Array;
  readonly skinInfluencesPerVertex: number;
}

function isSkinInfluenceAttribute(attrName: string): boolean {
  const match = /^(JOINTS|WEIGHTS)_([0-9]+)$/u.exec(attrName);
  if (match == null) return false;
  const setIndex = Number(match[2]);
  return Number.isSafeInteger(setIndex) &&
    attrName === `${match[1]}_${setIndex}`;
}

function collectSkinInfluenceSetIndices(
  attributes: GltfPrimitive['attributes'],
  primitivePath: string,
): number[] {
  const sets = new Set<number>();
  for (const attrName of Object.keys(attributes ?? {})) {
    const match = /^(JOINTS|WEIGHTS)_([0-9]+)$/u.exec(attrName);
    if (!match) continue;
    const setIndex = parseCanonicalSetSemantic(
      attrName,
      match[1] as 'JOINTS' | 'WEIGHTS',
      `${primitivePath}.attributes.${attrName}`,
      'ignored-primitive-attribute',
    );
    if (setIndex !== undefined) sets.add(setIndex);
  }
  return [...sets].sort((a, b) => a - b);
}

function packSkinInfluenceSets(
  sets: readonly SkinInfluenceSet[],
  vertexCount: number,
  resourceLedger: ImportResourceLedger,
  allocationPath: string,
): PackedSkinInfluences {
  if (!Number.isSafeInteger(vertexCount) || vertexCount < 0) {
    throw new RangeError(
      `[vitrum/gltf-adapter] ${allocationPath} vertex count must be a non-negative safe integer.`,
    );
  }
  let skinInfluencesPerVertex = 4;

  for (let v = 0; v < vertexCount; v++) {
    const sortedInfluences = mergedSkinInfluencesForVertex(sets, v);
    skinInfluencesPerVertex = Math.max(skinInfluencesPerVertex, sortedInfluences.length);
  }

  const outputAllocation = geometryArrayAllocation(
    [vertexCount, skinInfluencesPerVertex],
    Uint32Array.BYTES_PER_ELEMENT,
    allocationPath,
  );
  resourceLedger.chargeDecodedGeometryBytes(
    checkedGeometryProduct([outputAllocation.byteLength, 2], `${allocationPath} byte length`),
    allocationPath,
  );
  const skinIndices = new Uint32Array(outputAllocation.elementCount);
  const skinWeights = new Float32Array(outputAllocation.elementCount);
  for (let v = 0; v < vertexCount; v += 1) {
    const influences = mergedSkinInfluencesForVertex(sets, v);
    const sum = influences.reduce((acc, [, weight]) => acc + weight, 0);
    if (sum <= 0) continue;
    for (let lane = 0; lane < influences.length; lane += 1) {
      const [joint, weight] = influences[lane]!;
      const offset = v * skinInfluencesPerVertex + lane;
      skinIndices[offset] = joint;
      skinWeights[offset] = weight / sum;
    }
  }

  return { skinIndices, skinWeights, skinInfluencesPerVertex };
}

function mergedSkinInfluencesForVertex(
  sets: readonly SkinInfluenceSet[],
  vertex: number,
): Array<readonly [joint: number, weight: number]> {
  const merged = new Map<number, number>();
  for (const set of sets) {
    const base = vertex * 4;
    for (let lane = 0; lane < 4; lane += 1) {
      const weight = set.weights[base + lane] ?? 0;
      if (!Number.isFinite(weight) || weight <= 0) continue;
      const joint = set.joints[base + lane] ?? 0;
      merged.set(joint, (merged.get(joint) ?? 0) + weight);
    }
  }
  return [...merged.entries()].sort(
    ([jointA, weightA], [jointB, weightB]) => weightB - weightA || jointA - jointB,
  );
}

function _unpackJoints(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
  accessorIndex: number,
  onAccessorDiagnostic: (diagnostic: GltfAccessorDiagnostic) => void,
  semantic: string,
  resourceLedger: ImportResourceLedger,
): Uint32Array {
  const accessor = gltf.accessors?.[accessorIndex];
  if (!accessor) {
    throw new Error(`[vitrum/gltf-adapter] Accessor ${accessorIndex} not found`);
  }
  if (accessor.type !== 'VEC4') {
    throw new Error(
      `[vitrum/gltf-adapter] ${semantic} accessor must be VEC4, got "${accessor.type}"`,
    );
  }
  if (validateAccessorNormalization(accessor, `accessors[${accessorIndex}]`)) {
    throw new TypeError(`[vitrum/gltf-adapter] ${semantic} accessor must not be normalized.`);
  }
  const ct = accessor.componentType;
  if (ct !== GltfComponentType.UNSIGNED_BYTE && ct !== GltfComponentType.UNSIGNED_SHORT) {
    throw new Error(
      `[vitrum/gltf-adapter] ${semantic} componentType must be UNSIGNED_BYTE or UNSIGNED_SHORT, ` +
        `got ${ct}`,
    );
  }

  const count = accessor.count;
  const resultAllocation = geometryArrayAllocation(
    [count, 4],
    Uint32Array.BYTES_PER_ELEMENT,
    `accessors[${accessorIndex}] ${semantic}`,
  );
  resourceLedger.chargeDecodedGeometryBytes(
    resultAllocation.byteLength,
    `accessors[${accessorIndex}] ${semantic}`,
  );
  const result = new Uint32Array(resultAllocation.elementCount);

  const compSize = ct === GltfComponentType.UNSIGNED_BYTE ? 1 : 2;

  if (accessor.bufferView !== undefined) {
    const bvIdx = accessor.bufferView;
    const bv = gltf.bufferViews?.[bvIdx];
    if (!bv) throw new Error(`[vitrum/gltf-adapter] BufferView ${bvIdx} not found`);

    const buf = buffers.get(bv.buffer);
    if (!buf) {
      throw new Error(`[vitrum/gltf-adapter] Buffer ${bv.buffer} is not available (${semantic}).`);
    }

    const bvOffset = bv.byteOffset ?? 0;
    const range = accessorBufferViewRange(accessor, bv, 4);
    validateBufferViewAccess(buf, bvIdx, bv, range.requiredByteLength, `${semantic} accessor`);
    if (bvOffset % compSize !== 0) {
      throw new Error(
        `[vitrum/gltf-adapter] ${semantic} bufferView ${bvIdx}.byteOffset ${bvOffset} ` +
          `is not aligned to component size ${compSize}.`,
      );
    }
    const dataView = new DataView(buf, bvOffset, bv.byteLength);

    for (let i = 0; i < count; i++) {
      const base = range.byteOffset + i * range.byteStride;
      for (let c = 0; c < 4; c++) {
        result[i * 4 + c] = readJointComponent(dataView, base + c * compSize, ct);
      }
    }
  }

  if (accessor.sparse) {
    applySparseJointPatch(
      gltf,
      buffers,
      accessorIndex,
      accessor,
      result,
      onAccessorDiagnostic,
      semantic,
    );
  }

  return result;
}

function applySparseJointPatch(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
  accessorIndex: number,
  accessor: GltfAccessor,
  result: Uint32Array,
  onAccessorDiagnostic: (diagnostic: GltfAccessorDiagnostic) => void,
  semantic: string,
): void {
  const sparse = accessor.sparse!;
  if (!Number.isSafeInteger(sparse.count) || sparse.count <= 0 || sparse.count > accessor.count) {
    throw new RangeError(
      `[vitrum/gltf-adapter] ${semantic} sparse.count must be a positive safe integer no greater than accessor.count.`,
    );
  }
  onAccessorDiagnostic({
    severity: 'warning',
    code: 'sparse-accessor-applied',
    path: `accessors[${accessorIndex}].sparse`,
    message: `[vitrum/gltf-adapter] ${semantic} accessor uses sparse storage (count=${sparse.count}); applying patch.`,
    accessorIndex,
  });

  const indicesBufferView = gltf.bufferViews?.[sparse.indices.bufferView];
  if (!indicesBufferView) {
    const message = '[vitrum/gltf-adapter] Sparse indices bufferView not found; accessor rejected.';
    onAccessorDiagnostic({
      severity: 'error',
      code: 'sparse-indices-buffer-view-not-found',
      path: `accessors[${accessorIndex}].sparse.indices.bufferView`,
      message,
      accessorIndex,
      bufferViewIndex: sparse.indices.bufferView,
    });
    throw new Error(message);
  }
  const indicesBuffer = buffers.get(indicesBufferView.buffer);
  if (!indicesBuffer) {
    const message = `[vitrum/gltf-adapter] Sparse indices buffer ${indicesBufferView.buffer} unavailable; accessor rejected.`;
    onAccessorDiagnostic({
      severity: 'error',
      code: 'sparse-indices-buffer-unavailable',
      path: `accessors[${accessorIndex}].sparse.indices.bufferView`,
      message,
      accessorIndex,
      bufferViewIndex: sparse.indices.bufferView,
      bufferIndex: indicesBufferView.buffer,
    });
    throw new Error(message);
  }

  const valuesBufferView = gltf.bufferViews?.[sparse.values.bufferView];
  if (!valuesBufferView) {
    const message = '[vitrum/gltf-adapter] Sparse values bufferView not found; accessor rejected.';
    onAccessorDiagnostic({
      severity: 'error',
      code: 'sparse-values-buffer-view-not-found',
      path: `accessors[${accessorIndex}].sparse.values.bufferView`,
      message,
      accessorIndex,
      bufferViewIndex: sparse.values.bufferView,
    });
    throw new Error(message);
  }
  const valuesBuffer = buffers.get(valuesBufferView.buffer);
  if (!valuesBuffer) {
    const message = `[vitrum/gltf-adapter] Sparse values buffer ${valuesBufferView.buffer} unavailable; accessor rejected.`;
    onAccessorDiagnostic({
      severity: 'error',
      code: 'sparse-values-buffer-unavailable',
      path: `accessors[${accessorIndex}].sparse.values.bufferView`,
      message,
      accessorIndex,
      bufferViewIndex: sparse.values.bufferView,
      bufferIndex: valuesBufferView.buffer,
    });
    throw new Error(message);
  }

  const sparseIndexComponentType = sparse.indices.componentType;
  if (
    sparseIndexComponentType !== GltfComponentType.UNSIGNED_BYTE &&
    sparseIndexComponentType !== GltfComponentType.UNSIGNED_SHORT &&
    sparseIndexComponentType !== GltfComponentType.UNSIGNED_INT
  ) {
    const message =
      `[vitrum/gltf-adapter] Sparse indices componentType ${sparseIndexComponentType} is invalid; ` +
      'expected UNSIGNED_BYTE, UNSIGNED_SHORT, or UNSIGNED_INT. Accessor rejected.';
    onAccessorDiagnostic({
      severity: 'error',
      code: 'invalid-sparse-indices-component-type',
      path: `accessors[${accessorIndex}].sparse.indices.componentType`,
      message,
      accessorIndex,
      componentType: sparseIndexComponentType,
    });
    throw new Error(message);
  }

  const indexCompSize =
    sparseIndexComponentType === GltfComponentType.UNSIGNED_BYTE
      ? 1
      : sparseIndexComponentType === GltfComponentType.UNSIGNED_SHORT
        ? 2
        : 4;
  const valueCompSize = accessor.componentType === GltfComponentType.UNSIGNED_BYTE ? 1 : 2;
  const indexByteOffset = sparse.indices.byteOffset ?? 0;
  const valueByteOffset = sparse.values.byteOffset ?? 0;
  const requiredIndexBytes = checkedGeometrySum(
    [
      indexByteOffset,
      checkedGeometryProduct(
        [sparse.count, indexCompSize],
        `accessors[${accessorIndex}] sparse ${semantic} index byte length`,
      ),
    ],
    `accessors[${accessorIndex}] sparse ${semantic} index range`,
  );
  const requiredValueBytes = checkedGeometrySum(
    [
      valueByteOffset,
      checkedGeometryProduct(
        [sparse.count, 4, valueCompSize],
        `accessors[${accessorIndex}] sparse ${semantic} value byte length`,
      ),
    ],
    `accessors[${accessorIndex}] sparse ${semantic} value range`,
  );
  if (indexByteOffset % indexCompSize !== 0) {
    throw new Error(
      `[vitrum/gltf-adapter] sparse ${semantic} indices byteOffset ${indexByteOffset} ` +
        `is not aligned to component size ${indexCompSize}.`,
    );
  }
  if (valueByteOffset % valueCompSize !== 0) {
    throw new Error(
      `[vitrum/gltf-adapter] sparse ${semantic} values byteOffset ${valueByteOffset} ` +
        `is not aligned to component size ${valueCompSize}.`,
    );
  }
  try {
    validateBufferViewAccess(
      indicesBuffer,
      sparse.indices.bufferView,
      indicesBufferView,
      requiredIndexBytes,
      `sparse ${semantic} indices`,
    );
    validateBufferViewAccess(
      valuesBuffer,
      sparse.values.bufferView,
      valuesBufferView,
      requiredValueBytes,
      `sparse ${semantic} values`,
    );
  } catch (error) {
    const message = `${String(error)} Accessor rejected.`;
    onAccessorDiagnostic({
      severity: 'error',
      code: String(error).includes('indices')
        ? 'sparse-indices-buffer-view-truncated'
        : 'sparse-values-buffer-view-truncated',
      path: String(error).includes('indices')
        ? `accessors[${accessorIndex}].sparse.indices.bufferView`
        : `accessors[${accessorIndex}].sparse.values.bufferView`,
      message,
      accessorIndex,
      bufferViewIndex: String(error).includes('indices')
        ? sparse.indices.bufferView
        : sparse.values.bufferView,
    });
    throw new Error(message, { cause: error });
  }
  const indexBufferViewOffset = indicesBufferView.byteOffset ?? 0;
  const valueBufferViewOffset = valuesBufferView.byteOffset ?? 0;
  if (indexBufferViewOffset % indexCompSize !== 0) {
    throw new Error(
      `[vitrum/gltf-adapter] sparse ${semantic} indices bufferView ` +
        `${sparse.indices.bufferView}.byteOffset ${indexBufferViewOffset} is not aligned ` +
        `to component size ${indexCompSize}.`,
    );
  }
  if (valueBufferViewOffset % valueCompSize !== 0) {
    throw new Error(
      `[vitrum/gltf-adapter] sparse ${semantic} values bufferView ` +
        `${sparse.values.bufferView}.byteOffset ${valueBufferViewOffset} is not aligned ` +
        `to component size ${valueCompSize}.`,
    );
  }
  const indexView = new DataView(
    indicesBuffer,
    indexBufferViewOffset,
    indicesBufferView.byteLength,
  );
  const valueView = new DataView(valuesBuffer, valueBufferViewOffset, valuesBufferView.byteLength);

  let previousIndex = -1;
  for (let s = 0; s < sparse.count; s += 1) {
    const jointIndex = readSparseIndexComponent(
      indexView,
      indexByteOffset + s * indexCompSize,
      sparseIndexComponentType,
    );
    if (jointIndex < 0 || jointIndex >= accessor.count || jointIndex <= previousIndex) {
      const isOrderingViolation = jointIndex <= previousIndex;
      const message = isOrderingViolation
        ? `[vitrum/gltf-adapter] Sparse indices must be strictly increasing; ${jointIndex} follows ${previousIndex}.`
        : `[vitrum/gltf-adapter] Sparse index ${jointIndex} is outside accessor count ${accessor.count}.`;
      onAccessorDiagnostic({
        severity: 'error',
        code: isOrderingViolation
          ? 'sparse-indices-not-strictly-increasing'
          : 'sparse-index-out-of-range',
        path: `accessors[${accessorIndex}].sparse.indices[${s}]`,
        message,
        accessorIndex,
        sparseEntryIndex: s,
      });
      throw new Error(message);
    }
    previousIndex = jointIndex;
    for (let c = 0; c < 4; c += 1) {
      result[jointIndex * 4 + c] = readJointComponent(
        valueView,
        valueByteOffset + (s * 4 + c) * valueCompSize,
        accessor.componentType,
      );
    }
  }
}

function readSparseIndexComponent(
  view: DataView,
  byteOffset: number,
  componentType: GltfComponentType,
): number {
  if (componentType === GltfComponentType.UNSIGNED_BYTE) return view.getUint8(byteOffset);
  if (componentType === GltfComponentType.UNSIGNED_SHORT) return view.getUint16(byteOffset, true);
  return view.getUint32(byteOffset, true);
}

function readJointComponent(
  view: DataView,
  byteOffset: number,
  componentType: GltfComponentType,
): number {
  return componentType === GltfComponentType.UNSIGNED_BYTE
    ? view.getUint8(byteOffset)
    : view.getUint16(byteOffset, true);
}
