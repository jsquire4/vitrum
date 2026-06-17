// gltfToScene.ts — Top-level glTF 2.0 → @vitrum/core Scene converter.
//
// Entry point: gltfToScene(input, opts) → Promise<{ scene, warnings }>
//
// Design constraints:
//   - Zero runtime dependencies (hand-rolled GLB parsing + accessor unpacking).
//   - Browser + Node compatible (no DOM-only APIs in core path; image decode is
//     pluggable via opts.decodeImage).
//   - Honest scope: v1 supports the core profile; exclusions are documented in
//     README.md and emitted as per-file warnings.
//
// Supported:
//   - Skins → SkinnedMeshPrimitive when a mesh node binds `skin` and the
//     primitive provides both JOINTS_0 + WEIGHTS_0 (JOINTS_0 u8/u16,
//     WEIGHTS_0 float/u8/u16, inverseBindMatrices, rest-pose joint transforms
//     converted into the skinned mesh node's local space).
//   - Morph targets → SkinnedMeshPrimitive.morphTargets / morphTargetNormals /
//     morphTargetTangents / morphWeights (POSITION + NORMAL + TANGENT deltas;
//     node/mesh weights; unskinned morphed meshes are promoted with a synthesized
//     identity skeleton).
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
//   - KHR_draco_mesh_compression / EXT/KHR_meshopt_compression → resolved via
//     HOST-SUPPLIED decoder hooks (opts.dracoDecode / opts.meshoptDecode; the
//     package bundles no decoder). Without a hook the spec fallback is used
//     when present (Draco fallback accessors / meshopt fallback buffer);
//     extensionsRequired without a hook or fallback throws. See compression.ts.
//
// Out of scope: cameras,
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
  type AnimationClip,
  type MaterialSpec,
  type Mat4,
  type Scene,
  type SceneEmitter,
  type ScenePrimitive,
  type TextureRef,
} from '@vitrum/core';
import type {
  GltfJson,
  GltfNode,
  GltfPrimitive,
  KhrLightsPunctualRoot,
} from './gltfTypes.js';
import { GltfComponentType } from './gltfTypes.js';
import {
  buildTextureHandleMap,
  GLTF_TEXTURE_SOURCE_EXTENSIONS,
  type DecodeImageFn,
  type GltfImageBytesMap,
  type GltfTextureAcquisitionDiagnosticCode,
  type GltfTextureSourceExtension,
} from './textures.js';
import { parseGlb } from './glbParser.js';
import {
  unpackAccessorFloat,
  unpackAccessorUint32,
  type GltfAccessorDiagnostic,
  type GltfAccessorDiagnosticCode,
} from './accessors.js';
import { buildWorldTransforms, composeTrsMat4, mat4Invert, mat4Mul } from './transforms.js';
import {
  convertMaterial,
  GLTF_DEFAULT_MATERIAL,
  type GltfMaterialDiagnostic,
  type GltfMaterialDiagnosticCode,
} from './materials.js';
import { generateFlatNormals } from './normals.js';
import { generateTangents } from './tangents.js';
import { animationNodeId, convertAnimations, type GltfAnimationImportDiagnosticCode } from './animations.js';
import { resolveCompression } from './compression.js';
import type { DracoDecodeFn, MeshoptDecodeFn } from './compression.js';
import {
  GLTF_MODE_TRIANGLE_FAN,
  GLTF_MODE_TRIANGLE_STRIP,
  sequentialIndices,
  triangulateTopology,
} from './triangulation.js';
import {
  buildPointLineFallbackGeometry,
  isPointLineMode,
  pointLineModeName,
} from './primitiveModeFallback.js';

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

type MaterialTextureRefField = typeof MATERIAL_TEXTURE_REF_FIELDS[number];

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
  // Accessor unpacking already converts BYTE/SHORT normalized attributes to
  // float32, which is the representation contract KHR_mesh_quantization needs.
  'KHR_mesh_quantization',
  'KHR_texture_transform',
  'EXT_mesh_gpu_instancing',
]);

export interface GltfToSceneOptions {
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
   *     Result is an `ImageBitmap`, accepted by pt-webgpu and pt-webgl2.
   *   - Non-browser: returns `{ kind: 'raw-image', mimeType, data: Uint8Array }`.
   *     pt-webgpu and pt-webgl2 require an ImageBitmap / canvas-compatible handle;
   *     supply this callback when running outside a browser.
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
   * Host-supplied Draco decode hook for `KHR_draco_mesh_compression`.
   *
   * The adapter bundles NO decoder; wire one from `draco3d` /
   * `DracoDecoderModule` (see README "Compressed geometry"). Receives the
   * compressed blob plus the extension's semantic → Draco-attribute-unique-id
   * map; must return decoded typed arrays whose counts/types match the
   * primitive's declared accessors (which per spec describe the decoded
   * data — the adapter then applies its standard accessor conversion,
   * including `normalized` handling). May be sync or async.
   *
   * Without this hook: primitives with uncompressed fallback accessors import
   * from the fallback (warn); otherwise the primitive is skipped (warn), or
   * an Error is thrown when the extension is in `extensionsRequired`.
   */
  readonly dracoDecode?: DracoDecodeFn;

  /**
   * Host-supplied meshopt decode hook for `EXT_meshopt_compression`.
   *
   * Mirrors `MeshoptDecoder.decodeGltfBuffer` from `meshoptimizer` (see README
   * "Compressed geometry"): receives the compressed bytes plus the extension's
   * `count` / `byteStride` / `mode` / `filter` and must return exactly
   * `count × byteStride` decoded bytes. Decoding happens at bufferView
   * resolution, so geometry, animation and image consumers all transparently
   * see decompressed data. May be sync or async.
   *
   * Without this hook: bufferViews whose underlying buffer is a real (non
   * `fallback: true`) uncompressed fallback are read directly (warn);
   * otherwise dependent accessors fail and their primitives are skipped
   * (warn), or an Error is thrown when the extension is in
   * `extensionsRequired`.
   */
  readonly meshoptDecode?: MeshoptDecodeFn;

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
   * Primitive provenance for `EXT_mesh_gpu_instancing`. Each local instance
   * matrix is the accessor-authored TRS before the node world transform is
   * applied. `GltfSceneController` uses these to update `InstancedMeshPrimitive`
   * instance matrices when a node or ancestor animates.
   */
  readonly instancingBindings?: ReadonlyArray<GltfInstancingBinding>;
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
}

export interface GltfInstancingBinding {
  readonly primitiveId: string;
  readonly nodeIndex: number;
  readonly localInstanceTransforms: ReadonlyArray<Mat4>;
}

export type GltfImportDiagnosticCode =
  | GltfTextureAcquisitionDiagnosticCode
  | GltfAnimationImportDiagnosticCode
  | GltfAccessorDiagnosticCode
  | GltfMaterialDiagnosticCode
  | 'unsupported-version'
  | 'unsupported-required-extension'
  | 'ignored-camera'
  | 'double-sided-material'
  | 'generated-tangents'
  | 'missing-tangent-texcoord'
  | 'tangent-generation-failed'
  | 'skin-rest-pose'
  | 'ignored-skin-attributes'
  | 'incomplete-skin-attributes'
  | 'scene-not-found'
  | 'ignored-gpu-instancing'
  | 'unsupported-primitive-mode'
  | 'fallback-generated-primitive-mode'
  | 'unresolved-compression'
  | 'missing-position'
  | 'unreadable-position'
  | 'unreadable-indices'
  | 'ignored-vertex-color-set'
  | 'empty-triangulated-primitive'
  | 'ignored-material-texcoord'
  | 'ignored-morph-target-texcoord';

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

function emitImportDiagnostic(
  warnings: string[],
  diagnostics: GltfImportDiagnostic[],
  diagnostic: GltfImportDiagnostic,
): void {
  diagnostics.push(diagnostic);
  warnings.push(diagnostic.message);
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
export async function gltfToScene(
  input: ArrayBuffer | GltfJson,
  opts: GltfToSceneOptions = {},
): Promise<GltfToSceneResult> {
  const warnings: string[] = [];

  // ── 1. Parse input ─────────────────────────────────────────────────────────
  let gltf: GltfJson;
  const buffers = new Map<number, ArrayBuffer>();

  // Seed from opts.buffers.
  if (opts.buffers) {
    if (opts.buffers instanceof Map) {
      for (const [k, v] of opts.buffers as Map<number, ArrayBuffer>) buffers.set(k, v);
    } else {
      for (const [k, v] of Object.entries(opts.buffers as Record<string, ArrayBuffer>)) {
        buffers.set(Number(k), v);
      }
    }
  }

  if (input instanceof ArrayBuffer) {
    // Detect GLB by magic bytes.
    if (input.byteLength >= 4) {
      const magic = new DataView(input).getUint32(0, true);
      if (magic === 0x46546c67) {
        const glb = parseGlb(input);
        gltf = glb.json;
        if (glb.binChunk !== undefined && !buffers.has(0)) {
          buffers.set(0, glb.binChunk);
        }
      } else {
        // Treat as UTF-8 JSON.
        const text = new TextDecoder().decode(input);
        gltf = JSON.parse(text) as GltfJson;
      }
    } else {
      throw new Error('[vitrum/gltf-adapter] Input ArrayBuffer is too small to be a valid glTF');
    }
  } else {
    gltf = input;
  }

  const diagnostics: GltfImportDiagnostic[] = [];
  const onAccessorDiagnostic = (diagnostic: GltfAccessorDiagnostic): void => {
    emitImportDiagnostic(warnings, diagnostics, diagnostic);
  };
  const onMaterialDiagnostic = (diagnostic: GltfMaterialDiagnostic): void => {
    emitImportDiagnostic(warnings, diagnostics, diagnostic);
  };

  // ── 2. Validate version ────────────────────────────────────────────────────
  const version = gltf.asset?.version;
  if (version && !version.startsWith('2.')) {
    emitImportDiagnostic(warnings, diagnostics, {
      severity: 'warning',
      code: 'unsupported-version',
      path: 'asset.version',
      message:
        `[vitrum/gltf-adapter] glTF asset version is "${version}"; only 2.x is supported. ` +
        'Conversion will proceed but results may be incorrect.',
    });
  }

  const requiredExtensions = gltf.extensionsRequired ?? [];
  for (let i = 0; i < requiredExtensions.length; i += 1) {
    const ext = requiredExtensions[i]!;
    if (!isRequiredExtensionSupported(ext, opts.textureSourceExtensions)) {
      const message =
        `[vitrum/gltf-adapter] extensionsRequired includes unsupported extension "${ext}". ` +
        'Required glTF extensions cannot be safely ignored.';
      throw new GltfImportError(message, [{
        severity: 'error',
        code: 'unsupported-required-extension',
        path: `extensionsRequired[${i}]`,
        message,
      }]);
    }
  }

  // ── 3. Warn on out-of-scope top-level features ─────────────────────────────
  if (gltf.cameras && (gltf.cameras).length > 0) {
    for (const [cameraIndex] of gltf.cameras.entries()) {
      emitImportDiagnostic(warnings, diagnostics, {
        severity: 'warning',
        code: 'ignored-camera',
        path: `cameras[${cameraIndex}]`,
        message:
          '[vitrum/gltf-adapter] Camera nodes are present but ignored (cameras are not part of the ' +
          '@vitrum/core Scene contract; pass camera data via FrameInput instead).',
      });
    }
  }
  if (gltf.skins && gltf.skins.length > 0) {
    emitImportDiagnostic(warnings, diagnostics, {
      severity: 'warning',
      code: 'skin-rest-pose',
      path: 'skins',
      message:
        `[vitrum/gltf-adapter] This glTF has ${gltf.skins.length} skin(s). ` +
        'Skinned nodes are imported as SkinnedMeshPrimitive at rest pose. ' +
        'The engine does not advance clips itself: drive the pose host-side by ' +
        'sampling the imported animations (sampleAnimationClip), rebuilding bone ' +
        'matrices, and re-running solveSkin.',
    });
  }

  const extUsed = gltf.extensionsUsed ?? [];

  // ── 3.5. Resolve compressed geometry (GLTF-02) ─────────────────────────────
  // KHR_draco_mesh_compression + EXT_meshopt_compression via the host-supplied
  // opts.dracoDecode / opts.meshoptDecode hooks. Runs BEFORE texture/accessor
  // reads so every downstream consumer sees decompressed bufferViews. Returns
  // a clone when compression is present (the caller's GltfJson is never
  // mutated); throws when a hook-less required extension has no spec fallback.
  gltf = await resolveCompression(
    gltf,
    buffers,
    { dracoDecode: opts.dracoDecode, meshoptDecode: opts.meshoptDecode },
    warnings,
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
  );

  // ── 5. Pre-convert materials ───────────────────────────────────────────────
  const coreMaterials = (gltf.materials ?? []).map((m, materialIndex) =>
    convertMaterial(
      m,
      handleMap,
      warnings,
      gltf,
      materialIndex,
      opts.textureSourceExtensions,
      onMaterialDiagnostic,
    ),
  );
  for (const [materialIndex, material] of (gltf.materials ?? []).entries()) {
    if (material.doubleSided !== true) continue;
    emitImportDiagnostic(warnings, diagnostics, {
      severity: 'warning',
      code: 'double-sided-material',
      path: `materials[${materialIndex}].doubleSided`,
      message:
        `[vitrum/gltf-adapter] Material "${material.name ?? materialIndex}" sets doubleSided=true. ` +
        'The flag is preserved at MaterialSpec.extensions.doubleSided for host/backend inspection, ' +
        'but @vitrum/core has no first-class two-sided/backface-normal material contract; ' +
        'backend compatibility may report an approximate doubleSided row.',
    });
  }
  const selectedMaterialVariant = _resolveMaterialVariantSelection(
    gltf,
    opts.materialVariant,
    warnings,
  );

  // ── 6. Pick the target scene ───────────────────────────────────────────────
  const sceneIndex = opts.sceneIndex ?? gltf.scene ?? 0;
  const gltfScene = gltf.scenes?.[sceneIndex];
  const rootNodes = gltfScene?.nodes ?? [];

  if (gltf.scenes && !gltfScene) {
    emitImportDiagnostic(warnings, diagnostics, {
      severity: 'warning',
      code: 'scene-not-found',
      path: `scenes[${sceneIndex}]`,
      message:
        `[vitrum/gltf-adapter] Scene index ${sceneIndex} not found (total: ${gltf.scenes.length}). ` +
        'Falling back to an empty scene.',
    });
  }

  // ── 7. Build world transforms for all nodes ────────────────────────────────
  const worldTransforms = buildWorldTransforms(gltf, rootNodes);

  // ── 8. Flatten node → mesh → primitives ───────────────────────────────────
  const primitives: ScenePrimitive[] = [];
  const animationTargets: Record<string, string[]> = {};
  const materialVariantBindings: GltfMaterialVariantBinding[] = [];
  const instancingBindings: GltfInstancingBinding[] = [];
  let primIdCounter = 0;

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
    );
    let instanceFallbackWarned = false;

    const mesh = gltfMeshes[node.mesh];
    if (!mesh) continue;

    // ── Skin data for this node (if any) ──────────────────────────────────
    // glTF 2.0 §3.8: a node may reference a skin by index. All primitives in
    // the node's mesh share the same skin.
    const skinData = _extractSkinData(
      gltf, buffers, gltfSkins, node.skin, worldMat, worldTransforms, warnings, onAccessorDiagnostic,
    );
    const { bones, boneInverses } = skinData ?? {};

    for (const [primitiveIndex, prim] of mesh.primitives.entries()) {
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
        emitImportDiagnostic(warnings, diagnostics, {
          severity: 'warning',
          code: 'unsupported-primitive-mode',
          path: `meshes[${node.mesh}].primitives[${primitiveIndex}].mode`,
          message:
            `[vitrum/gltf-adapter] Mesh "${mesh.name ?? node.mesh}" primitive has unsupported ` +
            `mode ${mode} (${pointLineModeName(mode)}). Supported modes are POINTS (0), ` +
            'LINES (1), LINE_LOOP (2), LINE_STRIP (3), TRIANGLES (4), ' +
            'TRIANGLE_STRIP (5) and TRIANGLE_FAN (6). This primitive is SKIPPED.',
        });
        continue;
      }

      // Compression left UNRESOLVED by resolveCompression (no hook + no spec
      // fallback, or the decode hook failed) — skip honestly.
      const primExtKeys = Object.keys(prim.extensions ?? {});
      if (primExtKeys.includes('KHR_draco_mesh_compression')) {
        emitImportDiagnostic(warnings, diagnostics, {
          severity: 'warning',
          code: 'unresolved-compression',
          path: `meshes[${node.mesh}].primitives[${primitiveIndex}].extensions.KHR_draco_mesh_compression`,
          message:
            `[vitrum/gltf-adapter] Mesh "${mesh.name ?? node.mesh}" primitive has unresolved ` +
            'KHR_draco_mesh_compression geometry (no opts.dracoDecode hook / decode failed, ' +
            'and no uncompressed fallback). Primitive SKIPPED.',
        });
        continue;
      }

      // ── Unpack attributes ──────────────────────────────────────────────────
      const posIdx = prim.attributes['POSITION'];
      if (posIdx === undefined) {
        emitImportDiagnostic(warnings, diagnostics, {
          severity: 'warning',
          code: 'missing-position',
          path: `meshes[${node.mesh}].primitives[${primitiveIndex}].attributes.POSITION`,
          message:
            `[vitrum/gltf-adapter] Mesh "${mesh.name ?? node.mesh}" primitive has no POSITION ` +
            'attribute. Primitive SKIPPED.',
        });
        continue;
      }

      let positions: Float32Array;
      try {
        positions = unpackAccessorFloat(gltf, buffers, posIdx, warnings, onAccessorDiagnostic);
      } catch (e) {
        emitImportDiagnostic(warnings, diagnostics, {
          severity: 'warning',
          code: 'unreadable-position',
          path: `meshes[${node.mesh}].primitives[${primitiveIndex}].attributes.POSITION`,
          message:
            `[vitrum/gltf-adapter] Failed to read POSITION for mesh "${mesh.name ?? node.mesh}": ` +
            String(e) + ' Primitive SKIPPED.',
        });
        continue;
      }

      // Indices (optional).
      let indices: Uint32Array | undefined;
      if (prim.indices !== undefined) {
        try {
          indices = unpackAccessorUint32(gltf, buffers, prim.indices, null, onAccessorDiagnostic);
        } catch (e) {
          emitImportDiagnostic(warnings, diagnostics, {
            severity: 'warning',
            code: 'unreadable-indices',
            path: `meshes[${node.mesh}].primitives[${primitiveIndex}].indices`,
            message:
              `[vitrum/gltf-adapter] Failed to read indices for mesh "${mesh.name ?? node.mesh}": ` +
              String(e) + ' Primitive SKIPPED.',
          });
          continue;
        }
      }

      // TRIANGLE_STRIP / TRIANGLE_FAN → indexed triangle list (GLTF-05).
      // Works for indexed and non-indexed inputs; degenerate (repeated-index)
      // triangles are dropped per glTF §3.7.2.1 winding rules.
      if (mode === GLTF_MODE_TRIANGLE_STRIP || mode === GLTF_MODE_TRIANGLE_FAN) {
        const src = indices ?? sequentialIndices(positions.length / 3);
        const tris = triangulateTopology(src, mode);
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
          continue;
        }
        indices = tris;
      }

      // Normals — generate flat normals if absent or unreadable.
      const normIdx = prim.attributes['NORMAL'];
      const normAttempt = _tryUnpackFloat(
        gltf, buffers, normIdx,
        `NORMAL for mesh "${mesh.name ?? node.mesh}"`, warnings, onAccessorDiagnostic,
      );
      if (normAttempt === undefined && normIdx === undefined) {
        warnings.push(
          `[vitrum/gltf-adapter] Mesh "${mesh.name ?? node.mesh}" has no NORMAL attribute. ` +
            'Generating flat normals.',
        );
      } else if (normAttempt === undefined) {
        warnings.push(
          `[vitrum/gltf-adapter] NORMAL unreadable for mesh "${mesh.name ?? node.mesh}". ` +
            'Generating flat normals.',
        );
      }
      let normals: Float32Array = normAttempt ?? generateFlatNormals(positions, indices);

      // UVs — optional.
      let uvs = _tryUnpackFloat(
        gltf, buffers, prim.attributes['TEXCOORD_0'],
        `TEXCOORD_0 for "${mesh.name ?? node.mesh}"`, warnings, onAccessorDiagnostic,
      );
      let uv1 = _tryUnpackFloat(
        gltf, buffers, prim.attributes['TEXCOORD_1'],
        `TEXCOORD_1 for "${mesh.name ?? node.mesh}"`, warnings, onAccessorDiagnostic,
      );

      // Tangents — optional (xyzw per vertex).
      let tangents = _tryUnpackFloat(
        gltf, buffers, prim.attributes['TANGENT'],
        `TANGENT for "${mesh.name ?? node.mesh}"`, warnings, onAccessorDiagnostic,
      );

      // Vertex colors — optional (COLOR_0).
      let colors = _tryUnpackFloat(
        gltf, buffers, prim.attributes['COLOR_0'],
        `COLOR_0 for "${mesh.name ?? node.mesh}"`, warnings, onAccessorDiagnostic,
      );
      for (const attrName of Object.keys(prim.attributes).sort()) {
        if (/^COLOR_[1-9][0-9]*$/.test(attrName)) {
          emitImportDiagnostic(warnings, diagnostics, {
            severity: 'warning',
            code: 'ignored-vertex-color-set',
            path: `${primitivePath}.attributes.${attrName}`,
            message:
              `[vitrum/gltf-adapter] Mesh "${mesh.name ?? node.mesh}" primitive includes ${attrName}, ` +
              'but the core Scene contract currently imports only COLOR_0. This secondary vertex-color set is ignored.',
          });
        }
      }

      // ── Skinning attributes ────────────────────────────────────────────────
      // Only unpacked when this node has a skin. JOINTS_0 / WEIGHTS_0 without
      // node.skin carry no glTF skinning semantics, but report the ignored data
      // so strict one-call loading can reject the degradation before rendering.
      let skinIndices: Uint32Array | undefined;
      let skinWeights: Float32Array | undefined;
      const jointsIdx = prim.attributes['JOINTS_0'];
      const weightsIdx = prim.attributes['WEIGHTS_0'];
      if (!skinData && (jointsIdx !== undefined || weightsIdx !== undefined)) {
        emitImportDiagnostic(warnings, diagnostics, {
          severity: 'warning',
          code: 'ignored-skin-attributes',
          path: jointsIdx !== undefined
            ? `${primitivePath}.attributes.JOINTS_0`
            : `${primitivePath}.attributes.WEIGHTS_0`,
          message:
            `[vitrum/gltf-adapter] Mesh "${mesh.name ?? node.mesh}" primitive includes ` +
            'JOINTS_0/WEIGHTS_0 data, but the node does not bind a skin. ' +
            'Skin attributes are ignored and the primitive is imported as a static mesh.',
        });
      }
      if (skinData && bones && boneInverses) {
        if (jointsIdx !== undefined && weightsIdx !== undefined) {
          try {
            skinIndices = _unpackJoints(gltf, buffers, jointsIdx);
          } catch (e) {
            warnings.push(
              `[vitrum/gltf-adapter] Failed to read JOINTS_0 for "${mesh.name ?? node.mesh}": ` +
                String(e) + '. Falling back to static mesh.',
            );
          }
          try {
            skinWeights = unpackAccessorFloat(gltf, buffers, weightsIdx, warnings, onAccessorDiagnostic);
          } catch (e) {
            warnings.push(
              `[vitrum/gltf-adapter] Failed to read WEIGHTS_0 for "${mesh.name ?? node.mesh}": ` +
                String(e) + '. Falling back to static mesh.',
            );
            skinIndices = undefined; // don't emit skinned if weights failed
          }
        } else {
          emitImportDiagnostic(warnings, diagnostics, {
            severity: 'warning',
            code: 'incomplete-skin-attributes',
            path: `${primitivePath}.attributes.${jointsIdx === undefined ? 'JOINTS_0' : 'WEIGHTS_0'}`,
            message:
              `[vitrum/gltf-adapter] Mesh "${mesh.name ?? node.mesh}" node binds a skin, ` +
              'but the primitive does not provide both JOINTS_0 and WEIGHTS_0. ' +
              'Skinning is omitted and the primitive is imported as a static mesh.',
          });
        }
      }

      // Warn on unsupported primitive attributes.
      for (const attrName of Object.keys(prim.attributes)) {
        if (
          !['POSITION', 'NORMAL', 'TEXCOORD_0', 'TEXCOORD_1', 'TANGENT', 'COLOR_0',
            'JOINTS_0', 'WEIGHTS_0'].includes(attrName) &&
          !attrName.startsWith('TEXCOORD_') &&
          !attrName.startsWith('COLOR_')
        ) {
          warnings.push(
            `[vitrum/gltf-adapter] Unknown primitive attribute "${attrName}" in mesh ` +
              `"${mesh.name ?? node.mesh}" is ignored.`,
          );
        }
      }

      // Morph targets (GLTF-04) — POSITION/NORMAL/TANGENT deltas + node/mesh weights.
      let morph = _extractMorphTargets(
        gltf, buffers, prim.targets, node.weights ?? mesh.weights,
        positions.length / 3, `${mesh.name ?? node.mesh}`, primitivePath, warnings, diagnostics, onAccessorDiagnostic,
      );

      // Material.
      const materialIndex = _resolvePrimitiveMaterialIndex(
        gltf,
        prim,
        prim.material,
        selectedMaterialVariant,
        warnings,
        `${mesh.name ?? node.mesh}`,
      );
      const material =
        materialIndex !== undefined && materialIndex < coreMaterials.length
          ? (coreMaterials[materialIndex] ?? GLTF_DEFAULT_MATERIAL)
          : GLTF_DEFAULT_MATERIAL;
      const uvResolvedMaterial = _resolvePrimitiveUvMaterial(
        gltf,
        buffers,
        prim,
        material,
        uv1,
        warnings,
        diagnostics,
        primitivePath,
        mesh.name ?? node.mesh,
        onAccessorDiagnostic,
      );
      uv1 = uvResolvedMaterial.uv1;
      if (isPointLineMode(mode)) {
        const originalVertexCount = Math.floor(positions.length / 3);
        const fallback = buildPointLineFallbackGeometry(
          positions,
          indices,
          mode,
          opts.pointLineFallbackRadius,
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
          continue;
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
        uvs = remapVec2Attribute(uvs, fallback.sourceVertices);
        uv1 = remapVec2Attribute(uv1, fallback.sourceVertices);
        colors = remapVertexColors(colors, originalVertexCount, fallback.sourceVertices);
        tangents = undefined;
        if (skinIndices && skinWeights) {
          skinIndices = remapVec4UintAttribute(skinIndices, fallback.sourceVertices);
          skinWeights = remapVec4FloatAttribute(skinWeights, fallback.sourceVertices);
        }
        morph = remapMorphData(morph, fallback.sourceVertices);
        positions = fallback.positions;
        normals = fallback.normals;
        indices = fallback.indices;
      }
      const finalTangents = tangents ?? _maybeGenerateTangents(
        positions,
        normals,
        uvs,
        uv1,
        indices,
        uvResolvedMaterial.material,
        `${mesh.name ?? node.mesh}`,
        warnings,
        diagnostics,
        primitivePath,
      );

      const id = `gltf-prim-${primIdCounter++}`;
      (animationTargets[animationNodeId(nodeIdx)] ??= []).push(id);
      if ((prim.extensions?.KHR_materials_variants?.mappings?.length ?? 0) > 0) {
        materialVariantBindings.push({
          primitiveId: id,
          meshIndex: node.mesh,
          primitiveIndex,
          ...(prim.material !== undefined ? { baseMaterialIndex: prim.material } : {}),
        });
      }

      let skinArg = (skinIndices && skinWeights && bones && boneInverses)
        ? {
            skinIndices,
            skinWeights,
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
        const identitySkinWeights = new Float32Array(vertexCount * 4);
        for (let v = 0; v < vertexCount; v++) identitySkinWeights[v * 4] = 1;
        const identityBone = new Float32Array(16);
        identityBone[0] = 1; identityBone[5] = 1; identityBone[10] = 1; identityBone[15] = 1;
        skinArg = {
          skinIndices: new Uint32Array(vertexCount * 4), // all bone 0
          skinWeights: identitySkinWeights,
          bones: identityBone,
          boneInverses: new Float32Array(identityBone),
        };
        warnings.push(
          `[vitrum/gltf-adapter] Mesh "${mesh.name ?? node.mesh}" has morph targets but no skin; ` +
            'promoted to SkinnedMeshPrimitive with a synthesized identity skeleton (1 bone). ' +
            'Drive morphWeights and re-solve via @vitrum/core solveSkin to animate the blend shapes.',
        );
      }

      let primitiveInstances = instanceTransforms?.worldInstanceTransforms;
      if (primitiveInstances && skinArg) {
        if (!instanceFallbackWarned) {
          emitImportDiagnostic(warnings, diagnostics, {
            severity: 'warning',
            code: 'ignored-gpu-instancing',
            path: `nodes[${nodeIdx}].extensions.EXT_mesh_gpu_instancing`,
            message:
              `[vitrum/gltf-adapter] Node "${node.name ?? nodeIdx}" uses EXT_mesh_gpu_instancing ` +
              'on a skinned or morphed mesh. @vitrum/core has no instanced skinned/morphed primitive ' +
              'contract yet, so the mesh is imported once with its normal skin/morph representation.',
          });
          instanceFallbackWarned = true;
        }
        primitiveInstances = undefined;
      }
      if (primitiveInstances && instanceTransforms) {
        instancingBindings.push({
          primitiveId: id,
          nodeIndex: nodeIdx,
          localInstanceTransforms: instanceTransforms.localInstanceTransforms,
        });
      }

      primitives.push(_buildPrimitive(
        id, worldMat, positions, normals, indices,
        uvs, uvResolvedMaterial.uv1, finalTangents, colors, uvResolvedMaterial.material, skinArg, morph, primitiveInstances,
      ));
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
  let emitterIdCounter = 0;

  if (extUsed.includes('KHR_lights_punctual')) {
    const rootExt = gltf.extensions?.['KHR_lights_punctual'] as KhrLightsPunctualRoot | undefined;
    const lights = rootExt?.lights ?? [];

    for (const [nodeIdx, worldMat] of worldTransforms) {
      const node = gltfNodes[nodeIdx];
      if (!node?.extensions) continue;

      const nodeLightRef = node.extensions['KHR_lights_punctual'] as
        | { light: number }
        | undefined;
      if (nodeLightRef === undefined || typeof nodeLightRef.light !== 'number') continue;

      const light = lights[nodeLightRef.light];
      if (!light) {
        warnings.push(
          `[vitrum/gltf-adapter] Node ${nodeIdx} references KHR_lights_punctual light ` +
            `index ${nodeLightRef.light} which does not exist. Emitter skipped.`,
        );
        continue;
      }

      const id = `gltf-light-${emitterIdCounter++}`;
      const emitter = _convertPunctualLight(light, worldMat, id, warnings);
      if (emitter) emitters.push(emitter);
    }
  }

  // ── 10. Convert animations (GLTF-03) ───────────────────────────────────────
  const animations = convertAnimations(gltf, buffers, warnings, (diagnostic) => {
    diagnostics.push(diagnostic);
  }, onAccessorDiagnostic);

  const scene: Scene = {
    primitives,
    emitters,
    environment: { kind: 'none' },
  };

  return {
    scene,
    animations,
    animationTargets,
    convertedMaterials: coreMaterials,
    materialVariantBindings,
    instancingBindings,
    warnings,
    diagnostics,
  };
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
): number | undefined {
  if (selector === undefined) return undefined;
  const variants = gltf.extensions?.KHR_materials_variants?.variants ?? [];
  if (typeof selector === 'number') {
    if (Number.isInteger(selector) && selector >= 0 && selector < variants.length) return selector;
    warnings.push(
      `[vitrum/gltf-adapter] materialVariant index ${selector} was requested, ` +
        `but this asset declares ${variants.length} variant(s). Base materials are used.`,
    );
    return undefined;
  }
  const index = variants.findIndex((variant) => variant.name === selector);
  if (index >= 0) return index;
  warnings.push(
    `[vitrum/gltf-adapter] materialVariant "${selector}" was requested, but no ` +
      'KHR_materials_variants entry with that name exists. Base materials are used.',
  );
  return undefined;
}

function _resolvePrimitiveMaterialIndex(
  gltf: GltfJson,
  primitive: GltfPrimitive,
  baseMaterialIndex: number | undefined,
  selectedVariantIndex: number | undefined,
  warnings: string[],
  meshLabel: string,
): number | undefined {
  if (selectedVariantIndex === undefined) return baseMaterialIndex;
  const mappings = primitive.extensions?.KHR_materials_variants?.mappings ?? [];
  const mapping = mappings.find((candidate) => candidate.variants.includes(selectedVariantIndex));
  if (!mapping) return baseMaterialIndex;
  if (mapping.material < 0 || mapping.material >= (gltf.materials?.length ?? 0)) {
    warnings.push(
      `[vitrum/gltf-adapter] Mesh "${meshLabel}" KHR_materials_variants mapping for ` +
        `variant ${selectedVariantIndex} references missing material ${mapping.material}. ` +
        'Base material is used.',
    );
    return baseMaterialIndex;
  }
  return mapping.material;
}

function _maybeGenerateTangents(
  positions: Float32Array,
  normals: Float32Array,
  uvs: Float32Array | undefined,
  uv1: Float32Array | undefined,
  indices: Uint32Array | undefined,
  material: MaterialSpec,
  meshLabel: string,
  warnings: string[],
  diagnostics: GltfImportDiagnostic[],
  primitivePath: string,
): Float32Array | undefined {
  if (!materialNeedsTangentFrame(material)) return undefined;
  const tangentUvSet = tangentFrameTexCoord(material);
  if (tangentUvSet == null) {
    emitImportDiagnostic(warnings, diagnostics, {
      severity: 'warning',
      code: 'missing-tangent-texcoord',
      path: `${primitivePath}.attributes.TANGENT`,
      message:
        `[vitrum/gltf-adapter] Mesh "${meshLabel}" uses tangent-space material maps ` +
        'on multiple UV channels, but @vitrum/core carries one generated tangent frame per primitive. ' +
        'Tangents could not be generated; provide authored TANGENT data for this asset.',
    });
    return undefined;
  }
  if (tangentUvSet > 1) {
    emitImportDiagnostic(warnings, diagnostics, {
      severity: 'warning',
      code: 'missing-tangent-texcoord',
      path: `${primitivePath}.attributes.TEXCOORD_${tangentUvSet}`,
      message:
        `[vitrum/gltf-adapter] Mesh "${meshLabel}" uses a tangent-space material map ` +
        `on TEXCOORD_${tangentUvSet}, but @vitrum/core currently imports only TEXCOORD_0 ` +
        'and TEXCOORD_1. Tangents could not be generated; normal-map-like texture(s) may be ignored or approximate.',
    });
    return undefined;
  }
  const tangentUvs = tangentUvSet === 1 ? uv1 : uvs;
  if (!tangentUvs) {
    emitImportDiagnostic(warnings, diagnostics, {
      severity: 'warning',
      code: 'missing-tangent-texcoord',
      path: `${primitivePath}.attributes.TEXCOORD_${tangentUvSet}`,
      message:
        `[vitrum/gltf-adapter] Mesh "${meshLabel}" uses a tangent-space material map ` +
        `on TEXCOORD_${tangentUvSet}, but the primitive has no TEXCOORD_${tangentUvSet}. ` +
        'Tangents could not be generated; normal-map-like texture(s) may be ignored or approximate.',
    });
    return undefined;
  }
  const generated = generateTangents(positions, normals, tangentUvs, indices);
  if (!generated) {
    emitImportDiagnostic(warnings, diagnostics, {
      severity: 'warning',
      code: 'tangent-generation-failed',
      path: `${primitivePath}.attributes.TANGENT`,
      message:
        `[vitrum/gltf-adapter] Mesh "${meshLabel}" uses a tangent-space material map ` +
        `but tangents could not be generated from POSITION/NORMAL/TEXCOORD_${tangentUvSet}.`,
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
  return material.normalMap !== undefined ||
    material.clearcoatNormalMap !== undefined ||
    material.bumpMap !== undefined;
}

function tangentFrameTexCoord(material: MaterialSpec): number | null {
  const candidates = [
    material.normalMap?.texCoord,
    material.clearcoatNormalMap?.texCoord,
    material.bumpMap?.texCoord,
  ].filter((texCoord): texCoord is number => texCoord !== undefined);
  if (candidates.length === 0) return 0;
  const channels = new Set(candidates.map((texCoord) => Math.max(0, Math.floor(texCoord))));
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
): MeshGpuInstancingTransforms | undefined {
  const extension = node.extensions?.EXT_mesh_gpu_instancing;
  if (extension === undefined) return undefined;
  const pathBase = `nodes[${nodeIdx}].extensions.EXT_mesh_gpu_instancing`;
  const attributes = isObject(extension) ? extension.attributes : undefined;
  if (!isObject(extension) || !isObject(attributes)) {
    emitImportDiagnostic(warnings, diagnostics, {
      severity: 'warning',
      code: 'ignored-gpu-instancing',
      path: pathBase,
      message:
        `[vitrum/gltf-adapter] Node "${node.name ?? nodeIdx}" uses EXT_mesh_gpu_instancing ` +
        'without an attributes object. The base mesh is imported once with the node transform.',
    });
    return undefined;
  }

  const attrs = attributes;
  let failed = false;
  let instanceCount: number | undefined;

  const fail = (path: string, message: string): void => {
    failed = true;
    emitImportDiagnostic(warnings, diagnostics, {
      severity: 'warning',
      code: 'ignored-gpu-instancing',
      path,
      message,
    });
  };

  const readAccessor = (
    semantic: keyof typeof GPU_INSTANCE_ATTRIBUTE_SPECS,
  ): Float32Array | undefined => {
    const rawAccessorIndex = attrs[semantic];
    if (rawAccessorIndex === undefined) return undefined;
    const attrPath = `${pathBase}.attributes.${semantic}`;
    if (typeof rawAccessorIndex !== 'number' || !Number.isInteger(rawAccessorIndex) || rawAccessorIndex < 0) {
      fail(
        attrPath,
        `[vitrum/gltf-adapter] Node "${node.name ?? nodeIdx}" EXT_mesh_gpu_instancing ` +
          `${semantic} must reference a non-negative accessor index. The base mesh is imported once.`,
      );
      return undefined;
    }
    const accessorIndex = rawAccessorIndex;
    const accessor = gltf.accessors?.[accessorIndex];
    const spec = GPU_INSTANCE_ATTRIBUTE_SPECS[semantic];
    if (!accessor) {
      fail(
        attrPath,
        `[vitrum/gltf-adapter] Node "${node.name ?? nodeIdx}" EXT_mesh_gpu_instancing ` +
          `${semantic} references missing accessor ${accessorIndex}. The base mesh is imported once.`,
      );
      return undefined;
    }
    if (accessor.type !== spec.type) {
      fail(
        attrPath,
        `[vitrum/gltf-adapter] Node "${node.name ?? nodeIdx}" EXT_mesh_gpu_instancing ` +
          `${semantic} accessor must be ${spec.type}, got ${accessor.type}. The base mesh is imported once.`,
      );
      return undefined;
    }
    if (instanceCount === undefined) {
      instanceCount = accessor.count;
    } else if (accessor.count !== instanceCount) {
      fail(
        attrPath,
        `[vitrum/gltf-adapter] Node "${node.name ?? nodeIdx}" EXT_mesh_gpu_instancing ` +
          `${semantic} count ${accessor.count} does not match instance count ${instanceCount}. ` +
          'The base mesh is imported once.',
      );
      return undefined;
    }
    try {
      return unpackAccessorFloat(gltf, buffers, accessorIndex, warnings, onAccessorDiagnostic);
    } catch (e) {
      fail(
        attrPath,
        `[vitrum/gltf-adapter] Failed to read EXT_mesh_gpu_instancing ${semantic} for ` +
          `node "${node.name ?? nodeIdx}": ${String(e)} The base mesh is imported once.`,
      );
      return undefined;
    }
  };

  const translations = readAccessor('TRANSLATION');
  const rotations = readAccessor('ROTATION');
  const scales = readAccessor('SCALE');

  for (const key of Object.keys(attrs)) {
    if (key in GPU_INSTANCE_ATTRIBUTE_SPECS) continue;
    warnings.push(
      `[vitrum/gltf-adapter] Node "${node.name ?? nodeIdx}" EXT_mesh_gpu_instancing ` +
        `attribute "${key}" is custom/non-transform data and is ignored.`,
    );
  }

  if (failed) return undefined;
  if (instanceCount === undefined || instanceCount <= 0) {
    emitImportDiagnostic(warnings, diagnostics, {
      severity: 'warning',
      code: 'ignored-gpu-instancing',
      path: `${pathBase}.attributes`,
      message:
        `[vitrum/gltf-adapter] Node "${node.name ?? nodeIdx}" uses EXT_mesh_gpu_instancing ` +
        'without TRANSLATION, ROTATION, or SCALE accessors. The base mesh is imported once.',
    });
    return undefined;
  }

  const worldInstanceTransforms: Mat4[] = [];
  const localInstanceTransforms: Mat4[] = [];
  for (let i = 0; i < instanceCount; i += 1) {
    const t: [number, number, number] = translations
      ? [
          translations[i * 3 + 0] ?? 0,
          translations[i * 3 + 1] ?? 0,
          translations[i * 3 + 2] ?? 0,
        ]
      : [0, 0, 0];
    const r: [number, number, number, number] = rotations
      ? [
          rotations[i * 4 + 0] ?? 0,
          rotations[i * 4 + 1] ?? 0,
          rotations[i * 4 + 2] ?? 0,
          rotations[i * 4 + 3] ?? 1,
        ]
      : [0, 0, 0, 1];
    const s: [number, number, number] = scales
      ? [
          scales[i * 3 + 0] ?? 1,
          scales[i * 3 + 1] ?? 1,
          scales[i * 3 + 2] ?? 1,
        ]
      : [1, 1, 1];
    const local = asMat4(composeTrsMat4(t, r, s));
    localInstanceTransforms.push(local);
    worldInstanceTransforms.push(asMat4(mat4Mul(worldMat, local)));
  }
  return { worldInstanceTransforms, localInstanceTransforms };
}

function isObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/**
 * Attempt to unpack a float accessor, returning `undefined` and appending a
 * warning on failure.  Used for optional attributes (NORMAL, TEXCOORD_*, etc.)
 * that the caller may substitute or skip when missing.
 */
function _tryUnpackFloat(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
  accessorIndex: number | undefined,
  label: string,
  warnings: string[],
  onAccessorDiagnostic: (diagnostic: GltfAccessorDiagnostic) => void,
): Float32Array | undefined {
  if (accessorIndex === undefined) return undefined;
  try {
    return unpackAccessorFloat(gltf, buffers, accessorIndex, warnings, onAccessorDiagnostic);
  } catch (e) {
    warnings.push(`[vitrum/gltf-adapter] Failed to read ${label}: ${String(e)}`);
    return undefined;
  }
}

interface SkinData {
  bones: Float32Array;
  boneInverses: Float32Array;
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
  onAccessorDiagnostic: (diagnostic: GltfAccessorDiagnostic) => void,
): SkinData | undefined {
  if (skinIdx === undefined) return undefined;
  const gltfSkin = gltfSkins[skinIdx];
  if (!gltfSkin) return undefined;
  const meshWorldInverse = mat4Invert(meshWorld);
  if (!meshWorldInverse) {
    warnings.push(
      `[vitrum/gltf-adapter] Skinned mesh node for skin "${gltfSkin.name ?? skinIdx}" has a ` +
        'singular world transform. Skinning was omitted for this mesh.',
    );
    return undefined;
  }

  const jointCount = gltfSkin.joints.length;
  const bones = new Float32Array(jointCount * 16);
  for (let j = 0; j < jointCount; j++) {
    const jointNodeIdx = gltfSkin.joints[j]!;
    const jointWorld = worldTransforms.get(jointNodeIdx);
    if (jointWorld) {
      const jointMeshLocal = mat4Mul(meshWorldInverse, jointWorld);
      bones.set(jointMeshLocal, j * 16);
    } else {
      // Joint node not in the scene graph — use identity in mesh-local space.
      bones[j * 16 + 0] = 1; bones[j * 16 + 5] = 1;
      bones[j * 16 + 10] = 1; bones[j * 16 + 15] = 1;
    }
  }

  let boneInverses: Float32Array | undefined;
  if (gltfSkin.inverseBindMatrices !== undefined) {
    try {
      boneInverses = unpackAccessorFloat(
        gltf,
        buffers,
        gltfSkin.inverseBindMatrices,
        warnings,
        onAccessorDiagnostic,
      );
    } catch (e) {
      warnings.push(
        `[vitrum/gltf-adapter] Failed to read inverseBindMatrices for skin "${gltfSkin.name ?? skinIdx}": ` +
          String(e) + '. Using identity inverses.',
      );
    }
  }
  if (!boneInverses || boneInverses.length < jointCount * 16) {
    boneInverses = new Float32Array(jointCount * 16);
    for (let j = 0; j < jointCount; j++) {
      boneInverses[j * 16 + 0] = 1; boneInverses[j * 16 + 5] = 1;
      boneInverses[j * 16 + 10] = 1; boneInverses[j * 16 + 15] = 1;
    }
  }

  return { bones, boneInverses };
}

interface PrimitiveUvMaterialResolution {
  readonly material: MaterialSpec;
  readonly uv1?: Float32Array;
}

function _isTextureRef(value: unknown): value is TextureRef {
  return value !== null && typeof value === 'object' && 'handle' in value;
}

function _cloneMaterialWithTextureRef(
  material: MaterialSpec,
  field: MaterialTextureRefField,
  ref: TextureRef,
): MaterialSpec {
  return {
    ...material,
    [field]: ref,
  } as MaterialSpec;
}

function _cloneMaterialWithoutTextureRefs(
  material: MaterialSpec,
  fields: readonly MaterialTextureRefField[],
): MaterialSpec {
  const next: Record<string, unknown> = { ...material };
  for (const field of fields) delete next[field];
  return next as unknown as MaterialSpec;
}

function _resolvePrimitiveUvMaterial(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
  primitive: GltfPrimitive,
  material: MaterialSpec,
  uv1: Float32Array | undefined,
  warnings: string[],
  diagnostics: GltfImportDiagnostic[],
  primitivePath: string,
  meshName: string | number,
  onAccessorDiagnostic: (diagnostic: GltfAccessorDiagnostic) => void,
): PrimitiveUvMaterialResolution {
  const highFields: { field: MaterialTextureRefField; ref: TextureRef; texCoord: number }[] = [];
  const highTexCoords = new Set<number>();
  let usesUv1 = false;

  for (const field of MATERIAL_TEXTURE_REF_FIELDS) {
    const ref = material[field];
    if (!_isTextureRef(ref)) continue;
    const texCoord = Math.max(0, Math.floor(ref.texCoord ?? 0));
    if (texCoord === 1) {
      usesUv1 = true;
    } else if (texCoord > 1) {
      highFields.push({ field, ref, texCoord });
      highTexCoords.add(texCoord);
    }
  }

  if (highFields.length === 0) return { material, ...(uv1 ? { uv1 } : {}) };

  if (highTexCoords.size === 1 && !usesUv1) {
    const texCoord = [...highTexCoords][0]!;
    const attrName = `TEXCOORD_${texCoord}`;
    const remapUv = _tryUnpackFloat(
      gltf,
      buffers,
      primitive.attributes[attrName],
      `${attrName} for "${meshName}"`,
      warnings,
      onAccessorDiagnostic,
    );
    if (remapUv !== undefined) {
      let remapped = material;
      for (const { field, ref } of highFields) {
        remapped = _cloneMaterialWithTextureRef(remapped, field, { ...ref, texCoord: 1 });
      }
      return { material: remapped, uv1: remapUv };
    }
  }

  const dropFields = highFields.map(({ field }) => field);
  const dropped = _cloneMaterialWithoutTextureRefs(material, dropFields);
  const conflictReason = highTexCoords.size > 1
    ? `material references multiple high UV sets (${[...highTexCoords].sort((a, b) => a - b).map((n) => `TEXCOORD_${n}`).join(', ')})`
    : usesUv1
      ? 'material already references TEXCOORD_1, so the high UV set cannot be losslessly remapped into uv1'
      : `primitive has no readable TEXCOORD_${[...highTexCoords][0] ?? '?'} accessor`;

  for (const { field, texCoord } of highFields) {
    emitImportDiagnostic(warnings, diagnostics, {
      severity: 'warning',
      code: 'ignored-material-texcoord',
      path: `${primitivePath}.material`,
      message:
        `[vitrum/gltf-adapter] Mesh "${meshName}" material field "${field}" references ` +
        `TEXCOORD_${texCoord}, but ${conflictReason}. The texture field is ignored ` +
        'for this primitive instead of being sampled with the wrong UV channel.',
    });
  }
  return { material: dropped, ...(uv1 ? { uv1 } : {}) };
}

function remapVec2Attribute(
  attr: Float32Array | undefined,
  sourceVertices: Uint32Array,
): Float32Array | undefined {
  if (attr == null) return undefined;
  const out = new Float32Array(sourceVertices.length * 2);
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
): Float32Array | undefined {
  if (colors == null || sourceVertexCount <= 0) return undefined;
  const components = Math.max(1, Math.floor(colors.length / sourceVertexCount));
  const out = new Float32Array(sourceVertices.length * components);
  for (let i = 0; i < sourceVertices.length; i += 1) {
    const src = sourceVertices[i]! * components;
    const dst = i * components;
    for (let c = 0; c < components; c += 1) out[dst + c] = colors[src + c] ?? (c === 3 ? 1 : 0);
  }
  return out;
}

function remapVec4UintAttribute(values: Uint32Array, sourceVertices: Uint32Array): Uint32Array {
  const out = new Uint32Array(sourceVertices.length * 4);
  for (let i = 0; i < sourceVertices.length; i += 1) {
    const src = sourceVertices[i]! * 4;
    out.set(values.subarray(src, src + 4), i * 4);
  }
  return out;
}

function remapVec4FloatAttribute(values: Float32Array, sourceVertices: Uint32Array): Float32Array {
  const out = new Float32Array(sourceVertices.length * 4);
  for (let i = 0; i < sourceVertices.length; i += 1) {
    const src = sourceVertices[i]! * 4;
    out.set(values.subarray(src, src + 4), i * 4);
  }
  return out;
}

function remapMorphData(
  morph: MorphData | undefined,
  sourceVertices: Uint32Array,
): MorphData | undefined {
  if (morph == null) return undefined;
  return {
    morphTargets: morph.morphTargets.map((target) => remapVec3Attribute(target, sourceVertices)),
    ...(morph.morphTargetNormals != null
      ? { morphTargetNormals: morph.morphTargetNormals.map((target) => remapVec3Attribute(target, sourceVertices)) }
      : {}),
    ...(morph.morphTargetTangents != null
      ? { morphTargetTangents: morph.morphTargetTangents.map((target) => remapVec3Attribute(target, sourceVertices)) }
      : {}),
    morphWeights: new Float32Array(morph.morphWeights),
  };
}

function remapVec3Attribute(values: Float32Array, sourceVertices: Uint32Array): Float32Array {
  const out = new Float32Array(sourceVertices.length * 3);
  for (let i = 0; i < sourceVertices.length; i += 1) {
    const src = sourceVertices[i]! * 3;
    out[i * 3 + 0] = values[src + 0] ?? 0;
    out[i * 3 + 1] = values[src + 1] ?? 0;
    out[i * 3 + 2] = values[src + 2] ?? 0;
  }
  return out;
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
  tangents: Float32Array | undefined,
  colors: Float32Array | undefined,
  material: MaterialSpec,
  skin?: {
    skinIndices: Uint32Array;
    skinWeights: Float32Array;
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
    ...(tangents ? { tangents } : {}),
    ...(colors ? { colors } : {}),
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
      bones: skin.bones,
      boneInverses: skin.boneInverses,
      ...(skin.bindMatrix ? { bindMatrix: skin.bindMatrix } : {}),
      ...(skin.bindMatrixInverse ? { bindMatrixInverse: skin.bindMatrixInverse } : {}),
      ...(morph ? {
        morphTargets: morph.morphTargets,
        ...(morph.morphTargetNormals ? { morphTargetNormals: morph.morphTargetNormals } : {}),
        ...(morph.morphTargetTangents ? { morphTargetTangents: morph.morphTargetTangents } : {}),
        morphWeights: morph.morphWeights,
      } : {}),
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
  /** Initial per-target weights from `node.weights ?? mesh.weights` (zeros
   *  when neither is authored). */
  morphWeights: Float32Array;
}

/**
 * Parse glTF `primitive.targets` into core morph-target delta arrays.
 *
 * glTF §3.7.2.2: each target maps attribute names to accessors carrying
 * DELTAS from the base attribute (sparse accessors are common here and are
 * handled by `unpackAccessorFloat`). POSITION, NORMAL, and TANGENT deltas map
 * onto `SkinnedMeshPrimitive.morphTargets` / `.morphTargetNormals` /
 * `.morphTargetTangents`. TANGENT deltas are preserved for host/backend
 * inspection; current solvers still derive posed tangents from solved geometry.
 *
 * Returns `undefined` when the primitive has no targets.
 */
function _extractMorphTargets(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
  targets: ReadonlyArray<Record<string, number>> | undefined,
  authoredWeights: number[] | undefined,
  vertexCount: number,
  meshLabel: string,
  primitivePath: string,
  warnings: string[],
  diagnostics: GltfImportDiagnostic[],
  onAccessorDiagnostic: (diagnostic: GltfAccessorDiagnostic) => void,
): MorphData | undefined {
  if (!targets || targets.length === 0) return undefined;
  const tCount = targets.length;

  const morphTargets: Float32Array[] = [];
  const normalDeltas: (Float32Array | null)[] = [];
  const tangentDeltas: (Float32Array | null)[] = [];
  let anyNormals = false;
  let anyTangents = false;

  for (let t = 0; t < tCount; t++) {
    const target = targets[t]!;

    // POSITION delta.
    let posDelta = _tryUnpackFloat(
      gltf, buffers, target['POSITION'],
      `morph target ${t} POSITION for "${meshLabel}"`, warnings, onAccessorDiagnostic,
    );
    if (posDelta && posDelta.length !== vertexCount * 3) {
      warnings.push(
        `[vitrum/gltf-adapter] Morph target ${t} POSITION delta length ${posDelta.length} ` +
          `!= ${vertexCount * 3} for "${meshLabel}". Using zero deltas for this target.`,
      );
      posDelta = undefined;
    }
    morphTargets.push(posDelta ?? new Float32Array(vertexCount * 3));

    // NORMAL delta.
    let nrmDelta = _tryUnpackFloat(
      gltf, buffers, target['NORMAL'],
      `morph target ${t} NORMAL for "${meshLabel}"`, warnings, onAccessorDiagnostic,
    );
    if (nrmDelta && nrmDelta.length !== vertexCount * 3) {
      warnings.push(
        `[vitrum/gltf-adapter] Morph target ${t} NORMAL delta length ${nrmDelta.length} ` +
          `!= ${vertexCount * 3} for "${meshLabel}". Using zero deltas for this target.`,
      );
      nrmDelta = undefined;
    }
    if (nrmDelta) anyNormals = true;
    normalDeltas.push(nrmDelta ?? null);

    // TANGENT direction delta (glTF morph target TANGENT is VEC3; base tangent
    // handedness remains in the base TANGENT.w lane).
    let tanDelta = _tryUnpackFloat(
      gltf, buffers, target['TANGENT'],
      `morph target ${t} TANGENT for "${meshLabel}"`, warnings, onAccessorDiagnostic,
    );
    if (tanDelta && tanDelta.length !== vertexCount * 3) {
      warnings.push(
        `[vitrum/gltf-adapter] Morph target ${t} TANGENT delta length ${tanDelta.length} ` +
          `!= ${vertexCount * 3} for "${meshLabel}". Using zero deltas for this target.`,
      );
      tanDelta = undefined;
    }
    if (tanDelta) anyTangents = true;
    tangentDeltas.push(tanDelta ?? null);

    for (const attr of Object.keys(target)) {
      if (attr !== 'POSITION' && attr !== 'NORMAL' && attr !== 'TANGENT') {
        if (/^TEXCOORD_\d+$/.test(attr)) {
          emitImportDiagnostic(warnings, diagnostics, {
            severity: 'warning',
            code: 'ignored-morph-target-texcoord',
            path: `${primitivePath}.targets[${t}].${attr}`,
            message:
              `[vitrum/gltf-adapter] Morph target ${t} attribute "${attr}" in mesh ` +
              `"${meshLabel}" is ignored because @vitrum/core has no morph UV-delta lane; ` +
              'textured morph animation keeps the rest-pose UVs.',
          });
        } else {
          warnings.push(
            `[vitrum/gltf-adapter] Morph target ${t} attribute "${attr}" in mesh ` +
              `"${meshLabel}" is ignored.`,
          );
        }
      }
    }
  }

  // Initial weights: node-level overrides mesh-level per glTF §3.7.2.2;
  // absent weights default to 0 (rest pose).
  const morphWeights = new Float32Array(tCount);
  if (authoredWeights) {
    if (authoredWeights.length !== tCount) {
      warnings.push(
        `[vitrum/gltf-adapter] Mesh "${meshLabel}" morph weights length ` +
          `${authoredWeights.length} != target count ${tCount}; extra entries are ` +
          'dropped / missing entries default to 0.',
      );
    }
    for (let t = 0; t < tCount; t++) morphWeights[t] = authoredWeights[t] ?? 0;
  }

  return {
    morphTargets,
    ...(anyNormals
      ? { morphTargetNormals: normalDeltas.map(n => n ?? new Float32Array(vertexCount * 3)) }
      : {}),
    ...(anyTangents
      ? { morphTargetTangents: tangentDeltas.map(t => t ?? new Float32Array(vertexCount * 3)) }
      : {}),
    morphWeights,
  };
}

/**
 * Convert one KHR_lights_punctual light (with its node world transform) to a
 * core SceneEmitter. Returns `null` if the light type is unsupported.
 */
function _convertPunctualLight(
  light: NonNullable<KhrLightsPunctualRoot['lights']>[number],
  worldMat: Mat4,
  id: string,
  warnings: string[],
): SceneEmitter | null {
  const color: [number, number, number] = light.color
    ? [light.color[0], light.color[1], light.color[2]]
    : [1, 1, 1];
  const intensity = light.intensity ?? 1;

  // Column-major 4×4: translation at indices 12, 13, 14.
  const px = worldMat[12] ?? 0;
  const py = worldMat[13] ?? 0;
  const pz = worldMat[14] ?? 0;

  // glTF lights point along -Z in local space; world -Z column is at indices 8,9,10.
  const lzx = -(worldMat[8] ?? 0);
  const lzy = -(worldMat[9] ?? 0);
  const lzz = -(worldMat[10] ?? 0);
  const lzLen = Math.hypot(lzx, lzy, lzz);
  const dirX = lzLen > 1e-10 ? lzx / lzLen : 0;
  const dirY = lzLen > 1e-10 ? lzy / lzLen : 0;
  const dirZ = lzLen > 1e-10 ? lzz / lzLen : 1;

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
      const outer = light.spot?.outerConeAngle ?? (Math.PI / 4);
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
      warnings.push(
        `[vitrum/gltf-adapter] KHR_lights_punctual light "${light.name ?? id}" ` +
          `has unsupported type "${(light as { type: string }).type}". Emitter skipped.`,
      );
      return null;
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
function _unpackJoints(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
  accessorIndex: number,
): Uint32Array {
  const accessor = gltf.accessors?.[accessorIndex];
  if (!accessor) {
    throw new Error(`[vitrum/gltf-adapter] Accessor ${accessorIndex} not found`);
  }
  if (accessor.type !== 'VEC4') {
    throw new Error(
      `[vitrum/gltf-adapter] JOINTS_0 accessor must be VEC4, got "${accessor.type}"`,
    );
  }
  const ct = accessor.componentType;
  if (ct !== GltfComponentType.UNSIGNED_BYTE && ct !== GltfComponentType.UNSIGNED_SHORT) {
    throw new Error(
      `[vitrum/gltf-adapter] JOINTS_0 componentType must be UNSIGNED_BYTE or UNSIGNED_SHORT, ` +
        `got ${ct}`,
    );
  }

  const count = accessor.count;
  const result = new Uint32Array(count * 4);

  if (accessor.bufferView === undefined) return result; // zero-initialized

  const bvIdx = accessor.bufferView;
  const bv = gltf.bufferViews?.[bvIdx];
  if (!bv) throw new Error(`[vitrum/gltf-adapter] BufferView ${bvIdx} not found`);

  const buf = buffers.get(bv.buffer);
  if (!buf) {
    throw new Error(
      `[vitrum/gltf-adapter] Buffer ${bv.buffer} is not available (JOINTS_0).`,
    );
  }

  const compSize = ct === GltfComponentType.UNSIGNED_BYTE ? 1 : 2;
  const bvOffset = bv.byteOffset ?? 0;
  const accOffset = accessor.byteOffset ?? 0;
  const stride = bv.byteStride ?? compSize * 4;
  const dataView = new DataView(buf, bvOffset + accOffset);

  for (let i = 0; i < count; i++) {
    const base = i * stride;
    for (let c = 0; c < 4; c++) {
      result[i * 4 + c] = ct === GltfComponentType.UNSIGNED_BYTE
        ? (dataView.getUint8(base + c))
        : (dataView.getUint16(base + c * 2, true));
    }
  }

  return result;
}
