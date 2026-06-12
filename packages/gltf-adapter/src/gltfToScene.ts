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
//   - Skins → SkinnedMeshPrimitive (JOINTS_0 u8/u16, WEIGHTS_0 float/u8/u16,
//     inverseBindMatrices, rest-pose joint world transforms, bindMatrix /
//     bindMatrixInverse from the skinned node transform).
//   - Morph targets → SkinnedMeshPrimitive.morphTargets / morphTargetNormals /
//     morphWeights (POSITION + NORMAL deltas; node/mesh weights; unskinned
//     morphed meshes are promoted with a synthesized identity skeleton).
//     TANGENT deltas are warn-skipped (core has no morph-tangent field).
//   - Animations → core AnimationClip[] on the result (LINEAR / STEP /
//     CUBICSPLINE; translation / rotation / scale / weights channels; channel
//     node ids are `gltf-node-<i>`, resolved to primitives via
//     result.animationTargets). Geometry is still imported at rest pose; the
//     host samples clips (sampleAnimationClip) and pushes updates.
//   - KHR_lights_punctual → SceneEmitter[] (point, spot, directional;
//     world-transform applied to position/direction).
//
//   - KHR_draco_mesh_compression / EXT_meshopt_compression → resolved via
//     HOST-SUPPLIED decoder hooks (opts.dracoDecode / opts.meshoptDecode; the
//     package bundles no decoder). Without a hook the spec fallback is used
//     when present (Draco fallback accessors / meshopt fallback buffer);
//     extensionsRequired without a hook or fallback throws. See compression.ts.
//
// Out of scope: cameras, morph TANGENT deltas,
//
// Primitive modes: TRIANGLES (4) is converted directly; TRIANGLE_STRIP (5) and
// TRIANGLE_FAN (6) are triangulated into indexed triangle lists (winding per
// glTF §3.7.2.1, degenerates dropped). POINTS/LINES/LINE_LOOP/LINE_STRIP emit
// a warning and the primitive is skipped (core has no point/line primitive).
//
// References:
//   - glTF 2.0 specification (Khronos Group)
//     https://www.khronos.org/registry/glTF/specs/2.0/glTF-2.0.html
//   - KHR_lights_punctual extension
//     https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_lights_punctual/README.md

import type {
  Scene, ScenePrimitive, SceneEmitter, MaterialSpec, Mat4, AnimationClip,
} from '@vitrum/core';
import type { GltfJson, GltfPrimitive, KhrLightsPunctualRoot } from './gltfTypes.js';
import { GltfComponentType } from './gltfTypes.js';
import {
  buildTextureHandleMap,
  GLTF_TEXTURE_SOURCE_EXTENSIONS,
  type DecodeImageFn,
  type GltfImageBytesMap,
  type GltfTextureSourceExtension,
} from './textures.js';
import { parseGlb } from './glbParser.js';
import { unpackAccessorFloat, unpackAccessorUint32 } from './accessors.js';
import { buildWorldTransforms } from './transforms.js';
import { convertMaterial, GLTF_DEFAULT_MATERIAL } from './materials.js';
import { generateFlatNormals } from './normals.js';
import { generateTangents } from './tangents.js';
import { animationNodeId, convertAnimations } from './animations.js';
import { resolveCompression } from './compression.js';
import type { DracoDecodeFn, MeshoptDecodeFn } from './compression.js';
import {
  GLTF_MODE_TRIANGLE_FAN,
  GLTF_MODE_TRIANGLE_STRIP,
  sequentialIndices,
  triangulateTopology,
} from './triangulation.js';

const GLTF_PRIMITIVE_MODE_TRIANGLES = 4;

const SUPPORTED_REQUIRED_EXTENSIONS = new Set([
  'KHR_draco_mesh_compression',
  'EXT_meshopt_compression',
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
  'KHR_texture_transform',
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
   *     local TRS from the sampled values, then push the new world transform
   *     through `engine.updatePrimitive(primId, { transform })` for each
   *     mapped primitive. NOTE: the adapter flattens the node hierarchy into
   *     world transforms at import; channels animating an ANCESTOR of a mesh
   *     node (or a joint node) have no mapped primitives — hosts that need
   *     full scene-graph animation must retain the GltfJson hierarchy and
   *     recompute world transforms themselves.
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
  /** Non-fatal issues encountered during conversion. Inspect these for skipped
   *  primitives, unsupported extensions, missing buffers, sparse patches, etc. */
  readonly warnings: string[];
}

export interface GltfMaterialVariantBinding {
  readonly primitiveId: string;
  readonly meshIndex: number;
  readonly primitiveIndex: number;
  readonly baseMaterialIndex?: number;
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

  // ── 2. Validate version ────────────────────────────────────────────────────
  const version = gltf.asset?.version;
  if (version && !version.startsWith('2.')) {
    warnings.push(
      `[vitrum/gltf-adapter] glTF asset version is "${version}"; only 2.x is supported. ` +
        'Conversion will proceed but results may be incorrect.',
    );
  }

  for (const ext of gltf.extensionsRequired ?? []) {
    if (!isRequiredExtensionSupported(ext, opts.textureSourceExtensions)) {
      throw new Error(
        `[vitrum/gltf-adapter] extensionsRequired includes unsupported extension "${ext}". ` +
          'Required glTF extensions cannot be safely ignored.',
      );
    }
  }

  // ── 3. Warn on out-of-scope top-level features ─────────────────────────────
  if (gltf.cameras && (gltf.cameras).length > 0) {
    warnings.push(
      '[vitrum/gltf-adapter] Camera nodes are present but ignored (cameras are not part of the ' +
        '@vitrum/core Scene contract; pass camera data via FrameInput instead).',
    );
  }
  if (gltf.skins && gltf.skins.length > 0) {
    warnings.push(
      `[vitrum/gltf-adapter] This glTF has ${gltf.skins.length} skin(s). ` +
        'Skinned nodes are imported as SkinnedMeshPrimitive at rest pose. ' +
        'The engine does not advance clips itself: drive the pose host-side by ' +
        'sampling the imported animations (sampleAnimationClip), rebuilding bone ' +
        'matrices, and re-running solveSkin.',
    );
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
  );

  // ── 5. Pre-convert materials ───────────────────────────────────────────────
  const coreMaterials = (gltf.materials ?? []).map((m) =>
    convertMaterial(m, handleMap, warnings, gltf),
  );
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
    warnings.push(
      `[vitrum/gltf-adapter] Scene index ${sceneIndex} not found (total: ${gltf.scenes.length}). ` +
        'Falling back to an empty scene.',
    );
  }

  // ── 7. Build world transforms for all nodes ────────────────────────────────
  const worldTransforms = buildWorldTransforms(gltf, rootNodes);

  // ── 8. Flatten node → mesh → primitives ───────────────────────────────────
  const primitives: ScenePrimitive[] = [];
  const animationTargets: Record<string, string[]> = {};
  const materialVariantBindings: GltfMaterialVariantBinding[] = [];
  let primIdCounter = 0;

  const gltfNodes = gltf.nodes ?? [];
  const gltfMeshes = gltf.meshes ?? [];
  const gltfSkins = gltf.skins ?? [];

  for (const [nodeIdx, worldMat] of worldTransforms) {
    const node = gltfNodes[nodeIdx];
    if (!node || node.mesh === undefined) continue;
    if (node.extensions?.EXT_mesh_gpu_instancing !== undefined) {
      warnings.push(
        `[vitrum/gltf-adapter] Node "${node.name ?? nodeIdx}" uses EXT_mesh_gpu_instancing, ` +
          'but this adapter does not import accessor-driven instance transforms yet. ' +
          'The base mesh is imported once with the node transform; instance attributes are ignored.',
      );
    }

    const mesh = gltfMeshes[node.mesh];
    if (!mesh) continue;

    // ── Skin data for this node (if any) ──────────────────────────────────
    // glTF 2.0 §3.8: a node may reference a skin by index. All primitives in
    // the node's mesh share the same skin.
    const skinData = _extractSkinData(
      gltf, buffers, gltfSkins, node.skin, worldTransforms, warnings,
    );
    const { bones, boneInverses } = skinData ?? {};

    for (const [primitiveIndex, prim] of mesh.primitives.entries()) {
      // Mode check — TRIANGLES (4, default), TRIANGLE_STRIP (5) and
      // TRIANGLE_FAN (6) are supported (strip/fan are triangulated into an
      // indexed triangle list below). Point/line modes are skipped: the core
      // Scene contract has no point/line primitive.
      const mode = prim.mode ?? GLTF_PRIMITIVE_MODE_TRIANGLES;
      if (
        mode !== GLTF_PRIMITIVE_MODE_TRIANGLES &&
        mode !== GLTF_MODE_TRIANGLE_STRIP &&
        mode !== GLTF_MODE_TRIANGLE_FAN
      ) {
        const modeNames: Record<number, string> = {
          0: 'POINTS', 1: 'LINES', 2: 'LINE_LOOP', 3: 'LINE_STRIP',
        };
        warnings.push(
          `[vitrum/gltf-adapter] Mesh "${mesh.name ?? node.mesh}" primitive has unsupported ` +
            `mode ${mode} (${modeNames[mode] ?? 'UNKNOWN'}). Only TRIANGLES (4), ` +
            'TRIANGLE_STRIP (5) and TRIANGLE_FAN (6) are supported (core has no ' +
            'point/line primitive). This primitive is SKIPPED.',
        );
        continue;
      }

      // Compression left UNRESOLVED by resolveCompression (no hook + no spec
      // fallback, or the decode hook failed) — skip honestly.
      const primExtKeys = Object.keys(prim.extensions ?? {});
      if (primExtKeys.includes('KHR_draco_mesh_compression')) {
        warnings.push(
          `[vitrum/gltf-adapter] Mesh "${mesh.name ?? node.mesh}" primitive has unresolved ` +
            'KHR_draco_mesh_compression geometry (no opts.dracoDecode hook / decode failed, ' +
            'and no uncompressed fallback). Primitive SKIPPED.',
        );
        continue;
      }

      // ── Unpack attributes ──────────────────────────────────────────────────
      const posIdx = prim.attributes['POSITION'];
      if (posIdx === undefined) {
        warnings.push(
          `[vitrum/gltf-adapter] Mesh "${mesh.name ?? node.mesh}" primitive has no POSITION ` +
            'attribute. Primitive SKIPPED.',
        );
        continue;
      }

      let positions: Float32Array;
      try {
        positions = unpackAccessorFloat(gltf, buffers, posIdx, warnings);
      } catch (e) {
        warnings.push(
          `[vitrum/gltf-adapter] Failed to read POSITION for mesh "${mesh.name ?? node.mesh}": ` +
            String(e) + ' Primitive SKIPPED.',
        );
        continue;
      }

      // Indices (optional).
      let indices: Uint32Array | undefined;
      if (prim.indices !== undefined) {
        try {
          indices = unpackAccessorUint32(gltf, buffers, prim.indices);
        } catch (e) {
          warnings.push(
            `[vitrum/gltf-adapter] Failed to read indices for mesh "${mesh.name ?? node.mesh}": ` +
              String(e) + ' Primitive SKIPPED.',
          );
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
          warnings.push(
            `[vitrum/gltf-adapter] Mesh "${mesh.name ?? node.mesh}" ` +
              `${mode === GLTF_MODE_TRIANGLE_STRIP ? 'TRIANGLE_STRIP' : 'TRIANGLE_FAN'} primitive ` +
              'yields no non-degenerate triangles. Primitive SKIPPED.',
          );
          continue;
        }
        indices = tris;
      }

      // Normals — generate flat normals if absent or unreadable.
      const normIdx = prim.attributes['NORMAL'];
      const normAttempt = _tryUnpackFloat(
        gltf, buffers, normIdx,
        `NORMAL for mesh "${mesh.name ?? node.mesh}"`, warnings,
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
      const normals: Float32Array = normAttempt ?? generateFlatNormals(positions, indices);

      // UVs — optional.
      const uvs = _tryUnpackFloat(
        gltf, buffers, prim.attributes['TEXCOORD_0'],
        `TEXCOORD_0 for "${mesh.name ?? node.mesh}"`, warnings,
      );
      const uv1 = _tryUnpackFloat(
        gltf, buffers, prim.attributes['TEXCOORD_1'],
        `TEXCOORD_1 for "${mesh.name ?? node.mesh}"`, warnings,
      );

      // Tangents — optional (xyzw per vertex).
      const tangents = _tryUnpackFloat(
        gltf, buffers, prim.attributes['TANGENT'],
        `TANGENT for "${mesh.name ?? node.mesh}"`, warnings,
      );

      // Vertex colors — optional (COLOR_0).
      const colors = _tryUnpackFloat(
        gltf, buffers, prim.attributes['COLOR_0'],
        `COLOR_0 for "${mesh.name ?? node.mesh}"`, warnings,
      );

      // ── Skinning attributes ────────────────────────────────────────────────
      // Only unpacked when this node has a skin; JOINTS_0 / WEIGHTS_0 without a
      // node.skin are silently ignored (they carry no semantics without a skin).
      let skinIndices: Uint32Array | undefined;
      let skinWeights: Float32Array | undefined;
      if (skinData && bones && boneInverses) {
        const jointsIdx = prim.attributes['JOINTS_0'];
        const weightsIdx = prim.attributes['WEIGHTS_0'];
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
            skinWeights = unpackAccessorFloat(gltf, buffers, weightsIdx, warnings);
          } catch (e) {
            warnings.push(
              `[vitrum/gltf-adapter] Failed to read WEIGHTS_0 for "${mesh.name ?? node.mesh}": ` +
                String(e) + '. Falling back to static mesh.',
            );
            skinIndices = undefined; // don't emit skinned if weights failed
          }
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

      // Morph targets (GLTF-04) — POSITION/NORMAL deltas + node/mesh weights.
      // TANGENT deltas are warn-skipped (core SkinnedMeshPrimitive has no
      // morph-tangent field).
      const morph = _extractMorphTargets(
        gltf, buffers, prim.targets, node.weights ?? mesh.weights,
        positions.length / 3, `${mesh.name ?? node.mesh}`, warnings,
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
      const finalTangents = tangents ?? _maybeGenerateTangents(
        positions, normals, uvs, indices, material, `${mesh.name ?? node.mesh}`, warnings,
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

      // glTF §3.8: the skinned mesh node transform is the mesh bind matrix for
      // the imported rest pose. Preserve it so the core skin solver can return
      // mesh-local positions before the primitive transform is applied.
      const bindMatrixInverse = (skinIndices && skinWeights && bones && boneInverses)
        ? _invertMat4(worldMat)
        : null;
      if (skinIndices && skinWeights && bones && boneInverses && bindMatrixInverse === null) {
        warnings.push(
          `[vitrum/gltf-adapter] Skinned mesh node "${mesh.name ?? node.mesh}" has a singular ` +
            'bind matrix. bindMatrix/bindMatrixInverse were omitted.',
        );
      }
      let skinArg = (skinIndices && skinWeights && bones && boneInverses)
        ? {
            skinIndices,
            skinWeights,
            bones,
            boneInverses,
            ...(bindMatrixInverse ? {
              bindMatrix: new Float32Array(worldMat),
              bindMatrixInverse,
            } : {}),
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

      primitives.push(_buildPrimitive(
        id, worldMat, positions, normals, indices,
        uvs, uv1, finalTangents, colors, material, skinArg, morph,
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
  const animations = convertAnimations(gltf, buffers, warnings);

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
    warnings,
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
  indices: Uint32Array | undefined,
  material: MaterialSpec,
  meshLabel: string,
  warnings: string[],
): Float32Array | undefined {
  if (!materialNeedsTangentFrame(material)) return undefined;
  if (!uvs) {
    warnings.push(
      `[vitrum/gltf-adapter] Mesh "${meshLabel}" uses a tangent-space material map ` +
        'but has no TEXCOORD_0. Tangents could not be generated; normal-map-like texture(s) may be ignored or approximate.',
    );
    return undefined;
  }
  const generated = generateTangents(positions, normals, uvs, indices);
  if (!generated) {
    warnings.push(
      `[vitrum/gltf-adapter] Mesh "${meshLabel}" uses a tangent-space material map ` +
        'but tangents could not be generated from POSITION/NORMAL/TEXCOORD_0.',
    );
    return undefined;
  }
  warnings.push(
    `[vitrum/gltf-adapter] Mesh "${meshLabel}" uses a tangent-space material map without ` +
      'TANGENT; generated per-vertex tangents from POSITION/NORMAL/TEXCOORD_0.',
  );
  return generated;
}

function materialNeedsTangentFrame(material: MaterialSpec): boolean {
  return material.normalMap !== undefined ||
    material.clearcoatNormalMap !== undefined ||
    material.bumpMap !== undefined;
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
): Float32Array | undefined {
  if (accessorIndex === undefined) return undefined;
  try {
    return unpackAccessorFloat(gltf, buffers, accessorIndex, warnings);
  } catch (e) {
    warnings.push(`[vitrum/gltf-adapter] Failed to read ${label}: ${String(e)}`);
    return undefined;
  }
}

function _invertMat4(m: ArrayLike<number>): Float32Array | null {
  const a00 = m[0] ?? 0;
  const a01 = m[1] ?? 0;
  const a02 = m[2] ?? 0;
  const a03 = m[3] ?? 0;
  const a10 = m[4] ?? 0;
  const a11 = m[5] ?? 0;
  const a12 = m[6] ?? 0;
  const a13 = m[7] ?? 0;
  const a20 = m[8] ?? 0;
  const a21 = m[9] ?? 0;
  const a22 = m[10] ?? 0;
  const a23 = m[11] ?? 0;
  const a30 = m[12] ?? 0;
  const a31 = m[13] ?? 0;
  const a32 = m[14] ?? 0;
  const a33 = m[15] ?? 0;

  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;

  const det =
    b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!Number.isFinite(det) || Math.abs(det) < 1e-12) return null;
  const invDet = 1.0 / det;

  const out = new Float32Array(16);
  out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * invDet;
  out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * invDet;
  out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * invDet;
  out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * invDet;
  out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * invDet;
  out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * invDet;
  out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * invDet;
  out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * invDet;
  out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * invDet;
  out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * invDet;
  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * invDet;
  out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * invDet;
  out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * invDet;
  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * invDet;
  out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * invDet;
  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * invDet;
  return out;
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
  worldTransforms: Map<number, Float32Array>,
  warnings: string[],
): SkinData | undefined {
  if (skinIdx === undefined) return undefined;
  const gltfSkin = gltfSkins[skinIdx];
  if (!gltfSkin) return undefined;

  const jointCount = gltfSkin.joints.length;
  const bones = new Float32Array(jointCount * 16);
  for (let j = 0; j < jointCount; j++) {
    const jointNodeIdx = gltfSkin.joints[j]!;
    const jointWorld = worldTransforms.get(jointNodeIdx);
    if (jointWorld) {
      for (let k = 0; k < 16; k++) {
        bones[j * 16 + k] = jointWorld[k] ?? 0;
      }
    } else {
      // Joint node not in the scene graph — use identity.
      bones[j * 16 + 0] = 1; bones[j * 16 + 5] = 1;
      bones[j * 16 + 10] = 1; bones[j * 16 + 15] = 1;
    }
  }

  let boneInverses: Float32Array | undefined;
  if (gltfSkin.inverseBindMatrices !== undefined) {
    try {
      boneInverses = unpackAccessorFloat(gltf, buffers, gltfSkin.inverseBindMatrices, warnings);
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
    transform: worldMat,
  };
  if (skin) {
    return {
      kind: 'skinned-mesh' as const,
      ...base,
      skinIndices: skin.skinIndices,
      skinWeights: skin.skinWeights,
      bones: skin.bones,
      boneInverses: skin.boneInverses,
      ...(skin.bindMatrix ? { bindMatrix: skin.bindMatrix } : {}),
      ...(skin.bindMatrixInverse ? { bindMatrixInverse: skin.bindMatrixInverse } : {}),
      ...(morph ? {
        morphTargets: morph.morphTargets,
        ...(morph.morphTargetNormals ? { morphTargetNormals: morph.morphTargetNormals } : {}),
        morphWeights: morph.morphWeights,
      } : {}),
    };
  }
  return { kind: 'mesh' as const, ...base };
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
  /** Initial per-target weights from `node.weights ?? mesh.weights` (zeros
   *  when neither is authored). */
  morphWeights: Float32Array;
}

/**
 * Parse glTF `primitive.targets` into core morph-target delta arrays.
 *
 * glTF §3.7.2.2: each target maps attribute names to accessors carrying
 * DELTAS from the base attribute (sparse accessors are common here and are
 * handled by `unpackAccessorFloat`). POSITION and NORMAL deltas map onto
 * `SkinnedMeshPrimitive.morphTargets` / `.morphTargetNormals`; TANGENT deltas
 * are warn-skipped because core does not model morph-tangent deltas.
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
  warnings: string[],
): MorphData | undefined {
  if (!targets || targets.length === 0) return undefined;
  const tCount = targets.length;

  const morphTargets: Float32Array[] = [];
  const normalDeltas: (Float32Array | null)[] = [];
  let anyNormals = false;
  let warnedTangent = false;

  for (let t = 0; t < tCount; t++) {
    const target = targets[t]!;

    // POSITION delta.
    let posDelta = _tryUnpackFloat(
      gltf, buffers, target['POSITION'],
      `morph target ${t} POSITION for "${meshLabel}"`, warnings,
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
      `morph target ${t} NORMAL for "${meshLabel}"`, warnings,
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

    // TANGENT delta — core has no morph-tangent field.
    if (target['TANGENT'] !== undefined && !warnedTangent) {
      warnings.push(
        `[vitrum/gltf-adapter] Mesh "${meshLabel}" morph targets carry TANGENT deltas, ` +
          'which @vitrum/core does not model (SkinnedMeshPrimitive has morphTargets / ' +
          'morphTargetNormals only). TANGENT deltas are ignored.',
      );
      warnedTangent = true;
    }

    for (const attr of Object.keys(target)) {
      if (attr !== 'POSITION' && attr !== 'NORMAL' && attr !== 'TANGENT') {
        warnings.push(
          `[vitrum/gltf-adapter] Morph target ${t} attribute "${attr}" in mesh ` +
            `"${meshLabel}" is ignored.`,
        );
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
