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
// Out of scope (v1): animations, skins (imported as static rest-pose with warning),
// cameras, morph targets, KHR_lights_punctual (emits a warning).
//
// Primitive modes: only TRIANGLES (4) is converted; other modes emit a warning
// and the primitive is skipped.
//
// Reference: glTF 2.0 specification (Khronos Group).

import type { Scene, ScenePrimitive } from '@vitrum/core';
import type { GltfJson } from './gltfTypes.js';
import type { DecodeImageFn } from './textures.js';
import { parseGlb } from './glbParser.js';
import { unpackAccessorFloat, unpackAccessorUint32 } from './accessors.js';
import { buildWorldTransforms } from './transforms.js';
import { buildTextureHandleMap } from './textures.js';
import { convertMaterial, GLTF_DEFAULT_MATERIAL } from './materials.js';
import { generateFlatNormals } from './normals.js';

const GLTF_PRIMITIVE_MODE_TRIANGLES = 4;

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
   * Index of the glTF scene to import (0-based). Defaults to gltf.scene or 0.
   */
  readonly sceneIndex?: number;
}

export interface GltfToSceneResult {
  readonly scene: Scene;
  /** Non-fatal issues encountered during conversion. Inspect these for skipped
   *  primitives, unsupported extensions, missing buffers, sparse patches, etc. */
  readonly warnings: string[];
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
      for (const [k, v] of opts.buffers) buffers.set(k, v);
    } else {
      for (const [k, v] of Object.entries(opts.buffers)) {
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

  // ── 3. Warn on out-of-scope top-level features ─────────────────────────────
  if (gltf.animations && gltf.animations.length > 0) {
    warnings.push(
      `[vitrum/gltf-adapter] This glTF has ${gltf.animations.length} animation(s). ` +
        'Animations are NOT supported in v1. Geometry will be imported at rest pose.',
    );
  }
  if (gltf.cameras && (gltf.cameras as unknown[]).length > 0) {
    warnings.push(
      '[vitrum/gltf-adapter] Camera nodes are present but ignored (cameras are not part of the ' +
        '@vitrum/core Scene contract; pass camera data via FrameInput instead).',
    );
  }
  if (gltf.skins && gltf.skins.length > 0) {
    warnings.push(
      `[vitrum/gltf-adapter] This glTF has ${gltf.skins.length} skin(s). ` +
        'Skinned meshes are imported as STATIC REST-POSE geometry (v1 limitation). ' +
        'Full skinning (SkinnedMeshPrimitive remapping) is planned for v2.',
    );
  }

  // Warn about KHR_lights_punctual (extension that would map to emitters).
  const extUsed = gltf.extensionsUsed ?? [];
  if (extUsed.includes('KHR_lights_punctual')) {
    warnings.push(
      '[vitrum/gltf-adapter] KHR_lights_punctual is present but NOT imported as SceneEmitters in v1. ' +
        'Construct SceneEmitter objects manually from your scene graph and pass them to the Scene. ' +
        'This extension is tracked for v2 support.',
    );
  }
  if (extUsed.includes('KHR_draco_mesh_compression')) {
    warnings.push(
      '[vitrum/gltf-adapter] KHR_draco_mesh_compression is present but NOT supported. ' +
        'Affected primitives will be skipped. Decode the mesh with a Draco decoder before ' +
        'passing to the adapter, or export without Draco compression.',
    );
  }
  if (extUsed.includes('EXT_meshopt_compression')) {
    warnings.push(
      '[vitrum/gltf-adapter] EXT_meshopt_compression is present but NOT supported. ' +
        'Affected primitives will be skipped. Decode with a MeshOpt decoder before passing.',
    );
  }

  // ── 4. Resolve textures ────────────────────────────────────────────────────
  const handleMap = await buildTextureHandleMap(
    gltf,
    buffers,
    opts.decodeImage,
    warnings,
  );

  // ── 5. Pre-convert materials ───────────────────────────────────────────────
  const coreMaterials = (gltf.materials ?? []).map((m) =>
    convertMaterial(m, handleMap, warnings),
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
  let primIdCounter = 0;

  const gltfNodes = gltf.nodes ?? [];
  const gltfMeshes = gltf.meshes ?? [];

  for (const [nodeIdx, worldMat] of worldTransforms) {
    const node = gltfNodes[nodeIdx];
    if (!node || node.mesh === undefined) continue;

    const mesh = gltfMeshes[node.mesh];
    if (!mesh) continue;

    for (const prim of mesh.primitives) {
      // Mode check — only TRIANGLES (4) or absent (default=4) is supported.
      const mode = prim.mode ?? GLTF_PRIMITIVE_MODE_TRIANGLES;
      if (mode !== GLTF_PRIMITIVE_MODE_TRIANGLES) {
        const modeNames: Record<number, string> = {
          0: 'POINTS', 1: 'LINES', 2: 'LINE_LOOP', 3: 'LINE_STRIP',
          5: 'TRIANGLE_STRIP', 6: 'TRIANGLE_FAN',
        };
        warnings.push(
          `[vitrum/gltf-adapter] Mesh "${mesh.name ?? node.mesh}" primitive has unsupported ` +
            `mode ${mode} (${modeNames[mode] ?? 'UNKNOWN'}). Only TRIANGLES (4) is supported. ` +
            'This primitive is SKIPPED.',
        );
        continue;
      }

      // Check for unsupported compression extensions.
      const primExtKeys = Object.keys(prim.extensions ?? {});
      if (primExtKeys.includes('KHR_draco_mesh_compression') || primExtKeys.includes('EXT_meshopt_compression')) {
        warnings.push(
          `[vitrum/gltf-adapter] Mesh "${mesh.name ?? node.mesh}" primitive uses compressed geometry ` +
            `(${primExtKeys.join(', ')}) which is not supported. Primitive SKIPPED.`,
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

      // Normals — generate flat normals if absent.
      let normals: Float32Array;
      const normIdx = prim.attributes['NORMAL'];
      if (normIdx !== undefined) {
        try {
          normals = unpackAccessorFloat(gltf, buffers, normIdx, warnings);
        } catch (e) {
          warnings.push(
            `[vitrum/gltf-adapter] Failed to read NORMAL for mesh "${mesh.name ?? node.mesh}": ` +
              String(e) + '. Generating flat normals.',
          );
          normals = generateFlatNormals(positions, indices);
        }
      } else {
        warnings.push(
          `[vitrum/gltf-adapter] Mesh "${mesh.name ?? node.mesh}" has no NORMAL attribute. ` +
            'Generating flat normals.',
        );
        normals = generateFlatNormals(positions, indices);
      }

      // UVs — optional.
      let uvs: Float32Array | undefined;
      const uv0Idx = prim.attributes['TEXCOORD_0'];
      if (uv0Idx !== undefined) {
        try {
          uvs = unpackAccessorFloat(gltf, buffers, uv0Idx, warnings);
        } catch (e) {
          warnings.push(
            `[vitrum/gltf-adapter] Failed to read TEXCOORD_0 for "${mesh.name ?? node.mesh}": ` +
              String(e),
          );
        }
      }

      let uv1: Float32Array | undefined;
      const uv1Idx = prim.attributes['TEXCOORD_1'];
      if (uv1Idx !== undefined) {
        try {
          uv1 = unpackAccessorFloat(gltf, buffers, uv1Idx, warnings);
        } catch (e) {
          warnings.push(
            `[vitrum/gltf-adapter] Failed to read TEXCOORD_1 for "${mesh.name ?? node.mesh}": ` +
              String(e),
          );
        }
      }

      // Tangents — optional (xyzw per vertex).
      let tangents: Float32Array | undefined;
      const tangentIdx = prim.attributes['TANGENT'];
      if (tangentIdx !== undefined) {
        try {
          tangents = unpackAccessorFloat(gltf, buffers, tangentIdx, warnings);
        } catch (e) {
          warnings.push(
            `[vitrum/gltf-adapter] Failed to read TANGENT for "${mesh.name ?? node.mesh}": ` +
              String(e),
          );
        }
      }

      // Vertex colors — optional (COLOR_0).
      let colors: Float32Array | undefined;
      const colorIdx = prim.attributes['COLOR_0'];
      if (colorIdx !== undefined) {
        try {
          colors = unpackAccessorFloat(gltf, buffers, colorIdx, warnings);
        } catch (e) {
          warnings.push(
            `[vitrum/gltf-adapter] Failed to read COLOR_0 for "${mesh.name ?? node.mesh}": ` +
              String(e),
          );
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

      // Morph targets — warn and skip.
      if (prim.targets && prim.targets.length > 0) {
        warnings.push(
          `[vitrum/gltf-adapter] Mesh "${mesh.name ?? node.mesh}" has ${prim.targets.length} ` +
            'morph target(s). Morph targets are NOT supported in v1 and are ignored.',
        );
      }

      // Material.
      const material =
        prim.material !== undefined && prim.material < coreMaterials.length
          ? (coreMaterials[prim.material] ?? GLTF_DEFAULT_MATERIAL)
          : GLTF_DEFAULT_MATERIAL;

      // Construct the MeshPrimitive.
      const id = `gltf-prim-${primIdCounter++}`;

      primitives.push({
        kind: 'mesh',
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
      });
    }
  }

  const scene: Scene = {
    primitives,
    emitters: [],
    environment: { kind: 'none' },
  };

  return { scene, warnings };
}
