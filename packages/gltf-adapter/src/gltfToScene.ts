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
//     inverseBindMatrices, rest-pose joint world transforms).
//   - KHR_lights_punctual → SceneEmitter[] (point, spot, directional;
//     world-transform applied to position/direction).
//
// Out of scope: animations (rest-pose only), cameras, morph targets,
//   KHR_draco_mesh_compression, EXT_meshopt_compression.
//
// Primitive modes: only TRIANGLES (4) is converted; other modes emit a warning
// and the primitive is skipped.
//
// References:
//   - glTF 2.0 specification (Khronos Group)
//     https://www.khronos.org/registry/glTF/specs/2.0/glTF-2.0.html
//   - KHR_lights_punctual extension
//     https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_lights_punctual/README.md

import type { Scene, ScenePrimitive, SceneEmitter } from '@vitrum/core';
import type { GltfJson, KhrLightsPunctualRoot } from './gltfTypes.js';
import { GltfComponentType } from './gltfTypes.js';
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

  // ── 3. Warn on out-of-scope top-level features ─────────────────────────────
  if (gltf.animations && gltf.animations.length > 0) {
    warnings.push(
      `[vitrum/gltf-adapter] This glTF has ${gltf.animations.length} animation(s). ` +
        'Animations are NOT supported in v1. Geometry will be imported at rest pose.',
    );
  }
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
        'Animations are NOT supported; the skeleton will not move after import.',
    );
  }

  const extUsed = gltf.extensionsUsed ?? [];
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
  const gltfSkins = gltf.skins ?? [];

  for (const [nodeIdx, worldMat] of worldTransforms) {
    const node = gltfNodes[nodeIdx];
    if (!node || node.mesh === undefined) continue;

    const mesh = gltfMeshes[node.mesh];
    if (!mesh) continue;

    // ── Skin data for this node (if any) ──────────────────────────────────
    // glTF 2.0 §3.8: a node may reference a skin by index. All primitives in
    // the node's mesh share the same skin.
    const skinIdx = node.skin;
    const gltfSkin = skinIdx !== undefined ? gltfSkins[skinIdx] : undefined;

    // Build per-joint rest-pose bone matrices (world transforms of joint nodes).
    // We use the same worldTransforms map already computed from the scene graph,
    // so the joint matrices are in world space — matching SkinnedMeshPrimitive
    // contract: `bones` = bone-local-to-world column-major f32s.
    let bones: Float32Array | undefined;
    let boneInverses: Float32Array | undefined;
    if (gltfSkin) {
      const jointCount = gltfSkin.joints.length;
      bones = new Float32Array(jointCount * 16);
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

      // Inverse bind matrices — required by glTF spec to exist when skinning is
      // used, though the accessor is technically optional (identity if absent).
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
        // Spec fallback: identity inverse bind matrices.
        boneInverses = new Float32Array(jointCount * 16);
        for (let j = 0; j < jointCount; j++) {
          boneInverses[j * 16 + 0] = 1; boneInverses[j * 16 + 5] = 1;
          boneInverses[j * 16 + 10] = 1; boneInverses[j * 16 + 15] = 1;
        }
      }
    }

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

      // ── Skinning attributes ────────────────────────────────────────────────
      // Only unpacked when this node has a skin; JOINTS_0 / WEIGHTS_0 without a
      // node.skin are silently ignored (they carry no semantics without a skin).
      let skinIndices: Uint32Array | undefined;
      let skinWeights: Float32Array | undefined;
      if (gltfSkin && bones && boneInverses) {
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

      const id = `gltf-prim-${primIdCounter++}`;

      if (skinIndices && skinWeights && bones && boneInverses) {
        // ── SkinnedMeshPrimitive ─────────────────────────────────────────────
        // glTF §3.8: for typical glTF files (mesh at origin, bind at origin),
        // bindMatrix is identity and can be omitted. Pose = rest pose from the
        // scene graph (animations are not imported).
        primitives.push({
          kind: 'skinned-mesh',
          id,
          positions,
          normals,
          ...(uvs ? { uvs } : {}),
          ...(uv1 ? { uv1 } : {}),
          ...(tangents ? { tangents } : {}),
          ...(colors ? { colors } : {}),
          ...(indices ? { indices } : {}),
          skinIndices,
          skinWeights,
          bones,
          boneInverses,
          material,
          transform: worldMat,
        });
      } else {
        // ── MeshPrimitive ────────────────────────────────────────────────────
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

      const color: [number, number, number] = light.color
        ? [light.color[0], light.color[1], light.color[2]]
        : [1, 1, 1];
      const intensity = light.intensity ?? 1;
      const id = `gltf-light-${emitterIdCounter++}`;

      // Extract world-space position from the node's world matrix (column 3).
      // Column-major 4×4: translation is at indices 12, 13, 14.
      const px = worldMat[12] ?? 0;
      const py = worldMat[13] ?? 0;
      const pz = worldMat[14] ?? 0;

      // Extract the forward direction (-Z in glTF/node space) transformed to world.
      // In a column-major matrix, the local Z axis (third column) is at indices 8,9,10.
      // glTF lights point along -Z in their local space.
      const lzx = -(worldMat[8] ?? 0);
      const lzy = -(worldMat[9] ?? 0);
      const lzz = -(worldMat[10] ?? 0);
      const lzLen = Math.hypot(lzx, lzy, lzz);
      const dirX = lzLen > 1e-10 ? lzx / lzLen : 0;
      const dirY = lzLen > 1e-10 ? lzy / lzLen : 0;
      const dirZ = lzLen > 1e-10 ? lzz / lzLen : 1;

      switch (light.type) {
        case 'point':
          emitters.push({
            kind: 'point',
            id,
            color,
            intensity,
            position: [px, py, pz],
            ...(light.range != null && light.range > 0 ? { distance: light.range } : {}),
            decay: 2, // glTF punctual always uses physical inverse-square falloff
          });
          break;

        case 'spot': {
          // glTF innerConeAngle / outerConeAngle are half-angles in radians.
          // vitrum SpotEmitter.angle = half-cone angle in radians (= outerConeAngle).
          // vitrum SpotEmitter.penumbra = 0..1 blend fraction from inner to outer edge.
          const inner = light.spot?.innerConeAngle ?? 0;
          const outer = light.spot?.outerConeAngle ?? (Math.PI / 4);
          // penumbra: fraction of the cone that is the penumbra band.
          // 0 = hard edge (inner === outer), 1 = full penumbra (inner === 0).
          const penumbra = outer > 1e-10 ? 1 - inner / outer : 0;
          emitters.push({
            kind: 'spot',
            id,
            color,
            intensity,
            position: [px, py, pz],
            direction: [dirX, dirY, dirZ],
            angle: outer,
            penumbra: Math.max(0, Math.min(1, penumbra)),
            ...(light.range != null && light.range > 0 ? { distance: light.range } : {}),
            decay: 2, // physical inverse-square
          });
          break;
        }

        case 'directional':
          // Core contract: DirectionalEmitter.direction points AT the light
          // (i.e., the direction light COMES FROM in the scene). In glTF, the
          // light faces -Z in local space, so the world-space -Z column is the
          // direction rays travel FROM the light. Negate once more so that
          // direction = "toward the light source" per core contract.
          emitters.push({
            kind: 'directional',
            id,
            color,
            intensity,
            direction: [-dirX, -dirY, -dirZ],
          });
          break;

        default:
          warnings.push(
            `[vitrum/gltf-adapter] KHR_lights_punctual light "${light.name ?? id}" ` +
              `has unsupported type "${(light as { type: string }).type}". Emitter skipped.`,
          );
      }
    }
  }

  const scene: Scene = {
    primitives,
    emitters,
    environment: { kind: 'none' },
  };

  return { scene, warnings };
}

// ── Private helpers ──────────────────────────────────────────────────────────

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
