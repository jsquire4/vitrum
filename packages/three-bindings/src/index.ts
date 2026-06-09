/**
 * @vitrum/three-bindings — THREE.Scene → @vitrum/core Scene adapter.
 *
 * Public API: `sceneFromThreeJS` plus conversion helpers and loaders.
 * All sub-converters live in dedicated modules:
 *
 *   material.ts   — convertMaterial (MeshStandard + MeshPhysical) + convertBasicMaterial
 *   mesh.ts       — convertMesh + geometry attribute extractors
 *   lights.ts     — convertLight (directional, rect-area, point, spot)
 *   environment.ts — resolveEnvironment (HDRI vs none)
 *   vitrumSceneToThree.ts — core Scene → THREE (for pt-webgl + walkaround BVH)
 */

import type * as THREE from 'three';
import type { Scene, ScenePrimitive, SceneEmitter } from '@vitrum/core';
import {
  convertInstancedMeshToPrimitives,
  convertMeshToPrimitives,
  convertSkinnedMeshToPrimitives,
  emissiveMaterialAreaEmitter,
  stripEmissive,
  type MeshConversionOptions,
  type ThreeMaterialConverter,
} from './mesh.js';
import { convertLight } from './lights.js';
import { resolveEnvironment, type EnvironmentPayloadMode } from './environment.js';

export {
  vitrumSceneToThree,
  disposeVitrumThreeSceneRoot,
  applyEnvironment,
  applyVitrumMaterialToMesh,
  findMeshByPrimitiveId,
} from './vitrumSceneToThree.js';
export { combineSkinMatrices, solveSkin, mat3InverseTranspose } from '@vitrum/core';
export { loadGltfScene, type LoadedGltf, type GltfCamera, type LoadGltfSceneOptions } from './gltfLoader.js';
export { convertAnimations } from './animationImport.js';
export { VITRUM_USER_DATA_KEYS } from './userDataKeys.js';
export {
  extractThreePbrScalars,
  PBR_DEFAULTS_DEFAULT,
  colorToVec3,
  convertMaterial,
  convertBasicMaterial,
} from './material.js';
export type { PbrScalars, PbrDefaults } from './material.js';

function firstMaterial(material: THREE.Material | THREE.Material[]): THREE.Material | null {
  return Array.isArray(material) ? material[0] ?? null : material ?? null;
}

export interface SceneFromThreeJSOptions {
  /**
   * Optional host converter for material classes the default adapter does not
   * understand, such as ShaderMaterial/RawShaderMaterial.
   *
   * Return a complete core MaterialSpec to accept the material; return
   * null/undefined to fall back to the built-in converter/diagnostic.
   */
  readonly materialConverter?: ThreeMaterialConverter;

  /**
   * How the resolved `environment.hdri` handle is represented:
   *  - `'texture'` (default) — a `THREE.Texture`, for the fork-wrapping
   *    `@vitrum/pt-webgl` (its `vitrumSceneToThree` reads the texture back).
   *  - `'raw'` — a backend-neutral `{ width, height, data }` equirect payload, so
   *    the THREE-free path tracers (`@vitrum/pt-webgl2`, `@vitrum/pt-webgpu`) can
   *    sample the IBL. Falls back to the texture when CPU pixels aren't readable.
   * Set `'raw'` when the target backend is a THREE-free engine.
   */
  readonly environmentPayload?: EnvironmentPayloadMode;
}

function assertSupportedRenderableMaterial(
  rawMat: THREE.Material | null,
  label: string,
  options: SceneFromThreeJSOptions,
): void {
  if (
    rawMat != null &&
    ((rawMat as THREE.ShaderMaterial).isShaderMaterial === true ||
      (rawMat as THREE.RawShaderMaterial).isRawShaderMaterial === true)
  ) {
    if (options.materialConverter != null) return;
    throw new Error(
      `Unsupported THREE type at "${label}": ${(rawMat as object).constructor.name}. Supported types are listed in the backend's EngineCapabilities.`,
    );
  }
}

function shouldSkipRenderableObject(obj: THREE.Object3D, rawMat: THREE.Material | null): boolean {
  if (obj.visible === false) return true;
  if (rawMat?.visible === false) return true;
  return (
    rawMat != null &&
    (rawMat as THREE.MeshBasicMaterial).isMeshBasicMaterial === true &&
    ((rawMat as THREE.MeshBasicMaterial).transparent === true ||
      ((rawMat as THREE.MeshBasicMaterial).opacity ?? 1) <= 0.01)
  );
}

/**
 * Converts a THREE.Scene into a @vitrum/core Scene.
 *
 * Call this whenever scene topology changes (geometry added/removed, materials
 * swapped). For property-only edits (color sliders, intensity), prefer
 * engine.updatePrimitive / engine.updateEmitter if the backend supports
 * incremental updates.
 *
 * **Warning semantics:** Unsupported-but-skippable light types (AmbientLight,
 * HemisphereLight) emit one `console.warn` per unsupported type per `sceneFromThreeJS`
 * call. The warning fires again on the next call — it is not suppressed across
 * calls. This ensures scene hot-reloads in dev don't permanently silence warnings.
 */
export function sceneFromThreeJS(
  threeScene: THREE.Scene,
  options: SceneFromThreeJSOptions = {},
): Scene {
  threeScene.updateMatrixWorld(true);

  const primitives: ScenePrimitive[] = [];
  const emitters: SceneEmitter[] = [];
  const meshConversionOptions: MeshConversionOptions = {
    ...(options.materialConverter != null
      ? { materialConverter: options.materialConverter }
      : {}),
  };

  // Per-call warning dedup set — scoped to this call, not module-global.
  // Prevents duplicate warnings for the same type within one traversal while
  // allowing each new sceneFromThreeJS() call to warn again if needed.
  // (M-2 fix: was a module-level Set that suppressed warnings across all calls.)
  const warnedTypes = new Set<string>();

  threeScene.traverse((obj: THREE.Object3D) => {
    const label = obj.name || obj.uuid;

    // ── Unsupported mesh sub-types ──────────────────────────────────────────
    if ((obj as THREE.InstancedMesh).isInstancedMesh === true) {
      const inst = obj as THREE.InstancedMesh;
      const rawMat = firstMaterial(inst.material);
      assertSupportedRenderableMaterial(rawMat, label, options);
      if (shouldSkipRenderableObject(obj, rawMat)) return;
      const splitMaterials = Array.isArray(inst.material) && inst.geometry.groups.length > 0
        ? inst.material
        : null;
      const instancedPrimitives = convertInstancedMeshToPrimitives(inst, meshConversionOptions);
      for (let primitiveIndex = 0; primitiveIndex < instancedPrimitives.length; primitiveIndex += 1) {
        const prim = instancedPrimitives[primitiveIndex]!;
        const sourceMaterial = splitMaterials != null
          ? splitMaterials[inst.geometry.groups[primitiveIndex]!.materialIndex ?? 0] ?? null
          : rawMat ?? null;
        const meshEmitter = emissiveMaterialAreaEmitter(sourceMaterial, prim.id);
        if (meshEmitter != null) {
          emitters.push(meshEmitter);
          primitives.push(stripEmissive(prim));
          continue;
        }
        primitives.push(prim);
      }
      return;
    }

    // ── Skinned meshes ─────────────────────────────────────────────────────
    // C1 (2026-05-19) — convert rest-pose + current pose. Per-frame skin
    // updates are pushed via `engine.updatePrimitive(...)`. Engines that
    // don't implement skinning report so via EngineCapabilities and may
    // render the rest pose statically.
    if ((obj as THREE.SkinnedMesh).isSkinnedMesh === true) {
      const skinned = obj as THREE.SkinnedMesh;
      const rawMat = firstMaterial(skinned.material);
      assertSupportedRenderableMaterial(rawMat, label, options);
      if (shouldSkipRenderableObject(obj, rawMat)) return;
      const splitMaterials = Array.isArray(skinned.material) && skinned.geometry.groups.length > 0
        ? skinned.material
        : null;
      const skinnedPrimitives = convertSkinnedMeshToPrimitives(skinned, meshConversionOptions);
      for (let primitiveIndex = 0; primitiveIndex < skinnedPrimitives.length; primitiveIndex += 1) {
        const prim = skinnedPrimitives[primitiveIndex]!;
        const sourceMaterial = splitMaterials != null
          ? splitMaterials[skinned.geometry.groups[primitiveIndex]!.materialIndex ?? 0] ?? null
          : rawMat ?? null;
        const meshEmitter = emissiveMaterialAreaEmitter(sourceMaterial, prim.id);
        if (meshEmitter != null) {
          emitters.push(meshEmitter);
          primitives.push(stripEmissive(prim));
          continue;
        }
        primitives.push(prim);
      }
      return;
    }

    // ── Meshes ──────────────────────────────────────────────────────────────
    if ((obj as THREE.Mesh).isMesh === true) {
      const mesh = obj as THREE.Mesh;
      const rawMat = firstMaterial(mesh.material);
      assertSupportedRenderableMaterial(rawMat, label, options);
      // Skip meshes that aren't visually rendered — these are pointer-
      // capture planes (CanvasEventRouter's 10000×10000 plane with
      // `visible={false}`), edge hot-zones (EdgeHotZone with opacity=0),
      // selection-handle overlays, etc. Without skipping, they enter the
      // path-traced / walkaround scene as opaque flat-emissive surfaces
      // and occlude the actual geometry. PT/walkaround panel-black bug
      // 2026-05-12.
      if (shouldSkipRenderableObject(obj, rawMat)) return;
      const splitMaterials = Array.isArray(mesh.material) && mesh.geometry.groups.length > 0
        ? mesh.material
        : null;
      const meshPrimitives = convertMeshToPrimitives(mesh, meshConversionOptions);
      for (let primitiveIndex = 0; primitiveIndex < meshPrimitives.length; primitiveIndex += 1) {
        const prim = meshPrimitives[primitiveIndex]!;
        const sourceMaterial = splitMaterials != null
          ? splitMaterials[mesh.geometry.groups[primitiveIndex]!.materialIndex ?? 0] ?? null
          : rawMat ?? null;
        const meshEmitter = emissiveMaterialAreaEmitter(sourceMaterial, prim.id);
        if (meshEmitter != null) {
          emitters.push(meshEmitter);
          primitives.push(stripEmissive(prim));
          continue;
        }
        primitives.push(prim);
      }
      return;
    }

    // ── Lights ──────────────────────────────────────────────────────────────
    if ((obj as THREE.Light).isLight !== true) return;

    const emitter = convertLight(obj as THREE.Light, warnedTypes);
    if (emitter != null) {
      emitters.push(emitter);
    }
  });

  const environment = resolveEnvironment(threeScene, options.environmentPayload);

  return { primitives, emitters, environment };
}
