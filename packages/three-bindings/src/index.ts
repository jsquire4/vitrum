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
  convertInstancedMesh,
  convertMesh,
  convertSkinnedMesh,
  emissiveMeshAreaEmitter,
  stripEmissive,
} from './mesh.js';
import { convertLight } from './lights.js';
import { resolveEnvironment } from './environment.js';

export {
  vitrumSceneToThree,
  disposeVitrumThreeSceneRoot,
  applyEnvironment,
  applyVitrumMaterialToMesh,
} from './vitrumSceneToThree.js';
export { solveSkin } from './skinSolver.js';
export { loadGltfScene, type LoadedGltf, type GltfCamera, type LoadGltfSceneOptions } from './gltfLoader.js';
export { VITRUM_USER_DATA_KEYS } from './userDataKeys.js';
export {
  extractThreePbrScalars,
  PBR_DEFAULTS_DEFAULT,
  colorToVec3,
  convertMaterial,
  convertBasicMaterial,
} from './material.js';
export type { PbrScalars, PbrDefaults, ThreeStdMat, ThreePhysMat } from './material.js';

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
export function sceneFromThreeJS(threeScene: THREE.Scene): Scene {
  threeScene.updateMatrixWorld(true);

  const primitives: ScenePrimitive[] = [];
  const emitters: SceneEmitter[] = [];

  // Per-call warning dedup set — scoped to this call, not module-global.
  // Prevents duplicate warnings for the same type within one traversal while
  // allowing each new sceneFromThreeJS() call to warn again if needed.
  // (M-2 fix: was a module-level Set that suppressed warnings across all calls.)
  const warnedTypes = new Set<string>();

  threeScene.traverse((obj: THREE.Object3D) => {
    const label = obj.name || obj.uuid;

    // ── Unsupported mesh sub-types ──────────────────────────────────────────
    if ((obj as THREE.InstancedMesh).isInstancedMesh === true) {
      if (obj.visible === false) return;
      const inst = obj as THREE.InstancedMesh;
      const rawMat = Array.isArray(inst.material) ? inst.material[0] : inst.material;
      if ((rawMat as THREE.Material | null)?.visible === false) return;
      primitives.push(convertInstancedMesh(inst));
      return;
    }

    // ── Skinned meshes ─────────────────────────────────────────────────────
    // C1 (2026-05-19) — convert rest-pose + current pose. Per-frame skin
    // updates are pushed via `engine.updatePrimitive(...)`. Engines that
    // don't implement skinning report so via EngineCapabilities and may
    // render the rest pose statically.
    if ((obj as THREE.SkinnedMesh).isSkinnedMesh === true) {
      if (obj.visible === false) return;
      const skinned = obj as THREE.SkinnedMesh;
      const prim = convertSkinnedMesh(skinned);
      const meshEmitter = emissiveMeshAreaEmitter(skinned);
      if (meshEmitter != null) {
        emitters.push(meshEmitter);
        primitives.push({
          ...prim,
          material: { ...prim.material, emissive: [0, 0, 0], emissiveIntensity: 0 },
        });
        return;
      }
      primitives.push(prim);
      return;
    }

    // ── Meshes ──────────────────────────────────────────────────────────────
    if ((obj as THREE.Mesh).isMesh === true) {
      const mesh = obj as THREE.Mesh;
      const rawMat = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
      if (
        rawMat != null &&
        ((rawMat as THREE.ShaderMaterial).isShaderMaterial === true ||
          (rawMat as THREE.RawShaderMaterial).isRawShaderMaterial === true)
      ) {
        throw new Error(
          `Unsupported THREE type at "${label}": ${(rawMat as object).constructor.name}. Supported types are added per Phase 6 sprint.`,
        );
      }
      // Skip meshes that aren't visually rendered — these are pointer-
      // capture planes (CanvasEventRouter's 10000×10000 plane with
      // `visible={false}`), edge hot-zones (EdgeHotZone with opacity=0),
      // selection-handle overlays, etc. Without skipping, they enter the
      // path-traced / walkaround scene as opaque flat-emissive surfaces
      // and occlude the actual geometry. PT/walkaround panel-black bug
      // 2026-05-12.
      if (obj.visible === false) return;
      if ((rawMat as THREE.Material | null)?.visible === false) return;
      const isBasicTransparent =
        rawMat != null &&
        (rawMat as THREE.MeshBasicMaterial).isMeshBasicMaterial === true &&
        ((rawMat as THREE.MeshBasicMaterial).transparent === true ||
         ((rawMat as THREE.MeshBasicMaterial).opacity ?? 1) <= 0.01);
      if (isBasicTransparent) return;
      const prim = convertMesh(mesh);
      const meshEmitter = emissiveMeshAreaEmitter(mesh);
      if (meshEmitter != null) {
        emitters.push(meshEmitter);
        primitives.push(stripEmissive(prim));
        return;
      }
      primitives.push(prim);
      return;
    }

    // ── Lights ──────────────────────────────────────────────────────────────
    if ((obj as THREE.Light).isLight !== true) return;

    const emitter = convertLight(obj as THREE.Light, warnedTypes);
    if (emitter != null) {
      emitters.push(emitter);
    }
  });

  const environment = resolveEnvironment(threeScene);

  return { primitives, emitters, environment };
}
