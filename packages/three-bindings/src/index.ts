/**
 * @vitrum/three-bindings — THREE.Scene → @vitrum/core Scene adapter.
 *
 * Public API: `sceneFromThreeJS` plus Tier 1 extension constants (`spectral.ts`).
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
import { convertMesh, emissiveMeshAreaEmitter, stripEmissive } from './mesh.js';
import { convertLight } from './lights.js';
import { resolveEnvironment } from './environment.js';

export { vitrumSceneToThree, disposeVitrumThreeSceneRoot } from './vitrumSceneToThree.js';
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
      throw new Error(
        `Unsupported THREE type at "${label}": InstancedMesh. Supported types are added per Phase 6 sprint.`,
      );
    }
    if ((obj as THREE.SkinnedMesh).isSkinnedMesh === true) {
      throw new Error(
        `Unsupported THREE type at "${label}": SkinnedMesh. Supported types are added per Phase 6 sprint.`,
      );
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
