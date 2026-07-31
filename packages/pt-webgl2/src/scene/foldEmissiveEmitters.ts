// foldEmissiveEmitters — re-attach `mesh-area` emitter radiance onto the emissive
// surface materials the fork integrator samples.
//
// Core-scene importers follow a NEE-first convention: an emissive surface is
// extracted into a `mesh-area` SceneEmitter and the source primitive's material
// emissive is zeroed so NEE-based backends (walkaround, the light-tree NEE in
// pt-webgpu) don't double-count it as both a sampled light and a path-traced
// emissive surface.
//
// The pt-webgl2 fork integrator lights area sources the OTHER way — by tracing a
// path ray that HITS the emissive surface and accumulating `surf.emission` (MIS).
// And `mesh-area` emitters are deliberately NOT analytic lights in the lights
// texture (`packLightsTexture` filters `kind === 'mesh-area'` out). So with the
// material emissive stripped AND no analytic light entry, a Cornell light falls
// through the crack between both mechanisms and the scene renders black.
//
// This pure pass folds each `mesh-area` emitter's radiance back onto its referenced
// primitive's material as `emissive = color, emissiveIntensity = intensity` (the
// materials packer computes `emissiveIntensity * emissive` = the original
// `color * intensity` radiance). It runs ONLY inside pt-webgl2's scene-texture
// build — every other backend keeps consuming the original (stripped) scene, so
// the cross-backend NEE contract is untouched.
//
// pt-webgl2 also builds an explicit mesh-area NEE texture in `meshAreaLights.ts`.
// This fold keeps camera/path-hit emissive radiance and mesh-light NEE radiance in
// sync with the same authored `mesh-area` emitter.

import type { MaterialSpec, Scene, ScenePrimitive, SceneNodeId, Vec3 } from '@vitrum/core';
import { materialWithExplicitMeshEmitterAuthority } from './meshEmitterPolicy.js';

type FoldedMeshAreaEmitterMaterial = MaterialSpec & {
  readonly meshEmitterCastShadowDisabled?: boolean;
};

/**
 * Return a scene whose primitives carry the emissive their `mesh-area` emitters
 * describe. If the scene has no `mesh-area` emitters the input is returned
 * unchanged (referential identity preserved — no needless reallocation).
 */
export function foldMeshAreaEmittersIntoMaterials(scene: Scene): Scene {
  const radianceByMesh = new Map<SceneNodeId, {
    color: Vec3;
    intensity: number;
    castShadowDisabled: boolean;
  }>();
  for (const e of scene.emitters) {
    if (e.kind === 'mesh-area') {
      radianceByMesh.set(e.meshId, {
        color: e.color,
        intensity: e.intensity,
        castShadowDisabled: e.castShadow === false,
      });
    }
  }
  if (radianceByMesh.size === 0) return scene;

  const primitives = scene.primitives.map((prim): ScenePrimitive => {
    const r = radianceByMesh.get(prim.id);
    if (r == null) return prim;
    const authoritativeMaterial = materialWithExplicitMeshEmitterAuthority(prim.material);
    const material: FoldedMeshAreaEmitterMaterial = {
      ...authoritativeMaterial,
      emissive: r.color,
      emissiveIntensity: r.intensity,
      ...(r.castShadowDisabled ? { meshEmitterCastShadowDisabled: true } : {}),
    };
    return { ...prim, material };
  });

  return { ...scene, primitives };
}
