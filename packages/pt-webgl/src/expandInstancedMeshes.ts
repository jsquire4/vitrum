// Instanced-mesh expansion for the WebGL2 path tracer.
//
// WHY THIS EXISTS
// ---------------
// `@vitrum/three-bindings`'s `instancedMeshPrimitiveToThree` deliberately
// produces a single `THREE.InstancedMesh` (one shared geometry + material, N
// per-instance transforms in `instanceMatrix`). That is the right shape for
// `@vitrum/walkaround-hybrid`, whose TLAS path traverses the per-instance
// transforms directly. It is the WRONG shape for pt-webgl, because the
// absorbed three-gpu-pathtracer fork's geometry generator
// (`StaticGeometryGenerator` → `convertToStaticGeometry`) bakes ONLY
// `mesh.matrixWorld` (verified: `convertToStaticGeometry.js:283` applies
// `mesh.matrixWorld` to each position; nothing reads `instanceMatrix`). An
// `InstancedMesh` has an identity `matrixWorld`, so all N instances would
// collapse to ONE copy at the local origin.
//
// We must NOT change the shared `instancedMeshPrimitiveToThree` (walkaround
// needs the single InstancedMesh). Instead, pt-webgl expands each
// `THREE.InstancedMesh` into N separate `THREE.Mesh` objects AFTER
// `vitrumSceneToThree` builds the scene and BEFORE the fork's BVH/geometry
// generator runs, baking `parent.matrixWorld · instanceMatrix[i]` into each
// child mesh's `matrixWorld`. The N children share the InstancedMesh's
// geometry + material (so material edits via `updatePrimitive` still resolve
// through `findMeshByPrimitiveId`, which matches on `name`/`uuid`).

import { Matrix4, Mesh } from 'three';
import type { InstancedMesh, Mesh as TMesh, Object3D, Scene as ThreeScene } from 'three';

/** Per-instance scratch matrix — reused across the whole expansion pass. */
const _instanceMatrix = new Matrix4();
/** Composed `parent.matrixWorld · instanceMatrix[i]` scratch. */
const _composed = new Matrix4();

function isInstancedMesh(o: Object3D): o is InstancedMesh {
  return (o as InstancedMesh).isInstancedMesh === true;
}

/**
 * Expand a single `THREE.InstancedMesh` into `count` standalone `THREE.Mesh`
 * objects, one per `instanceMatrix[i]`. Each child:
 *   - shares the InstancedMesh's geometry + material (no clone — the fork's
 *     generator reads `mesh.geometry`/`mesh.material` and bakes a static copy
 *     of the geometry per mesh anyway, so sharing is safe and material edits
 *     stay single-sourced);
 *   - has `matrixWorld = parentMatrixWorld · instanceMatrix[i]` baked, with
 *     `matrixAutoUpdate = false` so the fork doesn't recompute it from a
 *     (nonexistent) parent chain;
 *   - keeps the primitive `name` so `findMeshByPrimitiveId` still resolves the
 *     primitive id (multiple meshes now share one id — that is intentional;
 *     `findMeshByPrimitiveId` returns the first, which is sufficient for the
 *     material-only fast path that re-uploads one material slot).
 *
 * Returns the array of expanded child meshes. Does NOT mutate the scene graph
 * here — the caller swaps the InstancedMesh out for the children.
 */
export function expandInstancedMesh(im: InstancedMesh): Mesh[] {
  const count = im.count;
  const out: Mesh[] = [];
  // The InstancedMesh's own matrixWorld (the per-primitive transform of the
  // instanced-mesh node itself). vitrumSceneToThree leaves it identity, but
  // composing is robust against future per-primitive transforms.
  const parentWorld = im.matrixWorld;
  for (let i = 0; i < count; i += 1) {
    im.getMatrixAt(i, _instanceMatrix);
    _composed.multiplyMatrices(parentWorld, _instanceMatrix);

    const mesh = new Mesh(im.geometry, im.material);
    mesh.name = im.name;
    mesh.castShadow = im.castShadow;
    mesh.receiveShadow = im.receiveShadow;
    mesh.visible = im.visible;
    mesh.matrixAutoUpdate = false;
    mesh.matrix.copy(_composed);
    mesh.matrixWorld.copy(_composed);
    // Tag the expanded children so callers can recognise them (e.g. dispose
    // logic that must avoid double-disposing the shared geometry/material).
    mesh.userData['vitrumExpandedInstanceOf'] = im.uuid;
    mesh.userData['vitrumExpandedInstanceIndex'] = i;
    out.push(mesh);
  }
  return out;
}

/**
 * Walk a `THREE.Scene` and replace every `THREE.InstancedMesh` with its N
 * expanded `THREE.Mesh` children (see {@link expandInstancedMesh}). Mutates
 * the scene in place and returns it. The geometry + material of each
 * InstancedMesh are NOT disposed — they are now owned by the expanded
 * children (which share the single instances), so disposal happens once when
 * the children are torn down.
 *
 * Idempotency: a scene with no InstancedMesh is returned unchanged.
 */
export function expandInstancedMeshesInScene(scene: ThreeScene): ThreeScene {
  // Ensure world matrices are current before reading parentWorld /
  // instanceMatrix — vitrumSceneToThree does not call updateMatrixWorld, and
  // some hosts may have mutated transforms. Guarded because callers may pass a
  // minimal scene-like object (e.g. test stubs) that lacks the full
  // Object3D surface; world-matrix refresh is a best-effort correctness step,
  // not a hard precondition for the per-instance bake below.
  if (typeof scene.updateMatrixWorld === 'function') {
    scene.updateMatrixWorld(true);
  }
  if (typeof scene.traverse !== 'function') return scene;

  // Collect first (mutating children during traversal is unsafe).
  const instanced: InstancedMesh[] = [];
  scene.traverse((o) => {
    if (isInstancedMesh(o)) instanced.push(o);
  });

  for (const im of instanced) {
    const children = expandInstancedMesh(im);
    const parent = im.parent;
    if (parent == null) continue;
    parent.remove(im);
    for (const child of children) parent.add(child);
  }
  return scene;
}

/**
 * Collect EVERY mesh in `root` whose `name`/`uuid` equals `id`. For an
 * expanded instanced-mesh this returns all N children (they share the
 * primitive id); for a plain mesh it returns the single match.
 *
 * `findMeshByPrimitiveId` (in `@vitrum/three-bindings`) returns only the
 * FIRST match — fine for resolving a mesh, but a material-only
 * `updatePrimitive` on an expanded instanced-mesh must re-point the material
 * on ALL N children (each holds its own `mesh.material` reference, and
 * `applyVitrumMaterialToMesh` REPLACES `mesh.material` rather than mutating
 * in place), or only one instance would pick up the edit. This helper exists
 * so the engine's material fast path stays correct for instanced meshes.
 */
export function findAllMeshesByPrimitiveId(root: Object3D, id: string): TMesh[] {
  const out: TMesh[] = [];
  root.traverseVisible((obj) => {
    if ((obj as TMesh).isMesh !== true) return;
    if (obj.uuid === id || obj.name === id) out.push(obj as TMesh);
  });
  return out;
}
