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

import { InstancedMesh as TInstancedMesh, Matrix4, Mesh } from 'three';
import type { InstancedMesh, Mesh as TMesh, Object3D, Scene as ThreeScene } from 'three';
import type { Mat4 } from '@vitrum/core';

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

/**
 * Re-expand a single `instanced-mesh` primitive after an `instances`-only patch
 * (an instance-COUNT change, or a per-instance transform change) WITHOUT a full
 * `setScene` teardown.
 *
 * At `setScene` time, `expandInstancedMeshesInScene` already replaced the source
 * `THREE.InstancedMesh` with N standalone `THREE.Mesh` children (all sharing the
 * single geometry + material, all carrying `name === primitiveId`). This helper:
 *   1. locates those existing children by primitive id;
 *   2. reuses their SHARED geometry + material (so the MaterialsTexture slot is
 *      untouched — instances never change material, which is why the caller can
 *      take the fork's geometry-only regen that skips `updateMaterials()`);
 *   3. removes the old children from their common parent;
 *   4. bakes a fresh set of N' standalone meshes from the new `instances`
 *      matrices (via {@link expandInstancedMesh} on a throwaway InstancedMesh
 *      that is never added to the scene graph) and adds them to that parent.
 *
 * The new children re-use the SAME `vitrumExpandedInstanceOf` uuid tag as the
 * old ones, so `disposeObject3DTree`'s identity-dedup (which collapses the N
 * shared-geometry/material disposes into one) keeps working unchanged: every
 * child still points at the one shared geometry + material object, and the tag
 * documents that they are expanded instances of one source.
 *
 * Returns `false` when the primitive has no existing expanded children in the
 * root (nothing to re-expand — caller must fall back to a full `setScene`);
 * `true` when the swap completed. Does NOT trigger the fork geometry regen — the
 * caller does that after the swap (it owns the path-tracer handle).
 */
export function reexpandInstancedMeshInScene(
  root: Object3D,
  primitiveId: string,
  instances: ReadonlyArray<Mat4>,
): boolean {
  const existing = findAllMeshesByPrimitiveId(root, primitiveId);
  if (existing.length === 0) return false;

  // The expanded children all share ONE geometry + ONE material (built once by
  // the shared converter, never cloned in expansion). Read them off the first
  // child; reusing them keeps the material slot single-sourced.
  const first = existing[0]!;
  const sharedGeometry = first.geometry;
  const sharedMaterial = first.material;
  const parent = first.parent;
  if (parent == null) return false;

  // Preserve render flags + the source-InstancedMesh uuid tag so the new
  // children look identical (modulo count) to a fresh setScene expansion.
  const castShadow = first.castShadow;
  const receiveShadow = first.receiveShadow;
  const visible = first.visible;
  const sourceUuidTag = first.userData['vitrumExpandedInstanceOf'] as string | undefined;

  // Build a throwaway InstancedMesh purely to drive expandInstancedMesh — it is
  // never added to the scene. It reuses the shared geometry/material so the
  // expanded children point at the SAME objects the (removed) children did.
  const temp = new TInstancedMesh(sharedGeometry, sharedMaterial, instances.length);
  temp.name = primitiveId;
  temp.castShadow = castShadow;
  temp.receiveShadow = receiveShadow;
  temp.visible = visible;
  // The source InstancedMesh node carried identity matrixWorld in setScene
  // (instancedMeshPrimitiveToThree leaves it identity); keep that so each
  // child's baked transform is exactly the per-instance matrix.
  temp.matrixAutoUpdate = false;
  const m = new Matrix4();
  for (let i = 0; i < instances.length; i += 1) {
    m.fromArray(instances[i]! as unknown as ArrayLike<number>);
    temp.setMatrixAt(i, m);
  }
  temp.instanceMatrix.needsUpdate = true;

  const children = expandInstancedMesh(temp);
  // expandInstancedMesh tags children with the THROWAWAY temp.uuid; rewrite the
  // tag to the value the old children carried (when present) so the dispose
  // dedup + any tag-based introspection stays stable across re-expansions.
  if (sourceUuidTag != null) {
    for (const child of children) {
      child.userData['vitrumExpandedInstanceOf'] = sourceUuidTag;
    }
  }

  for (const child of existing) parent.remove(child);
  for (const child of children) parent.add(child);
  return true;
}
