/**
 * Shared scene-walk helpers for ReSTIR BVH construction (merged + TLAS paths).
 */

import * as THREE from 'three';

/**
 * Walk `sceneRoots` once to find each named mesh and snapshot its
 * `matrixWorld.elements` — required by `HybridEngine.updatePrimitive`'s
 * transform-only refit path.
 */
export function enrichMeshVertexRangesWithMatrix(
  sceneRoots: THREE.Object3D[],
  rawRanges: ReadonlyArray<{
    name: string;
    vertexStart: number;
    vertexCount: number;
    triStart: number;
    triCount: number;
  }>,
): ReadonlyArray<{
  name: string;
  vertexStart: number;
  vertexCount: number;
  triStart: number;
  triCount: number;
  matrixWorldAtBuild: Float32Array;
}> {
  const byName = new Map<string, THREE.Object3D>();
  for (const root of sceneRoots) {
    root.traverseVisible((obj) => {
      if (!byName.has(obj.name)) byName.set(obj.name, obj);
    });
  }
  return rawRanges.map((r) => {
    const obj = byName.get(r.name);
    const m = obj
      ? new Float32Array(obj.matrixWorld.elements)
      : new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
    return {
      name: r.name,
      vertexStart: r.vertexStart,
      vertexCount: r.vertexCount,
      triStart: r.triStart,
      triCount: r.triCount,
      matrixWorldAtBuild: m,
    };
  });
}

/** RectAreaLight → emitter triangles for ReSTIR DI (not in merged BVH). */
export function collectRectAreaLightEmitterTris(
  sceneRoots: THREE.Object3D[],
): {
  vA: [number, number, number];
  vB: [number, number, number];
  vC: [number, number, number];
  normal: [number, number, number];
  area: number;
  Le: [number, number, number];
}[] {
  const out: {
    vA: [number, number, number];
    vB: [number, number, number];
    vC: [number, number, number];
    normal: [number, number, number];
    area: number;
    Le: [number, number, number];
  }[] = [];
  const _ll = new THREE.Vector3();
  const _lr = new THREE.Vector3();
  const _ur = new THREE.Vector3();
  const _ul = new THREE.Vector3();
  const _normal = new THREE.Vector3();
  const _ab = new THREE.Vector3();
  const _ac = new THREE.Vector3();

  for (const root of sceneRoots) {
    root.updateMatrixWorld(true);
    root.traverseVisible((obj) => {
      if (!(obj instanceof THREE.RectAreaLight)) return;
      const light = obj;
      const wHalf = light.width * 0.5;
      const hHalf = light.height * 0.5;

      _ll.set(-wHalf, -hHalf, 0).applyMatrix4(light.matrixWorld);
      _lr.set(wHalf, -hHalf, 0).applyMatrix4(light.matrixWorld);
      _ur.set(wHalf, hHalf, 0).applyMatrix4(light.matrixWorld);
      _ul.set(-wHalf, hHalf, 0).applyMatrix4(light.matrixWorld);

      _ab.subVectors(_lr, _ll);
      _ac.subVectors(_ur, _ll);
      _normal.crossVectors(_ab, _ac);
      const crossLen = _normal.length();
      if (crossLen < 1e-8) return;

      _normal.setFromMatrixColumn(light.matrixWorld, 2).normalize().negate();

      const triArea = crossLen * 0.5;
      const c = light.color;
      const I = light.intensity;
      const Le: [number, number, number] = [c.r * I, c.g * I, c.b * I];
      const N: [number, number, number] = [_normal.x, _normal.y, _normal.z];

      out.push({
        vA: [_ll.x, _ll.y, _ll.z],
        vB: [_lr.x, _lr.y, _lr.z],
        vC: [_ur.x, _ur.y, _ur.z],
        normal: N,
        area: triArea,
        Le,
      });
      out.push({
        vA: [_ll.x, _ll.y, _ll.z],
        vB: [_ur.x, _ur.y, _ur.z],
        vC: [_ul.x, _ul.y, _ul.z],
        normal: N,
        area: triArea,
        Le,
      });
    });
  }
  return out;
}
