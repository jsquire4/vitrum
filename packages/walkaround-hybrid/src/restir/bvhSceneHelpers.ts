/**
 * Shared scene-walk helpers for ReSTIR BVH construction (merged + TLAS paths).
 */

import * as THREE from 'three';
import type { Scene, Vec3 } from '@vitrum/core';

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

interface ExtraEmitterTri {
  vA: [number, number, number];
  vB: [number, number, number];
  vC: [number, number, number];
  normal: [number, number, number];
  area: number;
  Le: [number, number, number];
}

/**
 * THREE-free counterpart to {@link collectRectAreaLightEmitterTris}: derive the
 * extra ReSTIR emitter triangles for every `kind: 'rect-area'` emitter DIRECTLY
 * from a `@vitrum/core` `Scene`, with NO `vitrumSceneToThree` round-trip and no
 * `THREE.RectAreaLight`.
 *
 * Geometry parity with the THREE path (verified algebraically against
 * `vitrumSceneToThree.buildRectAreaLight` + `collectRectAreaLightEmitterTris`):
 *
 *  - `buildRectAreaLight` builds an orthonormal basis X=normalize(uAxis),
 *    Y=normalize(vAxis), Z=X×Y, with full width `2|uAxis|` / height `2|vAxis|`,
 *    then `collectRectAreaLightEmitterTris` derives the four corners as
 *    `(±w/2, ±h/2, 0)·matrixWorld`. Because the half-width is EXACTLY `|uAxis|`
 *    and X is EXACTLY `normalize(uAxis)`, that corner reduces to
 *    `position ± uAxis ± vAxis` — the four corners computed here directly.
 *  - The face normal is `-(Z column of matrixWorld) = -normalize(X×Y) =
 *    -normalize(uAxis × vAxis)` (THREE negates the basis Z).
 *  - Per-tri area is `0.5·|(LR−LL) × (UR−LL)|` — the same Möller cross the THREE
 *    path takes (LL,LR,UR) over.
 *  - `Le = color · intensity` per emitter (folded into Le, as the THREE path
 *    does — the WGSL `EmitterTri.Le` is the only radiance the shade kernel reads).
 *
 * Rejections mirror the THREE path exactly: a degenerate basis
 * (`|uAxis × vAxis|² < 1e-12`, the `buildRectAreaLight` null-return) and a
 * sub-`1e-8` triangle cross-length are both skipped.
 */
export function collectRectAreaEmitterTrisFromCore(scene: Scene): ExtraEmitterTri[] {
  const out: ExtraEmitterTri[] = [];
  for (const e of scene.emitters) {
    if (e.kind !== 'rect-area') continue;
    const p = e.position;
    const u = e.uAxis;
    const v = e.vAxis;

    // Z = uAxis × vAxis (un-normalized) — its length gates the degenerate basis.
    const zx = u[1] * v[2] - u[2] * v[1];
    const zy = u[2] * v[0] - u[0] * v[2];
    const zz = u[0] * v[1] - u[1] * v[0];
    const zLenSq = zx * zx + zy * zy + zz * zz;
    // buildRectAreaLight returns null (→ skipped) when the basis Z is degenerate.
    if (zLenSq < 1e-12) continue;
    const zLen = Math.sqrt(zLenSq);
    // Face normal = -normalize(X × Y) = -normalize(uAxis × vAxis).
    const N: [number, number, number] = [-zx / zLen, -zy / zLen, -zz / zLen];

    // Four world corners = position ± uAxis ± vAxis (see docstring for the
    // exact-identity reduction from the THREE matrixWorld corners).
    const ll: [number, number, number] = [p[0] - u[0] - v[0], p[1] - u[1] - v[1], p[2] - u[2] - v[2]];
    const lr: [number, number, number] = [p[0] + u[0] - v[0], p[1] + u[1] - v[1], p[2] + u[2] - v[2]];
    const ur: [number, number, number] = [p[0] + u[0] + v[0], p[1] + u[1] + v[1], p[2] + u[2] + v[2]];
    const ul: [number, number, number] = [p[0] - u[0] + v[0], p[1] - u[1] + v[1], p[2] - u[2] + v[2]];

    // Per-tri area = 0.5·|(LR−LL) × (UR−LL)| (THREE takes this over LL,LR,UR).
    const abx = lr[0] - ll[0], aby = lr[1] - ll[1], abz = lr[2] - ll[2];
    const acx = ur[0] - ll[0], acy = ur[1] - ll[1], acz = ur[2] - ll[2];
    const cx = aby * acz - abz * acy;
    const cy = abz * acx - abx * acz;
    const cz = abx * acy - aby * acx;
    const crossLen = Math.sqrt(cx * cx + cy * cy + cz * cz);
    if (crossLen < 1e-8) continue;
    const triArea = crossLen * 0.5;

    const col: Vec3 = e.color;
    const I = e.intensity;
    const Le: [number, number, number] = [col[0] * I, col[1] * I, col[2] * I];

    // Two tris (LL,LR,UR) + (LL,UR,UL) — identical winding to the THREE path.
    out.push({ vA: ll, vB: lr, vC: ur, normal: N, area: triArea, Le });
    out.push({ vA: ll, vB: ur, vC: ul, normal: N, area: triArea, Le });
  }
  return out;
}
