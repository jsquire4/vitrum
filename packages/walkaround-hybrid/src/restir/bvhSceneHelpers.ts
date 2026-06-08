/**
 * Shared scene-walk helpers for ReSTIR BVH construction (merged + TLAS paths).
 */

import type { Mat4, Scene, Vec3 } from '@vitrum/core';

const IDENTITY_MAT4 = new Float32Array([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
]);

type RawMeshVertexRange = {
  name: string;
  vertexStart: number;
  vertexCount: number;
  triStart: number;
  triCount: number;
};

type MeshVertexRangeWithMatrix = RawMeshVertexRange & {
  matrixWorldAtBuild: Float32Array;
};

interface MatrixWorldLike {
  readonly elements: ArrayLike<number>;
}

interface Object3DLike {
  readonly name: string;
  readonly uuid?: string;
  readonly type?: string;
  readonly isRectAreaLight?: boolean;
  readonly matrixWorld: MatrixWorldLike;
  updateMatrixWorld?: (force?: boolean) => void;
  traverseVisible: (cb: (obj: Object3DLike) => void) => void;
}

interface RectAreaLightLike extends Object3DLike {
  readonly width: number;
  readonly height: number;
  readonly color: { readonly r: number; readonly g: number; readonly b: number };
  readonly intensity: number;
}

function cloneMat4(m: Mat4 | Float32Array | undefined): Float32Array {
  return m != null ? new Float32Array(m) : new Float32Array(IDENTITY_MAT4);
}

function isRectAreaLightLike(obj: Object3DLike): obj is RectAreaLightLike {
  const candidate = obj as Partial<RectAreaLightLike>;
  return (
    (obj.isRectAreaLight === true || obj.type === 'RectAreaLight') &&
    typeof candidate.width === 'number' &&
    typeof candidate.height === 'number' &&
    candidate.color != null &&
    typeof candidate.intensity === 'number'
  );
}

function transformPoint(
  m: ArrayLike<number>,
  x: number,
  y: number,
  z: number,
): [number, number, number] {
  const w = m[3]! * x + m[7]! * y + m[11]! * z + m[15]!;
  const invW = w !== 0 && Number.isFinite(w) ? 1 / w : 1;
  return [
    (m[0]! * x + m[4]! * y + m[8]! * z + m[12]!) * invW,
    (m[1]! * x + m[5]! * y + m[9]! * z + m[13]!) * invW,
    (m[2]! * x + m[6]! * y + m[10]! * z + m[14]!) * invW,
  ];
}

function negatedNormalizedColumnZ(m: ArrayLike<number>): [number, number, number] {
  let x = -(m[8] ?? 0);
  let y = -(m[9] ?? 0);
  let z = -(m[10] ?? 1);
  const len = Math.sqrt(x * x + y * y + z * z);
  if (len > 1e-8) {
    x /= len;
    y /= len;
    z /= len;
  }
  return [x, y, z];
}

/**
 * Walk `sceneRoots` once to find each named mesh and snapshot its
 * `matrixWorld.elements` — required by `HybridEngine.updatePrimitive`'s
 * transform-only refit path.
 */
export function enrichMeshVertexRangesWithMatrix(
  sceneRoots: Object3DLike[],
  rawRanges: ReadonlyArray<RawMeshVertexRange>,
): ReadonlyArray<MeshVertexRangeWithMatrix> {
  const byName = new Map<string, Object3DLike>();
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

/**
 * Core-scene counterpart to {@link enrichMeshVertexRangesWithMatrix}. It
 * derives the build-time world matrix snapshots from the core primitive
 * transforms directly, so core BVH/update paths do not need a synthesized
 * `THREE.Scene` solely to populate `matrixWorldAtBuild`.
 */
export function enrichMeshVertexRangesWithCoreMatrix(
  scene: Scene,
  rawRanges: ReadonlyArray<RawMeshVertexRange>,
): ReadonlyArray<MeshVertexRangeWithMatrix> {
  const primitiveById = new Map<string, Scene['primitives'][number]>();
  for (const primitive of scene.primitives) {
    primitiveById.set(String(primitive.id), primitive);
  }
  const instancedOccurrences = new Map<string, number>();
  return rawRanges.map((r) => {
    const primitive = primitiveById.get(r.name);
    let matrix: Float32Array;
    if (primitive?.kind === 'instanced-mesh') {
      const occurrence = instancedOccurrences.get(r.name) ?? 0;
      instancedOccurrences.set(r.name, occurrence + 1);
      matrix = cloneMat4(primitive.instances[occurrence]);
    } else if (primitive?.kind === 'mesh' || primitive?.kind === 'skinned-mesh') {
      matrix = cloneMat4(primitive.transform);
    } else {
      matrix = cloneMat4(undefined);
    }
    return {
      name: r.name,
      vertexStart: r.vertexStart,
      vertexCount: r.vertexCount,
      triStart: r.triStart,
      triCount: r.triCount,
      matrixWorldAtBuild: matrix,
    };
  });
}

/** RectAreaLight → emitter triangles for ReSTIR DI (not in merged BVH). */
export function collectRectAreaLightEmitterTris(
  sceneRoots: Object3DLike[],
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
  for (const root of sceneRoots) {
    root.updateMatrixWorld?.(true);
    root.traverseVisible((obj) => {
      if (!isRectAreaLightLike(obj)) return;
      const light = obj;
      const wHalf = light.width * 0.5;
      const hHalf = light.height * 0.5;

      const m = light.matrixWorld.elements;
      const ll = transformPoint(m, -wHalf, -hHalf, 0);
      const lr = transformPoint(m, wHalf, -hHalf, 0);
      const ur = transformPoint(m, wHalf, hHalf, 0);
      const ul = transformPoint(m, -wHalf, hHalf, 0);

      const abx = lr[0] - ll[0], aby = lr[1] - ll[1], abz = lr[2] - ll[2];
      const acx = ur[0] - ll[0], acy = ur[1] - ll[1], acz = ur[2] - ll[2];
      const cx = aby * acz - abz * acy;
      const cy = abz * acx - abx * acz;
      const cz = abx * acy - aby * acx;
      const crossLen = Math.sqrt(cx * cx + cy * cy + cz * cz);
      if (crossLen < 1e-8) return;

      const N = negatedNormalizedColumnZ(m);

      const triArea = crossLen * 0.5;
      const c = light.color;
      const I = light.intensity;
      const Le: [number, number, number] = [c.r * I, c.g * I, c.b * I];

      out.push({
        vA: ll,
        vB: lr,
        vC: ur,
        normal: N,
        area: triArea,
        Le,
      });
      out.push({
        vA: ll,
        vB: ur,
        vC: ul,
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
