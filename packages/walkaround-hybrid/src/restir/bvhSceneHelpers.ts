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

export interface ExtraEmitterTri {
  vA: [number, number, number];
  vB: [number, number, number];
  vC: [number, number, number];
  normal: [number, number, number];
  area: number;
  Le: [number, number, number];
}

/**
 * H18 — Pack an `ExtraEmitterTri[]` into the 80-byte-stride (5 × vec4f)
 * Float32Array layout consumed by the DDGI probe-ray kernel's
 * `ddgiEmitterTris` storage buffer and the RC `rc_emitters` buffer.
 *
 * Layout per entry (20 floats):
 *   [0..2]  vA.xyz + pad(0)
 *   [4..6]  vB.xyz + pad(0)
 *   [8..10] vC.xyz + pad(0)
 *   [12..14] normal.xyz + area
 *   [16..18] Le.rgb + pad(0)
 *
 * Returns a zero-count dummy (empty Float32Array) when `tris` is empty so
 * callers can safely pass `data` to a placeholder GPU buffer.
 */
export function packEmitterTrisForDDGI(tris: readonly ExtraEmitterTri[]): {
  data: Float32Array;
  count: number;
} {
  const count = tris.length;
  if (count === 0) return { data: new Float32Array(0), count: 0 };
  const STRIDE = 20; // 5 × vec4f = 20 floats = 80 bytes
  const data = new Float32Array(count * STRIDE);
  for (let i = 0; i < count; i++) {
    const t = tris[i]!;
    const base = i * STRIDE;
    data[base + 0]  = t.vA[0]!; data[base + 1]  = t.vA[1]!; data[base + 2]  = t.vA[2]!; data[base + 3]  = 0;
    data[base + 4]  = t.vB[0]!; data[base + 5]  = t.vB[1]!; data[base + 6]  = t.vB[2]!; data[base + 7]  = 0;
    data[base + 8]  = t.vC[0]!; data[base + 9]  = t.vC[1]!; data[base + 10] = t.vC[2]!; data[base + 11] = 0;
    data[base + 12] = t.normal[0]!; data[base + 13] = t.normal[1]!; data[base + 14] = t.normal[2]!; data[base + 15] = t.area;
    data[base + 16] = t.Le[0]!; data[base + 17] = t.Le[1]!; data[base + 18] = t.Le[2]!; data[base + 19] = 0;
  }
  return { data, count };
}

const DISC_AREA_TRIANGLE_COUNT = 32;
const TAU = Math.PI * 2;

function emitterLe(color: Vec3, intensity: number): [number, number, number] {
  return [color[0] * intensity, color[1] * intensity, color[2] * intensity];
}

function normalizeVec3(v: Vec3): [number, number, number] | null {
  const x = v[0];
  const y = v[1];
  const z = v[2];
  const lenSq = x * x + y * y + z * z;
  if (lenSq < 1e-12 || !Number.isFinite(lenSq)) return null;
  const invLen = 1 / Math.sqrt(lenSq);
  return [x * invLen, y * invLen, z * invLen];
}

function discTangentBasis(n: [number, number, number]): {
  tangent: [number, number, number];
  bitangent: [number, number, number];
} {
  const up: [number, number, number] = Math.abs(n[1]) > 0.999 ? [1, 0, 0] : [0, 1, 0];
  let tx = up[1] * n[2] - up[2] * n[1];
  let ty = up[2] * n[0] - up[0] * n[2];
  let tz = up[0] * n[1] - up[1] * n[0];
  const tLen = Math.sqrt(tx * tx + ty * ty + tz * tz);
  if (tLen > 1e-8) {
    tx /= tLen;
    ty /= tLen;
    tz /= tLen;
  }
  const bx = n[1] * tz - n[2] * ty;
  const by = n[2] * tx - n[0] * tz;
  const bz = n[0] * ty - n[1] * tx;
  return { tangent: [tx, ty, tz], bitangent: [bx, by, bz] };
}

function discPoint(
  center: Vec3,
  tangent: [number, number, number],
  bitangent: [number, number, number],
  radius: number,
  theta: number,
): [number, number, number] {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  return [
    center[0] + radius * (tangent[0] * c + bitangent[0] * s),
    center[1] + radius * (tangent[1] * c + bitangent[1] * s),
    center[2] + radius * (tangent[2] * c + bitangent[2] * s),
  ];
}

/**
 * Core-scene counterpart to {@link collectRectAreaLightEmitterTris}: derive the
 * extra ReSTIR emitter triangles for every `kind: 'rect-area'` emitter DIRECTLY
 * from a `@vitrum/core` `Scene`.
 *
 * Geometry parity with the historical bridge path:
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
 *
 * Despite the legacy rect-only name, this also collects `disc-area` emitters as
 * a 32-triangle equal-area fan. The fan radius is scaled so the represented
 * polygon area equals the analytic disc area, preserving total power and the
 * area-PDF used by ReSTIR. `mesh-area` emitters are not expanded here: material
 * emissive mesh triangles already enter through the world-space emitter stream.
 */
export function collectRectAreaEmitterTrisFromCore(scene: Scene): ExtraEmitterTri[] {
  const out: ExtraEmitterTri[] = [];
  for (const e of scene.emitters) {
    if (e.kind === 'disc-area') {
      if (e.radius <= 0 || !Number.isFinite(e.radius)) continue;
      const n = normalizeVec3(e.normal);
      if (n == null) continue;

      const { tangent, bitangent } = discTangentBasis(n);
      const segmentAngle = TAU / DISC_AREA_TRIANGLE_COUNT;
      const areaPreservingRadius = e.radius * Math.sqrt(segmentAngle / Math.sin(segmentAngle));
      const triArea = Math.PI * e.radius * e.radius / DISC_AREA_TRIANGLE_COUNT;
      const N: [number, number, number] = [-n[0], -n[1], -n[2]];
      const Le = emitterLe(e.color, e.intensity);

      for (let i = 0; i < DISC_AREA_TRIANGLE_COUNT; i += 1) {
        const curr = discPoint(e.position, tangent, bitangent, areaPreservingRadius, i * segmentAngle);
        const next = discPoint(e.position, tangent, bitangent, areaPreservingRadius, (i + 1) * segmentAngle);
        out.push({
          vA: [e.position[0], e.position[1], e.position[2]],
          vB: next,
          vC: curr,
          normal: N,
          area: triArea,
          Le,
        });
      }
      continue;
    }

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

    const Le = emitterLe(e.color, e.intensity);

    // Two tris (LL,LR,UR) + (LL,UR,UL) — identical winding to the THREE path.
    out.push({ vA: ll, vB: lr, vC: ur, normal: N, area: triArea, Le });
    out.push({ vA: ll, vB: ur, vC: ul, normal: N, area: triArea, Le });
  }
  return out;
}

/**
 * H41 — Analytic point/spot emitter struct layout for the shade NEE buffer.
 *
 * Stride: 4 × vec4f = 64 bytes = 16 floats per entry.
 *   [0..2]  position.xyz + pad(0)
 *   [4..6]  color.rgb (linear, pre-multiplied by intensity) + pad(0)
 *   [8..10] direction.xyz (toward-light; (0,0,0) for omnidirectional point)
 *          + cosInner (cosine of inner cone half-angle; 1.0 for point = no cone)
 *   [12]   cosOuter (cosine of outer cone half-angle; 0.0 for point = no cone)
 *           pad×3
 *
 * Point emitters: direction=(0,0,0), cosInner=1, cosOuter=0.
 * Spot emitters: direction=normalize(axis), cosInner=cos(innerHalfAngle),
 *                cosOuter=cos(outerHalfAngle).
 *
 * The WGSL shade NEE loop uses `Le/(d²+ε)·cosθ·smoothstep` falloff:
 *   smoothstep(cosOuter, cosInner, dot(-dir, towardLight)) for spot cone
 *   = 1 for point (cosOuter=0 < cosInner=1, always within cone).
 */
const ANALYTIC_LIGHT_STRIDE_FLOATS = 16; // 4 × vec4f

export interface PackedAnalyticLights {
  data: Float32Array;
  count: number;
}

/**
 * Pack `point` and `spot` emitters from a `@vitrum/core` `Scene` into the
 * 64-byte-stride analytic-lights buffer for shade NEE (H41).
 *
 * Returns a 16-float placeholder (1 dummy entry, count=0) when the scene has
 * no point/spot emitters, so the bind group is always valid (WebGPU storage
 * bindings must be non-empty).
 */
export function packAnalyticPointSpotEmitters(scene: Scene): PackedAnalyticLights {
  const emitters = scene.emitters.filter(
    (e) => e.kind === 'point' || e.kind === 'spot',
  );
  if (emitters.length === 0) {
    // Dummy placeholder — 1 zeroed entry, count=0 so the shader skips the loop.
    return { data: new Float32Array(ANALYTIC_LIGHT_STRIDE_FLOATS), count: 0 };
  }

  const S = ANALYTIC_LIGHT_STRIDE_FLOATS;
  const data = new Float32Array(emitters.length * S);
  let out = 0;
  for (const e of emitters) {
    if (e.kind === 'point') {
      const [r, g, b] = e.color;
      const i = e.intensity;
      const [px, py, pz] = e.position;
      data[out * S + 0]  = px;  data[out * S + 1]  = py;  data[out * S + 2]  = pz;  data[out * S + 3]  = 0;
      data[out * S + 4]  = r * i; data[out * S + 5] = g * i; data[out * S + 6] = b * i; data[out * S + 7] = 0;
      // point: direction=(0,0,0), cosInner=1 (no cone)
      data[out * S + 8]  = 0;  data[out * S + 9]  = 0;  data[out * S + 10] = 0;  data[out * S + 11] = 1;
      // cosOuter=0 (no cone outer), pad
      data[out * S + 12] = 0;  data[out * S + 13] = 0;  data[out * S + 14] = 0;  data[out * S + 15] = 0;
      out++;
    } else if (e.kind === 'spot') {
      const [r, g, b] = e.color;
      const i = e.intensity;
      const [px, py, pz] = e.position;
      // SpotEmitter.direction is the direction the spot points (unit vector).
      const dirRaw = e.direction;
      const dx = dirRaw[0], dy = dirRaw[1], dz = dirRaw[2];
      const dLen = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const nx = dLen > 1e-8 ? dx / dLen : 0;
      const ny = dLen > 1e-8 ? dy / dLen : -1;
      const nz = dLen > 1e-8 ? dz / dLen : 0;
      // SpotEmitter.angle is the outer half-cone. penumbra [0..1] shrinks the
      // inner cone: innerAngle = outerAngle * (1 - penumbra). At penumbra=0 →
      // hard edge (inner==outer); at penumbra=1 → full penumbra (inner=0).
      const outerHalf = e.angle;  // radians, half-cone
      const penumbra = e.penumbra ?? 0;
      const innerHalf = outerHalf * (1 - penumbra);
      const cosInner = Math.cos(innerHalf);
      const cosOuter = Math.cos(outerHalf);
      data[out * S + 0]  = px;  data[out * S + 1]  = py;  data[out * S + 2]  = pz;  data[out * S + 3]  = 0;
      data[out * S + 4]  = r * i; data[out * S + 5] = g * i; data[out * S + 6] = b * i; data[out * S + 7] = 0;
      data[out * S + 8]  = nx;  data[out * S + 9]  = ny;  data[out * S + 10] = nz;  data[out * S + 11] = cosInner;
      data[out * S + 12] = cosOuter; data[out * S + 13] = 0; data[out * S + 14] = 0; data[out * S + 15] = 0;
      out++;
    }
  }
  return { data, count: out };
}
