/**
 * Emitter-geometry helpers for ReSTIR and DDGI area-light packing.
 *
 * Converts `@vitrum/core` emitter definitions (rect-area, disc-area,
 * mesh-area, point, spot) into the flat GPU buffer layouts consumed by
 * the DDGI probe-NEE kernel and the ReSTIR shade pass.
 */

import type { Mat4, Scene, Vec3 } from '@vitrum/core';

const IDENTITY_MAT4 = new Float32Array([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
]);

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
 * Derive the extra ReSTIR emitter triangles for every `kind: 'rect-area'` emitter
 * DIRECTLY from a `@vitrum/core` `Scene`.
 *
 * Despite the legacy rect-only name, this also collects `disc-area` emitters as
 * a 32-triangle equal-area fan. `mesh-area` emitters are NOT expanded here because
 * the ReSTIR path receives their triangles through the merged world-space
 * geometry stream (adding them again as extraEmitters would double-count them).
 * For the DDGI probe-NEE path, which has no geometry stream, use
 * {@link collectMeshAreaEmitterTrisFromCore} and concatenate:
 * `[...collectRectAreaEmitterTrisFromCore(s), ...collectMeshAreaEmitterTrisFromCore(s)]`.
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
    if (zLenSq < 1e-12) continue;
    const zLen = Math.sqrt(zLenSq);
    // Face normal = -normalize(X × Y) = -normalize(uAxis × vAxis).
    const N: [number, number, number] = [-zx / zLen, -zy / zLen, -zz / zLen];

    // Four world corners = position ± uAxis ± vAxis.
    const ll: [number, number, number] = [p[0] - u[0] - v[0], p[1] - u[1] - v[1], p[2] - u[2] - v[2]];
    const lr: [number, number, number] = [p[0] + u[0] - v[0], p[1] + u[1] - v[1], p[2] + u[2] - v[2]];
    const ur: [number, number, number] = [p[0] + u[0] + v[0], p[1] + u[1] + v[1], p[2] + u[2] + v[2]];
    const ul: [number, number, number] = [p[0] - u[0] + v[0], p[1] - u[1] + v[1], p[2] - u[2] + v[2]];

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
 * Expand `mesh-area` emitters from a `@vitrum/core` `Scene` into world-space
 * triangles for DDGI probe NEE. This is the DDGI-only counterpart to
 * {@link collectRectAreaEmitterTrisFromCore}; do NOT add this output to the
 * ReSTIR `extraEmitters` stream — that would double-count the triangles that
 * the merged world-space geometry stream already carries.
 */
export function collectMeshAreaEmitterTrisFromCore(scene: Scene): ExtraEmitterTri[] {
  const meshAreaEmitters = scene.emitters.filter((e) => e.kind === 'mesh-area');
  if (meshAreaEmitters.length === 0) return [];

  const primById = new Map<string, Scene['primitives'][number]>();
  for (const p of scene.primitives) {
    primById.set(String(p.id), p);
  }

  const out: ExtraEmitterTri[] = [];
  for (const e of meshAreaEmitters) {
    if (e.kind !== 'mesh-area') continue;
    const prim = primById.get(String(e.meshId));
    if (prim == null) {
      console.warn(
        `[collectMeshAreaEmitterTrisFromCore] mesh-area emitter id="${String(e.id)}" ` +
        `references meshId="${String(e.meshId)}" which is not in scene.primitives — skipped.`,
      );
      continue;
    }
    if (prim.kind !== 'mesh' && prim.kind !== 'skinned-mesh') continue;

    const positions = prim.positions;
    const indices = prim.indices;
    const transform: Mat4 | undefined = (prim as { transform?: Mat4 }).transform;
    const Le = emitterLe(e.color, e.intensity);

    const m: ArrayLike<number> = transform != null ? (transform) : IDENTITY_MAT4;

    const triCount = indices != null
      ? Math.floor(indices.length / 3)
      : Math.floor(positions.length / 9);

    for (let ti = 0; ti < triCount; ti++) {
      const i0 = indices != null ? (indices[ti * 3]!)     : (ti * 3);
      const i1 = indices != null ? (indices[ti * 3 + 1]!) : (ti * 3 + 1);
      const i2 = indices != null ? (indices[ti * 3 + 2]!) : (ti * 3 + 2);

      const vA = transformPoint(m, positions[i0 * 3]!, positions[i0 * 3 + 1]!, positions[i0 * 3 + 2]!);
      const vB = transformPoint(m, positions[i1 * 3]!, positions[i1 * 3 + 1]!, positions[i1 * 3 + 2]!);
      const vC = transformPoint(m, positions[i2 * 3]!, positions[i2 * 3 + 1]!, positions[i2 * 3 + 2]!);

      const abx = vB[0] - vA[0], aby = vB[1] - vA[1], abz = vB[2] - vA[2];
      const acx = vC[0] - vA[0], acy = vC[1] - vA[1], acz = vC[2] - vA[2];
      const nx = aby * acz - abz * acy;
      const ny = abz * acx - abx * acz;
      const nz = abx * acy - aby * acx;
      const crossLen = Math.sqrt(nx * nx + ny * ny + nz * nz);
      if (crossLen < 1e-8) continue;
      const area = crossLen * 0.5;
      const invLen = 1 / crossLen;
      out.push({
        vA, vB, vC,
        normal: [nx * invLen, ny * invLen, nz * invLen],
        area,
        Le,
      });
    }
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
      data[out * S + 8]  = 0;  data[out * S + 9]  = 0;  data[out * S + 10] = 0;  data[out * S + 11] = 1;
      data[out * S + 12] = 0;  data[out * S + 13] = 0;  data[out * S + 14] = 0;  data[out * S + 15] = 0;
      out++;
    } else if (e.kind === 'spot') {
      const [r, g, b] = e.color;
      const i = e.intensity;
      const [px, py, pz] = e.position;
      const dirRaw = e.direction;
      const dx = dirRaw[0], dy = dirRaw[1], dz = dirRaw[2];
      const dLen = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const nx = dLen > 1e-8 ? dx / dLen : 0;
      const ny = dLen > 1e-8 ? dy / dLen : -1;
      const nz = dLen > 1e-8 ? dz / dLen : 0;
      const outerHalf = e.angle;
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
