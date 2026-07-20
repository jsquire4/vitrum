/**
 * Emitter-geometry helpers for ReSTIR and DDGI area-light packing.
 *
 * Converts `@vitrum/core` emitter definitions (rect-area, disc-area,
 * mesh-area, point, spot) into the flat GPU buffer layouts consumed by
 * the DDGI probe-NEE kernel and the ReSTIR shade pass.
 */

import type { EngineWarning, Mat4, MaterialSpec, Scene, Vec3 } from '@vitrum/core';
import {
  emissiveMapTriangleSubdivisionLevel,
  estimateMaterialSpecEmissiveLeOverTriangle,
  forEachBarycentricSubTriangle,
  forEachEmissiveMapTexelSubTriangle,
  type BarycentricWeights,
  type PrimitiveTlasBinding,
} from '@vitrum/shared-bvh';

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

function uvAt(uvs: Float32Array | undefined, vertex: number): [number, number] {
  if (uvs == null) return [0, 0];
  return [uvs[vertex * 2] ?? 0, uvs[vertex * 2 + 1] ?? 0];
}

function baryVec3(
  a: [number, number, number],
  b: [number, number, number],
  c: [number, number, number],
  w: BarycentricWeights,
): [number, number, number] {
  return [
    a[0] * w[0] + b[0] * w[1] + c[0] * w[2],
    a[1] * w[0] + b[1] * w[1] + c[1] * w[2],
    a[2] * w[0] + b[2] * w[1] + c[2] * w[2],
  ];
}

function baryUv2(
  a: readonly [number, number],
  b: readonly [number, number],
  c: readonly [number, number],
  w: BarycentricWeights,
): [number, number] {
  return [
    a[0] * w[0] + b[0] * w[1] + c[0] * w[2],
    a[1] * w[0] + b[1] * w[1] + c[1] * w[2],
  ];
}

export interface ExtraEmitterTri {
  vA: [number, number, number];
  vB: [number, number, number];
  vC: [number, number, number];
  normal: [number, number, number];
  area: number;
  Le: [number, number, number];
  castShadow?: boolean;
  sourceTriIndex?: number;
  sourceSubdivLevel?: number;
  sourceSubdivOrdinal?: number;
}

/**
 * H18 — Pack an `ExtraEmitterTri[]` into the 80-byte-stride (5 × vec4f)
 * Float32Array layout consumed by the DDGI probe-ray kernel's
 * `ddgiEmitterTris` storage buffer and the RC `rc_emitters` buffer.
 *
 * Layout per entry (20 floats):
 *   [0..2]  vA.xyz + sourceTriIndex (-1 = scalar fallback)
 *   [4..6]  vB.xyz + sourceSubdivLevel
 *   [8..10] vC.xyz + sourceSubdivOrdinal
 *   [12..14] normal.xyz + area
 *   [16..18] Le.rgb + castShadowDisabled
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
    const sourceTriIndex = t.sourceTriIndex ?? -1;
    data[base + 0]  = t.vA[0]!; data[base + 1]  = t.vA[1]!; data[base + 2]  = t.vA[2]!; data[base + 3]  = sourceTriIndex;
    data[base + 4]  = t.vB[0]!; data[base + 5]  = t.vB[1]!; data[base + 6]  = t.vB[2]!; data[base + 7]  = sourceTriIndex !== -1 ? t.sourceSubdivLevel ?? 1 : 1;
    data[base + 8]  = t.vC[0]!; data[base + 9]  = t.vC[1]!; data[base + 10] = t.vC[2]!; data[base + 11] = sourceTriIndex !== -1 ? t.sourceSubdivOrdinal ?? 0 : 0;
    data[base + 12] = t.normal[0]!; data[base + 13] = t.normal[1]!; data[base + 14] = t.normal[2]!; data[base + 15] = t.area;
    data[base + 16] = t.Le[0]!; data[base + 17] = t.Le[1]!; data[base + 18] = t.Le[2]!; data[base + 19] = t.castShadow === false ? 1 : 0;
  }
  return { data, count };
}

export interface CollectMeshAreaEmitterTrisOptions {
  readonly tlasPrimitiveBindings?: readonly PrimitiveTlasBinding[];
  readonly onWarning?: (warning: EngineWarning) => void;
  readonly warningPhase?: EngineWarning['phase'];
  readonly warningMethod?: string;
}

const DISC_AREA_TRIANGLE_COUNT = 32;
const TAU = Math.PI * 2;

function emitterLe(color: Vec3, intensity: number): [number, number, number] {
  return [color[0] * intensity, color[1] * intensity, color[2] * intensity];
}

function warnCollectMeshAreaEmitter(
  options: CollectMeshAreaEmitterTrisOptions,
  warning: EngineWarning,
): void {
  if (options.onWarning) {
    try {
      options.onWarning(warning);
    } catch {
      // Host warning callbacks must not break emitter packing.
    }
    return;
  }
  console.warn(warning.message);
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
          ...(e.castShadow !== undefined ? { castShadow: e.castShadow } : {}),
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
    out.push({
      vA: ll,
      vB: lr,
      vC: ur,
      normal: N,
      area: triArea,
      Le,
      ...(e.castShadow !== undefined ? { castShadow: e.castShadow } : {}),
    });
    out.push({
      vA: ll,
      vB: ur,
      vC: ul,
      normal: N,
      area: triArea,
      Le,
      ...(e.castShadow !== undefined ? { castShadow: e.castShadow } : {}),
    });
  }
  return out;
}

/** A world-space sub-triangle: 3 positions + its UV0 and UV1 vertex pairs.
 *  Groups the former 9 positional vertex args of the `pushTri` closure. */
interface SubTriangle {
  readonly pos: readonly [
    [number, number, number],
    [number, number, number],
    [number, number, number],
  ];
  readonly uv0: readonly [readonly [number, number], readonly [number, number], readonly [number, number]];
  readonly uv1: readonly [readonly [number, number], readonly [number, number], readonly [number, number]];
}

/** Optional per-sub-triangle metadata (provenance + radiance override). */
interface SubTriangleMeta {
  readonly sourceSubdivLevel?: number;
  readonly sourceSubdivOrdinal?: number;
  readonly radianceOverride?: readonly [number, number, number];
  readonly forceScalarLe?: boolean;
}

/**
 * Expand `mesh-area` emitters from a `@vitrum/core` `Scene` into world-space
 * triangles for DDGI probe NEE. This is the DDGI-only counterpart to
 * {@link collectRectAreaEmitterTrisFromCore}; do NOT add this output to the
 * ReSTIR `extraEmitters` stream — that would double-count the triangles that
 * the merged world-space geometry stream already carries.
 */
export function collectMeshAreaEmitterTrisFromCore(
  scene: Scene,
  options: CollectMeshAreaEmitterTrisOptions = {},
): ExtraEmitterTri[] {
  const meshAreaEmitters = scene.emitters.filter((e) => e.kind === 'mesh-area');
  if (meshAreaEmitters.length === 0) return [];

  const primById = new Map<string, Scene['primitives'][number]>();
  for (const p of scene.primitives) {
    primById.set(String(p.id), p);
  }
  const sourceTriIndexFor = buildMeshAreaTlasSourceTriResolver(scene, options.tlasPrimitiveBindings);

  const out: ExtraEmitterTri[] = [];
  for (const e of meshAreaEmitters) {
    if (e.kind !== 'mesh-area') continue;
    const prim = primById.get(String(e.meshId));
    if (prim == null) {
      warnCollectMeshAreaEmitter(options, {
        code: 'walkaround-hybrid.mesh-area-emitter-missing-mesh',
        backend: 'walkaround-hybrid',
        phase: options.warningPhase ?? 'lifecycle',
        method: options.warningMethod ?? 'syncDdgiFromCoreScene',
        message:
          `[vitrum/walkaround-hybrid] mesh-area emitter "${String(e.id)}" ` +
          `references meshId="${String(e.meshId)}" which matches no scene primitive; ` +
          `the emitter is skipped for DDGI probe lighting.`,
        details: {
          emitterId: String(e.id),
          meshId: String(e.meshId),
          source: 'ddgi-probe-emitter-tris',
          fallback: 'emitter skipped',
        },
      });
      continue;
    }
    if (prim.kind !== 'mesh' && prim.kind !== 'skinned-mesh') continue;

    const positions = prim.positions;
    const indices = prim.indices;
    const transform: Mat4 | undefined = (prim as { transform?: Mat4 }).transform;
    const Le = emitterLe(e.color, e.intensity);
    const mappedRadianceMaterial: MaterialSpec | undefined = prim.material.emissiveMap != null
      ? { ...prim.material, emissive: Le, emissiveIntensity: 1 }
      : undefined;

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
      const uv0A = uvAt(prim.uvs, i0);
      const uv0B = uvAt(prim.uvs, i1);
      const uv0C = uvAt(prim.uvs, i2);
      const uv1A = uvAt(prim.uv1, i0);
      const uv1B = uvAt(prim.uv1, i1);
      const uv1C = uvAt(prim.uv1, i2);

      const abx = vB[0] - vA[0], aby = vB[1] - vA[1], abz = vB[2] - vA[2];
      const acx = vC[0] - vA[0], acy = vC[1] - vA[1], acz = vC[2] - vA[2];
      const nx = aby * acz - abz * acy;
      const ny = abz * acx - abx * acz;
      const nz = abx * acy - aby * acx;
      const crossLen = Math.sqrt(nx * nx + ny * ny + nz * nz);
      if (crossLen < 1e-8) continue;
      const invLen = 1 / crossLen;
      const normal: [number, number, number] = [nx * invLen, ny * invLen, nz * invLen];
      const sourceTriIndex = sourceTriIndexFor?.(String(prim.id), ti);
      // A world-space sub-triangle: 3 positions + its two UV sets, plus optional
      // provenance/override metadata. Replaces the former 13-positional-arg
      // pushTri closure (D6-6).
      const pushTri = (sub: SubTriangle, meta: SubTriangleMeta = {}): void => {
        const [triA, triB, triC] = sub.pos;
        const [tuv0A, tuv0B, tuv0C] = sub.uv0;
        const [tuv1A, tuv1B, tuv1C] = sub.uv1;
        const { sourceSubdivLevel, sourceSubdivOrdinal, radianceOverride, forceScalarLe = false } = meta;
        const sx = triB[0] - triA[0], sy = triB[1] - triA[1], sz = triB[2] - triA[2];
        const tx = triC[0] - triA[0], ty = triC[1] - triA[1], tz = triC[2] - triA[2];
        const triArea = 0.5 * Math.sqrt(
          (sy * tz - sz * ty) ** 2 +
          (sz * tx - sx * tz) ** 2 +
          (sx * ty - sy * tx) ** 2,
        );
        if (triArea < 1e-12) return;
        const triLe = radianceOverride ?? (mappedRadianceMaterial == null
          ? Le
          : estimateMaterialSpecEmissiveLeOverTriangle(
              mappedRadianceMaterial,
              tuv0A,
              tuv0B,
              tuv0C,
              tuv1A,
              tuv1B,
              tuv1C,
            ));
        if (triLe == null) return;
        out.push({
          vA: triA,
          vB: triB,
          vC: triC,
          normal,
          area: triArea,
          Le: [triLe[0], triLe[1], triLe[2]],
          ...(e.castShadow !== undefined ? { castShadow: e.castShadow } : {}),
          ...(sourceTriIndex != null && !forceScalarLe ? { sourceTriIndex } : {}),
          ...(sourceSubdivLevel != null && !forceScalarLe ? { sourceSubdivLevel } : {}),
          ...(sourceSubdivOrdinal != null && !forceScalarLe ? { sourceSubdivOrdinal } : {}),
        });
      };

      // Interpolate the parent triangle's positions + both UV sets at the given
      // barycentric weights into a SubTriangle.
      const subTriangleAt = (
        wa: readonly [number, number, number],
        wb: readonly [number, number, number],
        wc: readonly [number, number, number],
      ): SubTriangle => ({
        pos: [baryVec3(vA, vB, vC, wa), baryVec3(vA, vB, vC, wb), baryVec3(vA, vB, vC, wc)],
        uv0: [baryUv2(uv0A, uv0B, uv0C, wa), baryUv2(uv0A, uv0B, uv0C, wb), baryUv2(uv0A, uv0B, uv0C, wc)],
        uv1: [baryUv2(uv1A, uv1B, uv1C, wa), baryUv2(uv1A, uv1B, uv1C, wb), baryUv2(uv1A, uv1B, uv1C, wc)],
      });

      const exactTexelHandled = mappedRadianceMaterial == null
        ? false
        : forEachEmissiveMapTexelSubTriangle(
            mappedRadianceMaterial,
            uv0A,
            uv0B,
            uv0C,
            uv1A,
            uv1B,
            uv1C,
            (wa, wb, wc, texelLe) => {
              pushTri(subTriangleAt(wa, wb, wc), { radianceOverride: texelLe, forceScalarLe: true });
            },
          );
      if (exactTexelHandled) continue;

      const subdiv = mappedRadianceMaterial == null
        ? 1
        : emissiveMapTriangleSubdivisionLevel(mappedRadianceMaterial);
      if (subdiv <= 1) {
        pushTri({ pos: [vA, vB, vC], uv0: [uv0A, uv0B, uv0C], uv1: [uv1A, uv1B, uv1C] });
      } else {
        let ordinal = 0;
        forEachBarycentricSubTriangle(subdiv, (wa, wb, wc) => {
          pushTri(subTriangleAt(wa, wb, wc), { sourceSubdivLevel: subdiv, sourceSubdivOrdinal: ordinal });
          ordinal += 1;
        });
      }
    }
  }
  return out;
}

function determinantSignOfLinear(m: ArrayLike<number> | undefined): number {
  const a = m?.[0] ?? 1;
  const b = m?.[4] ?? 0;
  const c = m?.[8] ?? 0;
  const d = m?.[1] ?? 0;
  const e = m?.[5] ?? 1;
  const f = m?.[9] ?? 0;
  const g = m?.[2] ?? 0;
  const h = m?.[6] ?? 0;
  const i = m?.[10] ?? 1;
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  return det < 0 ? -1 : 1;
}

function buildMeshAreaTlasSourceTriResolver(
  scene: Scene,
  bindings: readonly PrimitiveTlasBinding[] | undefined,
): ((primitiveId: string, localTri: number) => number) | null {
  if (bindings == null || bindings.length === 0) return null;
  const bindingByPrimitiveId = new Map<string, PrimitiveTlasBinding>();
  for (const binding of bindings) {
    if (!bindingByPrimitiveId.has(binding.primitiveId)) {
      bindingByPrimitiveId.set(binding.primitiveId, binding);
    }
  }

  const windingSignByPrimitiveId = new Map<string, number>();
  for (const p of scene.primitives) {
    if (p.kind !== 'mesh' && p.kind !== 'skinned-mesh') continue;
    windingSignByPrimitiveId.set(String(p.id), determinantSignOfLinear(p.transform));
  }

  return (primitiveId: string, localTri: number): number => {
    const binding = bindingByPrimitiveId.get(primitiveId);
    if (binding == null || localTri < 0 || localTri >= binding.triCount) return -1;
    const sourceTri = binding.triStart + localTri;
    return (windingSignByPrimitiveId.get(primitiveId) ?? 1) < 0
      ? -(sourceTri + 2)
      : sourceTri;
  };
}

/**
 * H41 — Analytic point/spot emitter struct layout for the shade NEE buffer.
 *
 * Stride: 4 × vec4f = 64 bytes = 16 floats per entry.
 *   [0..2]  position.xyz + pad(0)
 *   [4..6]  color.rgb (linear, pre-multiplied by intensity) + pad(0)
 *   [8..10] direction.xyz (forward beam/travel axis; (0,0,0) for omnidirectional point)
 *          + cosInner (cosine of inner cone half-angle; 1.0 for point = no cone)
 *   [12]   cosOuter (cosine of outer cone half-angle; 0.0 for point = no cone)
 *   [13]   castShadowDisabled (1.0 when emitter.castShadow === false)
 *   [14]   distance (0.0 = no cutoff)
 *   [15]   decay (0.0 = no falloff, 2.0 = physical inverse-square)
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
      const dist = typeof e.distance === 'number' && e.distance > 0 ? e.distance : 0;
      const decay = typeof e.decay === 'number' ? e.decay : 2;
      data[out * S + 0]  = px;  data[out * S + 1]  = py;  data[out * S + 2]  = pz;  data[out * S + 3]  = 0;
      data[out * S + 4]  = r * i; data[out * S + 5] = g * i; data[out * S + 6] = b * i; data[out * S + 7] = 0;
      data[out * S + 8]  = 0;  data[out * S + 9]  = 0;  data[out * S + 10] = 0;  data[out * S + 11] = 1;
      data[out * S + 12] = 0;  data[out * S + 13] = e.castShadow === false ? 1 : 0;  data[out * S + 14] = dist;  data[out * S + 15] = decay;
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
      const dist = typeof e.distance === 'number' && e.distance > 0 ? e.distance : 0;
      const decay = typeof e.decay === 'number' ? e.decay : 2;
      data[out * S + 0]  = px;  data[out * S + 1]  = py;  data[out * S + 2]  = pz;  data[out * S + 3]  = 0;
      data[out * S + 4]  = r * i; data[out * S + 5] = g * i; data[out * S + 6] = b * i; data[out * S + 7] = 0;
      data[out * S + 8]  = nx;  data[out * S + 9]  = ny;  data[out * S + 10] = nz;  data[out * S + 11] = cosInner;
      data[out * S + 12] = cosOuter; data[out * S + 13] = e.castShadow === false ? 1 : 0; data[out * S + 14] = dist; data[out * S + 15] = decay;
      out++;
    }
  }
  return { data, count: out };
}
