// meshAreaLights.ts — B4: pack `mesh-area` emitters into a triangle-light buffer for
// next-event estimation (NEE).
//
// Background (see foldEmissiveEmitters.ts for the prior state):
//   pt-webgl2's fork integrator lights area sources by HITTING the emissive surface
//   and accumulating `surf.emission` (a pure BSDF-sampling estimator — unbiased but
//   high variance, since a path only finds a small light by luck). `mesh-area`
//   emitters were therefore left OUT of the analytic lights texture (no NEE).
//
//   B4 adds an explicit-connection (NEE) strategy: each emissive mesh-area triangle
//   becomes a triangle light the integrator can sample directly. To keep the result
//   unbiased we MIS-combine the two strategies — the forward emissive hit
//   (BSDF sampling) and the NEE triangle sample — with the balance/power heuristic.
//
// Double-count guard / MIS algebra (the crux):
//   NEE selects an emissive triangle with probability proportional to its AREA, i.e.
//   p_tri = area_tri / totalEmissiveArea, and samples a point uniformly on it
//   (area density 1/area_tri). The resulting SOLID-ANGLE pdf at a surface receiving
//   the connection is
//        p_NEE(ω) = (dist² / (area_tri·|cosθ_light|)) · p_tri
//                 = dist² / (totalEmissiveArea·|cosθ_light|)          [area_tri cancels]
//   which is INDEPENDENT of which triangle was chosen — it needs only the GLOBAL
//   `totalEmissiveArea`. That is what lets the forward emissive hit (which does NOT
//   know which light-list index it struck) compute the very same p_NEE and form the
//   MIS weight `misHeuristic(bsdfPdf, p_NEE)` without a triangle→light-index map.
//   The forward hit then adds `surf.emission · w_bsdf` and NEE adds its sample with
//   `w_nee` — together exactly one estimate of the area light (no double-count).
//
// Geometry source: the world-space merged tri stream (`WorldSpaceMergeResult`) — each
// mesh-area emitter's `meshId` maps to a `meshVertexRanges` entry (triStart/triCount),
// whose triangles are read from `mergedIndices` + `positions` (merge order).
//
// Layout: reuses the 6-texel light slot (LIGHT_PIXELS=6) with a new TRI_AREA type:
//   s0 = (v0.xyz, type=TRI_AREA=5)
//   s1 = (radiance.rgb = color*intensity, 0)
//   s2 = (v1.xyz, 0)
//   s3 = (v2.xyz, triArea)          — geometric area |(v1-v0)×(v2-v0)|/2
// (s4/s5 unused — TRI_AREA reads only s0..s3, like rect/disc.)

import type { Scene, SceneNodeId, Vec3 } from '@vitrum/core';
import type { WorldSpaceMergeResult } from '@vitrum/shared-bvh';

/** Triangle-light type id — must match the GLSL `#define TRI_AREA_LIGHT_TYPE`. */
export const TRI_AREA_LIGHT_TYPE = 5;
/** Texels per triangle light (same 6-slot stride as the analytic lights texture). */
export const TRI_LIGHT_PIXELS = 6;

export interface MeshAreaLightsData {
  /** RGBA32F square grid (dim×dim) packing `triLightCount` triangle lights, or null
   *  when the scene has no emissive mesh-area triangles. */
  readonly data: Float32Array | null;
  readonly dim: number;
  readonly triLightCount: number;
  /** Σ triangle areas — the global the forward-hit MIS weight needs (see header). */
  readonly totalEmissiveArea: number;
}

function v3(p: Float32Array, vi: number, stride: number): Vec3 {
  const b = vi * stride;
  return [p[b] ?? 0, p[b + 1] ?? 0, p[b + 2] ?? 0];
}

function sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function length(a: Vec3): number {
  return Math.hypot(a[0], a[1], a[2]);
}

/**
 * Pack the emissive `mesh-area` triangles into a triangle-light grid for NEE.
 *
 * @param scene   the capability-filtered scene (its `mesh-area` emitters drive this)
 * @param merged  the world-space merged tri stream (geometry source via meshVertexRanges)
 */
export function packMeshAreaLights(scene: Scene, merged: WorldSpaceMergeResult): MeshAreaLightsData {
  const radianceByMesh = new Map<SceneNodeId, Vec3>();
  for (const e of scene.emitters) {
    if (e.kind === 'mesh-area') {
      radianceByMesh.set(e.meshId, [e.color[0] * e.intensity, e.color[1] * e.intensity, e.color[2] * e.intensity]);
    }
  }
  if (radianceByMesh.size === 0) {
    return { data: null, dim: 1, triLightCount: 0, totalEmissiveArea: 0 };
  }

  const stride = merged.positionStrideFloats;
  const idx = merged.mergedIndices;
  const pos = merged.positions;

  // Collect (v0,v1,v2,radiance,area) for every triangle of every emissive mesh.
  const tris: { v0: Vec3; v1: Vec3; v2: Vec3; rad: Vec3; area: number }[] = [];
  let totalEmissiveArea = 0;
  for (const range of merged.meshVertexRanges) {
    const rad = radianceByMesh.get(range.name);
    if (rad == null) continue;
    for (let t = 0; t < range.triCount; t += 1) {
      const tri = range.triStart + t;
      const i0 = idx[tri * 3] ?? 0;
      const i1 = idx[tri * 3 + 1] ?? 0;
      const i2 = idx[tri * 3 + 2] ?? 0;
      const v0 = v3(pos, i0, stride);
      const v1 = v3(pos, i1, stride);
      const v2 = v3(pos, i2, stride);
      const area = 0.5 * length(cross(sub(v1, v0), sub(v2, v0)));
      if (area <= 0) continue; // degenerate triangle — contributes no light
      tris.push({ v0, v1, v2, rad, area });
      totalEmissiveArea += area;
    }
  }

  if (tris.length === 0) {
    return { data: null, dim: 1, triLightCount: 0, totalEmissiveArea: 0 };
  }

  const pixelCount = tris.length * TRI_LIGHT_PIXELS;
  const dim = Math.ceil(Math.sqrt(pixelCount));
  const data = new Float32Array(dim * dim * 4);
  for (let i = 0; i < tris.length; i += 1) {
    const { v0, v1, v2, rad, area } = tris[i]!;
    const base = i * TRI_LIGHT_PIXELS * 4;
    // s0: v0 / type
    data[base + 0] = v0[0]; data[base + 1] = v0[1]; data[base + 2] = v0[2]; data[base + 3] = TRI_AREA_LIGHT_TYPE;
    // s1: radiance / 0
    data[base + 4] = rad[0]; data[base + 5] = rad[1]; data[base + 6] = rad[2]; data[base + 7] = 0;
    // s2: v1 / 0
    data[base + 8] = v1[0]; data[base + 9] = v1[1]; data[base + 10] = v1[2]; data[base + 11] = 0;
    // s3: v2 / triArea
    data[base + 12] = v2[0]; data[base + 13] = v2[1]; data[base + 14] = v2[2]; data[base + 15] = area;
  }

  return { data, dim, triLightCount: tris.length, totalEmissiveArea };
}
