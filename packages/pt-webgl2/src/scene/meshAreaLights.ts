// meshAreaLights.ts — B4: pack `mesh-area` emitters into a triangle-light buffer for
// next-event estimation (NEE).
//
// Background (see foldEmissiveEmitters.ts for the prior state):
//   pt-webgl2's fork integrator lights area sources by HITTING the emissive surface
//   and accumulating `surf.emission` (a pure BSDF-sampling estimator — unbiased but
//   high variance, since a path only finds a small light by luck). `mesh-area`
//   emitters were therefore left OUT of the analytic lights texture (no NEE).
//
//   B4 adds an explicit-connection (NEE) strategy: each explicit `mesh-area`
//   triangle, plus each ordinary mesh whose material has nonzero emissive energy,
//   becomes one or more triangle lights the integrator can sample directly. CPU-
//   readable emissive maps use bounded barycentric micro-triangles so UV-varying
//   emission is spatially localized instead of collapsed to one source-triangle
//   radiance. To keep the result unbiased we MIS-combine the two strategies —
//   the forward emissive hit
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
//   s4 = (0, 0, 0, 0)
//   s5 = (0, castShadowDisabled, 0, 0) — s5.g mirrors analytic light slots

import type { MaterialSpec, Scene, SceneNodeId, TextureRef, Vec3 } from '@vitrum/core';
import {
  emissiveMapTriangleSubdivisionLevel,
  estimateMaterialSpecEmissiveLeOverTriangle,
  forEachBarycentricSubTriangle,
  forEachEmissiveMapTexelSubTriangle,
  mergeUv1FromCore,
  type BarycentricWeights,
  type WorldSpaceMergeResult,
} from '@vitrum/shared-bvh';

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
  readonly warnings: readonly string[];
}

interface RawTexturePayload {
  readonly width: number;
  readonly height: number;
  readonly data: ArrayBufferView;
}

const IMPLICIT_EMITTER_LUMINANCE_THRESHOLD = 1e-6;

function v3(p: Float32Array, vi: number, stride: number): Vec3 {
  const b = vi * stride;
  return [p[b] ?? 0, p[b + 1] ?? 0, p[b + 2] ?? 0];
}

function uvAt(uvs: Float32Array | undefined, vi: number): [number, number] {
  if (uvs == null) return [0, 0];
  const b = vi * 2;
  return [uvs[b] ?? 0, uvs[b + 1] ?? 0];
}

function baryVec3(a: Vec3, b: Vec3, c: Vec3, w: BarycentricWeights): Vec3 {
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

function sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function length(a: Vec3): number {
  return Math.hypot(a[0], a[1], a[2]);
}

function luminanceRgb(rgb: Vec3): number {
  return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
}

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, Number.isFinite(value) ? value : 0));
}

function srgbToLinear(value: number): number {
  const c = clamp01(value);
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

function numericChannelMax(data: ArrayBufferView): number | null {
  if (data instanceof Float32Array || data instanceof Float64Array) return 1;
  if (data instanceof Uint8Array || data instanceof Uint8ClampedArray) return 255;
  if (data instanceof Uint16Array) return 65535;
  if (data instanceof Uint32Array) return 4294967295;
  if (data instanceof Int8Array) return 127;
  if (data instanceof Int16Array) return 32767;
  if (data instanceof Int32Array) return 2147483647;
  return null;
}

function rawPayloadOfTexture(ref: TextureRef | undefined): RawTexturePayload | null {
  const source = ref?.handle;
  if (source == null || typeof source !== 'object') return null;
  const img = ('image' in source && (source as { image?: unknown }).image != null
    ? (source as { image?: unknown }).image
    : source) as Record<string, unknown>;
  if (img == null || typeof img !== 'object') return null;
  const width = typeof img.width === 'number' ? img.width : 0;
  const height = typeof img.height === 'number' ? img.height : 0;
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    return null;
  }
  return ArrayBuffer.isView(img.data)
    ? { width: Math.floor(width), height: Math.floor(height), data: img.data }
    : null;
}

function averageSrgbTextureRgb(ref: TextureRef | undefined): Vec3 | null {
  const payload = rawPayloadOfTexture(ref);
  if (payload == null) return null;
  const pixelCount = payload.width * payload.height;
  if (pixelCount <= 0) return null;
  const data = payload.data as unknown as ArrayLike<number>;
  const channelCount = data.length / pixelCount;
  if (![1, 2, 3, 4].includes(channelCount) || !Number.isInteger(channelCount)) {
    return null;
  }
  const maxValue = numericChannelMax(payload.data);
  if (maxValue == null) return null;
  let r = 0;
  let g = 0;
  let b = 0;
  for (let i = 0; i < pixelCount; i += 1) {
    const src = i * channelCount;
    const rawR = Number(data[src] ?? 0);
    const rawG = channelCount >= 2 ? Number(data[src + 1] ?? 0) : rawR;
    const rawB = channelCount >= 3 ? Number(data[src + 2] ?? 0) : rawR;
    r += srgbToLinear(rawR / maxValue);
    g += srgbToLinear(rawG / maxValue);
    b += srgbToLinear(rawB / maxValue);
  }
  const inv = 1 / pixelCount;
  return [r * inv, g * inv, b * inv];
}

function emissiveRadianceForMaterial(
  material: MaterialSpec,
  primitiveId: string,
  warnings?: string[],
): Vec3 {
  const emissive = material.emissive ?? [0, 0, 0];
  const intensity = material.emissiveIntensity ?? 1;
  const mapAverage = averageSrgbTextureRgb(material.emissiveMap);
  if (material.emissiveMap != null && mapAverage == null && warnings != null) {
    warnings.push(
      `@vitrum/pt-webgl2: primitive "${primitiveId}" has an emissiveMap without CPU-readable texels; ` +
        'implicit mesh-area NEE uses scalar emissive radiance only.',
    );
  }
  const map = mapAverage ?? [1, 1, 1];
  return [
    emissive[0] * intensity * map[0],
    emissive[1] * intensity * map[1],
    emissive[2] * intensity * map[2],
  ];
}

function isMeshLikePrimitive(primitive: Scene['primitives'][number]): primitive is Extract<
  Scene['primitives'][number],
  { kind: 'mesh' | 'instanced-mesh' | 'skinned-mesh' }
> {
  return primitive.kind === 'mesh' || primitive.kind === 'instanced-mesh' || primitive.kind === 'skinned-mesh';
}

function collectMeshAreaSources(
  scene: Scene,
  warnings: string[],
): Map<SceneNodeId, { radiance: Vec3; castShadowDisabled: number; implicitMaterial?: MaterialSpec }> {
  const emitterByMesh = new Map<SceneNodeId, { radiance: Vec3; castShadowDisabled: number; implicitMaterial?: MaterialSpec }>();
  for (const e of scene.emitters) {
    if (e.kind !== 'mesh-area') continue;
    const primitive = scene.primitives.find((p) => String(p.id) === String(e.meshId));
    const material = primitive != null && isMeshLikePrimitive(primitive) ? primitive.material : undefined;
    const radiance: Vec3 = [e.color[0] * e.intensity, e.color[1] * e.intensity, e.color[2] * e.intensity];
    emitterByMesh.set(e.meshId, {
      radiance,
      castShadowDisabled: e.castShadow === false ? 1 : 0,
      ...(material?.emissiveMap != null
        ? { implicitMaterial: { ...material, emissive: radiance, emissiveIntensity: 1 } }
        : {}),
    });
  }

  for (const primitive of scene.primitives) {
    if (!isMeshLikePrimitive(primitive)) continue;
    if (emitterByMesh.has(primitive.id)) continue;
    const radiance = emissiveRadianceForMaterial(primitive.material, String(primitive.id), warnings);
    if (luminanceRgb(radiance) < IMPLICIT_EMITTER_LUMINANCE_THRESHOLD) continue;
    emitterByMesh.set(primitive.id, {
      radiance,
      castShadowDisabled: primitive.castShadow === false ? 1 : 0,
      implicitMaterial: primitive.material,
    });
  }

  return emitterByMesh;
}

export function hasMeshAreaLightForPrimitive(scene: Scene, primitiveId: string): boolean {
  for (const e of scene.emitters) {
    if (e.kind === 'mesh-area' && String(e.meshId) === primitiveId) return true;
  }
  const primitive = scene.primitives.find((p) => String(p.id) === primitiveId);
  if (primitive == null || !isMeshLikePrimitive(primitive)) return false;
  const radiance = emissiveRadianceForMaterial(primitive.material, primitiveId);
  return luminanceRgb(radiance) >= IMPLICIT_EMITTER_LUMINANCE_THRESHOLD;
}

/**
 * Pack explicit `mesh-area` emitters and implicit emissive-material meshes into a
 * triangle-light grid for NEE.
 *
 * @param scene   the capability-filtered scene (emitters + material emission drive this)
 * @param merged  the world-space merged tri stream (geometry source via meshVertexRanges)
 */
export function packMeshAreaLights(
  scene: Scene,
  merged: WorldSpaceMergeResult,
): MeshAreaLightsData {
  const warnings: string[] = [];
  const emitterByMesh = collectMeshAreaSources(scene, warnings);
  if (emitterByMesh.size === 0) {
    return { data: null, dim: 1, triLightCount: 0, totalEmissiveArea: 0, warnings };
  }

  const stride = merged.positionStrideFloats;
  const idx = merged.mergedIndices;
  const pos = merged.positions;
  const uv0 = merged.uvs;
  const uv1 = mergeUv1FromCore(scene, merged.meshVertexRanges, merged.vertexCount);

  // Collect (v0,v1,v2,radiance,area,shadow flag) for every triangle of every emissive mesh.
  const tris: {
    v0: Vec3;
    v1: Vec3;
    v2: Vec3;
    rad: Vec3;
    area: number;
    castShadowDisabled: number;
  }[] = [];
  let totalEmissiveArea = 0;
  for (const range of merged.meshVertexRanges) {
    const emitter = emitterByMesh.get(range.name);
    if (emitter == null) continue;
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
      const uv0A = uvAt(uv0, i0);
      const uv0B = uvAt(uv0, i1);
      const uv0C = uvAt(uv0, i2);
      const uv1A = uvAt(uv1, i0);
      const uv1B = uvAt(uv1, i1);
      const uv1C = uvAt(uv1, i2);

      const pushLight = (
        tv0: Vec3,
        tv1: Vec3,
        tv2: Vec3,
        tuv0A: readonly [number, number],
        tuv0B: readonly [number, number],
        tuv0C: readonly [number, number],
        tuv1A: readonly [number, number],
        tuv1B: readonly [number, number],
        tuv1C: readonly [number, number],
        radianceOverride?: readonly [number, number, number],
      ): void => {
        const triArea = 0.5 * length(cross(sub(tv1, tv0), sub(tv2, tv0)));
        if (triArea <= 0) return;
        const rad = radianceOverride ?? (emitter.implicitMaterial == null
          ? emitter.radiance
          : estimateMaterialSpecEmissiveLeOverTriangle(
              emitter.implicitMaterial,
              tuv0A,
              tuv0B,
              tuv0C,
              tuv1A,
              tuv1B,
              tuv1C,
            ));
        if (rad == null || luminanceRgb(rad) < IMPLICIT_EMITTER_LUMINANCE_THRESHOLD) {
          return;
        }
        tris.push({
          v0: tv0,
          v1: tv1,
          v2: tv2,
          rad,
          area: triArea,
          castShadowDisabled: emitter.castShadowDisabled,
        });
        totalEmissiveArea += triArea;
      };

      const exactTexelHandled = emitter.implicitMaterial == null
        ? false
        : forEachEmissiveMapTexelSubTriangle(
            emitter.implicitMaterial,
            uv0A,
            uv0B,
            uv0C,
            uv1A,
            uv1B,
            uv1C,
            (wa, wb, wc, texelRadiance) => {
              pushLight(
                baryVec3(v0, v1, v2, wa),
                baryVec3(v0, v1, v2, wb),
                baryVec3(v0, v1, v2, wc),
                baryUv2(uv0A, uv0B, uv0C, wa),
                baryUv2(uv0A, uv0B, uv0C, wb),
                baryUv2(uv0A, uv0B, uv0C, wc),
                baryUv2(uv1A, uv1B, uv1C, wa),
                baryUv2(uv1A, uv1B, uv1C, wb),
                baryUv2(uv1A, uv1B, uv1C, wc),
                texelRadiance,
              );
            },
          );
      if (exactTexelHandled) continue;

      const subdiv = emitter.implicitMaterial == null
        ? 1
        : emissiveMapTriangleSubdivisionLevel(emitter.implicitMaterial);
      if (subdiv <= 1) {
        pushLight(v0, v1, v2, uv0A, uv0B, uv0C, uv1A, uv1B, uv1C);
      } else {
        forEachBarycentricSubTriangle(subdiv, (wa, wb, wc) => {
          pushLight(
            baryVec3(v0, v1, v2, wa),
            baryVec3(v0, v1, v2, wb),
            baryVec3(v0, v1, v2, wc),
            baryUv2(uv0A, uv0B, uv0C, wa),
            baryUv2(uv0A, uv0B, uv0C, wb),
            baryUv2(uv0A, uv0B, uv0C, wc),
            baryUv2(uv1A, uv1B, uv1C, wa),
            baryUv2(uv1A, uv1B, uv1C, wb),
            baryUv2(uv1A, uv1B, uv1C, wc),
          );
        });
      }
    }
  }

  if (tris.length === 0) {
    return { data: null, dim: 1, triLightCount: 0, totalEmissiveArea: 0, warnings };
  }

  const pixelCount = tris.length * TRI_LIGHT_PIXELS;
  const dim = Math.ceil(Math.sqrt(pixelCount));
  const data = new Float32Array(dim * dim * 4);
  for (let i = 0; i < tris.length; i += 1) {
    const { v0, v1, v2, rad, area, castShadowDisabled } = tris[i]!;
    const base = i * TRI_LIGHT_PIXELS * 4;
    // s0: v0 / type
    data[base + 0] = v0[0];
    data[base + 1] = v0[1];
    data[base + 2] = v0[2];
    data[base + 3] = TRI_AREA_LIGHT_TYPE;
    // s1: radiance / 0
    data[base + 4] = rad[0];
    data[base + 5] = rad[1];
    data[base + 6] = rad[2];
    data[base + 7] = 0;
    // s2: v1 / 0
    data[base + 8] = v1[0];
    data[base + 9] = v1[1];
    data[base + 10] = v1[2];
    data[base + 11] = 0;
    // s3: v2 / triArea
    data[base + 12] = v2[0];
    data[base + 13] = v2[1];
    data[base + 14] = v2[2];
    data[base + 15] = area;
    // s5.g: castShadowDisabled (s4 and other s5 channels remain zero).
    data[base + 21] = castShadowDisabled;
  }

  return { data, dim, triLightCount: tris.length, totalEmissiveArea, warnings };
}
