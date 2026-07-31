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
//   readable nearest/no-mip emissive maps are partitioned exactly at texel-cell
//   boundaries. Their decoded texels are rounded through the same RGBA16F
//   level-0 storage as forward hits, and emissive scalar operands/products follow
//   the shader's RGBA32F arithmetic order. UV-varying emission is therefore never
//   collapsed to an approximate or differently-rounded source-triangle radiance.
//   To keep the result unbiased we MIS-combine the two strategies —
//   the forward emissive hit
//   (BSDF sampling) and the NEE triangle sample — with the balance/power heuristic.
//
// Double-count guard / MIS algebra (the crux):
//   NEE selects an emissive triangle or bounded texel cell with probability
//   proportional to its emitted power:
//        power_tri = luminance(radiance_tri) · area_tri
//        p_tri = power_tri / totalEmissivePower
//   and samples a point uniformly on it (area density 1/area_tri). The resulting
//   SOLID-ANGLE pdf at a surface receiving the connection is
//        p_NEE(ω) = (dist² / (area_tri·|cosθ_light|)) · p_tri
//                 = luminance(radiance_tri) · dist²
//                   / (totalEmissivePower·|cosθ_light|)          [area_tri cancels]
//   which is reconstructible from the hit surface's emitted radiance and the GLOBAL
//   `totalEmissivePower`. That is what lets the forward emissive hit (which does NOT
//   know which light-list index it struck) compute the very same p_NEE and form the
//   MIS weight `misHeuristic(bsdfPdf, p_NEE)` without a triangle→index map.
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
//   s4 = (selectionPower, 0, 0, 0)  — luminance(radiance) * triArea
//   s5 = (faceIdLo16, castShadowDisabled, twoSided, faceIdHi16)
//        — exact BVH/index-texture target identity plus shadow/material-sidedness.
//          The two 16-bit words are individually exact in RGBA32F; combining the
//          id into one float would lose integer identity above 2^24.

import type { MaterialSpec, Scene, SceneNodeId, Vec3 } from '@vitrum/core';
import {
  forEachEmissiveMapTexelSubTriangle,
  isTextureRefCpuReadable,
  materialSpecEmissiveLe,
  type EmissiveMapTexelRadianceResolver,
  type BarycentricWeights,
  type WorldSpaceMergeResult,
} from '@vitrum/shared-bvh';
import { luminance } from '@vitrum/shared-samplers';
import { classifyTriangleAreaF32 } from './areaEmitterGeometry.js';
import { quantizeFiniteFloat16 } from './halfFloat.js';
import { buildUvAttributeLayout } from './uvAttributeLayout.js';
import {
  materialEmissionExcludedFromMeshNee,
  materialWithExplicitMeshEmitterAuthority,
} from './meshEmitterPolicy.js';

/** Triangle-light type id — must match the GLSL `#define TRI_AREA_LIGHT_TYPE`. */
export const TRI_AREA_LIGHT_TYPE = 5;
/** Texels per triangle light (same 6-slot stride as the analytic lights texture). */
export const TRI_LIGHT_PIXELS = 6;
const UINT32_MAX = 0xffff_ffff;
const UINT16_BASE = 0x1_0000;

export interface MeshAreaLightsData {
  /** RGBA32F square grid (dim×dim) packing `triLightCount` triangle lights, or null
   *  when the scene has no emissive mesh-area triangles. */
  readonly data: Float32Array | null;
  readonly dim: number;
  readonly triLightCount: number;
  /** Σ triangle areas — the global the forward-hit MIS weight needs (see header). */
  readonly totalEmissiveArea: number;
  /** Σ luminance(radiance)·area — energy-weighted selection mass for mesh NEE. */
  readonly totalEmissivePower: number;
}

function addMeshSelectionPowerF32(
  sum: number,
  power: number,
  context: string,
): number {
  const next = Math.fround(sum + power);
  if (!Number.isFinite(next)) {
    throw new RangeError(
      '@vitrum/pt-webgl2: mesh-emitter selection-power sum overflows float32.',
    );
  }
  if (power > 0 && next === sum) {
    throw new RangeError(
      `${context} loses proposal support in the cumulative float32 distribution.`,
    );
  }
  return next;
}

/**
 * Invert the producer's BVH-face -> merged-face permutation. Triangle lights are
 * enumerated in deterministic merged order, while WebGL2 intersection reports
 * the BVH/index-texture slot in `SurfaceHit.faceIndices.w`; visibility therefore
 * has to carry the inverse-mapped id.
 */
function buildBvhFaceByMergedFace(merged: WorldSpaceMergeResult): Uint32Array {
  const triangleCount = merged.triangleCount;
  if (
    !Number.isSafeInteger(triangleCount) ||
    triangleCount < 0 ||
    triangleCount > UINT32_MAX
  ) {
    throw new RangeError(
      '@vitrum/pt-webgl2: mesh-emitter triangle count is outside uint32 range.',
    );
  }
  if (merged.bvhTriToMergedTri.length !== triangleCount) {
    throw new RangeError(
      '@vitrum/pt-webgl2: mesh-emitter BVH-to-merged face mapping is incomplete.',
    );
  }

  const bvhFaceByMergedFace = new Uint32Array(triangleCount);
  const seenMergedFaces = new Uint8Array(triangleCount);
  for (let bvhFaceIndex = 0; bvhFaceIndex < triangleCount; bvhFaceIndex += 1) {
    if (bvhFaceIndex > UINT32_MAX) {
      throw new RangeError(
        '@vitrum/pt-webgl2: mesh-emitter BVH face id is outside uint32 range.',
      );
    }
    const mergedFaceIndex = merged.bvhTriToMergedTri[bvhFaceIndex]!;
    if (mergedFaceIndex >= triangleCount) {
      throw new RangeError(
        '@vitrum/pt-webgl2: mesh-emitter BVH-to-merged face mapping references an out-of-range face.',
      );
    }
    if (seenMergedFaces[mergedFaceIndex] !== 0) {
      throw new RangeError(
        '@vitrum/pt-webgl2: mesh-emitter BVH-to-merged face mapping is non-bijective.',
      );
    }
    seenMergedFaces[mergedFaceIndex] = 1;
    bvhFaceByMergedFace[mergedFaceIndex] = bvhFaceIndex;
  }
  for (let mergedFaceIndex = 0; mergedFaceIndex < triangleCount; mergedFaceIndex += 1) {
    if (seenMergedFaces[mergedFaceIndex] === 0) {
      throw new RangeError(
        '@vitrum/pt-webgl2: mesh-emitter BVH-to-merged face mapping is incomplete.',
      );
    }
  }
  return bvhFaceByMergedFace;
}

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

// `luminanceRgb` is a thin tuple wrapper over the shared Rec.709
// `luminance(r,g,b)` so the emitted-power field stays byte-for-byte identical.
function luminanceRgb(rgb: Vec3): number {
  return luminance(rgb[0], rgb[1], rgb[2]);
}

function materialF32Operand(value: number, context: string): number {
  if (!Number.isFinite(value)) {
    throw new RangeError(`${context} must be finite for material RGBA32F storage.`);
  }
  const stored = Math.fround(value);
  if (!Number.isFinite(stored)) {
    throw new RangeError(`${context} overflows material RGBA32F storage.`);
  }
  if (value > 0 && stored === 0) {
    throw new RangeError(`${context} underflows material RGBA32F storage.`);
  }
  return stored;
}

function multiplyForwardF32(a: number, b: number, context: string): number {
  const exact = a * b;
  const stored = Math.fround(exact);
  if (!Number.isFinite(exact) || !Number.isFinite(stored)) {
    throw new RangeError(`${context} overflows forward f32 emissive arithmetic.`);
  }
  if (exact !== 0 && stored === 0) {
    throw new RangeError(`${context} underflows forward f32 emissive arithmetic.`);
  }
  return stored;
}

/**
 * Mirror `packMaterialsTexture` + GLSL exactly: operands first enter separate
 * RGBA32F slots, then `emissiveIntensity * emissive.rgb` is evaluated in f32.
 */
function storedMaterialEmissiveScalar(
  emissive: readonly [number, number, number] | undefined,
  emissiveIntensity: number | undefined,
  context: string,
): Vec3 | null {
  if (emissive == null) return null;
  const authoredIntensity = emissiveIntensity ?? 1;
  if (!Number.isFinite(authoredIntensity)) {
    throw new RangeError(`${context} emissiveIntensity must be finite.`);
  }
  if (authoredIntensity < 0) {
    throw new RangeError(`${context} emissiveIntensity must be non-negative.`);
  }
  if (authoredIntensity === 0) return null;
  const intensity = materialF32Operand(
    authoredIntensity,
    `${context} emissiveIntensity`,
  );
  const out: [number, number, number] = [0, 0, 0];
  for (let channel = 0; channel < 3; channel += 1) {
    const authoredComponent = emissive[channel]!;
    if (authoredComponent < 0) {
      throw new RangeError(`${context} emissive[${channel}] must be non-negative.`);
    }
    const component = materialF32Operand(
      authoredComponent,
      `${context} emissive[${channel}]`,
    );
    out[channel] = multiplyForwardF32(
      intensity,
      component,
      `${context} emissive scalar[${channel}]`,
    );
  }
  return out[0] > 0 || out[1] > 0 || out[2] > 0 ? out : null;
}

const resolveWebGl2MappedEmitterRadiance: EmissiveMapTexelRadianceResolver = (
  material,
  decodedLinearRgb,
  texelX,
  texelY,
) => {
  const scalar = storedMaterialEmissiveScalar(
    material.emissive,
    material.emissiveIntensity,
    'mapped emitter material',
  );
  if (scalar == null) return null;
  return [0, 1, 2].map((channel) => {
    if (decodedLinearRgb[channel]! < 0) {
      throw new RangeError(
        `emissiveMap texel (${texelX}, ${texelY}) channel ${channel} ` +
          'must be non-negative outgoing radiance.',
      );
    }
    const mapValue = quantizeFiniteFloat16(
      decodedLinearRgb[channel]!,
      `emissiveMap texel (${texelX}, ${texelY}) channel ${channel}`,
    );
    return multiplyForwardF32(
      scalar[channel]!,
      mapValue,
      `mapped emitter radiance[${channel}]`,
    );
  }) as [number, number, number];
};

function assertRadianceSurvivesF32(rgb: readonly [number, number, number], context: string): void {
  for (let channel = 0; channel < 3; channel += 1) {
    const value = rgb[channel]!;
    const stored = Math.fround(value);
    if (!Number.isFinite(value) || value < 0 || !Number.isFinite(stored)) {
      throw new RangeError(`${context} radiance[${channel}] must be finite, non-negative, and f32-representable.`);
    }
    if (value > 0 && stored === 0) {
      throw new RangeError(`${context} radiance[${channel}] underflows RGBA32F storage.`);
    }
  }
}

function assertPositiveValueSurvivesF32(value: number, context: string): void {
  const stored = Math.fround(value);
  if (!(value > 0) || !Number.isFinite(value) || !Number.isFinite(stored) || stored === 0) {
    throw new RangeError(`${context} must be positive and survive f32 storage.`);
  }
}

function assertEmissiveMapCpuReadable(material: MaterialSpec, primitiveId: string): void {
  if (material.emissiveMap == null || isTextureRefCpuReadable(material.emissiveMap, 'srgb')) {
    return;
  }
  throw new TypeError(
    `@vitrum/pt-webgl2: primitive "${primitiveId}" uses an emissiveMap without ` +
      'complete CPU-readable texels. Emissive-map NEE cannot substitute scalar emission ' +
      'without biasing forward-hit MIS. Supply raw texture pixels or an exact cpuMirror.',
  );
}

function emissiveRadianceForMaterial(
  material: MaterialSpec,
  primitiveId: string,
): Vec3 {
  const scalar = storedMaterialEmissiveScalar(
    material.emissive,
    material.emissiveIntensity,
    `implicit emitter primitive ${primitiveId}`,
  );
  if (scalar == null) return [0, 0, 0];
  assertEmissiveMapCpuReadable(material, primitiveId);
  // Mapped radiance is evaluated per exact texel partition below. This scalar
  // source term is only the potential-emitter marker and unmapped radiance.
  return scalar;
}

/** Fail before GL allocation when an emissive-map NEE source has no exact CPU texels. */
export function assertSceneEmissiveMapsCpuReadable(scene: Scene): void {
  const explicitMeshIds = new Set<string>();
  for (const emitter of scene.emitters) {
    if (emitter.kind !== 'mesh-area') continue;
    explicitMeshIds.add(String(emitter.meshId));
    const radiance = storedMaterialEmissiveScalar(
      emitter.color,
      emitter.intensity,
      `mesh-area emitter ${String(emitter.id)}`,
    );
    if (radiance == null || !(luminanceRgb(radiance) > 0)) continue;
    const primitive = scene.primitives.find((p) => String(p.id) === String(emitter.meshId));
    if (primitive != null && isMeshLikePrimitive(primitive)) {
      const material = materialWithExplicitMeshEmitterAuthority(primitive.material);
      if (!materialEmissionExcludedFromMeshNee(material)) {
        assertEmissiveMapCpuReadable(material, String(primitive.id));
      }
    }
  }
  for (const primitive of scene.primitives) {
    if (!isMeshLikePrimitive(primitive) || explicitMeshIds.has(String(primitive.id))) continue;
    if (materialEmissionExcludedFromMeshNee(primitive.material)) continue;
    if (
      storedMaterialEmissiveScalar(
        primitive.material.emissive,
        primitive.material.emissiveIntensity,
        `implicit emitter primitive ${String(primitive.id)}`,
      ) != null
    ) {
      assertEmissiveMapCpuReadable(primitive.material, String(primitive.id));
    }
  }
}

function isMeshLikePrimitive(primitive: Scene['primitives'][number]): primitive is Extract<
  Scene['primitives'][number],
  { kind: 'mesh' | 'instanced-mesh' | 'skinned-mesh' }
> {
  return primitive.kind === 'mesh' || primitive.kind === 'instanced-mesh' || primitive.kind === 'skinned-mesh';
}

function collectMeshAreaSources(
  scene: Scene,
): Map<SceneNodeId, {
  radiance: Vec3;
  castShadowDisabled: number;
  twoSided: number;
  implicitMaterial?: MaterialSpec;
}> {
  assertSceneEmissiveMapsCpuReadable(scene);
  const emitterByMesh = new Map<SceneNodeId, {
    radiance: Vec3;
    castShadowDisabled: number;
    twoSided: number;
    implicitMaterial?: MaterialSpec;
  }>();
  for (const e of scene.emitters) {
    if (e.kind !== 'mesh-area') continue;
    const primitive = scene.primitives.find((p) => String(p.id) === String(e.meshId));
    const material = primitive != null && isMeshLikePrimitive(primitive)
      ? materialWithExplicitMeshEmitterAuthority(primitive.material)
      : undefined;
    const radiance = storedMaterialEmissiveScalar(
      e.color,
      e.intensity,
      `mesh-area emitter ${String(e.id)}`,
    );
    if (radiance == null || !(luminanceRgb(radiance) > 0)) continue;
    assertRadianceSurvivesF32(radiance, `mesh-area emitter ${String(e.id)}`);
    if (material != null && materialEmissionExcludedFromMeshNee(material)) continue;
    emitterByMesh.set(e.meshId, {
      radiance,
      castShadowDisabled: e.castShadow === false ? 1 : 0,
      twoSided: material?.doubleSided === true ? 1 : 0,
      ...(material?.emissiveMap != null
        ? {
            implicitMaterial: {
              ...material,
              emissive: e.color,
              emissiveIntensity: e.intensity,
            },
          }
        : {}),
    });
  }

  for (const primitive of scene.primitives) {
    if (!isMeshLikePrimitive(primitive)) continue;
    if (emitterByMesh.has(primitive.id)) continue;
    if (materialEmissionExcludedFromMeshNee(primitive.material)) continue;
    const radiance = emissiveRadianceForMaterial(primitive.material, String(primitive.id));
    if (!(luminanceRgb(radiance) > 0)) continue;
    assertRadianceSurvivesF32(radiance, `implicit emitter primitive ${String(primitive.id)}`);
    emitterByMesh.set(primitive.id, {
      radiance,
      castShadowDisabled: primitive.castShadow === false ? 1 : 0,
      twoSided: primitive.material.doubleSided === true ? 1 : 0,
      implicitMaterial: primitive.material,
    });
  }

  return emitterByMesh;
}

export function hasMeshAreaLightForPrimitive(scene: Scene, primitiveId: string): boolean {
  for (const e of scene.emitters) {
    if (e.kind === 'mesh-area' && String(e.meshId) === primitiveId) {
      const primitive = scene.primitives.find((p) => String(p.id) === primitiveId);
      if (
        primitive != null &&
        isMeshLikePrimitive(primitive) &&
        materialEmissionExcludedFromMeshNee(
          materialWithExplicitMeshEmitterAuthority(primitive.material),
        )
      ) {
        return false;
      }
      return (
        storedMaterialEmissiveScalar(
          e.color,
          e.intensity,
          `mesh-area emitter ${String(e.id)}`,
        ) != null
      );
    }
  }
  const primitive = scene.primitives.find((p) => String(p.id) === primitiveId);
  if (primitive == null || !isMeshLikePrimitive(primitive)) return false;
  if (materialEmissionExcludedFromMeshNee(primitive.material)) return false;
  if (
    storedMaterialEmissiveScalar(
      primitive.material.emissive,
      primitive.material.emissiveIntensity,
      `implicit emitter primitive ${primitiveId}`,
    ) == null
  ) {
    return false;
  }
  assertEmissiveMapCpuReadable(primitive.material, primitiveId);
  return materialSpecEmissiveLe(primitive.material) != null;
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
  mergedUvByTexCoord?: ReadonlyMap<number, Float32Array>,
): MeshAreaLightsData {
  const emitterByMesh = collectMeshAreaSources(scene);
  if (emitterByMesh.size === 0) {
    return { data: null, dim: 1, triLightCount: 0, totalEmissiveArea: 0, totalEmissivePower: 0 };
  }

  const stride = merged.positionStrideFloats;
  const idx = merged.mergedIndices;
  const pos = merged.positions;
  const bvhFaceByMergedFace = buildBvhFaceByMergedFace(merged);
  const uvStreams = mergedUvByTexCoord ??
    buildUvAttributeLayout(scene, merged, merged.materials).mergedByTexCoord;

  // Collect (v0,v1,v2,radiance,area,shadow flag) for every triangle of every emissive mesh.
  const tris: {
    v0: Vec3;
    v1: Vec3;
    v2: Vec3;
    rad: Vec3;
    area: number;
    power: number;
    castShadowDisabled: number;
    twoSided: number;
    targetFaceIndex: number;
  }[] = [];
  let totalEmissiveArea = 0;
  let totalEmissivePower = 0;
  for (const range of merged.meshVertexRanges) {
    const emitter = emitterByMesh.get(range.name);
    if (emitter == null) continue;
    for (let t = 0; t < range.triCount; t += 1) {
      const tri = range.triStart + t;
      if (tri >= bvhFaceByMergedFace.length) {
        throw new RangeError(
          `@vitrum/pt-webgl2: emissive triangle ${range.name}:${t} has no BVH face identity.`,
        );
      }
      const targetFaceIndex = bvhFaceByMergedFace[tri]!;
      const i0 = idx[tri * 3] ?? 0;
      const i1 = idx[tri * 3 + 1] ?? 0;
      const i2 = idx[tri * 3 + 2] ?? 0;
      const v0 = v3(pos, i0, stride);
      const v1 = v3(pos, i1, stride);
      const v2 = v3(pos, i2, stride);
      const sourceAreaMeasure = classifyTriangleAreaF32(v0, v1, v2);
      if (!sourceAreaMeasure.valid) {
        if (sourceAreaMeasure.reason === 'degenerate') continue;
        throw new RangeError(
          `@vitrum/pt-webgl2: emissive triangle ${range.name}:${t} has ` +
          `${sourceAreaMeasure.reason.replaceAll('-', ' ')} geometry.`,
        );
      }
      const authoredTexCoord = emitter.implicitMaterial?.emissiveMap?.texCoord ?? 0;
      const selectedUvs = uvStreams.get(authoredTexCoord);
      const uvA = uvAt(selectedUvs, i0);
      const uvB = uvAt(selectedUvs, i1);
      const uvC = uvAt(selectedUvs, i2);
      // Shared CPU emissive helpers accept UV0/UV1 positional inputs. Feed the
      // exact authored stream as UV0 and normalize only the helper-local ref;
      // this preserves texCoord semantics without changing shared-bvh's API.
      const samplingMaterial = emitter.implicitMaterial?.emissiveMap == null || authoredTexCoord === 0
        ? emitter.implicitMaterial
        : {
            ...emitter.implicitMaterial,
            emissiveMap: { ...emitter.implicitMaterial.emissiveMap, texCoord: 0 },
          };

      const pushLight = (
        tv0: Vec3,
        tv1: Vec3,
        tv2: Vec3,
        radianceOverride?: readonly [number, number, number],
      ): void => {
        const areaMeasure = classifyTriangleAreaF32(tv0, tv1, tv2);
        if (!areaMeasure.valid) {
          if (areaMeasure.reason === 'degenerate') return;
          throw new RangeError(
            `@vitrum/pt-webgl2: emissive triangle ${range.name}:${t} has ` +
            `${areaMeasure.reason.replaceAll('-', ' ')} sub-triangle geometry.`,
          );
        }
        const triArea = areaMeasure.area;
        const rad = radianceOverride ?? emitter.radiance;
        assertRadianceSurvivesF32(rad, `emissive triangle ${range.name}:${t}`);
        const emittedLuminance = luminanceRgb(rad);
        if (!(emittedLuminance > 0)) return;
        assertPositiveValueSurvivesF32(triArea, `emissive triangle ${range.name}:${t} area`);
        const exactPower = emittedLuminance * triArea;
        assertPositiveValueSurvivesF32(
          exactPower,
          `emissive triangle ${range.name}:${t} selection power`,
        );
        const power = Math.fround(exactPower);
        const nextTotalEmissivePower = addMeshSelectionPowerF32(
          totalEmissivePower,
          power,
          `@vitrum/pt-webgl2: emissive triangle ${range.name}:${t}`,
        );
        tris.push({
          v0: tv0,
          v1: tv1,
          v2: tv2,
          rad,
          area: triArea,
          power,
          castShadowDisabled: emitter.castShadowDisabled,
          twoSided: emitter.twoSided,
          targetFaceIndex,
        });
        totalEmissiveArea += triArea;
        totalEmissivePower = nextTotalEmissivePower;
      };

      if (samplingMaterial?.emissiveMap == null) {
        pushLight(v0, v1, v2);
        continue;
      }

      const exactTexelHandled = forEachEmissiveMapTexelSubTriangle(
            samplingMaterial,
            uvA,
            uvB,
            uvC,
            undefined,
            undefined,
            undefined,
            (wa, wb, wc, texelRadiance) => {
              pushLight(
                baryVec3(v0, v1, v2, wa),
                baryVec3(v0, v1, v2, wb),
                baryVec3(v0, v1, v2, wc),
                texelRadiance,
              );
            },
            4096,
            undefined,
            resolveWebGl2MappedEmitterRadiance,
          );
      if (exactTexelHandled) continue;
      throw new TypeError(
        `@vitrum/pt-webgl2: primitive "${range.name}" emissiveMap cannot be represented by exact ` +
        'texel-cell NEE. Exact mapped emitters require CPU-readable pixels, nearest mag/min filters, ' +
        'mipFilter "none", and at most 4096 covered texel cells.',
      );
    }
  }

  if (tris.length === 0) {
    return { data: null, dim: 1, triLightCount: 0, totalEmissiveArea: 0, totalEmissivePower: 0 };
  }
  for (const tri of tris) {
    if (!(Math.fround(tri.power / totalEmissivePower) > 0)) {
      throw new RangeError(
        '@vitrum/pt-webgl2: a positive mesh-emitter selection probability ' +
          'collapses to zero in float32.',
      );
    }
  }

  const pixelCount = tris.length * TRI_LIGHT_PIXELS;
  const dim = Math.ceil(Math.sqrt(pixelCount));
  const data = new Float32Array(dim * dim * 4);
  for (let i = 0; i < tris.length; i += 1) {
    const {
      v0,
      v1,
      v2,
      rad,
      area,
      power,
      castShadowDisabled,
      twoSided,
      targetFaceIndex,
    } = tris[i]!;
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
    // s4.r: selection power (luminance(radiance) * area) for energy-weighted NEE.
    data[base + 16] = power;
    // s5.r/a: exact uint32 BVH face id as individually exact 16-bit words.
    data[base + 20] = targetFaceIndex % UINT16_BASE;
    data[base + 21] = castShadowDisabled;
    data[base + 22] = twoSided;
    data[base + 23] = Math.floor(targetFaceIndex / UINT16_BASE);
  }

  return { data, dim, triLightCount: tris.length, totalEmissiveArea, totalEmissivePower };
}
