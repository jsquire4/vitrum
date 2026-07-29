import {
  type DiscAreaEmitter,
  type Mat4,
  type MaterialSpec,
  type MeshAreaEmitter,
  type Scene,
} from '@vitrum/core';
import {
  invertMat4,
  isTextureRefCpuReadable,
  materialSpecEmissiveLe,
  materialSpecScalarEmissiveLe,
  materialSpecSkipEmitter,
  resolveDisplacedGeometry,
} from '@vitrum/shared-bvh';
import { luminance, type LightTreeBuildInput } from '@vitrum/shared-samplers';
import { transformPoint } from '../math/mat4.js';
import { environmentParams } from './environmentPacking.js';
import {
  discArea,
  meshTriangleArea,
  rectQuadArea,
  walkPositionalEmitters,
} from '../bdpt/flatEmitterWalk.js';

/**
 * Directional light layout (2 vec4 = 8 floats, N-directional expansion):
 *   vec4 0: direction.xyz (normalized toward light), angularDiameter (radians)
 *   vec4 1: irradiance.rgb, mean_irradiance
 *   angularDiameter = 0 ⟹ perfect delta directional (historical exact path, byte-identical).
 *   mean_irradiance = (r+g+b)/3 — cached for storage/texture consumers.
 *   SHADOW-01: emitter castShadow:false is SIGN-ENCODED into the angularDiameter
 *   lane (both vec4s are otherwise full): packed = -1 - angularDiameter when the
 *   flag is false; the kernel decodes shadowDisabled = (raw < 0) and
 *   angularDiameter = -1 - raw. castShadow:true packs the raw value (≥ 0),
 *   byte-identical to the pre-SHADOW-01 layout.
 *
 * The full tier reads this storage buffer directly; the lite tier receives the
 * same records through its packed sampled-light texture. No first-directional
 * frame-UBO mirror exists.
 */
export const DIRECTIONAL_LIGHT_FLOAT_STRIDE = 8;

/**
 * Point light layout (3 vec4 = 12 floats, H51-D):
 *   vec4 0: position.xyz, 0
 *   vec4 1: radiance.rgb, 0
 *   vec4 2: distance, decay, castShadowDisabled, 0
 *   distance = 0 ⟹ no cutoff (infinite range); decay = 0 ⟹ no falloff.
 *   castShadowDisabled (SHADOW-01) = 1.0 ⟺ emitter castShadow:false; 0.0 default.
 */
export const POINT_LIGHT_FLOAT_STRIDE = 12;

/**
 * Spot light layout (4 vec4 = 16 floats, H51-D):
 *   vec4 0: position.xyz, 0
 *   vec4 1: direction.xyz, cos(outerAngle)
 *   vec4 2: radiance.rgb, cos(innerAngle)   — innerAngle = outerAngle·(1−penumbra)
 *   vec4 3: distance, decay, castShadowDisabled, 0
 *   distance = 0 ⟹ no cutoff; decay = 0 ⟹ no falloff.
 *   castShadowDisabled (SHADOW-01) = 1.0 ⟺ emitter castShadow:false; 0.0 default.
 */
export const SPOT_LIGHT_FLOAT_STRIDE = 16;
export const RECT_AREA_LIGHT_FLOAT_STRIDE = 16;
/**
 * Mesh-area record (7 vec4 = 28 floats):
 *   0..2 position vertices; 3 average Le + castShadowDisabled;
 *   4 raw emissive UV A/B; 5 raw UV C + (materialId+1) + source world area;
 *   6 authored base Le. A zero materialId+1 marks an untextured emitter.
 */
export const MESH_AREA_LIGHT_FLOAT_STRIDE = 28;

/**
 * Rect/disc area light record layout (4 vec4 = 16 floats):
 *   vec4 0: center.xyz, castShadowDisabled (SHADOW-01: 1.0 ⟺ castShadow:false; 0.0 default)
 *   vec4 1: uAxis.xyz, 0          — for rect: half-extent along u; for disc: tangent × radius
 *   vec4 2: vAxis.xyz, 0          — for rect: half-extent along v; for disc: bitangent × radius
 *   vec4 3: radiance.rgb, shape   — shape = 0.0 (rect) | 1.0 (disc, analytic concentric-map)
 *
 * Rect sampling: uniform in [-1,1]² → p = center + uAxis*u + vAxis*v; area = 4·|u×v|
 * Disc sampling: concentric-map (xi₁,xi₂)→(r,φ) → p = center + uAxis*(r·cos φ) + vAxis*(r·sin φ)
 *   area = π·|uAxis×vAxis| (= π·r² for the canonical orthogonal radius frame).
 *   pdf = dist² / (cosLight · area) — the same solid-angle conversion as a rect.
 * Forward-hit MIS (connect.wgsl): ray-plane + |rel_u|²+|rel_v|²≤1 circle test for disc;
 *   uCoord/vCoord ∈ [-1,1] box test for rect.  |shape.w - 1.0| < 0.5 discriminates disc.
 */
const RECT_DISC_SHAPE_RECT = 0.0;
const RECT_DISC_SHAPE_DISC = 1.0;

type Vec3 = [number, number, number];

type PackedMeshAreaTriangle = {
  readonly triA: Vec3;
  readonly triB: Vec3;
  readonly triC: Vec3;
  readonly radiance: Vec3;
  readonly baseRadiance: Vec3;
  readonly emissiveRawUvs: readonly [
    readonly [number, number],
    readonly [number, number],
    readonly [number, number],
  ];
  readonly mappedMaterialId: number | null;
  readonly sourceWorldArea: number;
  /** Emitted-power proxy used by the NEE cap and light tree: luminance(Le) · area. */
  readonly power: number;
  /** SHADOW-01 — true ⟺ source mesh-area emitter set castShadow:false.
   *  Packed as 1.0 into the radiance vec4's .w lane (0.0 default). */
  readonly castShadowDisabled: boolean;
};

type MeshAreaTrianglePackOptions = {
  readonly materialIdByPrimitive?: ReadonlyMap<string, number>;
};

/**
 * Mesh-area production proposal limit — prevents a large emissive mesh from
 * producing an unbounded GPU buffer (28 floats × 4 bytes per triangle) and an oversized CPU
 * light-tree build (O(N log N)).
 *
 * Limit = 65 536 triangles ≈ 7 MiB for the primary records, plus the
 * light-tree storage.
 *
 * Rationale:
 *   - A 1M-triangle emissive mesh would require at least 112 MB for primary
 *     records alone and a slow tree build per setScene.
 *   - Production fails synchronously above the limit. Silently dropping even a
 *     dim triangle would leave forward-hit emission with zero NEE proposal
 *     support and bias MIS.
 */
export const MESH_AREA_LIGHT_TRI_CAP = 65536;

export interface PackedEmitterArrays {
  readonly warnings: string[];
  readonly directionalLightCount: number;
  readonly pointLightCount: number;
  readonly spotLightCount: number;
  readonly rectAreaLightCount: number;
  readonly meshAreaLightCount: number;
  readonly directionalLightsData: Float32Array;
  readonly pointLightsData: Float32Array;
  readonly spotLightsData: Float32Array;
  readonly rectAreaLightsData: Float32Array;
  readonly meshAreaLightsData: Float32Array;
}

function pushVec4(
  out: number[],
  v: readonly [number, number, number],
  w = 0,
): void {
  out.push(v[0], v[1], v[2], w);
}

function packedFloatData(
  values: readonly number[],
  count: number,
  stride: number,
  label: string,
): Float32Array {
  const expected = count * stride;
  if (values.length !== expected) {
    throw new Error(
      `@vitrum/pt-webgpu: internal ${label} packing mismatch (${values.length} floats, expected ${expected}).`,
    );
  }
  return new Float32Array(values);
}

function emitterRadiance(
  e: Pick<DiscAreaEmitter | MeshAreaEmitter, 'color' | 'intensity'>,
): Vec3 {
  return [
    e.color[0] * e.intensity,
    e.color[1] * e.intensity,
    e.color[2] * e.intensity,
  ];
}

function uvAt(uvs: Float32Array | undefined, vertex: number): [number, number] {
  if (uvs == null) return [0, 0];
  return [uvs[vertex * 2] ?? 0, uvs[vertex * 2 + 1] ?? 0];
}

function assertEmissiveMapCpuReadable(material: MaterialSpec, primitiveId: string): void {
  if (material.emissiveMap == null || isTextureRefCpuReadable(material.emissiveMap, 'srgb')) {
    return;
  }
  throw new TypeError(
    `@vitrum/pt-webgpu: primitive "${primitiveId}" uses an emissiveMap without ` +
      'complete CPU-readable texels. Emissive-map NEE cannot substitute scalar emission ' +
      'without biasing forward-hit MIS. Supply a CPU-readable texture payload, or wrap a ' +
      'GPUTexture with createPtWebgpuTextureSource(..., { cpuMirror }).',
  );
}

function emissiveRadianceForMaterial(
  material: MaterialSpec,
  primitiveId: string,
): Vec3 {
  if (materialSpecScalarEmissiveLe(material) == null) return [0, 0, 0];
  assertEmissiveMapCpuReadable(material, primitiveId);
  return materialSpecEmissiveLe(material) ?? [0, 0, 0];
}

function hasPositiveRadiance(radiance: readonly [number, number, number]): boolean {
  return radiance[0] > 0 || radiance[1] > 0 || radiance[2] > 0;
}

function f32Dot3(x: number, y: number, z: number): number {
  const xx = Math.fround(Math.fround(x) * Math.fround(x));
  const yy = Math.fround(Math.fround(y) * Math.fround(y));
  const zz = Math.fround(Math.fround(z) * Math.fround(z));
  return Math.fround(Math.fround(xx + yy) + zz);
}

function f32Cross(
  a: readonly [number, number, number],
  b: readonly [number, number, number],
): Vec3 {
  const ax = Math.fround(a[0]);
  const ay = Math.fround(a[1]);
  const az = Math.fround(a[2]);
  const bx = Math.fround(b[0]);
  const by = Math.fround(b[1]);
  const bz = Math.fround(b[2]);
  return [
    Math.fround(Math.fround(ay * bz) - Math.fround(az * by)),
    Math.fround(Math.fround(az * bx) - Math.fround(ax * bz)),
    Math.fround(Math.fround(ax * by) - Math.fround(ay * bx)),
  ];
}

function f32TriangleHasPositiveArea(a: Vec3, b: Vec3, c: Vec3): boolean {
  const ab: Vec3 = [
    Math.fround(Math.fround(b[0]) - Math.fround(a[0])),
    Math.fround(Math.fround(b[1]) - Math.fround(a[1])),
    Math.fround(Math.fround(b[2]) - Math.fround(a[2])),
  ];
  const ac: Vec3 = [
    Math.fround(Math.fround(c[0]) - Math.fround(a[0])),
    Math.fround(Math.fround(c[1]) - Math.fround(a[1])),
    Math.fround(Math.fround(c[2]) - Math.fround(a[2])),
  ];
  const cross = f32Cross(ab, ac);
  return f32Dot3(cross[0], cross[1], cross[2]) > 0;
}

function assertUniqueMeshAreaEmitterOwnership(scene: Scene): void {
  const ownerByMeshId = new Map<string, string>();
  for (const emitter of scene.emitters) {
    if (emitter.kind !== 'mesh-area') continue;
    const previousOwner = ownerByMeshId.get(emitter.meshId);
    if (previousOwner != null) {
      throw new TypeError(
        `@vitrum/pt-webgpu: mesh primitive "${emitter.meshId}" is referenced by multiple ` +
          `mesh-area emitters ("${previousOwner}" and "${emitter.id}"). A surface may have ` +
          'only one explicit mesh-area emitter so forward-hit radiance and its MIS proposal ' +
          'have one unambiguous owner.',
      );
    }
    ownerByMeshId.set(emitter.meshId, emitter.id);
  }
}

/**
 * Pack a disc-area emitter as a rect-area record with a disc shape tag (1.0) in
 * the emission.w lane. The uAxis/vAxis carry the radius-scaled orthonormal tangent
 * basis so the WGSL sampler can recover radius = |uAxis| = |vAxis|.
 *
 * Record layout (4 × vec4f, 16 floats):
 *   [0] center.xyz, 0
 *   [1] (tangent × radius).xyz, 0        — uAxis
 *   [2] (bitangent × radius).xyz, 0      — vAxis
 *   [3] radiance.rgb, RECT_DISC_SHAPE_DISC
 *
 * The WGSL side reads shape = rectAreaLights[rb+3].w and branches:
 *   shape ≈ 0 → rect sampling (uniform [-1,1]², area = 4|u×v|)
 *   shape ≈ 1 → disc sampling (concentric-map, area = π|u×v|)
 *
 * Native analytic disc emitters replace the 32-triangle fan, 2026-06-10 —
 * RENDER-CHANGING for disc-lit scenes, A/B in R9-B.
 */
function packDiscAsRect(e: DiscAreaEmitter): readonly number[] {
  if (
    !Number.isFinite(e.radius) ||
    !(e.radius > 0) ||
    !(f32Dot3(e.radius, 0, 0) > 0)
  ) {
    throw new RangeError(
      `@vitrum/pt-webgpu: disc-area emitter "${e.id}" has a non-positive or ` +
        'non-f32-representable radius.',
    );
  }
  const nx = e.normal[0];
  const ny = e.normal[1];
  const nz = e.normal[2];
  const nLen = Math.hypot(nx, ny, nz);
  if (!Number.isFinite(nLen) || !(nLen > 0)) {
    throw new RangeError(
      `@vitrum/pt-webgpu: disc-area emitter "${e.id}" has a degenerate normal.`,
    );
  }
  // Build orthonormal tangent basis (tangent, bitangent) for the disc plane.
  const ux = nx / nLen;
  const uy = ny / nLen;
  const uz = nz / nLen;
  // Choose a helper vector not parallel to the normal.
  let ax = 0, ay = 1, az = 0;
  if (Math.abs(uy) > 0.999) { ax = 1; ay = 0; az = 0; }
  const tx = ay * uz - az * uy;
  const ty = az * ux - ax * uz;
  const tz = ax * uy - ay * ux;
  const tLen = Math.hypot(tx, ty, tz);
  if (!Number.isFinite(tLen) || !(tLen > 0)) {
    throw new RangeError(
      `@vitrum/pt-webgpu: disc-area emitter "${e.id}" has a degenerate tangent basis.`,
    );
  }
  const tcx = tx / tLen;
  const tcy = ty / tLen;
  const tcz = tz / tLen;
  // Bitangent = normal × tangent (right-hand rule, unit length because n and t are orthonormal).
  const bx = uy * tcz - uz * tcy;
  const by = uz * tcx - ux * tcz;
  const bz = ux * tcy - uy * tcx;
  // Scale both axes by radius — the WGSL recovers radius as |uAxis|.
  const r = e.radius;
  const uAxis: Vec3 = [tcx * r, tcy * r, tcz * r];
  const vAxis: Vec3 = [bx * r, by * r, bz * r];
  const axisCross = f32Cross(uAxis, vAxis);
  if (
    !(f32Dot3(uAxis[0], uAxis[1], uAxis[2]) > 0) ||
    !(f32Dot3(vAxis[0], vAxis[1], vAxis[2]) > 0) ||
    !(f32Dot3(axisCross[0], axisCross[1], axisCross[2]) > 0)
  ) {
    throw new RangeError(
      `@vitrum/pt-webgpu: disc-area emitter "${e.id}" does not retain ` +
        'strictly positive area in f32 shader arithmetic.',
    );
  }
  const rad = emitterRadiance(e);
  return [
    // vec4 0: center (.w = SHADOW-01 castShadowDisabled, 0.0 default)
    e.position[0], e.position[1], e.position[2], e.castShadow === false ? 1 : 0,
    // vec4 1: uAxis = tangent × radius
    uAxis[0], uAxis[1], uAxis[2], 0,
    // vec4 2: vAxis = bitangent × radius
    vAxis[0], vAxis[1], vAxis[2], 0,
    // vec4 3: radiance.rgb, shape = DISC (1.0)
    rad[0], rad[1], rad[2], RECT_DISC_SHAPE_DISC,
  ];
}

function packMeshAreaTriangles(
  emitter: MeshAreaEmitter,
  scene: Scene,
  warnings: string[],
  options: MeshAreaTrianglePackOptions = {},
): readonly PackedMeshAreaTriangle[] {
  const primitive = scene.primitives.find((p) => p.id === emitter.meshId);
  if (primitive == null || primitive.kind === 'analytic') {
    warnings.push(`Mesh-area emitter "${emitter.id}" references missing or non-mesh primitive "${emitter.meshId}".`);
    return [];
  }
  // Resolve through the exact canonical geometry preamble used by both
  // packSceneFromCore (TLAS tier) and mergeWorldSpaceFromCore (lite tier).
  // Sampling authored vertices while the BVH intersects displaced/diced
  // vertices gives NEE and forward hits disjoint geometric support.
  const resolved = resolveDisplacedGeometry(
    primitive,
    (warning) => warnings.push(warning),
  );
  const positions = resolved.sourcePositions;
  const indices = resolved.baseIndicesSource;
  const uv0 = resolved.baseUvs;
  const uv1 = resolved.baseUv1;
  const uvSets = resolved.baseUvSets;
  const vertexCount = Math.floor(positions.length / 3);
  const triangleIndexCount = indices?.length ?? vertexCount;
  const triangleCount = Math.floor(triangleIndexCount / 3);
  if (triangleCount < 1 || vertexCount < 3 || positions.length < 9) {
    warnings.push(`Mesh-area emitter "${emitter.id}" references primitive "${emitter.meshId}" with no triangles.`);
    return [];
  }
  const fetchPos = (idx: number): [number, number, number] => [
    positions[idx * 3] ?? 0,
    positions[idx * 3 + 1] ?? 0,
    positions[idx * 3 + 2] ?? 0,
  ];
  const indexAt = (offset: number): number => indices?.[offset] ?? offset;
  const transforms: readonly (Mat4 | undefined)[] = primitive.kind === 'instanced-mesh'
    ? primitive.instances
    : [primitive.transform];
  if (transforms.length === 0) {
    warnings.push(`Mesh-area emitter "${emitter.id}" references instanced primitive "${emitter.meshId}" with no instances.`);
    return [];
  }
  const radiance = emitterRadiance(emitter);
  if (hasPositiveRadiance(radiance)) {
    assertEmissiveMapCpuReadable(primitive.material, primitive.id);
  }
  const implicitMaterial = emitter.id === `__implicit__${primitive.id}` ? primitive.material : undefined;
  const mappedRadianceMaterial: MaterialSpec | undefined = implicitMaterial ??
    (primitive.material.emissiveMap != null
      ? {
          ...primitive.material,
          emissive: [radiance[0], radiance[1], radiance[2]],
          emissiveIntensity: 1,
        }
      : undefined);
  const mappedBaseRadiance: Vec3 = implicitMaterial == null
    ? radiance
    : [
        (implicitMaterial.emissive?.[0] ?? 0) * (implicitMaterial.emissiveIntensity ?? 1),
        (implicitMaterial.emissive?.[1] ?? 0) * (implicitMaterial.emissiveIntensity ?? 1),
        (implicitMaterial.emissive?.[2] ?? 0) * (implicitMaterial.emissiveIntensity ?? 1),
      ];
  const mappedMaterialId = mappedRadianceMaterial == null
    ? null
    : options.materialIdByPrimitive?.get(primitive.id) ?? (() => {
        let materialId = 0;
        for (const candidate of scene.primitives) {
          if (candidate.id === primitive.id) return materialId;
          // packEmitterArrays receives the capability-filtered scene in production;
          // every surviving primitive contributes one dense material slot.
          materialId += 1;
        }
        return -1;
      })();
  if (mappedRadianceMaterial != null && (mappedMaterialId == null || mappedMaterialId < 0)) {
    throw new Error(
      `@vitrum/pt-webgpu: cannot resolve material slot for mapped mesh emitter "${emitter.id}".`,
    );
  }
  // SHADOW-01 — carry the emitter's castShadow flag onto every packed triangle.
  const castShadowDisabled = emitter.castShadow === false;
  const packed: PackedMeshAreaTriangle[] = [];
  let invalidTriangleCount = 0;
  let degenerateTriangleCount = 0;
  for (const transform of transforms) {
    // packSceneFromCore deliberately omits singular TLAS instances. Matching
    // that publication rule here prevents NEE from sampling emitter geometry
    // that no renderer hit can ever reach.
    if (transform != null && invertMat4(transform) == null) {
      warnings.push(
        `Mesh-area emitter "${emitter.id}" skipped a non-invertible instance ` +
          `transform for primitive "${emitter.meshId}".`,
      );
      continue;
    }
    for (let tri = 0; tri < triangleCount; tri += 1) {
      const base = tri * 3;
      const i0 = indexAt(base);
      const i1 = indexAt(base + 1);
      const i2 = indexAt(base + 2);
      if (
        i0 < 0 || i0 >= vertexCount ||
        i1 < 0 || i1 >= vertexCount ||
        i2 < 0 || i2 >= vertexCount
      ) {
        invalidTriangleCount += 1;
        continue;
      }
      let a = fetchPos(i0);
      let b = fetchPos(i1);
      let c = fetchPos(i2);
      if (transform != null) {
        a = transformPoint(transform, a);
        b = transformPoint(transform, b);
        c = transformPoint(transform, c);
      }
      const sourceArea = meshTriangleArea(a, b, c);
      if (!Number.isFinite(sourceArea)) {
        throw new RangeError(
          `@vitrum/pt-webgpu: mesh-area emitter "${emitter.id}" produced ` +
            'non-finite transformed triangle geometry.',
        );
      }
      if (!(sourceArea > 0) || !f32TriangleHasPositiveArea(a, b, c)) {
        degenerateTriangleCount += 1;
        continue;
      }
      const uv0A = uvAt(uv0, i0);
      const uv0B = uvAt(uv0, i1);
      const uv0C = uvAt(uv0, i2);
      const uv1A = uvAt(uv1, i0);
      const uv1B = uvAt(uv1, i1);
      const uv1C = uvAt(uv1, i2);
      const mappedTexCoord = mappedRadianceMaterial?.emissiveMap?.texCoord ?? 0;
      const highUvStream = mappedTexCoord > 1
        ? uvSets?.[mappedTexCoord]
        : undefined;
      const selectedHighUv = highUvStream == null
        ? undefined
        : [
            uvAt(highUvStream, i0),
            uvAt(highUvStream, i1),
            uvAt(highUvStream, i2),
          ] as const;
      const selectedRawUvs = mappedTexCoord === 0
        ? [uv0A, uv0B, uv0C] as const
        : mappedTexCoord === 1
          ? [uv1A, uv1B, uv1C] as const
          : selectedHighUv ?? ([[0, 0], [0, 0], [0, 0]] as const);

      const pushTriangle = (
        triA: Vec3,
        triB: Vec3,
        triC: Vec3,
        rawEmissiveUvA: readonly [number, number],
        rawEmissiveUvB: readonly [number, number],
        rawEmissiveUvC: readonly [number, number],
      ): void => {
        const area = meshTriangleArea(triA, triB, triC);
        if (!Number.isFinite(area)) {
          throw new RangeError(
            `@vitrum/pt-webgpu: mesh-area emitter "${emitter.id}" produced ` +
              'non-finite subdivided triangle geometry.',
          );
        }
        if (!(area > 0) || !f32TriangleHasPositiveArea(triA, triB, triC)) return;
        // This record's RGB is a strictly-positive proposal proxy whenever the
        // authored base Le can emit. The GPU evaluates the exact mapped Le at
        // the sampled point; using the base Le here guarantees non-zero PMF
        // support even for a single bright texel, bilinear seam, or mip bleed.
        const triangleRadiance: Vec3 = [
          mappedBaseRadiance[0],
          mappedBaseRadiance[1],
          mappedBaseRadiance[2],
        ];
        const emittedLuminance = luminance(triangleRadiance[0], triangleRadiance[1], triangleRadiance[2]);
        if (!hasPositiveRadiance(triangleRadiance)) {
          return;
        }
        packed.push({
          triA,
          triB,
          triC,
          radiance: [triangleRadiance[0], triangleRadiance[1], triangleRadiance[2]],
          baseRadiance: mappedRadianceMaterial == null
            ? [triangleRadiance[0], triangleRadiance[1], triangleRadiance[2]]
            : [mappedBaseRadiance[0], mappedBaseRadiance[1], mappedBaseRadiance[2]],
          emissiveRawUvs: [
            [rawEmissiveUvA[0], rawEmissiveUvA[1]],
            [rawEmissiveUvB[0], rawEmissiveUvB[1]],
            [rawEmissiveUvC[0], rawEmissiveUvC[1]],
          ],
          mappedMaterialId,
          sourceWorldArea: area,
          power: emittedLuminance * area,
          castShadowDisabled,
        });
      };

      // One record retains the complete source triangle. Subdivision or texel
      // pruning cannot preserve support under authored bilinear/mipmap/wrap
      // filtering and is therefore not a valid production fallback.
      pushTriangle(
        a, b, c,
        selectedRawUvs[0], selectedRawUvs[1], selectedRawUvs[2],
      );
    }
  }
  if (invalidTriangleCount > 0) {
    warnings.push(
      `Mesh-area emitter "${emitter.id}" skipped ${invalidTriangleCount} triangle(s) with out-of-range indices.`,
    );
  }
  if (degenerateTriangleCount > 0) {
    warnings.push(
      `Mesh-area emitter "${emitter.id}" skipped ${degenerateTriangleCount} degenerate triangle(s).`,
    );
  }
  if (packed.length === 0) {
    warnings.push(`Mesh-area emitter "${emitter.id}" produced no non-degenerate triangles.`);
  }
  return packed;
}

/**
 * H14-A — Synthesize implicit mesh-area emitters for every mesh-like primitive
 * whose material has any finite positive emissive energy AND that has NO
 * explicit `mesh-area` emitter already referencing it. There is deliberately
 * no artistic luminance cutoff: every surface that can contribute forward-hit
 * radiance must retain light-sampling support, however dim.
 *
 * Without this synthesis, an emissive mesh is invisible to NEE/BDPT — the
 * kernel's emissive-on-hit term fires, but the direct-lighting estimators never
 * enumerate it. The synthetic emitters are virtual (id = `__implicit__<primitiveId>`)
 * and carry the material's coarse average emissive radiance as `color · intensity = 1`.
 * Production mesh-area packing retains each complete source triangle and the GPU
 * evaluates the exact authored emissive sample at the selected surface point.
 * Opaque/unreadable map handles are rejected before GPU allocation because scalar
 * substitution would disagree with forward-hit MIS.
 *
 * Guard: explicit mesh-area emitters take priority — if a `mesh-area` entry in
 * `scene.emitters` already references a primitive, that primitive is skipped to
 * prevent double-counting. Disc-area emitters are packed natively into the rect
 * stream and do NOT generate mesh-area triangles, so they are not excluded here.
 *
 * This function is the single source of truth for the implicit-emitter synthesis
 * logic and is reused by `hasMeshAreaEmitterForPrimitive`.
 */
function synthesizeImplicitEmitters(
  scene: Scene,
  /** When set, only consider this primitive id (early-exit fast path for the
   *  `hasMeshAreaEmitterForPrimitive` staleness predicate). */
  onlyPrimitiveId?: string,
  _warnings?: string[],
): Extract<Scene['emitters'][number], { kind: 'mesh-area' }>[] {
  const explicitMeshAreaIds = new Set<string>(
    scene.emitters
      .filter((e) => e.kind === 'mesh-area')
      .map((e) => (e as { meshId?: string }).meshId ?? '')
      .filter(Boolean),
  );
  const result: Extract<Scene['emitters'][number], { kind: 'mesh-area' }>[] = [];
  for (const primitive of scene.primitives) {
    if (onlyPrimitiveId !== undefined && primitive.id !== onlyPrimitiveId) continue;
    if (primitive.kind === 'analytic') continue;
    if (explicitMeshAreaIds.has(primitive.id)) continue;
    if (materialSpecSkipEmitter(primitive.material)) continue;
    const [emR, emG, emB] = emissiveRadianceForMaterial(primitive.material, primitive.id);
    if (!hasPositiveRadiance([emR, emG, emB])) continue;
    result.push({
      kind: 'mesh-area',
      id: `__implicit__${primitive.id}`,
      meshId: primitive.id,
      color: [emR, emG, emB],
      intensity: 1,
    });
  }
  return result;
}

export function packEmitterArrays(
  scene: Scene,
  options: { readonly materialIdByPrimitive?: ReadonlyMap<string, number> } = {},
): PackedEmitterArrays {
  assertUniqueMeshAreaEmitterOwnership(scene);
  const warnings: string[] = [];

  // N-directional packing — all directional emitters go into a flat storage-buffer
  // array. Lite serializes these same records into its sampled light texture.
  const directionalLights: number[] = [];
  let directionalLightCount = 0;
  for (const e of scene.emitters) {
    if (e.kind !== 'directional') continue;
    const x = e.direction[0];
    const y = e.direction[1];
    const z = e.direction[2];
    const len = Math.hypot(x, y, z);
    // Normalize direction toward the light (convention: direction points AT the light,
    // incoming light direction is -direction, but for NEE we want the "toward light" vec).
    // Core contract: direction points AT the light → incoming is -direction.
    // In NEE we fire a ray TOWARD the light, so the direction to the light is: -dir/len.
    const ndx = len < 1e-8 ? 0 : -x / len;
    const ndy = len < 1e-8 ? 1 : -y / len;
    const ndz = len < 1e-8 ? 0 : -z / len;
    const ad = e.angularDiameter;
    let angularDiameter = ad != null && Number.isFinite(ad) && ad > 0 ? ad : 0;
    // SHADOW-01 — castShadow:false is sign-encoded into the angularDiameter
    // lane (see the layout doc above): packed = -1 - ad. Default (true) keeps
    // the raw non-negative value, byte-identical to the pre-SHADOW-01 pack.
    if (e.castShadow === false) {
      angularDiameter = -1 - angularDiameter;
    }
    const scale = e.intensity;
    const irrR = e.color[0] * scale;
    const irrG = e.color[1] * scale;
    const irrB = e.color[2] * scale;
    const meanIrr = (irrR + irrG + irrB) / 3;
    // vec4 0: towardLight.xyz, angularDiameter
    directionalLights.push(ndx, ndy, ndz, angularDiameter);
    // vec4 1: irradiance.rgb, mean_irradiance
    directionalLights.push(irrR, irrG, irrB, meanIrr);
    directionalLightCount += 1;
  }
  const directionalLightsData = packedFloatData(
    directionalLights,
    directionalLightCount,
    DIRECTIONAL_LIGHT_FLOAT_STRIDE,
    'directional-light',
  );

  const pointLights: number[] = [];
  let pointLightCount = 0;
  for (const e of scene.emitters) {
    if (e.kind !== 'point') continue;
    pushVec4(pointLights, [e.position[0], e.position[1], e.position[2]]);
    pushVec4(pointLights, [
      e.color[0] * e.intensity,
      e.color[1] * e.intensity,
      e.color[2] * e.intensity,
    ]);
    // H51-D: distance (0 = no cutoff) + decay (0 = no falloff)
    const ptDist = typeof e.distance === 'number' && e.distance > 0 ? e.distance : 0;
    const ptDecay = typeof e.decay === 'number' ? e.decay : 2;
    // SHADOW-01 — lane .z carries castShadowDisabled (0.0 default).
    pointLights.push(ptDist, ptDecay, e.castShadow === false ? 1 : 0, 0);
    pointLightCount += 1;
  }
  const pointLightsData = packedFloatData(
    pointLights,
    pointLightCount,
    POINT_LIGHT_FLOAT_STRIDE,
    'point-light',
  );

  const spotLights: number[] = [];
  let spotLightCount = 0;
  for (const e of scene.emitters) {
    if (e.kind !== 'spot') continue;
    const d = e.direction;
    const len = Math.hypot(d[0], d[1], d[2]);
    const dir: readonly [number, number, number] =
      len < 1e-8 ? [0, -1, 0] : [d[0] / len, d[1] / len, d[2] / len];
    // H51-D: penumbra [0,1] defines the soft inner cone.
    // innerAngle = outerAngle * (1 − penumbra).
    // cos(innerAngle) > cos(outerAngle) ⟹ smoothstep from inner→outer.
    const penumbra = Math.min(1, Math.max(0, e.penumbra ?? 0));
    const outerAngle = e.angle;
    const innerAngle = outerAngle * (1 - penumbra);
    const cosOuter = Math.cos(outerAngle);
    const cosInner = Math.cos(innerAngle);
    pushVec4(spotLights, [e.position[0], e.position[1], e.position[2]]);
    pushVec4(spotLights, [dir[0], dir[1], dir[2]], cosOuter);
    pushVec4(spotLights, [
      e.color[0] * e.intensity,
      e.color[1] * e.intensity,
      e.color[2] * e.intensity,
    ], cosInner);
    // H51-D: distance (0 = no cutoff) + decay (0 = no falloff, 2 = physical)
    const spDist = typeof e.distance === 'number' && e.distance > 0 ? e.distance : 0;
    const spDecay = typeof e.decay === 'number' ? e.decay : 2;
    // SHADOW-01 — lane .z carries castShadowDisabled (0.0 default).
    spotLights.push(spDist, spDecay, e.castShadow === false ? 1 : 0, 0);
    spotLightCount += 1;
  }
  const spotLightsData = packedFloatData(
    spotLights,
    spotLightCount,
    SPOT_LIGHT_FLOAT_STRIDE,
    'spot-light',
  );

  const rectAreaLights: number[] = [];
  let rectAreaLightCount = 0;
  // Rect-area emitters: shape tag = 0.0 (RECT_DISC_SHAPE_RECT).
  for (const e of scene.emitters) {
    if (e.kind !== 'rect-area') continue;
    const uAxis: Vec3 = [Math.fround(e.uAxis[0]), Math.fround(e.uAxis[1]), Math.fround(e.uAxis[2])];
    const vAxis: Vec3 = [Math.fround(e.vAxis[0]), Math.fround(e.vAxis[1]), Math.fround(e.vAxis[2])];
    const axisCross = f32Cross(uAxis, vAxis);
    if (
      !(f32Dot3(uAxis[0], uAxis[1], uAxis[2]) > 0) ||
      !(f32Dot3(vAxis[0], vAxis[1], vAxis[2]) > 0) ||
      !(f32Dot3(axisCross[0], axisCross[1], axisCross[2]) > 0)
    ) {
      throw new RangeError(
        `@vitrum/pt-webgpu: rect-area emitter "${e.id}" does not retain ` +
          'strictly positive area in f32 shader arithmetic.',
      );
    }
    rectAreaLights.push(
      // SHADOW-01 — center .w carries castShadowDisabled (0.0 default).
      e.position[0], e.position[1], e.position[2], e.castShadow === false ? 1 : 0,
      e.uAxis[0], e.uAxis[1], e.uAxis[2], 0,
      e.vAxis[0], e.vAxis[1], e.vAxis[2], 0,
      e.color[0] * e.intensity, e.color[1] * e.intensity, e.color[2] * e.intensity,
      RECT_DISC_SHAPE_RECT,
    );
    rectAreaLightCount += 1;
  }
  // Disc-area emitters: native analytic packing into the rect stream, shape tag = 1.0.
  // Native analytic disc emitters replace the 32-triangle fan, 2026-06-10 —
  // RENDER-CHANGING for disc-lit scenes, A/B in R9-B.
  for (const e of scene.emitters) {
    if (e.kind !== 'disc-area') continue;
    const discRecord = packDiscAsRect(e);
    rectAreaLights.push(...discRecord);
    rectAreaLightCount += 1;
  }
  const rectAreaLightsData = packedFloatData(
    rectAreaLights,
    rectAreaLightCount,
    RECT_AREA_LIGHT_FLOAT_STRIDE,
    'rect-area-light',
  );

  const meshAreaTriangles: PackedMeshAreaTriangle[] = [];
  for (const emitter of scene.emitters) {
    if (emitter.kind !== 'mesh-area') continue;
    meshAreaTriangles.push(...packMeshAreaTriangles(
      emitter, scene, warnings,
      {
        ...(options.materialIdByPrimitive == null
          ? {}
          : { materialIdByPrimitive: options.materialIdByPrimitive }),
      },
    ));
  }

  // H14-A: synthesize implicit mesh-area emitters and pack their triangles.
  // See `synthesizeImplicitEmitters` for the synthesis contract.
  for (const synthetic of synthesizeImplicitEmitters(scene, undefined, warnings)) {
    meshAreaTriangles.push(...packMeshAreaTriangles(
      synthetic, scene, warnings,
      {
        ...(options.materialIdByPrimitive == null
          ? {}
          : { materialIdByPrimitive: options.materialIdByPrimitive }),
      },
    ));
  }

  // Every source triangle whose exact filtered Le can be non-zero must remain
  // in the proposal. Silently dropping a tail at the storage cap would create
  // forward-hit support with zero light-sampling density, so fail before upload.
  if (meshAreaTriangles.length > MESH_AREA_LIGHT_TRI_CAP) {
    throw new RangeError(
      `@vitrum/pt-webgpu: mesh-area proposal requires ${meshAreaTriangles.length} triangles, ` +
        `exceeding the exact-support limit of ${MESH_AREA_LIGHT_TRI_CAP}. ` +
        'Split the scene or reduce emissive geometry; partial proposal support is rejected.',
    );
  }
  const cappedTriangles = meshAreaTriangles;

  const meshAreaLights: number[] = [];
  for (const tri of cappedTriangles) {
    pushVec4(meshAreaLights, tri.triA);
    pushVec4(meshAreaLights, tri.triB);
    pushVec4(meshAreaLights, tri.triC);
    // SHADOW-01 — radiance vec4 .w carries castShadowDisabled (0.0 default).
    pushVec4(meshAreaLights, tri.radiance, tri.castShadowDisabled ? 1 : 0);
    meshAreaLights.push(
      tri.emissiveRawUvs[0][0], tri.emissiveRawUvs[0][1],
      tri.emissiveRawUvs[1][0], tri.emissiveRawUvs[1][1],
      tri.emissiveRawUvs[2][0], tri.emissiveRawUvs[2][1],
      tri.mappedMaterialId == null ? 0 : tri.mappedMaterialId + 1,
      tri.sourceWorldArea,
    );
    pushVec4(meshAreaLights, tri.baseRadiance);
  }
  const meshAreaLightCount = cappedTriangles.length;
  const meshAreaLightsData = packedFloatData(
    meshAreaLights,
    meshAreaLightCount,
    MESH_AREA_LIGHT_FLOAT_STRIDE,
    'mesh-area-light',
  );
  return {
    warnings,
    directionalLightCount,
    pointLightCount,
    spotLightCount,
    rectAreaLightCount,
    meshAreaLightCount,
    directionalLightsData,
    pointLightsData,
    spotLightsData,
    rectAreaLightsData,
    meshAreaLightsData,
  };
}

// ────────────────────────────────────────────────────────────────────────────
// WS2 — Many-light importance sampling: per-emitter power + light-tree input.
//
// pt-webgpu's NEE picks ONE selectable light per shading event. The uniform pick
// (`floor(rand·lightCount)`, compensated by `·lightCount`) has variance that grows
// with the number of lights and ignores their relative brightness. Replacing the
// pick with a power-weighted light-tree descent concentrates samples on the bright
// / nearby lights, cutting NEE variance at equal cost. The tree is the CPU-built
// `@vitrum/shared-samplers` binary tree (Shirley 1996 median split, power-as-cost)
// with the Estévez & Kulla 2018 distance-weighted importance descent on the GPU.
//
// References:
//   - Conty Estévez, A. & Kulla, C. 2018, "Importance Sampling of Many Lights
//     with Adaptive Tree Splitting", Proc. ACM CGIT (power × proximity descent).
//   - Shirley, Smits, Wang, Zimmerman 1996, "Monte Carlo Techniques for Direct
//     Lighting Calculations", ACM TOG (power-weighted light-list partition).
// ────────────────────────────────────────────────────────────────────────────

/**
 * H11 — Returns `true` if `scene` contains a `mesh-area` (or lowered `disc-area`)
 * emitter whose `meshId` matches `primitiveId`, OR if the primitive has an
 * implicit mesh-area emitter synthesized at pack time (i.e. it is a non-analytic
 * primitive whose material has non-zero emissive energy AND no explicit mesh-area
 * emitter already references it).
 *
 * Used by the geometry/transform fast paths in `SceneMutationRouter` to detect
 * when moving a primitive's vertices/transform also shifts a mesh-area emitter's
 * world-space triangles — in which case `packEmitterArrays` must be re-run and
 * re-uploaded so the GPU NEE data stays in sync.
 *
 * Item 2b — the original check covered only explicit `scene.emitters` entries,
 * making it blind to the `__implicit__<id>` synthetic emitters generated by the
 * H14-A loop in `packEmitterArrays` (~404-422 above). An emissive mesh moving
 * without an explicit emitter entry would leave the GPU NEE triangle cache stale.
 * Now covers both: explicit match OR (non-analytic primitive with emissive > 0
 * AND no explicit mesh-area emitter already claiming it).
 *
 * Disc-area emitters are lowered into the mesh-triangle section at pack time, so
 * they are treated as mesh-area emitters for the staleness check.
 */
export function hasMeshAreaEmitterForPrimitive(scene: Scene, primitiveId: string): boolean {
  // disc-area emitters are now packed natively into the rect stream (no longer
  // lowered into mesh-area triangles), so only mesh-area emitters are checked here.
  for (const e of scene.emitters) {
    if (e.kind !== 'mesh-area') continue;
    const meshId = (e as { meshId?: string }).meshId;
    if (meshId === primitiveId) return true;
  }
  // Item 2b — also check for implicit emitters synthesized at pack time.
  // `synthesizeImplicitEmitters` is the single source of the any-positive
  // radiance check, so the staleness predicate cannot drift from the synthesis
  // path in `packEmitterArrays`.
  return synthesizeImplicitEmitters(scene, primitiveId).length > 0;
}

/** The core positional area-emitter kinds (carry a finite area for the power term). */
export const AREA_LIGHT_KINDS: ReadonlySet<string> = new Set([
  'rect-area',
  'disc-area',
  'mesh-area',
]);

type EmitterPowerKind = { readonly kind: 'delta' } | { readonly kind: 'area'; readonly area: number };

/**
 * Per-emitter luminous power used as the light-tree split cost + leaf weight.
 *
 * Delta lights (point / spot / directional) have no finite area, so their power
 * is the Rec.709 luminance of their radiance directly. Area lights (rect / disc /
 * mesh) integrate that luminance over their surface, so power is `luminance·area`
 * (the relative-brightness ordering is what the tree needs — an exact radiometric
 * watt value is unnecessary for *selection*, only the *relative* weights matter).
 *
 * Always ≥ 0 (a black emitter contributes zero power and is never preferentially
 * selected).
 */
export function emitterPower(
  radiance: readonly [number, number, number],
  kind: EmitterPowerKind,
): number {
  const lum = Math.max(0, luminance(radiance[0], radiance[1], radiance[2]));
  return kind.kind === 'area' ? lum * Math.max(0, kind.area) : lum;
}

function pointAabb(p: Vec3): { min: Vec3; max: Vec3 } {
  return { min: [p[0], p[1], p[2]], max: [p[0], p[1], p[2]] };
}

/**
 * Narrow environment summary consumed by `buildLightTreeInputForScene` to
 * determine whether the env leaf is present and what radiance proxy to use.
 * Extracted from `EnvironmentParams` (environmentPacking.ts) so the caller can
 * pass the already-computed result without re-running the full HDRI/sky bake.
 */
export interface EnvSummaryForTree {
  /** Whether the scene has a valid HDRI map (routes through HDRI CDF sampling). */
  readonly hasHdri: boolean;
  /**
   * Solid-angle-integrated environment luminance, with map intensity applied
   * exactly once. Derived by `environmentParams`; zero means the selectable
   * environment is radiometrically black.
   */
  readonly lightTreePower: number;
}

/**
 * Build the `@vitrum/shared-samplers` light-tree input over the SELECTABLE lights
 * of `scene`, in the EXACT order pt-webgpu's NEE walk iterates them so the tree's
 * `emitterIndex` aligns 1:1 with the kernel's linear `current` index:
 *
 *   directional? · point[] · spot[] · rect-area[] · mesh-triangle-area[] · env?
 *
 * `disc-area` emitters are packed natively by `packEmitterArrays` into the
 * rect-area stream with a disc shape tag (shapeTag = 1.0), so every tree leaf
 * still matches one GPU slot in the rect-area buffer.
 *
 * The `directional` / `env` slots are non-positional, so they are given the union
 * AABB of all positional lights (the "lit region"). Inside that AABB the descent's
 * distance term floors out, so those slots compete by POWER alone — exactly the
 * desired behaviour for an infinitely-distant directional light or a hemispherical
 * environment (no meaningful spatial proximity). When there are no positional
 * lights the AABB collapses to the origin and the floor dominates ⇒ pure
 * power-weighted selection, which is still a valid pmf.
 *
 * The leaf count MUST equal `lightCount` as computed in the kernel:
 *   `directional + pointCount + spotCount + rectAreaCount + meshAreaCount + env`.
 * The caller (`buildPackedScene`) passes the SAME packed counts/radiances the
 * GPU loops consume, so index alignment cannot drift from the capability-capped
 * packed arrays.
 *
 * `precomputed` — optional already-computed sub-results. When provided, the
 * function skips the corresponding internal calls (avoids re-running the
 * potentially-expensive HDRI/sky bake a second time for the same scene):
 *   - `packed`: result of a prior `packEmitterArrays(scene)` call.
 *   - `envSummary`: narrow env metadata derived from a prior `environmentParams(scene)`.
 * Both are assumed to come from the SAME scene to preserve byte-identity.
 *
 * Callers (all within `@vitrum/pt-webgpu` — not exported to external packages):
 *   - `buildPackedScene` (uploadSceneBuffers.ts): always passes both `packed` and
 *     `envSummary` from the same-scene results already computed above it.
 *   - `packLightTreeForScene` (uploadSceneBuffers.ts): forwards its own optional
 *     `precomputed` parameter; callers supply it when available.
 *   - `sceneMutationRouter.ts` (via `packLightTreeForScene`): passes `packed`
 *     and/or `envSummary` from already-computed incremental update results.
 *   - Test files (`lightTreeImportance`, `scenePack.emitters`, `nDirectionalPacking`,
 *     `packingNoDoubleWork`): legitimately omit `precomputed` — they exercise the
 *     internal recompute path for correctness verification.
 *
 * Because test callers legitimately omit `precomputed`, the parameter MUST remain
 * optional. No `console.warn` is appropriate here — omitting precomputed is valid
 * and intentional for tests and for incremental callers that don't hold stale
 * intermediate results.
 */
export function buildLightTreeInputForScene(
  scene: Scene,
  precomputed?: { packed?: PackedEmitterArrays; envSummary?: EnvSummaryForTree },
): LightTreeBuildInput {
  const packed = precomputed?.packed ?? packEmitterArrays(scene);
  // N-directional expansion: build one tree leaf PER packed directional, matching
  // the kernel loop and its unconditional `current` increment. Zero-radiance
  // records remain zero-power leaves: dropping one would densely renumber every
  // later leaf while the kernel still addresses the original packed slot, causing
  // a tree sample to shade the wrong emitter. Directional leaves are inserted at
  // the front in packed order, exactly mirroring the kernel's slot walk.
  //
  // NOTE: directionals deliberately stay OUTSIDE the light tree's spatial structure
  // (they are given the union-AABB of positional lights so the distance term floors
  // out and they compete by power alone — the original V22 rationale). Keeping them
  // outside a separate "directional group" is sound: an infinitely-distant
  // directional has no meaningful proximity, so a power-only selection is correct
  // and unbiased (the 1/p_select compensates). This matches the existing single-
  // directional behaviour and extends cleanly to N.
  const directionalLeaves: { power: number; dir: readonly [number, number, number] }[] = [];
  for (let di = 0; di < packed.directionalLightCount; di++) {
    const base = di * DIRECTIONAL_LIGHT_FLOAT_STRIDE;
    const irrR = packed.directionalLightsData[base + 4] ?? 0;
    const irrG = packed.directionalLightsData[base + 5] ?? 0;
    const irrB = packed.directionalLightsData[base + 6] ?? 0;
    directionalLeaves.push({
      power: emitterPower([irrR, irrG, irrB], { kind: 'delta' }),
      dir: [irrR, irrG, irrB], // kept for potential future per-directional diagnostics
    });
  }

  // The packed array is the single authoritative directional-light gate.
  const hasDirectional = directionalLeaves.length > 0;

  // Mirror the kernel's environment NEE gate exactly: the environment leaf is
  // present only when a valid baked HDRI/procedural-sky map is available.
  // When the caller already has an EnvSummaryForTree (from a prior environmentParams
  // call for the same scene), use it directly to avoid re-running the HDRI/sky bake.
  const envSummary: EnvSummaryForTree = precomputed?.envSummary ?? (() => {
    const p = environmentParams(scene);
    return {
      hasHdri: p.hasHdri,
      lightTreePower: p.lightTreePower,
    };
  })();
  const hasEnv = envSummary.hasHdri;

  const powers: number[] = [];
  const centroids: Vec3[] = [];
  const aabbs: { min: Vec3; max: Vec3 }[] = [];
  // B8 — per-emitter orientation cones, in lockstep with powers/centroids/aabbs.
  // `undefined` ⇒ full-sphere (no orientation culling). Point lights + the env /
  // directional slots stay undefined (isotropic / no spatial orientation);
  // spotlights and single-sided area lights get an oriented cone.
  type ConeEntry = { axis: readonly [number, number, number]; thetaO?: number; thetaE?: number } | undefined;
  const cones: ConeEntry[] = [];
  const HEMISPHERE = Math.PI / 2; // one-sided cosine emission lobe (area lights)
  const norm3 = (v: Vec3): readonly [number, number, number] => {
    const l = Math.hypot(v[0], v[1], v[2]);
    return l > 1e-12 ? [v[0] / l, v[1] / l, v[2] / l] : [0, 0, 0];
  };
  const cross3 = (a: Vec3, b: Vec3): Vec3 => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];

  // Track the union AABB of positional lights for the non-positional slots.
  let uMinX = Infinity, uMinY = Infinity, uMinZ = Infinity;
  let uMaxX = -Infinity, uMaxY = -Infinity, uMaxZ = -Infinity;
  const extend = (p: Vec3): void => {
    uMinX = Math.min(uMinX, p[0]); uMinY = Math.min(uMinY, p[1]); uMinZ = Math.min(uMinZ, p[2]);
    uMaxX = Math.max(uMaxX, p[0]); uMaxY = Math.max(uMaxY, p[1]); uMaxZ = Math.max(uMaxZ, p[2]);
  };

  // Deferred non-positional pushes (need the union AABB computed first). We record
  // their target index so the leaf order matches the kernel walk exactly.
  // For N directionals the deferred power array matches directionalLeaves in order.
  const directionalPowers = directionalLeaves.map((l) => l.power);

  // Positional selectable lights, in the EXACT walk order shared with the BDPT
  // emitter-pick oracle (point[8] → spot[12] → rect[16] → mesh[16]). The stride
  // arithmetic is single-sourced in `walkPositionalEmitters`; the per-kind power /
  // centroid / AABB derivation stays here because it is light-tree-specific.
  for (const e of walkPositionalEmitters(packed)) {
    switch (e.kind) {
      case 'point': {
        // Point: isotropic delta light at `position`, AABB collapses to the point.
        const p = e.position;
        extend(p);
        powers.push(emitterPower(e.radiance, { kind: 'delta' }));
        centroids.push(p);
        aabbs.push(pointAabb(p));
        cones.push(undefined); // isotropic — full sphere
        break;
      }
      case 'spot': {
        // Spot: delta light at `position` with a forward beam along `axis`. B8 —
        // orient the cone along the beam so the light tree culls points behind it.
        const p = e.position;
        extend(p);
        powers.push(emitterPower(e.radiance, { kind: 'delta' }));
        centroids.push(p);
        aabbs.push(pointAabb(p));
        cones.push({
          axis: norm3(e.axis),
          thetaO: 0,
          thetaE: Math.acos(Math.max(-1, Math.min(1, e.cosOuter))),
        });
        break;
      }
      case 'rect': {
        // rect-area / disc-area records (both packed in the rect stream).
        // Disc records carry shape tag = 1.0 in shapeTag; rect records = 0.0.
        // Area formula: disc → π·|u×v|; rect → 4·|u×v|.
        // AABB: disc → sphere-bounding-box with radius = |u| (conservative);
        //        rect → four corners p ± u ± v.
        const p = e.position;
        const u = e.uAxis;
        const v = e.vAxis;
        const isDisc = Math.abs((e.shapeTag ?? 0) - 1.0) < 0.5;
        const area = isDisc ? discArea(u, v) : rectQuadArea(u, v);
        let min: Vec3;
        let max: Vec3;
        if (isDisc) {
          // Bounding box for a disc: center ± radius along each axis.
          const r = Math.hypot(u[0], u[1], u[2]);
          min = [p[0] - r, p[1] - r, p[2] - r];
          max = [p[0] + r, p[1] + r, p[2] + r];
        } else {
          const cx = [Math.abs(u[0]) + Math.abs(v[0]), Math.abs(u[1]) + Math.abs(v[1]), Math.abs(u[2]) + Math.abs(v[2])] as const;
          min = [p[0] - cx[0], p[1] - cx[1], p[2] - cx[2]];
          max = [p[0] + cx[0], p[1] + cx[1], p[2] + cx[2]];
        }
        extend(min); extend(max);
        powers.push(emitterPower(e.radiance, { kind: 'area', area }));
        centroids.push(p);
        aabbs.push({ min, max });
        // B8 — single-sided cosine lobe along the emitter normal (u × v for
        // rect; same formula works for disc since u and v are orthonormal).
        cones.push({ axis: norm3(cross3(u, v)), thetaO: 0, thetaE: HEMISPHERE });
        break;
      }
      case 'mesh': {
        // mesh-area triangle records — triangle area = 0.5·|(B−A)×(C−A)|.
        const a = e.triA;
        const b = e.triB;
        const c = e.triC;
        const area = meshTriangleArea(a, b, c);
        const centroid: Vec3 = [(a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3, (a[2] + b[2] + c[2]) / 3];
        const min: Vec3 = [Math.min(a[0], b[0], c[0]), Math.min(a[1], b[1], c[1]), Math.min(a[2], b[2], c[2])];
        const max: Vec3 = [Math.max(a[0], b[0], c[0]), Math.max(a[1], b[1], c[1]), Math.max(a[2], b[2], c[2])];
        extend(min); extend(max);
        powers.push(emitterPower(e.radiance, { kind: 'area', area }));
        centroids.push(centroid);
        aabbs.push({ min, max });
        // B8 — mesh-area triangle emits from one side along (B−A)×(C−A): the WGSL
        // mesh-area NEE is one-sided (cosLight = max(dot(lightNormal, -wi), 0) >
        // 0 gate, kernel.wgsl), so a hemisphere lobe along the geometric normal
        // exactly matches the lit region — culling the dark back side is a pure
        // unbiased win.
        cones.push({ axis: norm3(cross3([b[0] - a[0], b[1] - a[1], b[2] - a[2]], [c[0] - a[0], c[1] - a[1], c[2] - a[2]])), thetaO: 0, thetaE: HEMISPHERE });
        break;
      }
    }
  }

  // Union AABB of positional lights (origin if there were none).
  const unionMin: Vec3 = Number.isFinite(uMinX) ? [uMinX, uMinY, uMinZ] : [0, 0, 0];
  const unionMax: Vec3 = Number.isFinite(uMaxX) ? [uMaxX, uMaxY, uMaxZ] : [0, 0, 0];
  const unionCentroid: Vec3 = [
    (unionMin[0] + unionMax[0]) / 2,
    (unionMin[1] + unionMax[1]) / 2,
    (unionMin[2] + unionMax[2]) / 2,
  ];

  // The directional + env leaves must occupy the SAME positions the kernel walk
  // assigns them: directionals are indices 0..N-1 (before point lights), env is LAST.
  // We built the positional leaves in [point, spot, rect, mesh] order above; now
  // splice the non-positional slots into the correct ends.
  //
  // For N directionals we unshift all N leaves in REVERSE order (unshift prepends
  // one-at-a-time, so the last unshift ends up at index 0 — we reverse so directional[0]
  // lands at index 0, directional[1] at index 1, etc.).
  if (hasDirectional) {
    for (let di = directionalPowers.length - 1; di >= 0; di--) {
      powers.unshift(directionalPowers[di]!);
      centroids.unshift(unionCentroid);
      aabbs.unshift({ min: unionMin, max: unionMax });
      // B8 — directional slots have no spatial position (infinitely far);
      // full-sphere (undefined) — an orientation cone would only mis-cull.
      cones.unshift(undefined);
    }
  }
  if (hasEnv) {
    // environmentParams integrates the actual map luminance over solid angle
    // and applies HDRI intensity exactly once. Nonblack underflow is clamped
    // there only to the smallest positive f32; black / zero-intensity maps stay
    // exactly zero rather than receiving an artificial unit floor.
    if (
      !Number.isFinite(envSummary.lightTreePower) ||
      envSummary.lightTreePower < 0 ||
      !Number.isFinite(Math.fround(envSummary.lightTreePower)) ||
      (envSummary.lightTreePower > 0 &&
        Math.fround(envSummary.lightTreePower) === 0)
    ) {
      throw new RangeError(
        'buildLightTreeInputForScene: envSummary.lightTreePower must be a ' +
          'finite, non-negative, Float32-representable value derived from environmentParams.',
      );
    }
    powers.push(envSummary.lightTreePower);
    centroids.push(unionCentroid);
    aabbs.push({ min: unionMin, max: unionMax });
    cones.push(undefined); // env dome — full sphere
  }

  return { powers, centroids, aabbs, cones };
}
