import type { DiscAreaEmitter, Mat4, MaterialSpec, MeshAreaEmitter, Scene, TextureRef } from '@vitrum/core';
import { estimateMaterialSpecEmissiveLeOverTriangle } from '@vitrum/shared-bvh';
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
 *   mean_irradiance = (r+g+b)/3 — cached for the kernel gate (lightDir.w analog).
 *   SHADOW-01: emitter castShadow:false is SIGN-ENCODED into the angularDiameter
 *   lane (both vec4s are otherwise full): packed = -1 - angularDiameter when the
 *   flag is false; the kernel decodes shadowDisabled = (raw < 0) and
 *   angularDiameter = -1 - raw. castShadow:true packs the raw value (≥ 0),
 *   byte-identical to the pre-SHADOW-01 layout.
 *
 * Directional[0] is ALSO mirrored into the frame-UBO lightDir/cameraPos.w lanes for
 * backward compatibility with lite-tier single-directional direct lighting.
 * The cameraPos.w mirror uses the SAME sign-encoded angularDiameter convention
 * as the storage-buffer lane so lite can honor castShadow:false without adding
 * a storage-buffer binding. For N > 1, lights [1..N-1] are read ONLY from this
 * storage buffer; the full kernel loops params.directionalLightCount records.
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
export const MESH_AREA_LIGHT_FLOAT_STRIDE = 16;

/**
 * Rect/disc area light record layout (4 vec4 = 16 floats):
 *   vec4 0: center.xyz, castShadowDisabled (SHADOW-01: 1.0 ⟺ castShadow:false; 0.0 default)
 *   vec4 1: uAxis.xyz, 0          — for rect: half-extent along u; for disc: tangent × radius
 *   vec4 2: vAxis.xyz, 0          — for rect: half-extent along v; for disc: bitangent × radius
 *   vec4 3: radiance.rgb, shape   — shape = 0.0 (rect) | 1.0 (disc, analytic concentric-map)
 *
 * Rect sampling: uniform in [-1,1]² → p = center + uAxis*u + vAxis*v; area = 4·|u×v|
 * Disc sampling: concentric-map (xi₁,xi₂)→(r,φ) → p = center + uAxis*(r·cos φ) + vAxis*(r·sin φ)
 *   area = π·|uAxis|² (= π·r² when |uAxis|=|vAxis|=radius; enforced by packDiscAsRect).
 *   pdf = dist² / (cosLight · π·r²)  — identical solid-angle formula as rect but with disc area.
 * Forward-hit MIS (connect.wgsl): ray-plane + |rel_u|²+|rel_v|²≤1 circle test for disc;
 *   uCoord/vCoord ∈ [-1,1] box test for rect.  |shape.w - 1.0| < 0.5 discriminates disc.
 */
const RECT_DISC_SHAPE_RECT = 0.0;
const RECT_DISC_SHAPE_DISC = 1.0;

type Vec3 = [number, number, number];

interface RawTexturePayload {
  readonly width: number;
  readonly height: number;
  readonly data: ArrayBufferView;
}

type PackedMeshAreaTriangle = {
  readonly triA: Vec3;
  readonly triB: Vec3;
  readonly triC: Vec3;
  readonly radiance: Vec3;
  /** SHADOW-01 — true ⟺ source mesh-area emitter set castShadow:false.
   *  Packed as 1.0 into the radiance vec4's .w lane (0.0 default). */
  readonly castShadowDisabled: boolean;
};

/**
 * Mesh-area NEE triangle cap — prevents a large emissive mesh from producing an
 * unbounded GPU buffer (16 floats × 4 bytes per triangle) and an oversized CPU
 * light-tree build (O(N log N)).
 *
 * Cap = 65 536 triangles ≈ 4 MB buffer + a still-manageable light tree.
 *
 * Rationale:
 *   - A 1M-triangle emissive mesh = 64 MB buffer + slow tree build per setScene.
 *   - Dropped triangles still emit via the BSDF/forward path (energy not lost,
 *     only NEE efficiency for the dropped fraction).
 *   - Selection: LARGEST-AREA-FIRST — drops the lowest-contribution triangles,
 *     biasing NEE variance minimally (small triangles contribute little power).
 *     Energy-proportional is equivalent but requires per-emitter irradiance
 *     sorting; area is the simpler and correct proxy for same-radiance emitters.
 *   - Warn ONCE per emitter that exceeds the cap.
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

function uvAt(uvs: Float32Array | undefined, vertex: number): [number, number] {
  if (uvs == null) return [0, 0];
  return [uvs[vertex * 2] ?? 0, uvs[vertex * 2 + 1] ?? 0];
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
      `@vitrum/pt-webgpu: primitive "${primitiveId}" has an emissiveMap without CPU-readable texels; ` +
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
 *   shape ≈ 1 → disc sampling (concentric-map, area = π|u|²)
 *
 * Native analytic disc emitters replace the 32-triangle fan, 2026-06-10 —
 * RENDER-CHANGING for disc-lit scenes, A/B in R9-B.
 */
function packDiscAsRect(
  e: DiscAreaEmitter,
  warnings: string[],
): readonly number[] {
  if (!Number.isFinite(e.radius) || e.radius < 1e-8) {
    warnings.push(
      `@vitrum/pt-webgpu: disc-area emitter "${e.id}" has near-zero radius; skipped.`,
    );
    return [];
  }
  const nx = e.normal[0];
  const ny = e.normal[1];
  const nz = e.normal[2];
  const nLen = Math.hypot(nx, ny, nz);
  if (nLen < 1e-8) {
    warnings.push(
      `@vitrum/pt-webgpu: disc-area emitter "${e.id}" has degenerate normal; skipped.`,
    );
    return [];
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
  if (tLen < 1e-8) {
    warnings.push(
      `@vitrum/pt-webgpu: disc-area emitter "${e.id}" has degenerate tangent basis; skipped.`,
    );
    return [];
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
  const rad = emitterRadiance(e);
  return [
    // vec4 0: center (.w = SHADOW-01 castShadowDisabled, 0.0 default)
    e.position[0], e.position[1], e.position[2], e.castShadow === false ? 1 : 0,
    // vec4 1: uAxis = tangent × radius
    tcx * r, tcy * r, tcz * r, 0,
    // vec4 2: vAxis = bitangent × radius
    bx * r, by * r, bz * r, 0,
    // vec4 3: radiance.rgb, shape = DISC (1.0)
    rad[0], rad[1], rad[2], RECT_DISC_SHAPE_DISC,
  ];
}

export function defaultDirectionalLight(scene: Scene): readonly [number, number, number] {
  const directional = scene.emitters.find((e) => e.kind === 'directional');
  if (directional == null) return [0.4, 1.0, 0.2];
  const x = directional.direction[0];
  const y = directional.direction[1];
  const z = directional.direction[2];
  const len = Math.hypot(x, y, z);
  if (len < 1e-8) return [0.4, 1.0, 0.2];
  // Core contract: direction points AT the light, so incoming light is -direction.
  return [-x / len, -y / len, -z / len];
}

export function defaultDirectionalIrradiance(scene: Scene): readonly [number, number, number] {
  const directional = scene.emitters.find((e) => e.kind === 'directional');
  // No directional emitter ⇒ NO directional light. The former [1,1,1] default
  // fabricated a phantom directional in EVERY directional-less scene (the kernel
  // gates directional NEE on `lightDir.w = mean(thisIrradiance) > 1e-6`), which
  // is physically wrong AND skewed the power-weighted light tree: the phantom's
  // leaf was assigned the union-AABB of all positional lights, so its dist²≈0
  // inside the scene made its importance dominate the descent, starving the real
  // lights of selection probability and inflating their 1/pdf NEE weights (V22
  // showed the tree raising variance ~76% over a uniform pick). [0,0,0] removes
  // the phantom from both the kernel NEE and the tree. (V22, 2026-05-29.)
  if (directional == null) return [0, 0, 0];
  const scale = directional.intensity;
  return [
    directional.color[0] * scale,
    directional.color[1] * scale,
    directional.color[2] * scale,
  ];
}

/**
 * D3/SHADOW-01 — signed soft-sun angular diameter mirror for the scene's first
 * directional emitter.
 * 0 (the default, and when no directional emitter is present) = a perfect delta
 * directional, the historical exact path (byte-identical). A positive value turns
 * the directional NEE into a cone sampler over the sun's solid angle for soft
 * shadows. If the first directional has `castShadow:false`, the returned mirror
 * is sign-encoded as `-1 - angularDiameter`, matching the storage-buffer lane.
 * Carried in the frame UBO's `cameraPos.w` lane (a previously-unused `.w` slot
 * — see frameParamsPacker) so no UBO byte-size/layout change is needed.
 * Ref: DirectionalEmitter.angularDiameter (core contract).
 */
export function defaultDirectionalAngularDiameter(scene: Scene): number {
  const directional = scene.emitters.find((e) => e.kind === 'directional');
  if (directional == null) return 0;
  const ad = directional.angularDiameter;
  const angularDiameter = ad != null && Number.isFinite(ad) && ad > 0 ? ad : 0;
  return directional.castShadow === false ? -1 - angularDiameter : angularDiameter;
}

function packMeshAreaTriangles(
  emitter: MeshAreaEmitter,
  scene: Scene,
  warnings: string[],
): readonly PackedMeshAreaTriangle[] {
  const primitive = scene.primitives.find((p) => p.id === emitter.meshId);
  if (primitive == null || primitive.kind === 'analytic') {
    warnings.push(`Mesh-area emitter "${emitter.id}" references missing or non-mesh primitive "${emitter.meshId}".`);
    return [];
  }
  const positions = primitive.positions;
  const vertexCount = Math.floor(positions.length / 3);
  const triangleIndexCount = primitive.indices?.length ?? vertexCount;
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
  const indexAt = (offset: number): number => primitive.indices?.[offset] ?? offset;
  const transforms: readonly (Mat4 | undefined)[] = primitive.kind === 'instanced-mesh'
    ? primitive.instances
    : [primitive.transform];
  if (transforms.length === 0) {
    warnings.push(`Mesh-area emitter "${emitter.id}" references instanced primitive "${emitter.meshId}" with no instances.`);
    return [];
  }
  const implicitMaterial = emitter.id === `__implicit__${primitive.id}` ? primitive.material : undefined;
  const radiance = emitterRadiance(emitter);
  // SHADOW-01 — carry the emitter's castShadow flag onto every packed triangle.
  const castShadowDisabled = emitter.castShadow === false;
  const packed: PackedMeshAreaTriangle[] = [];
  let invalidTriangleCount = 0;
  let degenerateTriangleCount = 0;
  for (const transform of transforms) {
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
      if (meshTriangleArea(a, b, c) < 1e-12) {
        degenerateTriangleCount += 1;
        continue;
      }
      const triangleRadiance = implicitMaterial == null
        ? radiance
        : estimateMaterialSpecEmissiveLeOverTriangle(
            implicitMaterial,
            uvAt(primitive.uvs, i0),
            uvAt(primitive.uvs, i1),
            uvAt(primitive.uvs, i2),
            uvAt(primitive.uv1, i0),
            uvAt(primitive.uv1, i1),
            uvAt(primitive.uv1, i2),
          );
      if (triangleRadiance == null || luminance(triangleRadiance[0], triangleRadiance[1], triangleRadiance[2]) < IMPLICIT_EMITTER_LUMINANCE_THRESHOLD) {
        continue;
      }
      packed.push({ triA: a, triB: b, triC: c, radiance: triangleRadiance, castShadowDisabled });
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
 * Luminance threshold for implicit mesh-area emitter synthesis (H14-A).
 * A material must have `luminance(emissive · emissiveIntensity · avg(emissiveMap))`
 * ≥ IMPLICIT_EMITTER_THRESHOLD to be treated as an area light by NEE/BDPT. The same
 * helper is used by both `packEmitterArrays` (synthesis) and
 * `hasMeshAreaEmitterForPrimitive` (staleness check), so the threshold cannot drift
 * between the two paths.
 */
const IMPLICIT_EMITTER_LUMINANCE_THRESHOLD = 1e-6;

/**
 * H14-A — Synthesize implicit mesh-area emitters for every mesh-like primitive
 * whose material has non-zero emissive energy (luminance ≥
 * {@link IMPLICIT_EMITTER_LUMINANCE_THRESHOLD}) AND that has NO explicit
 * `mesh-area` emitter already referencing it.
 *
 * Without this synthesis, an emissive mesh is invisible to NEE/BDPT — the
 * kernel's emissive-on-hit term fires, but the direct-lighting estimators never
 * enumerate it. The synthetic emitters are virtual (id = `__implicit__<primitiveId>`)
 * and carry the material's average emissive radiance as `color · intensity = 1`.
 * CPU-readable emissive maps modulate the radiance by their average sRGB-decoded
 * linear RGB. Opaque/unreadable map handles keep the old scalar fallback and emit
 * a warning from `packEmitterArrays`.
 *
 * Guard: explicit mesh-area emitters take priority — if a `mesh-area` entry in
 * `scene.emitters` already references a primitive, that primitive is skipped to
 * prevent double-counting. Disc-area emitters are packed natively into the rect
 * stream and do NOT generate mesh-area triangles, so they are not excluded here.
 *
 * This function is the single source of truth for the implicit-emitter synthesis
 * logic and is reused by `hasMeshAreaEmitterForPrimitive` to avoid duplicating
 * the threshold check.
 */
function synthesizeImplicitEmitters(
  scene: Scene,
  /** When set, only consider this primitive id (early-exit fast path for the
   *  `hasMeshAreaEmitterForPrimitive` staleness predicate). */
  onlyPrimitiveId?: string,
  warnings?: string[],
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
    const [emR, emG, emB] = emissiveRadianceForMaterial(primitive.material, primitive.id, warnings);
    if (luminance(emR, emG, emB) < IMPLICIT_EMITTER_LUMINANCE_THRESHOLD) continue;
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

export function packEmitterArrays(scene: Scene): PackedEmitterArrays {
  const warnings: string[] = [];

  // N-directional packing — all directional emitters go into a flat storage-buffer
  // array. The first directional[0] is ALSO mirrored into the frame-UBO lightDir
  // and signed cameraPos.w lanes by frameParamsPacker.ts (backward-compat lite NEE).
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
    const discRecord = packDiscAsRect(e, warnings);
    if (discRecord.length === 0) continue;
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
    meshAreaTriangles.push(...packMeshAreaTriangles(emitter, scene, warnings));
  }

  // H14-A: synthesize implicit mesh-area emitters and pack their triangles.
  // See `synthesizeImplicitEmitters` for the synthesis contract.
  for (const synthetic of synthesizeImplicitEmitters(scene, undefined, warnings)) {
    meshAreaTriangles.push(...packMeshAreaTriangles(synthetic, scene, warnings));
  }

  // Mesh-area NEE cap: cap the total triangle count to MESH_AREA_LIGHT_TRI_CAP.
  // Strategy: LARGEST-AREA-FIRST (keeps the highest-contribution triangles for NEE;
  // dropped triangles still emit via the BSDF/forward path — energy is not lost,
  // only NEE efficiency for the dropped fraction).
  let cappedTriangles = meshAreaTriangles;
  if (meshAreaTriangles.length > MESH_AREA_LIGHT_TRI_CAP) {
    warnings.push(
      `@vitrum/pt-webgpu: mesh-area NEE triangle count (${meshAreaTriangles.length}) exceeds cap ` +
        `(${MESH_AREA_LIGHT_TRI_CAP}); keeping the ${MESH_AREA_LIGHT_TRI_CAP} largest-area triangles. ` +
        `Dropped triangles still emit via the BSDF/forward path (no energy loss, NEE-only efficiency reduction).`,
    );
    // Sort descending by triangle area; keep the first MESH_AREA_LIGHT_TRI_CAP.
    const withArea = meshAreaTriangles.map((tri) => ({
      tri,
      area: meshTriangleArea(tri.triA, tri.triB, tri.triC),
    }));
    withArea.sort((a, b) => b.area - a.area);
    cappedTriangles = withArea.slice(0, MESH_AREA_LIGHT_TRI_CAP).map((e) => e.tri);
  }

  const meshAreaLights: number[] = [];
  for (const tri of cappedTriangles) {
    pushVec4(meshAreaLights, tri.triA);
    pushVec4(meshAreaLights, tri.triB);
    pushVec4(meshAreaLights, tri.triC);
    // SHADOW-01 — radiance vec4 .w carries castShadowDisabled (0.0 default).
    pushVec4(meshAreaLights, tri.radiance, tri.castShadowDisabled ? 1 : 0);
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
  // `synthesizeImplicitEmitters` is the single source of the threshold check
  // (IMPLICIT_EMITTER_LUMINANCE_THRESHOLD), so the staleness predicate here
  // cannot drift from the synthesis path in `packEmitterArrays`.
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
  /** Procedural-sky / HDRI sun-strength scalar (drives the env NEE gate). */
  readonly sunStrength: number;
  /** Environment tint used as the env-leaf radiance proxy in the power estimate. */
  readonly tint: readonly [number, number, number];
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
 *   - `rebuildLightTreeForScene` (uploadSceneBuffers.ts): forwards its own optional
 *     `precomputed` parameter; callers supply it when available.
 *   - `sceneMutationRouter.ts` (via `rebuildLightTreeForScene`): passes `packed`
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
  // N-directional expansion: build one tree leaf PER directional emitter (matching
  // the kernel's loop `for (var di = 0u; di < params.directionalLightCount; di++)`).
  // Each leaf uses the per-directional irradiance as the power proxy; a directional
  // with mean_irradiance ≤ 1e-6 is silently skipped (matches the kernel's inner gate
  // `if (d_meanIrr > 1e-6)`). The leaves for directionals[0..N-1] are inserted at
  // the FRONT of the arrays (indices 0..N-1) in walk order, exactly mirroring the
  // kernel's `current` counter which starts at 0.
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
    const meanIrr = (irrR + irrG + irrB) / 3;
    if (meanIrr <= 1e-6) continue; // kernel gate: skip black/absent directionals
    directionalLeaves.push({
      power: emitterPower([irrR, irrG, irrB], { kind: 'delta' }),
      dir: [irrR, irrG, irrB], // kept for potential future per-directional diagnostics
    });
  }

  // For backward-compat gates in buildLightTreeInputForScene (lightDir.w was the
  // single gate); now we check the packed array directly.
  const hasDirectional = directionalLeaves.length > 0;

  // Mirror the kernel's env NEE gate EXACTLY: `hasEnvironmentMap || sunStrength
  // > 1e-6`, both derived from the SAME `environmentParams` the GPU uploads.
  // When the caller already has an EnvSummaryForTree (from a prior environmentParams
  // call for the same scene), use it directly to avoid re-running the HDRI/sky bake.
  const envSummary: EnvSummaryForTree = precomputed?.envSummary ?? (() => {
    const p = environmentParams(scene);
    return { hasHdri: p.hasHdri, sunStrength: p.sunStrength, tint: p.tint };
  })();
  const hasEnv = envSummary.hasHdri || envSummary.sunStrength > 1e-6;

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
  // Spotlights: a forward beam. We don't carry the spot half-angle in the walked
  // record, so use a conservative wide-ish lobe (≈60°) — it still culls the rear
  // hemisphere (the common, high-value case) without over-tightening selection.
  const SPOT_LOBE = Math.PI / 3;
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
        cones.push({ axis: norm3(e.axis), thetaO: 0, thetaE: SPOT_LOBE });
        break;
      }
      case 'rect': {
        // rect-area / disc-area records (both packed in the rect stream).
        // Disc records carry shape tag = 1.0 in shapeTag; rect records = 0.0.
        // Area formula: disc → π·|u|² (= π·r²); rect → 4·|u×v|.
        // AABB: disc → sphere-bounding-box with radius = |u| (conservative);
        //        rect → four corners p ± u ± v.
        const p = e.position;
        const u = e.uAxis;
        const v = e.vAxis;
        const isDisc = Math.abs((e.shapeTag ?? 0) - 1.0) < 0.5;
        const area = isDisc ? discArea(u) : rectQuadArea(u, v);
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
    // Env radiance proxy: the dome tint scaled by the dome brightness (sun
    // strength), no per-pixel HDRI integral on the CPU. The luminance of that
    // proxy gives a stable RELATIVE weight against the local lights. A floor of
    // 1 on the strength keeps the env slot selectable (its leaf pdf must be > 0)
    // even for a faint sky.
    const strength = Math.max(envSummary.sunStrength, 1);
    const t = envSummary.tint;
    const envRad: Vec3 = [t[0] * strength, t[1] * strength, t[2] * strength];
    powers.push(emitterPower(envRad, { kind: 'delta' }));
    centroids.push(unionCentroid);
    aabbs.push({ min: unionMin, max: unionMax });
    cones.push(undefined); // env dome — full sphere
  }

  return { powers, centroids, aabbs, cones };
}
