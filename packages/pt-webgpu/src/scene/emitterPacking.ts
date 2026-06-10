import type { DiscAreaEmitter, Mat4, MeshAreaEmitter, Scene } from '@vitrum/core';
import { luminance, type LightTreeBuildInput } from '@vitrum/shared-samplers';
import { transformPoint } from '../math/mat4.js';
import { environmentParams } from './environmentPacking.js';
import {
  meshTriangleArea,
  rectQuadArea,
  walkPositionalEmitters,
} from '../bdpt/flatEmitterWalk.js';

/**
 * Point light layout (3 vec4 = 12 floats, H51-D):
 *   vec4 0: position.xyz, 0
 *   vec4 1: radiance.rgb, 0
 *   vec4 2: distance, decay, 0, 0
 *   distance = 0 ⟹ no cutoff (infinite range); decay = 0 ⟹ no falloff.
 */
export const POINT_LIGHT_FLOAT_STRIDE = 12;

/**
 * Spot light layout (4 vec4 = 16 floats, H51-D):
 *   vec4 0: position.xyz, 0
 *   vec4 1: direction.xyz, cos(outerAngle)
 *   vec4 2: radiance.rgb, cos(innerAngle)   — innerAngle = outerAngle·(1−penumbra)
 *   vec4 3: distance, decay, 0, 0
 *   distance = 0 ⟹ no cutoff; decay = 0 ⟹ no falloff.
 */
export const SPOT_LIGHT_FLOAT_STRIDE = 16;
export const RECT_AREA_LIGHT_FLOAT_STRIDE = 16;
export const MESH_AREA_LIGHT_FLOAT_STRIDE = 16;

/**
 * Disc emitters lower into mesh-area triangle records so the existing WGSL
 * triangle sampler is used instead of pretending a disc is a rectangle. The
 * regular fan radius is scaled so its total triangle area equals pi*r^2.
 */
const DISC_AREA_TRIANGLE_SEGMENTS = 32;

type Vec3 = [number, number, number];

type PackedMeshAreaTriangle = {
  readonly triA: Vec3;
  readonly triB: Vec3;
  readonly triC: Vec3;
  readonly radiance: Vec3;
};

export interface PackedEmitterArrays {
  readonly warnings: string[];
  readonly pointLightCount: number;
  readonly spotLightCount: number;
  readonly rectAreaLightCount: number;
  readonly meshAreaLightCount: number;
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

function discAreaPackedAsTriangles(
  e: DiscAreaEmitter,
  warnings: string[],
): readonly PackedMeshAreaTriangle[] {
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
  const ux = nx / nLen;
  const uy = ny / nLen;
  const uz = nz / nLen;
  let ax = 0;
  let ay = 1;
  let az = 0;
  if (Math.abs(uy) > 0.999) {
    ax = 1;
    ay = 0;
    az = 0;
  }
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
  const bx = uy * tcz - uz * tcy;
  const by = uz * tcx - ux * tcz;
  const bz = ux * tcy - uy * tcx;
  const center: Vec3 = [e.position[0], e.position[1], e.position[2]];
  const polygonAreaFactor = 0.5
    * DISC_AREA_TRIANGLE_SEGMENTS
    * Math.sin((2 * Math.PI) / DISC_AREA_TRIANGLE_SEGMENTS);
  const fanRadius = e.radius * Math.sqrt(Math.PI / polygonAreaFactor);
  const radiance = emitterRadiance(e);
  const triangles: PackedMeshAreaTriangle[] = [];
  for (let i = 0; i < DISC_AREA_TRIANGLE_SEGMENTS; i += 1) {
    const a0 = (2 * Math.PI * i) / DISC_AREA_TRIANGLE_SEGMENTS;
    const a1 = (2 * Math.PI * (i + 1)) / DISC_AREA_TRIANGLE_SEGMENTS;
    const p0: Vec3 = [
      center[0] + (tcx * Math.cos(a0) + bx * Math.sin(a0)) * fanRadius,
      center[1] + (tcy * Math.cos(a0) + by * Math.sin(a0)) * fanRadius,
      center[2] + (tcz * Math.cos(a0) + bz * Math.sin(a0)) * fanRadius,
    ];
    const p1: Vec3 = [
      center[0] + (tcx * Math.cos(a1) + bx * Math.sin(a1)) * fanRadius,
      center[1] + (tcy * Math.cos(a1) + by * Math.sin(a1)) * fanRadius,
      center[2] + (tcz * Math.cos(a1) + bz * Math.sin(a1)) * fanRadius,
    ];
    triangles.push({ triA: center, triB: p0, triC: p1, radiance });
  }
  return triangles;
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
  const radiance = emitterRadiance(emitter);
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
      packed.push({ triA: a, triB: b, triC: c, radiance });
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

export function packEmitterArrays(scene: Scene): PackedEmitterArrays {
  const warnings: string[] = [];
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
    pointLights.push(ptDist, ptDecay, 0, 0);
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
    spotLights.push(spDist, spDecay, 0, 0);
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
  for (const e of scene.emitters) {
    if (e.kind !== 'rect-area') continue;
    pushVec4(rectAreaLights, [e.position[0], e.position[1], e.position[2]]);
    pushVec4(rectAreaLights, [e.uAxis[0], e.uAxis[1], e.uAxis[2]]);
    pushVec4(rectAreaLights, [e.vAxis[0], e.vAxis[1], e.vAxis[2]]);
    pushVec4(rectAreaLights, [
      e.color[0] * e.intensity,
      e.color[1] * e.intensity,
      e.color[2] * e.intensity,
    ]);
    rectAreaLightCount += 1;
  }
  const rectAreaLightsData = packedFloatData(
    rectAreaLights,
    rectAreaLightCount,
    RECT_AREA_LIGHT_FLOAT_STRIDE,
    'rect-area-light',
  );

  const meshAreaTriangles: PackedMeshAreaTriangle[] = [];
  // Disc-area emitters lower into the mesh-area triangle section. Keeping this
  // loop before mesh-area preserves the old flat walk's type block order:
  // rect/disc area came before mesh area.
  for (const emitter of scene.emitters) {
    if (emitter.kind !== 'disc-area') continue;
    meshAreaTriangles.push(...discAreaPackedAsTriangles(emitter, warnings));
  }
  for (const emitter of scene.emitters) {
    if (emitter.kind !== 'mesh-area') continue;
    meshAreaTriangles.push(...packMeshAreaTriangles(emitter, scene, warnings));
  }

  // H14-A: synthesize an implicit mesh-area emitter for every mesh-like primitive
  // whose material has non-zero emissive energy (luminance > 0) AND that has NO
  // explicit `mesh-area` emitter already pointing to it. Without this synthesis,
  // the emissive primitive is invisible to NEE/BDPT because the light loops never
  // enumerate it — the kernel's emissive-on-hit term fires, but there's no sampled
  // contribution from direct-lighting estimators.
  //
  // Guard: explicit emitters take priority — if a `mesh-area` emitter already
  // references this primitive we skip it (no double-counting). Disc-area emitters
  // that were lowered above are also excluded.
  const explicitMeshAreaIds = new Set<string>(
    scene.emitters
      .filter((e) => e.kind === 'mesh-area' || e.kind === 'disc-area')
      .map((e) => (e as { meshId?: string }).meshId ?? '')
      .filter(Boolean),
  );
  for (const primitive of scene.primitives) {
    if (primitive.kind === 'analytic') continue;
    if (explicitMeshAreaIds.has(primitive.id)) continue;
    const em = primitive.material.emissive ?? [0, 0, 0];
    const ei = primitive.material.emissiveIntensity ?? 1;
    const emR = em[0] * ei;
    const emG = em[1] * ei;
    const emB = em[2] * ei;
    if (luminance(emR, emG, emB) < 1e-6) continue;
    // Synthesize a virtual mesh-area emitter carrying the material's emissive radiance.
    const syntheticEmitter: Extract<Scene['emitters'][number], { kind: 'mesh-area' }> = {
      kind: 'mesh-area',
      id: `__implicit__${primitive.id}`,
      meshId: primitive.id,
      color: [emR, emG, emB],
      intensity: 1,
    };
    meshAreaTriangles.push(...packMeshAreaTriangles(syntheticEmitter, scene, warnings));
  }

  const meshAreaLights: number[] = [];
  for (const tri of meshAreaTriangles) {
    pushVec4(meshAreaLights, tri.triA);
    pushVec4(meshAreaLights, tri.triB);
    pushVec4(meshAreaLights, tri.triC);
    pushVec4(meshAreaLights, tri.radiance);
  }
  const meshAreaLightCount = meshAreaTriangles.length;
  const meshAreaLightsData = packedFloatData(
    meshAreaLights,
    meshAreaLightCount,
    MESH_AREA_LIGHT_FLOAT_STRIDE,
    'mesh-area-light',
  );

  return {
    warnings,
    pointLightCount,
    spotLightCount,
    rectAreaLightCount,
    meshAreaLightCount,
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
 * emitter whose `meshId` matches `primitiveId`.
 *
 * Used by the geometry/transform fast paths in `SceneMutationRouter` to detect
 * when moving a primitive's vertices/transform also shifts a mesh-area emitter's
 * world-space triangles — in which case `packEmitterArrays` must be re-run and
 * re-uploaded so the GPU NEE data stays in sync.
 *
 * Disc-area emitters are lowered into the mesh-triangle section at pack time, so
 * they are treated as mesh-area emitters for the staleness check.
 */
export function hasMeshAreaEmitterForPrimitive(scene: Scene, primitiveId: string): boolean {
  for (const e of scene.emitters) {
    if (e.kind !== 'mesh-area' && e.kind !== 'disc-area') continue;
    // Both kinds carry a `meshId` on the SceneEmitter type.
    const meshId = (e as { meshId?: string }).meshId;
    if (meshId === primitiveId) return true;
  }
  return false;
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
 * Build the `@vitrum/shared-samplers` light-tree input over the SELECTABLE lights
 * of `scene`, in the EXACT order pt-webgpu's NEE walk iterates them so the tree's
 * `emitterIndex` aligns 1:1 with the kernel's linear `current` index:
 *
 *   directional? · point[] · spot[] · rect-area[] · mesh-triangle-area[] · env?
 *
 * `disc-area` emitters are lowered by `packEmitterArrays` into the mesh-triangle
 * section as an equal-area fan, so every tree leaf still matches one GPU slot.
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
 */
export function buildLightTreeInputForScene(scene: Scene): LightTreeBuildInput {
  const packed = packEmitterArrays(scene);
  const dirIrr = defaultDirectionalIrradiance(scene);
  // Mirror the kernel's directional NEE gate EXACTLY: the kernel iterates the
  // directional slot iff `params.lightDir.w > 1e-6`, where `lightDir.w` is the
  // mean of `directionalIrradiance` = `defaultDirectionalIrradiance(scene)`. With
  // no directional emitter that default is [0,0,0] (V22 fix), so the mean is 0,
  // the kernel skips the directional slot, AND we omit its tree leaf — keeping
  // the tree's leaf order in lockstep with the kernel walk. Deriving
  // `hasDirectional` from the SAME `dirIrr` the kernel sees (rather than from a
  // separate "directional emitter exists?" check) is what guarantees the two
  // never disagree on whether the directional leaf is present. (V22.)
  const hasDirectional = (dirIrr[0] + dirIrr[1] + dirIrr[2]) / 3 > 1e-6;

  // Mirror the kernel's env NEE gate EXACTLY: `hasEnvironmentMap || sunStrength
  // > 1e-6`, both derived from the SAME `environmentParams` the GPU uploads.
  const envParams = environmentParams(scene);
  const hasEnv = envParams.hasHdri || envParams.sunStrength > 1e-6;

  const powers: number[] = [];
  const centroids: Vec3[] = [];
  const aabbs: { min: Vec3; max: Vec3 }[] = [];

  // Track the union AABB of positional lights for the non-positional slots.
  let uMinX = Infinity, uMinY = Infinity, uMinZ = Infinity;
  let uMaxX = -Infinity, uMaxY = -Infinity, uMaxZ = -Infinity;
  const extend = (p: Vec3): void => {
    uMinX = Math.min(uMinX, p[0]); uMinY = Math.min(uMinY, p[1]); uMinZ = Math.min(uMinZ, p[2]);
    uMaxX = Math.max(uMaxX, p[0]); uMaxY = Math.max(uMaxY, p[1]); uMaxZ = Math.max(uMaxZ, p[2]);
  };

  // Deferred non-positional pushes (need the union AABB computed first). We record
  // their target index so the leaf order matches the kernel walk exactly.
  let directionalPower = 0;
  if (hasDirectional) {
    directionalPower = emitterPower(dirIrr, { kind: 'delta' });
  }

  // Positional selectable lights, in the EXACT walk order shared with the BDPT
  // emitter-pick oracle (point[8] → spot[12] → rect[16] → mesh[16]). The stride
  // arithmetic is single-sourced in `walkPositionalEmitters`; the per-kind power /
  // centroid / AABB derivation stays here because it is light-tree-specific.
  for (const e of walkPositionalEmitters(packed)) {
    switch (e.kind) {
      case 'point':
      case 'spot': {
        // Point + spot: delta lights at `position`, AABB collapses to the point.
        const p = e.position;
        extend(p);
        powers.push(emitterPower(e.radiance, { kind: 'delta' }));
        centroids.push(p);
        aabbs.push(pointAabb(p));
        break;
      }
      case 'rect': {
        // rect-area — quad area = 4·|u×v| (matches the WGSL area-light NEE
        // term). AABB = the four corners p ± u ± v.
        const p = e.position;
        const u = e.uAxis;
        const v = e.vAxis;
        const area = rectQuadArea(u, v);
        const cx = [Math.abs(u[0]) + Math.abs(v[0]), Math.abs(u[1]) + Math.abs(v[1]), Math.abs(u[2]) + Math.abs(v[2])] as const;
        const min: Vec3 = [p[0] - cx[0], p[1] - cx[1], p[2] - cx[2]];
        const max: Vec3 = [p[0] + cx[0], p[1] + cx[1], p[2] + cx[2]];
        extend(min); extend(max);
        powers.push(emitterPower(e.radiance, { kind: 'area', area }));
        centroids.push(p);
        aabbs.push({ min, max });
        break;
      }
      case 'mesh': {
        // mesh-area and lowered disc-area records — triangle area =
        // 0.5·|(B−A)×(C−A)| (matches the WGSL term).
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
  // assigns them: directional is index 0 (before point lights), env is LAST. We
  // built the positional leaves in [point, spot, rect, mesh] order above; now
  // splice the non-positional slots into the correct ends.
  if (hasDirectional) {
    powers.unshift(directionalPower);
    centroids.unshift(unionCentroid);
    aabbs.unshift({ min: unionMin, max: unionMax });
  }
  if (hasEnv) {
    // Env radiance proxy: the dome tint scaled by the dome brightness (sun
    // strength), no per-pixel HDRI integral on the CPU. The luminance of that
    // proxy gives a stable RELATIVE weight against the local lights. A floor of
    // 1 on the strength keeps the env slot selectable (its leaf pdf must be > 0)
    // even for a faint sky.
    const strength = Math.max(envParams.sunStrength, 1);
    const t = envParams.tint;
    const envRad: Vec3 = [t[0] * strength, t[1] * strength, t[2] * strength];
    powers.push(emitterPower(envRad, { kind: 'delta' }));
    centroids.push(unionCentroid);
    aabbs.push({ min: unionMin, max: unionMax });
  }

  return { powers, centroids, aabbs };
}
