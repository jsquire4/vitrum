import type { DiscAreaEmitter, MeshAreaEmitter, Scene } from '@vitrum/core';
import { luminance, type LightTreeBuildInput } from '@vitrum/shared-samplers';
import { transformPoint } from '../math/mat4.js';
import { environmentParams } from './environmentPacking.js';

// Per-emitter capacity caps + float strides — file-local. No external
// consumers (2026-05-18 dead-code sweep verified workspace-wide). Re-exports
// previously surfaced from uploadSceneBuffers.ts were also dropped in the
// same sweep.
const MAX_POINT_LIGHTS = 16;
const MAX_SPOT_LIGHTS = 8;
const MAX_RECT_AREA_LIGHTS = 8;
const MAX_MESH_AREA_LIGHTS = 8;

/** vec4 pairs per point light: position, radiance */
const POINT_LIGHT_FLOAT_STRIDE = 8;
const SPOT_LIGHT_FLOAT_STRIDE = 12;
const RECT_AREA_LIGHT_FLOAT_STRIDE = 16;
const MESH_AREA_LIGHT_FLOAT_STRIDE = 16;

/** Map disc emitter to rect-axis payload — half-span √(π)/2·radius on each orthogonal tangent so WGSL quad area (=4|u×v|) equals π·r². Sampling differs from a true disc. */
function discAreaPackedAsRect(e: DiscAreaEmitter): {
  readonly position: readonly [number, number, number];
  readonly uAxis: readonly [number, number, number];
  readonly vAxis: readonly [number, number, number];
  readonly radiance: readonly [number, number, number];
} {
  const nx = e.normal[0];
  const ny = e.normal[1];
  const nz = e.normal[2];
  const nLen = Math.hypot(nx, ny, nz);
  if (nLen < 1e-8) {
    return {
      position: [e.position[0], e.position[1], e.position[2]],
      uAxis: [0, 0, 0],
      vAxis: [0, 0, 0],
      radiance: [0, 0, 0],
    };
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
    return {
      position: [e.position[0], e.position[1], e.position[2]],
      uAxis: [0, 0, 0],
      vAxis: [0, 0, 0],
      radiance: [0, 0, 0],
    };
  }
  const tcx = tx / tLen;
  const tcy = ty / tLen;
  const tcz = tz / tLen;
  const bx = uy * tcz - uz * tcy;
  const by = uz * tcx - ux * tcz;
  const bz = ux * tcy - uy * tcx;
  const s = (Math.sqrt(Math.PI) * e.radius) / 2;
  const uAxis = [tcx * s, tcy * s, tcz * s] as const;
  const vAxis = [bx * s, by * s, bz * s] as const;
  return {
    position: [e.position[0], e.position[1], e.position[2]],
    uAxis,
    vAxis,
    radiance: [
      e.color[0] * e.intensity,
      e.color[1] * e.intensity,
      e.color[2] * e.intensity,
    ],
  };
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
  if (directional == null) return [1, 1, 1];
  const scale = directional.intensity;
  return [
    directional.color[0] * scale,
    directional.color[1] * scale,
    directional.color[2] * scale,
  ];
}

/**
 * Pack a single mesh-area emitter's first triangle (positions in world space)
 * and per-emitter radiance. Returns `null` when the emitter's referenced
 * primitive is missing, not a mesh, or has fewer than one triangle — in those
 * cases a warning is emitted via the `warnings` accumulator.
 *
 * Extracted from the legacy `firstMeshAreaLight` helper so `packEmitterArrays`
 * can iterate emitters directly without faking a single-emitter scene.
 */
function packMeshAreaTriangle(
  emitter: MeshAreaEmitter,
  scene: Scene,
  warnings: string[],
): {
  readonly triA: readonly [number, number, number];
  readonly triB: readonly [number, number, number];
  readonly triC: readonly [number, number, number];
  readonly radiance: readonly [number, number, number];
} | null {
  const primitive = scene.primitives.find((p) => p.id === emitter.meshId);
  if (primitive == null || primitive.kind === 'analytic') {
    warnings.push(`Mesh-area emitter "${emitter.id}" references missing or non-mesh primitive "${emitter.meshId}".`);
    return null;
  }
  const positions = primitive.positions;
  const indices =
    primitive.indices ??
    (() => {
      const generated = new Uint32Array(positions.length / 3);
      for (let i = 0; i < generated.length; i += 1) generated[i] = i;
      return generated;
    })();
  if (indices.length < 3 || positions.length < 9) {
    warnings.push(`Mesh-area emitter "${emitter.id}" references primitive "${emitter.meshId}" with no triangles.`);
    return null;
  }
  const i0 = indices[0] ?? 0;
  const i1 = indices[1] ?? 0;
  const i2 = indices[2] ?? 0;
  const fetchPos = (idx: number): [number, number, number] => [
    positions[idx * 3] ?? 0,
    positions[idx * 3 + 1] ?? 0,
    positions[idx * 3 + 2] ?? 0,
  ];
  let a = fetchPos(i0);
  let b = fetchPos(i1);
  let c = fetchPos(i2);
  const transform = primitive.kind === 'instanced-mesh' ? primitive.instances[0] : primitive.transform;
  if (transform != null) {
    a = transformPoint(transform, a);
    b = transformPoint(transform, b);
    c = transformPoint(transform, c);
  }
  return {
    triA: a,
    triB: b,
    triC: c,
    radiance: [
      emitter.color[0] * emitter.intensity,
      emitter.color[1] * emitter.intensity,
      emitter.color[2] * emitter.intensity,
    ],
  };
}

export function packEmitterArrays(scene: Scene): {
  readonly warnings: string[];
  readonly pointLightCount: number;
  readonly spotLightCount: number;
  readonly rectAreaLightCount: number;
  readonly meshAreaLightCount: number;
  readonly pointLightsData: Float32Array;
  readonly spotLightsData: Float32Array;
  readonly rectAreaLightsData: Float32Array;
  readonly meshAreaLightsData: Float32Array;
} {
  const warnings: string[] = [];
  const pointLightsData = new Float32Array(MAX_POINT_LIGHTS * POINT_LIGHT_FLOAT_STRIDE).fill(0);
  let pointLightCount = 0;
  for (const e of scene.emitters) {
    if (e.kind !== 'point') continue;
    if (pointLightCount >= MAX_POINT_LIGHTS) {
      warnings.push(
        `@vitrum/pt-webgpu: point lights capped at ${MAX_POINT_LIGHTS}; emitter "${e.id}" and subsequent point emitters omitted.`,
      );
      break;
    }
    const o = pointLightCount * POINT_LIGHT_FLOAT_STRIDE;
    pointLightsData[o + 0] = e.position[0];
    pointLightsData[o + 1] = e.position[1];
    pointLightsData[o + 2] = e.position[2];
    pointLightsData[o + 4] = e.color[0] * e.intensity;
    pointLightsData[o + 5] = e.color[1] * e.intensity;
    pointLightsData[o + 6] = e.color[2] * e.intensity;
    pointLightCount += 1;
  }

  const spotLightsData = new Float32Array(MAX_SPOT_LIGHTS * SPOT_LIGHT_FLOAT_STRIDE).fill(0);
  let spotLightCount = 0;
  for (const e of scene.emitters) {
    if (e.kind !== 'spot') continue;
    if (spotLightCount >= MAX_SPOT_LIGHTS) {
      warnings.push(
        `@vitrum/pt-webgpu: spot lights capped at ${MAX_SPOT_LIGHTS}; emitter "${e.id}" and subsequent spot emitters omitted.`,
      );
      break;
    }
    const d = e.direction;
    const len = Math.hypot(d[0], d[1], d[2]);
    const dir: readonly [number, number, number] =
      len < 1e-8 ? [0, -1, 0] : [d[0] / len, d[1] / len, d[2] / len];
    const o = spotLightCount * SPOT_LIGHT_FLOAT_STRIDE;
    spotLightsData[o + 0] = e.position[0];
    spotLightsData[o + 1] = e.position[1];
    spotLightsData[o + 2] = e.position[2];
    spotLightsData[o + 4] = dir[0];
    spotLightsData[o + 5] = dir[1];
    spotLightsData[o + 6] = dir[2];
    spotLightsData[o + 7] = Math.cos(e.angle);
    spotLightsData[o + 8] = e.color[0] * e.intensity;
    spotLightsData[o + 9] = e.color[1] * e.intensity;
    spotLightsData[o + 10] = e.color[2] * e.intensity;
    spotLightCount += 1;
  }

  const rectAreaLightsData = new Float32Array(MAX_RECT_AREA_LIGHTS * RECT_AREA_LIGHT_FLOAT_STRIDE).fill(0);
  let rectAreaLightCount = 0;
  for (const e of scene.emitters) {
    if (e.kind !== 'rect-area' && e.kind !== 'disc-area') continue;

    let position: readonly [number, number, number];
    let uAxis: readonly [number, number, number];
    let vAxis: readonly [number, number, number];
    let rgb: readonly [number, number, number];

    if (e.kind === 'rect-area') {
      position = [e.position[0], e.position[1], e.position[2]];
      uAxis = [e.uAxis[0], e.uAxis[1], e.uAxis[2]];
      vAxis = [e.vAxis[0], e.vAxis[1], e.vAxis[2]];
      rgb = [
        e.color[0] * e.intensity,
        e.color[1] * e.intensity,
        e.color[2] * e.intensity,
      ];
    } else {
      if (Number.isFinite(e.radius) && e.radius < 1e-8) {
        warnings.push(
          `@vitrum/pt-webgpu: disc-area emitter "${e.id}" has near-zero radius; skipped.`,
        );
        continue;
      }
      if (Math.hypot(e.normal[0], e.normal[1], e.normal[2]) < 1e-8) {
        warnings.push(
          `@vitrum/pt-webgpu: disc-area emitter "${e.id}" has degenerate normal; skipped.`,
        );
        continue;
      }
      const d = discAreaPackedAsRect(e);
      position = d.position;
      uAxis = d.uAxis;
      vAxis = d.vAxis;
      rgb = d.radiance;
    }

    if (rectAreaLightCount >= MAX_RECT_AREA_LIGHTS) {
      warnings.push(
        `@vitrum/pt-webgpu: rect-/disc-area lights capped at ${MAX_RECT_AREA_LIGHTS}; emitter "${e.id}" and subsequent area emitters omitted.`,
      );
      break;
    }

    const o = rectAreaLightCount * RECT_AREA_LIGHT_FLOAT_STRIDE;
    rectAreaLightsData[o + 0] = position[0];
    rectAreaLightsData[o + 1] = position[1];
    rectAreaLightsData[o + 2] = position[2];
    rectAreaLightsData[o + 4] = uAxis[0];
    rectAreaLightsData[o + 5] = uAxis[1];
    rectAreaLightsData[o + 6] = uAxis[2];
    rectAreaLightsData[o + 8] = vAxis[0];
    rectAreaLightsData[o + 9] = vAxis[1];
    rectAreaLightsData[o + 10] = vAxis[2];
    rectAreaLightsData[o + 12] = rgb[0];
    rectAreaLightsData[o + 13] = rgb[1];
    rectAreaLightsData[o + 14] = rgb[2];
    rectAreaLightCount += 1;
  }

  const meshAreaLightsData = new Float32Array(MAX_MESH_AREA_LIGHTS * MESH_AREA_LIGHT_FLOAT_STRIDE).fill(0);
  let meshAreaLightCount = 0;
  for (const emitter of scene.emitters) {
    if (emitter.kind !== 'mesh-area') continue;
    if (meshAreaLightCount >= MAX_MESH_AREA_LIGHTS) {
      warnings.push(
        `@vitrum/pt-webgpu: mesh-area lights capped at ${MAX_MESH_AREA_LIGHTS}; emitter "${emitter.id}" and subsequent omitted.`,
      );
      break;
    }
    const packedOne = packMeshAreaTriangle(emitter, scene, warnings);
    if (packedOne == null) {
      continue;
    }
    const o = meshAreaLightCount * MESH_AREA_LIGHT_FLOAT_STRIDE;
    meshAreaLightsData[o + 0] = packedOne.triA[0];
    meshAreaLightsData[o + 1] = packedOne.triA[1];
    meshAreaLightsData[o + 2] = packedOne.triA[2];
    meshAreaLightsData[o + 4] = packedOne.triB[0];
    meshAreaLightsData[o + 5] = packedOne.triB[1];
    meshAreaLightsData[o + 6] = packedOne.triB[2];
    meshAreaLightsData[o + 8] = packedOne.triC[0];
    meshAreaLightsData[o + 9] = packedOne.triC[1];
    meshAreaLightsData[o + 10] = packedOne.triC[2];
    meshAreaLightsData[o + 12] = packedOne.radiance[0];
    meshAreaLightsData[o + 13] = packedOne.radiance[1];
    meshAreaLightsData[o + 14] = packedOne.radiance[2];
    meshAreaLightCount += 1;
  }

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

/** The positional area-emitter kinds (carry a finite area for the power term). */
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

type Vec3 = [number, number, number];

function pointAabb(p: Vec3): { min: Vec3; max: Vec3 } {
  return { min: [p[0], p[1], p[2]], max: [p[0], p[1], p[2]] };
}

/**
 * Build the `@vitrum/shared-samplers` light-tree input over the SELECTABLE lights
 * of `scene`, in the EXACT order pt-webgpu's NEE walk iterates them so the tree's
 * `emitterIndex` aligns 1:1 with the kernel's linear `current` index:
 *
 *   directional? · point[] · spot[] · (rect|disc)-area[] · mesh-area[] · env?
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
  const hasDirectional =
    scene.emitters.some((e) => e.kind === 'directional') &&
    (dirIrr[0] + dirIrr[1] + dirIrr[2]) / 3 > 1e-6;

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

  // 1. point lights — stride 8 (vec2: position, radiance).
  for (let i = 0; i < packed.pointLightCount; i++) {
    const o = i * 8;
    const p: Vec3 = [packed.pointLightsData[o]!, packed.pointLightsData[o + 1]!, packed.pointLightsData[o + 2]!];
    const rad: Vec3 = [packed.pointLightsData[o + 4]!, packed.pointLightsData[o + 5]!, packed.pointLightsData[o + 6]!];
    extend(p);
    powers.push(emitterPower(rad, { kind: 'delta' }));
    centroids.push(p);
    aabbs.push(pointAabb(p));
  }
  // 2. spot lights — stride 12 (position, dir+cos, radiance).
  for (let i = 0; i < packed.spotLightCount; i++) {
    const o = i * 12;
    const p: Vec3 = [packed.spotLightsData[o]!, packed.spotLightsData[o + 1]!, packed.spotLightsData[o + 2]!];
    const rad: Vec3 = [packed.spotLightsData[o + 8]!, packed.spotLightsData[o + 9]!, packed.spotLightsData[o + 10]!];
    extend(p);
    powers.push(emitterPower(rad, { kind: 'delta' }));
    centroids.push(p);
    aabbs.push(pointAabb(p));
  }
  // 3. rect/disc-area lights — stride 16 (position, uAxis, vAxis, radiance).
  //    Quad area = 4·|u×v| (matches the WGSL area-light NEE term).
  for (let i = 0; i < packed.rectAreaLightCount; i++) {
    const o = i * 16;
    const p: Vec3 = [packed.rectAreaLightsData[o]!, packed.rectAreaLightsData[o + 1]!, packed.rectAreaLightsData[o + 2]!];
    const u: Vec3 = [packed.rectAreaLightsData[o + 4]!, packed.rectAreaLightsData[o + 5]!, packed.rectAreaLightsData[o + 6]!];
    const v: Vec3 = [packed.rectAreaLightsData[o + 8]!, packed.rectAreaLightsData[o + 9]!, packed.rectAreaLightsData[o + 10]!];
    const rad: Vec3 = [packed.rectAreaLightsData[o + 12]!, packed.rectAreaLightsData[o + 13]!, packed.rectAreaLightsData[o + 14]!];
    const cross: Vec3 = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
    const area = 4 * Math.hypot(cross[0], cross[1], cross[2]);
    // AABB = the four corners p ± u ± v.
    const cx = [Math.abs(u[0]) + Math.abs(v[0]), Math.abs(u[1]) + Math.abs(v[1]), Math.abs(u[2]) + Math.abs(v[2])] as const;
    const min: Vec3 = [p[0] - cx[0], p[1] - cx[1], p[2] - cx[2]];
    const max: Vec3 = [p[0] + cx[0], p[1] + cx[1], p[2] + cx[2]];
    extend(min); extend(max);
    powers.push(emitterPower(rad, { kind: 'area', area }));
    centroids.push(p);
    aabbs.push({ min, max });
  }
  // 4. mesh-area lights — stride 16 (triA, triB, triC, radiance).
  //    Triangle area = 0.5·|（B−A)×(C−A)| (matches the WGSL mesh-area NEE term).
  for (let i = 0; i < packed.meshAreaLightCount; i++) {
    const o = i * 16;
    const a: Vec3 = [packed.meshAreaLightsData[o]!, packed.meshAreaLightsData[o + 1]!, packed.meshAreaLightsData[o + 2]!];
    const b: Vec3 = [packed.meshAreaLightsData[o + 4]!, packed.meshAreaLightsData[o + 5]!, packed.meshAreaLightsData[o + 6]!];
    const c: Vec3 = [packed.meshAreaLightsData[o + 8]!, packed.meshAreaLightsData[o + 9]!, packed.meshAreaLightsData[o + 10]!];
    const rad: Vec3 = [packed.meshAreaLightsData[o + 12]!, packed.meshAreaLightsData[o + 13]!, packed.meshAreaLightsData[o + 14]!];
    const ab: Vec3 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    const ac: Vec3 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    const cross: Vec3 = [ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2], ab[0] * ac[1] - ab[1] * ac[0]];
    const area = 0.5 * Math.hypot(cross[0], cross[1], cross[2]);
    const centroid: Vec3 = [(a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3, (a[2] + b[2] + c[2]) / 3];
    const min: Vec3 = [Math.min(a[0], b[0], c[0]), Math.min(a[1], b[1], c[1]), Math.min(a[2], b[2], c[2])];
    const max: Vec3 = [Math.max(a[0], b[0], c[0]), Math.max(a[1], b[1], c[1]), Math.max(a[2], b[2], c[2])];
    extend(min); extend(max);
    powers.push(emitterPower(rad, { kind: 'area', area }));
    centroids.push(centroid);
    aabbs.push({ min, max });
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
