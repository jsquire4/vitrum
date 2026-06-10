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
 * Directional light layout (2 vec4 = 8 floats, N-directional expansion):
 *   vec4 0: direction.xyz (normalized toward light), angularDiameter (radians)
 *   vec4 1: irradiance.rgb, mean_irradiance
 *   angularDiameter = 0 ⟹ perfect delta directional (historical exact path, byte-identical).
 *   mean_irradiance = (r+g+b)/3 — cached for the kernel gate (lightDir.w analog).
 *
 * Directional[0] is ALSO mirrored into the frame-UBO lightDir/cameraPos.w lanes for
 * backward compatibility with the single-directional path read by in-medium NEE.
 * For N > 1, lights [1..N-1] are read ONLY from this storage buffer; the kernel
 * loops params.directionalLightCount records.
 */
export const DIRECTIONAL_LIGHT_FLOAT_STRIDE = 8;

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

/**
 * D3 — soft-sun angular diameter (radians) for the scene's directional emitter.
 * 0 (the default, and when no directional emitter is present) = a perfect delta
 * directional, the historical exact path (byte-identical). A positive value turns
 * the directional NEE into a cone sampler over the sun's solid angle for soft
 * shadows. Carried in the frame UBO's `cameraPos.w` lane (a previously-unused
 * `.w` slot — see frameParamsPacker) so no UBO byte-size/layout change is needed.
 * Ref: DirectionalEmitter.angularDiameter (core contract).
 */
export function defaultDirectionalAngularDiameter(scene: Scene): number {
  const directional = scene.emitters.find((e) => e.kind === 'directional');
  if (directional == null) return 0;
  const ad = directional.angularDiameter;
  return ad != null && Number.isFinite(ad) && ad > 0 ? ad : 0;
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

  // N-directional packing — all directional emitters go into a flat storage-buffer
  // array. The first directional[0] is ALSO mirrored into the frame-UBO lightDir
  // and cameraPos.w lanes by frameParamsPacker.ts (backward-compat, in-medium NEE).
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
    const angularDiameter = ad != null && Number.isFinite(ad) && ad > 0 ? ad : 0;
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
    pushVec4(meshAreaLights, tri.radiance);
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
  // Check explicit mesh-area / disc-area emitters first.
  for (const e of scene.emitters) {
    if (e.kind !== 'mesh-area' && e.kind !== 'disc-area') continue;
    // Both kinds carry a `meshId` on the SceneEmitter type.
    const meshId = (e as { meshId?: string }).meshId;
    if (meshId === primitiveId) return true;
  }
  // Item 2b — also check for implicit emitters: a non-analytic primitive with
  // non-zero emissive luminance synthesizes `__implicit__<id>` at pack time.
  // Only check when the primitive has no explicit mesh-area emitter (already
  // handled above) to avoid double-counting.
  const primitive = scene.primitives.find((p) => p.id === primitiveId);
  if (primitive != null && primitive.kind !== 'analytic') {
    const em = primitive.material.emissive ?? [0, 0, 0];
    const ei = primitive.material.emissiveIntensity ?? 1;
    if (luminance(em[0] * ei, em[1] * ei, em[2] * ei) >= 1e-6) {
      return true;
    }
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
 *
 * `precomputed` — optional already-computed sub-results. When provided, the
 * function skips the corresponding internal calls (avoids re-running the
 * potentially-expensive HDRI/sky bake a second time for the same scene):
 *   - `packed`: result of a prior `packEmitterArrays(scene)` call.
 *   - `envSummary`: narrow env metadata derived from a prior `environmentParams(scene)`.
 * Both are verified to be passed from the SAME scene to preserve byte-identity.
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
        // B8 — rect-area light emits from one side along its geometric normal
        // (u × v). Single-sided cosine lobe ⇒ thetaE = π/2 (hemisphere).
        cones.push({ axis: norm3(cross3(u, v)), thetaO: 0, thetaE: HEMISPHERE });
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
