/**
 * EmitterTri construction for ReSTIR DI.
 *
 * Builds the GPU-side emitter list, power CDF, and per-emitter cell-power
 * buffer from a scene's triangle / material data. Extracted from
 * `bvhCompute.ts` (was ~190 lines inline).
 *
 * Selection rules (per triangle, priority order):
 *   1. emissive (luminance > 0 AND emissiveIntensity > 0) → direct emitter
 *   2. transmission > 0.1 AND !userData.skipEmitter → sun-attenuated
 *      secondary emitter (gated on primaryLightDot > 0.05)
 *   3. otherwise → skipped
 *
 * Emitters with power < 1e-8 are dropped. If the resulting list is empty,
 * a synthetic dummy emitter is inserted so the GPU buffer is non-empty
 * (WGSL bind groups can't be size 0).
 */

import * as THREE from 'three';

/**
 * EmitterTri struct layout (80 bytes, 16-byte aligned, 20 f32 per entry):
 *   0..11  : vertexA (12 bytes)
 *   12..23 : vertexB (12 bytes)
 *   24..35 : vertexC (12 bytes)
 *   36..47 : normal  (12 bytes)
 *   48..51 : area    ( 4 bytes)
 *   52..63 : color   (12 bytes)
 *   64..67 : intensity (4 bytes)
 * Padded to 80 bytes (5 × vec4f) for 16-byte alignment.
 */
// EMITTER_STRIDE / EMITTER_FLOATS — file-local (only used inside this
// module's emitter packer). The matching layout comment in
// restir/bvhCompute.ts:131 references the names by spelling, not by
// import. 2026-05-18 dead-code sweep verified zero non-self consumers.
const EMITTER_STRIDE = 80;
const EMITTER_FLOATS = EMITTER_STRIDE / 4;

function luminance(r: number, g: number, b: number): number {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/**
 * Classify a material + face normal as an emitter, or null if the face
 * isn't selected. Implements the priority order described in the file
 * header (emissive > transmissive with skipEmitter override). Extracted so
 * the per-triangle loop body is "classify → if not null, accumulate."
 *
 * `lightDir` should already be the configured primary-light direction;
 * `intensity` is the configured primary-light irradiance.
 */
function classifyTriangleEmitter(
  mat: THREE.Material,
  normal: { x: number; y: number; z: number },
  lightDir: THREE.Vector3,
  primaryIntensity: number,
): { color: [number, number, number]; intensity: number } | null {
  const meshMat = mat as THREE.MeshStandardMaterial;
  const emissiveLum = meshMat.emissive
    ? luminance(meshMat.emissive.r, meshMat.emissive.g, meshMat.emissive.b)
    : 0;
  if (emissiveLum > 0 && meshMat.emissiveIntensity && meshMat.emissiveIntensity > 0) {
    return {
      color: [
        meshMat.emissive.r * meshMat.emissiveIntensity,
        meshMat.emissive.g * meshMat.emissiveIntensity,
        meshMat.emissive.b * meshMat.emissiveIntensity,
      ],
      intensity: meshMat.emissiveIntensity,
    };
  }
  const physMat = mat as THREE.MeshPhysicalMaterial;
  if (!physMat.transmission || physMat.transmission <= 0.1) return null;

  const skipEmitter = (mat.userData as { skipEmitter?: boolean } | undefined)?.skipEmitter === true;
  if (skipEmitter) return null;

  const sunDot = Math.abs(
    lightDir.x * normal.x + lightDir.y * normal.y + lightDir.z * normal.z,
  );
  if (sunDot <= 0.05) return null;

  const baseColor = physMat.color ?? new THREE.Color(1, 1, 1);
  const attenColor = physMat.attenuationColor ?? new THREE.Color(1, 1, 1);
  const trans = physMat.transmission;
  return {
    color: [
      baseColor.r * attenColor.r * trans * primaryIntensity * sunDot,
      baseColor.g * attenColor.g * trans * primaryIntensity * sunDot,
      baseColor.b * attenColor.b * trans * primaryIntensity * sunDot,
    ],
    intensity: primaryIntensity * trans * sunDot,
  };
}

export interface EmitterListOptions {
  primaryLightDir?: THREE.Vector3;
  primaryLightIntensity?: number;
  /**
   * Additional emitter triangles from non-mesh sources (e.g. THREE.RectAreaLight
   * or other scene-graph lights that do not appear in the BVH). These are
   * appended verbatim AFTER the BVH-iteration produces its own emitter list,
   * and suppress the synthetic-placeholder fallback when present.
   *
   * The caller is responsible for folding any per-light intensity into
   * `Le` (the WGSL `EmitterTri.Le` field is the only radiance source the
   * shade kernel reads — the `intensity` field is legacy and ignored by
   * WGSL).
   */
  extraEmitters?: ReadonlyArray<{
    vA: [number, number, number];
    vB: [number, number, number];
    vC: [number, number, number];
    normal: [number, number, number];
    area: number;
    Le: [number, number, number];
  }>;
}

export function buildEmitterList(
  indices: Uint32Array,
  positions: Float32Array,    // stride-4: read .xyz only
  normals: Float32Array,      // stride-4: read .xyz only
  triMatIdMap: Uint32Array,
  materials: THREE.Material[],
  options: EmitterListOptions,
): {
  emitterFloats: Float32Array;
  cdfArray: Float32Array;
  cellPowerArray: Float32Array;
  totalEmissivePower: number;
} {
  const triCount = indices.length / 3;

  const emitterData: {
    triIdx: number;
    vA: [number, number, number];
    vB: [number, number, number];
    vC: [number, number, number];
    normal: [number, number, number];
    area: number;
    color: [number, number, number];
    intensity: number;
    power: number;
  }[] = [];

  const _va = new THREE.Vector3();
  const _vb = new THREE.Vector3();
  const _vc = new THREE.Vector3();
  const _ab = new THREE.Vector3();
  const _ac = new THREE.Vector3();
  const _cross = new THREE.Vector3();

  for (let t = 0; t < triCount; t++) {
    const i0 = indices[t * 3 + 0]!;
    const i1 = indices[t * 3 + 1]!;
    const i2 = indices[t * 3 + 2]!;

    _va.set(positions[i0 * 4]!, positions[i0 * 4 + 1]!, positions[i0 * 4 + 2]!);
    _vb.set(positions[i1 * 4]!, positions[i1 * 4 + 1]!, positions[i1 * 4 + 2]!);
    _vc.set(positions[i2 * 4]!, positions[i2 * 4 + 1]!, positions[i2 * 4 + 2]!);

    _ab.subVectors(_vb, _va);
    _ac.subVectors(_vc, _va);
    _cross.crossVectors(_ab, _ac);
    const crossLen = _cross.length();
    if (crossLen < 1e-8) continue;
    const area = crossLen * 0.5;
    const invLen = 1.0 / crossLen;
    let nx = _cross.x * invLen;
    let ny = _cross.y * invLen;
    let nz = _cross.z * invLen;
    const n0x = normals[i0 * 4]!;
    const n0y = normals[i0 * 4 + 1]!;
    const n0z = normals[i0 * 4 + 2]!;
    const hasNormals = (n0x !== 0 || n0y !== 0 || n0z !== 0);
    if (hasNormals) {
      nx = (n0x + normals[i1 * 4]! + normals[i2 * 4]!) / 3;
      ny = (n0y + normals[i1 * 4 + 1]! + normals[i2 * 4 + 1]!) / 3;
      nz = (n0z + normals[i1 * 4 + 2]! + normals[i2 * 4 + 2]!) / 3;
      const nlen = Math.sqrt(nx * nx + ny * ny + nz * nz);
      if (nlen > 1e-6) { nx /= nlen; ny /= nlen; nz /= nlen; }
    }

    const matId = triMatIdMap[t]!;
    const mat = materials[matId];
    if (!mat) continue;

    const lightDir = options.primaryLightDir ?? new THREE.Vector3(0, 1, 0);
    const primaryIntensity = options.primaryLightIntensity ?? 3.0;
    const classified = classifyTriangleEmitter(
      mat,
      { x: nx, y: ny, z: nz },
      lightDir,
      primaryIntensity,
    );
    if (!classified) continue;
    const [cr, cg, cb] = classified.color;
    const intensity = classified.intensity;

    const power = luminance(cr, cg, cb) * area;
    if (power < 1e-8) continue;

    emitterData.push({
      triIdx: t,
      vA: [_va.x, _va.y, _va.z],
      vB: [_vb.x, _vb.y, _vb.z],
      vC: [_vc.x, _vc.y, _vc.z],
      normal: [nx, ny, nz],
      area,
      color: [cr, cg, cb],
      intensity,
      power,
    });
  }

  if (options.extraEmitters) {
    for (const ex of options.extraEmitters) {
      const lum = luminance(ex.Le[0], ex.Le[1], ex.Le[2]);
      const power = lum * ex.area;
      if (power < 1e-8) continue;
      emitterData.push({
        triIdx: -1,
        vA: ex.vA, vB: ex.vB, vC: ex.vC,
        normal: ex.normal,
        area: ex.area,
        color: ex.Le,
        intensity: 1,
        power,
      });
    }
  }

  if (emitterData.length === 0) {
    emitterData.push({
      triIdx: 0,
      vA: [0, 10, 0], vB: [1, 10, 0], vC: [0.5, 10, 1],
      normal: [0, -1, 0],
      area: 0.5,
      color: [1, 1, 1],
      intensity: 1,
      power: 0.5,
    });
  }

  const emitterCount = emitterData.length;
  const emitterFloats = new Float32Array(emitterCount * EMITTER_FLOATS);
  let totalEmissivePower = 0;

  for (let i = 0; i < emitterCount; i++) {
    const e = emitterData[i]!;
    const base = i * EMITTER_FLOATS;
    emitterFloats[base + 0] = e.vA[0]; emitterFloats[base + 1] = e.vA[1]; emitterFloats[base + 2] = e.vA[2]; emitterFloats[base + 3] = 0;
    emitterFloats[base + 4] = e.vB[0]; emitterFloats[base + 5] = e.vB[1]; emitterFloats[base + 6] = e.vB[2]; emitterFloats[base + 7] = 0;
    emitterFloats[base + 8] = e.vC[0]; emitterFloats[base + 9] = e.vC[1]; emitterFloats[base + 10] = e.vC[2]; emitterFloats[base + 11] = 0;
    emitterFloats[base + 12] = e.normal[0]; emitterFloats[base + 13] = e.normal[1]; emitterFloats[base + 14] = e.normal[2]; emitterFloats[base + 15] = e.area;
    emitterFloats[base + 16] = e.color[0]; emitterFloats[base + 17] = e.color[1]; emitterFloats[base + 18] = e.color[2]; emitterFloats[base + 19] = e.intensity;
    totalEmissivePower += e.power;
  }

  const cdfArray = new Float32Array(emitterCount);
  let runningSum = 0;
  for (let i = 0; i < emitterCount; i++) {
    runningSum += emitterData[i]!.power;
    cdfArray[i] = runningSum / totalEmissivePower;
  }

  const cellPowerArray = new Float32Array(emitterCount);
  for (let i = 0; i < emitterCount; i++) {
    cellPowerArray[i] = emitterData[i]!.power;
  }

  return { emitterFloats, cdfArray, cellPowerArray, totalEmissivePower };
}
