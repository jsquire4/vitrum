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
import {
  luminance,
  buildLightTree,
  packLightTreeForGPU,
  LIGHT_TREE_FLOATS_PER_NODE,
} from '@vitrum/shared-samplers';
import { materialEmissiveLe } from './packingHelpers.js';

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
  // Emissive surface → direct emitter. Shares `materialEmissiveLe` with the
  // camera-visible-glow packer (packBVHEmissiveLe) so the NEE-sampled radiance
  // and the camera glow Le are GUARANTEED identical (no drift).
  const emissiveLe = materialEmissiveLe(mat);
  if (emissiveLe != null) {
    return { color: emissiveLe, intensity: meshMat.emissiveIntensity ?? 1 };
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

interface EmitterListOptions {
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

/**
 * Per-emitter light-tree build inputs, aligned 1:1 with the emitter list (same
 * order/indices as `emitterFloats` / `cdfArray`). Consumed by
 * `@vitrum/shared-samplers buildLightTree` so leaf `emitterIndex` values index
 * the same GPU emitter array RIS samples a point on. Power is the SAME
 * `luminance(color)·area` quantity the flat power CDF is built from, so the
 * tree's power weighting and the CDF agree exactly.
 */
export interface EmitterTreeInput {
  /** luminance(color)·area per emitter (matches CDF power). */
  powers: number[];
  /** Triangle centroid (vA+vB+vC)/3 per emitter (spatial-split heuristic). */
  centroids: [number, number, number][];
  /** Per-emitter triangle AABB (min/max over the 3 vertices). */
  aabbs: { min: [number, number, number]; max: [number, number, number] }[];
}

/**
 * Built light-tree GPU artifact: the packed flat-f32 node buffer + node count +
 * a gate. `enabled` is `true` only when there are ≥ 2 emitters (a 1-emitter
 * scene gains nothing from a tree — selection is deterministic). When disabled,
 * `nodes` is a single zeroed placeholder node so the GPU storage buffer is
 * non-empty (WGSL bind groups cannot be size 0) but is never dereferenced (RIS
 * branches on the `lightTreeEnabled` UBO gate before touching the buffer).
 */
export interface LightTreeBuffer {
  /** Packed flat-f32 nodes (LIGHT_TREE_FLOATS_PER_NODE per node). */
  nodes: Float32Array;
  /** Number of nodes (0 when disabled). */
  nodeCount: number;
  /** Whether RIS should select via the tree (≥ 2 emitters). */
  enabled: boolean;
}

/**
 * Build + serialise the ReSTIR-DI light-selection tree from the emitter-list
 * tree inputs. Returns a zeroed 1-node placeholder + `enabled:false` when there
 * are fewer than 2 emitters (or no positive-power emitters), so RIS falls back
 * to the flat power-CDF path. Leaf `emitterIndex` values index the SAME GPU
 * emitter array RIS samples a point on (built 1:1 with `buildEmitterList`).
 */
export function buildLightTreeBuffer(treeInput: EmitterTreeInput): LightTreeBuffer {
  const n = treeInput.powers.length;
  const totalPower = treeInput.powers.reduce((a, b) => a + b, 0);
  // < 2 emitters → deterministic selection; > 0 power required for a meaningful
  // power-weighted tree. Either way fall back to the flat CDF path.
  if (n < 2 || totalPower <= 0) {
    return {
      nodes: new Float32Array(LIGHT_TREE_FLOATS_PER_NODE), // 1 zeroed placeholder node
      nodeCount: 0,
      enabled: false,
    };
  }
  const { nodes } = buildLightTree({
    powers: treeInput.powers,
    centroids: treeInput.centroids,
    aabbs: treeInput.aabbs,
  });
  return {
    nodes: packLightTreeForGPU(nodes),
    nodeCount: nodes.length,
    enabled: true,
  };
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
  totalEmissivePower: number;
  /** Light-tree build inputs aligned 1:1 with the emitter list. */
  treeInput: EmitterTreeInput;
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

  // Light-tree build inputs, aligned 1:1 with the emitter list (leaf
  // emitterIndex == emitter array index). Power is the SAME luminance·area
  // quantity the flat power CDF accumulates, so the two selection
  // distributions share their power weighting exactly.
  const treeInput: EmitterTreeInput = { powers: [], centroids: [], aabbs: [] };

  for (let i = 0; i < emitterCount; i++) {
    const e = emitterData[i]!;
    const base = i * EMITTER_FLOATS;
    emitterFloats[base + 0] = e.vA[0]; emitterFloats[base + 1] = e.vA[1]; emitterFloats[base + 2] = e.vA[2]; emitterFloats[base + 3] = 0;
    emitterFloats[base + 4] = e.vB[0]; emitterFloats[base + 5] = e.vB[1]; emitterFloats[base + 6] = e.vB[2]; emitterFloats[base + 7] = 0;
    emitterFloats[base + 8] = e.vC[0]; emitterFloats[base + 9] = e.vC[1]; emitterFloats[base + 10] = e.vC[2]; emitterFloats[base + 11] = 0;
    emitterFloats[base + 12] = e.normal[0]; emitterFloats[base + 13] = e.normal[1]; emitterFloats[base + 14] = e.normal[2]; emitterFloats[base + 15] = e.area;
    emitterFloats[base + 16] = e.color[0]; emitterFloats[base + 17] = e.color[1]; emitterFloats[base + 18] = e.color[2]; emitterFloats[base + 19] = e.intensity;
    totalEmissivePower += e.power;

    treeInput.powers.push(e.power);
    treeInput.centroids.push([
      (e.vA[0] + e.vB[0] + e.vC[0]) / 3,
      (e.vA[1] + e.vB[1] + e.vC[1]) / 3,
      (e.vA[2] + e.vB[2] + e.vC[2]) / 3,
    ]);
    treeInput.aabbs.push({
      min: [
        Math.min(e.vA[0], e.vB[0], e.vC[0]),
        Math.min(e.vA[1], e.vB[1], e.vC[1]),
        Math.min(e.vA[2], e.vB[2], e.vC[2]),
      ],
      max: [
        Math.max(e.vA[0], e.vB[0], e.vC[0]),
        Math.max(e.vA[1], e.vB[1], e.vC[1]),
        Math.max(e.vA[2], e.vB[2], e.vC[2]),
      ],
    });
  }

  const cdfArray = new Float32Array(emitterCount);
  let runningSum = 0;
  for (let i = 0; i < emitterCount; i++) {
    runningSum += emitterData[i]!.power;
    cdfArray[i] = runningSum / totalEmissivePower;
  }

  return { emitterFloats, cdfArray, totalEmissivePower, treeInput };
}
