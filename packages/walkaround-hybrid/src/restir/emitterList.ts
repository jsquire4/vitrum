/**
 * EmitterTri construction for ReSTIR DI.
 *
 * Builds the GPU-side emitter list, power CDF, and per-emitter cell-power
 * buffer from a scene's triangle / material data. Extracted from
 * the legacy `legacy/three/restirBvhCompute.ts` mixed builder (was ~190 lines inline).
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

import type { MaterialSpec } from '@vitrum/core';
import {
  luminance,
  buildLightTree,
  packLightTreeForGPU,
  LIGHT_TREE_FLOATS_PER_NODE,
} from '@vitrum/shared-samplers';
import {
  type BarycentricWeights,
  classifyTriangleEmitterCore,
  emissiveMapTriangleSubdivisionLevel,
  estimateMaterialSpecEmissiveLeOverTriangle,
  forEachBarycentricSubTriangle,
  forEachEmissiveMapTexelSubTriangle,
  materialSpecScalarEmissiveLe,
} from '@vitrum/shared-bvh';

interface Vector3Like {
  readonly x: number;
  readonly y: number;
  readonly z: number;
}


/**
 * EmitterTri struct layout (80 bytes, 16-byte aligned, 20 f32 per entry):
 *   0..15  : vertexA.xyz + sourceTriIndex (-1 for non-BVH/placeholder emitters)
 *   16..31 : vertexB.xyz + sourceSubdivLevel
 *   32..47 : vertexC.xyz + sourceSubdivOrdinal
 *   48..63 : normal.xyz + area
 *   64..79 : Le.rgb + castShadowDisabled
 * Padded to 80 bytes (5 × vec4f) for 16-byte alignment.
 */
// Canonical byte stride for the shared EmitterTri storage layout. Import this
// anywhere host-side code has to derive or validate an emitter count from raw
// packed bytes.
export const EMITTER_TRI_STRIDE_BYTES = 80;
const EMITTER_FLOATS = EMITTER_TRI_STRIDE_BYTES / 4;


interface EmitterListOptions {
  primaryLightDir?: Vector3Like;
  primaryLightIntensity?: number;
  /**
   * Additional emitter triangles from non-mesh sources (e.g. THREE.RectAreaLight
   * or other scene-graph lights that do not appear in the BVH). These are
   * appended verbatim AFTER the BVH-iteration produces its own emitter list,
   * and suppress the synthetic-placeholder fallback when present.
   *
   * The caller is responsible for folding any per-light intensity into
   * `Le`. `castShadow:false` is packed into the legacy `.w` lane of the
   * fifth vec4 so direct-light shadow rays can skip the occlusion test for
   * author-opted emitters without changing the 80-byte stride.
   */
  extraEmitters?: ReadonlyArray<{
    vA: [number, number, number];
    vB: [number, number, number];
    vC: [number, number, number];
    normal: [number, number, number];
    area: number;
    Le: [number, number, number];
    castShadow?: boolean;
  }>;
  /**
   * When true, mesh-material emitters pack their triangle index into the first
   * padding lane so DI shaders can resample UV-varying emissive maps at the
   * stored candidate `xi`. The index must be in the active render buffers'
   * `bvh_index` / material-atlas triangle space; callers with a world-expanded
   * emitter stream can provide `sourceTriIndexForTriangle` to translate.
   * Negative encodings are reserved: `-1` means fallback to averaged radiance,
   * while `-(tri + 2)` means source triangle `tri` with reversed barycentric
   * orientation for mirrored TLAS instances.
   */
  packSourceTriIndex?: boolean;
  /**
   * Optional mapper from the emitter-list triangle index to the active render
   * buffers' triangle index. Returning a negative/non-finite value keeps that
   * emitter on the averaged-radiance fallback.
   */
  sourceTriIndexForTriangle?: (triIdx: number) => number;
  /** Merged UV0 stream, stride-2 and aligned with `positions`. */
  uvs?: Float32Array;
  /** Optional merged UV1 stream, stride-2 and aligned with `positions`. */
  uv1s?: Float32Array;
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
 * emitter array RIS samples a point on (built 1:1 with the emitter list).
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

/**
 * THREE-free ReSTIR-DI emitter list builder from a merged world-space triangle
 * stream + core `MaterialSpec[]`. Thin wrapper over {@link buildEmitterListCore}
 * that supplies the core-material classifier ({@link classifyTriangleEmitterCore}).
 * Used when the geometry is ingested from a `@vitrum/core` `Scene` via
 * `mergeWorldSpaceFromCore`.
 */
export function buildEmitterListFromCore(
  indices: Uint32Array,
  positions: Float32Array,    // stride-4: read .xyz only
  normals: Float32Array,      // stride-4: read .xyz only
  triMatIdMap: Uint32Array,
  materials: readonly MaterialSpec[],
  options: EmitterListOptions,
): {
  emitterFloats: Float32Array;
  cdfArray: Float32Array;
  totalEmissivePower: number;
  /** Light-tree build inputs aligned 1:1 with the emitter list. */
  treeInput: EmitterTreeInput;
} {
  const ld = options.primaryLightDir;
  const lightDir = ld != null ? { x: ld.x, y: ld.y, z: ld.z } : { x: 0, y: 1, z: 0 };
  const primaryIntensity = options.primaryLightIntensity ?? 3.0;
  return buildEmitterListCore(
    indices,
    positions,
    normals,
    (t, normal) => {
      const mat = materials[triMatIdMap[t]!];
      if (!mat) return null;
      const classified = classifyTriangleEmitterCore(mat, normal, lightDir, primaryIntensity);
      const castShadowDisabled = (mat as MaterialSpec & { readonly castShadow?: boolean }).castShadow === false;
      if (classified == null) return null;
      if (options.packSourceTriIndex !== true) {
        return { ...classified, castShadowDisabled };
      }
      const scalarLe = scalarMaterialEmissiveLe(mat);
      if (scalarLe == null) return { ...classified, castShadowDisabled };
      const selectionColor = estimateTriangleEmissiveLe(
        mat,
        indices,
        t,
        options.uvs,
        options.uv1s,
      ) ?? classified.color;
      const exactTexelSubTriangles: {
        a: BarycentricWeights;
        b: BarycentricWeights;
        c: BarycentricWeights;
        radiance: [number, number, number];
      }[] = [];
      const i0 = indices[t * 3 + 0] ?? 0;
      const i1 = indices[t * 3 + 1] ?? 0;
      const i2 = indices[t * 3 + 2] ?? 0;
      const exactTexelHandled = mat.emissiveMap != null &&
        forEachEmissiveMapTexelSubTriangle(
          mat,
          uvAt(options.uvs, i0),
          uvAt(options.uvs, i1),
          uvAt(options.uvs, i2),
          uvAt(options.uv1s, i0),
          uvAt(options.uv1s, i1),
          uvAt(options.uv1s, i2),
          (wa, wb, wc, radiance) => {
            exactTexelSubTriangles.push({
              a: wa,
              b: wb,
              c: wc,
              radiance: [radiance[0], radiance[1], radiance[2]],
            });
          },
        );
      const sourceTriIndex = options.sourceTriIndexForTriangle?.(t) ?? t;
      if (exactTexelHandled) {
        return {
          ...classified,
          castShadowDisabled,
          color: scalarLe,
          selectionColor,
          texelSubTriangles: exactTexelSubTriangles,
        };
      }
      if (
        !Number.isFinite(sourceTriIndex) ||
        (sourceTriIndex < 0 && Math.trunc(sourceTriIndex) !== sourceTriIndex) ||
        sourceTriIndex === -1
      ) {
        return classified;
      }
      return {
        ...classified,
        castShadowDisabled,
        color: scalarLe,
        selectionColor,
        sourceTriIndex: Math.trunc(sourceTriIndex),
        subdivisionLevel: emissiveMapTriangleSubdivisionLevel(mat),
        subdivisionSelectionColor: (wa: BarycentricWeights, wb: BarycentricWeights, wc: BarycentricWeights) =>
          estimateSubTriangleEmissiveLe(
            mat,
            indices,
            t,
            wa,
            wb,
            wc,
            options.uvs,
            options.uv1s,
          ) ?? selectionColor,
      };
    },
    options,
  );
}

/**
 * A per-triangle emitter classifier: given a triangle index and its (already
 * area-weighted, normalized) world face normal, return the emitter
 * `{ color, intensity }` or `null` to skip. This is the ONLY material-typed step
 * in the emitter-list build, so the THREE and core variants share everything
 * else (geometry derivation, power gate, packing, CDF, light-tree input) by
 * supplying their own classifier here.
 */
type TriangleEmitterClassifier = (
  triIdx: number,
  normal: { x: number; y: number; z: number },
) => {
  /** Radiance packed into EmitterTri.Le for shader evaluation. */
  color: [number, number, number];
  intensity: number;
  /**
   * Radiance used for power-CDF / light-tree selection. This differs from
   * `color` when shader evaluation applies a UV-varying emissive map at
   * candidate time but selection still uses the stable CPU average.
   */
  selectionColor?: [number, number, number];
  /** Valid atlas/BVH triangle id for emissive-map sampling, or absent for fallback. */
  sourceTriIndex?: number;
  /** Barycentric micro-triangle split for UV-varying emitter selection. */
  subdivisionLevel?: number;
  /** Per-micro-triangle selection radiance estimate in parent barycentric space. */
  subdivisionSelectionColor?: (
    wa: BarycentricWeights,
    wb: BarycentricWeights,
    wc: BarycentricWeights,
  ) => [number, number, number];
  /** Exact CPU-readable emissive-map texel cells, expressed in parent barycentrics. */
  texelSubTriangles?: readonly {
    a: BarycentricWeights;
    b: BarycentricWeights;
    c: BarycentricWeights;
    radiance: [number, number, number];
  }[];
  /** Source primitive explicitly opted out of shadow casting. */
  castShadowDisabled?: boolean;
} | null;

function scalarMaterialEmissiveLe(material: MaterialSpec): [number, number, number] | null {
  return materialSpecScalarEmissiveLe(material);
}

function uvAt(uvs: Float32Array | undefined, vertex: number): [number, number] {
  if (uvs == null) return [0, 0];
  return [uvs[vertex * 2] ?? 0, uvs[vertex * 2 + 1] ?? 0];
}

function baryVec3(
  a: [number, number, number],
  b: [number, number, number],
  c: [number, number, number],
  w: BarycentricWeights,
): [number, number, number] {
  return [
    a[0] * w[0] + b[0] * w[1] + c[0] * w[2],
    a[1] * w[0] + b[1] * w[1] + c[1] * w[2],
    a[2] * w[0] + b[2] * w[1] + c[2] * w[2],
  ];
}

function baryUv(
  a: [number, number],
  b: [number, number],
  c: [number, number],
  w: BarycentricWeights,
): [number, number] {
  return [
    a[0] * w[0] + b[0] * w[1] + c[0] * w[2],
    a[1] * w[0] + b[1] * w[1] + c[1] * w[2],
  ];
}

function estimateTriangleEmissiveLe(
  material: MaterialSpec,
  indices: Uint32Array,
  triIdx: number,
  uvs?: Float32Array,
  uv1s?: Float32Array,
): [number, number, number] | null {
  if (material.emissiveMap == null) return materialSpecScalarEmissiveLe(material);
  const i0 = indices[triIdx * 3 + 0] ?? 0;
  const i1 = indices[triIdx * 3 + 1] ?? 0;
  const i2 = indices[triIdx * 3 + 2] ?? 0;
  return estimateMaterialSpecEmissiveLeOverTriangle(
    material,
    uvAt(uvs, i0),
    uvAt(uvs, i1),
    uvAt(uvs, i2),
    uvAt(uv1s, i0),
    uvAt(uv1s, i1),
    uvAt(uv1s, i2),
  );
}

function estimateSubTriangleEmissiveLe(
  material: MaterialSpec,
  indices: Uint32Array,
  triIdx: number,
  wa: BarycentricWeights,
  wb: BarycentricWeights,
  wc: BarycentricWeights,
  uvs?: Float32Array,
  uv1s?: Float32Array,
): [number, number, number] | null {
  const i0 = indices[triIdx * 3 + 0] ?? 0;
  const i1 = indices[triIdx * 3 + 1] ?? 0;
  const i2 = indices[triIdx * 3 + 2] ?? 0;
  const uv0a = uvAt(uvs, i0);
  const uv0b = uvAt(uvs, i1);
  const uv0c = uvAt(uvs, i2);
  const uv1a = uvAt(uv1s, i0);
  const uv1b = uvAt(uv1s, i1);
  const uv1c = uvAt(uv1s, i2);
  return estimateMaterialSpecEmissiveLeOverTriangle(
    material,
    baryUv(uv0a, uv0b, uv0c, wa),
    baryUv(uv0a, uv0b, uv0c, wb),
    baryUv(uv0a, uv0b, uv0c, wc),
    baryUv(uv1a, uv1b, uv1c, wa),
    baryUv(uv1a, uv1b, uv1c, wb),
    baryUv(uv1a, uv1b, uv1c, wc),
  );
}

/**
 * Shared emitter-list builder core. Iterates the merged world-space triangle
 * stream, derives each triangle's area + face normal (cross-product, then
 * the per-vertex-normal-average override identical to the original
 * `buildEmitterListCore`), runs the supplied `classify` callback, gates on
 * `power < 1e-8`, then packs the 80-byte EmitterTri buffer + power CDF +
 * light-tree inputs. {@link buildEmitterListFromCore} (`MaterialSpec[]`) calls
 * this with the core-material classifier — the geometry + packing math is
 * independent of the material system.
 */
function buildEmitterListCore(
  indices: Uint32Array,
  positions: Float32Array,    // stride-4: read .xyz only
  normals: Float32Array,      // stride-4: read .xyz only
  classify: TriangleEmitterClassifier,
  options: EmitterListOptions,
): {
  emitterFloats: Float32Array;
  cdfArray: Float32Array;
  totalEmissivePower: number;
  treeInput: EmitterTreeInput;
} {
  const triCount = indices.length / 3;

  const emitterData: {
    triIdx: number;
    sourceTriIndex: number;
    sourceSubdivLevel: number;
    sourceSubdivOrdinal: number;
    vA: [number, number, number];
    vB: [number, number, number];
    vC: [number, number, number];
    normal: [number, number, number];
    area: number;
    color: [number, number, number];
    intensity: number;
    castShadowDisabled: boolean;
    power: number;
  }[] = [];

  const pushEmitter = (e: {
    triIdx: number;
    sourceTriIndex: number;
    sourceSubdivLevel?: number;
    sourceSubdivOrdinal?: number;
    vA: [number, number, number];
    vB: [number, number, number];
    vC: [number, number, number];
    normal: [number, number, number];
    area: number;
    color: [number, number, number];
    intensity: number;
    castShadowDisabled: boolean;
    selectionColor?: [number, number, number];
  }): void => {
    const powerColor = e.selectionColor ?? e.color;
    const power = luminance(powerColor[0], powerColor[1], powerColor[2]) * e.area;
    if (power < 1e-8) return;
    emitterData.push({
      triIdx: e.triIdx,
      sourceTriIndex: e.sourceTriIndex,
      sourceSubdivLevel: e.sourceSubdivLevel ?? 1,
      sourceSubdivOrdinal: e.sourceSubdivOrdinal ?? 0,
      vA: e.vA,
      vB: e.vB,
      vC: e.vC,
      normal: e.normal,
      area: e.area,
      color: e.color,
      intensity: e.intensity,
      castShadowDisabled: e.castShadowDisabled,
      power,
    });
  };

  for (let t = 0; t < triCount; t++) {
    const i0 = indices[t * 3 + 0]!;
    const i1 = indices[t * 3 + 1]!;
    const i2 = indices[t * 3 + 2]!;

    const ax = positions[i0 * 4]!, ay = positions[i0 * 4 + 1]!, az = positions[i0 * 4 + 2]!;
    const bx = positions[i1 * 4]!, by = positions[i1 * 4 + 1]!, bz = positions[i1 * 4 + 2]!;
    const cx0 = positions[i2 * 4]!, cy0 = positions[i2 * 4 + 1]!, cz0 = positions[i2 * 4 + 2]!;

    const abx = bx - ax, aby = by - ay, abz = bz - az;
    const acx = cx0 - ax, acy = cy0 - ay, acz = cz0 - az;
    const crossX = aby * acz - abz * acy;
    const crossY = abz * acx - abx * acz;
    const crossZ = abx * acy - aby * acx;
    const crossLen = Math.sqrt(crossX * crossX + crossY * crossY + crossZ * crossZ);
    if (crossLen < 1e-8) continue;
    const area = crossLen * 0.5;
    const invLen = 1.0 / crossLen;
    let nx = crossX * invLen;
    let ny = crossY * invLen;
    let nz = crossZ * invLen;
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

    const classified = classify(t, { x: nx, y: ny, z: nz });
    if (!classified) continue;
    const [cr, cg, cb] = classified.color;
    const intensity = classified.intensity;
    const castShadowDisabled = classified.castShadowDisabled === true;

    const parentA: [number, number, number] = [ax, ay, az];
    const parentB: [number, number, number] = [bx, by, bz];
    const parentC: [number, number, number] = [cx0, cy0, cz0];
    const normal: [number, number, number] = [nx, ny, nz];
    if (classified.texelSubTriangles != null) {
      for (const patch of classified.texelSubTriangles) {
        const subA = baryVec3(parentA, parentB, parentC, patch.a);
        const subB = baryVec3(parentA, parentB, parentC, patch.b);
        const subC = baryVec3(parentA, parentB, parentC, patch.c);
        const sx = subB[0] - subA[0], sy = subB[1] - subA[1], sz = subB[2] - subA[2];
        const tx = subC[0] - subA[0], ty = subC[1] - subA[1], tz = subC[2] - subA[2];
        const subArea = 0.5 * Math.sqrt(
          (sy * tz - sz * ty) ** 2 +
          (sz * tx - sx * tz) ** 2 +
          (sx * ty - sy * tx) ** 2,
        );
        if (subArea < 1e-12) continue;
        pushEmitter({
          triIdx: t,
          sourceTriIndex: -1,
          vA: subA,
          vB: subB,
          vC: subC,
          normal,
          area: subArea,
          color: patch.radiance,
          intensity: 1,
          castShadowDisabled,
          selectionColor: patch.radiance,
        });
      }
      continue;
    }
    const sourceTriIndex = classified.sourceTriIndex ?? -1;
    const subdivisionLevel = Math.max(1, Math.floor(classified.subdivisionLevel ?? 1));
    if (sourceTriIndex !== -1 && subdivisionLevel > 1) {
      let ordinal = 0;
      forEachBarycentricSubTriangle(subdivisionLevel, (wa, wb, wc) => {
        const subA = baryVec3(parentA, parentB, parentC, wa);
        const subB = baryVec3(parentA, parentB, parentC, wb);
        const subC = baryVec3(parentA, parentB, parentC, wc);
        const subSelectionColor = classified.subdivisionSelectionColor?.(wa, wb, wc) ??
          classified.selectionColor ??
          classified.color;
        pushEmitter({
          triIdx: t,
          sourceTriIndex,
          sourceSubdivLevel: subdivisionLevel,
          sourceSubdivOrdinal: ordinal,
          vA: subA,
          vB: subB,
          vC: subC,
          normal,
          area: area / (subdivisionLevel * subdivisionLevel),
          color: [cr, cg, cb],
          intensity,
          castShadowDisabled,
          selectionColor: subSelectionColor,
        });
        ordinal += 1;
      });
      continue;
    }

    pushEmitter({
      triIdx: t,
      sourceTriIndex,
      vA: parentA,
      vB: parentB,
      vC: parentC,
      normal,
      area,
      color: [cr, cg, cb],
      intensity,
      castShadowDisabled,
      ...(classified.selectionColor != null ? { selectionColor: classified.selectionColor } : {}),
    });
  }

  if (options.extraEmitters) {
    for (const ex of options.extraEmitters) {
      const lum = luminance(ex.Le[0], ex.Le[1], ex.Le[2]);
      const power = lum * ex.area;
      if (power < 1e-8) continue;
      emitterData.push({
        triIdx: -1,
        sourceTriIndex: -1,
        sourceSubdivLevel: 1,
        sourceSubdivOrdinal: 0,
        vA: ex.vA, vB: ex.vB, vC: ex.vC,
        normal: ex.normal,
        area: ex.area,
        color: ex.Le,
        intensity: 1,
        castShadowDisabled: ex.castShadow === false,
        power,
      });
    }
  }

  if (emitterData.length === 0) {
    // H22 — the phantom placeholder is a structural requirement: the GPU buffer
    // binding needs at least one element to avoid a zero-byte storage binding
    // (WebGPU validation error). However, the placeholder MUST NOT contribute
    // light: setting Le=[0,0,0] (intensity=0) ensures pHat = 0 at every RIS
    // candidate evaluation → the RIS weight w=0 → no reservoir update → the
    // shade pass returns vec3f(0) on the empty `r.W <= 0` guard. The shade.wgsl
    // `if (lid >= ubo.emitterCount)` guard does NOT fire because emitterCount=1
    // (the placeholder slot), but the zero-radiance Le makes the placeholder
    // contribution exactly 0. Verified: ris.wgsl:232 `let pHat = luminance(...)`
    // is 0 when Le=0, so `let w = select(0.0, pHat/pX, pHat > 0.0)` returns 0.
    emitterData.push({
      triIdx: -1,
      sourceTriIndex: -1,
      sourceSubdivLevel: 1,
      sourceSubdivOrdinal: 0,
      vA: [0, 10, 0], vB: [1, 10, 0], vC: [0.5, 10, 1],
      normal: [0, -1, 0],
      area: 0.5,
      color: [0, 0, 0],  // zero Le → pHat = 0 → inert (H22)
      intensity: 0,
      castShadowDisabled: false,
      power: 0,          // power = 0 → excluded from CDF (totalEmissivePower = 0)
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
    const sourceTriIndex = options.packSourceTriIndex === true ? e.sourceTriIndex : -1;
    emitterFloats[base + 0] = e.vA[0]; emitterFloats[base + 1] = e.vA[1]; emitterFloats[base + 2] = e.vA[2]; emitterFloats[base + 3] = sourceTriIndex;
    emitterFloats[base + 4] = e.vB[0]; emitterFloats[base + 5] = e.vB[1]; emitterFloats[base + 6] = e.vB[2]; emitterFloats[base + 7] = sourceTriIndex !== -1 ? e.sourceSubdivLevel : 1;
    emitterFloats[base + 8] = e.vC[0]; emitterFloats[base + 9] = e.vC[1]; emitterFloats[base + 10] = e.vC[2]; emitterFloats[base + 11] = sourceTriIndex !== -1 ? e.sourceSubdivOrdinal : 0;
    emitterFloats[base + 12] = e.normal[0]; emitterFloats[base + 13] = e.normal[1]; emitterFloats[base + 14] = e.normal[2]; emitterFloats[base + 15] = e.area;
    emitterFloats[base + 16] = e.color[0]; emitterFloats[base + 17] = e.color[1]; emitterFloats[base + 18] = e.color[2]; emitterFloats[base + 19] = e.castShadowDisabled ? 1 : 0;
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
    // H22 guard: when totalEmissivePower = 0 (the zero-real-emitter placeholder
    // path), set a uniform CDF so the CDF buffer contains valid f32 values (not
    // NaN). The placeholder's Le=0 makes every candidate pHat=0 → RIS weight=0
    // → the reservoir stays empty → shade returns 0. CDF content is irrelevant
    // when all weights are zero, but NaN in the buffer would produce undefined
    // GPU behaviour.
    cdfArray[i] = totalEmissivePower > 0 ? runningSum / totalEmissivePower : (i + 1) / emitterCount;
  }

  return { emitterFloats, cdfArray, totalEmissivePower, treeInput };
}
