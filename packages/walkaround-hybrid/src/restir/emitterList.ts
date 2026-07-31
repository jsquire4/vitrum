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
 * Only non-positive or non-finite-power emitters are dropped. If the resulting
 * list is empty,
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
  classifyTriangleEmitterCore,
  materialSpecScalarEmissiveLe,
  multiplyNonNegativeRadianceScalarsF32,
  packNonNegativeRadianceRgbF32,
  packNonNegativeRadianceScalarF32,
} from '@vitrum/shared-bvh';
import {
  classifyTriangleAreaF32,
  normalizeDirectionF32,
} from './areaEmitterGeometry.js';

/**
 * EmitterTri struct layout (80 bytes, 16-byte aligned, 20 f32 per entry):
 *   0..15  : vertexA.xyz + sourceTriIndex (-1 for non-BVH/placeholder emitters)
 *   16..31 : vertexB.xyz + sourceSubdivLevel
 *   32..47 : vertexC.xyz + sourceSubdivOrdinal
 *   48..63 : normal.xyz + area
 *   64..79 : Le.rgb + emitterFlags
 *             bit 0 = castShadowDisabled
 *             bit 1 = twoSided
 * Padded to 80 bytes (5 × vec4f) for 16-byte alignment.
 */
// Canonical byte stride for the shared EmitterTri storage layout. Import this
// anywhere host-side code has to derive or validate an emitter count from raw
// packed bytes.
export const EMITTER_TRI_STRIDE_BYTES = 80;
const EMITTER_FLOATS = EMITTER_TRI_STRIDE_BYTES / 4;
export const EMITTER_TRI_CAST_SHADOW_DISABLED_FLAG = 1;
export const EMITTER_TRI_TWO_SIDED_FLAG = 2;


interface EmitterListOptions {
  /**
   * Additional emitter triangles from non-mesh sources (e.g. THREE.RectAreaLight
   * or other scene-graph lights that do not appear in the BVH). These are
   * appended AFTER the BVH iteration produces its own emitter list, then
   * canonicalized through the same geometry/radiance f32 envelope. Their
   * presence suppresses the synthetic-placeholder fallback.
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
    twoSided?: boolean;
  }>;
  /**
   * When true, mesh-material emitters pack their triangle index into the first
   * padding lane so DI shaders can resample UV-varying emissive maps at the
   * stored candidate `xi`. The index must be in the active render buffers'
   * `bvh_index` / material-atlas triangle space; callers with a world-expanded
   * emitter stream can provide `sourceTriIndexForTriangle` to translate.
   * Negative encodings are reserved: `-1` means constant packed radiance,
   * while `-(tri + 2)` means source triangle `tri` with reversed barycentric
   * orientation for mirrored TLAS instances.
   */
  packSourceTriIndex?: boolean;
  /**
   * Optional mapper from the emitter-list triangle index to the active render
   * buffers' triangle index. `-1`, a non-integer/non-finite value, or a value
   * that cannot round-trip through the packed f32 lane keeps a scalar emitter
   * on constant packed radiance. A mapped emitter rejects such a value because
   * evaluating its authored map requires an exact source triangle.
   */
  sourceTriIndexForTriangle?: (triIdx: number) => number;
}

/**
 * Per-emitter light-tree build inputs, aligned 1:1 with the emitter list (same
 * order/indices as `emitterFloats` / `cdfArray`). Consumed by
 * `@vitrum/shared-samplers buildLightTree` so leaf `emitterIndex` values index
 * the same GPU emitter array RIS samples a point on. Power is the exact
 * represented f32 CDF interval, so tree, alias, and flat-CDF proposal
 * probabilities agree even when authored powers span extreme dynamic range.
 */
export interface EmitterTreeInput {
  /** Positive f32 CDF intervals represented on the GPU wire. */
  powers: number[];
  /** Triangle centroid (vA+vB+vC)/3 per emitter (spatial-split heuristic). */
  centroids: [number, number, number][];
  /** Per-emitter triangle AABB (min/max over the 3 vertices). */
  aabbs: { min: [number, number, number]; max: [number, number, number] }[];
}

/**
 * Exact number of equal f32 CDF buckets in [0,1] when endpoints are encoded as
 * integer multiples of 2^-24. Every integer in this range is exactly
 * representable by binary32, including the final endpoint 1.
 */
export const EMITTER_CDF_F32_BUCKETS = 0x01000000;

/**
 * Quantize non-negative source weights to a monotone binary32 CDF.
 *
 * Every positive source receives at least one 2^-24 interval. Remaining
 * buckets are distributed with Hamilton's largest-remainder method, preserving
 * the authored ratios as closely as this wire format allows. The returned PMF
 * is derived from those exact intervals and is therefore the distribution
 * shader-side `emitterCdfPmf` observes.
 */
export function buildRepresentedEmitterCdfF32(
  weights: readonly number[],
): {
  readonly cdf: Float32Array;
  readonly representedPmf: Float32Array;
} {
  const count = weights.length;
  if (count === 0) {
    return {
      cdf: new Float32Array(0),
      representedPmf: new Float32Array(0),
    };
  }
  const packed = weights.map((weight, index) =>
    packNonNegativeRadianceScalarF32(
      weight,
      `emitter CDF weight[${index}]`,
    )
  );
  const positiveIndices = packed
    .map((weight, index) => weight > 0 ? index : -1)
    .filter((index) => index >= 0);
  const representedSupportCount = positiveIndices.length === 0
    ? count
    : positiveIndices.length;
  if (representedSupportCount > EMITTER_CDF_F32_BUCKETS) {
    throw new RangeError(
      'emitter CDF has more positive intervals than binary32 can represent.',
    );
  }

  const buckets = new Uint32Array(count);
  if (positiveIndices.length === 0) {
    const base = Math.floor(EMITTER_CDF_F32_BUCKETS / count);
    let remainder = EMITTER_CDF_F32_BUCKETS - base * count;
    for (let index = 0; index < count; index += 1) {
      buckets[index] = base + (remainder > 0 ? 1 : 0);
      if (remainder > 0) remainder -= 1;
    }
  } else {
    let maxWeight = 0;
    for (const index of positiveIndices) {
      maxWeight = Math.max(maxWeight, packed[index]!);
      buckets[index] = 1;
    }
    let relativeSum = 0;
    let compensation = 0;
    const relativeWeights = new Float64Array(count);
    for (const index of positiveIndices) {
      const relative = packed[index]! / maxWeight;
      relativeWeights[index] = relative;
      const corrected = relative - compensation;
      const next = relativeSum + corrected;
      compensation = (next - relativeSum) - corrected;
      relativeSum = next;
    }

    const available = EMITTER_CDF_F32_BUCKETS - positiveIndices.length;
    const remainders: Array<{ index: number; fraction: number }> = [];
    let allocated = positiveIndices.length;
    for (const index of positiveIndices) {
      const exactExtra = available * relativeWeights[index]! / relativeSum;
      const whole = Math.floor(exactExtra);
      buckets[index] = buckets[index]! + whole;
      allocated += whole;
      remainders.push({ index, fraction: exactExtra - whole });
    }
    if (allocated > EMITTER_CDF_F32_BUCKETS) {
      throw new RangeError('emitter CDF bucket allocation exceeded its exact f32 domain.');
    }
    remainders.sort((a, b) =>
      b.fraction - a.fraction || a.index - b.index
    );
    let remaining = EMITTER_CDF_F32_BUCKETS - allocated;
    for (let ordinal = 0; remaining > 0; ordinal += 1, remaining -= 1) {
      const target = remainders[ordinal % remainders.length]!;
      buckets[target.index] = buckets[target.index]! + 1;
    }
  }

  const cdf = new Float32Array(count);
  const representedPmf = new Float32Array(count);
  let cumulativeBuckets = 0;
  for (let index = 0; index < count; index += 1) {
    const intervalBuckets = buckets[index]!;
    cumulativeBuckets += intervalBuckets;
    representedPmf[index] = intervalBuckets / EMITTER_CDF_F32_BUCKETS;
    cdf[index] = cumulativeBuckets / EMITTER_CDF_F32_BUCKETS;
    if (packed[index]! > 0 && representedPmf[index] === 0) {
      throw new RangeError(
        `emitter CDF lost positive weight[${index}] during Float32 publication.`,
      );
    }
    if (index > 0 && packed[index]! > 0 && !(cdf[index]! > cdf[index - 1]!)) {
      throw new RangeError(
        `emitter CDF flattened positive weight[${index}] during Float32 publication.`,
      );
    }
  }
  if (cumulativeBuckets !== EMITTER_CDF_F32_BUCKETS || cdf[count - 1] !== 1) {
    throw new RangeError('emitter CDF failed to terminate at exactly one.');
  }
  return { cdf, representedPmf };
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
  return buildEmitterListCore(
    indices,
    positions,
    normals,
    (t) => {
      const mat = materials[triMatIdMap[t]!];
      if (!mat) return null;
      const classified = classifyTriangleEmitterCore(mat);
      const castShadowDisabled = (mat as MaterialSpec & { readonly castShadow?: boolean }).castShadow === false;
      const twoSided = mat.doubleSided === true;
      if (classified == null) return null;
      const scalarLe = scalarMaterialEmissiveLe(mat);
      if (scalarLe == null) return { ...classified, castShadowDisabled, twoSided };
      const sourceTriIndex = options.sourceTriIndexForTriangle?.(t) ?? t;
      const sourceTriIndexIsPackable =
        Number.isSafeInteger(sourceTriIndex) &&
        sourceTriIndex !== -1 &&
        Math.fround(sourceTriIndex) === sourceTriIndex;
      if (mat.emissiveMap != null) {
        if (!sourceTriIndexIsPackable) {
          throw new RangeError(
            `Mapped emitter triangle ${t} requires an exact f32-packable source triangle index.`,
          );
        }
        // Mapped emitters use a bounded-memory uniform-area conditional
        // proposal over the parent triangle. The candidate shader evaluates
        // the exact atlas texel (including authored transform, wrap, filter,
        // and mip state) at the sampled barycentrics; reservoir weights divide
        // by this proposal, so no texel-cell geometry expansion is required
        // for unbiased transport. `selectionColor` only improves the top-level
        // alias proposal and is never substituted for candidate radiance.
        return {
          ...classified,
          castShadowDisabled,
          twoSided,
          color: scalarLe,
          selectionColor: classified.color,
          sourceTriIndex,
        };
      }
      if (options.packSourceTriIndex !== true) {
        return { ...classified, castShadowDisabled, twoSided };
      }
      if (!sourceTriIndexIsPackable) {
        return { ...classified, castShadowDisabled, twoSided, color: scalarLe };
      }
      return {
        ...classified,
        castShadowDisabled,
        twoSided,
        color: scalarLe,
        sourceTriIndex,
      };
    },
    options,
  );
}

/**
 * A per-triangle emitter classifier: given a triangle index and its (already
 * area-weighted, normalized) world face normal, return the emitter
 * final folded `color` or `null` to skip. This is the ONLY material-typed step
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
  /** Radiance used for power-CDF / light-tree selection. */
  selectionColor?: [number, number, number];
  /** Valid atlas/BVH triangle id for candidate-time sampling, or absent for constant radiance. */
  sourceTriIndex?: number;
  /** Source primitive explicitly opted out of shadow casting. */
  castShadowDisabled?: boolean;
  /** Source material emits from both geometric orientations. */
  twoSided?: boolean;
} | null;

function scalarMaterialEmissiveLe(material: MaterialSpec): [number, number, number] | null {
  return materialSpecScalarEmissiveLe(material);
}


/**
 * Shared emitter-list builder core. Iterates the merged world-space triangle
 * stream, derives each triangle's area + face normal (cross-product, then
 * the per-vertex-normal-average override identical to the original
 * `buildEmitterListCore`), runs the supplied `classify` callback, gates on
 * positive finite power, then packs the 80-byte EmitterTri buffer + power CDF +
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
    castShadowDisabled: boolean;
    twoSided: boolean;
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
    castShadowDisabled: boolean;
    twoSided: boolean;
    selectionColor?: [number, number, number];
  }): void => {
    const color = packNonNegativeRadianceRgbF32(
      e.color,
      `emitter triangle ${e.triIdx} radiance`,
    );
    const powerColor = packNonNegativeRadianceRgbF32(
      e.selectionColor ?? color,
      `emitter triangle ${e.triIdx} selection radiance`,
    );
    const area = packNonNegativeRadianceScalarF32(
      e.area,
      `emitter triangle ${e.triIdx} area`,
    );
    const emittedLuminance = packNonNegativeRadianceScalarF32(
      luminance(powerColor[0], powerColor[1], powerColor[2]),
      `emitter triangle ${e.triIdx} luminance`,
    );
    const power = multiplyNonNegativeRadianceScalarsF32(
      emittedLuminance,
      area,
      `emitter triangle ${e.triIdx} luminance×area`,
    );
    if (!(power > 0)) return;
    emitterData.push({
      triIdx: e.triIdx,
      sourceTriIndex: e.sourceTriIndex,
      sourceSubdivLevel: e.sourceSubdivLevel ?? 1,
      sourceSubdivOrdinal: e.sourceSubdivOrdinal ?? 0,
      vA: e.vA,
      vB: e.vB,
      vC: e.vC,
      normal: e.normal,
      area,
      color,
      castShadowDisabled: e.castShadowDisabled,
      twoSided: e.twoSided,
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

    const parentA: [number, number, number] = [ax, ay, az];
    const parentB: [number, number, number] = [bx, by, bz];
    const parentC: [number, number, number] = [cx0, cy0, cz0];
    const areaMeasure = classifyTriangleAreaF32(parentA, parentB, parentC);
    if (!areaMeasure.valid) {
      if (areaMeasure.reason === 'degenerate') continue;
      throw new RangeError(
        `@vitrum/walkaround-hybrid: emitter triangle ${t} has ` +
        `${areaMeasure.reason.replaceAll('-', ' ')} geometry.`,
      );
    }
    const area = areaMeasure.area;
    let [nx, ny, nz] = areaMeasure.normal;
    const n0x = normals[i0 * 4]!;
    const n0y = normals[i0 * 4 + 1]!;
    const n0z = normals[i0 * 4 + 2]!;
    const hasNormals = (n0x !== 0 || n0y !== 0 || n0z !== 0);
    if (hasNormals) {
      const n1x = normals[i1 * 4]!, n1y = normals[i1 * 4 + 1]!, n1z = normals[i1 * 4 + 2]!;
      const n2x = normals[i2 * 4]!, n2y = normals[i2 * 4 + 1]!, n2z = normals[i2 * 4 + 2]!;
      const normalScale = Math.max(
        Math.abs(n0x), Math.abs(n0y), Math.abs(n0z),
        Math.abs(n1x), Math.abs(n1y), Math.abs(n1z),
        Math.abs(n2x), Math.abs(n2y), Math.abs(n2z),
      );
      if (normalScale > 0 && Number.isFinite(normalScale)) {
        const averaged = normalizeDirectionF32([
          n0x / normalScale + n1x / normalScale + n2x / normalScale,
          n0y / normalScale + n1y / normalScale + n2y / normalScale,
          n0z / normalScale + n1z / normalScale + n2z / normalScale,
        ]);
        if (averaged != null) [nx, ny, nz] = averaged;
      }
    }

    const classified = classify(t, { x: nx, y: ny, z: nz });
    if (!classified) continue;
    const [cr, cg, cb] = classified.color;
    const castShadowDisabled = classified.castShadowDisabled === true;
    const twoSided = classified.twoSided === true;

    const normal: [number, number, number] = [nx, ny, nz];
    const sourceTriIndex = classified.sourceTriIndex ?? -1;
    pushEmitter({
      triIdx: t,
      sourceTriIndex,
      vA: parentA,
      vB: parentB,
      vC: parentC,
      normal,
      area,
      color: [cr, cg, cb],
      castShadowDisabled,
      twoSided,
      ...(classified.selectionColor != null ? { selectionColor: classified.selectionColor } : {}),
    });
  }

  if (options.extraEmitters) {
    for (const ex of options.extraEmitters) {
      const geometryMeasure = classifyTriangleAreaF32(ex.vA, ex.vB, ex.vC);
      if (!geometryMeasure.valid) {
        if (geometryMeasure.reason === 'degenerate') continue;
        throw new RangeError(
          '@vitrum/walkaround-hybrid: extra emitter has ' +
          `${geometryMeasure.reason.replaceAll('-', ' ')} geometry.`,
        );
      }
      const publishedArea = Math.fround(ex.area);
      const area = geometryMeasure.area;
      const normal = normalizeDirectionF32(ex.normal);
      if (!(publishedArea > 0) || !Number.isFinite(publishedArea) || normal == null) {
        throw new RangeError(
          '@vitrum/walkaround-hybrid: extra emitter area and normal must retain ' +
          'finite non-degenerate Float32 values.',
        );
      }
      pushEmitter({
        triIdx: -1,
        sourceTriIndex: -1,
        sourceSubdivLevel: 1,
        sourceSubdivOrdinal: 0,
        vA: ex.vA, vB: ex.vB, vC: ex.vC,
        normal,
        area,
        color: packNonNegativeRadianceRgbF32(
          ex.Le,
          'extra emitter radiance',
        ),
        castShadowDisabled: ex.castShadow === false,
        twoSided: ex.twoSided === true,
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
      castShadowDisabled: false,
      twoSided: false,
      power: 0,          // power = 0 → excluded from CDF (totalEmissivePower = 0)
    });
  }

  const emitterCount = emitterData.length;
  const emitterFloats = new Float32Array(emitterCount * EMITTER_FLOATS);
  let totalEmissivePower = 0;

  // Light-tree build inputs, aligned 1:1 with the emitter list (leaf
  // emitterIndex == emitter array index). Powers are populated below from the
  // exact represented CDF intervals so every selection structure agrees.
  const treeInput: EmitterTreeInput = { powers: [], centroids: [], aabbs: [] };

  for (let i = 0; i < emitterCount; i++) {
    const e = emitterData[i]!;
    const base = i * EMITTER_FLOATS;
    const sourceTriIndex = options.packSourceTriIndex === true ? e.sourceTriIndex : -1;
    emitterFloats[base + 0] = e.vA[0]; emitterFloats[base + 1] = e.vA[1]; emitterFloats[base + 2] = e.vA[2]; emitterFloats[base + 3] = sourceTriIndex;
    emitterFloats[base + 4] = e.vB[0]; emitterFloats[base + 5] = e.vB[1]; emitterFloats[base + 6] = e.vB[2]; emitterFloats[base + 7] = sourceTriIndex !== -1 ? e.sourceSubdivLevel : 1;
    emitterFloats[base + 8] = e.vC[0]; emitterFloats[base + 9] = e.vC[1]; emitterFloats[base + 10] = e.vC[2]; emitterFloats[base + 11] = sourceTriIndex !== -1 ? e.sourceSubdivOrdinal : 0;
    emitterFloats[base + 12] = e.normal[0]; emitterFloats[base + 13] = e.normal[1]; emitterFloats[base + 14] = e.normal[2]; emitterFloats[base + 15] = e.area;
    emitterFloats[base + 16] = e.color[0]; emitterFloats[base + 17] = e.color[1]; emitterFloats[base + 18] = e.color[2];
    emitterFloats[base + 19] =
      (e.castShadowDisabled ? EMITTER_TRI_CAST_SHADOW_DISABLED_FLAG : 0) |
      (e.twoSided ? EMITTER_TRI_TWO_SIDED_FLAG : 0);
    totalEmissivePower += e.power;
    if (!Number.isFinite(totalEmissivePower)) {
      throw new RangeError('total emissive power exceeds the finite Number domain.');
    }

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

  // The light tree and alias table consume the same represented CDF intervals
  // rather than re-normalizing ideal binary64 weights. This keeps every
  // proposal's stored power finite and preserves support for arbitrarily dim
  // positive emitters beside much brighter ones.
  const represented = buildRepresentedEmitterCdfF32(
    emitterData.map((emitter) => emitter.power),
  );
  const cdfArray = represented.cdf;
  treeInput.powers.push(...represented.representedPmf);

  return { emitterFloats, cdfArray, totalEmissivePower, treeInput };
}
