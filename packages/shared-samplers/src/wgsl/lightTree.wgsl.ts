/**
 * lightTree.wgsl.ts — binding-agnostic GPU light-tree importance traversal.
 *
 * This is the canonical WGSL port of the represented 24-bit bucket proposal in
 * `shared-samplers/lightTree.ts`. Both path tracers and ReGIR instantiate this
 * generator, so distance, cone, saturation, support, and PDF semantics cannot
 * drift between backends.
 *
 * Packed node layout (16 f32 lanes):
 *   [0] emitter index                [1] represented proposal power
 *   [2] left child                   [3] right child
 *   [4..6] AABB min                  [7..9] AABB max
 *   [10..12] cone axis               [13] cos(thetaO)
 *   [14] cos(thetaO + thetaE)        [15] exact subtree leaf count
 *
 * Positive raw powers are normalized by a common maximum and floored to the
 * smallest normal f32 before lane 1 is published. Lane 15 lets traversal
 * reserve one of the 2^24 root buckets for every leaf. The returned PMF is the
 * leaf's exact bucket count / 2^24, rather than a deep product that can round to
 * zero.
 */

import {
  LIGHT_TREE_BUCKET_COUNT,
  LIGHT_TREE_FLOATS_PER_NODE,
} from '../lightTree.js';
import { requireInteger } from '../numericGuards.js';

/** Module name for include-graph consumers (walkaround's `WgslModule`). */
export const LIGHT_TREE_MODULE_NAME = 'lightTree';

export interface LightTreeTraversalWgslOptions {
  /** Flat `array<f32>` storage variable containing the packed tree. */
  readonly storageVariable?: string;
  /** Prefix for private helper functions/constants. */
  readonly helperPrefix?: string;
  readonly strideConstantName?: string;
  readonly sampleStructName?: string;
  readonly sampleFunctionName?: string;
  readonly rngStateType?: string;
}

function wgslIdentifier(value: string, label: string): string {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) {
    throw new TypeError(`${label} must be a WGSL identifier`);
  }
  return value;
}

/**
 * Generate the binding-independent traversal body. The selected storage
 * variable and `rand_f32(ptr<function, rngStateType>)` must already be in scope.
 * `nodeIndex` is returned with the leaf so ReGIR can evaluate qHat through the
 * exact same canonical importance helper.
 */
export function lightTreeTraversalWgsl(
  opts: LightTreeTraversalWgslOptions = {},
): string {
  const storage = wgslIdentifier(
    opts.storageVariable ?? 'lightTree',
    'lightTreeTraversalWgsl.storageVariable',
  );
  const prefix = wgslIdentifier(
    opts.helperPrefix ?? 'lt',
    'lightTreeTraversalWgsl.helperPrefix',
  );
  const stride = wgslIdentifier(
    opts.strideConstantName ?? 'LIGHT_TREE_STRIDE',
    'lightTreeTraversalWgsl.strideConstantName',
  );
  const sampleStruct = wgslIdentifier(
    opts.sampleStructName ?? 'LightTreeSample',
    'lightTreeTraversalWgsl.sampleStructName',
  );
  const sampleFunction = wgslIdentifier(
    opts.sampleFunctionName ?? 'sampleLightTree',
    'lightTreeTraversalWgsl.sampleFunctionName',
  );
  const rngStateType = wgslIdentifier(
    opts.rngStateType ?? 'u32',
    'lightTreeTraversalWgsl.rngStateType',
  );
  const dist2ToAabb = `${prefix}_dist2ToAabb`;
  const coneFactor = `${prefix}_coneFactor`;
  const importance = `${prefix}_importance`;
  const pairFirst = `${prefix}_pairFirst`;
  const leftBucketCount = `${prefix}_leftBucketCount`;
  const f32Max = `${prefix}_F32_MAX`;
  const f32MinNormal = `${prefix}_F32_MIN_NORMAL`;
  const rootBuckets = `${prefix}_ROOT_BUCKETS`;

  return /* wgsl */ `
const ${stride}: u32 = ${LIGHT_TREE_FLOATS_PER_NODE}u;
const ${rootBuckets}: u32 = ${LIGHT_TREE_BUCKET_COUNT}u;
const ${f32Max}: f32 = 3.402823466e38;
const ${f32MinNormal}: f32 = 1.175494351e-38;

struct ${sampleStruct} {
  emitterIndex: i32,
  pdf:          f32,
  nodeIndex:    u32,
};

// Squared point/AABB distance, saturated before any f32 intermediate can
// overflow. Half-coordinate differences remain finite for opposite extrema.
fn ${dist2ToAabb}(p: vec3f, bmin: vec3f, bmax: vec3f) -> f32 {
  let halfD = max(
    max(0.5 * bmin - 0.5 * p, vec3f(0.0)),
    0.5 * p - 0.5 * bmax,
  );
  let halfScale = max(abs(halfD.x), max(abs(halfD.y), abs(halfD.z)));
  if (!(halfScale >= ${f32MinNormal})) { return 0.0; }
  if (halfScale > 1.701411733e38) { return ${f32Max}; }
  let scaled = halfD / halfScale;
  let scaledSquared = dot(scaled, scaled);
  let actualScale = 2.0 * halfScale;
  if (actualScale >= sqrt(${f32Max} / scaledSquared)) { return ${f32Max}; }
  return min((actualScale * actualScale) * scaledSquared, ${f32Max});
}

// Conty-Estévez orientation factor including the AABB's conservative angular
// radius. All geometry uses scaled vectors; no raw length or centre subtraction
// can overflow.
fn ${coneFactor}(
  axis: vec3f,
  cosThetaO: f32,
  cosThetaOE: f32,
  p: vec3f,
  bmin: vec3f,
  bmax: vec3f,
) -> f32 {
  let axisScale = max(abs(axis.x), max(abs(axis.y), abs(axis.z)));
  if (!(axisScale > 0.0) || axisScale > ${f32Max}) {
    return 1.0;
  }
  let scaledAxis = axis / axisScale;
  let axisLength = length(scaledAxis);

  // dvHalf points from the AABB centre to p, scaled by one half. radiusHalf
  // has the same scale, so their ratio and direction are unchanged.
  let dvHalf = 0.5 * p - 0.25 * bmin - 0.25 * bmax;
  let radiusVectorHalf = 0.25 * bmax - 0.25 * bmin;
  let distanceScale = max(abs(dvHalf.x), max(abs(dvHalf.y), abs(dvHalf.z)));
  if (!(distanceScale >= ${f32MinNormal}) || distanceScale > ${f32Max}) {
    return 1.0;
  }
  let scaledDistance = dvHalf / distanceScale;
  let scaledDistanceLength = length(scaledDistance);

  let radiusScale = max(
    abs(radiusVectorHalf.x),
    max(abs(radiusVectorHalf.y), abs(radiusVectorHalf.z)),
  );
  var radiusOverDistance = 0.0;
  if (radiusScale >= ${f32MinNormal}) {
    let scaledRadiusLength = length(radiusVectorHalf / radiusScale);
    let logRatio =
      log2(radiusScale) + log2(scaledRadiusLength) -
      log2(distanceScale) - log2(scaledDistanceLength);
    if (logRatio >= 0.0) { return 1.0; }
    radiusOverDistance = exp2(max(logRatio, -126.0));
  }

  let d = scaledDistance / scaledDistanceLength;
  let a = scaledAxis / axisLength;
  let cosTheta = clamp(dot(a, d), -1.0, 1.0);
  let sinThetaU = clamp(radiusOverDistance, 0.0, 1.0);
  let cosThetaU = sqrt(max(0.0, 1.0 - sinThetaU * sinThetaU));
  var cosAdjusted = 1.0;
  if (cosTheta < cosThetaU) {
    let sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    cosAdjusted = clamp(
      cosTheta * cosThetaU + sinTheta * sinThetaU,
      -1.0,
      1.0,
    );
  }
  if (cosAdjusted < cosThetaOE) { return 0.0; }
  if (cosAdjusted >= cosThetaO) { return 1.0; }
  let cosineSpan = cosThetaO - cosThetaOE;
  if (!(cosineSpan > 0.0)) { return 1.0; }
  return clamp(
    (cosAdjusted - cosThetaOE) / cosineSpan,
    ${f32MinNormal},
    1.0,
  );
}

// Overflow-safe, FTZ-safe represented importance. A positive power with a
// positive cone factor returns a normal positive f32; only true zero support is
// encoded as zero.
fn ${importance}(base: u32, p: vec3f, dist2Floor: f32) -> f32 {
  let power = ${storage}[base + 1u];
  if (!(power > 0.0)) { return 0.0; }
  let bmin = vec3f(${storage}[base + 4u], ${storage}[base + 5u], ${storage}[base + 6u]);
  let bmax = vec3f(${storage}[base + 7u], ${storage}[base + 8u], ${storage}[base + 9u]);
  let d2 = max(max(${dist2ToAabb}(p, bmin, bmax), dist2Floor), ${f32MinNormal});
  let axis = vec3f(${storage}[base + 10u], ${storage}[base + 11u], ${storage}[base + 12u]);
  let cone = ${coneFactor}(
    axis,
    ${storage}[base + 13u],
    ${storage}[base + 14u],
    p,
    bmin,
    bmax,
  );
  if (!(cone > 0.0)) { return 0.0; }
  let logImportance =
    log2(power) - log2(d2) + log2(max(cone, ${f32MinNormal}));
  if (logImportance >= 127.99999) { return ${f32Max}; }
  if (logImportance <= -126.0) { return ${f32MinNormal}; }
  return clamp(exp2(logImportance), ${f32MinNormal}, ${f32Max});
}

fn ${pairFirst}(left: f32, right: f32) -> f32 {
  let scale = max(left, right);
  if (!(scale > 0.0)) { return 0.5; }
  let l = left / scale;
  let r = right / scale;
  return l / (l + r);
}

// Reserve one bucket for every descendant leaf, then assign all spare buckets
// to the represented importance target. The child counts sum exactly to the
// parent's interval and neither positive subtree can disappear.
fn ${leftBucketCount}(
  currentBuckets: u32,
  leftLeaves: u32,
  rightLeaves: u32,
  leftImportance: f32,
  rightImportance: f32,
) -> u32 {
  let leafCount = leftLeaves + rightLeaves;
  let remaining = currentBuckets - leafCount;
  let pLeft = ${pairFirst}(leftImportance, rightImportance);
  let roundedExtra = floor(f32(remaining) * pLeft + 0.5);
  let leftExtra = min(remaining, u32(max(0.0, roundedExtra)));
  return leftLeaves + leftExtra;
}

fn ${sampleFunction}(
  p: vec3f,
  dist2Floor: f32,
  nodeCount: u32,
  rng: ptr<function, ${rngStateType}>,
) -> ${sampleStruct} {
  var nodeIdx = 0u;
  var currentBuckets = ${rootBuckets};
  if (nodeCount == 0u) {
    var invalid: ${sampleStruct};
    invalid.emitterIndex = -1;
    invalid.pdf = 0.0;
    invalid.nodeIndex = 0u;
    return invalid;
  }
  // Both supported RNG adapters map their high 24 bits to [0,1), so this is an
  // exact root-bucket draw and consumes only one RNG value for the whole tree.
  // A one-leaf tree consumes no RNG value, matching the CPU oracle and the old
  // totality contract.
  var localBucket = 0u;
  let rootLeft = i32(${storage}[2u]);
  let rootRight = i32(${storage}[3u]);
  if (rootLeft >= 0 && rootRight >= 0) {
    localBucket = min(
      ${rootBuckets} - 1u,
      u32(floor(rand_f32(rng) * f32(${rootBuckets}))),
    );
  }
  // packLightTreeForGPU caps nodeCount below 2^24, so nodeCount + 1u cannot
  // overflow the u32 loop bound.
  for (var guard = 0u; guard < nodeCount + 1u; guard = guard + 1u) {
    let base = nodeIdx * ${stride};
    let leftChild = i32(${storage}[base + 2u]);
    let rightChild = i32(${storage}[base + 3u]);
    if (leftChild < 0 || rightChild < 0) {
      var result: ${sampleStruct};
      result.emitterIndex = i32(${storage}[base + 0u]);
      result.pdf = f32(currentBuckets) / f32(${rootBuckets});
      result.nodeIndex = nodeIdx;
      return result;
    }
    let leftBase = u32(leftChild) * ${stride};
    let rightBase = u32(rightChild) * ${stride};
    let leftBuckets = ${leftBucketCount}(
      currentBuckets,
      u32(${storage}[leftBase + 15u]),
      u32(${storage}[rightBase + 15u]),
      ${importance}(leftBase, p, dist2Floor),
      ${importance}(rightBase, p, dist2Floor),
    );
    if (localBucket < leftBuckets) {
      currentBuckets = leftBuckets;
      nodeIdx = u32(leftChild);
    } else {
      localBucket = localBucket - leftBuckets;
      currentBuckets = currentBuckets - leftBuckets;
      nodeIdx = u32(rightChild);
    }
  }
  let base = nodeIdx * ${stride};
  var result: ${sampleStruct};
  result.emitterIndex = i32(${storage}[base + 0u]);
  result.pdf = f32(currentBuckets) / f32(${rootBuckets});
  result.nodeIndex = nodeIdx;
  return result;
}
`;
}

/** Default traversal used by pt-webgpu and walkaround ReSTIR-DI. */
export const LIGHT_TREE_TRAVERSAL_WGSL = lightTreeTraversalWgsl();

/** Storage binding declaration, kept separate for include-graph consumers. */
export function lightTreeBindingWgsl(group: number, binding: number): string {
  requireInteger(group, 'lightTreeBindingWgsl.group', 0, 65535);
  requireInteger(binding, 'lightTreeBindingWgsl.binding', 0, 65535);
  return `\n// Flat f32 node array, ${LIGHT_TREE_FLOATS_PER_NODE} floats per node (see packLightTreeForGPU).\n@group(${group}) @binding(${binding}) var<storage, read> lightTree: array<f32>;\n`;
}

/** Full binding + canonical traversal fragment. */
export function lightTreeWgsl(opts: {
  readonly group: number;
  readonly binding: number;
  readonly rngStateType?: string;
}): string {
  return lightTreeBindingWgsl(opts.group, opts.binding) + lightTreeTraversalWgsl(
    opts.rngStateType == null ? {} : { rngStateType: opts.rngStateType },
  );
}
