/**
 * lightTree.wgsl.ts — binding-agnostic GPU light-tree importance traversal.
 *
 * The canonical WGSL port of `sampleLightTreeCPU` (shared-samplers/lightTree.ts),
 * byte-for-byte in branch logic. Hoisted here so EVERY backend that imports the
 * CPU `buildLightTree` / `packLightTreeForGPU` consumes the SAME traversal — no
 * per-package copies (walkaround-hybrid ReSTIR-DI and pt-webgpu NEE both build
 * their light-tree WGSL from this one source).
 *
 * The `@group/@binding` of the flat node storage buffer differs per backend
 * (walkaround puts it at a RIS-only group(3); pt-webgpu also at group(3) but in a
 * different pipeline layout), so the binding declaration is PARAMETERISED via
 * `lightTreeWgsl({ group, binding, rngStateType })`. The default RNG state is
 * `u32`, matching the shared PCG module. Backends with an explicit state struct
 * pass its WGSL identifier through `rngStateType`; the matching `rand_f32`
 * implementation must already be in scope.
 *
 * Layout per node (flat f32, stride `LIGHT_TREE_FLOATS_PER_NODE` = 16, B8 grew
 * this from 12 to carry the orientation cone), identical to `packLightTreeForGPU`:
 *   [0] emitterIndex (-1 internal)  [1] totalPower
 *   [2] leftChild (-1 leaf)         [3] rightChild (-1 leaf)
 *   [4..6] aabbMin.xyz              [7..9] aabbMax.xyz
 *   [10..12] cone.axis.xyz          [13] cos(thetaO)   [14] cos(thetaO+thetaE)
 *   [15] padding
 *
 * The cone slots let `lt_importance` cull a node whose emitters point away from
 * the shading point (Conty-Estévez 2018 orientation term). A full-sphere node
 * (axis 0, both cosines = −1) reads as "no culling" ⇒ cone factor 1, recovering
 * the pre-B8 spatial-only descent byte-for-byte.
 *
 * References:
 *   - Conty Estévez & Kulla 2018 — distance-weighted importance descent + cone.
 *   - Shirley, Smits, Wang, Zimmerman 1996 — power-weighted light-list partition.
 */

import { LIGHT_TREE_FLOATS_PER_NODE } from '../lightTree.js';
import { requireInteger } from '../numericGuards.js';

/** Module name for include-graph consumers (walkaround's `WgslModule`). */
export const LIGHT_TREE_MODULE_NAME = 'lightTree';

/**
 * The binding-INDEPENDENT body: the `LightTreeSample` struct + the
 * `lt_dist2ToAabb` / `lt_importance` / `sampleLightTree` traversal. Assumes a
 * module-scope `lightTree: array<f32>` storage buffer is already declared (by
 * `lightTreeBindingWgsl`) and `rand_f32` is in scope.
 */
export const LIGHT_TREE_TRAVERSAL_WGSL = /* wgsl */ `
const LIGHT_TREE_STRIDE: u32 = 16u;

struct LightTreeSample {
  emitterIndex: i32,
  pdf:          f32,   // selection pmf of the chosen emitter (root→leaf product)
};

// Squared distance from point p to the AABB [bmin, bmax]; 0 inside.
fn lt_dist2ToAabb(p: vec3f, bmin: vec3f, bmax: vec3f) -> f32 {
  let d = max(max(bmin - p, vec3f(0.0)), p - bmax);
  return dot(d, d);
}

// Orientation-cone factor including the node AABB's conservative angular
// radius. Mirrors the CPU coneImportanceFactor branch-for-branch.
fn lt_coneFactor(
  axis: vec3f,
  cosThetaO: f32,
  cosThetaOE: f32,
  p: vec3f,
  c: vec3f,
  radius: f32,
) -> f32 {
  let axisScale = max(abs(axis.x), max(abs(axis.y), abs(axis.z)));
  if (!(axisScale > 0.0) || axisScale > 3.402823e38) {
    return 1.0; // unoriented / full sphere, or invalid input
  }
  let scaledAxis = axis / axisScale;
  let axisLength = length(scaledAxis);
  let dv = p - c;
  let distanceScale = max(abs(dv.x), max(abs(dv.y), abs(dv.z)));
  if (!(distanceScale > 0.0) || distanceScale > 3.402823e38) {
    return 1.0;
  }
  let scaledDistance = dv / distanceScale;
  let scaledDistanceLength = length(scaledDistance);
  let radiusOverDistance = (radius / distanceScale) / scaledDistanceLength;
  if (radiusOverDistance >= 1.0) { return 1.0; } // inside bounding sphere
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
  let sinAdjusted = sqrt(max(0.0, 1.0 - cosAdjusted * cosAdjusted));
  let sinThetaO = sqrt(max(0.0, 1.0 - cosThetaO * cosThetaO));
  return max(0.0, cosAdjusted * cosThetaO + sinAdjusted * sinThetaO);
}

// Node importance for shading point p: (power / max(dist², floor)) * coneFactor.
// dist2Floor is the SAME floor the RIS / NEE geometry term uses so near-light
// selection and evaluation stay consistent (no divide-by-zero inside an AABB).
fn lt_importance(base: u32, p: vec3f, dist2Floor: f32) -> f32 {
  let power = lightTree[base + 1u];
  if (power <= 0.0) { return 0.0; }
  let bmin = vec3f(lightTree[base + 4u], lightTree[base + 5u], lightTree[base + 6u]);
  let bmax = vec3f(lightTree[base + 7u], lightTree[base + 8u], lightTree[base + 9u]);
  let d2 = max(lt_dist2ToAabb(p, bmin, bmax), dist2Floor);
  let axis = vec3f(lightTree[base + 10u], lightTree[base + 11u], lightTree[base + 12u]);
  let cosThetaO  = lightTree[base + 13u];
  let cosThetaOE = lightTree[base + 14u];
  let center = 0.5 * (bmin + bmax);
  let radius = length(0.5 * (bmax - bmin));
  let coneFactor = lt_coneFactor(axis, cosThetaO, cosThetaOE, p, center, radius);
  let importance = (power / d2) * coneFactor;
  return min(importance, 3.402823466e38);
}

// Importance-sample one emitter (leaf) from the tree for shading point p.
// Returns the chosen emitterIndex + the selection pdf (root→leaf branch-product).
// Mirrors sampleLightTreeCPU in shared-samplers byte-for-byte.
fn sampleLightTree(p: vec3f, dist2Floor: f32, nodeCount: u32, rng: ptr<function, u32>) -> LightTreeSample {
  var nodeIdx: u32 = 0u;
  var pdf: f32 = 1.0;
  // Bounded descent: a binary tree over N leaves has depth ≤ N. The +1 guard
  // matches the CPU reference loop bound (WGSL forbids unbounded while).
  for (var guard: u32 = 0u; guard < nodeCount + 1u; guard = guard + 1u) {
    let base = nodeIdx * LIGHT_TREE_STRIDE;
    let leftChild  = i32(lightTree[base + 2u]);
    let rightChild = i32(lightTree[base + 3u]);
    if (leftChild < 0 || rightChild < 0) {
      // Leaf.
      var s: LightTreeSample;
      s.emitterIndex = i32(lightTree[base + 0u]);
      s.pdf = pdf;
      return s;
    }
    let lBase = u32(leftChild) * LIGHT_TREE_STRIDE;
    let rBase = u32(rightChild) * LIGHT_TREE_STRIDE;
    let impL = lt_importance(lBase, p, dist2Floor);
    let impR = lt_importance(rBase, p, dist2Floor);
    let scale = max(impL, impR);
    // Degenerate (both children zero importance): uniform 50/50 so the descent
    // terminates with a strictly-positive pdf (never an infinite RIS weight).
    var pL = 0.5;
    if (scale > 0.0) {
      pL = (impL / scale) / ((impL / scale) + (impR / scale));
    }
    if (rand_f32(rng) < pL) {
      pdf = pdf * pL;
      nodeIdx = u32(leftChild);
    } else {
      pdf = pdf * (1.0 - pL);
      nodeIdx = u32(rightChild);
    }
  }
  // Unreachable for a well-formed tree.
  let base = nodeIdx * LIGHT_TREE_STRIDE;
  var s: LightTreeSample;
  s.emitterIndex = i32(lightTree[base + 0u]);
  s.pdf = pdf;
  return s;
}
`;

/**
 * The `@group(group) @binding(binding) var<storage, read> lightTree` declaration.
 * Separate from the traversal body so a backend can place the buffer wherever its
 * pipeline layout has room.
 */
export function lightTreeBindingWgsl(group: number, binding: number): string {
  requireInteger(group, 'lightTreeBindingWgsl.group', 0, 65535);
  requireInteger(binding, 'lightTreeBindingWgsl.binding', 0, 65535);
  return `\n// Flat f32 node array, ${LIGHT_TREE_FLOATS_PER_NODE} floats per node (see packLightTreeForGPU).\n@group(${group}) @binding(${binding}) var<storage, read> lightTree: array<f32>;\n`;
}

/**
 * Full light-tree WGSL fragment: the binding declaration + the traversal body.
 * The matching `rand_f32` implementation must already be in scope. The optional `rngStateType` defaults to `u32`.
 */
export function lightTreeWgsl(opts: {
  readonly group: number;
  readonly binding: number;
  readonly rngStateType?: string;
}): string {
  const rngStateType = opts.rngStateType ?? 'u32';
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(rngStateType)) {
    throw new TypeError('lightTreeWgsl.rngStateType must be a WGSL identifier');
  }
  const traversal = rngStateType === 'u32'
    ? LIGHT_TREE_TRAVERSAL_WGSL
    : LIGHT_TREE_TRAVERSAL_WGSL.replace(
      'rng: ptr<function, u32>',
      `rng: ptr<function, ${rngStateType}>`,
    );
  return lightTreeBindingWgsl(opts.group, opts.binding) + traversal;
}
