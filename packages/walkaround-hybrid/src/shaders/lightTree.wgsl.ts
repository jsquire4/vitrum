/**
 * Light-tree importance sampling for ReSTIR-DI initial-candidate light SELECTION.
 *
 * Wires the CPU-built `@vitrum/shared-samplers` light tree (Shirley 1996 median
 * split, power-as-cost) into the GPU RIS candidate loop. The tree is serialised
 * by `packLightTreeForGPU` (12 f32 / node) and uploaded as a flat `array<f32>`
 * storage buffer at `@group(3) @binding(0)` — a RIS-ONLY bind group, separate
 * from the shared `scene` group, so the heavier shade pass (which already sits
 * at the `maxStorageBuffersPerShaderStage = 16` full-tier floor) is unaffected.
 *
 * Traversal (mirrors `sampleLightTreeCPU` in shared-samplers/src/lightTree.ts
 * branch-for-branch): descend from the root, at each internal node choosing a
 * child with probability proportional to its **importance**
 *   importance(child) = child.totalPower / max(dist²(x, childAABB), dist2Floor)
 * The returned `pdf` is the product of the branch probabilities along the
 * root→leaf path — a proper pmf over the emitter set (sums to 1). RIS uses this
 * pdf as the source pmf `p(emitter)` in the WRS weight `w = p̂ / p_source`, so
 * the estimator stays UNBIASED while importance-sampling near/bright lights.
 *
 * Gate: `ubo.lightTreeEnabled == 0u` (built with < 2 emitters, or the host
 * disabled it) ⇒ callers fall back to the flat power-CDF path (`sampleEmitterIdx`
 * + the `emitterPmf` weight) verbatim. When disabled the buffer is a single
 * zeroed placeholder node, never dereferenced (the caller branches on the gate
 * first).
 *
 * Layout per node (flat f32, stride 12), identical to `packLightTreeForGPU`:
 *   [0] emitterIndex (-1 internal)  [1] totalPower
 *   [2] leftChild (-1 leaf)         [3] rightChild (-1 leaf)
 *   [4..6] aabbMin.xyz              [7..9] aabbMax.xyz
 *   [10..11] padding
 *
 * References:
 *   - Estévez & Kulla 2018 — distance-weighted importance descent.
 *   - Shirley et al. 1996 — power-weighted light-list partition.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const LIGHT_TREE_WGSL = /* wgsl */ `// ============================================================
// Light-tree storage buffer (RIS-only @group(3)) + importance traversal
// ============================================================

// Flat f32 node array, 12 floats per node (see packLightTreeForGPU).
@group(3) @binding(0) var<storage, read> lightTree: array<f32>;

const LIGHT_TREE_STRIDE: u32 = 12u;

struct LightTreeSample {
  emitterIndex: i32,
  pdf:          f32,   // selection pmf of the chosen emitter (root→leaf product)
};

// Squared distance from point p to the AABB [bmin, bmax]; 0 inside.
fn lt_dist2ToAabb(p: vec3f, bmin: vec3f, bmax: vec3f) -> f32 {
  let d = max(max(bmin - p, vec3f(0.0)), p - bmax);
  return dot(d, d);
}

// Node importance for shading point p: power / max(dist², floor).
// dist2Floor is the SAME UBO floor the RIS geometry term uses so near-light
// selection and evaluation stay consistent (no divide-by-zero inside an AABB).
fn lt_importance(base: u32, p: vec3f, dist2Floor: f32) -> f32 {
  let power = lightTree[base + 1u];
  if (power <= 0.0) { return 0.0; }
  let bmin = vec3f(lightTree[base + 4u], lightTree[base + 5u], lightTree[base + 6u]);
  let bmax = vec3f(lightTree[base + 7u], lightTree[base + 8u], lightTree[base + 9u]);
  let d2 = max(lt_dist2ToAabb(p, bmin, bmax), dist2Floor);
  return power / d2;
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
    let sum = impL + impR;
    // Degenerate (both children zero importance): uniform 50/50 so the descent
    // terminates with a strictly-positive pdf (never an infinite RIS weight).
    let pL = select(0.5, impL / sum, sum > 0.0);
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
 * Include-graph entry. Requires `common` for `rand_f32` (PCG) + the
 * `WalkaroundUBO` struct; the RIS module pulls this in transitively.
 */
export const LIGHT_TREE_MODULE: WgslModule = {
  name: 'lightTree',
  source: LIGHT_TREE_WGSL,
  requires: ['common'],
};
