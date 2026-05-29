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
 * `lightTreeWgsl({ group, binding })`. The traversal functions reference the
 * `rand_f32(ptr<function,u32>)` PCG primitive from `@vitrum/shared-samplers`'
 * `PCG_WGSL` (both backends already include it), so the caller must concatenate
 * `PCG_WGSL` (or `requires:['common']` in walkaround's include graph) earlier.
 *
 * Layout per node (flat f32, stride `LIGHT_TREE_FLOATS_PER_NODE` = 12), identical
 * to `packLightTreeForGPU`:
 *   [0] emitterIndex (-1 internal)  [1] totalPower
 *   [2] leftChild (-1 leaf)         [3] rightChild (-1 leaf)
 *   [4..6] aabbMin.xyz              [7..9] aabbMax.xyz
 *   [10..11] padding
 *
 * References:
 *   - Conty Estévez & Kulla 2018 — distance-weighted importance descent.
 *   - Shirley, Smits, Wang, Zimmerman 1996 — power-weighted light-list partition.
 */

/** Module name for include-graph consumers (walkaround's `WgslModule`). */
export const LIGHT_TREE_MODULE_NAME = 'lightTree';

/**
 * The binding-INDEPENDENT body: the `LightTreeSample` struct + the
 * `lt_dist2ToAabb` / `lt_importance` / `sampleLightTree` traversal. Assumes a
 * module-scope `lightTree: array<f32>` storage buffer is already declared (by
 * `lightTreeBindingWgsl`) and `rand_f32` is in scope.
 */
export const LIGHT_TREE_TRAVERSAL_WGSL = /* wgsl */ `
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
// dist2Floor is the SAME floor the RIS / NEE geometry term uses so near-light
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
 * The `@group(group) @binding(binding) var<storage, read> lightTree` declaration.
 * Separate from the traversal body so a backend can place the buffer wherever its
 * pipeline layout has room.
 */
export function lightTreeBindingWgsl(group: number, binding: number): string {
  return `\n// Flat f32 node array, ${'12'} floats per node (see packLightTreeForGPU).\n@group(${group}) @binding(${binding}) var<storage, read> lightTree: array<f32>;\n`;
}

/**
 * Full light-tree WGSL fragment: the binding declaration + the traversal body.
 * `rand_f32` (PCG) must already be in scope (concatenate `PCG_WGSL` earlier).
 */
export function lightTreeWgsl(opts: { readonly group: number; readonly binding: number }): string {
  return lightTreeBindingWgsl(opts.group, opts.binding) + LIGHT_TREE_TRAVERSAL_WGSL;
}
