/**
 * Light-tree importance sampling for ReSTIR-DI initial-candidate light SELECTION.
 *
 * Wires the CPU-built `@vitrum/shared-samplers` light tree (Shirley 1996 median
 * split, power-as-cost) into the GPU RIS candidate loop. The tree is serialised
 * by `packLightTreeForGPU` (16 f32 / node — B8 grew it from 12 for the
 * orientation cone) and uploaded as a flat `array<f32>`
 * storage buffer at `@group(3) @binding(0)` — a RIS-ONLY bind group, separate
 * from the shared `scene` group, so the heavier shade pass (which already sits
 * at the guaranteed `maxStorageBuffersPerShaderStage = 8` floor) is unaffected.
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
 * Layout per node (flat f32, stride 16), identical to `packLightTreeForGPU`:
 *   [0] emitterIndex (-1 internal)  [1] totalPower
 *   [2] leftChild (-1 leaf)         [3] rightChild (-1 leaf)
 *   [4..6] aabbMin.xyz              [7..9] aabbMax.xyz
 *   [10..12] cone.axis.xyz          [13] cos(thetaO)  [14] cos(thetaO+thetaE)
 *   [15] padding
 *
 * References:
 *   - Estévez & Kulla 2018 — distance-weighted importance descent.
 *   - Shirley et al. 1996 — power-weighted light-list partition.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';
import { lightTreeWgsl } from '@vitrum/shared-samplers';

// RIS-only @group(3) binding(0) (separate from the shared `scene` group at the
// eight-storage-buffer full-tier floor). Body is the canonical traversal hoisted to
// `@vitrum/shared-samplers/wgsl/lightTree.wgsl.ts` — single source of truth across
// walkaround-hybrid + pt-webgpu (no per-package copies).
export const LIGHT_TREE_WGSL = /* wgsl */ `// ============================================================
// Light-tree storage buffer (RIS-only @group(3)) + importance traversal
// ============================================================
${lightTreeWgsl({ group: 3, binding: 0 })}
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
