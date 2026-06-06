/**
 * PPG tree layout constants — shared by ppgUpdate and ppgPdf.
 *
 * These constants mirror the flat-buffer serialisation layout produced by
 * `serialise.ts`. If the serialiser changes these values, all three shader
 * modules must be updated in lock-step.
 *
 * Reference: Müller et al. 2017 "Practical Path Guiding for Efficient
 * Light-Transport Simulation", §3.1 (sTree layout) and §3.2 (dTree layout).
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const PPG_TREE_LAYOUT_WGSL = /* wgsl */ `
// ── PPG tree layout constants — MUST stay in sync with serialise.ts ──────────
// dTree flat-buffer header size and per-node stride (f32 elements).
const DTREE_HEADER_F32 : u32 = 4u;
const DTREE_NODE_STRIDE: u32 = 8u;
// sTree flat-buffer header size and per-node stride (f32 elements).
const STREE_HEADER_F32 : u32 = 4u;
const STREE_NODE_STRIDE: u32 = 16u;
`;

/** PPG tree layout constants — required by ppgUpdate and ppgPdf. */
export const PPG_TREE_LAYOUT_MODULE: WgslModule = {
  name: 'ppgTreeLayout',
  source: PPG_TREE_LAYOUT_WGSL,
  requires: [],
};
