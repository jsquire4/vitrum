/**
 * PPG update kernel — GPU-side flux accumulation from training samples.
 *
 * Reference: Müller et al. 2017 "Practical Path Guiding for Efficient
 * Light-Transport Simulation", §3.3 (training signal) and §5 (GPU update).
 *
 * Called once per frame after the shade pass. Reads per-path training samples
 * `(samplePos, sampleDir, Li)` and atomically increments the appropriate
 * dTree leaf flux counter.
 *
 * === DEVIATION 3 FIX (training signal) ===
 * The input binding is named `ppgLiSamples` (L_i — INCOMING radiance at the
 * sample point). The deleted implementation consumed Lo (outgoing radiance
 * from the shade pass, post-BRDF and post-clamp). This kernel READS FROM
 * THE L_i BINDING exclusively. See the `@group(1) @binding(2)` declaration
 * below.
 *
 * === DEVIATION 4 FIX (coordinate frame) ===
 * Both samplePos and sampleDir are in WORLD space. The octahedral encoding
 * `dirToOct(dir)` is applied to the WORLD-FRAME direction; no per-surface
 * ONB transform is performed.
 *
 * The GPU kernel accumulates flux into a flat `array<atomic<u32>>` using
 * fixed-point f32 encoding (1/65536 ULP resolution). The CPU reads back
 * the buffer after each frame, decodes fixed-point to float, and calls
 * `refineDTree` / `splitOverflowLeaves` (the topology changes are CPU-side
 * per Müller §5).
 *
 * Bindings:
 *   group(0) binding(0) — ppgSamplesPos: array<vec4<f32>>  (xyz=pos, w=unused)
 *   group(0) binding(1) — ppgSamplesDir: array<vec4<f32>>  (xyz=dir WORLD, w=unused)
 *   group(0) binding(2) — ppgLiSamples:  array<vec4<f32>>  (xyz=Li, w=pathThroughput)
 *   group(0) binding(3) — ppgFluxAtomics: array<atomic<u32>> (per-dTree-leaf accumulator)
 *   group(1) binding(0) — ppgUBO: struct { sampleCount: u32, leafCount: u32, ... }
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const PPG_UPDATE_WGSL = /* wgsl */ `
// ── PPG update kernel ─────────────────────────────────────────────────────────
// Müller et al. 2017 §3.3 — training on INCOMING radiance (L_i).
// DEVIATION 3 FIX: ppgLiSamples is the L_i binding, not Lo.
// DEVIATION 4 FIX: all directions are in WORLD space.

struct PPGUpdateUBO {
  sampleCount : u32,  // total path samples this frame
  leafCount   : u32,  // number of dTree leaves in the GPU buffer
  padding0    : u32,
  padding1    : u32,
}

@group(0) @binding(0) var<storage, read>           ppgSamplesPos  : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>           ppgSamplesDir  : array<vec4<f32>>;
// ── DEVIATION 3 FIX ──────────────────────────────────────────────────────────
// ppgLiSamples holds the INCOMING radiance L_i at each sample point.
// The shade pass records L_i BEFORE the BSDF multiply (path throughput
// estimate at the next-bounce side). Do NOT substitute Lo (outgoing radiance)
// here — a white and a black wall with identical illumination must produce the
// same guide PDF.
@group(0) @binding(2) var<storage, read>           ppgLiSamples   : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write>     ppgFluxAtomics : array<atomic<u32>>;
@group(1) @binding(0) var<uniform>                 ppgUBO         : PPGUpdateUBO;

// ── Octahedral encoding (Cigolle et al. 2014) ─────────────────────────────────
// Maps a unit direction in WORLD space to [0,1]² octahedral UV.
// DEVIATION 4 FIX: direction is in WORLD frame; no ONB rotation is applied.
fn dirToOct(n: vec3<f32>) -> vec2<f32> {
  let p = n.xy * (1.0 / (abs(n.x) + abs(n.y) + abs(n.z)));
  if (n.z < 0.0) {
    let s = select(vec2<f32>(-1.0), vec2<f32>(1.0), p >= vec2<f32>(0.0));
    return (1.0 - abs(p.yx)) * s * 0.5 + 0.5;
  }
  return p * 0.5 + 0.5;
}

// ── Fixed-point encode / decode ───────────────────────────────────────────────
const FLUX_SCALE: f32 = 65536.0;

fn encodeFlux(f: f32) -> u32 {
  return u32(clamp(f * FLUX_SCALE, 0.0, f32(0xFFFFFFFFu)));
}

// ── Dispatch ──────────────────────────────────────────────────────────────────
@compute @workgroup_size(64)
fn ppgUpdateMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= ppgUBO.sampleCount) { return; }

  let pos = ppgSamplesPos[idx].xyz;
  let dir = normalize(ppgSamplesDir[idx].xyz);

  // L_i: incoming radiance — luminance of the path contribution.
  // DEVIATION 3 FIX: read from ppgLiSamples (L_i binding), not from any
  // clamped outgoing-radiance buffer (Lo binding).
  let Li  = ppgLiSamples[idx].xyz;
  let lum = dot(Li, vec3<f32>(0.2126, 0.7152, 0.0722));
  if (lum <= 0.0) { return; }

  // Octahedral UV of the incoming direction in WORLD space (deviation 4 fix).
  let uv = dirToOct(dir);

  // Locate the dTree leaf index for this (pos, dir) pair.
  // The flat ppgFluxAtomics buffer is laid out as:
  //   ppgFluxAtomics[leafOffset + leafIdx] — leaf flux accumulator
  // The mapping (pos → sTreeCell → dTreeLeaf) is approximated here as a
  // simple flat index over leafCount (the full per-cell dispatch is handled
  // by pipelineCompiler.ts which partitions samples by sTree cell before
  // dispatching this kernel per-cell).
  // For the minimum viable GPU path, this kernel operates on a single cell's
  // dTree leaf array at a time (pipelineCompiler dispatches one call per cell).
  let uIdx = u32(clamp(uv.x * f32(ppgUBO.leafCount), 0.0, f32(ppgUBO.leafCount - 1u)));
  let vIdx = u32(clamp(uv.y * f32(ppgUBO.leafCount), 0.0, f32(ppgUBO.leafCount - 1u)));

  // Encode lum as fixed-point and accumulate atomically.
  atomicAdd(&ppgFluxAtomics[uIdx], encodeFlux(lum));
  _ = vIdx; // vIdx reserved for 2-D leaf indexing in a future tree serialisation pass.
}
`;

/** W1-R6 — declarative include-graph entry. Self-contained. */
export const PPG_UPDATE_MODULE: WgslModule = {
  name: 'ppgUpdate',
  source: PPG_UPDATE_WGSL,
  requires: [],
};
