/**
 * PPG pdf-eval + guided-sampling helpers for the gi-ris RIS source pdf.
 *
 * Reference: Müller et al. 2017 "Practical Path Guiding for Efficient
 * Light-Transport Simulation", §3.2 (dTree sampling), §3.4 (MIS mixture).
 *
 * === Why this module exists ===
 * gi-ris compiles against the shared hybrid pipeline layout
 * (frame/scene/ubo/hybridLayers), so the guide sampler must read the learned
 * PPG trees from one packed query arena in the `hybridLayers` group (group 3,
 * binding 6 — there because the adapter caps `maxBindGroups = 4`) and provides:
 *
 *   1. `ppgEvalPdf(pos, wi)`        — the SOLID-ANGLE guide pdf p_guide(ωi) for
 *                                     an ARBITRARY world direction. Mirrors the
 *                                     CPU `dTreePdf` (dTree.ts) EXACTLY so the
 *                                     defensive MIS weight stays unbiased.
 *   2. `ppgSampleGuidedDir(pos,&rng)` — draw a world direction from the learned
 *                                     dTree (flux-proportional descent + leaf
 *                                     jitter), mirroring the CPU dTree sampler.
 *
 * Both descend the SAME serialised layout the producer (`serialise.ts`) and the
 * training kernel (`ppgUpdate.wgsl`) use — if the layout constants change there,
 * they must change here in lock-step.
 *
 * === Directional parametrisation (cylindrical equal-area) ===
 * The dTree stores UV in [0,1]². A world direction maps to that UV via the
 * cylindrical EQUAL-AREA map `ppgDirToUv` (inline below; u = (1−z)/2 uniform in
 * z, v = azimuth/2π) — the parametrisation Müller 2017 §3.2 actually uses, so
 * the dTree's `solidAngle = 4π·uvArea` is exact and the guide pdf equals the
 * uniform-in-UV sampling density (unbiased). The 2026-06-09 fix replaced the
 * earlier non-equal-area Cigolle octahedral map (which dropped a varying
 * Jacobian → biased MIS source pdf). Train (`ppgUpdate.wgsl`), pdf, and the
 * sampler (`ppgUvToDir`) all share this SAME map in lock-step. No octahedralCore
 * dependency: the shared octEncode/octDecode are no longer referenced here.
 *
 * === Bindings (group 3 = hybridLayers — see bindGroupLayouts.ts) ===
 *   @group(3) @binding(6) ppgQueryArena_gi : array<u32>
 *     (header + serialised sTree + concatenated dTrees + cell offsets)
 *
 * These are gated: gi-ris only calls into this module when `ubo.ppgEnabled == 1`,
 * so the 16-byte placeholders bound when PPG is off are never dereferenced.
 */

import type { WgslModule } from '../wgslTypes.js';
import {
  PPG_QUERY_ARENA_MAGIC,
  PPG_QUERY_ARENA_SCHEMA,
  PPG_QUERY_ARENA_VERSION,
} from './ppgQueryArena.js';

export const PPG_PDF_WGSL = /* wgsl */ `
// ── PPG guided-sampling + pdf-eval (gi-ris) ─────────────────────────────────
// Müller 2017 §3.2/§3.4. Reads the learned sTree/dTree from group(3).

@group(3) @binding(6) var<storage, read> ppgQueryArena_gi : array<u32>;

const PPG_QUERY_MAGIC_GI: u32 = ${PPG_QUERY_ARENA_MAGIC}u;
const PPG_QUERY_VERSION_GI: u32 = ${PPG_QUERY_ARENA_VERSION}u;
const PPG_QUERY_SCHEMA_GI: u32 = ${PPG_QUERY_ARENA_SCHEMA}u;
fn ppgQueryArenaValidGi() -> bool {
  return ppgQueryArena_gi[0] == PPG_QUERY_MAGIC_GI &&
    ppgQueryArena_gi[1] == PPG_QUERY_VERSION_GI &&
    ppgQueryArena_gi[2] != 0u &&
    ppgQueryArena_gi[3] == PPG_QUERY_SCHEMA_GI;
}
fn ppgArenaLoadSTreeF32(word: u32) -> f32 {
  return bitcast<f32>(ppgQueryArena_gi[ppgQueryArena_gi[4] + word]);
}
fn ppgArenaLoadDTreeF32(word: u32) -> f32 {
  return bitcast<f32>(ppgQueryArena_gi[ppgQueryArena_gi[7] + word]);
}
fn ppgArenaLoadDTreeOffset(word: u32) -> u32 {
  return ppgQueryArena_gi[ppgQueryArena_gi[10] + word];
}

// Layout constants provided by ppgTreeLayout (DTREE_HEADER_F32, DTREE_NODE_STRIDE,
// STREE_HEADER_F32, STREE_NODE_STRIDE — shared with ppgUpdate).
const PPG_FOUR_PI: f32 = 12.566370614359172; // 4π — uniform-fallback pdf = 1/4π

// ── World direction ↔ dTree [0,1]² UV (cylindrical EQUAL-AREA, Müller 2017 §3.2) ──
// FIX 2026-06-09: this used octEncode (Cigolle 2014, a NON-equal-area octahedral
// map) while dTree.ts stores solidAngle = 4π·uvArea ASSUMING equal-area. So the
// guide pdf (leafFlux/totalFlux)/solidAngle did NOT equal the uniform-in-UV
// sampling density (it dropped the octahedral Jacobian) → the gi-ris MIS source
// pdf was biased → guided GI GAINED energy, growing as the dTree refined into the
// distorted diagonal regions (g-p11: PPG-on 2.36× over-bright, Δ grows with frames).
// The cylindrical map (u = (1-z)/2 uniform in z — Archimedes' hat-box) IS equal-area,
// so solidAngle = 4π·uvArea is exact and uniform-in-UV = uniform-in-solid-angle.
// This is the parametrisation Müller's paper actually uses. Train (ppgUpdate.wgsl),
// pdf, and the sampler below MUST all use this SAME map — keep them in lock-step.
fn ppgDirToUv(dir: vec3<f32>) -> vec2<f32> {
  let u = (1.0 - clamp(dir.z, -1.0, 1.0)) * 0.5;            // z∈[+1,−1] → u∈[0,1]
  let v = atan2(dir.y, dir.x) * 0.15915494309189535 + 0.5; // azimuth/(2π) + 0.5 → [0,1]
  return vec2<f32>(u, clamp(v, 0.0, 1.0));
}
fn ppgUvToDir(uv: vec2<f32>) -> vec3<f32> {
  let z = 1.0 - 2.0 * uv.x;
  let r = sqrt(max(0.0, 1.0 - z * z));
  let phi = 6.283185307179586 * (uv.y - 0.5);              // inverse of ppgDirToUv azimuth
  return vec3<f32>(r * cos(phi), r * sin(phi), z);
}

// ── sTree descent (mirror of serialise.gpuTraverseSTreeLeaf) ────────────────
// Returns the f32 base offset of the sNode whose AABB contains 'pos'.
// MUST-MATCH: this descent body is semantically identical to sTreeFindLeafBase
// in ppgUpdate.wgsl.ts — only the buffer name differs (ppgSTreeBuf_gi here vs
// ppgSTreeBuf there). If you edit the logic here, mirror the change there,
// and vice versa. The ppgDescentDrift vitest gate enforces this automatically.
fn ppgSTreeFindLeafBase(pos: vec3<f32>) -> u32 {
  let nodeCount = u32(ppgArenaLoadSTreeF32(0));
  var idx: u32 = 0u;
  // sTree depth ≤ log2(16384) = 14; 32 is a generous safety cap.
  for (var step: u32 = 0u; step < 32u; step = step + 1u) {
    let base = STREE_HEADER_F32 + idx * STREE_NODE_STRIDE;
    let splitAxisF = ppgArenaLoadSTreeF32(base + 7u);
    if (splitAxisF < 0.0) { return base; } // leaf
    let splitVal = ppgArenaLoadSTreeF32(base + 3u);
    let leftChildF  = ppgArenaLoadSTreeF32(base + 8u);
    let rightChildF = ppgArenaLoadSTreeF32(base + 9u);
    let axis = u32(splitAxisF);
    var queryAxis: f32 = 0.0;
    if (axis == 0u)      { queryAxis = pos.x; }
    else if (axis == 1u) { queryAxis = pos.y; }
    else                 { queryAxis = pos.z; }
    if (queryAxis < splitVal) { idx = u32(leftChildF); }
    else                      { idx = u32(rightChildF); }
    if (idx >= nodeCount) { return base; } // defensive fallback
  }
  return STREE_HEADER_F32;
}

// ── dTree descent to the leaf containing an arbitrary UV ────────────────────
// Mirror of dTree.findDTreeLeaf: at each interior node, descend the quadrant
// of octUV. Returns the leaf's f32 base offset within ppgDTreeBuf_gi.
// MUST-MATCH: this descent body is semantically identical to dTreeFindLeafBase
// in ppgUpdate.wgsl.ts — only the buffer name differs (ppgDTreeBuf_gi here vs
// ppgDTreeBuf there). If you edit the logic here, mirror the change there,
// and vice versa. The ppgDescentDrift vitest gate enforces this automatically.
fn ppgDTreeFindLeafBase(dTreeOffset: u32, octUV: vec2<f32>) -> u32 {
  var idx: u32 = 0u;
  for (var step: u32 = 0u; step < 32u; step = step + 1u) {
    let base = dTreeOffset + DTREE_HEADER_F32 + idx * DTREE_NODE_STRIDE;
    let isLeafFlag = ppgArenaLoadDTreeF32(base + 7u);
    if (isLeafFlag > 0.5) { return base; }
    let u0 = ppgArenaLoadDTreeF32(base + 0u);
    let v0 = ppgArenaLoadDTreeF32(base + 1u);
    let u1 = ppgArenaLoadDTreeF32(base + 2u);
    let v1 = ppgArenaLoadDTreeF32(base + 3u);
    let uMid = (u0 + u1) * 0.5;
    let vMid = (v0 + v1) * 0.5;
    let firstChildF = ppgArenaLoadDTreeF32(base + 6u);
    if (firstChildF < 0.0) { return base; }
    let firstChild = u32(firstChildF);
    var off: u32 = 0u;
    if (octUV.x >= uMid) { off = off + 1u; }
    if (octUV.y >= vMid) { off = off + 2u; }
    idx = firstChild + off;
  }
  return dTreeOffset + DTREE_HEADER_F32;
}

// ── Guide pdf for an arbitrary world direction (Müller §3.2/§3.4) ───────────
// p_guide(ωi) = (leafFlux / totalFlux) / solidAngle_leaf, with a 1/(4π)
// uniform fallback when the cell has no training flux yet. Mirrors the CPU
// dTreePdf (dTree.ts) EXACTLY — this is the defensive evaluation that keeps
// the gi-ris mixture estimator unbiased.
fn ppgEvalPdf(pos: vec3<f32>, wi: vec3<f32>) -> f32 {
  if (!ppgQueryArenaValidGi()) { return 1.0 / PPG_FOUR_PI; }
  let sBase = ppgSTreeFindLeafBase(pos);
  let dTreeIndex = u32(ppgArenaLoadSTreeF32(sBase + 10u));
  let dOff = ppgArenaLoadDTreeOffset(dTreeIndex);
  let totalFlux = ppgArenaLoadDTreeF32(dOff + 2u);
  if (totalFlux <= 0.0) { return 1.0 / PPG_FOUR_PI; }
  let octUV = ppgDirToUv(wi);
  let leafBase = ppgDTreeFindLeafBase(dOff, octUV);
  let leafFlux = ppgArenaLoadDTreeF32(leafBase + 4u);
  let solidAng = ppgArenaLoadDTreeF32(leafBase + 5u);
  if (!(leafFlux > 0.0) || !(solidAng > 0.0)) {
    return 1.0 / PPG_FOUR_PI;
  }
  return (leafFlux / totalFlux) / solidAng;
}

// ── Flux-proportional dTree leaf sampler ─────────────────────────────────────
fn ppgDTreeSampleLeafBase(dTreeOffset: u32, rng: ptr<function, u32>) -> u32 {
  var idx: u32 = 0u;
  for (var step: u32 = 0u; step < 32u; step = step + 1u) {
    let base = dTreeOffset + DTREE_HEADER_F32 + idx * DTREE_NODE_STRIDE;
    let isLeafFlag = ppgArenaLoadDTreeF32(base + 7u);
    if (isLeafFlag > 0.5) { return base; }
    let firstChildF = ppgArenaLoadDTreeF32(base + 6u);
    if (firstChildF < 0.0) { return base; }
    let firstChild = u32(firstChildF);

    var sum: f32 = 0.0;
    var cFlux: array<f32, 4>;
    for (var ci: u32 = 0u; ci < 4u; ci = ci + 1u) {
      let cBase = dTreeOffset + DTREE_HEADER_F32 + (firstChild + ci) * DTREE_NODE_STRIDE;
      cFlux[ci] = ppgArenaLoadDTreeF32(cBase + 4u);
      sum = sum + cFlux[ci];
    }
    let r = rand_f32(rng);
    var pick: u32 = 3u;
    if (sum <= 0.0) {
      // Uniform fallback when the cell's children carry no flux (cold start).
      pick = min(u32(r * 4.0), 3u);
    } else {
      let targetFlux = r * sum;
      var cum: f32 = 0.0;
      for (var ci: u32 = 0u; ci < 4u; ci = ci + 1u) {
        cum = cum + cFlux[ci];
        if (targetFlux < cum && pick == 3u) { pick = ci; }
      }
    }
    idx = firstChild + pick;
  }
  return dTreeOffset + DTREE_HEADER_F32;
}

// ── Draw a guided world direction from the learned dTree ────────────────────
// Returns a unit world direction sampled ∝ leaf flux, jittered within the
// chosen leaf's octahedral patch. The caller evaluates the mixture pdf for the
// returned direction via ppgEvalPdf (and a cosine pdf) — this routine does NOT
// return a pdf, because the RIS source pdf is the α-mixture, not p_guide alone.
fn ppgSampleGuidedDir(pos: vec3<f32>, rng: ptr<function, u32>) -> vec3<f32> {
  if (!ppgQueryArenaValidGi()) {
    let z = rand_f32(rng) * 2.0 - 1.0;
    let phi = rand_f32(rng) * 6.283185307179586;
    let rxy = sqrt(max(0.0, 1.0 - z * z));
    return vec3<f32>(rxy * cos(phi), rxy * sin(phi), z);
  }
  let sBase = ppgSTreeFindLeafBase(pos);
  let dTreeIndex = u32(ppgArenaLoadSTreeF32(sBase + 10u));
  let dOff = ppgArenaLoadDTreeOffset(dTreeIndex);
  let totalFlux = ppgArenaLoadDTreeF32(dOff + 2u);
  // Degenerate cell (no training flux): fall back to a uniform-sphere sample
  // so the returned direction is still valid (its mixture pdf is dominated by
  // the cosine term anyway, and p_guide reduces to the 1/4π uniform fallback).
  if (totalFlux <= 0.0) {
    let z = rand_f32(rng) * 2.0 - 1.0;
    let phi = rand_f32(rng) * 6.283185307179586;
    let rxy = sqrt(max(0.0, 1.0 - z * z));
    return vec3<f32>(rxy * cos(phi), rxy * sin(phi), z);
  }
  let leafBase = ppgDTreeSampleLeafBase(dOff, rng);
  let u0 = ppgArenaLoadDTreeF32(leafBase + 0u);
  let v0 = ppgArenaLoadDTreeF32(leafBase + 1u);
  let u1 = ppgArenaLoadDTreeF32(leafBase + 2u);
  let v1 = ppgArenaLoadDTreeF32(leafBase + 3u);
  let r0 = rand_f32(rng);
  let r1 = rand_f32(rng);
  let uv = vec2<f32>(u0 + r0 * (u1 - u0), v0 + r1 * (v1 - v0));
  // [0,1]² UV → world dir via the cylindrical equal-area map (inverse of ppgDirToUv).
  return ppgUvToDir(uv);
}
`;

/** W1-R6 — declarative include-graph entry.
 *  Requires `ppgTreeLayout` for the shared DTREE_/STREE_ layout constants.
 *  No octahedralCore require: the 2026-06-09 equal-area fix replaced
 *  octEncode/octDecode with the inline cylindrical map (`ppgDirToUv`/
 *  `ppgUvToDir`), so the shared octahedral helpers are no longer referenced.
 *  The `rand_f32` RNG + the group(3) PPG bindings are provided by the gi-ris
 *  compilation unit (sharedPrimitives supplies `rand_f32`; risGi declares it
 *  requires this module). The bindings live here because only gi-ris consumes them. */
export const PPG_PDF_MODULE: WgslModule = {
  name: 'ppgPdf',
  source: PPG_PDF_WGSL,
  requires: ['ppgTreeLayout'],
};
