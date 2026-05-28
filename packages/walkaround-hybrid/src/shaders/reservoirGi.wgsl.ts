/**
 * ReSTIR-GI reservoir ADT + the `PrimarySurface` struct it builds on.
 *
 * Split out of common.wgsl.ts (T9-stepA): the `PrimarySurface` struct
 * (re-cast primary-ray receiver, shared by temporal/spatial and read inline
 * by shade), the `ReservoirGI` struct (80-byte / 20×u32), its
 * `emptyReservoirGI`, strided load/store helpers, and `updateReservoirGI`
 * (Sprint 16 ReSTIR-GI). `updateReservoirGI` forward-references `rand_f32`
 * (shared primitives) — see reservoirDi.wgsl.ts header note on ordering.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const RESERVOIR_GI_WGSL = /* wgsl */ `// ============================================================
// PrimarySurface — derived from re-casting the primary ray through the BVH.
// Replaces the pre-fix placeholder G-buffer reads that returned constant
// values for all pixels. Shared by temporal and spatial passes; shade.wgsl
// reads the same fields inline.
// ============================================================
struct PrimarySurface {
  hit:    bool,
  pos:    vec3f,
  normal: vec3f,
  wo:     vec3f,
  albedo: vec3f,
  rough:  f32,
  metal:  f32,
  depth:  f32,
};

// ============================================================
// ReSTIR GI Reservoir (80 bytes, co-located at pixel offset after DI)
// ============================================================
struct ReservoirGI {
  xv:      vec3f,   // visible point (primary hit)
  _pad0:   f32,
  nv:      vec3f,   // normal at xv
  W:       f32,
  xs:      vec3f,   // sample point (secondary bounce hit)
  w_sum:   f32,
  ns:      vec3f,   // normal at xs
  M:       u32,
  Lo:      vec3f,   // outgoing radiance at xs
  lightId: u32,
};

fn emptyReservoirGI() -> ReservoirGI {
  var r: ReservoirGI;
  r.xv = vec3f(0.0); r.nv = vec3f(0,1,0);
  r.xs = vec3f(0.0); r.ns = vec3f(0,1,0);
  r.Lo = vec3f(0.0); r.W = 0.0; r.w_sum = 0.0; r.M = 0u;
  r.lightId = 0u; r._pad0 = 0.0;
  return r;
}

// Sprint 16 — ReservoirGI byte layout (80 bytes = 20 × u32):
//   [0..2]  xv.xyz       [3]    _pad0
//   [4..6]  nv.xyz       [7]    W
//   [8..10] xs.xyz       [11]   w_sum
//   [12..14] ns.xyz      [15]   M
//   [16..18] Lo.xyz      [19]   lightId
// Strided storage in array<u32> (4-byte elements) — stride = 20 u32.
const RESERVOIR_GI_STRIDE: u32 = 20u;

fn loadReservoirGI_rw(buf: ptr<storage, array<u32>, read_write>, pixelIdx: u32) -> ReservoirGI {
  let b = pixelIdx * RESERVOIR_GI_STRIDE;
  var r: ReservoirGI;
  r.xv      = vec3f(bitcast<f32>(buf[b + 0u]), bitcast<f32>(buf[b + 1u]), bitcast<f32>(buf[b + 2u]));
  r._pad0   = bitcast<f32>(buf[b + 3u]);
  r.nv      = vec3f(bitcast<f32>(buf[b + 4u]), bitcast<f32>(buf[b + 5u]), bitcast<f32>(buf[b + 6u]));
  r.W       = bitcast<f32>(buf[b + 7u]);
  r.xs      = vec3f(bitcast<f32>(buf[b + 8u]), bitcast<f32>(buf[b + 9u]), bitcast<f32>(buf[b + 10u]));
  r.w_sum   = bitcast<f32>(buf[b + 11u]);
  r.ns      = vec3f(bitcast<f32>(buf[b + 12u]), bitcast<f32>(buf[b + 13u]), bitcast<f32>(buf[b + 14u]));
  r.M       = buf[b + 15u];
  r.Lo      = vec3f(bitcast<f32>(buf[b + 16u]), bitcast<f32>(buf[b + 17u]), bitcast<f32>(buf[b + 18u]));
  r.lightId = buf[b + 19u];
  return r;
}

fn loadReservoirGI_ro(buf: ptr<storage, array<u32>, read>, pixelIdx: u32) -> ReservoirGI {
  let b = pixelIdx * RESERVOIR_GI_STRIDE;
  var r: ReservoirGI;
  r.xv      = vec3f(bitcast<f32>(buf[b + 0u]), bitcast<f32>(buf[b + 1u]), bitcast<f32>(buf[b + 2u]));
  r._pad0   = bitcast<f32>(buf[b + 3u]);
  r.nv      = vec3f(bitcast<f32>(buf[b + 4u]), bitcast<f32>(buf[b + 5u]), bitcast<f32>(buf[b + 6u]));
  r.W       = bitcast<f32>(buf[b + 7u]);
  r.xs      = vec3f(bitcast<f32>(buf[b + 8u]), bitcast<f32>(buf[b + 9u]), bitcast<f32>(buf[b + 10u]));
  r.w_sum   = bitcast<f32>(buf[b + 11u]);
  r.ns      = vec3f(bitcast<f32>(buf[b + 12u]), bitcast<f32>(buf[b + 13u]), bitcast<f32>(buf[b + 14u]));
  r.M       = buf[b + 15u];
  r.Lo      = vec3f(bitcast<f32>(buf[b + 16u]), bitcast<f32>(buf[b + 17u]), bitcast<f32>(buf[b + 18u]));
  r.lightId = buf[b + 19u];
  return r;
}

fn storeReservoirGI_rw(buf: ptr<storage, array<u32>, read_write>, pixelIdx: u32, r: ReservoirGI) {
  let b = pixelIdx * RESERVOIR_GI_STRIDE;
  buf[b + 0u]  = bitcast<u32>(r.xv.x);
  buf[b + 1u]  = bitcast<u32>(r.xv.y);
  buf[b + 2u]  = bitcast<u32>(r.xv.z);
  buf[b + 3u]  = bitcast<u32>(r._pad0);
  buf[b + 4u]  = bitcast<u32>(r.nv.x);
  buf[b + 5u]  = bitcast<u32>(r.nv.y);
  buf[b + 6u]  = bitcast<u32>(r.nv.z);
  buf[b + 7u]  = bitcast<u32>(r.W);
  buf[b + 8u]  = bitcast<u32>(r.xs.x);
  buf[b + 9u]  = bitcast<u32>(r.xs.y);
  buf[b + 10u] = bitcast<u32>(r.xs.z);
  buf[b + 11u] = bitcast<u32>(r.w_sum);
  buf[b + 12u] = bitcast<u32>(r.ns.x);
  buf[b + 13u] = bitcast<u32>(r.ns.y);
  buf[b + 14u] = bitcast<u32>(r.ns.z);
  buf[b + 15u] = r.M;
  buf[b + 16u] = bitcast<u32>(r.Lo.x);
  buf[b + 17u] = bitcast<u32>(r.Lo.y);
  buf[b + 18u] = bitcast<u32>(r.Lo.z);
  buf[b + 19u] = r.lightId;
}

fn updateReservoirGI(
  r: ptr<function, ReservoirGI>,
  xs: vec3f, ns: vec3f, Lo: vec3f,
  w: f32,
  rng: ptr<function, u32>,
) {
  (*r).M = (*r).M + 1u;
  (*r).w_sum = (*r).w_sum + w;
  if (rand_f32(rng) * (*r).w_sum < w) {
    (*r).xs = xs;
    (*r).ns = ns;
    (*r).Lo = Lo;
  }
}

`;

/** T9-stepA — focused WGSL_MODULES entry split out of `common`. */
export const RESERVOIR_GI_MODULE: WgslModule = {
  name: "reservoirGi",
  source: RESERVOIR_GI_WGSL,
  requires: [],
};
