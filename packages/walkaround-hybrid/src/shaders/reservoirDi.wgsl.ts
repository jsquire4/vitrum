/**
 * ReSTIR-DI reservoir ADT + the emitter / G-buffer structs it co-locates.
 *
 * Split out of common.wgsl.ts (T9-stepA): the `EmitterTri` (80-byte) struct,
 * the `ReservoirDI` struct (with stored xi —
 * Bitterli 2020 §4), `emptyReservoirDI` / `updateReservoirDI`, and the
 * strided pack/unpack helpers (load/store, 6×u32 stride) shared by
 * ris/temporal/spatial.
 *
 * `updateReservoirDI` forward-references `rand_f32` (defined in the shared
 * primitives module). WGSL resolves module-scope functions regardless of
 * declaration order, and `common` aggregates the modules in the original
 * source order, so the forward reference is well-formed.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const RESERVOIR_DI_WGSL = /* wgsl */ `// ============================================================
// Emitter struct (80 bytes per emitter, 16-byte aligned)
// ============================================================
struct EmitterTri {
  vA:        vec3f,   // bytes 0-11
  _padA:     f32,     // bytes 12-15
  vB:        vec3f,   // bytes 16-27
  _padB:     f32,     // bytes 28-31
  vC:        vec3f,   // bytes 32-43
  _padC:     f32,     // bytes 44-47
  normal:    vec3f,   // bytes 48-59
  area:      f32,     // bytes 60-63
  Le:        vec3f,   // bytes 64-75
  intensity: f32,     // bytes 76-79
};

// ============================================================
// ReSTIR DI Reservoir (24 bytes — 6 × u32)
// ============================================================
//
// The xi field (2 × f32 stored as 2 × u32 via bitcast) captures the
// random sample params used by sampleEmitterPoint when this candidate
// won the WRS. Without it the visibility test stage couldn't reconstruct
// the original sample point and fell back to centroid — a real bias
// (visibility at centroid vs at the actual sample disagrees for any
// emitter whose extent is comparable to the occluder's). Bitterli 2020
// section 4 documents this as the canonical "store xi alongside lightId" path.
struct ReservoirDI {
  lightId: u32,
  M:       u32,
  w_sum:   f32,
  W:       f32,
  xi:      vec2f,    // sampled (u, v) on the chosen emitter
};

fn emptyReservoirDI() -> ReservoirDI {
  return ReservoirDI(0u, 0u, 0.0, 0.0, vec2f(0.0, 0.0));
}

fn updateReservoirDI(r: ptr<function, ReservoirDI>, lid: u32, xi: vec2f, w: f32, rng: ptr<function, u32>) {
  (*r).M += 1u;
  (*r).w_sum += w;
  if (rand_f32(rng) * (*r).w_sum < w) {
    (*r).lightId = lid;
    (*r).xi      = xi;
  }
}

// ============================================================
// ReservoirDI pack/unpack helpers — canonical, used by ris/temporal/spatial.
// 16 bytes = 4 × u32 per pixel. lightId, M are u32; w_sum and W are
// bit-cast to/from u32 to preserve f32 precision through the storage buffer.
// ============================================================
// 6 u32 = 24 bytes per reservoir (was 4 u32 = 16 bytes pre-xi).
const RESERVOIR_DI_STRIDE = 6u;

fn loadReservoirDI_rw(buf: ptr<storage, array<u32>, read_write>, pixelIdx: u32) -> ReservoirDI {
  let base = pixelIdx * RESERVOIR_DI_STRIDE;
  return ReservoirDI(
    buf[base],
    buf[base + 1u],
    bitcast<f32>(buf[base + 2u]),
    bitcast<f32>(buf[base + 3u]),
    vec2f(bitcast<f32>(buf[base + 4u]), bitcast<f32>(buf[base + 5u])),
  );
}

fn loadReservoirDI_ro(buf: ptr<storage, array<u32>, read>, pixelIdx: u32) -> ReservoirDI {
  let base = pixelIdx * RESERVOIR_DI_STRIDE;
  return ReservoirDI(
    buf[base],
    buf[base + 1u],
    bitcast<f32>(buf[base + 2u]),
    bitcast<f32>(buf[base + 3u]),
    vec2f(bitcast<f32>(buf[base + 4u]), bitcast<f32>(buf[base + 5u])),
  );
}

fn storeReservoirDI_rw(buf: ptr<storage, array<u32>, read_write>, pixelIdx: u32, r: ReservoirDI) {
  let base = pixelIdx * RESERVOIR_DI_STRIDE;
  buf[base + 0u] = r.lightId;
  buf[base + 1u] = r.M;
  buf[base + 2u] = bitcast<u32>(r.w_sum);
  buf[base + 3u] = bitcast<u32>(r.W);
  buf[base + 4u] = bitcast<u32>(r.xi.x);
  buf[base + 5u] = bitcast<u32>(r.xi.y);
}

`;

/** T9-stepA — focused WGSL_MODULES entry split out of `common`. */
export const RESERVOIR_DI_MODULE: WgslModule = {
  name: "reservoirDi",
  source: RESERVOIR_DI_WGSL,
  requires: [],
};
