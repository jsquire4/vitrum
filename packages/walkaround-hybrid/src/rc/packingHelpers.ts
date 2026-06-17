/**
 * RC GPU buffer packing helpers — owned by the rc/ subsystem.
 *
 * Extracted from `HybridEngineRC.ts` (D2.6, R6 E sweep, 2026-06-11) so the
 * RC subsystem owns its own packing logic. `HybridEngineRC.ts` re-exports all
 * symbols below for back-compatibility with existing callers.
 *
 * Exports:
 *   packRCParams           — pack the 64-byte RCParams struct
 *   packRCLights           — pack the 1040-byte RCLightBuffer
 *   RC_LIGHTS_BUFFER_BYTES — byte size of the full RCLightBuffer allocation
 *   RC_LIGHTS_HEADER_BYTES — byte size of the RCLightBuffer header section
 *   RC_LIGHT_ENTRY_BYTES   — byte size of one RCLight entry
 *   RCLightBufferHeaderOffset — byte offsets in the RCLightBuffer header
 *   RCLightEntryOffset        — byte offsets within one RCLight entry
 */

import type { DDGILight } from '../ddgi/types.js';
import { RC_PARAMS_BYTE_SIZE, RCParamsOffset } from './rcParamsLayout.generated.js';

// ─── packRCParams ─────────────────────────────────────────────────────────────

/**
 * Pack the WGSL `sampleCascadeC0.wgsl` `RCParams` struct (64 bytes).
 * Layout — must match the WGSL struct declaration exactly:
 *   probeOriginWorld: vec3f  (offset 0..11)
 *   rcWeight:         f32    (offset 12..15)
 *   roomSize:         vec3f  (offset 16..27)
 *   enabled:          u32    (offset 28..31)
 *   probeCount:       vec3u  (offset 32..43)
 *   raysPerProbe:     u32    (offset 44..47)
 *   rayGridSize:      u32    (offset 48..51)
 *   _pad0/1/2:        3 × u32 (offset 52..63)
 */
export function packRCParams(
  probeOriginWorld: readonly [number, number, number],
  roomSize:         readonly [number, number, number],
  probeCount:       readonly [number, number, number],
  raysPerProbe:     number,
  rcWeight:         number,
  enabled:          boolean,
): ArrayBuffer {
  // Use the generated layout constants (RC_PARAMS_BYTE_SIZE, RCParamsOffset) so
  // this packer is the single source of truth for the wire format and the codegen
  // test can verify both independently.
  const buf = new ArrayBuffer(RC_PARAMS_BYTE_SIZE);
  const f = new Float32Array(buf);
  const u = new Uint32Array(buf);
  // probeOriginWorld: vec3f at byte 0 → f32 words [0..2]
  f[RCParamsOffset.probeOriginWorld / 4 + 0] = probeOriginWorld[0];
  f[RCParamsOffset.probeOriginWorld / 4 + 1] = probeOriginWorld[1];
  f[RCParamsOffset.probeOriginWorld / 4 + 2] = probeOriginWorld[2];
  // rcWeight: f32 at byte 12 → f32 word [3]
  f[RCParamsOffset.rcWeight / 4] = rcWeight;
  // roomSize: vec3f at byte 16 → f32 words [4..6]
  f[RCParamsOffset.roomSize / 4 + 0] = roomSize[0];
  f[RCParamsOffset.roomSize / 4 + 1] = roomSize[1];
  f[RCParamsOffset.roomSize / 4 + 2] = roomSize[2];
  // enabled: u32 at byte 28 → u32 word [7]
  u[RCParamsOffset.enabled / 4] = enabled ? 1 : 0;
  // probeCount: vec3u at byte 32 → u32 words [8..10]
  u[RCParamsOffset.probeCount / 4 + 0] = probeCount[0];
  u[RCParamsOffset.probeCount / 4 + 1] = probeCount[1];
  u[RCParamsOffset.probeCount / 4 + 2] = probeCount[2];
  // raysPerProbe: u32 at byte 44 → u32 word [11]
  u[RCParamsOffset.raysPerProbe / 4] = raysPerProbe;
  // rayGridSize: u32 at byte 48 → u32 word [12]  (sqrt of raysPerProbe, ≥1)
  u[RCParamsOffset.rayGridSize / 4] = Math.max(1, Math.round(Math.sqrt(raysPerProbe)));
  // bytes 52..63 (_pad0/1/2): already zero from ArrayBuffer init.
  return buf;
}

// ─── RCLightBuffer packing (A7, 2026-06-10) ──────────────────────────────────
//
// Layout mirrors DDGI's DDGILightUniforms / DDGILight (probeUpdateLights.ts)
// so the same host-side DDGILight structs can be forwarded into RC. The WGSL
// struct is `RCLightBuffer` in probeRayCast.wgsl.ts:
//   [0]     count (u32)
//   [1..3]  _h0/h1/h2 pad (u32)
//   [4..]   items: array<RCLight, 16>  — 16 × 16 f32 = 256 f32 = 1024 bytes
// Total: 4 + 256 = 260 u32/f32 = 1040 bytes.
//
// Each RCLight (matches WGSL RCLight struct, 64 bytes = 16 floats):
//   [0]       kind (u32): low bits 0=skip, 1=point, 2=spot;
//             high bit set => castShadow:false on the source emitter
//   [1]       distance (f32) — 0 = no cutoff
//   [2]       decay (f32) — 0 = no falloff, 2 = physical inverse-square
//   [3]       _pad
//   [4..6]    position (vec3f)
//   [7]       intensity (f32)
//   [8..10]   direction (vec3f) — spot cone forward beam axis; zero for point
//   [11]      innerCone (f32)   — cosine of inner angle
//   [12..14]  color (vec3f)
//   [15]      outerCone (f32)   — cosine of outer angle
//
// Only 'fixture' and 'teaLight' kind DDGILights become point/spot RC lights;
// 'sun' kind is handled separately via sunDirection/sunColor in CascadeUniforms.
// Off-lights (l.on === false) are filtered out.
// Cap: 16 (matching DDGI's MAX_DDGI_PROBE_LIGHTS).

/** Byte size of the full RCLightBuffer GPU allocation. */
export const RC_LIGHTS_BUFFER_BYTES = (4 + 16 * 16) * 4; // 1040 bytes
export const RC_LIGHT_KIND_MASK = 0x7fffffff;
export const RC_LIGHT_CAST_SHADOW_DISABLED = 0x80000000;

/**
 * Byte offsets within the `RCLightBuffer` header section (first 16 bytes).
 * Mirrors the WGSL `struct RCLightBuffer` in `probeRayCast.wgsl.ts`:
 *   count: u32  @ 0
 *   _h0/h1/h2: u32 pad  @ 4/8/12
 */
export const RCLightBufferHeaderOffset = {
  count: 0,
} as const;

/**
 * Byte size of the RCLightBuffer header (16 bytes = 4 × u32).
 * Items array starts at this offset.
 */
export const RC_LIGHTS_HEADER_BYTES = 16;

/**
 * Byte size of one `RCLight` entry (64 bytes = 16 × f32/u32).
 * Mirrors the WGSL `struct RCLight` in `probeRayCast.wgsl.ts`.
 */
export const RC_LIGHT_ENTRY_BYTES = 64;

/**
 * Field byte offsets within one `RCLight` entry (relative to entry start).
 * Mirrors the WGSL `struct RCLight`:
 *   kind:      u32   @ 0
 *   distance:  f32   @ 4  (f32 word 1)
 *   decay:     f32   @ 8  (f32 word 2)
 *   _pad2:     f32   @ 12 (f32 word 3)
 *   position:  vec3f @ 16 (f32 words 4-6)
 *   intensity: f32   @ 28 (f32 word 7)
 *   direction: vec3f @ 32 (f32 words 8-10)
 *   innerCone: f32   @ 44 (f32 word 11)
 *   color:     vec3f @ 48 (f32 words 12-14)
 *   outerCone: f32   @ 60 (f32 word 15)
 */
export const RCLightEntryOffset = {
  kind:      0,
  distance:  4,
  decay:     8,
  position:  16,
  intensity: 28,
  direction: 32,
  innerCone: 44,
  color:     48,
  outerCone: 60,
} as const;

/**
 * Pack a `DDGILight[]` into the `RCLightBuffer` wire format.
 * Ignores 'sun' kind (RC uses sunDirection/sunColor in CascadeUniforms).
 * Truncates at 16 entries (matching DDGI's per-probe cap) with a warning.
 */
export function packRCLights(lights: readonly DDGILight[]): ArrayBuffer {
  const HEADER_FLOATS  = 4;
  const LIGHT_FLOATS   = 16;
  const MAX            = 16;
  const data = new Float32Array(HEADER_FLOATS + MAX * LIGHT_FLOATS);
  const ui   = new Uint32Array(data.buffer);

  const fixtures = lights.filter((l) => l.on && (l.kind === 'fixture' || l.kind === 'teaLight'));
  if (fixtures.length > MAX) {
    console.warn(
      `[RC] packRCLights: scene has ${fixtures.length} active fixtures but RC supports ` +
      `at most ${MAX}. Extra lights beyond the cap are dropped for probe-ray GI.`,
    );
  }
  const active = fixtures.slice(0, MAX);
  ui[0] = active.length;  // count

  active.forEach((l, i) => {
    const base = HEADER_FLOATS + i * LIGHT_FLOATS;
    const ub   = base;
    const isSpot = l.spotAxis != null && (l.spotCosInner != null || l.spotCosOuter != null);
    const shadowFlag = l.castShadow === false ? RC_LIGHT_CAST_SHADOW_DISABLED : 0;
    ui[ub]        = ((isSpot ? 2 : 1) | shadowFlag) >>> 0; // kind + shadow flag
    data[base + 1] = typeof l.distance === 'number' && l.distance > 0 ? l.distance : 0;
    data[base + 2] = typeof l.decay === 'number' ? l.decay : 2;
    data[base + 4] = l.position?.x ?? 0;
    data[base + 5] = l.position?.y ?? 0;
    data[base + 6] = l.position?.z ?? 0;
    data[base + 7] = l.intensity;
    // Spot axis (forward beam/travel direction; zero vector → point light → cone skipped).
    data[base + 8]  = l.spotAxis?.x ?? 0;
    data[base + 9]  = l.spotAxis?.y ?? 0;
    data[base + 10] = l.spotAxis?.z ?? 0;
    data[base + 11] = l.spotCosInner ?? 1;  // innerCone cos (1 = no inner cone)
    data[base + 12] = l.color?.r ?? 1;
    data[base + 13] = l.color?.g ?? 1;
    data[base + 14] = l.color?.b ?? 1;
    data[base + 15] = l.spotCosOuter ?? 0;  // outerCone cos (0 = point fallback)
  });

  return data.buffer;
}
