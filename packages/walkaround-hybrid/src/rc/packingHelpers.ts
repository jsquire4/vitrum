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

import type { EngineWarning } from '@vitrum/core';
import type { DDGILight } from '../ddgi/types.js';
import {
  RC_PARAMS_BYTE_SIZE,
  RCParamsOffset,
  RC_LIGHTS_BUFFER_BYTES,
  RC_LIGHTS_HEADER_BYTES,
  RC_LIGHT_ENTRY_BYTES,
  RC_LIGHTS_MAX,
  RCLightBufferHeaderOffset,
  RCLightEntryOffset,
} from './rcParamsLayout.generated.js';

// Re-export the generated RCLight/RCLightBuffer layout constants so existing
// consumers (HybridEngineRC.ts, rcLightsLayoutPin.test.ts) keep importing them
// from this module. The single source of truth is now the codegen
// (tools/generate-wgsl-layouts.mjs), which emits these to match the WGSL
// `struct RCLight` / `struct RCLightBuffer` in probeRayCast.wgsl.ts.
export {
  RC_LIGHTS_BUFFER_BYTES,
  RC_LIGHTS_HEADER_BYTES,
  RC_LIGHT_ENTRY_BYTES,
  RCLightBufferHeaderOffset,
  RCLightEntryOffset,
};

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

// RC_LIGHTS_BUFFER_BYTES, RC_LIGHTS_HEADER_BYTES, RC_LIGHT_ENTRY_BYTES,
// RCLightBufferHeaderOffset and RCLightEntryOffset are now imported from the
// codegen (rcParamsLayout.generated.ts) and re-exported at the top of this file.
export const RC_LIGHT_KIND_MASK = 0x7fffffff;
export const RC_LIGHT_CAST_SHADOW_DISABLED = 0x80000000;

export interface PackRCLightsOptions {
  readonly onWarning?: (warning: EngineWarning) => void;
  readonly phase?: EngineWarning['phase'];
  readonly method?: string;
}

/**
 * Pack a `DDGILight[]` into the `RCLightBuffer` wire format.
 * Ignores 'sun' kind (RC uses sunDirection/sunColor in CascadeUniforms).
 * Truncates at 16 entries (matching DDGI's per-probe cap) with a warning.
 */
export function packRCLights(
  lights: readonly DDGILight[],
  options: PackRCLightsOptions = {},
): ArrayBuffer {
  // Word (f32/u32) strides derived from the generated byte layout — the single
  // source of truth for the RCLightBuffer wire format (rcParamsLayout.generated.ts).
  const HEADER_FLOATS = RC_LIGHTS_HEADER_BYTES / 4;   // 4
  const LIGHT_FLOATS  = RC_LIGHT_ENTRY_BYTES / 4;     // 16
  const MAX           = RC_LIGHTS_MAX;                // 16
  const data = new Float32Array(RC_LIGHTS_BUFFER_BYTES / 4);
  const ui   = new Uint32Array(data.buffer);

  const fixtures = lights.filter((l) => l.on && (l.kind === 'fixture' || l.kind === 'teaLight'));
  if (fixtures.length > MAX) {
    const warning: EngineWarning = {
      code: 'walkaround-hybrid.rc-light-cap-exceeded',
      backend: 'walkaround-hybrid',
      phase: options.phase ?? 'renderFrame',
      method: options.method ?? 'renderFrame',
      message:
        `[RC] packRCLights: scene has ${fixtures.length} active fixtures but RC supports ` +
        `at most ${MAX}. Extra lights beyond the cap are dropped for probe-ray GI.`,
      details: {
        activeFixtureCount: fixtures.length,
        maxLights: MAX,
        droppedLightCount: fixtures.length - MAX,
        fallback: 'drop-extra-rc-lights',
      },
    };
    if (options.onWarning !== undefined) {
      try {
        options.onWarning(warning);
      } catch {
        // Host warning callbacks must not prevent RC light packing.
      }
    } else {
      console.warn(warning.message);
    }
  }
  const active = fixtures.slice(0, MAX);
  ui[RCLightBufferHeaderOffset.count / 4] = active.length;  // count

  // Entry field word indices from the generated byte offsets (all 4-aligned).
  const O = RCLightEntryOffset;
  active.forEach((l, i) => {
    const base = HEADER_FLOATS + i * LIGHT_FLOATS;
    const isSpot = l.spotAxis != null && (l.spotCosInner != null || l.spotCosOuter != null);
    const shadowFlag = l.castShadow === false ? RC_LIGHT_CAST_SHADOW_DISABLED : 0;
    ui[base + O.kind / 4]        = ((isSpot ? 2 : 1) | shadowFlag) >>> 0; // kind + shadow flag
    data[base + O.distance / 4]  = typeof l.distance === 'number' && l.distance > 0 ? l.distance : 0;
    data[base + O.decay / 4]     = typeof l.decay === 'number' ? l.decay : 2;
    data[base + O.position / 4 + 0] = l.position?.x ?? 0;
    data[base + O.position / 4 + 1] = l.position?.y ?? 0;
    data[base + O.position / 4 + 2] = l.position?.z ?? 0;
    data[base + O.intensity / 4] = l.intensity;
    // Spot axis (forward beam/travel direction; zero vector → point light → cone skipped).
    data[base + O.direction / 4 + 0] = l.spotAxis?.x ?? 0;
    data[base + O.direction / 4 + 1] = l.spotAxis?.y ?? 0;
    data[base + O.direction / 4 + 2] = l.spotAxis?.z ?? 0;
    data[base + O.innerCone / 4] = l.spotCosInner ?? 1;  // innerCone cos (1 = no inner cone)
    data[base + O.color / 4 + 0] = l.color?.r ?? 1;
    data[base + O.color / 4 + 1] = l.color?.g ?? 1;
    data[base + O.color / 4 + 2] = l.color?.b ?? 1;
    data[base + O.outerCone / 4] = l.spotCosOuter ?? 0;  // outerCone cos (0 = point fallback)
  });

  return data.buffer;
}
