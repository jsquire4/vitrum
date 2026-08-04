/**
 * RC GPU buffer packing helpers — owned by the rc/ subsystem.
 *
 * Extracted from `HybridEngineRC.ts` (D2.6, R6 E sweep, 2026-06-11) so the
 * RC subsystem owns its own packing logic. `HybridEngineRC.ts` re-exports all
 * symbols below for back-compatibility with existing callers.
 *
 * Exports:
 *   packRCParams           — pack the 64-byte RCParams struct
 *   packRCLights           — pack a runtime-sized RCLight + alias buffer
 *   RC_LIGHTS_BUFFER_BYTES — byte size of the canonical empty header
 *   RC_LIGHTS_HEADER_BYTES — byte size of the RCLightBuffer header section
 *   RC_LIGHT_ENTRY_BYTES   — byte size of one RCLight entry
 *   RCLightBufferHeaderOffset — byte offsets in the RCLightBuffer header
 *   RCLightEntryOffset        — byte offsets within one RCLight entry
 */

import { buildAliasTable, luminance } from '@vitrum/shared-samplers';
import type { DDGILight } from '../ddgi/types.js';
import {
  canonicalizeLightingDirectionF32,
  packFiniteLightingFloat32,
  packLightingRgbScaleEnvelopeF32,
  packNonNegativeLightingFloat32,
  packNonNegativeLightingRgbF32,
} from '../lightingFloat32.js';
import {
  RC_PARAMS_BYTE_SIZE,
  RCParamsOffset,
  RC_LIGHTS_BUFFER_BYTES,
  RC_LIGHTS_HEADER_BYTES,
  RC_LIGHT_ENTRY_BYTES,
  RC_LIGHT_ALIAS_ENTRY_BYTES,
  RC_LIGHTS_ABI_MAGIC,
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
  RC_LIGHT_ALIAS_ENTRY_BYTES,
  RC_LIGHTS_ABI_MAGIC,
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
//   [4..]   runtime RCLight records, followed by runtime alias entries
// Records are followed by one 16-byte alias entry per active light.
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
// Active fixture, teaLight, and sun DDGILights are retained; off lights are filtered.

// RC_LIGHTS_BUFFER_BYTES, RC_LIGHTS_HEADER_BYTES, RC_LIGHT_ENTRY_BYTES,
// RCLightBufferHeaderOffset and RCLightEntryOffset are now imported from the
// codegen (rcParamsLayout.generated.ts) and re-exported at the top of this file.
export const RC_LIGHT_KIND_MASK = 0x7fffffff;
export const RC_LIGHT_CAST_SHADOW_DISABLED = 0x80000000;
export const RC_LIGHT_KIND_POINT = 1;
export const RC_LIGHT_KIND_SPOT = 2;
export const RC_LIGHT_KIND_DIRECTIONAL = 3;

/**
 * Proposal weights are binary64 host-only inputs to the robust alias builder,
 * not GPU f32 fields. Preserve the wider product so `radiance × 4π` cannot
 * invalidate otherwise representable RCLight radiance.
 */
function rcAliasProposalWeight(
  emittedLuminance: number,
  solidAngle: number,
  label: string,
): number {
  const weight = emittedLuminance * solidAngle;
  if (!Number.isFinite(weight) || weight < 0) {
    throw new RangeError(`${label} must be finite and non-negative.`);
  }
  return weight;
}

export function rcLightsBufferByteLength(count: number): number {
  if (!Number.isSafeInteger(count) || count < 0 || count > 0xffff_ffff) {
    throw new RangeError('[RC] light count must fit in u32.');
  }
  const bytes = RC_LIGHTS_HEADER_BYTES
    + count * (RC_LIGHT_ENTRY_BYTES + RC_LIGHT_ALIAS_ENTRY_BYTES);
  if (!Number.isSafeInteger(bytes)) {
    throw new RangeError('[RC] runtime light buffer byte length exceeds Number.MAX_SAFE_INTEGER.');
  }
  return bytes;
}

/** Runtime-sized RCLight records followed by a represented-PMF alias table. */
export function packRCLights(lights: readonly DDGILight[]): ArrayBuffer {
  // Word (f32/u32) strides derived from the generated byte layout — the single
  // source of truth for the RCLightBuffer wire format (rcParamsLayout.generated.ts).
  const HEADER_FLOATS = RC_LIGHTS_HEADER_BYTES / 4;   // 4
  const LIGHT_FLOATS  = RC_LIGHT_ENTRY_BYTES / 4;     // 16
  const active = lights.filter((light) => light.on);
  for (const [index, light] of active.entries()) {
    if (light.kind !== 'fixture' && light.kind !== 'teaLight' && light.kind !== 'sun') {
      throw new RangeError(`[RC] active light[${index}] has unsupported kind ${JSON.stringify(light.kind)}.`);
    }
  }
  const data = new Float32Array(rcLightsBufferByteLength(active.length) / 4);
  const ui   = new Uint32Array(data.buffer);
  ui[RCLightBufferHeaderOffset.count / 4] = active.length;  // count
  ui[RCLightBufferHeaderOffset.entriesWordOffset / 4] = HEADER_FLOATS;
  ui[RCLightBufferHeaderOffset.aliasWordOffset / 4] = HEADER_FLOATS + active.length * LIGHT_FLOATS;
  ui[RCLightBufferHeaderOffset.abiMagic / 4] = RC_LIGHTS_ABI_MAGIC;

  // Entry field word indices from the generated byte offsets (all 4-aligned).
  const O = RCLightEntryOffset;
  const weights: number[] = [];
  active.forEach((l, i) => {
    const base = HEADER_FLOATS + i * LIGHT_FLOATS;
    const isSun = l.kind === 'sun';
    const isSpot = !isSun && l.spotAxis != null && (l.spotCosInner != null || l.spotCosOuter != null);
    const shadowFlag = l.castShadow === false ? RC_LIGHT_CAST_SHADOW_DISABLED : 0;
    const kind = isSun ? RC_LIGHT_KIND_DIRECTIONAL : isSpot ? RC_LIGHT_KIND_SPOT : RC_LIGHT_KIND_POINT;
    ui[base + O.kind / 4] = (kind | shadowFlag) >>> 0;
    data[base + O.distance / 4] = isSun
      ? 0
      : packNonNegativeLightingFloat32(
          l.distance ?? 0,
          `packRCLights lights[${i}].distance`,
        );
    data[base + O.decay / 4] = isSun
      ? 0
      : packNonNegativeLightingFloat32(
          l.decay ?? 2,
          `packRCLights lights[${i}].decay`,
        );
    data[base + O.position / 4 + 0] = packFiniteLightingFloat32(
      l.position?.x ?? 0,
      `packRCLights lights[${i}].position.x`,
    );
    data[base + O.position / 4 + 1] = packFiniteLightingFloat32(
      l.position?.y ?? 0,
      `packRCLights lights[${i}].position.y`,
    );
    data[base + O.position / 4 + 2] = packFiniteLightingFloat32(
      l.position?.z ?? 0,
      `packRCLights lights[${i}].position.z`,
    );
    const intensity = packNonNegativeLightingFloat32(
      l.intensity,
      `packRCLights lights[${i}].intensity`,
    );
    data[base + O.intensity / 4] = intensity;
    // Spot axis (forward beam/travel direction; zero vector → point light → cone skipped).
    const sourceDirection = isSun
      ? (l.direction ?? { x: 0, y: -1, z: 0 })
      : l.spotAxis;
    const direction = sourceDirection == null
      ? [0, 0, 0] as [number, number, number]
      : canonicalizeLightingDirectionF32(
          [sourceDirection.x, sourceDirection.y, sourceDirection.z],
          `packRCLights lights[${i}].direction`,
        );
    data[base + O.direction / 4 + 0] = direction[0];
    data[base + O.direction / 4 + 1] = direction[1];
    data[base + O.direction / 4 + 2] = direction[2];
    data[base + O.innerCone / 4] = isSun
      ? 0
      : packFiniteLightingFloat32(
          l.spotCosInner ?? 1,
          `packRCLights lights[${i}].innerCone`,
        );
    const fallback = isSun ? { r: 1, g: 0.95, b: 0.85 } : { r: 1, g: 1, b: 1 };
    const color = l.color ?? fallback;
    const [r, g, b] = packNonNegativeLightingRgbF32(
      [color.r, color.g, color.b],
      `packRCLights lights[${i}].color`,
    );
    data[base + O.color / 4 + 0] = r;
    data[base + O.color / 4 + 1] = g;
    data[base + O.color / 4 + 2] = b;
    const outer = isSun
      ? packNonNegativeLightingFloat32(
          l.angularRadius ?? 0,
          `packRCLights lights[${i}].angularRadius`,
        )
      : packFiniteLightingFloat32(
          l.spotCosOuter ?? 0,
          `packRCLights lights[${i}].outerCone`,
        );
    data[base + O.outerCone / 4] = outer;
    const emitted = packLightingRgbScaleEnvelopeF32(
      [r, g, b],
      intensity,
      `packRCLights lights[${i}]`,
    ).scaled;
    const emittedLuminance = rcAliasProposalWeight(
      luminance(emitted[0], emitted[1], emitted[2]),
      1,
      `packRCLights lights[${i}] emitted luminance`,
    );
    const solidAngle = isSun
      ? 1
      : isSpot
        ? 2 * Math.PI * Math.max(0, 1 - outer)
        : 4 * Math.PI;
    weights.push(rcAliasProposalWeight(
      emittedLuminance,
      solidAngle,
      `packRCLights lights[${i}] alias weight`,
    ));
  });

  const alias = buildAliasTable(weights);
  new Uint8Array(data.buffer, ui[RCLightBufferHeaderOffset.aliasWordOffset / 4]! * 4)
    .set(new Uint8Array(alias.data));

  return data.buffer;
}
