/**
 * Probe-update pass UBO layouts (W4c extract from probeUpdatePass.ts).
 */
import { defineUbo } from '@vitrum/shared-samplers';

/**
 * Compact producer/consumer record shared by the DDGI ray and blend passes:
 * hitRadiance.xyz + hitDistance, then direction.xyz + one alignment lane.
 */
export const PROBE_RAY_STRIDE_BYTES = 32;

export const DDGI_BORDER_UBO = defineUbo([
  { name: 'numProbes',   type: 'u32' },
  { name: 'atlasWidth',  type: 'u32' },
  { name: 'atlasHeight', type: 'u32' },
  { name: '_pad0',       type: 'u32' },
  { name: 'gridDimX',    type: 'u32' },
  { name: 'gridDimY',    type: 'u32' },
  { name: 'gridDimZ',    type: 'u32' },
  { name: '_pad1',       type: 'u32' },
] as const);

/**
 * Byte size of the DDGI border-update UBO — derived from the UBO definition so
 * this constant stays in sync with the field list above (no hand-maintenance). Item 18d.
 * Current layout: 8 × u32 (4 bytes each) = 32 bytes, rounded to max(4,4) = 32.
 */
export const DDGI_BORDER_UBO_BYTES: number = DDGI_BORDER_UBO.sizeBytes;

export const DDGI_FRAME_PARAMS_UBO = defineUbo([
  { name: 'randomRotation', type: 'vec3f' },
  { name: 'frameIndex',     type: 'u32'   },
  { name: 'skyTint',        type: 'vec3f' },
  { name: 'skyIrradiance',  type: 'f32'   },
  { name: 'glassMixScale',  type: 'f32'   },
  // H46-A — DDGI indirect-feedback gate (was the inert _pad2 slot; byte size
  // unchanged). 1 = multi-bounce diffuse EMA (maxBounces >= 2, default),
  // 0 = direct-only probes (maxBounces == 1).
  { name: 'indirectFeedback', type: 'u32' },
  // Wave 4 — HDRI into DDGI probe misses (2026-06-10).
  // hasEnv=1 gates the equirect sample path in sampleSkyColor; 0 keeps the
  // procedural gradient (byte-identical to the pre-Wave-4 path for no-HDRI
  // scenes). envRotationY + envIntensity mirror the convention in
  // environmentSample.wgsl (H6 RY(-rotY) world→map lookup).
  { name: 'hasEnv',         type: 'u32'   },   // was _pad3
  { name: 'envRotationY',   type: 'f32'   },   // was _pad4 — repurposed (same 4-byte slot)
  // Item 18d: the real UBO byte size is DDGI_FRAME_PARAMS_UBO.sizeBytes (computed by defineUbo).
  // Field offsets (std140/WGSL §14.4.4, maxAlign=16):
  //   randomRotation  @ 0   (vec3f, align=16, size=12) → cursor 12
  //   frameIndex @ 12; skyTint @ 16; skyIrradiance @ 28;
  //   glassMixScale @ 32; indirectFeedback @ 36; hasEnv @ 40;
  //   envRotationY @ 44; envIntensity @ 48; struct size = 64 bytes.
  { name: 'envIntensity',   type: 'f32'   },
] as const);

export const DDGI_BLEND_PARAMS_UBO = defineUbo([
  { name: 'hysteresis', type: 'f32' },
] as const);
