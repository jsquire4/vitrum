/**
 * UBO updater — writes per-frame camera + lighting + tunables into the
 * 368-byte WalkaroundUBO uniform buffer.
 *
 * UBO layout (mixed f32 / u32 — see WalkaroundUBO struct in common.wgsl):
 *   offset   0: viewMatrix                  (mat4×4f = 64 bytes)
 *   offset  64: projMatrix                  (64 bytes)
 *   offset 128: prevViewMatrix              (64 bytes)
 *   offset 192: cameraPos                   (vec3f = 12 bytes)
 *   offset 204: frameSeed                   (u32 = 4 bytes)
 *   offset 208: screenSize                  (vec2u = 8 bytes)
 *   offset 216: emitterCount                (u32 = 4 bytes)
 *   offset 220: totalEmPower                (f32 = 4 bytes)
 *   offset 224: primaryLightDir             (vec3f = 12 bytes)
 *   offset 236: primaryLightIntensity       (f32 = 4 bytes)
 *   offset 240: skyTint                     (vec3f = 12 bytes)
 *   offset 252: skyIrradiance               (f32 = 4 bytes)
 *   offset 256: emitterDist2Floor           (f32 = 4 bytes) — audit M12
 *   offset 260: directFireflyClamp          (f32 = 4 bytes) — audit B4
 *   offset 264: causticBoost                (f32 = 4 bytes) — audit B1
 *   offset 268: causticVisClamp             (f32 = 4 bytes) — audit B1
 *   offset 272: temporalMClampDI            (u32 = 4 bytes) — audit M6
 *   offset 276: spatialReuseRadiusPx        (f32 = 4 bytes) — audit M7
 *   offset 280: spatialDepthTolFloor        (f32 = 4 bytes) — audit M8
 *   offset 284: triIntersectEpsilon         (f32 = 4 bytes) — D12
 *   offset 288: glassMixScale               (f32 = 4 bytes) — sweep 2026-05-18
 *   offset 292: restirGiWCap                (f32 = 4 bytes) — sweep 2026-05-18
 *   offset 296: restirGiIrrClamp            (f32 = 4 bytes) — sweep 2026-05-18
 *   offset 300: restirGiMClamp              (u32 = 4 bytes) — sweep 2026-05-18
 *   offset 304: restirGiSpatialRadiusPx     (f32 = 4 bytes) — sweep 2026-05-18
 *   offset 308: restirGiSpatialNormalDotMin (f32 = 4 bytes) — sweep 2026-05-18
 *   offset 312: restirGiSpatialCoplanarTol  (f32 = 4 bytes) — sweep 2026-05-18
 *   offset 316: _padPreVec3                 (f32 = 4 bytes — align next vec3 to 16)
 *   offset 320: indirectFireflyClamp        (vec3f = 12 bytes) — sweep 2026-05-18
 *   offset 332: _padEnd                     (f32 = 4 bytes)
 *   offset 336: bvhMode                     (u32 = 4 bytes) — PR-3
 *   offset 340: tlasNodeCount               (u32 = 4 bytes)
 *   offset 344: stainedGlassFlags           (u32 = 4 bytes) — T5 (was _tracePad0)
 *   offset 348: ppgEnabled                  (u32 = 4 bytes) — PPG guided-sampling gate (was _tracePad1)
 *   offset 352: ppgMixAlpha                 (f32 = 4 bytes) — PPG MIS mixing weight α
 *   offset 356: lightTreeEnabled            (u32 = 4 bytes) — DI light-tree selection gate (was _ppgPad0)
 *   offset 360: lightTreeNodeCount          (u32 = 4 bytes) — packed light-tree node count (was _ppgPad1)
 *   offset 364: _ppgPad2                    (u32 = 4 bytes)
 * Total: 368 bytes (368 % 16 == 0).
 */

import type { PipelineFrameInputs } from './WalkaroundGPUPipeline.js';

/**
 * T5 — stained-glass opt-in flag bit masks. Bit 0 gates the sun-caustic term,
 * bit 1 gates the sky-aperture term. MUST match the `SG_FLAG_SUN_CAUSTIC` /
 * `SG_FLAG_SKY_APERTURE` constants in `shaders/walkaroundUbo.wgsl.ts` (the two
 * sides agree on bit positions). Exported so the engine + tests can pack the
 * flags without re-deriving the bit layout.
 */
export const SG_FLAG_SUN_CAUSTIC = 1; // bit 0
export const SG_FLAG_SKY_APERTURE = 2; // bit 1

/**
 * Pack the per-engine stained-glass opt-in booleans into the `u32` bitfield
 * that lands at UBO offset 344. Default (both `false`) → `0`, which makes
 * `lo_sg_caustic` / `lo_sg_aperture` early-return `vec3f(0)` — a generic scene
 * gets ZERO stained-glass caustic / aperture physics.
 */
export function packStainedGlassFlags(opts: {
  sunCaustic?: boolean | undefined;
  skyAperture?: boolean | undefined;
}): number {
  let flags = 0;
  if (opts.sunCaustic) flags |= SG_FLAG_SUN_CAUSTIC;
  if (opts.skyAperture) flags |= SG_FLAG_SKY_APERTURE;
  return flags >>> 0;
}

/** Size of the WalkaroundUBO in bytes. File-local — `resourceManager.ts`
 *  intentionally duplicates the literal `368` rather than import this name
 *  to avoid a circular import (see resourceManager.ts). */
const WALKAROUND_UBO_SIZE_BYTES = 368;

/**
 * Live PPG guided-sampling state injected by the pipeline (NOT part of the
 * host {@link PipelineFrameInputs} contract).
 *
 * The pipeline is the source of truth for whether PPG guided sampling is
 * live this frame: `enabled` mirrors `PPGCoordinator.enabled`, which is only
 * `true` when the host opted in AND both PPG compute pipelines compiled. When
 * `enabled` is false the kernel-side α collapses to 0, so gi-ris stays on the
 * pure-cosine path bit-for-bit. `mixAlpha` is the Müller §3.4 mixing weight.
 */
export interface PpgUboState {
  readonly enabled: boolean;
  readonly mixAlpha: number;
}

export function updateUBO(
  device: GPUDevice,
  uboBuffer: GPUBuffer,
  inputs: PipelineFrameInputs,
  /** Live PPG gate + α from the pipeline. Defaults to OFF so callers that
   *  don't run PPG (and the existing tests) keep the pure-cosine gi-ris path
   *  — ppgEnabled=0 and α=0 make the gi-ris RIS source pdf reduce exactly to
   *  cosθ/π, preserving ppg-OFF bit-identity. */
  ppg: PpgUboState = { enabled: false, mixAlpha: 0 },
): void {
  const data = new ArrayBuffer(WALKAROUND_UBO_SIZE_BYTES);
  const f32  = new Float32Array(data);
  const u32  = new Uint32Array(data);

  f32.set(inputs.viewMatrix,     0);    //  0..15 (64 bytes)
  f32.set(inputs.projMatrix,    16);    // 16..31 (64 bytes)
  f32.set(inputs.prevViewMatrix, 32);   // 32..47 (64 bytes)
  f32[48] = inputs.cameraPos[0];
  f32[49] = inputs.cameraPos[1];
  f32[50] = inputs.cameraPos[2];
  u32[51] = inputs.frameSeed >>> 0;
  u32[52] = inputs.screenWidth;
  u32[53] = inputs.screenHeight;
  u32[54] = inputs.emitterCount;
  f32[55] = inputs.totalEmissivePower;
  f32[56] = inputs.primaryLightDir[0];
  f32[57] = inputs.primaryLightDir[1];
  f32[58] = inputs.primaryLightDir[2];
  f32[59] = inputs.primaryLightIntensity;
  f32[60] = inputs.skyTint[0];
  f32[61] = inputs.skyTint[1];
  f32[62] = inputs.skyTint[2];
  f32[63] = inputs.skyIrradiance;
  // Library-generality tunables (audit follow-up).
  f32[64] = inputs.emitterDist2Floor;
  f32[65] = inputs.directFireflyClamp;
  f32[66] = inputs.causticBoost;
  f32[67] = inputs.causticVisClamp;
  u32[68] = inputs.temporalMClampDI >>> 0;
  f32[69] = inputs.spatialReuseRadiusPx;
  f32[70] = inputs.spatialDepthTolFloor;
  f32[71] = inputs.triIntersectEpsilon;
  // 2026-05-18 sweep — eight more Cornell-tuned magic constants threaded
  // through the UBO for library-consumer override.
  f32[72] = inputs.glassMixScale;
  f32[73] = inputs.restirGiWCap;
  f32[74] = inputs.restirGiIrrClamp;
  u32[75] = inputs.restirGiMClamp >>> 0;
  f32[76] = inputs.restirGiSpatialRadiusPx;
  f32[77] = inputs.restirGiSpatialNormalDotMin;
  f32[78] = inputs.restirGiSpatialCoplanarTol;
  // f32[79] = _padPreVec3 (zero — keeps indirectFireflyClamp vec3-aligned).
  f32[80] = inputs.indirectFireflyClamp[0];
  f32[81] = inputs.indirectFireflyClamp[1];
  f32[82] = inputs.indirectFireflyClamp[2];
  // f32[83] = _padEnd (zero).
  u32[84] = inputs.bvhMode >>> 0;
  u32[85] = inputs.tlasNodeCount >>> 0;
  // T5 — stained-glass opt-in flag bits (repurposed _tracePad0 at offset 344).
  // 0 → both terms OFF (generic-scene default).
  u32[86] = inputs.stainedGlassFlags >>> 0;
  // PPG guided sampling (W9 guided-sampling landing). offset 348 = ppgEnabled
  // (gate), offset 352 = ppgMixAlpha. When PPG is off, ppgEnabled stays 0 and
  // α stays 0 → gi-ris RIS source pdf = cosθ/π exactly (ppg-OFF bit-identity).
  u32[87] = ppg.enabled ? 1 : 0; //  offset 348 — ppgEnabled
  f32[88] = ppg.enabled ? ppg.mixAlpha : 0; // offset 352 — ppgMixAlpha
  // Light-tree DI light-SELECTION gate (offset 356) + node count (offset 360).
  // When disabled both stay 0 → RIS uses the flat power-CDF path exactly.
  u32[89] = (inputs.lightTreeEnabled ?? 0) >>> 0; // offset 356 — lightTreeEnabled
  u32[90] = (inputs.lightTreeNodeCount ?? 0) >>> 0; // offset 360 — lightTreeNodeCount
  // u32[91] = _ppgPad2 (zero).

  device.queue.writeBuffer(uboBuffer, 0, data);
}
