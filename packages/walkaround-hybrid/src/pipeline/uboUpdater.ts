/**
 * UBO updater — writes per-frame camera + lighting + tunables into the
 * 432-byte WalkaroundUBO uniform buffer (see WALKAROUND_UBO_SIZE_BYTES).
 *
 * UBO layout (mixed f32 / u32 — see WalkaroundUBO struct in common.wgsl):
 *   offset   0: viewMatrix                  (mat4×4f = 64 bytes)
 *   offset  64: projMatrix                  (64 bytes)
 *   offset 128: prevViewProjMatrix          (64 bytes)
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
 *   offset 316: frameParity                 (u32 = 4 bytes) — checkerboard frame phase (was _padPreVec3)
 *   offset 320: indirectFireflyClamp        (vec3f = 12 bytes) — sweep 2026-05-18
 *   offset 332: checkerboardOn              (u32 = 4 bytes) — checkerboard sparse-shade gate (was _padEnd)
 *   offset 336: bvhMode                     (u32 = 4 bytes) — PR-3
 *   offset 340: tlasNodeCount               (u32 = 4 bytes)
 *   offset 344: stainedGlassFlags           (u32 = 4 bytes) — shade flags bitfield
 *   offset 348: ppgEnabled                  (u32 = 4 bytes) — PPG guided-sampling gate (was _tracePad1)
 *   offset 352: ppgMixAlpha                 (f32 = 4 bytes) — PPG MIS mixing weight α
 *   offset 356: lightTreeEnabled            (u32 = 4 bytes) — DI light-tree selection gate (was _ppgPad0)
 *   offset 360: lightTreeNodeCount          (u32 = 4 bytes) — packed light-tree node count (was _ppgPad1)
 *   offset 364: nrcEnabled                  (u32 = 4 bytes) — NRC cache gate (was _ppgPad2)
 *   offset 368: regirOrigin                 (vec3f = 12 bytes) — ReGIR grid AABB min
 *   offset 380: regirInvCellSize            (f32 = 4 bytes) — 1 / cellSize
 *   offset 384: regirDims                   (vec3u = 12 bytes) — grid cell counts
 *   offset 396: regirEnabled                (u32 = 4 bytes) — ReGIR DI-selection gate
 *   offset 400: regirCandidatesPerCell      (u32 = 4 bytes) — M per sub-reservoir
 *   offset 404: regirSurvivorsPerCell       (u32 = 4 bytes) — K survivors per cell
 *   offset 408: regirGridFloatOffset        (u32 = 4 bytes) — grid-region float offset in combined buffer
 *   offset 412: restirPtReuse               (u32 = 4 bytes) — GRIS reconnection-shift reuse gate (was _regirPad)
 *   offset 416: sunAngular.x                (f32 = 4 bytes) — direct sun cone radius in radians
 *   offset 420: sunAngular.yzw              (3×f32 = 12 bytes) — padding / future sun controls
 * Total: 432 bytes (432 % 16 == 0).
 */

import type { PipelineFrameInputs } from './WalkaroundGPUPipeline.js';
import {
  WALKAROUND_DEFAULT_SUN_ANGULAR_RADIUS,
  WALKAROUND_UBO_SIZE_BYTES,
} from './constants.js';

/**
 * T5 — shade flag bit masks packed into `stainedGlassFlags`. Bits 0/1 are the
 * stained-glass opt-ins; bit 2 disables the direct-sun visibility ray when a
 * scene directional emitter sets `castShadow:false`. MUST match the constants
 * in `shaders/walkaroundUbo.wgsl.ts`.
 */
export const SG_FLAG_SUN_CAUSTIC = 1; // bit 0
export const SG_FLAG_SKY_APERTURE = 2; // bit 1
export const SHADE_FLAG_DIRECT_SUN_SHADOW_DISABLED = 4; // bit 2

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

/**
 * Live ReGIR grid state injected by the pipeline (NOT part of the host
 * {@link PipelineFrameInputs} contract — derived from the scene bounds +
 * resolved options by the ReGIR coordinator).
 *
 * `enabled` is `true` only when the host opted in AND the grid-build pipeline
 * compiled AND the light tree is live (ReGIR seeds cells via the tree). When
 * `enabled` is false every field is written as a zero/placeholder and the
 * kernel-side `regirEnabled == 0` gate makes RIS fall back to the light-tree
 * path bit-identically (and the grid-build pass early-returns).
 */
export interface RegirUboState {
  readonly enabled: boolean;
  /** World-space grid AABB min. */
  readonly origin: readonly [number, number, number];
  /** 1 / cellSize (uniform cubic cells). */
  readonly invCellSize: number;
  /** Grid cell counts (x, y, z). */
  readonly dims: readonly [number, number, number];
  /** M — WRS candidates per sub-reservoir. */
  readonly candidatesPerCell: number;
  /** K — survivors stored per cell. */
  readonly survivorsPerCell: number;
  /** Float offset of the grid region in the combined light-tree buffer
   *  (= lightTreeNodeCount × LIGHT_TREE_FLOATS_PER_NODE). */
  readonly gridFloatOffset: number;
}

/**
 * Live checkerboard half-res-shading state injected by the pipeline (NOT part of
 * the host {@link PipelineFrameInputs} contract — it is a pipeline-resolved
 * structural flag plus the per-frame parity phase).
 *
 * `enabled` mirrors the pipeline's `_checkerboard` field (the host opt-in). When
 * `enabled` is false BOTH packed fields stay 0, so the UBO is byte-identical to
 * the pre-checkerboard layout and `shadeMain`'s gap early-out is never taken.
 * `frameParity` is the SAME `frameCount & 1` phase ResolvePass writes into the
 * separate ResolveUniforms buffer, so the shade gap pixels match the resolve
 * gap-fill pixels exactly.
 */
export interface CheckerboardUboState {
  readonly enabled: boolean;
  /** frameCount & 1 — the checkerboard phase that is "shaded" this frame. */
  readonly frameParity: number;
}

/** Checkerboard-OFF default — both fields zero ⇒ shadeMain shades every pixel
 *  and the packed UBO is byte-identical to the pre-checkerboard layout. */
const CHECKERBOARD_OFF: CheckerboardUboState = { enabled: false, frameParity: 0 };

/** ReGIR-OFF default — every field zero ⇒ the kernel's `regirEnabled == 0`
 *  gate keeps RIS on the light-tree path bit-for-bit. */
const REGIR_OFF: RegirUboState = {
  enabled: false,
  origin: [0, 0, 0],
  invCellSize: 0,
  dims: [0, 0, 0],
  candidatesPerCell: 0,
  survivorsPerCell: 0,
  gridFloatOffset: 0,
};

/**
 * Pure packing core — fills and returns the 416-byte WalkaroundUBO ArrayBuffer
 * from structured inputs. No GPU types involved; safe to call in Node / Vitest.
 *
 * Extracted so the sentinel round-trip test (I3.3/D3.17) can verify every
 * field's packer index against the WGSL offset comment without requiring a
 * real GPUDevice. The existing `updateUBO` delegates here and calls
 * `device.queue.writeBuffer` on the result — byte-identical behaviour.
 */
export function packWalkaroundUBO(
  inputs: PipelineFrameInputs,
  ppg: PpgUboState = { enabled: false, mixAlpha: 0 },
  regir: RegirUboState = REGIR_OFF,
  checkerboard: CheckerboardUboState = CHECKERBOARD_OFF,
): ArrayBuffer {
  const data = new ArrayBuffer(WALKAROUND_UBO_SIZE_BYTES);
  const f32  = new Float32Array(data);
  const u32  = new Uint32Array(data);

  f32.set(inputs.camera.viewMatrix,     0);    //  0..15 (64 bytes)
  f32.set(inputs.camera.projMatrix,    16);    // 16..31 (64 bytes)
  f32.set(inputs.camera.prevViewProjMatrix, 32); // 32..47 (64 bytes)
  f32[48] = inputs.camera.cameraPos[0];
  f32[49] = inputs.camera.cameraPos[1];
  f32[50] = inputs.camera.cameraPos[2];
  u32[51] = inputs.screen.frameSeed >>> 0;
  u32[52] = inputs.screen.screenWidth;
  u32[53] = inputs.screen.screenHeight;
  u32[54] = inputs.lighting.emitterCount;
  f32[55] = inputs.lighting.totalEmissivePower;
  f32[56] = inputs.lighting.primaryLightDir[0];
  f32[57] = inputs.lighting.primaryLightDir[1];
  f32[58] = inputs.lighting.primaryLightDir[2];
  f32[59] = inputs.lighting.primaryLightIntensity;
  f32[60] = inputs.lighting.skyTint[0];
  f32[61] = inputs.lighting.skyTint[1];
  f32[62] = inputs.lighting.skyTint[2];
  f32[63] = inputs.lighting.skyIrradiance;
  // Library-generality tunables (audit follow-up).
  f32[64] = inputs.lighting.emitterDist2Floor;
  f32[65] = inputs.lighting.directFireflyClamp;
  f32[66] = inputs.lighting.causticBoost;
  f32[67] = inputs.lighting.causticVisClamp;
  u32[68] = inputs.restirDI.temporalMClampDI >>> 0;
  f32[69] = inputs.restirDI.spatialReuseRadiusPx;
  f32[70] = inputs.restirDI.spatialDepthTolFloor;
  f32[71] = inputs.filter.triIntersectEpsilon;
  // 2026-05-18 sweep — eight more Cornell-tuned magic constants threaded
  // through the UBO for library-consumer override.
  f32[72] = inputs.filter.glassMixScale;
  f32[73] = inputs.restirGI.restirGiWCap;
  f32[74] = inputs.restirGI.restirGiIrrClamp;
  u32[75] = inputs.restirGI.restirGiMClamp >>> 0;
  f32[76] = inputs.restirGI.restirGiSpatialRadiusPx;
  f32[77] = inputs.restirGI.restirGiSpatialNormalDotMin;
  f32[78] = inputs.restirGI.restirGiSpatialCoplanarTol;
  // u32[79] = frameParity (offset 316 — the former _padPreVec3 slot). The
  // checkerboard frame phase (frameCount & 1) the shade gap early-out compares
  // against. 0 when checkerboard is OFF, so the slot stays zero (byte-identity)
  // and indirectFireflyClamp below remains vec3-aligned.
  u32[79] = checkerboard.enabled ? (checkerboard.frameParity & 1) >>> 0 : 0;
  f32[80] = inputs.filter.indirectFireflyClamp[0];
  f32[81] = inputs.filter.indirectFireflyClamp[1];
  f32[82] = inputs.filter.indirectFireflyClamp[2];
  // u32[83] = checkerboardOn (offset 332 — the former _padEnd slot). 0 ⇒
  // shadeMain shades EVERY pixel (gap early-out never taken ⇒ bit-identity); 1 ⇒
  // shadeMain skips the gap pixels for resolve.wgsl to reproject. Absent ⇒ 0
  // (OFF), so callers + existing tests that never set it stay byte-identical.
  u32[83] = checkerboard.enabled ? 1 : 0;
  u32[84] = inputs.bvh.bvhMode >>> 0;
  u32[85] = inputs.bvh.tlasNodeCount >>> 0;
  // T5 — stained-glass opt-in flag bits (repurposed _tracePad0 at offset 344).
  // 0 → both terms OFF (generic-scene default).
  u32[86] = inputs.filter.stainedGlassFlags >>> 0;
  // PPG guided sampling (W9 guided-sampling landing). offset 348 = ppgEnabled
  // (gate), offset 352 = ppgMixAlpha. When PPG is off, ppgEnabled stays 0 and
  // α stays 0 → gi-ris RIS source pdf = cosθ/π exactly (ppg-OFF bit-identity).
  u32[87] = ppg.enabled ? 1 : 0; //  offset 348 — ppgEnabled
  f32[88] = ppg.enabled ? ppg.mixAlpha : 0; // offset 352 — ppgMixAlpha
  // Light-tree DI light-SELECTION gate (offset 356) + node count (offset 360).
  // When disabled both stay 0 → RIS uses the flat power-CDF path exactly.
  u32[89] = (inputs.lighting.lightTreeEnabled ?? 0) >>> 0; // offset 356 — lightTreeEnabled
  u32[90] = (inputs.lighting.lightTreeNodeCount ?? 0) >>> 0; // offset 360 — lightTreeNodeCount
  // NRC cache flag (offset 364 — the former _ppgPad2 slot). 0 keeps the gi-ris
  // suffix on the verbatim DDGI-atlas estimate (NRC-OFF bit-identity); 1 marks
  // the neural radiance cache on. Absent ⇒ 0 (OFF), so callers and existing
  // tests that never set it are byte-identical to before. NOTE: this UBO field
  // is an informational mirror — no shader reads u32[91]. The load-bearing gate
  // is compile-time (the risGiNrc variant is composed only when nrcEnabled). See V20.
  u32[91] = (inputs.nrc.nrcEnabled ?? 0) >>> 0; // offset 364 — nrcEnabled (was _ppgPad2)
  // ReGIR grid state (offsets 368..412). When ReGIR is off every field is 0,
  // so the kernel's `regirEnabled == 0` gate keeps RIS on the light-tree path
  // bit-for-bit (and the grid-build pass early-returns).
  const r = regir.enabled ? regir : REGIR_OFF;
  f32[92] = r.origin[0];       // offset 368 — regirOrigin.x
  f32[93] = r.origin[1];       // offset 372 — regirOrigin.y
  f32[94] = r.origin[2];       // offset 376 — regirOrigin.z
  f32[95] = r.invCellSize;     // offset 380 — regirInvCellSize
  u32[96] = r.dims[0] >>> 0;   // offset 384 — regirDims.x
  u32[97] = r.dims[1] >>> 0;   // offset 388 — regirDims.y
  u32[98] = r.dims[2] >>> 0;   // offset 392 — regirDims.z
  u32[99] = r.enabled ? 1 : 0; // offset 396 — regirEnabled
  u32[100] = r.candidatesPerCell >>> 0; // offset 400 — regirCandidatesPerCell (M)
  u32[101] = r.survivorsPerCell >>> 0;  // offset 404 — regirSurvivorsPerCell (K)
  u32[102] = r.gridFloatOffset >>> 0;   // offset 408 — regirGridFloatOffset
  // GRIS / ReSTIR-PT reconnection-shift reuse gate (offset 412 — the former
  // _regirPad slot). 0 keeps the GI spatial/temporal reuse on the legacy
  // clamped-Jacobian path bit-for-bit; 1 turns on the unbiased GRIS shift +
  // reconnection visibility + pairwise MIS. Absent ⇒ 0 (OFF), so callers and
  // existing tests that never set it are byte-identical to before.
  u32[103] = (inputs.restirGI.restirPtReuse ?? 0) >>> 0; // offset 412 — restirPtReuse
  const sunAngularRadius = inputs.lighting.sunAngularRadius;
  f32[104] = typeof sunAngularRadius === 'number' && Number.isFinite(sunAngularRadius)
    ? Math.max(0, sunAngularRadius)
    : WALKAROUND_DEFAULT_SUN_ANGULAR_RADIUS; // offset 416 — sunAngular.x
  f32[105] = 0; // offset 420 — sunAngular.y reserved
  f32[106] = 0; // offset 424 — sunAngular.z reserved
  f32[107] = 0; // offset 428 — sunAngular.w reserved

  return data;
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
  /** Live ReGIR grid state from the pipeline. Defaults to OFF so callers that
   *  don't run ReGIR (and the existing tests) keep the light-tree DI path
   *  bit-identically (regirEnabled=0). */
  regir: RegirUboState = REGIR_OFF,
  /** Live checkerboard half-res-shading state from the pipeline. Defaults to OFF
   *  so callers that don't run checkerboard (and the existing tests) keep the
   *  full-shade path with frameParity=0/checkerboardOn=0 — both pad slots stay
   *  zero, so the UBO is byte-identical to the pre-checkerboard layout. */
  checkerboard: CheckerboardUboState = CHECKERBOARD_OFF,
): void {
  const data = packWalkaroundUBO(inputs, ppg, regir, checkerboard);
  device.queue.writeBuffer(uboBuffer, 0, data);
}
