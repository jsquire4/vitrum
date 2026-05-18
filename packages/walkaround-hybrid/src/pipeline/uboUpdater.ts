/**
 * UBO updater — writes per-frame camera + lighting + tunables into the
 * 304-byte WalkaroundUBO uniform buffer.
 *
 * W2-C13 follow-up (vec2u vocab): the historical hand-rolled `f32[]`/`u32[]`
 * aliased typed-array packer has been replaced with a single `defineUbo`
 * field-spec. The pre-existing byte layout below is preserved exactly
 * (offset-for-offset, pad-for-pad).
 *
 * Pinned UBO layout (mirrored in `WalkaroundUBO` in common.wgsl):
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
 *   offset 288: _pad                        (u32 = 4 bytes)
 *   offset 292: _pad2                       (u32 = 4 bytes)
 *   offset 296: _pad3                       (u32 = 4 bytes)
 *   offset 300: _pad4                       (u32 = 4 bytes — 16-byte align)
 * Total: 304 bytes (304 % 16 == 0).
 *
 * The four trailing _pad fields are kept explicit because the WGSL struct
 * in common.wgsl declares them by name and downstream readers expect them.
 */

import { defineUbo } from '@vitrum/shared-samplers';
import type { PipelineFrameInputs } from './WalkaroundGPUPipeline.js';

/**
 * Single source of truth for the 304-byte WalkaroundUBO layout.
 *
 * Field order MUST match `struct WalkaroundUBO` in common.wgsl. The four
 * trailing `_padN` fields are intentional: they pad the struct out to 304
 * bytes (a multiple of 16, satisfying the std140 struct-size rule for the
 * vec3f-dominated tail) and match the named pad fields in the WGSL struct.
 */
const WALKAROUND_UBO = defineUbo([
  { name: 'viewMatrix',            type: 'mat4x4f' }, //   0
  { name: 'projMatrix',            type: 'mat4x4f' }, //  64
  { name: 'prevViewMatrix',        type: 'mat4x4f' }, // 128
  { name: 'cameraPos',             type: 'vec3f'   }, // 192
  { name: 'frameSeed',             type: 'u32'     }, // 204
  { name: 'screenSize',            type: 'vec2u'   }, // 208
  { name: 'emitterCount',          type: 'u32'     }, // 216
  { name: 'totalEmPower',          type: 'f32'     }, // 220
  { name: 'primaryLightDir',       type: 'vec3f'   }, // 224
  { name: 'primaryLightIntensity', type: 'f32'     }, // 236
  { name: 'skyTint',               type: 'vec3f'   }, // 240
  { name: 'skyIrradiance',         type: 'f32'     }, // 252
  { name: 'emitterDist2Floor',     type: 'f32'     }, // 256 — audit M12
  { name: 'directFireflyClamp',    type: 'f32'     }, // 260 — audit B4
  { name: 'causticBoost',          type: 'f32'     }, // 264 — audit B1
  { name: 'causticVisClamp',       type: 'f32'     }, // 268 — audit B1
  { name: 'temporalMClampDI',      type: 'u32'     }, // 272 — audit M6
  { name: 'spatialReuseRadiusPx',  type: 'f32'     }, // 276 — audit M7
  { name: 'spatialDepthTolFloor',  type: 'f32'     }, // 280 — audit M8
  { name: 'triIntersectEpsilon',   type: 'f32'     }, // 284 — D12
  // Explicit trailing pad — matches the WGSL struct's _pad/_pad2/_pad3/_pad4
  // fields and brings the struct to 304 bytes (304 % 16 == 0).
  { name: '_pad',                  type: 'u32'     }, // 288
  { name: '_pad2',                 type: 'u32'     }, // 292
  { name: '_pad3',                 type: 'u32'     }, // 296
  { name: '_pad4',                 type: 'u32'     }, // 300
] as const);

/**
 * Size of the WalkaroundUBO in bytes. Exported so the GPU buffer allocator
 * can match. Sourced from the codegen so it can never drift from the
 * actual packed layout.
 */
export const WALKAROUND_UBO_SIZE_BYTES = WALKAROUND_UBO.sizeBytes;

// Compile-time check: the layout must still be the pinned 304 bytes the
// resourceManager allocator and shader consumers expect.
if (WALKAROUND_UBO_SIZE_BYTES !== 304) {
  throw new Error(
    `WalkaroundUBO size drifted from pinned 304 bytes (got ${WALKAROUND_UBO_SIZE_BYTES}).` +
    ` Update the resourceManager allocator and common.wgsl struct in lockstep.`,
  );
}

// Per-frame allocation avoidance — the codegen pack() writes into a
// DataView; we reuse a single ArrayBuffer across frames to keep the
// per-frame allocation pattern identical to the pre-W2-C13 packer.
const FRAME_BUFFER = new ArrayBuffer(WALKAROUND_UBO_SIZE_BYTES);
const FRAME_VIEW = new DataView(FRAME_BUFFER);

/**
 * Coerce a 16-element Float32Array into the 16-tuple shape `defineUbo`'s
 * mat4x4f field expects. The codegen `pack` indexes via `[i]` so a
 * Float32Array works at runtime; this cast is purely structural for the
 * TypeScript type checker.
 */
type Mat4Tuple = readonly [
  number, number, number, number,
  number, number, number, number,
  number, number, number, number,
  number, number, number, number,
];

export function updateUBO(
  device: GPUDevice,
  uboBuffer: GPUBuffer,
  inputs: PipelineFrameInputs,
): void {
  WALKAROUND_UBO.pack(FRAME_VIEW, 0, {
    viewMatrix:            inputs.viewMatrix     as unknown as Mat4Tuple,
    projMatrix:            inputs.projMatrix     as unknown as Mat4Tuple,
    prevViewMatrix:        inputs.prevViewMatrix as unknown as Mat4Tuple,
    cameraPos:             inputs.cameraPos,
    frameSeed:             inputs.frameSeed >>> 0,
    screenSize:            [inputs.screenWidth, inputs.screenHeight] as const,
    emitterCount:          inputs.emitterCount,
    totalEmPower:          inputs.totalEmissivePower,
    primaryLightDir:       inputs.primaryLightDir,
    primaryLightIntensity: inputs.primaryLightIntensity,
    skyTint:               inputs.skyTint,
    skyIrradiance:         inputs.skyIrradiance,
    emitterDist2Floor:     inputs.emitterDist2Floor,
    directFireflyClamp:    inputs.directFireflyClamp,
    causticBoost:          inputs.causticBoost,
    causticVisClamp:       inputs.causticVisClamp,
    temporalMClampDI:      inputs.temporalMClampDI >>> 0,
    spatialReuseRadiusPx:  inputs.spatialReuseRadiusPx,
    spatialDepthTolFloor:  inputs.spatialDepthTolFloor,
    triIntersectEpsilon:   inputs.triIntersectEpsilon,
    _pad: 0, _pad2: 0, _pad3: 0, _pad4: 0,
  });
  device.queue.writeBuffer(uboBuffer, 0, FRAME_BUFFER);
}
