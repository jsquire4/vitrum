// FrameParamsPacker — the per-frame uniform-buffer (UBO) assembly extracted
// verbatim from PTEngineWebGPU.#buildParamsBuffer (Task 4.3, Theme A).
//
// CONTRACT: the packed bytes MUST be byte-identical to the pre-extraction
// `#buildParamsBuffer`. pathTraceBruteforce.wgsl's `params` struct reads these
// exact offsets, so this is a load-bearing GPU layout. The engine still OWNS
// its state (#scene / #sceneBuffers / config); this pure function operates on
// that state passed in (it does not duplicate or own engine state).
import type { FrameInput } from '@vitrum/core';
import { asMat4 } from '@vitrum/core';
import { FrameParamsSlot, FRAME_PARAMS_BYTE_SIZE } from './scene/frameParamsLayout.js';
import { invertMat4, multiplyMat4 } from './math/mat4.js';
import { X_CMF_INTEGRAL, Y_CMF_INTEGRAL, Z_CMF_INTEGRAL } from '@vitrum/shared-samplers';
import type { PtWebgpuTraceTier } from './traceTier.js';

// D8.11 — Module-load-time allocation guard.
//
// The buffer is always allocated as 512 bytes (128 u32/f32 slots, indices 0..127).
// FRAME_PARAMS_BYTE_SIZE is the WGSL-derived padded size of the struct
// (auto-generated; currently 416 bytes = 104 slots).  If the generator
// adds fields and the struct grows past 512 bytes, writes at slot ≥ 128 would
// silently go out of bounds.  Fail loudly here at module load instead.
//
// The checked invariant: FRAME_PARAMS_BYTE_SIZE must not exceed the 512-byte
// ArrayBuffer that `packFrameParams` allocates.
export const FRAME_PARAMS_BUFFER_ALLOC_BYTES = 512;
if (FRAME_PARAMS_BYTE_SIZE > FRAME_PARAMS_BUFFER_ALLOC_BYTES) {
  throw new Error(
    `frameParamsPacker: FRAME_PARAMS_BYTE_SIZE (${FRAME_PARAMS_BYTE_SIZE} B) exceeds the ` +
    `allocated buffer size (${FRAME_PARAMS_BUFFER_ALLOC_BYTES} B). ` +
    `Run tools/generate-wgsl-layouts.mjs and update the ArrayBuffer allocation.`,
  );
}
// Also guard the highest *named* slot index used by the packer.  Each slot is 4
// bytes; the 512-byte buffer accommodates slots 0..127.  If a generator update
// moves the highest slot past 127 without bumping the allocation, a silent OOB
// write would corrupt the GPU uniform.  We export this constant so tests can pin it.
export const FRAME_PARAMS_MAX_SLOT = 127;
const _highestSlot = Math.max(...(Object.values(FrameParamsSlot) as number[]));
// For vec4f/mat4x4f fields the slot value is the first slot; add their size:
//   invViewProj at slot 48, mat4x4f = 16 slots → last slot 63
//   prevViewProj at slot 80, mat4x4f = 16 slots → last slot 95
//   directionalLightCount at slot 96, u32 = 1 slot → last slot 96
//   sceneCenterX/Y/Z + sceneRadius at slots 97..100 → last slot 100
// The last slot actually WRITTEN is sceneRadius = 100. The mat fields span
// slots [48..95].
// We let the generator-derived FRAME_PARAMS_BYTE_SIZE / 4 give us the effective
// slot count (accounting for trailing struct padding):
const _effectiveSlots = FRAME_PARAMS_BYTE_SIZE / 4; // e.g. 416/4 = 104
if (_effectiveSlots > FRAME_PARAMS_MAX_SLOT + 1) {
  throw new Error(
    `frameParamsPacker: FRAME_PARAMS_BYTE_SIZE implies ${_effectiveSlots} slots but ` +
    `the buffer only holds ${FRAME_PARAMS_MAX_SLOT + 1} (0..${FRAME_PARAMS_MAX_SLOT}). ` +
    `Update FRAME_PARAMS_MAX_SLOT or bump the ArrayBuffer allocation.`,
  );
}

/**
 * The subset of {@link import('./scene/uploadSceneBuffers.js').UploadedSceneBuffers}
 * the params packer reads. Narrowed to exactly the fields touched so the packer
 * has no hidden coupling to the full upload struct.
 */
export interface FrameParamsSceneInputs {
  readonly triangleCount: number;
  readonly bvhNodeCount: number;
  readonly analyticCount: number;
  /** N-directional: total packed directional count (kernel loops this many storage-buffer records). */
  readonly directionalLightCount: number;
  readonly pointLightCount: number;
  readonly spotLightCount: number;
  readonly rectAreaLightCount: number;
  readonly meshAreaLightCount: number;
  readonly hasEnvironmentMap: boolean;
  readonly environmentMapWidth: number;
  readonly environmentMapHeight: number;
  readonly tlasNodeCount: number;
  readonly lightTreeEnabled: boolean;
  readonly lightTreeNodeCount: number;
  readonly directionalLight: readonly [number, number, number];
  readonly directionalIrradiance: readonly [number, number, number];
  /** D3 — soft-sun angular diameter in radians (0 = perfect delta directional).
   *  Written to the frame UBO's `cameraPos.w` lane (previously a constant 1, never
   *  read by any shader). 0 keeps the historical exact directional path. */
  readonly directionalAngularDiameter: number;
  /** Current scene bounds center, used by BDPT pseudo-distant emitters. */
  readonly sceneCenter: readonly [number, number, number];
  /** Half diagonal of current scene bounds, used to scale BDPT pseudo-distant emitters. */
  readonly sceneRadius: number;
  readonly environmentTint: readonly [number, number, number];
  readonly environmentSunDirection: readonly [number, number, number];
  readonly environmentSunStrength: number;
  /** H14-E: HDRI-only intensity lane — separate from environmentSunStrength.
   *  Written to params.environmentHdriIntensity so the equirect lookup is not
   *  gated by the procedural-sky sun-strength lane. */
  readonly environmentHdriIntensity: number;
  /**
   * H6: HDRI dome CCW Y-rotation in radians (default 0).
   * Packed into params.environmentTint.w (the previously-zero .w lane — no layout
   * change).  The WGSL equirect lookup rotates the direction by -rotationY before UV;
   * the importance sampler rotates the CDF-sampled direction by +rotationY.
   * Zero means no rotation: byte-identical to pre-H6 behaviour.
   */
  readonly environmentHdriRotationY: number;
}

/**
 * The per-frame engine configuration the packer reads. These are the engine
 * instance fields the original `#buildParamsBuffer` closed over; they are
 * passed in explicitly so the packer is a pure function over engine state.
 */
export interface FrameParamsEngineConfig {
  /** The active per-frame bounce limit (`#activeBounces`). */
  readonly activeBounces: number;
  readonly mneeMaxIterations: number;
  readonly mneeMaxChainLength: number;
  readonly causticStrategy: 'none' | 'manifold-nee' | 'photon-map';
  readonly spectralEnabled: boolean;
  readonly traceTier: PtWebgpuTraceTier;
  readonly bdpt: boolean;
  readonly bdptMaxLightBounces: number;
  readonly lightTreeImportanceSampling: boolean;
}

/**
 * Pack the per-frame uniform buffer (512 bytes, vec4-aligned). Layout is
 * pinned and pathTraceBruteforce.wgsl's `params` struct reads from these
 * exact offsets. Callers must have already validated that the scene buffers
 * are non-null (renderFrame's preconditions handle this).
 *
 * Layout (384 bytes used out of a 512-byte buffer; trailing bytes are zero):
 *
 * Authoritative field layout is auto-generated in scene/frameParamsLayout.generated.ts (FrameParamsSlot).
 *
 * Per-light data lives in dedicated storage buffers at bind slots 20..23;
 * see `packEmitterArrays` for the layout (8 f32 / point light, 12 / spot,
 * 16 / rect-area, 16 / mesh-area).
 */
export function packFrameParams(
  config: FrameParamsEngineConfig,
  sb: FrameParamsSceneInputs,
  input: FrameInput,
  width: number,
  height: number,
): ArrayBuffer {
  const vp = multiplyMat4(input.projMatrix, input.viewMatrix);
  const invVp = invertMat4(asMat4(vp));
  if (invVp == null) {
    throw new Error('renderFrame: non-invertible view-projection matrix');
  }

  const paramsArrayBuffer = new ArrayBuffer(512);
  const paramsU32 = new Uint32Array(paramsArrayBuffer);
  const paramsF32 = new Float32Array(paramsArrayBuffer);
  paramsU32[FrameParamsSlot.width] = width;
  paramsU32[FrameParamsSlot.height] = height;
  paramsU32[FrameParamsSlot.frameIndex] = input.frameIndex >>> 0;
  paramsU32[FrameParamsSlot.frameSeed] = input.frameSeed >>> 0;
  paramsU32[FrameParamsSlot.triangleCount] = sb.triangleCount >>> 0;
  paramsU32[FrameParamsSlot.maxBounces] = config.activeBounces >>> 0;
  paramsU32[FrameParamsSlot.bvhNodeCount] = sb.bvhNodeCount >>> 0;
  // Item 24 — analytic × lite phantom count: the lite kernel has no analytic
  // primitive path (no group-1 analyticHeaders/Params/LocalToWorld/WorldToLocal
  // bindings, no analytic intersection loop). Writing the real analyticCount on
  // lite tier would leave a phantom count in the UBO that a future lite-kernel
  // change could accidentally read and misinterpret. Zero it here at pack time so
  // the UBO never exposes an analytic count to a kernel that cannot use it.
  // Full-tier path: uses the real sb.analyticCount — byte-identical to prior behaviour.
  paramsU32[FrameParamsSlot.analyticCount] =
    config.traceTier === 'lite' ? 0 : sb.analyticCount >>> 0;
  paramsU32[FrameParamsSlot.pointLightCount] = sb.pointLightCount >>> 0;
  paramsU32[FrameParamsSlot.spotLightCount] = sb.spotLightCount >>> 0;
  paramsU32[FrameParamsSlot.rectAreaLightCount] = sb.rectAreaLightCount >>> 0;
  paramsU32[FrameParamsSlot.meshAreaLightCount] = sb.meshAreaLightCount >>> 0;
  paramsU32[FrameParamsSlot.mneeMaxIterations] = config.mneeMaxIterations >>> 0;
  paramsU32[FrameParamsSlot.mneeMaxChainLength] = config.mneeMaxChainLength >>> 0;
  paramsU32[FrameParamsSlot.hasEnvironmentMap] = sb.hasEnvironmentMap ? 1 : 0;
  paramsU32[FrameParamsSlot.causticStrategy] =
    config.causticStrategy === 'manifold-nee'
      ? 1
      : config.causticStrategy === 'photon-map'
        ? 2
        : 0;
  paramsU32[FrameParamsSlot.environmentMapWidth] = sb.environmentMapWidth >>> 0;
  paramsU32[FrameParamsSlot.environmentMapHeight] = sb.environmentMapHeight >>> 0;
  paramsF32[FrameParamsSlot.triIntersectEpsilon] = 1e-5; // triIntersectEpsilon: default metre-scale (D12)
  paramsU32[FrameParamsSlot.tlasNodeCount] = sb.tlasNodeCount >>> 0;
  paramsU32[FrameParamsSlot.spectralEnabled] = config.spectralEnabled ? 1 : 0;
  paramsF32[FrameParamsSlot.heroLambdaNm] = 550.0;
  paramsF32[FrameParamsSlot.heroPdf] = 1.0;
  paramsF32[FrameParamsSlot.cmfIntegralX] = X_CMF_INTEGRAL;
  paramsF32[FrameParamsSlot.cmfIntegralY] = Y_CMF_INTEGRAL;
  paramsF32[FrameParamsSlot.cmfIntegralZ] = Z_CMF_INTEGRAL;
  const bdptActive = config.bdpt && config.traceTier === 'full';
  paramsU32[FrameParamsSlot.bdptEnabled] = bdptActive ? 1 : 0;
  paramsU32[FrameParamsSlot.bdptMaxLightBounces] = config.bdptMaxLightBounces >>> 0;
  // Eye-subpath scratch depth = the active per-pixel bounce limit (<= 8).
  paramsU32[FrameParamsSlot.bdptMaxEyeDepth] = config.activeBounces >>> 0;
  // WS2 — power-weighted light selection. FULL tier only: the lite kernel keeps
  // the uniform pick and never composes the light-tree WGSL / group(3) binding.
  const lightTreeOn = config.traceTier === 'full' && sb.lightTreeEnabled && config.lightTreeImportanceSampling;
  paramsU32[FrameParamsSlot.lightTreeEnabled] = lightTreeOn ? 1 : 0;
  paramsU32[FrameParamsSlot.lightTreeNodeCount] = lightTreeOn ? sb.lightTreeNodeCount >>> 0 : 0;
  // N-directional: kernel loops this many records from the directionalLights storage buffer.
  paramsU32[FrameParamsSlot.directionalLightCount] = sb.directionalLightCount >>> 0;
  paramsF32[FrameParamsSlot.sceneCenterX] = sb.sceneCenter[0];
  paramsF32[FrameParamsSlot.sceneCenterY] = sb.sceneCenter[1];
  paramsF32[FrameParamsSlot.sceneCenterZ] = sb.sceneCenter[2];
  paramsF32[FrameParamsSlot.sceneRadius] = Math.max(1e-3, sb.sceneRadius);
  // H14-E: HDRI intensity lives in its own f32 slot (slot 31 = environmentHdriIntensity),
  // separate from environmentSun.w (procedural sky sun strength). This ensures the HDRI
  // equirect lookup is NOT silently zeroed when sun.w == 0 (e.g. night-only scenes).
  paramsF32[FrameParamsSlot.environmentHdriIntensity] = sb.environmentHdriIntensity;
  paramsF32[FrameParamsSlot.cameraPos] = input.cameraPosition[0];
  paramsF32[FrameParamsSlot.cameraPos + 1] = input.cameraPosition[1];
  paramsF32[FrameParamsSlot.cameraPos + 2] = input.cameraPosition[2];
  // D3 — soft-sun angular diameter (radians) in the previously-constant cameraPos.w
  // lane (no shader reads cameraPos.w; only .xyz is used). 0 ⇒ exact directional.
  paramsF32[FrameParamsSlot.cameraPos + 3] = sb.directionalAngularDiameter;
  paramsF32[FrameParamsSlot.lightDir] = sb.directionalLight[0];
  paramsF32[FrameParamsSlot.lightDir + 1] = sb.directionalLight[1];
  paramsF32[FrameParamsSlot.lightDir + 2] = sb.directionalLight[2];
  paramsF32[FrameParamsSlot.lightDir + 3] =
    (sb.directionalIrradiance[0] +
      sb.directionalIrradiance[1] +
      sb.directionalIrradiance[2]) /
    3;
  paramsF32[FrameParamsSlot.environmentTint] = sb.environmentTint[0];
  paramsF32[FrameParamsSlot.environmentTint + 1] = sb.environmentTint[1];
  paramsF32[FrameParamsSlot.environmentTint + 2] = sb.environmentTint[2];
  // H6: environmentTint.w was always 0 (unused). It now carries environmentHdriRotationY
  // so the WGSL equirect helpers can apply the CCW Y-rotation without a new UBO field.
  // rotationY = 0 → writes 0.0 → WGSL cos(0)=1, sin(0)=0 → identity → zero-rotation invariant.
  paramsF32[FrameParamsSlot.environmentTint + 3] = sb.environmentHdriRotationY;
  paramsF32[FrameParamsSlot.environmentSun] = sb.environmentSunDirection[0];
  paramsF32[FrameParamsSlot.environmentSun + 1] = sb.environmentSunDirection[1];
  paramsF32[FrameParamsSlot.environmentSun + 2] = sb.environmentSunDirection[2];
  paramsF32[FrameParamsSlot.environmentSun + 3] = sb.environmentSunStrength;
  paramsF32.set(invVp, FrameParamsSlot.invViewProj);
  paramsF32.set(vp, FrameParamsSlot.viewProj);
  const prevVp = multiplyMat4(
    input.prevProjMatrix ?? input.projMatrix,
    input.prevViewMatrix ?? input.viewMatrix,
  );
  paramsF32.set(prevVp, FrameParamsSlot.prevViewProj);
  return paramsArrayBuffer;
}
