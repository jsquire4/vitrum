// FrameParamsPacker — the per-frame uniform-buffer (UBO) assembly extracted
// verbatim from PTEngineWebGPU.#buildParamsBuffer (Task 4.3, Theme A).
//
// CONTRACT: the packed bytes MUST be byte-identical to the pre-extraction
// `#buildParamsBuffer`. pathTraceBruteforce.wgsl's `params` struct reads these
// exact offsets, so this is a load-bearing GPU layout. The engine still OWNS
// its state (#scene / #sceneBuffers / config); this pure function operates on
// that state passed in (it does not duplicate or own engine state).
import type { FrameInput } from '@vitrum/core';
import { asMat4, resolveFrameCameraPosition } from '@vitrum/core';
import { FrameParamsSlot, FRAME_PARAMS_BYTE_SIZE } from './scene/frameParamsLayout.js';
import { invertMat4, multiplyMat4 } from './math/mat4.js';
import type { PtWebgpuTraceTier } from './traceTier.js';
import {
  assertPtWebgpuDistantLaunchDiskRepresentable,
  ptWebgpuRayOriginBias,
  ptWebgpuRayTMin,
  resolvePtWebgpuSceneRadius,
} from './scene/sceneScalePolicy.js';
import {
  assertPtWebgpuEnvironmentScaleF32,
  packPtWebgpuEnvironmentRotationF32,
} from './environmentRadianceScale.js';

// The CPU payload, GPU buffer, and generated WGSL struct have one exact size.
// Keeping a larger fixed allocation would silently retain dead per-frame payload.
export const FRAME_PARAMS_BUFFER_ALLOC_BYTES = FRAME_PARAMS_BYTE_SIZE;
export const FRAME_PARAMS_MAX_SLOT = FRAME_PARAMS_BYTE_SIZE / 4 - 1;
if (
  !Number.isInteger(FRAME_PARAMS_MAX_SLOT) ||
  FRAME_PARAMS_BYTE_SIZE % 16 !== 0
) {
  throw new Error(
    `frameParamsPacker: generated FrameParams size (${FRAME_PARAMS_BYTE_SIZE} B) ` +
    'must be a 16-byte-aligned whole number of f32/u32 slots.',
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
  /** Current scene bounds center, used by BDPT pseudo-distant emitters. */
  readonly sceneCenter: readonly [number, number, number];
  /** Half diagonal of current scene bounds, used to scale BDPT pseudo-distant emitters. */
  readonly sceneRadius: number;
  readonly environmentTint: readonly [number, number, number];
  /** H14-E: map-backed environment-radiance intensity lane. */
  readonly environmentHdriIntensity: number;
  /** Exact represented PMF for the optional environment distant-light candidate. */
  readonly environmentDistantProposalPmf: number;
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
  readonly directLightingMode: 'sampled-selection' | 'summed-expectation';
}

/**
 * Pack the generated-size per-frame uniform buffer (vec4-aligned). Layout is
 * pinned and pathTraceBruteforce.wgsl's `params` struct reads from these
 * exact offsets. Callers must have already validated that the scene buffers
 * are non-null (renderFrame's preconditions handle this).
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
  const cameraPosition = resolveFrameCameraPosition(
    input,
    'PTEngineWebGPU.renderFrame',
  );
  const sceneRadius = resolvePtWebgpuSceneRadius(
    sb.sceneCenter,
    sb.sceneRadius,
  );
  const hasDistantEmitter =
    sb.directionalLightCount > 0 || sb.hasEnvironmentMap;
  const usesDistantLaunchDisk =
    (config.bdpt && config.traceTier === 'full') ||
    (config.causticStrategy === 'photon-map' && config.traceTier === 'full');
  if (hasDistantEmitter && usesDistantLaunchDisk) {
    assertPtWebgpuDistantLaunchDiskRepresentable(sceneRadius);
  }

  const paramsArrayBuffer = new ArrayBuffer(FRAME_PARAMS_BUFFER_ALLOC_BYTES);
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
  paramsF32[FrameParamsSlot.triIntersectEpsilon] =
    ptWebgpuRayTMin(sceneRadius);
  paramsU32[FrameParamsSlot.tlasNodeCount] = sb.tlasNodeCount >>> 0;
  paramsU32[FrameParamsSlot.spectralEnabled] = config.spectralEnabled ? 1 : 0;
  // Spectral kernels sample a hero wavelength per path invocation. These lanes
  // remain neutral for layout compatibility and non-kernel diagnostic readers.
  paramsF32[FrameParamsSlot.heroLambdaNm] = 550.0;
  paramsF32[FrameParamsSlot.heroPdf] = 1.0;
  const bdptActive = config.bdpt && config.traceTier === 'full';
  paramsU32[FrameParamsSlot.bdptEnabled] = bdptActive ? 1 : 0;
  paramsU32[FrameParamsSlot.bdptMaxLightBounces] = config.bdptMaxLightBounces >>> 0;
  // Active private eye-prefix depth (fixed shader capacity <= 8).
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
  paramsF32[FrameParamsSlot.sceneRadius] = sceneRadius;
  paramsU32[FrameParamsSlot.directLightingMode] =
    config.directLightingMode === 'summed-expectation' ? 1 : 0;
  paramsF32[FrameParamsSlot.rayOriginBias] =
    ptWebgpuRayOriginBias(sceneRadius, sb.sceneCenter);
  paramsF32[FrameParamsSlot.environmentDistantProposalPmf] =
    sb.environmentDistantProposalPmf;
  // H14-E: HDRI intensity lives in its own f32 slot (slot 31).
  paramsF32[FrameParamsSlot.environmentHdriIntensity] =
    assertPtWebgpuEnvironmentScaleF32(
      sb.environmentHdriIntensity,
      'frame HDRI intensity',
    );
  paramsF32[FrameParamsSlot.cameraPos] = cameraPosition[0];
  paramsF32[FrameParamsSlot.cameraPos + 1] = cameraPosition[1];
  paramsF32[FrameParamsSlot.cameraPos + 2] = cameraPosition[2];
  paramsF32[FrameParamsSlot.environmentTint] = sb.environmentTint[0];
  paramsF32[FrameParamsSlot.environmentTint + 1] = sb.environmentTint[1];
  paramsF32[FrameParamsSlot.environmentTint + 2] = sb.environmentTint[2];
  // H6: environmentTint.w was always 0 (unused). It now carries environmentHdriRotationY
  // so the WGSL equirect helpers can apply the CCW Y-rotation without a new UBO field.
  // rotationY = 0 → writes 0.0 → WGSL cos(0)=1, sin(0)=0 → identity → zero-rotation invariant.
  paramsF32[FrameParamsSlot.environmentTint + 3] =
    packPtWebgpuEnvironmentRotationF32(
      sb.environmentHdriRotationY,
      'frame HDRI rotationY',
    );
  paramsF32.set(invVp, FrameParamsSlot.invViewProj);
  paramsF32.set(vp, FrameParamsSlot.viewProj);
  const prevVp = multiplyMat4(
    input.prevProjMatrix ?? input.projMatrix,
    input.prevViewMatrix ?? input.viewMatrix,
  );
  paramsF32.set(prevVp, FrameParamsSlot.prevViewProj);
  return paramsArrayBuffer;
}
