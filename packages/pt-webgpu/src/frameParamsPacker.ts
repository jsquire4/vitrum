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
import { FrameParamsSlot } from './scene/frameParamsLayout.js';
import { invertMat4, multiplyMat4 } from './math/mat4.js';
import { X_CMF_INTEGRAL, Y_CMF_INTEGRAL, Z_CMF_INTEGRAL } from '@vitrum/shared-samplers';
import type { PtWebgpuTraceTier } from './traceTier.js';

/**
 * The subset of {@link import('./scene/uploadSceneBuffers.js').UploadedSceneBuffers}
 * the params packer reads. Narrowed to exactly the fields touched so the packer
 * has no hidden coupling to the full upload struct.
 */
export interface FrameParamsSceneInputs {
  readonly triangleCount: number;
  readonly bvhNodeCount: number;
  readonly analyticCount: number;
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
  readonly environmentTint: readonly [number, number, number];
  readonly environmentSunDirection: readonly [number, number, number];
  readonly environmentSunStrength: number;
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
  paramsU32[FrameParamsSlot.analyticCount] = sb.analyticCount >>> 0;
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
  paramsU32[FrameParamsSlot.heroStrategy] = 0;
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
  paramsF32[FrameParamsSlot.cameraPos] = input.cameraPosition[0];
  paramsF32[FrameParamsSlot.cameraPos + 1] = input.cameraPosition[1];
  paramsF32[FrameParamsSlot.cameraPos + 2] = input.cameraPosition[2];
  paramsF32[FrameParamsSlot.cameraPos + 3] = 1;
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
  paramsF32[FrameParamsSlot.environmentTint + 3] = 0;
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
