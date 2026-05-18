import type { FrameInput } from '@vitrum/core';
import { invertMat4, multiplyMat4 } from '../math/mat4.js';
import type { UploadedSceneBuffers } from '../scene/uploadSceneBuffers.js';

/**
 * Packs the per-frame uniform buffer (FrameParams UBO) for `PTEngineWebGPU`.
 *
 * Layout is pinned and pathTraceBruteforce.wgsl's `params` struct reads from
 * these exact offsets — see `frameParamsLayout.test.ts` for the contract. The
 * packer was extracted from `PTEngineWebGPU.#buildParamsBuffer` in W4-A9
 * without changing a single offset: the buffer is byte-for-byte identical.
 *
 * The buffer is 512 bytes; the first 336 bytes are populated and the trailing
 * 176 bytes are zero (ArrayBuffer-init contract).
 *
 *   u32 slot 0..1   width, height
 *   u32 slot 2..3   frameIndex, frameSeed
 *   u32 slot 4..7   triangleCount, activeBounces, bvhNodeCount, analyticCount
 *   u32 slot 8..11  pointLightCount, spotLightCount, rectAreaLightCount,
 *                   meshAreaLightCount
 *   u32 slot 12..13 mneeMaxIterations, mneeMaxChainLength
 *   u32 slot 14..15 hasEnvironmentMap (0/1), causticStrategy
 *                   (0=none, 1=manifold-nee, 2=photon-map)
 *   u32 slot 16..17 environmentMapWidth, environmentMapHeight
 *   f32 slot 18     triIntersectEpsilon (default 1e-5; metre-scale)
 *   u32 slot 19     _pad1   (zero; reserved)
 *
 *   f32 slot 20..23 cameraPos.xyz + 1.0
 *   f32 slot 24..27 lightDir.xyz + averageDirectionalIrradiance
 *   f32 slot 28..31 environmentTint.xyz + 0   (.w unused, write 0)
 *   f32 slot 32..35 environmentSun.xyz (sun dir) + sun strength
 *
 *   f32 slot 36..51 invViewProj (mat4x4f, 16 floats)
 *   f32 slot 52..67 viewProj    (mat4x4f, 16 floats)
 *   f32 slot 68..83 prevViewProj(mat4x4f, 16 floats)
 *
 * Per-light data lives in dedicated storage buffers at bind slots 20..23;
 * see `packEmitterArrays` for the layout (8 f32 / point light, 12 / spot,
 * 16 / rect-area, 16 / mesh-area).
 */

export interface PTFrameParamsInputs {
  readonly input: FrameInput;
  readonly width: number;
  readonly height: number;
  readonly activeBounces: number;
  readonly sceneBuffers: UploadedSceneBuffers;
  readonly mneeMaxIterations: number;
  readonly mneeMaxChainLength: number;
  readonly causticStrategy: 'none' | 'manifold-nee' | 'photon-map';
}

export const PARAMS_BUFFER_BYTES = 512;
/**
 * Identity matrix used as the invViewProj fallback when the per-frame
 * viewProj matrix is singular (mirrors the original inline allocation —
 * `invertMat4` returns null on degenerate input).
 */
function identityMat4(): Float32Array {
  return new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]);
}

export class PTEngineWebGPUParamsPacker {
  /**
   * Allocate and pack a fresh 512-byte ArrayBuffer for this frame. Returns the
   * raw buffer; the caller is responsible for `queue.writeBuffer`-ing it into
   * the params UBO.
   */
  pack(inputs: PTFrameParamsInputs): ArrayBuffer {
    const {
      input,
      width,
      height,
      activeBounces,
      sceneBuffers: sb,
      mneeMaxIterations,
      mneeMaxChainLength,
      causticStrategy,
    } = inputs;
    const vp = multiplyMat4(input.projMatrix, input.viewMatrix);
    const invVp = invertMat4(vp) ?? identityMat4();

    const paramsArrayBuffer = new ArrayBuffer(PARAMS_BUFFER_BYTES);
    const paramsU32 = new Uint32Array(paramsArrayBuffer);
    const paramsF32 = new Float32Array(paramsArrayBuffer);
    paramsU32[0] = width;
    paramsU32[1] = height;
    paramsU32[2] = input.frameIndex >>> 0;
    paramsU32[3] = input.frameSeed >>> 0;
    paramsU32[4] = sb.triangleCount >>> 0;
    paramsU32[5] = activeBounces >>> 0;
    paramsU32[6] = sb.bvhNodeCount >>> 0;
    paramsU32[7] = sb.analyticCount >>> 0;
    paramsU32[8] = sb.pointLightCount >>> 0;
    paramsU32[9] = sb.spotLightCount >>> 0;
    paramsU32[10] = sb.rectAreaLightCount >>> 0;
    paramsU32[11] = sb.meshAreaLightCount >>> 0;
    paramsU32[12] = mneeMaxIterations >>> 0;
    paramsU32[13] = mneeMaxChainLength >>> 0;
    paramsU32[14] = sb.hasEnvironmentMap ? 1 : 0;
    paramsU32[15] =
      causticStrategy === 'manifold-nee'
        ? 1
        : causticStrategy === 'photon-map'
          ? 2
          : 0;
    paramsU32[16] = sb.environmentMapWidth >>> 0;
    paramsU32[17] = sb.environmentMapHeight >>> 0;
    paramsF32[18] = 1e-5; // triIntersectEpsilon: default metre-scale (D12)
    // Slot 19 (_pad1) is padding; zero-initialized by ArrayBuffer.
    paramsF32[20] = input.cameraPosition[0];
    paramsF32[21] = input.cameraPosition[1];
    paramsF32[22] = input.cameraPosition[2];
    paramsF32[23] = 1;
    paramsF32[24] = sb.directionalLight[0];
    paramsF32[25] = sb.directionalLight[1];
    paramsF32[26] = sb.directionalLight[2];
    paramsF32[27] =
      (sb.directionalIrradiance[0] +
        sb.directionalIrradiance[1] +
        sb.directionalIrradiance[2]) /
      3;
    paramsF32[28] = sb.environmentTint[0];
    paramsF32[29] = sb.environmentTint[1];
    paramsF32[30] = sb.environmentTint[2];
    paramsF32[31] = 0;
    paramsF32[32] = sb.environmentSunDirection[0];
    paramsF32[33] = sb.environmentSunDirection[1];
    paramsF32[34] = sb.environmentSunDirection[2];
    paramsF32[35] = sb.environmentSunStrength;
    paramsF32.set(invVp, 36);
    paramsF32.set(vp, 52);
    const prevVp = multiplyMat4(
      input.prevProjMatrix ?? input.projMatrix,
      input.prevViewMatrix ?? input.viewMatrix,
    );
    paramsF32.set(prevVp, 68);
    return paramsArrayBuffer;
  }
}
