/**
 * AdjointPass — WS5 Phase-1 path-replay adjoint compute pass.
 *
 * Extracted from PTEngineWebGPU.#computeAdjointGradient (D8.10). Owns:
 *   - Lazy pipeline compilation (PT_WEBGPU_ADJOINT_PASS_WGSL, layout:'auto').
 *   - Per-step transient buffer allocation + guaranteed destroy on success and
 *     on mapAsync rejection (H14-D leak-guard accounting with adjointCreated /
 *     adjointDestroyed counters — these counters may be consumed by telemetry
 *     and tests; the semantics are preserved verbatim).
 *   - UBO packing (invViewProj + cameraPos + counts).
 *   - Dispatch + copyBufferToBuffer readback.
 *
 * The engine holds one lazy instance (`#adjointPass`) and delegates the
 * `computeAdjointGradient` hook to it.  The pipeline is cached for the lifetime
 * of the AdjointPass instance (engine-owned; freed via dispose()).
 */

import { asMat4 } from '@vitrum/core';
import { invertMat4, multiplyMat4 } from './math/mat4.js';
import {
  PT_WEBGPU_ADJOINT_PASS_WGSL,
  ADJOINT_PARAMS_UBO_BYTES,
  ADJOINT_FIELD_BASECOLOR,
  ADJOINT_FIELD_ROUGHNESS,
  ADJOINT_FIELD_EMISSIVE,
  ADJOINT_FIELD_SPECULAR_COLOR,
  ADJOINT_FIELD_SPECULAR_INTENSITY,
} from './wgsl/pathTrace/adjointPass.wgsl.js';
import { ADJOINT_GRAD_FP } from './wgsl/pathTrace/pathTraceAdjoint.wgsl.js';
import type { UploadedSceneBuffers } from './scene/uploadSceneBuffers.js';
import type { FrameInput, Scene } from '@vitrum/core';
import type { AdjointGradientRequest } from './inverse/inverseSession.js';

export class AdjointPass {
  readonly #device: GPUDevice;
  #pipeline: GPUComputePipeline | null = null;

  constructor(device: GPUDevice) {
    this.#device = device;
  }

  /**
   * WS5 Phase-1 path-replay adjoint pass.
   *
   * One dispatch of `PT_WEBGPU_ADJOINT_PASS_WGSL` over the live scene buffers:
   * per pixel it re-traces the frozen-seed primary ray (brute-force closest-hit)
   * and accumulates `∂loss/∂θ` for the optimized material params through the
   * GPU-validated partials + fixed-point `adjointScatter`:
   *  - baseColor / roughness — single-bounce point + rect-area direct-light NEE
   *    (the BRDF partials `dBrdf_dBaseColor` / `dBrdf_dRoughness`);
   *  - emissive — the camera-DIRECT emission at the primary hit (NOT a NEE term):
   *    `∂loss/∂emissive_c = dLoss_dR_c · emissiveIntensity`.
   *
   * The pipeline is lazily compiled and cached on the first call.
   * Transient buffers are allocated per step and destroyed in the finally block
   * (H14-D leak-guard: adjointCreated / adjointDestroyed counters preserved).
   */
  async computeGradient(
    req: AdjointGradientRequest,
    sb: UploadedSceneBuffers,
    last: FrameInput,
    scene: Scene,
    supportedAnalyticShapes: ReadonlySet<string>,
    materialIndexForPrimitive: (scene: Scene, id: string, shapes: ReadonlySet<string>) => number | null,
  ): Promise<Float32Array> {
    const device = this.#device;

    if (this.#pipeline == null) {
      const module = device.createShaderModule({ code: PT_WEBGPU_ADJOINT_PASS_WGSL });
      this.#pipeline = device.createComputePipeline({
        label: 'vitrum.pt-webgpu.adjointPass',
        layout: 'auto',
        compute: { module, entryPoint: 'main' },
      });
    }
    const pipeline = this.#pipeline;

    const { width, height, channels, params, gradientLength, dLoss_dRendered } = req;
    const sampleCount = Math.max(1, Math.floor(req.samples));

    // AdjointParams UBO: invViewProj(mat4) + cameraPos(vec4) + 2×uvec4 of counts.
    const vp = multiplyMat4(last.projMatrix, last.viewMatrix);
    const invVp = invertMat4(asMat4(vp));
    if (invVp == null) {
      throw new Error('computeAdjointGradient: camera viewProj is not invertible.');
    }
    const ubo = new ArrayBuffer(ADJOINT_PARAMS_UBO_BYTES);
    const uboF = new Float32Array(ubo);
    const uboU = new Uint32Array(ubo);
    uboF.set(invVp, 0); // mat4x4f at byte 0 (16 floats)
    uboF[16] = last.cameraPosition[0];
    uboF[17] = last.cameraPosition[1];
    uboF[18] = last.cameraPosition[2];
    uboF[19] = 1;
    uboU[20] = width >>> 0;
    uboU[21] = height >>> 0;
    uboU[22] = sb.triangleCount >>> 0;
    uboU[23] = sb.pointLightCount >>> 0;
    uboU[24] = params.length >>> 0;
    uboU[25] = channels >>> 0;
    uboU[26] = sb.rectAreaLightCount >>> 0;
    uboU[27] = sampleCount >>> 0;

    // adjointParamDescs: per param {matId, fieldCode, gradOffset, w}. For an
    // emissive param `w` carries the FIXED emissiveIntensity (bitcast f32) the
    // pass folds back in (the packed material folds intensity INTO emissive.rgb,
    // so the partial ∂rendered/∂emissive_param = throughput · emissiveIntensity);
    // baseColor/roughness leave it 0. A Float32 view aliases the same buffer so
    // the .w slot can hold an f32 the shader reads via bitcast<f32>.
    const descs = new Uint32Array(Math.max(params.length, 1) * 4);
    const descsF = new Float32Array(descs.buffer);
    for (let i = 0; i < params.length; i++) {
      const p = params[i]!;
      const matId = materialIndexForPrimitive(scene, p.id, supportedAnalyticShapes);
      if (matId == null) {
        throw new Error(`computeAdjointGradient: no material index for primitive "${p.id}".`);
      }
      const fieldCode =
        p.field === 'roughness'
          ? ADJOINT_FIELD_ROUGHNESS
          : p.field === 'emissive'
            ? ADJOINT_FIELD_EMISSIVE
            : p.field === 'specularColor'
              ? ADJOINT_FIELD_SPECULAR_COLOR
              : p.field === 'specularIntensity'
                ? ADJOINT_FIELD_SPECULAR_INTENSITY
                : ADJOINT_FIELD_BASECOLOR;
      descs[i * 4 + 0] = matId >>> 0;
      descs[i * 4 + 1] = fieldCode;
      descs[i * 4 + 2] = p.offset >>> 0;
      if (fieldCode === ADJOINT_FIELD_EMISSIVE) {
        // Read the live emissiveIntensity (held fixed during the fit) for the fold.
        const prim = scene.primitives.find((pr) => pr.id === p.id);
        descsF[i * 4 + 3] = prim?.material.emissiveIntensity ?? 1;
      }
    }

    // H14-D: track all transient buffers in a list; finally block destroys any
    // that survive a rejected mapAsync (device loss, OOM) so no GPU memory leaks.
    const U = (globalThis as { GPUBufferUsage: typeof GPUBufferUsage }).GPUBufferUsage;
    const adjointCreated: GPUBuffer[] = [];
    const adjointDestroyed = new Set<GPUBuffer>();
    const adjointDestroy = (buf: GPUBuffer) => {
      if (!adjointDestroyed.has(buf)) {
        adjointDestroyed.add(buf);
        buf.destroy();
      }
    };
    const mk = (size: number, usage: number, data?: ArrayBufferView): GPUBuffer => {
      const b = device.createBuffer({ size: Math.max(size, 16), usage });
      adjointCreated.push(b);
      if (data) device.queue.writeBuffer(b, 0, data.buffer, data.byteOffset, data.byteLength);
      return b;
    };
    const paramsBuf = mk(ADJOINT_PARAMS_UBO_BYTES, U.UNIFORM | U.COPY_DST, uboF);
    const dLossBuf = mk(dLoss_dRendered.byteLength, U.STORAGE | U.COPY_DST, dLoss_dRendered);
    const gradBuf = mk(gradientLength * 4, U.STORAGE | U.COPY_SRC | U.COPY_DST, new Int32Array(gradientLength));
    const descBuf = mk(descs.byteLength, U.STORAGE | U.COPY_DST, descs);
    const stage = mk(gradientLength * 4, U.MAP_READ | U.COPY_DST);

    const bind = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: sb.positionsBuffer } },
        { binding: 2, resource: { buffer: sb.indicesBuffer } },
        { binding: 3, resource: { buffer: sb.triMaterialIdsBuffer } },
        { binding: 4, resource: { buffer: sb.materialsBuffer } },
        { binding: 5, resource: { buffer: sb.normalsBuffer } },
        { binding: 6, resource: { buffer: sb.pointLightsBuffer } },
        { binding: 7, resource: { buffer: dLossBuf } },
        { binding: 8, resource: { buffer: gradBuf } },
        { binding: 9, resource: { buffer: descBuf } },
        { binding: 10, resource: { buffer: sb.rectAreaLightsBuffer } },
      ],
    });

    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
    pass.end();
    enc.copyBufferToBuffer(gradBuf, 0, stage, 0, gradientLength * 4);
    device.queue.submit([enc.finish()]);
    try {
      await stage.mapAsync((globalThis as { GPUMapMode: typeof GPUMapMode }).GPUMapMode.READ);
      const raw = new Int32Array(stage.getMappedRange().slice(0));
      stage.unmap();

      const grad = new Float32Array(gradientLength);
      for (let i = 0; i < gradientLength; i++) grad[i] = raw[i]! / ADJOINT_GRAD_FP;

      // Per-step transient buffers — free them (no leak; the pipeline is cached).
      for (const b of adjointCreated) adjointDestroy(b);
      return grad;
    } finally {
      // Destroy any buffers not already freed in the happy path above
      // (ensures no GPU memory leak on rejected mapAsync).
      for (const b of adjointCreated) adjointDestroy(b);
    }
  }

  /** Drop the cached pipeline reference (GPUComputePipeline has no destroy()). */
  dispose(): void {
    this.#pipeline = null;
  }
}
