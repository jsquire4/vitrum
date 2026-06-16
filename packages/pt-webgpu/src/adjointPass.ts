/**
 * AdjointPass — WS5 Phase-1 path-replay adjoint compute pass.
 *
 * Extracted from PTEngineWebGPU.#computeAdjointGradient (D8.10). Owns:
 *   - Lazy pipeline compilation (PT_WEBGPU_ADJOINT_PASS_WGSL, layout:'auto').
 *   - Per-step transient buffer allocation + guaranteed destroy on success and
 *     on mapAsync rejection (H14-D leak-guard accounting with adjointCreated /
 *     adjointDestroyed counters — these counters may be consumed by telemetry
 *     and tests; the semantics are preserved verbatim).
 *   - UBO packing (invViewProj + cameraPos + direct-light counts).
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
  ADJOINT_FIELD_METALLIC,
  ADJOINT_FIELD_EMISSIVE_INTENSITY,
  ADJOINT_FIELD_CLEARCOAT,
  ADJOINT_FIELD_CLEARCOAT_ROUGHNESS,
  ADJOINT_FIELD_SHEEN,
  ADJOINT_FIELD_SHEEN_ROUGHNESS,
  ADJOINT_FIELD_SHEEN_COLOR,
  ADJOINT_FIELD_IRIDESCENCE,
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
   *  - baseColor / roughness / metallic / specular / clearcoat / sheen scalar controls —
   *    single-bounce directional + point + spot + center-sampled rect/disc/mesh-area
   *    direct-light NEE (the BRDF partials in `pathTraceAdjoint.wgsl.ts`);
   *  - emissive / emissiveIntensity — the camera-DIRECT emission at the primary
   *    hit (NOT a NEE term): `∂loss/∂emissive_c = dLoss_dR_c · emissiveIntensity`
   *    and `∂loss/∂emissiveIntensity = dot(dLoss_dR, emissive)`, both modulated
   *    by the hit-local emissiveMap texel when one is authored.
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

    // AdjointParams UBO: invViewProj(mat4) + cameraPos(vec4) + 3×uvec4 of counts.
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
    uboU[28] = sb.directionalLightCount >>> 0;
    uboU[29] = sb.spotLightCount >>> 0;
    uboU[30] = sb.meshAreaLightCount >>> 0;

    // adjointParamDescs: two vec4u records per param:
    //   record 0: {matId, fieldCode, gradOffset, fieldPayloadBits}
    //   record 1: {payloadXBits, payloadYBits, payloadZBits, _}
    // For an emissive param, record0.w carries the FIXED emissiveIntensity
    // (bitcast f32). For an emissiveIntensity param, record1.xyz carries the
    // UNFACTORED material emissive RGB so intensity=0 remains differentiable.
    // Lit BRDF fields leave payloads 0. A Float32 view aliases the same buffer.
    const descs = new Uint32Array(Math.max(params.length, 1) * 8);
    const descsF = new Float32Array(descs.buffer);
    for (let i = 0; i < params.length; i++) {
      const p = params[i]!;
      const matId = materialIndexForPrimitive(scene, p.id, supportedAnalyticShapes);
      if (matId == null) {
        throw new Error(`computeAdjointGradient: no material index for primitive "${p.id}".`);
      }
      let fieldCode = ADJOINT_FIELD_BASECOLOR;
      switch (p.field) {
        case 'roughness':
          fieldCode = ADJOINT_FIELD_ROUGHNESS;
          break;
        case 'metallic':
          fieldCode = ADJOINT_FIELD_METALLIC;
          break;
        case 'emissive':
          fieldCode = ADJOINT_FIELD_EMISSIVE;
          break;
        case 'emissiveIntensity':
          fieldCode = ADJOINT_FIELD_EMISSIVE_INTENSITY;
          break;
        case 'clearcoat':
          fieldCode = ADJOINT_FIELD_CLEARCOAT;
          break;
        case 'clearcoatRoughness':
          fieldCode = ADJOINT_FIELD_CLEARCOAT_ROUGHNESS;
          break;
        case 'sheen':
          fieldCode = ADJOINT_FIELD_SHEEN;
          break;
        case 'sheenRoughness':
          fieldCode = ADJOINT_FIELD_SHEEN_ROUGHNESS;
          break;
        case 'sheenColor':
          fieldCode = ADJOINT_FIELD_SHEEN_COLOR;
          break;
        case 'iridescence':
          fieldCode = ADJOINT_FIELD_IRIDESCENCE;
          break;
        case 'specularColor':
          fieldCode = ADJOINT_FIELD_SPECULAR_COLOR;
          break;
        case 'specularIntensity':
          fieldCode = ADJOINT_FIELD_SPECULAR_INTENSITY;
          break;
        default:
          break;
      }
      const descBase = i * 8;
      descs[descBase + 0] = matId >>> 0;
      descs[descBase + 1] = fieldCode;
      descs[descBase + 2] = p.offset >>> 0;
      const prim = scene.primitives.find((pr) => pr.id === p.id);
      if (fieldCode === ADJOINT_FIELD_EMISSIVE) {
        // Read the live emissiveIntensity (held fixed during the fit) for the fold.
        descsF[descBase + 3] = prim?.material.emissiveIntensity ?? 1;
      } else if (fieldCode === ADJOINT_FIELD_EMISSIVE_INTENSITY) {
        const emissive = prim?.material.emissive ?? [0, 0, 0];
        descsF[descBase + 4] = emissive[0];
        descsF[descBase + 5] = emissive[1];
        descsF[descBase + 6] = emissive[2];
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
        { binding: 11, resource: { buffer: sb.directionalLightsBuffer } },
        { binding: 12, resource: { buffer: sb.spotLightsBuffer } },
        { binding: 13, resource: { buffer: sb.meshAreaLightsBuffer } },
        { binding: 14, resource: { buffer: sb.uvsBuffer } },
        { binding: 15, resource: { buffer: sb.materialTexDescriptorsBuffer } },
        { binding: 16, resource: sb.materialTextureView },
        { binding: 17, resource: sb.materialTextureSampler },
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
