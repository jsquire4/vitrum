/**
 * AdjointPass — WS5 Phase-1 path-replay adjoint compute pass.
 *
 * Extracted from PTEngineWebGPU.#computeAdjointGradient (D8.10). Owns:
 *   - Lazy pipeline compilation (PT_WEBGPU_ADJOINT_PASS_WGSL, layout:'auto').
 *   - Per-step transient buffer allocation + guaranteed destroy on success and
 *     on mapAsync rejection (H14-D leak-guard accounting with adjointCreated /
 *     adjointDestroyed counters — these counters may be consumed by telemetry
 *     and tests; the semantics are preserved verbatim).
 *   - UBO packing (invViewProj + cameraPos + direct-light counts + env map replay params).
 *   - Dispatch + copyBufferToBuffer readback.
 *
 * The engine holds one lazy instance (`#adjointPass`) and delegates the
 * `computeAdjointGradient` hook to it.  The pipeline is cached for the lifetime
 * of the AdjointPass instance (engine-owned; freed via dispose()).
 */

import {
  analyticPrimitiveToMesh,
  asMat4,
  type AnalyticPrimitive,
  type FrameInput,
  type Scene,
  type ScenePrimitive,
} from '@vitrum/core';
import {
  expandIndicesToStride4,
  mergeUv1FromCore,
  mergeWorldSpaceFromCore,
} from '@vitrum/shared-bvh';
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
  ADJOINT_FIELD_IRIDESCENCE_IOR,
  ADJOINT_FIELD_IRIDESCENCE_THICKNESS_RANGE,
  ADJOINT_FIELD_ANISOTROPY,
  ADJOINT_FIELD_ANISOTROPY_ROTATION,
  ADJOINT_FIELD_AO_MAP_INTENSITY,
  ADJOINT_FIELD_LIGHT_MAP_INTENSITY,
  ADJOINT_FIELD_ENV_MAP_INTENSITY,
  ADJOINT_FIELD_NORMAL_SCALE,
  ADJOINT_FIELD_BUMP_SCALE,
  ADJOINT_FIELD_CLEARCOAT_NORMAL_SCALE,
  ADJOINT_FIELD_EMITTER_COLOR,
  ADJOINT_FIELD_EMITTER_INTENSITY,
  ADJOINT_EMITTER_TARGET_DIRECTIONAL,
  ADJOINT_EMITTER_TARGET_POINT,
  ADJOINT_EMITTER_TARGET_SPOT,
  ADJOINT_EMITTER_TARGET_RECT,
  ADJOINT_EMITTER_TARGET_MESH,
} from './wgsl/pathTrace/adjointPass.wgsl.js';
import { ADJOINT_GRAD_FP } from './wgsl/pathTrace/pathTraceAdjoint.wgsl.js';
import type { UploadedSceneBuffers } from './scene/uploadSceneBuffers.js';
import type { AdjointGradientRequest } from './inverse/inverseSession.js';
import {
  meshAreaEmitterAdjointRangeForScene,
  packMeshAreaAdjointReplayArrays,
} from './scene/emitterPacking.js';

interface AdjointWorldSpaceGeometry {
  readonly positions: Float32Array;
  readonly normals: Float32Array;
  readonly uvs: Float32Array;
  readonly tangents: Float32Array;
  readonly colors: Float32Array;
  readonly indices: Uint32Array;
  readonly triMaterialIds: Uint32Array;
  readonly triangleCount: number;
}

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
   *    single-bounce directional + point + spot + stochastic area-measure rect/disc/mesh-area
   *    direct-light NEE (the BRDF partials in `pathTraceAdjoint.wgsl.ts`);
   *  - emissive / emissiveIntensity / lightMapIntensity — the camera-DIRECT
   *    emission at the primary hit (NOT a NEE term): `∂loss/∂emissive_c =
   *    dLoss_dR_c · emissiveIntensity`, `∂loss/∂emissiveIntensity =
   *    dot(dLoss_dR, emissive)`, and `∂loss/∂lightMapIntensity =
   *    dot(dLoss_dR, lightMapRadiance)`, modulated by the hit-local readable
   *    texture texels when authored.
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

    // AdjointParams UBO: invViewProj(mat4) + cameraPos(vec4) + 3×uvec4 of counts
    // plus an env-map uvec4 and env scalar vec4.
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
    uboU[32] = sb.environmentMapWidth >>> 0;
    uboU[33] = sb.environmentMapHeight >>> 0;
    uboU[34] = sb.hasEnvironmentMap ? 1 : 0;
    uboF[36] = sb.environmentHdriIntensity;
    uboF[37] = sb.environmentHdriRotationY;

    // adjointParamDescs: two vec4u records per param:
    //   material record 0: {matId, fieldCode, gradOffset, fieldPayloadBits}
    //   emitter record 0: {kind-local light slot/range start, fieldCode, gradOffset, emitterTargetMeta}
    //   record 1: {payloadXBits, payloadYBits, payloadZBits, payloadWBits}
    // For an emissive param, record0.w carries the FIXED emissiveIntensity
    // (bitcast f32). For an emissiveIntensity param, record1.xyz carries the
    // UNFACTORED material emissive RGB so intensity=0 remains differentiable.
    // Emitter color/intensity params use record1.xyz = unfactored color and
    // record1.w = fixed intensity. Mapped mesh-area emitters recover their
    // readable emissive-map multiplier from meshAreaLightSourceFactorsBuffer, so
    // zero authored color channels remain differentiable. The same vec4 stores
    // the explicit mesh-area owner slot in `.w`, so capped/power-sorted replay
    // can scatter by owner instead of assuming a contiguous packed range.
    // emitterTargetMeta packs kind in the low 8 bits and range count in the upper
    // bits (1 for scalar light streams).
    // Lit BRDF fields leave payloads 0.
    // A Float32 view aliases the same buffer.
    const needsMeshAreaAdjointReplay = params.some((p) =>
      p.domain === 'emitters' &&
      scene.emitters.some((e) => e.id === p.id && e.kind === 'mesh-area'));
    const meshAreaAdjointReplay = needsMeshAreaAdjointReplay
      ? packMeshAreaAdjointReplayArrays(scene)
      : null;
    uboU[30] = (meshAreaAdjointReplay?.meshAreaLightCount ?? sb.meshAreaLightCount) >>> 0;

    const descs = new Uint32Array(Math.max(params.length, 1) * 8);
    const descsF = new Float32Array(descs.buffer);
    for (let i = 0; i < params.length; i++) {
      const p = params[i]!;
      if (p.domain === 'emitters') {
        const target = adjointEmitterTargetForScene(scene, p.id);
        if (target == null) {
          throw new Error(
            `computeAdjointGradient: emitter "${p.id}" is outside the scoped adjoint direct-light target domain.`,
          );
        }
        let fieldCode: number;
        switch (p.field) {
          case 'intensity':
            fieldCode = ADJOINT_FIELD_EMITTER_INTENSITY;
            break;
          case 'color':
            fieldCode = ADJOINT_FIELD_EMITTER_COLOR;
            break;
          default:
            throw new Error(`computeAdjointGradient: unsupported emitter adjoint field "${String(p.field)}".`);
        }
        const descBase = i * 8;
        descs[descBase + 0] = target.slot >>> 0;
        descs[descBase + 1] = fieldCode;
        descs[descBase + 2] = p.offset >>> 0;
        descs[descBase + 3] = encodeAdjointEmitterTargetMeta(target.kind, target.count);
        descsF[descBase + 4] = target.color[0];
        descsF[descBase + 5] = target.color[1];
        descsF[descBase + 6] = target.color[2];
        descsF[descBase + 7] = target.intensity;
        continue;
      }
      const matId = materialIndexForPrimitive(scene, p.id, supportedAnalyticShapes);
      if (matId == null) {
        throw new Error(`computeAdjointGradient: no material index for primitive "${p.id}".`);
      }
      let fieldCode: number;
      switch (p.field) {
        case 'baseColor':
          fieldCode = ADJOINT_FIELD_BASECOLOR;
          break;
        case 'roughness':
          fieldCode = ADJOINT_FIELD_ROUGHNESS;
          break;
        case 'metallic':
          fieldCode = ADJOINT_FIELD_METALLIC;
          break;
        case 'aoMapIntensity':
          fieldCode = ADJOINT_FIELD_AO_MAP_INTENSITY;
          break;
        case 'lightMapIntensity':
          fieldCode = ADJOINT_FIELD_LIGHT_MAP_INTENSITY;
          break;
        case 'envMapIntensity':
          fieldCode = ADJOINT_FIELD_ENV_MAP_INTENSITY;
          break;
        case 'normalScale':
          fieldCode = ADJOINT_FIELD_NORMAL_SCALE;
          break;
        case 'bumpScale':
          fieldCode = ADJOINT_FIELD_BUMP_SCALE;
          break;
        case 'clearcoatNormalScale':
          fieldCode = ADJOINT_FIELD_CLEARCOAT_NORMAL_SCALE;
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
        case 'iridescenceIor':
          fieldCode = ADJOINT_FIELD_IRIDESCENCE_IOR;
          break;
        case 'iridescenceThicknessRange':
          fieldCode = ADJOINT_FIELD_IRIDESCENCE_THICKNESS_RANGE;
          break;
        case 'anisotropy':
          fieldCode = ADJOINT_FIELD_ANISOTROPY;
          break;
        case 'anisotropyRotation':
          fieldCode = ADJOINT_FIELD_ANISOTROPY_ROTATION;
          break;
        case 'specularColor':
          fieldCode = ADJOINT_FIELD_SPECULAR_COLOR;
          break;
        case 'specularIntensity':
          fieldCode = ADJOINT_FIELD_SPECULAR_INTENSITY;
          break;
        default:
          throw new Error(`computeAdjointGradient: unsupported material adjoint field "${String(p.field)}".`);
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

    const geometryOverride = buildAdjointWorldSpaceGeometryOverride(
      scene,
      supportedAnalyticShapes,
      materialIndexForPrimitive,
    );
    if (geometryOverride != null) {
      uboU[22] = geometryOverride.triangleCount >>> 0;
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
    const meshAreaLightsBuffer = meshAreaAdjointReplay == null
      ? sb.meshAreaLightsBuffer
      : mk(meshAreaAdjointReplay.meshAreaLightsData.byteLength, U.STORAGE | U.COPY_DST, meshAreaAdjointReplay.meshAreaLightsData);
    const meshAreaLightSourceFactorsBuffer = meshAreaAdjointReplay == null
      ? sb.meshAreaLightSourceFactorsBuffer
      : mk(
          meshAreaAdjointReplay.meshAreaLightSourceFactorsData.byteLength,
          U.STORAGE | U.COPY_DST,
          meshAreaAdjointReplay.meshAreaLightSourceFactorsData,
        );
    const positionsBuffer = geometryOverride == null
      ? sb.positionsBuffer
      : mk(geometryOverride.positions.byteLength, U.STORAGE | U.COPY_DST, geometryOverride.positions);
    const indicesBuffer = geometryOverride == null
      ? sb.indicesBuffer
      : mk(geometryOverride.indices.byteLength, U.STORAGE | U.COPY_DST, geometryOverride.indices);
    const triMaterialIdsBuffer = geometryOverride == null
      ? sb.triMaterialIdsBuffer
      : mk(geometryOverride.triMaterialIds.byteLength, U.STORAGE | U.COPY_DST, geometryOverride.triMaterialIds);
    const normalsBuffer = geometryOverride == null
      ? sb.normalsBuffer
      : mk(geometryOverride.normals.byteLength, U.STORAGE | U.COPY_DST, geometryOverride.normals);
    const uvsBuffer = geometryOverride == null
      ? sb.uvsBuffer
      : mk(geometryOverride.uvs.byteLength, U.STORAGE | U.COPY_DST, geometryOverride.uvs);
    const colorsBuffer = geometryOverride == null
      ? sb.colorsBuffer
      : mk(geometryOverride.colors.byteLength, U.STORAGE | U.COPY_DST, geometryOverride.colors);
    const tangentsBuffer = geometryOverride == null
      ? sb.tangentsBuffer
      : mk(geometryOverride.tangents.byteLength, U.STORAGE | U.COPY_DST, geometryOverride.tangents);

    const bind = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: positionsBuffer } },
        { binding: 2, resource: { buffer: indicesBuffer } },
        { binding: 3, resource: { buffer: triMaterialIdsBuffer } },
        { binding: 4, resource: { buffer: sb.materialsBuffer } },
        { binding: 5, resource: { buffer: normalsBuffer } },
        { binding: 6, resource: { buffer: sb.pointLightsBuffer } },
        { binding: 7, resource: { buffer: dLossBuf } },
        { binding: 8, resource: { buffer: gradBuf } },
        { binding: 9, resource: { buffer: descBuf } },
        { binding: 10, resource: { buffer: sb.rectAreaLightsBuffer } },
        { binding: 11, resource: { buffer: sb.directionalLightsBuffer } },
        { binding: 12, resource: { buffer: sb.spotLightsBuffer } },
        { binding: 13, resource: { buffer: meshAreaLightsBuffer } },
        { binding: 14, resource: { buffer: uvsBuffer } },
        { binding: 15, resource: { buffer: sb.materialTexDescriptorsBuffer } },
        { binding: 16, resource: sb.materialTextureView },
        { binding: 17, resource: sb.materialTextureSampler },
        { binding: 18, resource: { buffer: colorsBuffer } },
        { binding: 19, resource: sb.materialLinearTextureView },
        { binding: 20, resource: { buffer: sb.environmentMapTexelsBuffer } },
        { binding: 21, resource: { buffer: sb.environmentMapCdfBuffer } },
        { binding: 22, resource: { buffer: meshAreaLightSourceFactorsBuffer } },
        { binding: 23, resource: { buffer: tangentsBuffer } },
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

export function buildAdjointWorldSpaceGeometryOverride(
  scene: Scene,
  supportedAnalyticShapes: ReadonlySet<string>,
  materialIndexForPrimitive: (scene: Scene, id: string, shapes: ReadonlySet<string>) => number | null,
): AdjointWorldSpaceGeometry | null {
  const replayPrimitives: ScenePrimitive[] = [];
  let needsOverride = false;
  for (const primitive of scene.primitives) {
    const replayPrimitive = adjointReplayPrimitive(primitive, supportedAnalyticShapes);
    if (replayPrimitive == null) return null;
    replayPrimitives.push(replayPrimitive);
    needsOverride ||= primitive.kind === 'analytic' || needsAdjointWorldSpaceGeometryOverride(replayPrimitive);
  }
  if (!needsOverride) {
    return null;
  }

  const replayScene: Scene = { ...scene, primitives: replayPrimitives };
  const merged = mergeWorldSpaceFromCore(replayScene, {
    positionStride: 4,
    filter: isAdjointReplayMeshPrimitive,
  });
  if (merged.triangleCount === 0) return null;

  const mergedTriMaterialIds = new Uint32Array(merged.triangleCount);
  for (const range of merged.meshVertexRanges) {
    const matId = materialIndexForPrimitive(scene, range.name, supportedAnalyticShapes);
    if (matId == null) return null;
    const triEnd = Math.min(mergedTriMaterialIds.length, range.triStart + range.triCount);
    for (let tri = range.triStart; tri < triEnd; tri += 1) {
      mergedTriMaterialIds[tri] = matId >>> 0;
    }
  }

  const triMaterialIds = new Uint32Array(merged.triangleCount);
  for (let tri = 0; tri < merged.triangleCount; tri += 1) {
    const mergedTri = merged.bvhTriToMergedTri[tri] ?? tri;
    triMaterialIds[tri] = mergedTriMaterialIds[mergedTri] ?? 0;
  }

  const uv1 = mergeUv1FromCore(replayScene, merged.meshVertexRanges, merged.vertexCount);
  const uvs = new Float32Array(merged.vertexCount * 4);
  for (let vertex = 0; vertex < merged.vertexCount; vertex += 1) {
    uvs[vertex * 4] = merged.uvs[vertex * 2] ?? 0;
    uvs[vertex * 4 + 1] = merged.uvs[vertex * 2 + 1] ?? 0;
    if (uv1 != null) {
      uvs[vertex * 4 + 2] = uv1[vertex * 2] ?? 0;
      uvs[vertex * 4 + 3] = uv1[vertex * 2 + 1] ?? 0;
    }
  }

  return {
    positions: merged.positions,
    normals: merged.normals,
    uvs,
    tangents: merged.tangents,
    colors: merged.colors,
    indices: expandIndicesToStride4(merged.indices),
    triMaterialIds,
    triangleCount: merged.triangleCount,
  };
}

function adjointReplayPrimitive(
  primitive: ScenePrimitive,
  supportedAnalyticShapes: ReadonlySet<string>,
): Extract<ScenePrimitive, { kind: 'mesh' | 'skinned-mesh' | 'instanced-mesh' }> | null {
  if (isAdjointReplayMeshPrimitive(primitive)) return primitive;
  if (primitive.kind === 'analytic') {
    if (!supportedAnalyticShapes.has(primitive.shape)) return null;
    return analyticPrimitiveToMesh(primitive as AnalyticPrimitive);
  }
  return null;
}

function isAdjointReplayMeshPrimitive(
  primitive: ScenePrimitive,
): primitive is Extract<ScenePrimitive, { kind: 'mesh' | 'skinned-mesh' | 'instanced-mesh' }> {
  return (
    primitive.kind === 'mesh' ||
    primitive.kind === 'skinned-mesh' ||
    primitive.kind === 'instanced-mesh'
  );
}

function needsAdjointWorldSpaceGeometryOverride(
  primitive: Extract<
    ScenePrimitive,
    { kind: 'mesh' | 'skinned-mesh' | 'instanced-mesh' }
  >,
): boolean {
  if (primitive.kind === 'instanced-mesh') return primitive.instances.length > 0;
  return primitive.transform != null && !isIdentityMat4(primitive.transform);
}

function isIdentityMat4(transform: Float32Array): boolean {
  const expected = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  if (transform.length < 16) return false;
  for (let i = 0; i < 16; i += 1) {
    if (Math.abs((transform[i] ?? 0) - expected[i]!) > 1e-6) return false;
  }
  return true;
}

function adjointEmitterTargetForScene(
  scene: Scene,
  id: string,
): {
  readonly kind: number;
  readonly slot: number;
  readonly count: number;
  readonly color: readonly [number, number, number];
  readonly intensity: number;
} | null {
  let directionalSlot = 0;
  let pointSlot = 0;
  let spotSlot = 0;
  let rectSlot = 0;
  for (const emitter of scene.emitters) {
    switch (emitter.kind) {
      case 'directional': {
        const slot = directionalSlot;
        directionalSlot += 1;
        if (emitter.id !== id) break;
        return {
          kind: ADJOINT_EMITTER_TARGET_DIRECTIONAL,
          slot,
          count: 1,
          color: emitter.color,
          intensity: emitter.intensity,
        };
      }
      case 'point': {
        const slot = pointSlot;
        pointSlot += 1;
        if (emitter.id !== id) break;
        return {
          kind: ADJOINT_EMITTER_TARGET_POINT,
          slot,
          count: 1,
          color: emitter.color,
          intensity: emitter.intensity,
        };
      }
      case 'spot': {
        const slot = spotSlot;
        spotSlot += 1;
        if (emitter.id !== id) break;
        return {
          kind: ADJOINT_EMITTER_TARGET_SPOT,
          slot,
          count: 1,
          color: emitter.color,
          intensity: emitter.intensity,
        };
      }
      case 'rect-area':
      case 'disc-area': {
        const slot = rectSlot;
        rectSlot += 1;
        if (emitter.id !== id) break;
        return {
          kind: ADJOINT_EMITTER_TARGET_RECT,
          slot,
          count: 1,
          color: emitter.color,
          intensity: emitter.intensity,
        };
      }
      case 'mesh-area': {
        if (emitter.id !== id) break;
        const range = meshAreaEmitterAdjointRangeForScene(scene, emitter.id);
        if (range == null) return null;
        return {
          kind: ADJOINT_EMITTER_TARGET_MESH,
          slot: range.adjointEmitterSlot,
          count: 1,
          color: emitter.color,
          intensity: emitter.intensity,
        };
      }
      default: {
        const unknownEmitter = emitter as { readonly id?: string };
        if (unknownEmitter.id === id) return null;
        break;
      }
    }
  }
  return null;
}

function encodeAdjointEmitterTargetMeta(kind: number, count: number): number {
  const safeCount = Math.max(1, Math.min(0x00ffffff, Math.floor(count)));
  return ((kind & 0xff) | (safeCount << 8)) >>> 0;
}
