/**
 * GPU implementation of the certified pt-webgpu path-replay domain.
 *
 * Only one-bounce, camera-visible `MaterialSpec.emissive` is accepted. This
 * class validates that narrow descriptor domain before creating a shader module,
 * pipeline, buffer, bind group, or command encoder.
 */
import {
  asMat4,
  resolveFrameCameraPosition,
  type FrameInput,
  type Scene,
  type ScenePrimitive,
} from '@vitrum/core';
import { MAX_INVERSE_SAMPLES_PER_STEP } from '@vitrum/core/inverse-scaffolding';
import {
  expandIndicesToStride4,
  mergeWorldSpaceFromCore,
} from '@vitrum/shared-bvh';
import { resolveEmissiveIntensity } from '@vitrum/shared-samplers';
import { invertMat4, multiplyMat4 } from './math/mat4.js';
import {
  ADJOINT_FIELD_EMISSIVE,
  ADJOINT_PARAMS_UBO_BYTES,
  PT_WEBGPU_ADJOINT_PASS_WGSL,
  composePtWebgpuAdjointPassWgsl,
} from './wgsl/pathTrace/adjointPass.wgsl.js';
import type { PtWebgpuSamplingMode } from './wgsl/common.wgsl.js';
import {
  applySolveSkinToScene,
  type UploadedSceneBuffers,
} from './scene/uploadSceneBuffers.js';
import type { AdjointGradientRequest } from './inverse/inverseSession.js';
import {
  emissiveReplayPrimitiveIssue,
  emissiveReplaySceneIssue,
  emissiveReplayTargetIssue,
} from './inverse/emissivePathReplayDomain.js';

const NO_ANALYTIC_SHAPES: ReadonlySet<string> = new Set();
const DEFAULT_GRADIENT_SCALE = 1048576;
const I32_MAX = 2147483647;
const FIXED_POINT_MARGIN = 4096;
const U32_MAX = 0xffff_ffff;
const F32_MAX = 3.4028234663852886e38;

interface AdjointWorldSpaceGeometry {
  readonly positions: Float32Array;
  readonly indices: Uint32Array;
  readonly triMaterialIds: Uint32Array;
  readonly triangleCount: number;
}

type MaterialIndexForPrimitive = (
  scene: Scene,
  id: string,
  shapes: ReadonlySet<string>,
) => number | null;

function assertPositiveU32(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 1 || value > U32_MAX) {
    throw new RangeError(
      `computeAdjointGradient: ${label} must be a positive u32 integer.`,
    );
  }
}

function checkedProduct(label: string, ...factors: readonly number[]): number {
  let product = 1;
  for (const factor of factors) {
    product *= factor;
    if (!Number.isSafeInteger(product) || product > U32_MAX) {
      throw new RangeError(
        `computeAdjointGradient: ${label} exceeds the addressable u32 resource domain.`,
      );
    }
  }
  return product;
}

function validateDeviceResourceLimits(
  device: GPUDevice,
  req: AdjointGradientRequest,
  geometryOverride: AdjointWorldSpaceGeometry | null,
): void {
  const limits = device.limits;
  const maxBufferSize = Number(limits.maxBufferSize);
  const maxStorageBindingSize = Number(limits.maxStorageBufferBindingSize);
  const maxWorkgroups = Number(limits.maxComputeWorkgroupsPerDimension);
  if (
    !Number.isFinite(maxBufferSize) ||
    maxBufferSize < 1 ||
    !Number.isFinite(maxStorageBindingSize) ||
    maxStorageBindingSize < 1 ||
    !Number.isFinite(maxWorkgroups) ||
    maxWorkgroups < 1
  ) {
    throw new RangeError(
      'computeAdjointGradient: device did not report valid WebGPU resource limits.',
    );
  }
  const storageSizes = [
    req.dLoss_dRendered.byteLength,
    req.gradientLength * 4,
    req.params.length * 16,
    ...(geometryOverride == null
      ? []
      : [
          geometryOverride.positions.byteLength,
          geometryOverride.indices.byteLength,
          geometryOverride.triMaterialIds.byteLength,
        ]),
  ];
  for (const size of storageSizes) {
    if (size > maxBufferSize || size > maxStorageBindingSize) {
      throw new RangeError(
        `computeAdjointGradient: ${size}-byte storage resource exceeds this device's limits.`,
      );
    }
  }

  if (
    Math.ceil(req.width / 8) > maxWorkgroups ||
    Math.ceil(req.height / 8) > maxWorkgroups
  ) {
    throw new RangeError(
      'computeAdjointGradient: dispatch dimensions exceed this device\'s compute-workgroup limit.',
    );
  }
}

/**
 * Derive an exactly representable power-of-two fixed-point scale whose worst
 * possible absolute atomic sum stays below i32. The bound includes all legal
 * overlapping gradient offsets and a half-unit rounding allowance for every
 * per-pixel atomic add. A two-sided arithmetic margin covers f32 multiply/add
 * rounding in the shader. Requests with no safe positive f32 scale reject.
 */
export function computeAdjointGradientScale(
  req: AdjointGradientRequest,
  emissiveIntensities: readonly number[],
): number {
  if (emissiveIntensities.length !== req.params.length) {
    throw new RangeError(
      'computeAdjointGradient: emissive intensity count does not match parameters.',
    );
  }
  const pixelCount = req.width * req.height;
  if (!Number.isSafeInteger(pixelCount) || pixelCount < 1) {
    throw new RangeError('computeAdjointGradient: pixel count is not a safe integer.');
  }

  const lossAbsSums: [number, number, number] = [0, 0, 0];
  const lossAbsMaxima: [number, number, number] = [0, 0, 0];
  for (let pixel = 0; pixel < pixelCount; pixel += 1) {
    const base = pixel * req.channels;
    for (let channel = 0; channel < 3; channel += 1) {
      const value = req.dLoss_dRendered[base + channel]!;
      if (!Number.isFinite(value)) {
        throw new RangeError(
          `computeAdjointGradient: dLoss_dRendered[${base + channel}] is not finite.`,
        );
      }
      const absoluteValue = Math.abs(value);
      lossAbsSums[channel] = lossAbsSums[channel]! + absoluteValue;
      lossAbsMaxima[channel] = Math.max(
        lossAbsMaxima[channel]!,
        absoluteValue,
      );
      if (!Number.isFinite(lossAbsSums[channel])) {
        throw new RangeError('computeAdjointGradient: loss magnitude bound overflowed.');
      }
    }
  }

  const absoluteBounds = new Float64Array(req.gradientLength);
  const atomicAdds = new Float64Array(req.gradientLength);
  for (let parameter = 0; parameter < req.params.length; parameter += 1) {
    const param = req.params[parameter]!;
    const intensity = Math.fround(emissiveIntensities[parameter]!);
    if (!Number.isFinite(intensity)) {
      throw new RangeError(
        `computeAdjointGradient: emissive intensity for "${param.id}" is not finite.`,
      );
    }
    for (let channel = 0; channel < 3; channel += 1) {
      const gradientIndex = param.offset + channel;
      if (lossAbsMaxima[channel]! * Math.abs(intensity) > F32_MAX) {
        throw new RangeError(
          `computeAdjointGradient: per-pixel emissive gradient for "${param.id}" ` +
            `channel ${channel} exceeds finite f32 range before fixed-point scaling.`,
        );
      }
      absoluteBounds[gradientIndex] =
        absoluteBounds[gradientIndex]! +
        lossAbsSums[channel]! * Math.abs(intensity);
      if (
        !Number.isFinite(absoluteBounds[gradientIndex]) ||
        absoluteBounds[gradientIndex] > F32_MAX
      ) {
        throw new RangeError(
          `computeAdjointGradient: aggregate emissive gradient at offset ` +
            `${gradientIndex} exceeds finite f32 readback range.`,
        );
      }
      atomicAdds[gradientIndex] = atomicAdds[gradientIndex]! + pixelCount;
    }
  }

  let maximumScale = DEFAULT_GRADIENT_SCALE;
  for (let index = 0; index < req.gradientLength; index += 1) {
    const bound = absoluteBounds[index]!;
    if (bound === 0) continue;
    const roundingAllowance = Math.ceil(atomicAdds[index]! * 0.5);
    const available = I32_MAX - FIXED_POINT_MARGIN - roundingAllowance;
    if (available <= 0) {
      throw new RangeError(
        'computeAdjointGradient: no overflow-safe fixed-point accumulation range.',
      );
    }
    // 2x is intentionally conservative for f32 product and local-sum rounding.
    maximumScale = Math.min(maximumScale, available / (bound * 2));
  }
  if (!(maximumScale > 0) || !Number.isFinite(maximumScale)) {
    throw new RangeError(
      'computeAdjointGradient: no overflow-safe fixed-point accumulation scale.',
    );
  }

  const scale = maximumScale >= DEFAULT_GRADIENT_SCALE
    ? DEFAULT_GRADIENT_SCALE
    : 2 ** Math.floor(Math.log2(maximumScale));
  if (!(scale > 0) || !Number.isFinite(scale) || Math.fround(scale) !== scale) {
    throw new RangeError(
      'computeAdjointGradient: overflow-safe scale is not representable as f32.',
    );
  }
  return scale;
}

/**
 * Defense-in-depth validation at the dispatch boundary. InverseSession performs
 * the public preflight earlier; this guard prevents private callers from
 * compiling or dispatching a descriptor outside the certified domain.
 */
function validateAdjointRequest(
  req: AdjointGradientRequest,
  scene: Scene,
  cameraVisibleEmitters: boolean,
): void {
  if (cameraVisibleEmitters !== true) {
    throw new Error(
      'computeAdjointGradient: cameraVisibleEmitters must be true for exact primary-emission replay.',
    );
  }
  assertPositiveU32(req.width, 'width');
  assertPositiveU32(req.height, 'height');
  assertPositiveU32(req.samples, 'samples');
  assertPositiveU32(req.gradientLength, 'gradientLength');
  if (req.samples > MAX_INVERSE_SAMPLES_PER_STEP) {
    throw new RangeError(
      `computeAdjointGradient: samples exceeds the inverse-session limit ${MAX_INVERSE_SAMPLES_PER_STEP}.`,
    );
  }
  if (req.channels !== 3 && req.channels !== 4) {
    throw new RangeError('computeAdjointGradient: channels must be 3 or 4.');
  }
  const pixelCount = checkedProduct('pixel count', req.width, req.height);
  const requiredLossLength = checkedProduct(
    'loss element count',
    pixelCount,
    req.channels,
  );
  if (req.dLoss_dRendered.length !== requiredLossLength) {
    throw new RangeError(
      `computeAdjointGradient: dLoss_dRendered length ${req.dLoss_dRendered.length} ` +
        `does not equal ${requiredLossLength}.`,
    );
  }
  if (req.params.length < 1) {
    throw new RangeError('computeAdjointGradient: at least one emissive parameter is required.');
  }
  if (req.params.length > U32_MAX) {
    throw new RangeError(
      'computeAdjointGradient: parameter count exceeds the addressable u32 domain.',
    );
  }
  checkedProduct('loss buffer byte size', requiredLossLength, 4);
  checkedProduct('gradient buffer byte size', req.gradientLength, 4);
  checkedProduct('descriptor buffer byte size', req.params.length, 16);

  const sceneIssue = emissiveReplaySceneIssue(scene);
  if (sceneIssue != null) {
    throw new Error(
      `computeAdjointGradient: scene is outside exact emissive replay: ${sceneIssue.message}.`,
    );
  }

  for (const param of req.params) {
    if (
      param.domain !== 'materials' ||
      param.field !== 'emissive' ||
      param.length !== 3
    ) {
      throw new Error(
        'computeAdjointGradient: production path replay accepts only material emissive RGB.',
      );
    }
    if (
      !Number.isSafeInteger(param.offset) ||
      param.offset < 0 ||
      param.offset > U32_MAX - 2 ||
      param.offset + 3 > req.gradientLength
    ) {
      throw new RangeError(
        `computeAdjointGradient: invalid emissive gradient offset ${param.offset}.`,
      );
    }
    const primitive = scene.primitives.find((candidate) => candidate.id === param.id);
    if (primitive == null) {
      throw new Error(`computeAdjointGradient: primitive "${param.id}" does not exist.`);
    }
    const targetIssue = emissiveReplayTargetIssue(scene, primitive);
    if (targetIssue != null) {
      throw new Error(
        `computeAdjointGradient: target is outside exact emissive replay: ${targetIssue.message}.`,
      );
    }
  }
}

export class AdjointPass {
  readonly #device: GPUDevice;
  readonly #sampling: PtWebgpuSamplingMode;
  #pipeline: GPUComputePipeline | null = null;

  constructor(device: GPUDevice, sampling: PtWebgpuSamplingMode = 'pcg') {
    this.#device = device;
    this.#sampling = sampling;
  }

  async computeGradient(
    req: AdjointGradientRequest,
    sceneBuffers: UploadedSceneBuffers,
    lastFrame: FrameInput,
    scene: Scene,
    materialIndexForPrimitive: MaterialIndexForPrimitive,
    cameraVisibleEmitters: boolean,
  ): Promise<Float32Array> {
    // Nothing observable on the GPU may happen before this check completes.
    validateAdjointRequest(req, scene, cameraVisibleEmitters);

    const viewProjection = multiplyMat4(lastFrame.projMatrix, lastFrame.viewMatrix);
    const inverseViewProjection = invertMat4(asMat4(viewProjection));
    if (inverseViewProjection == null) {
      throw new Error('computeAdjointGradient: camera viewProj is not invertible.');
    }

    const descriptors = new Uint32Array(req.params.length * 4);
    const descriptorFloats = new Float32Array(descriptors.buffer);
    const emissiveIntensities: number[] = [];
    for (let index = 0; index < req.params.length; index += 1) {
      const param = req.params[index]!;
      const materialId = materialIndexForPrimitive(scene, param.id, NO_ANALYTIC_SHAPES);
      if (
        materialId == null ||
        !Number.isSafeInteger(materialId) ||
        materialId < 0 ||
        materialId > U32_MAX
      ) {
        throw new Error(`computeAdjointGradient: no material index for primitive "${param.id}".`);
      }
      const primitive = scene.primitives.find((candidate) => candidate.id === param.id)!;
      const base = index * 4;
      descriptors[base] = materialId >>> 0;
      descriptors[base + 1] = ADJOINT_FIELD_EMISSIVE;
      descriptors[base + 2] = param.offset >>> 0;
      const emissiveIntensity = resolveEmissiveIntensity(
        primitive.material.emissiveIntensity,
      );
      descriptorFloats[base + 3] = emissiveIntensity;
      emissiveIntensities.push(emissiveIntensity);
    }
    const gradientScale = computeAdjointGradientScale(req, emissiveIntensities);

    const geometryOverride = buildAdjointWorldSpaceGeometryOverride(
      scene,
      materialIndexForPrimitive,
    );
    const triangleCount = geometryOverride?.triangleCount ?? sceneBuffers.triangleCount;
    assertPositiveU32(triangleCount, 'triangleCount');
    if (geometryOverride != null) {
      checkedProduct(
        'world-space position buffer byte size',
        geometryOverride.positions.length,
        Float32Array.BYTES_PER_ELEMENT,
      );
      checkedProduct(
        'world-space index buffer byte size',
        geometryOverride.indices.length,
        Uint32Array.BYTES_PER_ELEMENT,
      );
      checkedProduct(
        'world-space material-id buffer byte size',
        geometryOverride.triMaterialIds.length,
        Uint32Array.BYTES_PER_ELEMENT,
      );
    }
    validateDeviceResourceLimits(this.#device, req, geometryOverride);

    const uniformData = new ArrayBuffer(ADJOINT_PARAMS_UBO_BYTES);
    const uniformFloats = new Float32Array(uniformData);
    const uniformU32 = new Uint32Array(uniformData);
    uniformFloats.set(inverseViewProjection, 0);
    const cameraPosition = resolveFrameCameraPosition(
      lastFrame,
      'PTEngineWebGPU.computeAdjointGradient',
    );
    uniformFloats[16] = cameraPosition[0];
    uniformFloats[17] = cameraPosition[1];
    uniformFloats[18] = cameraPosition[2];
    uniformFloats[19] = 1;
    uniformU32[20] = req.width >>> 0;
    uniformU32[21] = req.height >>> 0;
    uniformU32[22] = triangleCount >>> 0;
    uniformU32[23] = req.params.length >>> 0;
    uniformU32[24] = req.channels >>> 0;
    uniformU32[25] = req.samples >>> 0;
    uniformFloats[26] = gradientScale;

    const device = this.#device;
    if (this.#pipeline == null) {
      const module = device.createShaderModule({
        code:
          this.#sampling === 'pcg'
            ? PT_WEBGPU_ADJOINT_PASS_WGSL
            : composePtWebgpuAdjointPassWgsl(this.#sampling),
      });
      this.#pipeline = device.createComputePipeline({
        label: 'vitrum.pt-webgpu.adjointPass.emissive',
        layout: 'auto',
        compute: { module, entryPoint: 'main' },
      });
    }
    const pipeline = this.#pipeline;

    const usage = (globalThis as { GPUBufferUsage: typeof GPUBufferUsage }).GPUBufferUsage;
    const created: GPUBuffer[] = [];
    let stage: GPUBuffer | null = null;
    let stageMapped = false;
    const createBuffer = (
      size: number,
      bufferUsage: number,
      data?: ArrayBufferView,
    ): GPUBuffer => {
      const buffer = device.createBuffer({ size: Math.max(size, 16), usage: bufferUsage });
      created.push(buffer);
      if (data != null) {
        device.queue.writeBuffer(
          buffer,
          0,
          data.buffer,
          data.byteOffset,
          data.byteLength,
        );
      }
      return buffer;
    };

    let result: Float32Array | null = null;
    let operationError: unknown;
    let cleanupError: unknown;
    try {
      const paramsBuffer = createBuffer(
        ADJOINT_PARAMS_UBO_BYTES,
        usage.UNIFORM | usage.COPY_DST,
        uniformFloats,
      );
      const lossBuffer = createBuffer(
        req.dLoss_dRendered.byteLength,
        usage.STORAGE | usage.COPY_DST,
        req.dLoss_dRendered,
      );
      const gradientBuffer = createBuffer(
        req.gradientLength * 4,
        usage.STORAGE | usage.COPY_SRC | usage.COPY_DST,
        new Int32Array(req.gradientLength),
      );
      const descriptorBuffer = createBuffer(
        descriptors.byteLength,
        usage.STORAGE | usage.COPY_DST,
        descriptors,
      );
      stage = createBuffer(req.gradientLength * 4, usage.MAP_READ | usage.COPY_DST);

      const positionsBuffer = geometryOverride == null
        ? sceneBuffers.positionsBuffer
        : createBuffer(
            geometryOverride.positions.byteLength,
            usage.STORAGE | usage.COPY_DST,
            geometryOverride.positions,
          );
      const indicesBuffer = geometryOverride == null
        ? sceneBuffers.indicesBuffer
        : createBuffer(
            geometryOverride.indices.byteLength,
            usage.STORAGE | usage.COPY_DST,
            geometryOverride.indices,
          );
      const materialIdsBuffer = geometryOverride == null
        ? sceneBuffers.triMaterialIdsBuffer
        : createBuffer(
            geometryOverride.triMaterialIds.byteLength,
            usage.STORAGE | usage.COPY_DST,
            geometryOverride.triMaterialIds,
          );

      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: positionsBuffer } },
          { binding: 2, resource: { buffer: indicesBuffer } },
          { binding: 3, resource: { buffer: materialIdsBuffer } },
          { binding: 4, resource: { buffer: sceneBuffers.materialsBuffer } },
          { binding: 5, resource: { buffer: lossBuffer } },
          { binding: 6, resource: { buffer: gradientBuffer } },
          { binding: 7, resource: { buffer: descriptorBuffer } },
        ],
      });

      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(req.width / 8), Math.ceil(req.height / 8));
      pass.end();
      encoder.copyBufferToBuffer(
        gradientBuffer,
        0,
        stage,
        0,
        req.gradientLength * 4,
      );
      device.queue.submit([encoder.finish()]);

      await stage.mapAsync(
        (globalThis as { GPUMapMode: typeof GPUMapMode }).GPUMapMode.READ,
      );
      stageMapped = true;
      const fixedPoint = new Int32Array(stage.getMappedRange().slice(0));
      result = new Float32Array(req.gradientLength);
      for (let index = 0; index < req.gradientLength; index += 1) {
        result[index] = fixedPoint[index]! / gradientScale;
      }
    } catch (error) {
      operationError = error;
    } finally {
      if (stageMapped && stage != null) {
        try {
          stage.unmap();
        } catch (error) {
          cleanupError ??= error;
        }
      }
      for (const buffer of created) {
        try {
          buffer.destroy();
        } catch (error) {
          cleanupError ??= error;
        }
      }
    }

    if (operationError != null) {
      throw operationError instanceof Error
        ? operationError
        : new Error('computeAdjointGradient: GPU operation rejected with a non-Error value.');
    }
    if (cleanupError != null) {
      throw cleanupError instanceof Error
        ? cleanupError
        : new Error('computeAdjointGradient: cleanup rejected with a non-Error value.');
    }
    if (result == null) {
      throw new Error('computeAdjointGradient: readback completed without a gradient result.');
    }
    return result;
  }

  /** GPUComputePipeline has no explicit destroy method. */
  dispose(): void {
    this.#pipeline = null;
  }
}

/**
 * Flatten transforms and instances when the uploaded triangle buffer is not
 * already world-space. Analytic primitives are rejected before this helper.
 */
export function buildAdjointWorldSpaceGeometryOverride(
  scene: Scene,
  materialIndexForPrimitive: MaterialIndexForPrimitive,
): AdjointWorldSpaceGeometry | null {
  const solvedScene = applySolveSkinToScene(scene);
  const replayPrimitives =
    solvedScene.primitives.filter(isTriangleBackedPrimitive);
  if (replayPrimitives.length !== solvedScene.primitives.length) {
    throw new Error('buildAdjointWorldSpaceGeometryOverride: non-triangle primitive.');
  }
  for (const primitive of replayPrimitives) {
    const issue = emissiveReplayPrimitiveIssue(primitive);
    if (issue != null) {
      throw new Error(
        `buildAdjointWorldSpaceGeometryOverride: ${issue.message}.`,
      );
    }
  }
  const needsOverride = replayPrimitives.some(needsWorldSpaceOverride);
  if (!needsOverride) return null;

  const merged = mergeWorldSpaceFromCore(solvedScene, {
    positionStride: 4,
    filter: isTriangleBackedPrimitive,
  });
  if (merged.triangleCount < 1) {
    throw new Error('buildAdjointWorldSpaceGeometryOverride: replay geometry is empty.');
  }

  const mergedMaterialIds = new Uint32Array(merged.triangleCount);
  for (const range of merged.meshVertexRanges) {
    const materialId = materialIndexForPrimitive(
      solvedScene,
      range.name,
      NO_ANALYTIC_SHAPES,
    );
    if (materialId == null) {
      throw new Error(
        `buildAdjointWorldSpaceGeometryOverride: no material for "${range.name}".`,
      );
    }
    const end = Math.min(mergedMaterialIds.length, range.triStart + range.triCount);
    mergedMaterialIds.fill(materialId >>> 0, range.triStart, end);
  }

  const triMaterialIds = new Uint32Array(merged.triangleCount);
  for (let triangle = 0; triangle < merged.triangleCount; triangle += 1) {
    const mergedTriangle = merged.bvhTriToMergedTri[triangle] ?? triangle;
    triMaterialIds[triangle] = mergedMaterialIds[mergedTriangle] ?? 0;
  }

  return {
    positions: merged.positions,
    indices: expandIndicesToStride4(merged.indices),
    triMaterialIds,
    triangleCount: merged.triangleCount,
  };
}

function isTriangleBackedPrimitive(
  primitive: ScenePrimitive,
): primitive is Extract<
  ScenePrimitive,
  { kind: 'mesh' | 'skinned-mesh' | 'instanced-mesh' }
> {
  return (
    primitive.kind === 'mesh' ||
    primitive.kind === 'skinned-mesh' ||
    primitive.kind === 'instanced-mesh'
  );
}

function needsWorldSpaceOverride(
  primitive: Extract<
    ScenePrimitive,
    { kind: 'mesh' | 'skinned-mesh' | 'instanced-mesh' }
  >,
): boolean {
  if (primitive.kind === 'instanced-mesh') return primitive.instances.length > 0;
  return primitive.transform != null && !isIdentityMat4(primitive.transform);
}

function isIdentityMat4(transform: Float32Array): boolean {
  const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  if (transform.length < identity.length) return false;
  return identity.every(
    (expected, index) => Math.abs((transform[index] ?? 0) - expected) <= 1e-6,
  );
}
