/**
 * PR-7 — GPU linear blend skinning; readback completes between frames and is
 * applied at the start of the next `renderFrame` via `updatePrimitive`.
 */

import type { Scene, SkinnedMeshPrimitive } from '@vitrum/core';
import { combineSkinMatrices, solveSkin } from '@vitrum/three-bindings';
import { GPU_SKIN_LBS_WGSL } from './gpuSkinLbs.wgsl.js';
import type { HybridEngine } from '../HybridEngine.js';

const UNIFORM = 0x40;
const STORAGE = 0x80;
const COPY_DST = 0x02;
const MAP_READ = 0x01;

interface MeshGpuState {
  readonly vertexCount: number;
  readonly boneCount: number;
  readonly uniformBuffer: GPUBuffer;
  readonly boneBuffer: GPUBuffer;
  readonly restPosBuffer: GPUBuffer;
  readonly restNormBuffer: GPUBuffer;
  readonly skinIdxBuffer: GPUBuffer;
  readonly skinWeightBuffer: GPUBuffer;
  readonly outPosBuffer: GPUBuffer;
  readonly outNormBuffer: GPUBuffer;
  readonly stagingPos: GPUBuffer;
  readonly stagingNorm: GPUBuffer;
  readonly bindGroup: GPUBindGroup;
  readonly pipeline: GPUComputePipeline;
}

interface PendingReadback {
  readonly positions: Float32Array;
  readonly normals: Float32Array;
}

function packVec4Positions(positions: Float32Array, count: number): Float32Array {
  const out = new Float32Array(count * 4);
  for (let i = 0; i < count; i += 1) {
    const o = i * 4;
    const s = i * 3;
    out[o] = positions[s]!;
    out[o + 1] = positions[s + 1]!;
    out[o + 2] = positions[s + 2]!;
    out[o + 3] = 1;
  }
  return out;
}

function packVec4Normals(normals: Float32Array, count: number): Float32Array {
  const out = new Float32Array(count * 4);
  for (let i = 0; i < count; i += 1) {
    const o = i * 4;
    const s = i * 3;
    out[o] = normals[s]!;
    out[o + 1] = normals[s + 1]!;
    out[o + 2] = normals[s + 2]!;
    out[o + 3] = 0;
  }
  return out;
}

function unpackVec3(mapped: Float32Array, vertCount: number): Float32Array {
  const out = new Float32Array(vertCount * 3);
  for (let i = 0; i < vertCount; i += 1) {
    const o4 = i * 4;
    const o3 = i * 3;
    out[o3] = mapped[o4]!;
    out[o3 + 1] = mapped[o4 + 1]!;
    out[o3 + 2] = mapped[o4 + 2]!;
  }
  return out;
}

function destroyMeshState(state: MeshGpuState): void {
  state.uniformBuffer.destroy();
  state.boneBuffer.destroy();
  state.restPosBuffer.destroy();
  state.restNormBuffer.destroy();
  state.skinIdxBuffer.destroy();
  state.skinWeightBuffer.destroy();
  state.outPosBuffer.destroy();
  state.outNormBuffer.destroy();
  state.stagingPos.destroy();
  state.stagingNorm.destroy();
}

export class GpuSkinningSubsystem {
  readonly #device: GPUDevice;
  readonly #preferGpu: boolean;
  readonly #meshes = new Map<string, MeshGpuState>();
  readonly #pending = new Map<string, PendingReadback>();

  constructor(device: GPUDevice, preferGpu: boolean) {
    this.#device = device;
    this.#preferGpu = preferGpu;
  }

  dispose(): void {
    for (const state of this.#meshes.values()) {
      destroyMeshState(state);
    }
    this.#meshes.clear();
    this.#pending.clear();
  }

  run(engine: HybridEngine, scene: Scene): void {
    for (const [id, pending] of this.#pending) {
      engine.updatePrimitive(id, { positions: pending.positions, normals: pending.normals });
    }
    this.#pending.clear();

    for (const prim of scene.primitives) {
      if (prim.kind !== 'skinned-mesh') continue;
      const id = String(prim.id);
      const hasMorph =
        (prim.morphTargets?.length ?? 0) > 0 &&
        prim.morphWeights != null &&
        prim.morphWeights.some((w) => w !== 0);
      if (!this.#preferGpu || hasMorph || typeof this.#device.createComputePipeline !== 'function') {
        const { positions, normals } = solveSkin(prim);
        engine.updatePrimitive(id, { positions, normals });
        continue;
      }
      if (!this.#meshes.has(id)) {
        const boot = solveSkin(prim);
        engine.updatePrimitive(id, { positions: boot.positions, normals: boot.normals });
      }
      const state = this.#ensureMesh(prim);
      const combined = combineSkinMatrices(prim.bones, prim.boneInverses, state.boneCount);
      this.#device.queue.writeBuffer(state.boneBuffer, 0, new Float32Array(combined));
      const params = new Uint32Array([state.vertexCount, state.boneCount, 0, 0]);
      this.#device.queue.writeBuffer(state.uniformBuffer, 0, params);
      const encoder = this.#device.createCommandEncoder({ label: 'vitrum.gpuSkin.dispatch' });
      const pass = encoder.beginComputePass({ label: 'vitrum.gpuSkin.pass' });
      pass.setPipeline(state.pipeline);
      pass.setBindGroup(0, state.bindGroup);
      pass.dispatchWorkgroups(Math.ceil(state.vertexCount / 64), 1, 1);
      pass.end();
      encoder.copyBufferToBuffer(state.outPosBuffer, 0, state.stagingPos, 0, state.vertexCount * 16);
      encoder.copyBufferToBuffer(state.outNormBuffer, 0, state.stagingNorm, 0, state.vertexCount * 16);
      this.#device.queue.submit([encoder.finish()]);
      const vertCount = state.vertexCount;
      void this.#device.queue.onSubmittedWorkDone().then(async () => {
        await state.stagingPos.mapAsync(MAP_READ);
        await state.stagingNorm.mapAsync(MAP_READ);
        const posRange = state.stagingPos.getMappedRange(0, vertCount * 16);
        const normRange = state.stagingNorm.getMappedRange(0, vertCount * 16);
        const positions = unpackVec3(new Float32Array(posRange.slice(0)), vertCount);
        const normals = unpackVec3(new Float32Array(normRange.slice(0)), vertCount);
        state.stagingPos.unmap();
        state.stagingNorm.unmap();
        this.#pending.set(id, { positions, normals });
      });
    }
  }

  #ensureMesh(prim: SkinnedMeshPrimitive): MeshGpuState {
    const id = String(prim.id);
    const existing = this.#meshes.get(id);
    const vertCount = prim.positions.length / 3;
    const boneCount = prim.bones.length / 16;
    if (existing != null && existing.vertexCount === vertCount && existing.boneCount === boneCount) {
      return existing;
    }
    if (existing != null) {
      destroyMeshState(existing);
      this.#meshes.delete(id);
    }
    const device = this.#device;
    const restPos = packVec4Positions(prim.positions, vertCount);
    const restNorm = packVec4Normals(prim.normals, vertCount);
    const skinIdx = new Uint32Array(prim.skinIndices);
    const skinW = new Float32Array(prim.skinWeights);
    const vertBytes = vertCount * 16;
    const boneBytes = boneCount * 16 * 4;
    const mkStorage = (label: string, size: number) =>
      device.createBuffer({
        label,
        size: Math.max(16, size),
        usage: STORAGE | COPY_DST,
      });
    const restPosBuffer = mkStorage(`vitrum.gpuSkin.${id}.restPos`, vertBytes);
    const restNormBuffer = mkStorage(`vitrum.gpuSkin.${id}.restNorm`, vertBytes);
    const skinIdxBuffer = mkStorage(`vitrum.gpuSkin.${id}.skinIdx`, skinIdx.byteLength);
    const skinWeightBuffer = mkStorage(`vitrum.gpuSkin.${id}.skinW`, skinW.byteLength);
    const boneBuffer = mkStorage(`vitrum.gpuSkin.${id}.bones`, boneBytes);
    const outPosBuffer = mkStorage(`vitrum.gpuSkin.${id}.outPos`, vertBytes);
    const outNormBuffer = mkStorage(`vitrum.gpuSkin.${id}.outNorm`, vertBytes);
    const stagingPos = device.createBuffer({
      label: `vitrum.gpuSkin.${id}.stagingPos`,
      size: vertBytes,
      usage: MAP_READ | COPY_DST,
    });
    const stagingNorm = device.createBuffer({
      label: `vitrum.gpuSkin.${id}.stagingNorm`,
      size: vertBytes,
      usage: MAP_READ | COPY_DST,
    });
    const uniformBuffer = device.createBuffer({
      label: `vitrum.gpuSkin.${id}.uniform`,
      size: 16,
      usage: UNIFORM | COPY_DST,
    });
    device.queue.writeBuffer(restPosBuffer, 0, new Float32Array(restPos));
    device.queue.writeBuffer(restNormBuffer, 0, new Float32Array(restNorm));
    device.queue.writeBuffer(skinIdxBuffer, 0, new Uint32Array(skinIdx));
    device.queue.writeBuffer(skinWeightBuffer, 0, new Float32Array(skinW));
    const module = device.createShaderModule({
      label: 'vitrum.gpuSkin.module',
      code: GPU_SKIN_LBS_WGSL,
    });
    const pipeline = device.createComputePipeline({
      label: 'vitrum.gpuSkin.pipeline',
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    const bindGroup = device.createBindGroup({
      label: `vitrum.gpuSkin.${id}.bindGroup`,
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: restPosBuffer } },
        { binding: 2, resource: { buffer: restNormBuffer } },
        { binding: 3, resource: { buffer: skinIdxBuffer } },
        { binding: 4, resource: { buffer: skinWeightBuffer } },
        { binding: 5, resource: { buffer: boneBuffer } },
        { binding: 6, resource: { buffer: outPosBuffer } },
        { binding: 7, resource: { buffer: outNormBuffer } },
      ],
    });
    const state: MeshGpuState = {
      vertexCount: vertCount,
      boneCount,
      uniformBuffer,
      boneBuffer,
      restPosBuffer,
      restNormBuffer,
      skinIdxBuffer,
      skinWeightBuffer,
      outPosBuffer,
      outNormBuffer,
      stagingPos,
      stagingNorm,
      bindGroup,
      pipeline,
    };
    this.#meshes.set(id, state);
    return state;
  }
}
