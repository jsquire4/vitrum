/**
 * PR-7 — GPU linear blend skinning into the live ReSTIR `bvhPositions` buffer,
 * then same-frame CPU BVH refit (nodes-only upload) via `applyGpuSkinnedRefit`.
 */

import type { Scene, SkinnedMeshPrimitive } from '@vitrum/core';
import { combineSkinMatrices, solveSkin } from '@vitrum/three-bindings';
import { GPU_SKIN_BVH_WGSL } from './gpuSkinBvh.wgsl.js';
import type { HybridEngine } from '../HybridEngine.js';

const UNIFORM = 0x40;
const STORAGE = 0x80;
const COPY_DST = 0x02;

interface MeshGpuState {
  readonly vertexCount: number;
  readonly boneCount: number;
  readonly uniformBuffer: GPUBuffer;
  readonly boneBuffer: GPUBuffer;
  readonly restPosBuffer: GPUBuffer;
  readonly restNormBuffer: GPUBuffer;
  readonly skinIdxBuffer: GPUBuffer;
  readonly skinWeightBuffer: GPUBuffer;
  readonly bindGroup: GPUBindGroup;
  readonly pipeline: GPUComputePipeline;
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

function destroyMeshState(state: MeshGpuState): void {
  state.uniformBuffer.destroy();
  state.boneBuffer.destroy();
  state.restPosBuffer.destroy();
  state.restNormBuffer.destroy();
  state.skinIdxBuffer.destroy();
  state.skinWeightBuffer.destroy();
}

export class GpuSkinningSubsystem {
  readonly #device: GPUDevice;
  readonly #preferGpu: boolean;
  readonly #meshes = new Map<string, MeshGpuState>();
  #bvhPipeline: GPUComputePipeline | null = null;

  constructor(device: GPUDevice, preferGpu: boolean) {
    this.#device = device;
    this.#preferGpu = preferGpu;
  }

  dispose(): void {
    for (const state of this.#meshes.values()) {
      destroyMeshState(state);
    }
    this.#meshes.clear();
    this.#bvhPipeline = null;
  }

  run(engine: HybridEngine, scene: Scene): void {
    const bvhPositions = engine.getGpuSkinningBvhBuffer();
    const meshVertexRanges = engine.getMeshVertexRanges();
    if (bvhPositions == null || meshVertexRanges == null) {
      return;
    }

    const encoder = this.#device.createCommandEncoder({ label: 'vitrum.gpuSkinBvh' });
    const gpuSkinnedIds: Array<{
      id: string;
      positions: Float32Array;
      normals: Float32Array;
    }> = [];

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

      const range = meshVertexRanges.find((r) => r.name === id);
      if (range == null || range.vertexCount === 0) {
        const { positions, normals } = solveSkin(prim);
        engine.updatePrimitive(id, { positions, normals });
        continue;
      }

      const bvhMode = engine.getBvhMode();
      let baseVertex = range.vertexStart;
      if (bvhMode === 'tlas') {
        const binding = engine.getPrimitiveTlasBindings()?.find((b) => b.primitiveId === id);
        if (binding == null || binding.vertexCount === 0) {
          const { positions, normals } = solveSkin(prim);
          engine.updatePrimitive(id, { positions, normals });
          continue;
        }
        baseVertex = binding.vertexStart;
      }

      const state = this.#ensureMesh(prim, bvhPositions);
      const combined = combineSkinMatrices(prim.bones, prim.boneInverses, state.boneCount);
      this.#device.queue.writeBuffer(state.boneBuffer, 0, new Float32Array(combined));

      const uniformBytes = new ArrayBuffer(80);
      const u32 = new Uint32Array(uniformBytes);
      u32[0] = state.vertexCount;
      u32[1] = baseVertex;
      u32[2] = engine.getBvhMode() === 'tlas' ? 0 : 1;
      u32[3] = 0;
      new Float32Array(uniformBytes).set(range.matrixWorldAtBuild, 4);
      this.#device.queue.writeBuffer(state.uniformBuffer, 0, uniformBytes);

      const pass = encoder.beginComputePass({ label: `vitrum.gpuSkinBvh.${id}` });
      pass.setPipeline(state.pipeline);
      pass.setBindGroup(0, state.bindGroup);
      pass.dispatchWorkgroups(Math.ceil(state.vertexCount / 64), 1, 1);
      pass.end();

      const { positions, normals } = solveSkin(prim);
      gpuSkinnedIds.push({ id, positions, normals });
    }

    if (gpuSkinnedIds.length > 0) {
      this.#device.queue.submit([encoder.finish()]);
      for (const { id, positions, normals } of gpuSkinnedIds) {
        engine.applyGpuSkinnedRefit(id, positions, normals);
      }
    }
  }

  #ensureMesh(prim: SkinnedMeshPrimitive, bvhPositions: GPUBuffer): MeshGpuState {
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

    const restPosBuffer = mkStorage(`vitrum.gpuSkinBvh.${id}.restPos`, vertBytes);
    const restNormBuffer = mkStorage(`vitrum.gpuSkinBvh.${id}.restNorm`, vertBytes);
    const skinIdxBuffer = mkStorage(`vitrum.gpuSkinBvh.${id}.skinIdx`, skinIdx.byteLength);
    const skinWeightBuffer = mkStorage(`vitrum.gpuSkinBvh.${id}.skinW`, skinW.byteLength);
    const boneBuffer = mkStorage(`vitrum.gpuSkinBvh.${id}.bones`, boneBytes);
    const uniformBuffer = device.createBuffer({
      label: `vitrum.gpuSkinBvh.${id}.uniform`,
      size: 80,
      usage: UNIFORM | COPY_DST,
    });

    device.queue.writeBuffer(restPosBuffer, 0, new Float32Array(restPos));
    device.queue.writeBuffer(restNormBuffer, 0, new Float32Array(restNorm));
    device.queue.writeBuffer(skinIdxBuffer, 0, new Uint32Array(skinIdx));
    device.queue.writeBuffer(skinWeightBuffer, 0, new Float32Array(skinW));

    if (this.#bvhPipeline == null) {
      const module = device.createShaderModule({
        label: 'vitrum.gpuSkinBvh.module',
        code: GPU_SKIN_BVH_WGSL,
      });
      this.#bvhPipeline = device.createComputePipeline({
        label: 'vitrum.gpuSkinBvh.pipeline',
        layout: 'auto',
        compute: { module, entryPoint: 'main' },
      });
    }
    const pipeline = this.#bvhPipeline;

    const bindGroup = device.createBindGroup({
      label: `vitrum.gpuSkinBvh.${id}.bindGroup`,
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: restPosBuffer } },
        { binding: 2, resource: { buffer: restNormBuffer } },
        { binding: 3, resource: { buffer: skinIdxBuffer } },
        { binding: 4, resource: { buffer: skinWeightBuffer } },
        { binding: 5, resource: { buffer: boneBuffer } },
        { binding: 6, resource: { buffer: bvhPositions } },
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
      bindGroup,
      pipeline,
    };
    this.#meshes.set(id, state);
    return state;
  }
}
