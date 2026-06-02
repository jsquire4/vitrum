/**
 * PR-7 — GPU linear blend skinning into the live ReSTIR `bvhPositions` buffer,
 * then same-frame CPU BVH refit (nodes-only upload) via `applyGpuSkinnedRefit`.
 */

import type { Scene, ScenePrimitive, SkinnedMeshPrimitive } from '@vitrum/core';
import { combineSkinMatrices, solveSkin } from '@vitrum/core';
import { GPU_SKIN_BVH_WITH_NORMALS_WGSL } from './gpuSkinBvh.wgsl.js';
import type { ReSTIRBvhMode, SceneBVHBuffers } from '../restir/bvhCompute.js';

/**
 * Narrow back-reference the skinning subsystem needs from its host engine.
 *
 * Previously `run()` took the whole `HybridEngine`, which made
 * `skin/GpuSkinningSubsystem.ts` import `../HybridEngine.js` — the package's
 * only import cycle (HybridEngine imports the subsystem; the subsystem imported
 * HybridEngine). Depending on this 6-method surface instead breaks the cycle.
 * `HybridEngine` satisfies it structurally, so the engine passes `this`
 * unchanged (upcast to `GpuSkinningHost`).
 */
export interface GpuSkinningHost {
  /** Merged-BVH world-position SSBO target for the LBS compute write. */
  getGpuSkinningBvhBuffer(): GPUBuffer | null;
  /** WS1 — merged-BVH per-vertex normal SSBO target. The LBS compute writes
   *  inverse-transpose skinned normals here at `baseVertex+vi` so the smooth-
   *  shading-normal blend reads the deformed normal. Null before pipeline init. */
  getGpuSkinningNormalBuffer(): GPUBuffer | null;
  /** Per-mesh vertex ranges in the merged BVH. */
  getMeshVertexRanges(): SceneBVHBuffers['meshVertexRanges'] | null;
  /** Active ReSTIR BVH layout (`merged` world positions vs `tlas` local BLAS). */
  getBvhMode(): ReSTIRBvhMode | null;
  /** Per-primitive TLAS bindings (vertex ranges in `tlas` mode). */
  getPrimitiveTlasBindings(): SceneBVHBuffers['primitiveTlasBindings'] | null;
  /** CPU-skin fallback path: push solved positions/normals through the
   *  standard incremental geometry update. */
  updatePrimitive(id: string, patch: Partial<ScenePrimitive>): void;
  /** GPU-skin path: refit BVH nodes after the compute pass wrote world
   *  positions directly into the live `bvhPositions` buffer. */
  applyGpuSkinnedRefit(id: string, localPositions?: Float32Array, localNormals?: Float32Array): void;
}

const UNIFORM = 0x40;
const STORAGE = 0x80;
const COPY_DST = 0x02;

/** Epsilon for the bindMatrix identity comparison. The GPU skin kernel only
 *  applies the blended bone matrices (`combineSkinMatrices`), NOT the
 *  bindMatrix / bindMatrixInverse wrapping that the CPU `solveSkin` honours,
 *  so a primitive with a non-identity bindMatrix would skin WRONG on the GPU
 *  path. We route those to the CPU solver (same fallback shape as `hasMorph`).
 *  glTF-typical bind is identity, so the GPU fast path is preserved for the
 *  common case. */
const BIND_IDENTITY_EPS = 1e-6;

/** Column-major identity (== `THREE.Matrix4` default). */
const IDENTITY_MAT4 = new Float32Array([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
]);

/**
 * True when the primitive carries a `bindMatrix` that is NOT (within
 * {@link BIND_IDENTITY_EPS}) the identity. An absent or identity bindMatrix
 * returns false — the GPU fast path is safe because the kernel's
 * `combineSkinMatrices(bones, boneInverses)` collapses to the same transform
 * the CPU solver computes when `hasBind` is false. A non-identity bind makes
 * the two diverge (the CPU solver pre/post-multiplies by `bindMatrix` /
 * `bindMatrixInverse`), so this gate sends those meshes to the CPU solver.
 */
function hasNonIdentityBind(prim: SkinnedMeshPrimitive): boolean {
  const bm = prim.bindMatrix;
  // `solveSkin` only applies bind when BOTH bindMatrix and bindMatrixInverse
  // are present; mirror that so a half-specified bind doesn't force a fallback
  // the CPU solver would itself ignore.
  if (bm == null || prim.bindMatrixInverse == null) return false;
  if (bm.length !== 16) return true; // malformed → don't trust the fast path
  for (let i = 0; i < 16; i += 1) {
    if (Math.abs(bm[i]! - IDENTITY_MAT4[i]!) > BIND_IDENTITY_EPS) return true;
  }
  return false;
}

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
  /** WS1 — identities of the SHARED merged buffers the cached bind group was
   *  built against. A full BVH rebuild swaps these buffers (same vert/bone
   *  count), so we rebuild the bind group when either identity changes. */
  readonly boundPositionBuffer: GPUBuffer;
  readonly boundNormalBuffer: GPUBuffer;
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
  // boundPositionBuffer / boundNormalBuffer are SHARED merged buffers owned by
  // BvhBufferHost — do NOT destroy them here.
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

  run(host: GpuSkinningHost, scene: Scene): void {
    const bvhPositions = host.getGpuSkinningBvhBuffer();
    const bvhNormals = host.getGpuSkinningNormalBuffer();
    const meshVertexRanges = host.getMeshVertexRanges();
    if (bvhPositions == null || bvhNormals == null || meshVertexRanges == null) {
      return;
    }

    const encoder = this.#device.createCommandEncoder({ label: 'vitrum.gpuSkinBvh' });
    const gpuSkinnedIds: string[] = [];

    for (const prim of scene.primitives) {
      if (prim.kind !== 'skinned-mesh') continue;
      const id = String(prim.id);
      const hasMorph =
        (prim.morphTargets?.length ?? 0) > 0 &&
        prim.morphWeights != null &&
        prim.morphWeights.some((w) => w !== 0);
      // A non-identity bindMatrix must take the CPU `solveSkin` path: the GPU
      // kernel applies only `combineSkinMatrices(bones, boneInverses)` and does
      // NOT wrap by bindMatrix / bindMatrixInverse, so it would skin positions
      // AND normals incorrectly for a bound mesh. Same fallback shape as morphs.
      if (
        !this.#preferGpu ||
        hasMorph ||
        hasNonIdentityBind(prim) ||
        typeof this.#device.createComputePipeline !== 'function'
      ) {
        const { positions, normals } = solveSkin(prim);
        host.updatePrimitive(id, { positions, normals });
        continue;
      }

      const range = meshVertexRanges.find((r) => r.name === id);
      if (range == null || range.vertexCount === 0) {
        const { positions, normals } = solveSkin(prim);
        host.updatePrimitive(id, { positions, normals });
        continue;
      }

      const bvhMode = host.getBvhMode();
      let baseVertex = range.vertexStart;
      if (bvhMode === 'tlas') {
        const binding = host.getPrimitiveTlasBindings()?.find((b) => b.primitiveId === id);
        if (binding == null || binding.vertexCount === 0) {
          const { positions, normals } = solveSkin(prim);
          host.updatePrimitive(id, { positions, normals });
          continue;
        }
        baseVertex = binding.vertexStart;
      }

      const state = this.#ensureMesh(prim, bvhPositions, bvhNormals);
      const combined = combineSkinMatrices(prim.bones, prim.boneInverses, state.boneCount);
      this.#device.queue.writeBuffer(state.boneBuffer, 0, new Float32Array(combined));

      const uniformBytes = new ArrayBuffer(80);
      const u32 = new Uint32Array(uniformBytes);
      u32[0] = state.vertexCount;
      u32[1] = baseVertex;
      u32[2] = host.getBvhMode() === 'tlas' ? 0 : 1;
      u32[3] = 0;
      new Float32Array(uniformBytes).set(range.matrixWorldAtBuild, 4);
      this.#device.queue.writeBuffer(state.uniformBuffer, 0, uniformBytes);

      const pass = encoder.beginComputePass({ label: `vitrum.gpuSkinBvh.${id}` });
      pass.setPipeline(state.pipeline);
      pass.setBindGroup(0, state.bindGroup);
      pass.dispatchWorkgroups(Math.ceil(state.vertexCount / 64), 1, 1);
      pass.end();

      gpuSkinnedIds.push(id);
    }

    if (gpuSkinnedIds.length > 0) {
      this.#device.queue.submit([encoder.finish()]);
      for (const id of gpuSkinnedIds) {
        host.applyGpuSkinnedRefit(id);
      }
    }
  }

  #ensureMesh(
    prim: SkinnedMeshPrimitive,
    bvhPositions: GPUBuffer,
    bvhNormals: GPUBuffer,
  ): MeshGpuState {
    const id = String(prim.id);
    const existing = this.#meshes.get(id);
    const vertCount = prim.positions.length / 3;
    const boneCount = prim.bones.length / 16;
    // WS1 — reuse only if the cached bind group still points at the LIVE shared
    // position + normal buffers (a full BVH rebuild swaps both, even at the same
    // vert/bone count). A stale bind group would skin into the destroyed buffer.
    if (
      existing != null &&
      existing.vertexCount === vertCount &&
      existing.boneCount === boneCount &&
      existing.boundPositionBuffer === bvhPositions &&
      existing.boundNormalBuffer === bvhNormals
    ) {
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
        code: GPU_SKIN_BVH_WITH_NORMALS_WGSL,
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
        // WS1 — binding 7 is the SHARED merged bvh_normal buffer; the kernel
        // writes skinned normals at `baseVertex+vi` (NOT a per-mesh buffer).
        { binding: 7, resource: { buffer: bvhNormals } },
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
      boundPositionBuffer: bvhPositions,
      boundNormalBuffer: bvhNormals,
    };
    this.#meshes.set(id, state);
    return state;
  }
}
