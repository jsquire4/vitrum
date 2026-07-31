/**
 * PR-7 — GPU linear blend skinning into the live ReSTIR `bvhPositions` buffer,
 * then same-frame CPU BVH refit (nodes-only upload) via `applyGpuSkinnedRefit`.
 */

import type { Scene, ScenePrimitivePatch, SkinnedMeshPrimitive } from '@vitrum/core';
import {
  combineSkinMatrices,
  solveSkin,
  sparseArrayOwnIndices,
} from '@vitrum/core';
import { GPU_SKIN_BVH_WITH_NORMALS_WGSL } from './gpuSkinBvh.wgsl.js';
import type { ReSTIRBvhMode, SceneBVHBuffers } from '../restir/bvhTypes.js';
import {
  hasActiveSkinMorph,
  hasMorphControlledRenderStreams,
  solvedSkinRenderPatch,
} from './solvedSkinPatch.js';

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
  /** Exact scene-arena subrange containing packed BVH positions. */
  getGpuSkinningBvhBinding?(): GPUBufferBinding | null;
  /** WS1 — merged-BVH per-vertex normal SSBO target. The LBS compute writes
   *  inverse-transpose skinned normals here at `baseVertex+vi` so the smooth-
   *  shading-normal blend reads the deformed normal. Null before pipeline init. */
  getGpuSkinningNormalBuffer(): GPUBuffer | null;
  /** Exact scene-arena subrange containing packed BVH normals. */
  getGpuSkinningNormalBinding?(): GPUBufferBinding | null;
  /** Per-mesh vertex ranges in the merged BVH. */
  getMeshVertexRanges(): SceneBVHBuffers['meshVertexRanges'] | null;
  /** Active ReSTIR BVH layout (`merged` world positions vs `tlas` local BLAS). */
  getBvhMode(): ReSTIRBvhMode | null;
  /** Per-primitive TLAS bindings (vertex ranges in `tlas` mode). */
  getPrimitiveTlasBindings(): SceneBVHBuffers['primitiveTlasBindings'] | null;
  /** CPU-skin fallback path: push solved positions/normals through the
   *  standard incremental geometry update. */
  updatePrimitive(id: string, patch: ScenePrimitivePatch): void;
  /** GPU-skin path: refit BVH nodes after the compute pass wrote world
   *  positions directly into the live `bvhPositions` buffer. */
  applyGpuSkinnedRefit(id: string, localPositions?: Float32Array, localNormals?: Float32Array): void;
  /** Atomically publish a batch whose compute work has been encoded but not submitted. */
  applySkinningBatch(
    updates: readonly SkinningBatchUpdate[],
    skinCommands: GPUCommandBuffer | null,
  ): void;
}

export interface SkinningBatchUpdate {
  readonly id: string;
  readonly patch: ScenePrimitivePatch;
  readonly gpuWritten: boolean;
}

const UNIFORM = 0x40;
const STORAGE = 0x80;
const COPY_DST = 0x02;

/** The GPU skin kernel only applies the blended bone matrices
 *  (`combineSkinMatrices`), NOT the
 *  bindMatrix / bindMatrixInverse wrapping that the CPU `solveSkin` honours,
 *  so a primitive with a non-identity bindMatrix would skin WRONG on the GPU
 *  path. We route those to the CPU solver (same fallback shape as `hasMorph`).
 *  glTF-typical bind is identity, so the GPU fast path is preserved for the
 *  common case. */
/** Column-major identity (== `THREE.Matrix4` default). */
const IDENTITY_MAT4 = new Float32Array([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
]);

/**
 * True when the primitive carries a stored-f32 `bindMatrix` that is not exactly
 * identity. An absent or identity bindMatrix
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
    if (bm[i] !== IDENTITY_MAT4[i]) return true;
  }
  return false;
}

interface MeshGpuState {
  readonly vertexCount: number;
  readonly boneCount: number;
  /** Rest-pose and skinning inputs captured by the private storage buffers. */
  readonly sourcePositions: SkinnedMeshPrimitive['positions'];
  readonly sourceNormals: SkinnedMeshPrimitive['normals'];
  readonly sourceSkinIndices: SkinnedMeshPrimitive['skinIndices'];
  readonly sourceSkinWeights: SkinnedMeshPrimitive['skinWeights'];
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
  readonly boundPositionOffset: number;
  readonly boundPositionSize: number | undefined;
  readonly boundNormalBuffer: GPUBuffer;
  readonly boundNormalOffset: number;
  readonly boundNormalSize: number | undefined;
}

interface SkinRunSnapshot {
  readonly inputReferences: readonly unknown[];
  readonly bones: Float32Array;
  readonly boneInverses: Float32Array;
  readonly morphWeights?: Float32Array;
  readonly bindMatrix?: Float32Array;
  readonly bindMatrixInverse?: Float32Array;
  readonly matrixWorldAtBuild?: Float32Array;
  readonly skinInfluencesPerVertex: number;
  readonly bvhMode: ReSTIRBvhMode | null;
  readonly positionBuffer: GPUBuffer | null;
  readonly positionOffset: number;
  readonly positionSize: number | undefined;
  readonly normalBuffer: GPUBuffer | null;
  readonly normalOffset: number;
  readonly normalSize: number | undefined;
  readonly rangeVertexStart: number;
  readonly rangeVertexCount: number;
  readonly tlasVertexStart: number;
  readonly tlasVertexCount: number;
}

function pushTargetListReferences(
  target: unknown[],
  list: ReadonlyArray<Float32Array> | undefined,
): void {
  target.push(list);
  if (list == null) return;
  for (const stream of list) target.push(stream);
}

function pushSparseStreamReferences(
  target: unknown[],
  streams: ReadonlyArray<Float32Array | undefined> | undefined,
): void {
  target.push(streams);
  if (streams == null) return;
  for (const index of sparseArrayOwnIndices(streams)) {
    target.push(streams[index]);
  }
}

function pushSparseTargetReferences(
  target: unknown[],
  lanes:
    | SkinnedMeshPrimitive['morphTargetUvSets']
    | SkinnedMeshPrimitive['morphTargetColorSets'],
): void {
  target.push(lanes);
  if (lanes == null) return;
  for (const index of sparseArrayOwnIndices(lanes)) {
    pushTargetListReferences(target, lanes[index]);
  }
}

function skinInputReferences(
  primitive: SkinnedMeshPrimitive,
): readonly unknown[] {
  const references: unknown[] = [
    primitive.positions,
    primitive.normals,
    primitive.skinIndices,
    primitive.skinWeights,
    primitive.bones,
    primitive.boneInverses,
    primitive.bindMatrix,
    primitive.bindMatrixInverse,
    primitive.morphWeights,
    primitive.tangents,
    primitive.uvs,
    primitive.uv1,
    primitive.colors,
    primitive.transform,
  ];
  pushSparseStreamReferences(references, primitive.uvSets);
  pushSparseStreamReferences(references, primitive.colorSets);
  pushTargetListReferences(references, primitive.morphTargets);
  pushTargetListReferences(references, primitive.morphTargetNormals);
  pushTargetListReferences(references, primitive.morphTargetTangents);
  pushTargetListReferences(references, primitive.morphTargetUvs);
  pushTargetListReferences(references, primitive.morphTargetUv1s);
  pushTargetListReferences(references, primitive.morphTargetColors);
  pushSparseTargetReferences(references, primitive.morphTargetUvSets);
  pushSparseTargetReferences(references, primitive.morphTargetColorSets);
  return references;
}

function bindingOffset(binding: GPUBufferBinding | null): number {
  return Number(binding?.offset ?? 0);
}

function bindingSize(binding: GPUBufferBinding | null): number | undefined {
  return binding?.size == null ? undefined : Number(binding.size);
}

function makeSkinRunSnapshot(
  primitive: SkinnedMeshPrimitive,
  bvhMode: ReSTIRBvhMode | null,
  positionBinding: GPUBufferBinding | null,
  normalBinding: GPUBufferBinding | null,
  range: SceneBVHBuffers['meshVertexRanges'][number] | undefined,
  tlasBinding: SceneBVHBuffers['primitiveTlasBindings'][number] | undefined,
): SkinRunSnapshot {
  return {
    inputReferences: skinInputReferences(primitive),
    bones: new Float32Array(primitive.bones),
    boneInverses: new Float32Array(primitive.boneInverses),
    ...(primitive.morphWeights != null
      ? { morphWeights: new Float32Array(primitive.morphWeights) }
      : {}),
    ...(primitive.bindMatrix != null
      ? { bindMatrix: new Float32Array(primitive.bindMatrix) }
      : {}),
    ...(primitive.bindMatrixInverse != null
      ? { bindMatrixInverse: new Float32Array(primitive.bindMatrixInverse) }
      : {}),
    ...(range != null
      ? { matrixWorldAtBuild: new Float32Array(range.matrixWorldAtBuild) }
      : {}),
    skinInfluencesPerVertex: primitive.skinInfluencesPerVertex ?? 4,
    bvhMode,
    positionBuffer: positionBinding?.buffer ?? null,
    positionOffset: bindingOffset(positionBinding),
    positionSize: bindingSize(positionBinding),
    normalBuffer: normalBinding?.buffer ?? null,
    normalOffset: bindingOffset(normalBinding),
    normalSize: bindingSize(normalBinding),
    rangeVertexStart: range?.vertexStart ?? -1,
    rangeVertexCount: range?.vertexCount ?? -1,
    tlasVertexStart: tlasBinding?.vertexStart ?? -1,
    tlasVertexCount: tlasBinding?.vertexCount ?? -1,
  };
}

function exactFloat32Equal(
  a: Float32Array | undefined,
  b: Float32Array | undefined,
): boolean {
  if (a === b) return true;
  if (a == null || b == null || a.length !== b.length) return false;
  for (let index = 0; index < a.length; index += 1) {
    if (!Object.is(a[index], b[index])) return false;
  }
  return true;
}

function skinRunSnapshotEqual(
  a: SkinRunSnapshot | undefined,
  b: SkinRunSnapshot,
): boolean {
  if (a == null || a.inputReferences.length !== b.inputReferences.length) {
    return false;
  }
  for (let index = 0; index < a.inputReferences.length; index += 1) {
    if (a.inputReferences[index] !== b.inputReferences[index]) return false;
  }
  return (
    exactFloat32Equal(a.bones, b.bones) &&
    exactFloat32Equal(a.boneInverses, b.boneInverses) &&
    exactFloat32Equal(a.morphWeights, b.morphWeights) &&
    exactFloat32Equal(a.bindMatrix, b.bindMatrix) &&
    exactFloat32Equal(a.bindMatrixInverse, b.bindMatrixInverse) &&
    exactFloat32Equal(a.matrixWorldAtBuild, b.matrixWorldAtBuild) &&
    a.skinInfluencesPerVertex === b.skinInfluencesPerVertex &&
    a.bvhMode === b.bvhMode &&
    a.positionBuffer === b.positionBuffer &&
    a.positionOffset === b.positionOffset &&
    a.positionSize === b.positionSize &&
    a.normalBuffer === b.normalBuffer &&
    a.normalOffset === b.normalOffset &&
    a.normalSize === b.normalSize &&
    a.rangeVertexStart === b.rangeVertexStart &&
    a.rangeVertexCount === b.rangeVertexCount &&
    a.tlasVertexStart === b.tlasVertexStart &&
    a.tlasVertexCount === b.tlasVertexCount
  );
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

function destroyBuffersBestEffort(buffers: readonly GPUBuffer[]): void {
  for (const buffer of buffers) {
    try { buffer.destroy(); } catch { /* continue retiring remaining owned buffers */ }
  }
}

function destroyMeshState(state: MeshGpuState): void {
  destroyBuffersBestEffort([
    state.uniformBuffer,
    state.boneBuffer,
    state.restPosBuffer,
    state.restNormBuffer,
    state.skinIdxBuffer,
    state.skinWeightBuffer,
  ]);
  // boundPositionBuffer / boundNormalBuffer are SHARED merged buffers owned by
  // BvhBufferHost — do NOT destroy them here.
}

export class GpuSkinningSubsystem {
  readonly #device: GPUDevice;
  readonly #preferGpu: boolean;
  readonly #meshes = new Map<string, MeshGpuState>();
  readonly #lastAcceptedRuns = new Map<string, SkinRunSnapshot>();
  /** Primitives whose last accepted frame published non-zero UV/color morphs.
   *  The first zero-weight frame must restore their authored base streams. */
  readonly #activeRenderStreamMorphs = new Set<string>();
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
    this.#lastAcceptedRuns.clear();
    this.#activeRenderStreamMorphs.clear();
    this.#bvhPipeline = null;
  }

  run(host: GpuSkinningHost, scene: Scene): void {
    const bvhPositions = host.getGpuSkinningBvhBuffer();
    const bvhNormals = host.getGpuSkinningNormalBuffer();
    const bvhPositionBinding =
      host.getGpuSkinningBvhBinding?.() ??
      (bvhPositions == null ? null : { buffer: bvhPositions });
    const bvhNormalBinding =
      host.getGpuSkinningNormalBinding?.() ??
      (bvhNormals == null ? null : { buffer: bvhNormals });
    const meshVertexRanges = host.getMeshVertexRanges();
    const bvhMode = host.getBvhMode();
    const tlasBindings = host.getPrimitiveTlasBindings();
    const skinned = scene.primitives.filter(
      (primitive): primitive is SkinnedMeshPrimitive => primitive.kind === 'skinned-mesh',
    );
    const liveIds = new Set(skinned.map((primitive) => String(primitive.id)));
    for (const [id, state] of this.#meshes) {
      if (liveIds.has(id)) continue;
      destroyMeshState(state);
      this.#meshes.delete(id);
    }
    for (const id of this.#lastAcceptedRuns.keys()) {
      if (!liveIds.has(id)) this.#lastAcceptedRuns.delete(id);
    }
    if (skinned.length === 0) {
      this.#activeRenderStreamMorphs.clear();
      return;
    }
    for (const id of this.#activeRenderStreamMorphs) {
      if (!liveIds.has(id)) this.#activeRenderStreamMorphs.delete(id);
    }

    const activeRenderStreamMorphs = new Map<string, boolean>();
    const restoreAuthoredRenderStreams = new Map<string, boolean>();
    for (const primitive of skinned) {
      const id = String(primitive.id);
      const active =
        hasActiveSkinMorph(primitive) &&
        hasMorphControlledRenderStreams(primitive);
      activeRenderStreamMorphs.set(id, active);
      restoreAuthoredRenderStreams.set(
        id,
        !active && this.#activeRenderStreamMorphs.has(id),
      );
    }
    const acceptRenderStreamMorphState = (): void => {
      for (const [id, active] of activeRenderStreamMorphs) {
        if (active) this.#activeRenderStreamMorphs.add(id);
        else this.#activeRenderStreamMorphs.delete(id);
      }
    };
    const solvedPatch = (
      primitive: SkinnedMeshPrimitive,
    ): ScenePrimitivePatch => solvedSkinRenderPatch(
      primitive,
      solveSkin(primitive),
      restoreAuthoredRenderStreams.get(String(primitive.id)) === true,
    ) as ScenePrimitivePatch;
    const fallback = new Map<string, ScenePrimitivePatch>();
    const pendingSnapshots = new Map<string, SkinRunSnapshot>();
    const dirtySkinned: SkinnedMeshPrimitive[] = [];
    const gpuJobs: Array<{
      readonly primitive: SkinnedMeshPrimitive;
      readonly id: string;
      readonly range: SceneBVHBuffers['meshVertexRanges'][number];
      readonly baseVertex: number;
    }> = [];

    for (const primitive of skinned) {
      const id = String(primitive.id);
      const hasMorph = hasActiveSkinMorph(primitive);
      const restoresRenderStreams =
        restoreAuthoredRenderStreams.get(id) === true;
      const range = meshVertexRanges?.find((candidate) => candidate.name === id);
      let baseVertex = range?.vertexStart ?? 0;
      const tlasBinding =
        bvhMode === 'tlas'
          ? tlasBindings?.find((binding) => binding.primitiveId === id)
          : null;
      if (tlasBinding != null) baseVertex = tlasBinding.vertexStart;
      const snapshot = makeSkinRunSnapshot(
        primitive,
        bvhMode,
        bvhPositionBinding,
        bvhNormalBinding,
        range,
        tlasBinding ?? undefined,
      );
      pendingSnapshots.set(id, snapshot);
      if (
        !restoresRenderStreams &&
        skinRunSnapshotEqual(this.#lastAcceptedRuns.get(id), snapshot)
      ) {
        continue;
      }
      dirtySkinned.push(primitive);
      const canUseGpu =
        bvhPositionBinding != null &&
        bvhNormalBinding != null &&
        meshVertexRanges != null &&
        this.#preferGpu &&
        !hasMorph &&
        !restoresRenderStreams &&
        primitive.tangents == null &&
        (primitive.skinInfluencesPerVertex ?? 4) === 4 &&
        !hasNonIdentityBind(primitive) &&
        typeof this.#device.createComputePipeline === 'function' &&
        range != null &&
        range.vertexCount > 0 &&
        (bvhMode !== 'tlas' || (tlasBinding != null && tlasBinding.vertexCount > 0));
      if (!canUseGpu) {
        fallback.set(id, solvedPatch(primitive));
      } else {
        gpuJobs.push({ primitive, id, range, baseVertex });
      }
    }

    if (dirtySkinned.length === 0) return;

    const fallbackNeedsTopology = [...fallback.values()].some(
      (patch) =>
        patch.tangents != null ||
        patch.uvs != null ||
        patch.uv1 != null ||
        patch.uvSets != null ||
        patch.colors != null ||
        patch.colorSets != null,
    );
    if (fallbackNeedsTopology) {
      // A topology candidate replaces the live BVH buffers, so commands encoded
      // against their old identities cannot join the transaction. Solve every
      // skinned primitive on CPU and publish one full candidate instead.
      host.applySkinningBatch(
        skinned.map((primitive) => ({
          id: String(primitive.id),
          patch: solvedPatch(primitive),
          gpuWritten: false,
        })),
        null,
      );
      for (const primitive of skinned) {
        const id = String(primitive.id);
        this.#lastAcceptedRuns.set(id, pendingSnapshots.get(id)!);
      }
      acceptRenderStreamMorphState();
      return;
    }

    let skinCommands: GPUCommandBuffer | null = null;
    const gpuPatches = new Map<string, ScenePrimitivePatch>();
    if (gpuJobs.length > 0) {
      const encoder = this.#device.createCommandEncoder({ label: 'vitrum.gpuSkinBvh' });
      for (const job of gpuJobs) {
        const state = this.#ensureMesh(
          job.primitive,
          bvhPositionBinding!,
          bvhNormalBinding!,
        );
        const combined = combineSkinMatrices(
          job.primitive.bones,
          job.primitive.boneInverses,
          state.boneCount,
        );
        this.#device.queue.writeBuffer(state.boneBuffer, 0, new Float32Array(combined));

        const uniformBytes = new ArrayBuffer(80);
        const u32 = new Uint32Array(uniformBytes);
        u32[0] = state.vertexCount;
        u32[1] = job.baseVertex;
        u32[2] = bvhMode === 'tlas' ? 0 : 1;
        new Float32Array(uniformBytes).set(job.range.matrixWorldAtBuild, 4);
        this.#device.queue.writeBuffer(state.uniformBuffer, 0, uniformBytes);

        const pass = encoder.beginComputePass({
          label: `vitrum.gpuSkinBvh.${job.id}`,
        });
        pass.setPipeline(state.pipeline);
        pass.setBindGroup(0, state.bindGroup);
        pass.dispatchWorkgroups(Math.ceil(state.vertexCount / 64), 1, 1);
        pass.end();
        gpuPatches.set(job.id, solvedPatch(job.primitive));
      }
      skinCommands = encoder.finish();
    }

    const updates: SkinningBatchUpdate[] = dirtySkinned.map((primitive) => {
      const id = String(primitive.id);
      const gpuPatch = gpuPatches.get(id);
      return {
        id,
        patch: gpuPatch ?? fallback.get(id)!,
        gpuWritten: gpuPatch != null,
      };
    });
    host.applySkinningBatch(updates, skinCommands);
    for (const primitive of dirtySkinned) {
      const id = String(primitive.id);
      this.#lastAcceptedRuns.set(id, pendingSnapshots.get(id)!);
    }
    acceptRenderStreamMorphState();
  }

  #ensureMesh(
    prim: SkinnedMeshPrimitive,
    bvhPositionBinding: GPUBufferBinding,
    bvhNormalBinding: GPUBufferBinding,
  ): MeshGpuState {
    const bvhPositions = bvhPositionBinding.buffer;
    const bvhNormals = bvhNormalBinding.buffer;
    const bvhPositionOffset = Number(bvhPositionBinding.offset ?? 0);
    const bvhPositionSize =
      bvhPositionBinding.size == null ? undefined : Number(bvhPositionBinding.size);
    const bvhNormalOffset = Number(bvhNormalBinding.offset ?? 0);
    const bvhNormalSize =
      bvhNormalBinding.size == null ? undefined : Number(bvhNormalBinding.size);
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
      existing.sourcePositions === prim.positions &&
      existing.sourceNormals === prim.normals &&
      existing.sourceSkinIndices === prim.skinIndices &&
      existing.sourceSkinWeights === prim.skinWeights &&
      existing.boundPositionBuffer === bvhPositions &&
      existing.boundPositionOffset === bvhPositionOffset &&
      existing.boundPositionSize === bvhPositionSize &&
      existing.boundNormalBuffer === bvhNormals &&
      existing.boundNormalOffset === bvhNormalOffset &&
      existing.boundNormalSize === bvhNormalSize
    ) {
      return existing;
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

    const forbiddenBuffers = new Set<GPUBuffer>([bvhPositions, bvhNormals]);
    for (const cached of this.#meshes.values()) {
      forbiddenBuffers.add(cached.uniformBuffer);
      forbiddenBuffers.add(cached.boneBuffer);
      forbiddenBuffers.add(cached.restPosBuffer);
      forbiddenBuffers.add(cached.restNormBuffer);
      forbiddenBuffers.add(cached.skinIdxBuffer);
      forbiddenBuffers.add(cached.skinWeightBuffer);
    }
    const candidateBuffers: GPUBuffer[] = [];
    const candidateSet = new Set<GPUBuffer>();
    const registerCandidate = (buffer: GPUBuffer): GPUBuffer => {
      if (forbiddenBuffers.has(buffer)) {
        throw new Error(`GpuSkinningSubsystem: device returned a live/shared buffer alias for ${id}`);
      }
      if (candidateSet.has(buffer)) {
        throw new Error(`GpuSkinningSubsystem: device returned a duplicate candidate buffer for ${id}`);
      }
      candidateSet.add(buffer);
      candidateBuffers.push(buffer);
      return buffer;
    };
    try {
      const createStorage = (label: string, size: number): GPUBuffer =>
        registerCandidate(mkStorage(label, size));
      const restPosBuffer = createStorage(`vitrum.gpuSkinBvh.${id}.restPos`, vertBytes);
      const restNormBuffer = createStorage(`vitrum.gpuSkinBvh.${id}.restNorm`, vertBytes);
      const skinIdxBuffer = createStorage(`vitrum.gpuSkinBvh.${id}.skinIdx`, skinIdx.byteLength);
      const skinWeightBuffer = createStorage(`vitrum.gpuSkinBvh.${id}.skinW`, skinW.byteLength);
      const boneBuffer = createStorage(`vitrum.gpuSkinBvh.${id}.bones`, boneBytes);
      const uniformBuffer = registerCandidate(device.createBuffer({
        label: `vitrum.gpuSkinBvh.${id}.uniform`,
        size: 80,
        usage: UNIFORM | COPY_DST,
      }));

      device.queue.writeBuffer(restPosBuffer, 0, new Float32Array(restPos));
      device.queue.writeBuffer(restNormBuffer, 0, new Float32Array(restNorm));
      device.queue.writeBuffer(skinIdxBuffer, 0, new Uint32Array(skinIdx));
      device.queue.writeBuffer(skinWeightBuffer, 0, new Float32Array(skinW));

      let pipeline = this.#bvhPipeline;
      const createdPipeline = pipeline == null;
      if (pipeline == null) {
        const module = device.createShaderModule({
          label: 'vitrum.gpuSkinBvh.module',
          code: GPU_SKIN_BVH_WITH_NORMALS_WGSL,
        });
        pipeline = device.createComputePipeline({
          label: 'vitrum.gpuSkinBvh.pipeline',
          layout: 'auto',
          compute: { module, entryPoint: 'main' },
        });
      }
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
          { binding: 6, resource: bvhPositionBinding },
          { binding: 7, resource: bvhNormalBinding },
        ],
      });
      const state: MeshGpuState = {
        vertexCount: vertCount,
        boneCount,
        sourcePositions: prim.positions,
        sourceNormals: prim.normals,
        sourceSkinIndices: prim.skinIndices,
        sourceSkinWeights: prim.skinWeights,
        uniformBuffer,
        boneBuffer,
        restPosBuffer,
        restNormBuffer,
        skinIdxBuffer,
        skinWeightBuffer,
        bindGroup,
        pipeline,
        boundPositionBuffer: bvhPositions,
        boundPositionOffset: bvhPositionOffset,
        boundPositionSize: bvhPositionSize,
        boundNormalBuffer: bvhNormals,
        boundNormalOffset: bvhNormalOffset,
        boundNormalSize: bvhNormalSize,
      };
      // Publish only after the complete candidate succeeded.
      this.#meshes.set(id, state);
      if (createdPipeline) this.#bvhPipeline = pipeline;
      if (existing != null) destroyMeshState(existing);
      return state;
    } catch (error) {
      destroyBuffersBestEffort(candidateBuffers);
      throw error;
    }
  }
}
