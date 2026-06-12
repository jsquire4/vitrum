// sceneController.ts — glTF animation runtime bridge for Vitrum engines.
//
// The importer returns a flattened @vitrum/core Scene plus AnimationClip data.
// This controller keeps the original glTF node hierarchy around, evaluates a
// clip at time t, recomputes world transforms / skeleton bones, and pushes the
// resulting primitive patches into any Engine-like target.

import {
  asMat4,
  patchPrimitiveInScene,
  sampleAnimationClip,
  solveSkin,
  type AnimationClip,
  type Mat4,
  type SampledChannel,
  type Scene,
  type ScenePrimitive,
  type SkinnedMeshPrimitive,
} from '@vitrum/core';
import type { GltfJson, GltfNode } from './gltfTypes.js';
import type { GltfToSceneResult } from './gltfToScene.js';
import {
  IDENTITY_MAT4,
  composeTrsMat4,
  mat4Mul,
  nodeLocalMatrix,
} from './transforms.js';

export interface GltfScenePatchTarget {
  setScene(scene: Scene): void;
  updatePrimitive?(id: string, patch: Partial<ScenePrimitive>): void;
}

export interface GltfSceneControllerInput extends GltfToSceneResult {
  readonly gltf: GltfJson;
  readonly sceneIndex?: number;
}

export interface GltfSceneControllerOptions {
  readonly engine?: GltfScenePatchTarget;
  readonly setSceneOnAttach?: boolean;
}

export type GltfClipSelector = number | string | AnimationClip;

export interface GltfApplyAnimationOptions {
  readonly engine?: GltfScenePatchTarget;
  readonly loop?: boolean;
  readonly forceSetScene?: boolean;
}

export interface GltfPrimitivePatchRecord {
  readonly id: string;
  readonly patch: Partial<ScenePrimitive>;
}

export interface GltfAnimationApplyResult {
  readonly clip: AnimationClip;
  readonly requestedTime: number;
  readonly localTime: number;
  readonly primitivePatches: readonly GltfPrimitivePatchRecord[];
  readonly scene: Scene;
  readonly warnings: readonly string[];
  readonly usedSetScene: boolean;
}

interface NodeLocalState {
  translation: [number, number, number];
  rotation: [number, number, number, number];
  scale: [number, number, number];
  matrix?: Float32Array;
  matrixOverridden: boolean;
}

interface SkinBinding {
  meshNodeIndex: number;
  jointNodeIndices: readonly number[];
}

const NODE_ID_PREFIX = 'gltf-node-';

export function createGltfSceneController(
  input: GltfSceneControllerInput,
  options: GltfSceneControllerOptions = {},
): GltfSceneController {
  return new GltfSceneController(input, options);
}

export class GltfSceneController {
  readonly gltf: GltfJson;
  readonly animations: readonly AnimationClip[];
  readonly animationTargets: Readonly<Record<string, readonly string[]>>;

  #scene: Scene;
  #engine: GltfScenePatchTarget | undefined;
  #activeClip: AnimationClip | undefined;
  #clock = 0;
  readonly #sceneIndex: number;
  readonly #baseLocals: readonly NodeLocalState[];
  readonly #nodeToPrimitiveIds: ReadonlyMap<number, readonly string[]>;
  readonly #basePrimitiveById: ReadonlyMap<string, ScenePrimitive>;
  readonly #skinBindingsByPrimitiveId: ReadonlyMap<string, SkinBinding>;
  readonly #warnings: string[] = [];
  readonly #warnedMatrixOverrideNodes = new Set<number>();

  constructor(input: GltfSceneControllerInput, options: GltfSceneControllerOptions = {}) {
    this.gltf = input.gltf;
    this.animations = input.animations;
    this.animationTargets = input.animationTargets;
    this.#scene = input.scene;
    this.#sceneIndex = input.sceneIndex ?? input.gltf.scene ?? 0;
    this.#baseLocals = (input.gltf.nodes ?? []).map(baseLocalState);
    this.#nodeToPrimitiveIds = parseAnimationTargets(input.animationTargets);
    this.#basePrimitiveById = new Map(input.scene.primitives.map((p) => [String(p.id), p]));
    this.#skinBindingsByPrimitiveId = buildSkinBindings(input.gltf, this.#nodeToPrimitiveIds);
    this.#engine = options.engine;
    if (options.engine && (options.setSceneOnAttach ?? true)) {
      options.engine.setScene(this.#scene);
    }
  }

  get scene(): Scene {
    return this.#scene;
  }

  get warnings(): readonly string[] {
    return this.#warnings;
  }

  attachEngine(engine: GltfScenePatchTarget, options: { readonly setScene?: boolean } = {}): void {
    this.#engine = engine;
    if (options.setScene ?? true) {
      engine.setScene(this.#scene);
    }
  }

  setActiveClip(selector: GltfClipSelector, time = 0): AnimationClip {
    const clip = this.#resolveClip(selector);
    this.#activeClip = clip;
    this.#clock = time;
    return clip;
  }

  seek(time: number, options: GltfApplyAnimationOptions = {}): GltfAnimationApplyResult {
    if (!this.#activeClip) {
      if (this.animations.length === 0) {
        throw new Error('[vitrum/gltf-adapter] GltfSceneController.seek: asset has no animations.');
      }
      this.#activeClip = this.animations[0]!;
    }
    this.#clock = time;
    return this.applyAnimation(this.#activeClip, time, options);
  }

  advance(deltaSeconds: number, options: GltfApplyAnimationOptions = {}): GltfAnimationApplyResult {
    return this.seek(this.#clock + deltaSeconds, { loop: options.loop ?? true, ...options });
  }

  applyAnimation(
    selector: GltfClipSelector,
    time: number,
    options: GltfApplyAnimationOptions = {},
  ): GltfAnimationApplyResult {
    const clip = this.#resolveClip(selector);
    const localTime = normalizeClipTime(clip, time, options.loop ?? false);
    const samples = sampleAnimationClip(clip, localTime);
    const locals = cloneLocalStates(this.#baseLocals);
    const morphWeightsByNode = new Map<number, Float32Array>();
    const frameWarnings: string[] = [];

    for (const sample of samples) {
      this.#applySampleToLocals(sample, locals, morphWeightsByNode, frameWarnings);
    }

    const worldTransforms = buildWorldTransformsForLocals(this.gltf, this.#rootNodes(), locals);
    const patchMap = new Map<string, Partial<ScenePrimitive>>();

    for (const [nodeIndex, primitiveIds] of this.#nodeToPrimitiveIds) {
      const world = worldTransforms.get(nodeIndex);
      if (!world) continue;
      for (const id of primitiveIds) {
        const current = findPrimitive(this.#scene, id);
        if (!current) continue;
        if (!mat4AlmostEqual(primitiveTransform(current), world)) {
          mergePrimitivePatch(patchMap, id, { transform: world } as Partial<ScenePrimitive>);
        }
      }
    }

    for (const [nodeIndex, weights] of morphWeightsByNode) {
      const primitiveIds = this.#nodeToPrimitiveIds.get(nodeIndex) ?? [];
      for (const id of primitiveIds) {
        const patch = this.#buildMorphPatch(id, weights, frameWarnings);
        if (patch) mergePrimitivePatch(patchMap, id, patch);
      }
    }

    for (const [id, binding] of this.#skinBindingsByPrimitiveId) {
      const patch = this.#buildSkinPatch(id, binding, worldTransforms, patchMap.get(id), frameWarnings);
      if (patch) mergePrimitivePatch(patchMap, id, patch);
    }

    const primitivePatches = Array.from(patchMap, ([id, patch]) => ({ id, patch }));
    const target = options.engine ?? this.#engine;
    let nextScene = this.#scene;
    for (const { id, patch } of primitivePatches) {
      nextScene = patchPrimitiveInScene(nextScene, id, patch);
    }

    let usedSetScene = false;
    if (target && primitivePatches.length > 0) {
      if (!options.forceSetScene && target.updatePrimitive) {
        for (const { id, patch } of primitivePatches) {
          target.updatePrimitive(id, patch);
        }
      } else {
        target.setScene(nextScene);
        usedSetScene = true;
      }
    }

    this.#scene = nextScene;
    if (frameWarnings.length > 0) this.#warnings.push(...frameWarnings);

    return {
      clip,
      requestedTime: time,
      localTime,
      primitivePatches,
      scene: this.#scene,
      warnings: frameWarnings,
      usedSetScene,
    };
  }

  resetPose(options: { readonly engine?: GltfScenePatchTarget; readonly forceSetScene?: boolean } = {}): void {
    const target = options.engine ?? this.#engine;
    this.#scene = {
      ...this.#scene,
      primitives: this.#scene.primitives.map((primitive) =>
        this.#basePrimitiveById.get(String(primitive.id)) ?? primitive,
      ),
    };
    if (target) target.setScene(this.#scene);
  }

  #resolveClip(selector: GltfClipSelector): AnimationClip {
    if (typeof selector === 'number') {
      const clip = this.animations[selector];
      if (!clip) throw new Error(`[vitrum/gltf-adapter] Animation clip index ${selector} not found.`);
      return clip;
    }
    if (typeof selector === 'string') {
      const clip = this.animations.find((candidate) => candidate.name === selector);
      if (!clip) throw new Error(`[vitrum/gltf-adapter] Animation clip "${selector}" not found.`);
      return clip;
    }
    return selector;
  }

  #rootNodes(): readonly number[] {
    return this.gltf.scenes?.[this.#sceneIndex]?.nodes ?? [];
  }

  #applySampleToLocals(
    sample: SampledChannel,
    locals: NodeLocalState[],
    morphWeightsByNode: Map<number, Float32Array>,
    warnings: string[],
  ): void {
    const nodeIndex = parseAnimationNodeIndex(sample.node);
    if (nodeIndex == null || !locals[nodeIndex]) {
      warnings.push(
        `[vitrum/gltf-adapter] Animation target "${sample.node}" does not map to a glTF node; channel skipped.`,
      );
      return;
    }
    if (sample.path === 'weights') {
      morphWeightsByNode.set(nodeIndex, new Float32Array(sample.value));
      return;
    }

    const state = locals[nodeIndex]!;
    if (state.matrix && !this.#warnedMatrixOverrideNodes.has(nodeIndex)) {
      this.#warnedMatrixOverrideNodes.add(nodeIndex);
      warnings.push(
        `[vitrum/gltf-adapter] Animation targets node ${nodeIndex}, which imported from a matrix. ` +
          'The controller evaluates animated TRS channels over the matrix for this frame.',
      );
    }
    state.matrixOverridden = true;
    if (sample.path === 'translation') {
      state.translation = readVec3(sample.value, [0, 0, 0]);
    } else if (sample.path === 'rotation') {
      state.rotation = normalizeQuat(readQuat(sample.value, [0, 0, 0, 1]));
    } else if (sample.path === 'scale') {
      state.scale = readVec3(sample.value, [1, 1, 1]);
    }
  }

  #buildMorphPatch(
    primitiveId: string,
    weights: Float32Array,
    warnings: string[],
  ): Partial<ScenePrimitive> | undefined {
    const base = this.#basePrimitiveById.get(primitiveId);
    if (!isSkinnedMesh(base) || !base.morphTargets || base.morphTargets.length === 0) {
      warnings.push(
        `[vitrum/gltf-adapter] Animation weights target resolved to primitive "${primitiveId}", ` +
          'but that primitive has no morph targets; channel skipped.',
      );
      return undefined;
    }
    const morphWeights = fitMorphWeights(weights, base.morphTargets.length, primitiveId, warnings);
    const solved = solveSkin({ ...base, morphWeights });
    return {
      morphWeights,
      positions: solved.positions,
      normals: solved.normals,
    } as Partial<ScenePrimitive>;
  }

  #buildSkinPatch(
    primitiveId: string,
    binding: SkinBinding,
    worldTransforms: ReadonlyMap<number, Mat4>,
    existingPatch: Partial<ScenePrimitive> | undefined,
    warnings: string[],
  ): Partial<ScenePrimitive> | undefined {
    const base = this.#basePrimitiveById.get(primitiveId);
    if (!isSkinnedMesh(base)) return undefined;

    const bones = new Float32Array(binding.jointNodeIndices.length * 16);
    for (const [jointOffset, jointNodeIndex] of binding.jointNodeIndices.entries()) {
      const jointWorld = worldTransforms.get(jointNodeIndex);
      const outOffset = jointOffset * 16;
      if (jointWorld) {
        bones.set(jointWorld, outOffset);
      } else {
        bones.set(IDENTITY_MAT4, outOffset);
        warnings.push(
          `[vitrum/gltf-adapter] Skin for primitive "${primitiveId}" references joint node ` +
            `${jointNodeIndex}, which is not reachable from the imported scene; identity bone used.`,
        );
      }
    }

    const patchedMorphWeights = patchMorphWeights(existingPatch);
    if (matArrayAlmostEqual(base.bones, bones) && !patchedMorphWeights) {
      return undefined;
    }

    const currentPrimitive = findPrimitive(this.#scene, primitiveId);
    const currentMorphWeights =
      patchedMorphWeights
        ?? (isSkinnedMesh(currentPrimitive) ? currentPrimitive.morphWeights : undefined)
        ?? base.morphWeights;
    const solved = solveSkin({
      ...base,
      bones,
      ...(currentMorphWeights ? { morphWeights: currentMorphWeights } : {}),
    });
    return {
      bones,
      positions: solved.positions,
      normals: solved.normals,
    } as Partial<ScenePrimitive>;
  }
}

function baseLocalState(node: GltfNode): NodeLocalState {
  const state: NodeLocalState = {
    translation: node.translation ? [...node.translation] : [0, 0, 0],
    rotation: node.rotation ? [...node.rotation] : [0, 0, 0, 1],
    scale: node.scale ? [...node.scale] : [1, 1, 1],
    matrixOverridden: false,
  };
  if (node.matrix) state.matrix = new Float32Array(node.matrix);
  return state;
}

function cloneLocalStates(states: readonly NodeLocalState[]): NodeLocalState[] {
  return states.map((state) => ({
    translation: [...state.translation],
    rotation: [...state.rotation],
    scale: [...state.scale],
    ...(state.matrix ? { matrix: new Float32Array(state.matrix) } : {}),
    matrixOverridden: false,
  }));
}

function localMatrixForState(state: NodeLocalState | undefined, node: GltfNode | undefined): Float32Array {
  if (!state) return node ? nodeLocalMatrix(node) : new Float32Array(IDENTITY_MAT4);
  if (state.matrix && !state.matrixOverridden) return new Float32Array(state.matrix);
  return composeTrsMat4(state.translation, state.rotation, state.scale);
}

function buildWorldTransformsForLocals(
  gltf: GltfJson,
  rootNodeIndices: readonly number[],
  locals: readonly NodeLocalState[],
): Map<number, Mat4> {
  const result = new Map<number, Mat4>();
  const nodes = gltf.nodes ?? [];
  const stack: Array<{ nodeIdx: number; parentWorld: Float32Array }> = rootNodeIndices.map(
    (nodeIdx) => ({ nodeIdx, parentWorld: new Float32Array(IDENTITY_MAT4) }),
  );

  while (stack.length > 0) {
    const { nodeIdx, parentWorld } = stack.pop()!;
    const node = nodes[nodeIdx];
    if (!node || result.has(nodeIdx)) continue;
    const local = localMatrixForState(locals[nodeIdx], node);
    const world = mat4Mul(parentWorld, local);
    result.set(nodeIdx, asMat4(world));
    for (const childIdx of node.children ?? []) {
      stack.push({ nodeIdx: childIdx, parentWorld: world });
    }
  }
  return result;
}

function parseAnimationTargets(
  targets: Readonly<Record<string, readonly string[]>>,
): Map<number, readonly string[]> {
  const out = new Map<number, readonly string[]>();
  for (const [nodeId, primitiveIds] of Object.entries(targets)) {
    const nodeIndex = parseAnimationNodeIndex(nodeId);
    if (nodeIndex != null) out.set(nodeIndex, primitiveIds);
  }
  return out;
}

function buildSkinBindings(
  gltf: GltfJson,
  nodeToPrimitiveIds: ReadonlyMap<number, readonly string[]>,
): Map<string, SkinBinding> {
  const out = new Map<string, SkinBinding>();
  for (const [nodeIndex, primitiveIds] of nodeToPrimitiveIds) {
    const node = gltf.nodes?.[nodeIndex];
    if (!node || node.skin === undefined) continue;
    const skin = gltf.skins?.[node.skin];
    if (!skin) continue;
    for (const primitiveId of primitiveIds) {
      out.set(primitiveId, {
        meshNodeIndex: nodeIndex,
        jointNodeIndices: skin.joints,
      });
    }
  }
  return out;
}

function parseAnimationNodeIndex(nodeId: string): number | undefined {
  if (!nodeId.startsWith(NODE_ID_PREFIX)) return undefined;
  const raw = nodeId.slice(NODE_ID_PREFIX.length);
  if (!/^\d+$/.test(raw)) return undefined;
  return Number(raw);
}

function normalizeClipTime(clip: AnimationClip, time: number, loop: boolean): number {
  if (!Number.isFinite(time)) return 0;
  const duration = clip.duration;
  if (duration <= 0 || !Number.isFinite(duration)) return 0;
  if (loop) return ((time % duration) + duration) % duration;
  return Math.max(0, Math.min(duration, time));
}

function readVec3(value: Float32Array, fallback: [number, number, number]): [number, number, number] {
  return [
    value[0] ?? fallback[0],
    value[1] ?? fallback[1],
    value[2] ?? fallback[2],
  ];
}

function readQuat(
  value: Float32Array,
  fallback: [number, number, number, number],
): [number, number, number, number] {
  return [
    value[0] ?? fallback[0],
    value[1] ?? fallback[1],
    value[2] ?? fallback[2],
    value[3] ?? fallback[3],
  ];
}

function normalizeQuat(q: [number, number, number, number]): [number, number, number, number] {
  const len = Math.hypot(q[0], q[1], q[2], q[3]);
  if (!Number.isFinite(len) || len < 1e-8) return [0, 0, 0, 1];
  return [q[0] / len, q[1] / len, q[2] / len, q[3] / len];
}

function mergePrimitivePatch(
  map: Map<string, Partial<ScenePrimitive>>,
  id: string,
  patch: Partial<ScenePrimitive>,
): void {
  map.set(id, { ...(map.get(id) ?? {}), ...patch } as Partial<ScenePrimitive>);
}

function findPrimitive(scene: Scene, id: string): ScenePrimitive | undefined {
  return scene.primitives.find((primitive) => String(primitive.id) === id);
}

function primitiveTransform(primitive: ScenePrimitive): Mat4 | undefined {
  if (primitive.kind === 'instanced-mesh') return undefined;
  return primitive.transform;
}

function patchMorphWeights(patch: Partial<ScenePrimitive> | undefined): Float32Array | undefined {
  const value = (patch as { readonly morphWeights?: unknown } | undefined)?.morphWeights;
  return value instanceof Float32Array ? value : undefined;
}

function mat4AlmostEqual(a: ArrayLike<number> | undefined, b: ArrayLike<number>, eps = 1e-5): boolean {
  const aa = a ?? IDENTITY_MAT4;
  return matArrayAlmostEqual(aa, b, eps);
}

function matArrayAlmostEqual(a: ArrayLike<number>, b: ArrayLike<number>, eps = 1e-5): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) {
    if (Math.abs((a[i] ?? 0) - (b[i] ?? 0)) > eps) return false;
  }
  return true;
}

function isSkinnedMesh(primitive: ScenePrimitive | undefined): primitive is SkinnedMeshPrimitive {
  return primitive?.kind === 'skinned-mesh';
}

function fitMorphWeights(
  weights: Float32Array,
  targetCount: number,
  primitiveId: string,
  warnings: string[],
): Float32Array {
  if (weights.length === targetCount) return new Float32Array(weights);
  warnings.push(
    `[vitrum/gltf-adapter] Animation weights for primitive "${primitiveId}" have length ` +
      `${weights.length}, expected ${targetCount}; extra entries are dropped and missing entries default to 0.`,
  );
  const out = new Float32Array(targetCount);
  for (let i = 0; i < targetCount; i += 1) out[i] = weights[i] ?? 0;
  return out;
}
