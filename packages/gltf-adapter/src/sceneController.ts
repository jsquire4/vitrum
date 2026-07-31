// sceneController.ts — glTF animation runtime bridge for Vitrum engines.
//
// The importer returns a flattened @vitrum/core Scene plus AnimationClip data.
// This controller keeps the original glTF node hierarchy around, evaluates a
// clip at time t, recomputes world transforms / skeleton bones, and pushes the
// resulting primitive patches into any Engine-like target.

import {
  asMat4,
  createAnimationClipSampler,
  patchPrimitiveInScene,
  solveSkin,
  validateScene,
  type AnimationClip,
  type AnimationClipSampler,
  type InstancedMeshPrimitive,
  type MaterialSpec,
  type MaterialSpecPatch,
  type Mat4,
  type SampledChannel,
  type Scene,
  type SceneEmitter,
  type ScenePrimitive,
  type ScenePrimitivePatch,
  type SkinnedMeshPrimitive,
  type SurfaceAbsorptionLayerPatch,
  type TextureRef,
} from '@vitrum/core';
import type { GltfJson, GltfNode, GltfPrimitive } from './gltfTypes.js';
import type {
  GltfInstancingBinding,
  GltfMaterialBinding,
  GltfMaterialVariantBinding,
  GltfMaterialVariantPrimitivePatch,
  GltfPunctualEmitterBinding,
  GltfSceneCamera,
  GltfToSceneResult,
} from './gltfToScene.js';
import { GLTF_DEFAULT_MATERIAL } from './materials.js';
import { collectSceneCameras } from './cameraMetadata.js';
import {
  IDENTITY_MAT4,
  composeTrsMat4,
  mat4Invert,
  mat4Mul,
  nodeLocalMatrix,
} from './transforms.js';
import {
  applyGltfMaterialAnimationPointerValue,
  gltfAnimationPointerSampleValueError,
  resolveGltfAnimationPointer,
  type GltfAnimationPointerTarget,
} from './animationPointer.js';

export interface GltfScenePatchTarget {
  setScene(scene: Scene): void;
  updatePrimitive?(id: string, patch: ScenePrimitivePatch): void;
  updateEmitter?(id: string, patch: Partial<SceneEmitter>): void;
  reset?(): void;
}

export interface GltfSceneControllerInput extends Omit<GltfToSceneResult, 'cameras'> {
  readonly gltf: GltfJson;
  readonly sceneIndex?: number;
  readonly cameras?: readonly GltfSceneCamera[];
}

export interface GltfSceneControllerOptions {
  readonly engine?: GltfScenePatchTarget;
  readonly setSceneOnAttach?: boolean;
  /**
   * Maximum number of warnings and structured diagnostics retained in the
   * controller-wide history. Per-operation result arrays are never truncated.
   * Default: 256. Set to 0 to disable retained history.
   */
  readonly diagnosticHistoryLimit?: number;
}

export type GltfClipSelector = number | string | AnimationClip;

export interface GltfApplyAnimationOptions {
  readonly engine?: GltfScenePatchTarget;
  readonly loop?: boolean;
  readonly forceSetScene?: boolean;
}

export interface GltfPlaybackOptions extends GltfApplyAnimationOptions {
  readonly time?: number;
}

export interface GltfBlendAnimationOptions {
  readonly engine?: GltfScenePatchTarget;
  readonly loop?: boolean;
  readonly forceSetScene?: boolean;
  readonly times?: readonly number[];
}

export interface GltfApplyVariantOptions {
  readonly engine?: GltfScenePatchTarget;
  readonly forceSetScene?: boolean;
}

export interface GltfResetPoseOptions {
  readonly engine?: GltfScenePatchTarget;
  readonly forceSetScene?: boolean;
  readonly resetPlayback?: boolean;
}

export interface GltfPrimitivePatchRecord {
  readonly id: string;
  readonly patch: ScenePrimitivePatch;
}

export interface GltfEmitterPatchRecord {
  readonly id: string;
  readonly patch: Partial<SceneEmitter>;
}

export type GltfSceneControllerDiagnosticCode =
  | 'animation-matrix-overridden'
  | 'animation-matrix-trs-unavailable'
  | 'animation-morph-target-missing'
  | 'animation-skin-joint-unreachable'
  | 'animation-skin-mesh-transform-noninvertible'
  | 'animation-pointer-material-missing'
  | 'animation-pointer-material-unmapped'
  | 'animation-pointer-camera-unmapped'
  | 'animation-pointer-light-unmapped'
  | 'animation-pointer-value-invalid'
  | 'animation-pointer-unsupported'
  | 'animation-target-node-unmapped'
  | 'controller-update-primitive-failed'
  | 'controller-update-emitter-failed'
  | 'morph-weight-count-mismatch'
  | 'variant-bindings-missing'
  | 'variant-converted-materials-missing'
  | 'variant-list-malformed'
  | 'variant-material-index-missing'
  | 'variant-mapping-malformed'
  | 'variant-mapping-material-missing'
  | 'variant-primitive-missing-in-scene'
  | 'variant-provenance-missing-primitive'
  | 'variant-selection-not-found';

export interface GltfSceneControllerDiagnostic {
  readonly severity: 'warning';
  readonly code: GltfSceneControllerDiagnosticCode;
  readonly path: string;
  readonly message: string;
  readonly caller: 'applyAnimation' | 'blend' | 'setVariant' | 'resetPose';
  readonly primitiveId?: string;
  readonly emitterId?: string;
  readonly nodeIndex?: number;
  readonly jointNodeIndex?: number;
  readonly variantIndex?: number;
  readonly materialIndex?: number;
  readonly meshIndex?: number;
  readonly primitiveIndex?: number;
}

export interface GltfAnimationApplyResult {
  readonly clip: AnimationClip;
  readonly requestedTime: number;
  readonly localTime: number;
  readonly primitivePatches: readonly GltfPrimitivePatchRecord[];
  readonly emitterPatches: readonly GltfEmitterPatchRecord[];
  readonly cameras: readonly GltfSceneCamera[];
  readonly scene: Scene;
  readonly warnings: readonly string[];
  readonly diagnostics: readonly GltfSceneControllerDiagnostic[];
  readonly usedSetScene: boolean;
}

export interface GltfBlendApplyResult {
  readonly clips: readonly AnimationClip[];
  readonly requestedTimes: readonly number[];
  readonly localTimes: readonly number[];
  readonly weights: readonly number[];
  readonly primitivePatches: readonly GltfPrimitivePatchRecord[];
  readonly emitterPatches: readonly GltfEmitterPatchRecord[];
  readonly cameras: readonly GltfSceneCamera[];
  readonly scene: Scene;
  readonly warnings: readonly string[];
  readonly diagnostics: readonly GltfSceneControllerDiagnostic[];
  readonly usedSetScene: boolean;
}

export interface GltfVariantApplyResult {
  readonly requestedVariant: string | number | undefined;
  readonly variantIndex?: number;
  readonly primitivePatches: readonly GltfPrimitivePatchRecord[];
  readonly scene: Scene;
  readonly warnings: readonly string[];
  readonly diagnostics: readonly GltfSceneControllerDiagnostic[];
  readonly usedSetScene: boolean;
}

interface NodeLocalState {
  translation: [number, number, number];
  rotation: [number, number, number, number];
  scale: [number, number, number];
  matrix?: Float32Array;
  matrixResidual?: Float32Array;
  /** The raw matrix is valid, but no invertible TRS base exists for channel replacement. */
  matrixTrsUnavailable?: boolean;
  matrixOverridden: boolean;
}

interface SkinBinding {
  meshNodeIndex: number;
  jointNodeIndices: readonly number[];
}

type GltfSceneControllerDiagnosticDraft = Omit<
  GltfSceneControllerDiagnostic,
  'caller' | 'severity'
>;

interface GltfSceneControllerDiagnosticFrame {
  readonly warnings: string[];
  readonly diagnostics: GltfSceneControllerDiagnostic[];
  readonly caller: GltfSceneControllerDiagnostic['caller'];
}

interface ResolvedPointerSample {
  readonly sample: SampledChannel;
  readonly target: GltfAnimationPointerTarget;
}

const NODE_ID_PREFIX = 'gltf-node-';
const DEFAULT_DIAGNOSTIC_HISTORY_LIMIT = 256;

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

function diagnosticHistoryLimit(value: number | undefined): number {
  if (value === undefined) return DEFAULT_DIAGNOSTIC_HISTORY_LIMIT;
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new Error(
      '[vitrum/gltf-adapter] GltfSceneController: diagnosticHistoryLimit must be a ' +
        `non-negative safe integer; received ${String(value)}.`,
    );
  }
  return value;
}

function appendBounded<T>(target: T[], incoming: readonly T[], limit: number): void {
  if (limit === 0 || incoming.length === 0) return;
  const start = Math.max(0, incoming.length - limit);
  const retainedIncoming = incoming.length - start;
  const keepExisting = Math.max(0, limit - retainedIncoming);
  if (target.length > keepExisting) {
    target.splice(0, target.length - keepExisting);
  }
  for (let i = start; i < incoming.length; i += 1) {
    target.push(incoming[i]!);
  }
}

function resetAfterIncrementalPrimitivePatch(target: GltfScenePatchTarget): void {
  target.reset?.();
}

function primitiveResetPatch(
  primitive: ScenePrimitive,
  previous: ScenePrimitive | undefined,
): ScenePrimitivePatch {
  const patch: Record<string, unknown> = {};
  if (previous?.kind === primitive.kind) {
    const base = primitive as unknown as Record<string, unknown>;
    for (const key of Object.keys(previous)) {
      if (key !== 'id' && key !== 'kind' && !Object.prototype.hasOwnProperty.call(base, key)) {
        patch[key] = undefined;
      }
    }
  }
  for (const [key, value] of Object.entries(primitive as unknown as Record<string, unknown>)) {
    if (key === 'id' || key === 'kind') continue;
    patch[key] = value;
  }
  patch.material = materialReplacementPatch(primitive.material);
  return patch;
}

function emitterResetPatch(
  emitter: SceneEmitter,
  previous: SceneEmitter | undefined,
): Partial<SceneEmitter> {
  const patch: Record<string, unknown> = {};
  if (previous?.kind === emitter.kind) {
    const base = emitter as unknown as Record<string, unknown>;
    for (const key of Object.keys(previous)) {
      if (key !== 'id' && key !== 'kind' && !Object.prototype.hasOwnProperty.call(base, key)) {
        patch[key] = undefined;
      }
    }
  }
  for (const [key, value] of Object.entries(emitter as unknown as Record<string, unknown>)) {
    if (key === 'id' || key === 'kind') continue;
    patch[key] = value;
  }
  return patch;
}

function emitControllerDiagnostic(
  frame: GltfSceneControllerDiagnosticFrame,
  diagnostic: GltfSceneControllerDiagnosticDraft,
): void {
  frame.warnings.push(diagnostic.message);
  frame.diagnostics.push({
    severity: 'warning',
    caller: frame.caller,
    ...diagnostic,
  });
}

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
  readonly #baseCameras: readonly GltfSceneCamera[];
  #cameras: readonly GltfSceneCamera[];
  #engine: GltfScenePatchTarget | undefined;
  #activeClip: AnimationClip | undefined;
  #clock = 0;
  #playing = false;
  readonly #sceneIndex: number;
  readonly #baseLocals: readonly NodeLocalState[];
  readonly #nodeToPrimitiveIds: ReadonlyMap<number, readonly string[]>;
  readonly #primitiveNodeById: ReadonlyMap<string, number>;
  readonly #emitterNodeById: ReadonlyMap<string, number>;
  readonly #allPrimitiveIds: readonly string[];
  readonly #allEmitterIds: readonly string[];
  readonly #basePrimitiveById: Map<string, ScenePrimitive>;
  readonly #convertedMaterials: readonly MaterialSpec[];
  readonly #materialVariantBindings: readonly GltfMaterialVariantBinding[];
  readonly #materialBindingsByMaterialIndex: Map<number, string[]>;
  readonly #materialIndexByPrimitiveId: Map<string, number>;
  readonly #instancingBindingsByPrimitiveId: ReadonlyMap<string, GltfInstancingBinding>;
  readonly #punctualEmitterBindings: readonly GltfPunctualEmitterBinding[];
  readonly #baseEmitterById: ReadonlyMap<string, SceneEmitter>;
  readonly #baseNodeVisibility: readonly boolean[];
  readonly #skinBindingsByPrimitiveId: ReadonlyMap<string, SkinBinding>;
  #materialPointerPrimitiveIds = new Set<string>();
  readonly #warnings: string[] = [];
  readonly #diagnostics: GltfSceneControllerDiagnostic[] = [];
  readonly #warnedMatrixOverrideNodes = new Set<number>();
  readonly #warnedMatrixTrsUnavailableNodes = new Set<number>();
  readonly #samplersByClip = new WeakMap<AnimationClip, AnimationClipSampler>();
  readonly #diagnosticHistoryLimit: number;

  constructor(input: GltfSceneControllerInput, options: GltfSceneControllerOptions = {}) {
    // patchPrimitiveInScene intentionally validates only the changed primitive
    // on its hot path. Establish its documented precondition once at this
    // public boundary, including hidden node-visibility members that can enter
    // a later frame's snapshot.
    validateScene(input.scene);
    const allPrimitives = input.nodeVisibilityPrimitives ?? input.scene.primitives;
    const allEmitters = input.nodeVisibilityEmitters ?? input.scene.emitters;
    if (allPrimitives !== input.scene.primitives || allEmitters !== input.scene.emitters) {
      validateScene({
        ...input.scene,
        primitives: allPrimitives,
        emitters: allEmitters,
      });
    }
    this.#diagnosticHistoryLimit = diagnosticHistoryLimit(options.diagnosticHistoryLimit);
    this.gltf = input.gltf;
    this.animations = input.animations;
    this.animationTargets = input.animationTargets;
    this.#scene = input.scene;
    this.#sceneIndex = input.sceneIndex ?? input.gltf.scene ?? 0;
    this.#baseLocals = (input.gltf.nodes ?? []).map(baseLocalState);
    this.#baseCameras =
      input.cameras ??
      collectSceneCameras(
        input.gltf,
        buildWorldTransformsForLocals(input.gltf, this.#rootNodes(), this.#baseLocals),
      );
    this.#cameras = this.#baseCameras;
    this.#nodeToPrimitiveIds = parseAnimationTargets(input.animationTargets);
    this.#primitiveNodeById = invertNodePrimitiveBindings(this.#nodeToPrimitiveIds);
    this.#allPrimitiveIds = allPrimitives.map((primitive) => String(primitive.id));
    this.#basePrimitiveById = new Map(allPrimitives.map((p) => [String(p.id), p]));
    this.#convertedMaterials = input.convertedMaterials ?? [];
    this.#materialVariantBindings = input.materialVariantBindings ?? [];
    const materialBindings = input.materialBindings ?? [];
    this.#materialBindingsByMaterialIndex = buildMaterialBindings(materialBindings);
    this.#materialIndexByPrimitiveId = buildMaterialIndexByPrimitiveId(materialBindings);
    this.#instancingBindingsByPrimitiveId = new Map(
      (input.instancingBindings ?? []).map((binding) => [binding.primitiveId, binding]),
    );
    this.#punctualEmitterBindings = input.punctualEmitterBindings ?? [];
    this.#emitterNodeById = new Map(
      this.#punctualEmitterBindings.map((binding) => [binding.emitterId, binding.nodeIndex]),
    );
    this.#allEmitterIds = allEmitters.map((emitter) => String(emitter.id));
    this.#baseEmitterById = new Map(allEmitters.map((emitter) => [String(emitter.id), emitter]));
    this.#baseNodeVisibility = (input.gltf.nodes ?? []).map(nodeOwnVisibility);
    this.#skinBindingsByPrimitiveId = buildSkinBindings(input.gltf, this.#nodeToPrimitiveIds);
    this.#engine = options.engine;
    if (options.engine && (options.setSceneOnAttach ?? true)) {
      options.engine.setScene(this.#scene);
    }
  }

  get scene(): Scene {
    return this.#scene;
  }

  get cameras(): readonly GltfSceneCamera[] {
    return this.#cameras;
  }

  get warnings(): readonly string[] {
    return this.#warnings;
  }

  get diagnostics(): readonly GltfSceneControllerDiagnostic[] {
    return this.#diagnostics;
  }

  get activeClip(): AnimationClip | undefined {
    return this.#activeClip;
  }

  get currentTime(): number {
    return this.#clock;
  }

  get playing(): boolean {
    return this.#playing;
  }

  /** Release retained warning/diagnostic objects without affecting scene state. */
  clearDiagnosticHistory(): void {
    this.#warnings.length = 0;
    this.#diagnostics.length = 0;
  }

  /** Release the attached engine reference; later calls may pass an engine explicitly. */
  detachEngine(): void {
    this.#engine = undefined;
  }

  attachEngine(engine: GltfScenePatchTarget, options: { readonly setScene?: boolean } = {}): void {
    if (options.setScene ?? true) {
      engine.setScene(this.#scene);
    }
    this.#engine = engine;
  }

  #recordDiagnosticFrame(frame: GltfSceneControllerDiagnosticFrame): void {
    appendBounded(this.#warnings, frame.warnings, this.#diagnosticHistoryLimit);
    appendBounded(this.#diagnostics, frame.diagnostics, this.#diagnosticHistoryLimit);
  }

  setActiveClip(selector: GltfClipSelector, time = 0): AnimationClip {
    const clip = this.#resolveClip(selector);
    this.#activeClip = clip;
    this.#clock = time;
    return clip;
  }

  play(selector?: GltfClipSelector, options: GltfPlaybackOptions = {}): GltfAnimationApplyResult {
    const clip =
      selector !== undefined
        ? this.#resolveClip(selector)
        : (this.#activeClip ?? this.#defaultClip('play'));
    const { time, ...applyOptions } = options;
    const nextTime = time ?? this.#clock;
    const result = this.applyAnimation(clip, nextTime, applyOptions);
    this.#activeClip = clip;
    this.#clock = nextTime;
    this.#playing = true;
    return result;
  }

  pause(): void {
    this.#playing = false;
  }

  resume(options: GltfApplyAnimationOptions = {}): GltfAnimationApplyResult {
    const clip = this.#activeClip ?? this.#defaultClip('resume');
    const result = this.applyAnimation(clip, this.#clock, options);
    this.#activeClip = clip;
    this.#playing = true;
    return result;
  }

  tick(
    deltaSeconds: number,
    options: GltfApplyAnimationOptions = {},
  ): GltfAnimationApplyResult | undefined {
    if (!this.#playing) return undefined;
    return this.advance(deltaSeconds, options);
  }

  seek(time: number, options: GltfApplyAnimationOptions = {}): GltfAnimationApplyResult {
    let clip = this.#activeClip;
    if (!clip) {
      clip = this.animations[0];
      if (!clip) {
        throw new Error('[vitrum/gltf-adapter] GltfSceneController.seek: asset has no animations.');
      }
    }
    const result = this.applyAnimation(clip, time, options);
    this.#activeClip = clip;
    this.#clock = time;
    return result;
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
    const samples = this.#samplerForClip(clip).sample(localTime);
    const frameWarnings: string[] = [];
    const frameDiagnostics: GltfSceneControllerDiagnostic[] = [];
    const applied = this.#applySampledChannels(samples, options, {
      warnings: frameWarnings,
      diagnostics: frameDiagnostics,
      caller: 'applyAnimation',
    });

    return {
      clip,
      requestedTime: time,
      localTime,
      primitivePatches: applied.primitivePatches,
      emitterPatches: applied.emitterPatches,
      cameras: applied.cameras,
      scene: applied.scene,
      warnings: frameWarnings,
      diagnostics: frameDiagnostics,
      usedSetScene: applied.usedSetScene,
    };
  }

  blend(
    selectors: readonly GltfClipSelector[],
    weights: readonly number[],
    time = this.#clock,
    options: GltfBlendAnimationOptions = {},
  ): GltfBlendApplyResult {
    if (selectors.length === 0) {
      throw new Error(
        '[vitrum/gltf-adapter] GltfSceneController.blend: at least one clip is required.',
      );
    }
    if (selectors.length !== weights.length) {
      throw new Error(
        `[vitrum/gltf-adapter] GltfSceneController.blend: clip count (${selectors.length}) ` +
          `does not match weight count (${weights.length}).`,
      );
    }
    if (options.times && options.times.length !== selectors.length) {
      throw new Error(
        `[vitrum/gltf-adapter] GltfSceneController.blend: times count (${options.times.length}) ` +
          `does not match clip count (${selectors.length}).`,
      );
    }

    const clips = selectors.map((selector) => this.#resolveClip(selector));
    const positiveWeights = normalizeBlendWeights(weights);
    const requestedTimes = clips.map((_, index) => options.times?.[index] ?? time);
    const localTimes = clips.map((clip, index) =>
      normalizeClipTime(clip, requestedTimes[index] ?? 0, options.loop ?? false),
    );
    const perClipSamples = clips.map((clip, index) =>
      this.#samplerForClip(clip).sample(localTimes[index] ?? 0),
    );
    const samples = blendSampledChannels(perClipSamples, positiveWeights);
    const frameWarnings: string[] = [];
    const frameDiagnostics: GltfSceneControllerDiagnostic[] = [];
    const applied = this.#applySampledChannels(samples, options, {
      warnings: frameWarnings,
      diagnostics: frameDiagnostics,
      caller: 'blend',
    });

    this.#clock = time;
    return {
      clips,
      requestedTimes,
      localTimes,
      weights: positiveWeights,
      primitivePatches: applied.primitivePatches,
      emitterPatches: applied.emitterPatches,
      cameras: applied.cameras,
      scene: applied.scene,
      warnings: frameWarnings,
      diagnostics: frameDiagnostics,
      usedSetScene: applied.usedSetScene,
    };
  }

  #samplerForClip(clip: AnimationClip): AnimationClipSampler {
    let sampler = this.#samplersByClip.get(clip);
    if (sampler == null) {
      sampler = createAnimationClipSampler(clip);
      this.#samplersByClip.set(clip, sampler);
    }
    return sampler;
  }

  setVariant(
    selector: string | number | undefined = undefined,
    options: GltfApplyVariantOptions = {},
  ): GltfVariantApplyResult {
    const frameWarnings: string[] = [];
    const frameDiagnostics: GltfSceneControllerDiagnostic[] = [];
    const frame: GltfSceneControllerDiagnosticFrame = {
      warnings: frameWarnings,
      diagnostics: frameDiagnostics,
      caller: 'setVariant',
    };
    const selectedVariantIndex = resolveMaterialVariantSelection(this.gltf, selector, frame);
    const patchMap = new Map<string, ScenePrimitivePatch>();
    const materialBindingUpdates = new Map<string, number | undefined>();

    if (this.#materialVariantBindings.length === 0) {
      if ((this.gltf.extensions?.KHR_materials_variants?.variants?.length ?? 0) > 0) {
        emitControllerDiagnostic(frame, {
          code: 'variant-bindings-missing',
          path: 'materialVariantBindings',
          message:
            '[vitrum/gltf-adapter] GltfSceneController.setVariant: this controller input has no ' +
            'materialVariantBindings metadata. Recreate it from gltfToScene() before switching variants.',
        });
      }
    } else if (this.#convertedMaterials.length === 0 && (this.gltf.materials?.length ?? 0) > 0) {
      emitControllerDiagnostic(frame, {
        code: 'variant-converted-materials-missing',
        path: 'convertedMaterials',
        message:
          '[vitrum/gltf-adapter] GltfSceneController.setVariant: this controller input has no ' +
          'convertedMaterials metadata. Recreate it from gltfToScene() before switching variants.',
      });
    } else {
      for (const binding of this.#materialVariantBindings) {
        const mesh = this.gltf.meshes?.[binding.meshIndex];
        const primitive = mesh?.primitives?.[binding.primitiveIndex];
        if (!primitive) {
          emitControllerDiagnostic(frame, {
            code: 'variant-provenance-missing-primitive',
            path: `meshes[${binding.meshIndex}].primitives[${binding.primitiveIndex}]`,
            message:
              `[vitrum/gltf-adapter] GltfSceneController.setVariant: primitive provenance for ` +
              `"${binding.primitiveId}" points at missing meshes[${binding.meshIndex}].primitives[` +
              `${binding.primitiveIndex}]; patch skipped.`,
            primitiveId: binding.primitiveId,
            meshIndex: binding.meshIndex,
            primitiveIndex: binding.primitiveIndex,
          });
          continue;
        }
        if (!this.#basePrimitiveById.has(binding.primitiveId)) {
          emitControllerDiagnostic(frame, {
            code: 'variant-primitive-missing-in-scene',
            path: `scene.primitives["${binding.primitiveId}"]`,
            message:
              `[vitrum/gltf-adapter] GltfSceneController.setVariant: primitive "${binding.primitiveId}" ` +
              'is no longer present in the controller visual inventory; patch skipped.',
            primitiveId: binding.primitiveId,
          });
          continue;
        }

        const materialIndex = resolvePrimitiveMaterialIndex(
          this.gltf,
          primitive,
          binding.baseMaterialIndex,
          selectedVariantIndex,
          frame,
          `${mesh?.name ?? binding.meshIndex}`,
          binding.meshIndex,
          binding.primitiveIndex,
        );
        materialBindingUpdates.set(binding.primitiveId, materialIndex);
        const variantPatch = materialVariantPatchForSelection(binding, selectedVariantIndex);
        if (variantPatch) {
          const material = materialForVariantPatch(
            materialForIndex(this.#convertedMaterials, variantPatch.materialIndex, frame),
            variantPatch,
          );
          mergePrimitivePatch(
            patchMap,
            binding.primitiveId,
            scenePrimitivePatchForMaterialVariant(variantPatch, material),
          );
        } else {
          const material = materialForIndex(this.#convertedMaterials, materialIndex, frame);
          mergePrimitivePatch(patchMap, binding.primitiveId, {
            material: materialReplacementPatch(material),
          });
        }
      }
    }

    const primitivePatches = Array.from(patchMap, ([id, patch]) => ({ id, patch }));
    const target = options.engine ?? this.#engine;
    let nextScene = this.#scene;
    for (const { id, patch } of primitivePatches) {
      nextScene = patchPrimitiveInScene(nextScene, id, patch);
    }

    const usedSetScene = this.#commitSceneChange(
      target,
      primitivePatches.filter(({ id }) => findPrimitive(this.#scene, id) !== undefined),
      [],
      nextScene,
      frame,
      options.forceSetScene ?? false,
    );

    this.#scene = nextScene;
    for (const [id, materialIndex] of materialBindingUpdates) {
      this.#setCurrentMaterialBinding(id, materialIndex);
    }
    for (const { id, patch } of primitivePatches) {
      const base = this.#basePrimitiveById.get(id);
      if (base) this.#basePrimitiveById.set(id, { ...base, ...patch } as ScenePrimitive);
    }
    this.#recordDiagnosticFrame(frame);

    return {
      requestedVariant: selector,
      ...(selectedVariantIndex !== undefined ? { variantIndex: selectedVariantIndex } : {}),
      primitivePatches,
      scene: this.#scene,
      warnings: frameWarnings,
      diagnostics: frameDiagnostics,
      usedSetScene,
    };
  }

  resetPose(options: GltfResetPoseOptions = {}): void {
    const target = options.engine ?? this.#engine;
    const previousScene = this.#scene;
    const effectiveVisibility = buildEffectiveNodeVisibilityForController(
      this.gltf,
      this.#rootNodes(),
      this.#baseNodeVisibility,
    );
    const nextScene: Scene = {
      ...this.#scene,
      primitives: this.#allPrimitiveIds.flatMap((id) => {
        const nodeIndex = this.#primitiveNodeById.get(id);
        const primitive = this.#basePrimitiveById.get(id);
        return primitive &&
          (nodeIndex === undefined || effectiveVisibility.get(nodeIndex) !== false)
          ? [primitive]
          : [];
      }),
      emitters: this.#allEmitterIds.flatMap((id) => {
        const nodeIndex = this.#emitterNodeById.get(id);
        const emitter = this.#baseEmitterById.get(id);
        return emitter && (nodeIndex === undefined || effectiveVisibility.get(nodeIndex) !== false)
          ? [emitter]
          : [];
      }),
    };
    const publishResetState = (): void => {
      this.#scene = nextScene;
      this.#cameras = this.#baseCameras;
      this.#materialPointerPrimitiveIds.clear();
      if (options.resetPlayback === true) {
        this.#activeClip = undefined;
        this.#clock = 0;
        this.#playing = false;
      }
    };
    if (!target) {
      publishResetState();
      return;
    }
    const previousPrimitiveIds = new Set(
      previousScene.primitives.map((primitive) => String(primitive.id)),
    );
    const previousEmitterIds = new Set(previousScene.emitters.map((emitter) => String(emitter.id)));
    const nextPrimitiveIds = new Set(nextScene.primitives.map((primitive) => String(primitive.id)));
    const nextEmitterIds = new Set(nextScene.emitters.map((emitter) => String(emitter.id)));
    const visibilityChanged =
      !setsEqual(previousPrimitiveIds, nextPrimitiveIds) ||
      !setsEqual(previousEmitterIds, nextEmitterIds);
    const primitivePatches = nextScene.primitives
      .map((primitive) => {
        const id = String(primitive.id);
        const previous = findPrimitive(previousScene, id);
        return {
          id,
          patch: primitiveResetPatch(primitive, previous),
          changed: previous !== primitive,
        };
      })
      .filter((entry) => entry.changed)
      .map(({ id, patch }) => ({ id, patch }));
    const emitterPatches = nextScene.emitters
      .map((emitter) => {
        const id = String(emitter.id);
        const previous = findEmitter(previousScene, id);
        return {
          id,
          patch: emitterResetPatch(emitter, previous),
          changed: previous !== emitter,
        };
      })
      .filter((entry) => entry.changed)
      .map(({ id, patch }) => ({ id, patch }));
    if (primitivePatches.length === 0 && emitterPatches.length === 0 && !visibilityChanged) {
      if (options.forceSetScene === true) target.setScene(nextScene);
      publishResetState();
      return;
    }
    const frame: GltfSceneControllerDiagnosticFrame = {
      warnings: [],
      diagnostics: [],
      caller: 'resetPose',
    };
    this.#commitSceneChange(
      target,
      primitivePatches,
      emitterPatches,
      nextScene,
      frame,
      (options.forceSetScene ?? false) || visibilityChanged,
    );
    publishResetState();
    this.#recordDiagnosticFrame(frame);
  }

  /**
   * Commit a batch of primitive patches to the target engine (D15-5). Prefers
   * incremental `updatePrimitive`; on ANY failure it emits a diagnostic through
   * `frame` and falls back to a full `setScene(nextScene)`. Returns whether the
   * setScene fallback (or the forced-setScene path) was taken. Extracted from the
   * byte-identical patch-then-fallback blocks in `setVariant` and
   * `applyAnimationFrame`. (`resetPose` keeps its own early-return variant.)
   */
  #commitSceneChange(
    target: GltfScenePatchTarget | undefined,
    primitivePatches: readonly GltfPrimitivePatchRecord[],
    emitterPatches: readonly GltfEmitterPatchRecord[],
    nextScene: Scene,
    frame: GltfSceneControllerDiagnosticFrame,
    forceSetScene: boolean,
  ): boolean {
    if (!target) return false;
    if (forceSetScene) {
      target.setScene(nextScene);
      return true;
    }
    if (primitivePatches.length === 0 && emitterPatches.length === 0) return false;
    const canPatchPrimitives =
      primitivePatches.length === 0 || target.updatePrimitive !== undefined;
    const canPatchEmitters = emitterPatches.length === 0 || target.updateEmitter !== undefined;
    if (canPatchPrimitives && canPatchEmitters) {
      let attemptedPrimitiveId: string | undefined;
      let attemptedEmitterId: string | undefined;
      try {
        for (const { id, patch } of primitivePatches) {
          attemptedPrimitiveId = id;
          target.updatePrimitive!(id, patch);
        }
        for (const { id, patch } of emitterPatches) {
          attemptedEmitterId = id;
          target.updateEmitter!(id, patch);
        }
        resetAfterIncrementalPrimitivePatch(target);
        return false;
      } catch (err) {
        const message = errorMessage(err);
        if (attemptedEmitterId !== undefined) {
          emitControllerDiagnostic(frame, {
            code: 'controller-update-emitter-failed',
            path: `scene.emitters["${attemptedEmitterId}"]`,
            message:
              `[vitrum/gltf-adapter] GltfSceneController.${frame.caller}: ` +
              `engine.updateEmitter failed; falling back to setScene(nextScene). ${message}`,
            emitterId: attemptedEmitterId,
          });
        } else {
          emitControllerDiagnostic(frame, {
            code: 'controller-update-primitive-failed',
            path: attemptedPrimitiveId
              ? `scene.primitives["${attemptedPrimitiveId}"]`
              : 'scene.primitives',
            message:
              `[vitrum/gltf-adapter] GltfSceneController.${frame.caller}: ` +
              `engine.updatePrimitive failed; falling back to setScene(nextScene). ${message}`,
            ...(attemptedPrimitiveId !== undefined ? { primitiveId: attemptedPrimitiveId } : {}),
          });
        }
        target.setScene(nextScene);
        return true;
      }
    }
    target.setScene(nextScene);
    return true;
  }

  #setCurrentMaterialBinding(primitiveId: string, materialIndex: number | undefined): void {
    const previousIndex = this.#materialIndexByPrimitiveId.get(primitiveId);
    if (previousIndex === materialIndex) return;
    if (previousIndex !== undefined) {
      const previousIds = this.#materialBindingsByMaterialIndex.get(previousIndex);
      if (previousIds) {
        const filtered = previousIds.filter((id) => id !== primitiveId);
        if (filtered.length > 0) {
          this.#materialBindingsByMaterialIndex.set(previousIndex, filtered);
        } else {
          this.#materialBindingsByMaterialIndex.delete(previousIndex);
        }
      }
      this.#materialIndexByPrimitiveId.delete(primitiveId);
    }
    if (materialIndex === undefined) return;
    const nextIds = this.#materialBindingsByMaterialIndex.get(materialIndex);
    if (nextIds) {
      if (!nextIds.includes(primitiveId)) nextIds.push(primitiveId);
    } else {
      this.#materialBindingsByMaterialIndex.set(materialIndex, [primitiveId]);
    }
    this.#materialIndexByPrimitiveId.set(primitiveId, materialIndex);
  }

  #defaultClip(caller: 'play' | 'resume'): AnimationClip {
    const clip = this.animations[0];
    if (!clip) {
      throw new Error(
        `[vitrum/gltf-adapter] GltfSceneController.${caller}: asset has no animations.`,
      );
    }
    return clip;
  }

  #resolveClip(selector: GltfClipSelector): AnimationClip {
    if (typeof selector === 'number') {
      const clip = this.animations[selector];
      if (!clip)
        throw new Error(`[vitrum/gltf-adapter] Animation clip index ${selector} not found.`);
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
    frame: GltfSceneControllerDiagnosticFrame,
  ): void {
    const nodeIndex = parseAnimationNodeIndex(sample.node);
    if (nodeIndex == null || !locals[nodeIndex]) {
      emitControllerDiagnostic(frame, {
        code: 'animation-target-node-unmapped',
        path: `animations.channels.target.node["${sample.node}"]`,
        message: `[vitrum/gltf-adapter] Animation target "${sample.node}" does not map to a glTF node; channel skipped.`,
      });
      return;
    }
    if (sample.path === 'weights') {
      morphWeightsByNode.set(nodeIndex, new Float32Array(sample.value));
      return;
    }

    const state = locals[nodeIndex];
    if (state.matrix && state.matrixTrsUnavailable) {
      if (!this.#warnedMatrixTrsUnavailableNodes.has(nodeIndex)) {
        this.#warnedMatrixTrsUnavailableNodes.add(nodeIndex);
        emitControllerDiagnostic(frame, {
          code: 'animation-matrix-trs-unavailable',
          path: `nodes[${nodeIndex}].matrix`,
          message:
            `[vitrum/gltf-adapter] Animation targets node ${nodeIndex}, whose imported matrix ` +
            'has a singular basis and cannot be decomposed into an invertible TRS base. ' +
            'The TRS channel was skipped and the authored matrix remains active.',
          nodeIndex,
        });
      }
      return;
    }
    if (state.matrix && !this.#warnedMatrixOverrideNodes.has(nodeIndex)) {
      this.#warnedMatrixOverrideNodes.add(nodeIndex);
      emitControllerDiagnostic(frame, {
        code: 'animation-matrix-overridden',
        path: `nodes[${nodeIndex}].matrix`,
        message:
          `[vitrum/gltf-adapter] Animation targets node ${nodeIndex}, which imported from a matrix. ` +
          'The controller decomposes its base TRS, replaces only the animated channel, and preserves ' +
          'the remaining affine residual for this frame.',
        nodeIndex,
      });
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

  #applySampledChannels(
    samples: readonly SampledChannel[],
    options: Pick<GltfApplyAnimationOptions, 'engine' | 'forceSetScene'>,
    frame: GltfSceneControllerDiagnosticFrame,
  ): {
    primitivePatches: readonly GltfPrimitivePatchRecord[];
    emitterPatches: readonly GltfEmitterPatchRecord[];
    cameras: readonly GltfSceneCamera[];
    scene: Scene;
    usedSetScene: boolean;
  } {
    const locals = cloneLocalStates(this.#baseLocals);
    const morphWeightsByNode = new Map<number, Float32Array>();
    const materialPointerSamples: ResolvedPointerSample[] = [];
    const cameraPointerSamples: ResolvedPointerSample[] = [];
    const lightPointerSamples: ResolvedPointerSample[] = [];
    const directNodeVisibility = [...this.#baseNodeVisibility];

    for (const sample of samples) {
      if (sample.path === 'pointer') {
        const target = resolveGltfAnimationPointer(sample.pointer);
        if (target === undefined) {
          emitControllerDiagnostic(frame, {
            code: 'animation-pointer-unsupported',
            path: `animations.channels.target.extensions.KHR_animation_pointer.pointer["${sample.pointer ?? ''}"]`,
            message:
              `[vitrum/gltf-adapter] GltfSceneController.${frame.caller}: unsupported ` +
              `KHR_animation_pointer target "${String(sample.pointer)}"; channel skipped.`,
          });
          continue;
        }
        const sampledValueError = gltfAnimationPointerSampleValueError(
          this.gltf,
          target,
          sample.value,
        );
        if (sampledValueError !== undefined) {
          emitControllerDiagnostic(frame, {
            code: 'animation-pointer-value-invalid',
            path: target.pointer,
            message:
              `[vitrum/gltf-adapter] GltfSceneController.${frame.caller}: sampled ` +
              `KHR_animation_pointer value for "${target.pointer}" is invalid: ` +
              `${sampledValueError}. Channel skipped without clamping.`,
          });
          continue;
        }
        if (target.kind === 'node') {
          this.#applySampleToLocals(
            {
              node: `${NODE_ID_PREFIX}${target.nodeIndex}`,
              path: target.path,
              value: sample.value,
            },
            locals,
            morphWeightsByNode,
            frame,
          );
          continue;
        }
        if (target.kind === 'node-weight') {
          const weights =
            morphWeightsByNode.get(target.nodeIndex) ??
            baseMorphWeightsForNode(this.gltf, target.nodeIndex);
          if (weights === undefined || target.weightIndex >= weights.length) {
            emitControllerDiagnostic(frame, {
              code: 'animation-morph-target-missing',
              path: `nodes[${target.nodeIndex}].weights[${target.weightIndex}]`,
              message:
                `[vitrum/gltf-adapter] GltfSceneController.${frame.caller}: weight pointer ` +
                `"${target.pointer}" does not resolve to an imported morph target; channel skipped.`,
              nodeIndex: target.nodeIndex,
            });
          } else {
            weights[target.weightIndex] = sample.value[0]!;
            morphWeightsByNode.set(target.nodeIndex, weights);
          }
          continue;
        }
        if (target.kind === 'node-visibility') {
          if (directNodeVisibility[target.nodeIndex] === undefined) {
            emitControllerDiagnostic(frame, {
              code: 'animation-target-node-unmapped',
              path: `nodes[${target.nodeIndex}]`,
              message:
                `[vitrum/gltf-adapter] GltfSceneController.${frame.caller}: visibility pointer ` +
                `"${target.pointer}" references a missing glTF node; channel skipped.`,
              nodeIndex: target.nodeIndex,
            });
          } else {
            directNodeVisibility[target.nodeIndex] = (sample.value[0] ?? 0) !== 0;
          }
          continue;
        }
        const resolved = { sample, target };
        if (target.kind === 'material-property' || target.kind === 'material-texture-transform') {
          materialPointerSamples.push(resolved);
        } else if (target.kind === 'camera') {
          cameraPointerSamples.push(resolved);
        } else {
          lightPointerSamples.push(resolved);
        }
        continue;
      }
      this.#applySampleToLocals(sample, locals, morphWeightsByNode, frame);
    }

    const worldTransforms = buildWorldTransformsForLocals(this.gltf, this.#rootNodes(), locals);
    let cameras =
      (this.gltf.cameras?.length ?? 0) > 0
        ? collectSceneCameras(this.gltf, worldTransforms)
        : this.#baseCameras;
    cameras = applyCameraPointerSamples(cameras, cameraPointerSamples, frame);
    const patchMap = new Map<string, ScenePrimitivePatch>();

    for (const [nodeIndex, primitiveIds] of this.#nodeToPrimitiveIds) {
      const world = worldTransforms.get(nodeIndex);
      if (!world) continue;
      for (const id of primitiveIds) {
        const current = findPrimitive(this.#scene, id) ?? this.#basePrimitiveById.get(id);
        if (!current) continue;
        const instancingBinding = this.#instancingBindingsByPrimitiveId.get(id);
        if (instancingBinding && isInstancedMesh(current)) {
          const instances = buildAnimatedInstanceTransforms(
            world,
            instancingBinding.localInstanceTransforms,
          );
      if (!instanceMatricesEqual(current.instances, instances)) {
            mergePrimitivePatch(patchMap, id, { instances });
          }
          continue;
        }
        if (!mat4Equal(primitiveTransform(current), world)) {
          mergePrimitivePatch(patchMap, id, { transform: world });
        }
      }
    }

    const sampledMorphWeightsByPrimitiveId = new Map<string, Float32Array>();
    for (const [nodeIndex, weights] of morphWeightsByNode) {
      const primitiveIds = this.#nodeToPrimitiveIds.get(nodeIndex) ?? [];
      for (const id of primitiveIds) {
        const source = this.#deformationSource(id);
        if (!source?.morphTargets || source.morphTargets.length === 0) {
          emitControllerDiagnostic(frame, {
            code: 'animation-morph-target-missing',
            path: `scene.primitives["${id}"].morphTargets`,
            message:
              `[vitrum/gltf-adapter] GltfSceneController.${frame.caller}: animation weights ` +
              `target resolved to primitive "${id}", but that primitive has no morph targets; ` +
              'this primitive was skipped while morph-capable siblings continue.',
            primitiveId: id,
            nodeIndex,
          });
          continue;
        }
        sampledMorphWeightsByPrimitiveId.set(
          id,
          fitMorphWeights(weights, source.morphTargets.length, id, frame),
        );
      }
    }

    // Morph-only primitives have no glTF skin binding. True skinned primitives
    // publish only morph weights so engines solve once from their retained rest
    // streams. Native instanced primitives cannot carry skin state, so their
    // deformation source is still solved here into one shared geometry stream.
    for (const id of this.#allPrimitiveIds) {
      if (this.#skinBindingsByPrimitiveId.has(id)) continue;
      const source = this.#deformationSource(id);
      if (!source?.morphTargets || source.morphTargets.length === 0) continue;
      const weights =
        sampledMorphWeightsByPrimitiveId.get(id) ??
        source.morphWeights ??
        new Float32Array(source.morphTargets.length);
      const patch = this.#buildMorphPatch(id, weights);
      if (patch) mergePrimitivePatch(patchMap, id, patch);
    }

    for (const [id, binding] of this.#skinBindingsByPrimitiveId) {
      const patch = this.#buildSkinPatch(
        id,
        binding,
        worldTransforms,
        sampledMorphWeightsByPrimitiveId.get(id),
        frame,
      );
      if (patch) mergePrimitivePatch(patchMap, id, patch);
    }
    for (const id of this.#materialPointerPrimitiveIds) {
      const base = this.#basePrimitiveById.get(id);
      if (!base) continue;
      mergePrimitivePatch(patchMap, id, {
        material: materialReplacementPatch(base.material),
      });
    }
    const materialPointerPrimitiveIds = this.#applyMaterialPointerSamples(
      materialPointerSamples,
      patchMap,
      frame,
    );

    const primitivePatches = Array.from(patchMap, ([id, patch]) => ({ id, patch }));
    const emitterPatches = this.#buildAnimatedEmitterPatches(
      worldTransforms,
      lightPointerSamples,
      frame,
    );
    const effectiveVisibility = buildEffectiveNodeVisibilityForController(
      this.gltf,
      this.#rootNodes(),
      directNodeVisibility,
    );
    const visiblePrimitiveIds = new Set(
      this.#allPrimitiveIds.filter((id) => {
        const nodeIndex = this.#primitiveNodeById.get(id);
        return nodeIndex === undefined || effectiveVisibility.get(nodeIndex) !== false;
      }),
    );
    const visibleEmitterIds = new Set(
      this.#allEmitterIds.filter((id) => {
        const nodeIndex = this.#emitterNodeById.get(id);
        return nodeIndex === undefined || effectiveVisibility.get(nodeIndex) !== false;
      }),
    );
    const target = options.engine ?? this.#engine;
    const currentPrimitiveIds = new Set(
      this.#scene.primitives.map((primitive) => String(primitive.id)),
    );
    const currentEmitterIds = new Set(this.#scene.emitters.map((emitter) => String(emitter.id)));
    const nextScene: Scene = {
      ...this.#scene,
      primitives: this.#allPrimitiveIds.flatMap((id) => {
        if (!visiblePrimitiveIds.has(id)) return [];
        const source = findPrimitive(this.#scene, id) ?? this.#basePrimitiveById.get(id);
        if (!source) return [];
        const patch = patchMap.get(id);
        return [patch ? ({ ...source, ...patch } as ScenePrimitive) : source];
      }),
      emitters: this.#allEmitterIds.flatMap((id) => {
        if (!visibleEmitterIds.has(id)) return [];
        const source = findEmitter(this.#scene, id) ?? this.#baseEmitterById.get(id);
        if (!source) return [];
        const patch = emitterPatches.find((entry) => entry.id === id)?.patch;
        return [patch ? ({ ...source, ...patch } as SceneEmitter) : source];
      }),
    };
    const visibilityChanged =
      !setsEqual(currentPrimitiveIds, visiblePrimitiveIds) ||
      !setsEqual(currentEmitterIds, visibleEmitterIds);
    const committedPrimitivePatches = primitivePatches.filter(
      ({ id }) => currentPrimitiveIds.has(id) && visiblePrimitiveIds.has(id),
    );
    const committedEmitterPatches = emitterPatches.filter(
      ({ id }) => currentEmitterIds.has(id) && visibleEmitterIds.has(id),
    );

    const usedSetScene = this.#commitSceneChange(
      target,
      committedPrimitivePatches,
      committedEmitterPatches,
      nextScene,
      frame,
      (options.forceSetScene ?? false) || visibilityChanged,
    );

    this.#scene = nextScene;
    this.#cameras = cameras;
    this.#materialPointerPrimitiveIds = materialPointerPrimitiveIds;
    this.#recordDiagnosticFrame(frame);

    return {
      primitivePatches: primitivePatches.filter(({ id }) => visiblePrimitiveIds.has(id)),
      emitterPatches: emitterPatches.filter(({ id }) => visibleEmitterIds.has(id)),
      cameras,
      scene: this.#scene,
      usedSetScene,
    };
  }

  #buildAnimatedEmitterPatches(
    worldTransforms: ReadonlyMap<number, Mat4>,
    pointerSamples: readonly ResolvedPointerSample[],
    frame: GltfSceneControllerDiagnosticFrame,
  ): GltfEmitterPatchRecord[] {
    const patches: GltfEmitterPatchRecord[] = [];
    const mappedLightIndices = new Set(
      this.#punctualEmitterBindings.map((binding) => binding.lightIndex),
    );
    for (const { target } of pointerSamples) {
      if (target.kind !== 'punctual-light' || mappedLightIndices.has(target.lightIndex)) continue;
      emitControllerDiagnostic(frame, {
        code: 'animation-pointer-light-unmapped',
        path: `extensions.KHR_lights_punctual.lights[${target.lightIndex}]`,
        message:
          `[vitrum/gltf-adapter] GltfSceneController.${frame.caller}: light pointer ` +
          `"${target.pointer}" has no punctual emitter in the imported scene; channel skipped.`,
      });
    }
    for (const binding of this.#punctualEmitterBindings) {
      const world = worldTransforms.get(binding.nodeIndex);
      const base = this.#baseEmitterById.get(binding.emitterId);
      const current = findEmitter(this.#scene, binding.emitterId) ?? base;
      if (!world || !base || !current) continue;
      const posePatch = punctualEmitterPosePatch(base, world) ?? {};
      const lightSamples = pointerSamples.filter(
        ({ target }) =>
          target.kind === 'punctual-light' && target.lightIndex === binding.lightIndex,
      );
      const pointerPatch = punctualEmitterPointerPatch(base, lightSamples, frame);
      const desired = { ...base, ...posePatch, ...pointerPatch } as SceneEmitter;
      const patch = emitterDifferencePatch(desired, current);
      if (Object.keys(patch).length > 0) {
        patches.push({ id: binding.emitterId, patch });
      }
    }
    return patches;
  }

  #deformationSource(primitiveId: string): SkinnedMeshPrimitive | undefined {
    const base = this.#basePrimitiveById.get(primitiveId);
    if (isSkinnedMesh(base)) return base;
    return this.#instancingBindingsByPrimitiveId.get(primitiveId)?.deformationSource;
  }

  #buildMorphPatch(
    primitiveId: string,
    morphWeights: Float32Array,
  ): ScenePrimitivePatch | undefined {
    const source = this.#deformationSource(primitiveId);
    if (!source?.morphTargets || source.morphTargets.length === 0) return undefined;
    if (isSkinnedMesh(this.#basePrimitiveById.get(primitiveId))) {
      return { morphWeights };
    }
    const solved = solveSkin({ ...source, morphWeights });
    return solvedInstancedDeformationPatch(source, solved);
  }

  #buildSkinPatch(
    primitiveId: string,
    binding: SkinBinding,
    worldTransforms: ReadonlyMap<number, Mat4>,
    sampledMorphWeights: Float32Array | undefined,
    frame: GltfSceneControllerDiagnosticFrame,
  ): ScenePrimitivePatch | undefined {
    const source = this.#deformationSource(primitiveId);
    if (!source) return undefined;

    const meshWorld = worldTransforms.get(binding.meshNodeIndex);
    const meshWorldInverse = meshWorld ? mat4Invert(meshWorld) : null;
    if (!meshWorldInverse) {
      emitControllerDiagnostic(frame, {
        code: 'animation-skin-mesh-transform-noninvertible',
        path: `scene.primitives["${primitiveId}"].transform`,
        message:
          `[vitrum/gltf-adapter] Animation skin patch for primitive "${primitiveId}" ` +
          'could not invert the skinned mesh node transform; skin channel skipped.',
        primitiveId,
        nodeIndex: binding.meshNodeIndex,
      });
      return undefined;
    }

    const bones = new Float32Array(binding.jointNodeIndices.length * 16);
    for (const [jointOffset, jointNodeIndex] of binding.jointNodeIndices.entries()) {
      const jointWorld = worldTransforms.get(jointNodeIndex);
      const outOffset = jointOffset * 16;
      if (jointWorld) {
        bones.set(mat4Mul(meshWorldInverse, jointWorld), outOffset);
      } else {
        bones.set(IDENTITY_MAT4, outOffset);
        emitControllerDiagnostic(frame, {
          code: 'animation-skin-joint-unreachable',
          path: `skins.joints[${jointOffset}]`,
          message:
            `[vitrum/gltf-adapter] Skin for primitive "${primitiveId}" references joint node ` +
            `${jointNodeIndex}, which is not reachable from the imported scene; identity bone used.`,
          primitiveId,
          jointNodeIndex,
        });
      }
    }

    const morphWeights =
      source.morphTargets && source.morphTargets.length > 0
        ? (sampledMorphWeights ??
          source.morphWeights ??
          new Float32Array(source.morphTargets.length))
        : undefined;
    const basePrimitive = this.#basePrimitiveById.get(primitiveId);
    if (isSkinnedMesh(basePrimitive)) {
      const currentPrimitive = findPrimitive(this.#scene, primitiveId);
      if (
        currentPrimitive?.kind === 'skinned-mesh' &&
        arrayElementsEqual(currentPrimitive.bones, bones) &&
        morphWeights === undefined
      ) {
        return undefined;
      }
      return {
        bones,
        ...(morphWeights ? { morphWeights } : {}),
      };
    }

    // Instanced primitives cannot retain bones/morph weights. Resolve their
    // private deformation source once and patch the shared render attributes.
    const solved = solveSkin({
      ...source,
      bones,
      ...(morphWeights ? { morphWeights } : {}),
    });
    return solvedInstancedDeformationPatch(source, solved);
  }

  #applyMaterialPointerSamples(
    samples: readonly ResolvedPointerSample[],
    patchMap: Map<string, ScenePrimitivePatch>,
    frame: GltfSceneControllerDiagnosticFrame,
  ): Set<string> {
    const animatedPrimitiveIds = new Set<string>();
    if (samples.length === 0) return animatedPrimitiveIds;
    const animatedMaterials = new Map<number, MaterialSpec>();
    const touchedMaterials = new Set<number>();

    for (const { sample, target } of samples) {
      if (target.kind !== 'material-property' && target.kind !== 'material-texture-transform')
        continue;

      const base =
        animatedMaterials.get(target.materialIndex) ??
        this.#convertedMaterials[target.materialIndex];
      if (!base) {
        emitControllerDiagnostic(frame, {
          code: 'animation-pointer-material-missing',
          path: `materials[${target.materialIndex}]`,
          message:
            `[vitrum/gltf-adapter] GltfSceneController.${frame.caller}: ` +
            `KHR_animation_pointer target "${target.pointer}" references missing converted material ` +
            `${target.materialIndex}; channel skipped.`,
          materialIndex: target.materialIndex,
        });
        continue;
      }

      animatedMaterials.set(
        target.materialIndex,
        applyGltfMaterialAnimationPointerValue(base, target, sample.value),
      );
      touchedMaterials.add(target.materialIndex);
    }

    for (const materialIndex of touchedMaterials) {
      const material = animatedMaterials.get(materialIndex);
      if (!material) continue;
      const thicknessRange = material.iridescenceThicknessRange;
      if (
        thicknessRange !== undefined &&
        !(thicknessRange[0] >= 0 && thicknessRange[0] <= thicknessRange[1])
      ) {
        emitControllerDiagnostic(frame, {
          code: 'animation-pointer-value-invalid',
          path: `materials[${materialIndex}].extensions.KHR_materials_iridescence`,
          message:
            `[vitrum/gltf-adapter] GltfSceneController.${frame.caller}: sampled iridescence ` +
            `thickness range [${thicknessRange[0]}, ${thicknessRange[1]}] does not satisfy ` +
            '0 <= minimum <= maximum. Material pointer updates skipped without clamping.',
          materialIndex,
        });
        continue;
      }
      const primitiveIds = this.#materialBindingsByMaterialIndex.get(materialIndex) ?? [];
      if (primitiveIds.length === 0) {
        emitControllerDiagnostic(frame, {
          code: 'animation-pointer-material-unmapped',
          path: `materials[${materialIndex}]`,
          message:
            `[vitrum/gltf-adapter] GltfSceneController.${frame.caller}: ` +
            `KHR_animation_pointer animated material ${materialIndex}, but no imported primitive currently ` +
            'uses that material.',
          materialIndex,
        });
        continue;
      }

      for (const id of primitiveIds) {
        if (!this.#basePrimitiveById.has(id)) continue;
        animatedPrimitiveIds.add(id);
        mergePrimitivePatch(patchMap, id, {
          material: materialReplacementPatch(material),
        });
      }
    }
    return animatedPrimitiveIds;
  }
}

function baseLocalState(node: GltfNode): NodeLocalState {
  if (node.matrix) {
    const matrix = new Float32Array(node.matrix);
    try {
      const decomposed = decomposeAffineMatrix(matrix);
      const composedBase = composeTrsMat4(
        decomposed.translation,
        decomposed.rotation,
        decomposed.scale,
      );
      const inverseBase = mat4Invert(composedBase);
      if (inverseBase === null) {
        throw new Error(
          '[vitrum/gltf-adapter] Matrix-authored node could not be decomposed into an invertible TRS base.',
        );
      }
      return {
        ...decomposed,
        matrix,
        matrixResidual: mat4Mul(inverseBase, matrix),
        matrixOverridden: false,
      };
    } catch {
      // A zero-scale/singular matrix is still a valid authored local transform.
      // Preserve it verbatim so unrelated nodes and non-TRS animation continue
      // to work; #applySampleToLocals diagnoses and skips only a TRS channel
      // that actually targets this node.
      return {
        translation: [matrix[12] ?? 0, matrix[13] ?? 0, matrix[14] ?? 0],
        rotation: [0, 0, 0, 1],
        scale: [1, 1, 1],
        matrix,
        matrixTrsUnavailable: true,
        matrixOverridden: false,
      };
    }
  }
  return {
    translation: node.translation ? [...node.translation] : [0, 0, 0],
    rotation: node.rotation ? [...node.rotation] : [0, 0, 0, 1],
    scale: node.scale ? [...node.scale] : [1, 1, 1],
    matrixOverridden: false,
  };
}

function cloneLocalStates(states: readonly NodeLocalState[]): NodeLocalState[] {
  return states.map((state) => ({
    translation: [...state.translation],
    rotation: [...state.rotation],
    scale: [...state.scale],
    ...(state.matrix ? { matrix: new Float32Array(state.matrix) } : {}),
    ...(state.matrixResidual ? { matrixResidual: new Float32Array(state.matrixResidual) } : {}),
    ...(state.matrixTrsUnavailable ? { matrixTrsUnavailable: true } : {}),
    matrixOverridden: false,
  }));
}

function localMatrixForState(
  state: NodeLocalState | undefined,
  node: GltfNode | undefined,
): Float32Array {
  if (!state) return node ? nodeLocalMatrix(node) : new Float32Array(IDENTITY_MAT4);
  if (state.matrix && !state.matrixOverridden) return new Float32Array(state.matrix);
  const composed = composeTrsMat4(state.translation, state.rotation, state.scale);
  return state.matrixResidual ? mat4Mul(composed, state.matrixResidual) : composed;
}

function decomposeAffineMatrix(
  matrix: ArrayLike<number>,
): Pick<NodeLocalState, 'translation' | 'rotation' | 'scale'> {
  const translation: [number, number, number] = [matrix[12] ?? 0, matrix[13] ?? 0, matrix[14] ?? 0];
  const c0: [number, number, number] = [matrix[0] ?? 0, matrix[1] ?? 0, matrix[2] ?? 0];
  const c1: [number, number, number] = [matrix[4] ?? 0, matrix[5] ?? 0, matrix[6] ?? 0];
  const c2: [number, number, number] = [matrix[8] ?? 0, matrix[9] ?? 0, matrix[10] ?? 0];
  const basisScale = Math.max(
    Math.abs(c0[0]),
    Math.abs(c0[1]),
    Math.abs(c0[2]),
    Math.abs(c1[0]),
    Math.abs(c1[1]),
    Math.abs(c1[2]),
    Math.abs(c2[0]),
    Math.abs(c2[1]),
    Math.abs(c2[2]),
  );
  if (!(basisScale > 0) || !Number.isFinite(basisScale)) {
    throw new Error(
      '[vitrum/gltf-adapter] Matrix-authored node has a degenerate basis and cannot preserve TRS animation.',
    );
  }
  const n0: [number, number, number] = [c0[0] / basisScale, c0[1] / basisScale, c0[2] / basisScale];
  const n1: [number, number, number] = [c1[0] / basisScale, c1[1] / basisScale, c1[2] / basisScale];
  const n2: [number, number, number] = [c2[0] / basisScale, c2[1] / basisScale, c2[2] / basisScale];
  const sxNormalized = Math.hypot(...n0);
  if (!(sxNormalized > 1e-12) || !Number.isFinite(sxNormalized)) {
    throw new Error(
      '[vitrum/gltf-adapter] Matrix-authored node has a degenerate X basis and cannot preserve TRS animation.',
    );
  }
  const x: [number, number, number] = [
    n0[0] / sxNormalized,
    n0[1] / sxNormalized,
    n0[2] / sxNormalized,
  ];
  const xy = x[0] * n1[0] + x[1] * n1[1] + x[2] * n1[2];
  const yCandidate: [number, number, number] = [
    n1[0] - x[0] * xy,
    n1[1] - x[1] * xy,
    n1[2] - x[2] * xy,
  ];
  const syNormalized = Math.hypot(...yCandidate);
  if (!(syNormalized > 1e-12) || !Number.isFinite(syNormalized)) {
    throw new Error(
      '[vitrum/gltf-adapter] Matrix-authored node has a degenerate Y basis and cannot preserve TRS animation.',
    );
  }
  const y: [number, number, number] = [
    yCandidate[0] / syNormalized,
    yCandidate[1] / syNormalized,
    yCandidate[2] / syNormalized,
  ];
  const z: [number, number, number] = [
    x[1] * y[2] - x[2] * y[1],
    x[2] * y[0] - x[0] * y[2],
    x[0] * y[1] - x[1] * y[0],
  ];
  const szNormalized = z[0] * n2[0] + z[1] * n2[1] + z[2] * n2[2];
  if (!(Math.abs(szNormalized) > 1e-12) || !Number.isFinite(szNormalized)) {
    throw new Error(
      '[vitrum/gltf-adapter] Matrix-authored node has a degenerate Z basis and cannot preserve TRS animation.',
    );
  }
  return {
    translation,
    rotation: quaternionFromRotationColumns(x, y, z),
    scale: [sxNormalized * basisScale, syNormalized * basisScale, szNormalized * basisScale],
  };
}

function quaternionFromRotationColumns(
  x: readonly [number, number, number],
  y: readonly [number, number, number],
  z: readonly [number, number, number],
): [number, number, number, number] {
  const m00 = x[0],
    m01 = y[0],
    m02 = z[0];
  const m10 = x[1],
    m11 = y[1],
    m12 = z[1];
  const m20 = x[2],
    m21 = y[2],
    m22 = z[2];
  const trace = m00 + m11 + m22;
  let q: [number, number, number, number];
  if (trace > 0) {
    const s = Math.sqrt(trace + 1) * 2;
    q = [(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, s / 4];
  } else if (m00 > m11 && m00 > m22) {
    const s = Math.sqrt(1 + m00 - m11 - m22) * 2;
    q = [s / 4, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s];
  } else if (m11 > m22) {
    const s = Math.sqrt(1 + m11 - m00 - m22) * 2;
    q = [(m01 + m10) / s, s / 4, (m12 + m21) / s, (m02 - m20) / s];
  } else {
    const s = Math.sqrt(1 + m22 - m00 - m11) * 2;
    q = [(m02 + m20) / s, (m12 + m21) / s, s / 4, (m10 - m01) / s];
  }
  return normalizeQuat(q);
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

function invertNodePrimitiveBindings(
  bindings: ReadonlyMap<number, readonly string[]>,
): Map<string, number> {
  const out = new Map<string, number>();
  for (const [nodeIndex, primitiveIds] of bindings) {
    for (const primitiveId of primitiveIds) out.set(primitiveId, nodeIndex);
  }
  return out;
}

function nodeOwnVisibility(node: GltfNode): boolean {
  return node.extensions?.KHR_node_visibility?.visible !== false;
}

function baseMorphWeightsForNode(gltf: GltfJson, nodeIndex: number): Float32Array | undefined {
  const node = gltf.nodes?.[nodeIndex];
  if (!node || node.mesh === undefined) return undefined;
  const mesh = gltf.meshes?.[node.mesh];
  const targetCount = mesh?.primitives[0]?.targets?.length ?? 0;
  if (targetCount <= 0) return undefined;
  const source = node.weights ?? mesh?.weights ?? [];
  const result = new Float32Array(targetCount);
  for (let index = 0; index < targetCount; index += 1) result[index] = source[index] ?? 0;
  return result;
}

function buildEffectiveNodeVisibilityForController(
  gltf: GltfJson,
  rootNodeIndices: readonly number[],
  directVisibility: readonly boolean[],
): Map<number, boolean> {
  const result = new Map<number, boolean>();
  const nodes = gltf.nodes ?? [];
  const stack = rootNodeIndices.map((nodeIndex) => ({ nodeIndex, parentVisible: true }));
  while (stack.length > 0) {
    const { nodeIndex, parentVisible } = stack.pop()!;
    const node = nodes[nodeIndex];
    if (!node || result.has(nodeIndex)) continue;
    const visible = parentVisible && (directVisibility[nodeIndex] ?? true);
    result.set(nodeIndex, visible);
    for (const child of node.children ?? []) {
      stack.push({ nodeIndex: child, parentVisible: visible });
    }
  }
  return result;
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

function buildMaterialBindings(bindings: readonly GltfMaterialBinding[]): Map<number, string[]> {
  const out = new Map<number, string[]>();
  for (const binding of bindings) {
    const existing = out.get(binding.materialIndex);
    if (existing) {
      existing.push(binding.primitiveId);
    } else {
      out.set(binding.materialIndex, [binding.primitiveId]);
    }
  }
  return out;
}

function buildMaterialIndexByPrimitiveId(
  bindings: readonly GltfMaterialBinding[],
): Map<string, number> {
  return new Map(bindings.map((binding) => [binding.primitiveId, binding.materialIndex]));
}

function parseAnimationNodeIndex(nodeId: string): number | undefined {
  if (!nodeId.startsWith(NODE_ID_PREFIX)) return undefined;
  const raw = nodeId.slice(NODE_ID_PREFIX.length);
  if (!/^\d+$/.test(raw)) return undefined;
  return Number(raw);
}

function resolveMaterialVariantSelection(
  gltf: GltfJson,
  selector: string | number | undefined,
  frame: GltfSceneControllerDiagnosticFrame,
): number | undefined {
  if (selector === undefined) return undefined;
  const variantList = gltf.extensions?.KHR_materials_variants?.variants;
  if (variantList !== undefined && !Array.isArray(variantList)) {
    emitControllerDiagnostic(frame, {
      code: 'variant-list-malformed',
      path: 'extensions.KHR_materials_variants.variants',
      message:
        '[vitrum/gltf-adapter] GltfSceneController.setVariant: materialVariant was requested, ' +
        'but extensions.KHR_materials_variants.variants is missing or malformed. Base materials are used.',
    });
    return undefined;
  }
  const variants = variantList ?? [];
  if (typeof selector === 'number') {
    if (Number.isInteger(selector) && selector >= 0 && selector < variants.length) return selector;
    emitControllerDiagnostic(frame, {
      code: 'variant-selection-not-found',
      path: 'extensions.KHR_materials_variants.variants',
      message:
        `[vitrum/gltf-adapter] GltfSceneController.setVariant: materialVariant index ${selector} ` +
        `was requested, but this asset declares ${variants.length} variant(s). Base materials are used.`,
      variantIndex: selector,
    });
    return undefined;
  }
  const index = variants.findIndex((variant) => variant.name === selector);
  if (index >= 0) return index;
  emitControllerDiagnostic(frame, {
    code: 'variant-selection-not-found',
    path: 'extensions.KHR_materials_variants.variants',
    message:
      `[vitrum/gltf-adapter] GltfSceneController.setVariant: materialVariant "${selector}" was ` +
      'requested, but no KHR_materials_variants entry with that name exists. Base materials are used.',
  });
  return undefined;
}

function resolvePrimitiveMaterialIndex(
  gltf: GltfJson,
  primitive: GltfPrimitive,
  baseMaterialIndex: number | undefined,
  selectedVariantIndex: number | undefined,
  frame: GltfSceneControllerDiagnosticFrame,
  meshLabel: string,
  meshIndex: number,
  primitiveIndex: number,
): number | undefined {
  if (selectedVariantIndex === undefined) return baseMaterialIndex;
  const mappings = primitive.extensions?.KHR_materials_variants?.mappings ?? [];
  for (const [mappingIndex, mapping] of mappings.entries()) {
    if (Array.isArray(mapping.variants)) continue;
    emitControllerDiagnostic(frame, {
      code: 'variant-mapping-malformed',
      path: `meshes[${meshIndex}].primitives[${primitiveIndex}].extensions.KHR_materials_variants.mappings[${mappingIndex}].variants`,
      message:
        `[vitrum/gltf-adapter] GltfSceneController.setVariant: mesh "${meshLabel}" ` +
        `KHR_materials_variants mapping ${mappingIndex} has a missing or malformed variants array. ` +
        'Mapping skipped.',
      meshIndex,
      primitiveIndex,
    });
  }
  let matchedMapping:
    | { readonly mapping: (typeof mappings)[number]; readonly index: number }
    | undefined;
  for (const [mappingIndex, candidate] of mappings.entries()) {
    if (Array.isArray(candidate.variants) && candidate.variants.includes(selectedVariantIndex)) {
      matchedMapping = { mapping: candidate, index: mappingIndex };
      break;
    }
  }
  if (!matchedMapping) return baseMaterialIndex;
  const { mapping, index: mappingIndex } = matchedMapping;
  if (
    !Number.isSafeInteger(mapping.material) ||
    mapping.material < 0 ||
    mapping.material >= (gltf.materials?.length ?? 0)
  ) {
    emitControllerDiagnostic(frame, {
      code: 'variant-mapping-material-missing',
      path: `meshes[${meshIndex}].primitives[${primitiveIndex}].extensions.KHR_materials_variants.mappings[${mappingIndex}].material`,
      message:
        `[vitrum/gltf-adapter] GltfSceneController.setVariant: mesh "${meshLabel}" ` +
        `KHR_materials_variants mapping for variant ${selectedVariantIndex} references missing ` +
        `material ${mapping.material}. Base material is used.`,
      variantIndex: selectedVariantIndex,
      materialIndex: mapping.material,
      meshIndex,
      primitiveIndex,
    });
    return baseMaterialIndex;
  }
  return mapping.material;
}

function materialForIndex(
  materials: readonly MaterialSpec[],
  materialIndex: number | undefined,
  frame: GltfSceneControllerDiagnosticFrame,
): MaterialSpec {
  if (materialIndex === undefined) return GLTF_DEFAULT_MATERIAL;
  const material = materials[materialIndex];
  if (material) return material;
  emitControllerDiagnostic(frame, {
    code: 'variant-material-index-missing',
    path: `convertedMaterials[${materialIndex}]`,
    message:
      `[vitrum/gltf-adapter] GltfSceneController.setVariant: converted material ${materialIndex} ` +
      'is missing. The glTF default material is used.',
    materialIndex,
  });
  return GLTF_DEFAULT_MATERIAL;
}

const MATERIAL_REPLACEMENT_CLEAR_FIELDS = [
  'emissive',
  'emissiveIntensity',
  'shadingModel',
  'alphaMode',
  'alphaCutoff',
  'opacity',
  'doubleSided',
  'transmission',
  'ior',
  'attenuationColor',
  'attenuationDistance',
  'thickness',
  'baseColorMap',
  'normalMap',
  'normalScale',
  'roughnessMap',
  'metallicMap',
  'transmissionMap',
  'thicknessMap',
  'emissiveMap',
  'alphaMap',
  'aoMap',
  'aoMapIntensity',
  'clearcoatMap',
  'clearcoatRoughnessMap',
  'clearcoatNormalMap',
  'clearcoatNormalScale',
  'sheenColorMap',
  'sheenRoughnessMap',
  'iridescenceMap',
  'iridescenceThicknessMap',
  'anisotropyMap',
  'specularColorMap',
  'specularIntensityMap',
  'bumpMap',
  'bumpScale',
  'displacementMap',
  'displacementScale',
  'displacementBias',
  'displacementSubdivisions',
  'lightMap',
  'lightMapIntensity',
  'sheen',
  'sheenColor',
  'sheenRoughness',
  'clearcoat',
  'clearcoatRoughness',
  'iridescence',
  'iridescenceIor',
  'iridescenceThicknessRange',
  'specularIntensity',
  'specularColor',
  'envMapIntensity',
  'spectralAttenuation',
  'dispersionAbbeNumber',
  'scatteringCoefficient',
  'scatteringAnisotropy',
  'scatteringCoefficientRGB',
  'frontLayer',
  'backLayer',
  'thinFilmStack',
  'anisotropy',
  'anisotropyRotation',
  'extensions',
] as const satisfies readonly (keyof MaterialSpec)[];

const MATERIAL_TEXTURE_REF_FIELDS = [
  'baseColorMap',
  'normalMap',
  'roughnessMap',
  'metallicMap',
  'transmissionMap',
  'thicknessMap',
  'emissiveMap',
  'alphaMap',
  'aoMap',
  'clearcoatMap',
  'clearcoatRoughnessMap',
  'clearcoatNormalMap',
  'sheenColorMap',
  'sheenRoughnessMap',
  'iridescenceMap',
  'iridescenceThicknessMap',
  'anisotropyMap',
  'specularColorMap',
  'specularIntensityMap',
  'bumpMap',
  'displacementMap',
  'lightMap',
] as const satisfies readonly (keyof MaterialSpec)[];

function surfaceLayerReplacementPatch(
  layer: NonNullable<MaterialSpec['frontLayer']>,
): SurfaceAbsorptionLayerPatch {
  return {
    roughness: undefined,
    normalMap: undefined,
    normalScale: undefined,
    ...layer,
  };
}

function materialReplacementPatch(material: MaterialSpec): MaterialSpecPatch {
  const patch: Record<string, unknown> = {};
  for (const field of MATERIAL_REPLACEMENT_CLEAR_FIELDS) patch[field] = undefined;
  Object.assign(patch, material);
  if (material.frontLayer != null) {
    patch.frontLayer = surfaceLayerReplacementPatch(material.frontLayer);
  }
  if (material.backLayer != null) {
    patch.backLayer = surfaceLayerReplacementPatch(material.backLayer);
  }
  return patch;
}

function isTextureRef(value: unknown): value is TextureRef {
  return value !== null && typeof value === 'object' && 'handle' in value;
}

function textureRefWithRouting(handleRef: TextureRef, routingRef: TextureRef): TextureRef {
  return {
    ...routingRef,
    handle: handleRef.handle,
  };
}

function materialForVariantPatch(
  material: MaterialSpec,
  patch: GltfMaterialVariantPrimitivePatch,
): MaterialSpec {
  if (patch.materialRouting === undefined) return material;
  const routed: Record<string, unknown> = { ...material };
  for (const field of MATERIAL_TEXTURE_REF_FIELDS) {
    const routedRef = patch.materialRouting[field];
    const liveRef = material[field];
    if (isTextureRef(routedRef)) {
      routed[field] = isTextureRef(liveRef) ? textureRefWithRouting(liveRef, routedRef) : routedRef;
    }
  }
  return routed as unknown as MaterialSpec;
}

function materialVariantPatchForSelection(
  binding: GltfMaterialVariantBinding,
  selectedVariantIndex: number | undefined,
): GltfMaterialVariantPrimitivePatch | undefined {
  if (selectedVariantIndex === undefined) return binding.basePatch;
  return (
    binding.variantPatches?.find((patch) => patch.variantIndex === selectedVariantIndex)?.patch ??
    binding.basePatch
  );
}

function scenePrimitivePatchForMaterialVariant(
  patch: GltfMaterialVariantPrimitivePatch,
  material: MaterialSpec,
): ScenePrimitivePatch {
  return {
    material: materialReplacementPatch(material),
    uvs: patch.uvs,
    uv1: patch.uv1,
    uvSets: patch.uvSets,
    tangents: patch.tangents,
  };
}

function normalizeBlendWeights(weights: readonly number[]): number[] {
  const positive = weights.map((weight) => (Number.isFinite(weight) && weight > 0 ? weight : 0));
  const sum = positive.reduce((acc, weight) => acc + weight, 0);
  if (sum <= 0) {
    throw new Error(
      '[vitrum/gltf-adapter] GltfSceneController.blend: at least one weight must be positive.',
    );
  }
  return positive.map((weight) => weight / sum);
}

function blendSampledChannels(
  perClipSamples: readonly (readonly SampledChannel[])[],
  weights: readonly number[],
): SampledChannel[] {
  const accumulators = new Map<
    string,
    {
      node: SampledChannel['node'];
      path: SampledChannel['path'];
      pointer?: string;
      value: Float32Array;
      weightSum: number;
      quaternion: boolean;
      referenceQuat?: Float32Array;
    }
  >();

  for (let clipIndex = 0; clipIndex < perClipSamples.length; clipIndex += 1) {
    const weight = weights[clipIndex] ?? 0;
    if (weight <= 0) continue;
    for (const sample of perClipSamples[clipIndex] ?? []) {
      const key = `${sample.node}\u0000${sample.path}\u0000${sample.pointer ?? ''}`;
      const quaternion = sampledChannelIsNodeRotation(sample);
      let acc = accumulators.get(key);
      if (!acc) {
        acc = {
          node: sample.node,
          path: sample.path,
          ...(sample.pointer !== undefined ? { pointer: sample.pointer } : {}),
          value: new Float32Array(sample.value.length),
          weightSum: 0,
          quaternion,
          ...(quaternion ? { referenceQuat: new Float32Array(sample.value) } : {}),
        };
        accumulators.set(key, acc);
      }
      if (sample.value.length > acc.value.length) {
        const grown = new Float32Array(sample.value.length);
        grown.set(acc.value);
        acc.value = grown;
      }

      const sign =
        acc.quaternion && acc.referenceQuat && quatDot(acc.referenceQuat, sample.value) < 0
          ? -1
          : 1;
      for (let i = 0; i < sample.value.length; i += 1) {
        acc.value[i] = (acc.value[i] ?? 0) + (sample.value[i] ?? 0) * weight * sign;
      }
      acc.weightSum += weight;
    }
  }

  const out: SampledChannel[] = [];
  for (const acc of accumulators.values()) {
    if (acc.weightSum <= 0) continue;
    const value = new Float32Array(acc.value.length);
    if (acc.quaternion) {
      value.set(acc.value);
      normalizeQuatInPlace(value);
    } else {
      for (let i = 0; i < acc.value.length; i += 1) {
        value[i] = (acc.value[i] ?? 0) / acc.weightSum;
      }
    }
    out.push({
      node: acc.node,
      path: acc.path,
      ...(acc.pointer !== undefined ? { pointer: acc.pointer } : {}),
      value,
    });
  }
  return out;
}

function sampledChannelIsNodeRotation(sample: SampledChannel): boolean {
  if (sample.path === 'rotation') return true;
  if (sample.path !== 'pointer') return false;
  const target = resolveGltfAnimationPointer(sample.pointer);
  return target?.kind === 'node' && target.path === 'rotation';
}

function quatDot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  return (
    (a[0] ?? 0) * (b[0] ?? 0) +
    (a[1] ?? 0) * (b[1] ?? 0) +
    (a[2] ?? 0) * (b[2] ?? 0) +
    (a[3] ?? 1) * (b[3] ?? 1)
  );
}

function normalizeQuatInPlace(value: Float32Array): void {
  const len = Math.hypot(value[0] ?? 0, value[1] ?? 0, value[2] ?? 0, value[3] ?? 1);
  if (len > 1e-8 && Number.isFinite(len)) {
    value[0] = (value[0] ?? 0) / len;
    value[1] = (value[1] ?? 0) / len;
    value[2] = (value[2] ?? 0) / len;
    value[3] = (value[3] ?? 1) / len;
  } else {
    value[0] = 0;
    value[1] = 0;
    value[2] = 0;
    value[3] = 1;
  }
}

function normalizeClipTime(clip: AnimationClip, time: number, loop: boolean): number {
  if (!Number.isFinite(time)) return 0;
  const duration = clip.duration;
  if (duration <= 0 || !Number.isFinite(duration)) return 0;
  if (loop) return ((time % duration) + duration) % duration;
  return Math.max(0, Math.min(duration, time));
}

function readVec3(
  value: Float32Array,
  fallback: [number, number, number],
): [number, number, number] {
  return [value[0] ?? fallback[0], value[1] ?? fallback[1], value[2] ?? fallback[2]];
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
  map: Map<string, ScenePrimitivePatch>,
  id: string,
  patch: ScenePrimitivePatch,
): void {
  map.set(id, { ...(map.get(id) ?? {}), ...patch } as ScenePrimitivePatch);
}

/**
 * Instanced primitives cannot retain authored skin/morph state, so every
 * controller tick publishes a complete renderer-facing deformation surface.
 * `solveSkin()` intentionally omits UV/color outputs for an all-zero morph;
 * falling back to the source lanes here prevents the preceding active morph's
 * allocations from surviving through shallow scene-patch application.
 */
function solvedInstancedDeformationPatch(
  source: SkinnedMeshPrimitive,
  solved: ReturnType<typeof solveSkin>,
): ScenePrimitivePatch {
  return {
    positions: solved.positions,
    normals: solved.normals,
    tangents: solved.tangents ?? source.tangents,
    uvs: solved.uvs ?? source.uvs ?? source.uvSets?.[0],
    uv1: solved.uv1 ?? source.uv1 ?? source.uvSets?.[1],
    uvSets: solved.uvSets ?? source.uvSets,
    colors: solved.colors ?? source.colors ?? source.colorSets?.[0],
    colorSets: solved.colorSets ?? source.colorSets,
  };
}

function findPrimitive(scene: Scene, id: string): ScenePrimitive | undefined {
  return scene.primitives.find((primitive) => String(primitive.id) === id);
}

function findEmitter(scene: Scene, id: string): SceneEmitter | undefined {
  return scene.emitters.find((emitter) => String(emitter.id) === id);
}

function applyCameraPointerSamples(
  cameras: readonly GltfSceneCamera[],
  samples: readonly ResolvedPointerSample[],
  frame: GltfSceneControllerDiagnosticFrame,
): GltfSceneCamera[] {
  const cameraSamples = samples.filter((entry) => entry.target.kind === 'camera');
  const matchedPointers = new Set<string>();
  const result = cameras.map((camera) => {
    const matching = cameraSamples.filter(
      ({ target }) =>
        target.kind === 'camera' &&
        camera.cameraIndex === target.cameraIndex &&
        camera.type === target.cameraType,
    );
    if (matching.length === 0) return camera;
    let candidate = camera;
    for (const { sample, target } of matching) {
      if (target.kind !== 'camera') continue;
      matchedPointers.add(target.pointer);
      const value = sample.value[0]!;
      candidate =
        target.cameraType === 'perspective'
          ? {
              ...candidate,
              perspective: { ...(candidate.perspective ?? {}), [target.field]: value },
            }
          : {
              ...candidate,
              orthographic: { ...(candidate.orthographic ?? {}), [target.field]: value },
            };
    }
    const projection =
      candidate.type === 'perspective' ? candidate.perspective : candidate.orthographic;
    if (
      projection?.znear !== undefined &&
      projection.zfar !== undefined &&
      !(projection.zfar > projection.znear)
    ) {
      emitControllerDiagnostic(frame, {
        code: 'animation-pointer-value-invalid',
        path: `cameras[${camera.cameraIndex}].${camera.type}`,
        message:
          `[vitrum/gltf-adapter] GltfSceneController.${frame.caller}: sampled camera projection ` +
          `has zfar=${projection.zfar} but znear=${projection.znear}; zfar must be greater than ` +
          'znear. Camera pointer updates skipped without clamping.',
      });
      return camera;
    }
    return candidate;
  });
  for (const { target } of cameraSamples) {
    if (target.kind === 'camera' && !matchedPointers.has(target.pointer)) {
      emitControllerDiagnostic(frame, {
        code: 'animation-pointer-camera-unmapped',
        path: `cameras[${target.cameraIndex}].${target.cameraType}.${target.field}`,
        message:
          `[vitrum/gltf-adapter] GltfSceneController.${frame.caller}: camera pointer ` +
          `"${target.pointer}" has no camera instance in the imported scene; channel skipped.`,
      });
    }
  }
  return result;
}

function punctualEmitterPointerPatch(
  emitter: SceneEmitter,
  samples: readonly ResolvedPointerSample[],
  frame: GltfSceneControllerDiagnosticFrame,
): Partial<SceneEmitter> {
  const patch: Record<string, unknown> = {};
  let outerCone = emitter.kind === 'spot' ? emitter.angle : 0;
  let innerCone = emitter.kind === 'spot' ? emitter.angle * (1 - (emitter.penumbra ?? 0)) : 0;
  let coneChanged = false;
  for (const { sample, target } of samples) {
    if (target.kind !== 'punctual-light') continue;
    if (target.field === 'color') {
      patch.color = [sample.value[0]!, sample.value[1]!, sample.value[2]!];
    } else if (target.field === 'intensity') {
      patch.intensity = sample.value[0]!;
    } else if (target.field === 'range' && (emitter.kind === 'point' || emitter.kind === 'spot')) {
      patch.distance = sample.value[0]!;
    } else if (target.field === 'spotInnerConeAngle' && emitter.kind === 'spot') {
      innerCone = sample.value[0]!;
      coneChanged = true;
    } else if (target.field === 'spotOuterConeAngle' && emitter.kind === 'spot') {
      outerCone = sample.value[0]!;
      coneChanged = true;
    }
  }
  if (emitter.kind === 'spot' && coneChanged) {
    if (innerCone <= outerCone) {
      patch.angle = outerCone;
      patch.penumbra = 1 - innerCone / outerCone;
    } else {
      emitControllerDiagnostic(frame, {
        code: 'animation-pointer-value-invalid',
        path: 'extensions.KHR_lights_punctual.lights.spot',
        message:
          `[vitrum/gltf-adapter] GltfSceneController.${frame.caller}: sampled spot-light ` +
          `innerConeAngle=${innerCone} exceeds outerConeAngle=${outerCone}. Cone pointer updates ` +
          'skipped without clamping.',
      });
    }
  }
  return patch;
}

function emitterDifferencePatch(
  desired: SceneEmitter,
  current: SceneEmitter,
): Partial<SceneEmitter> {
  const patch: Record<string, unknown> = {};
  const desiredRecord = desired as unknown as Record<string, unknown>;
  const currentRecord = current as unknown as Record<string, unknown>;
  for (const key of Object.keys(currentRecord)) {
    if (
      key !== 'id' &&
      key !== 'kind' &&
      !Object.prototype.hasOwnProperty.call(desiredRecord, key)
    ) {
      patch[key] = undefined;
    }
  }
  for (const [key, next] of Object.entries(desiredRecord)) {
    if (key === 'id' || key === 'kind') continue;
    const previous = currentRecord[key];
    if (Array.isArray(previous) && Array.isArray(next)) {
      if (!arrayElementsEqual(previous, next)) patch[key] = next;
    } else if (!Object.is(previous, next)) {
      patch[key] = next;
    }
  }
  return patch;
}

function setsEqual<T>(a: ReadonlySet<T>, b: ReadonlySet<T>): boolean {
  if (a.size !== b.size) return false;
  for (const value of a) if (!b.has(value)) return false;
  return true;
}

function punctualEmitterPosePatch(
  emitter: SceneEmitter,
  world: Mat4,
): Partial<SceneEmitter> | undefined {
  const position: [number, number, number] = [world[12] ?? 0, world[13] ?? 0, world[14] ?? 0];
  const forward = normalizeDirection([-(world[8] ?? 0), -(world[9] ?? 0), -(world[10] ?? 0)]);
  if (emitter.kind === 'point') return { position };
  if (emitter.kind === 'spot') return { position, direction: forward };
  if (emitter.kind === 'directional') {
    return { direction: [-forward[0], -forward[1], -forward[2]] };
  }
  return undefined;
}

function normalizeDirection(value: readonly [number, number, number]): [number, number, number] {
  const scale = Math.max(
    Math.abs(value[0]),
    Math.abs(value[1]),
    Math.abs(value[2]),
  );
  if (!(scale > 0) || !Number.isFinite(scale)) return [0, 0, 1];
  const x = value[0] / scale;
  const y = value[1] / scale;
  const z = value[2] / scale;
  const length = Math.hypot(x, y, z);
  return [x / length, y / length, z / length];
}

function primitiveTransform(primitive: ScenePrimitive): Mat4 | undefined {
  if (primitive.kind === 'instanced-mesh') return undefined;
  return primitive.transform;
}

function buildAnimatedInstanceTransforms(
  nodeWorld: Mat4,
  localInstances: ReadonlyArray<Mat4>,
): readonly Mat4[] {
  return localInstances.map((local) => asMat4(mat4Mul(nodeWorld, local)));
}

function instanceMatricesEqual(a: ReadonlyArray<Mat4>, b: ReadonlyArray<Mat4>): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) {
    const ai = a[i];
    const bi = b[i];
    if (!ai || !bi || !arrayElementsEqual(ai, bi)) return false;
  }
  return true;
}

function mat4Equal(
  a: ArrayLike<number> | undefined,
  b: ArrayLike<number>,
): boolean {
  const aa = a ?? IDENTITY_MAT4;
  return arrayElementsEqual(aa, b);
}

function arrayElementsEqual(a: ArrayLike<number>, b: ArrayLike<number>): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) {
    if (!Object.is(a[i] ?? 0, b[i] ?? 0)) return false;
  }
  return true;
}

function isSkinnedMesh(primitive: ScenePrimitive | undefined): primitive is SkinnedMeshPrimitive {
  return primitive?.kind === 'skinned-mesh';
}

function isInstancedMesh(
  primitive: ScenePrimitive | undefined,
): primitive is InstancedMeshPrimitive {
  return primitive?.kind === 'instanced-mesh';
}

function fitMorphWeights(
  weights: Float32Array,
  targetCount: number,
  primitiveId: string,
  frame: GltfSceneControllerDiagnosticFrame,
): Float32Array {
  if (weights.length === targetCount) return new Float32Array(weights);
  const retainedCount = Math.min(weights.length, targetCount);
  const fitted = new Float32Array(targetCount);
  fitted.set(weights.subarray(0, retainedCount));
  emitControllerDiagnostic(frame, {
    code: 'morph-weight-count-mismatch',
    path: `scene.primitives["${primitiveId}"].morphWeights`,
    primitiveId,
    message:
      `[vitrum/gltf-adapter] Animation weights for primitive "${primitiveId}" have ` +
      `length ${weights.length}; expected ${targetCount}. ` +
      (weights.length > targetCount
        ? `The first ${targetCount} weight(s) were retained and the remainder were discarded.`
        : `The ${weights.length} supplied weight(s) were retained and the remainder were zero-filled.`),
  });
  return fitted;
}
