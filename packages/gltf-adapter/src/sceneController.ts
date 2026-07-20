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
  type InstancedMeshPrimitive,
  type MaterialSpec,
  type Mat4,
  type SampledChannel,
  type Scene,
  type ScenePrimitive,
  type SkinnedMeshPrimitive,
  type TextureRef,
} from '@vitrum/core';
import type { GltfJson, GltfNode, GltfPrimitive } from './gltfTypes.js';
import type {
  GltfInstancingBinding,
  GltfMaterialBinding,
  GltfMaterialVariantBinding,
  GltfMaterialVariantPrimitivePatch,
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
  applyGltfMaterialPointerValue,
  resolveGltfMaterialAnimationPointer,
} from './materialPointerAnimation.js';

export interface GltfScenePatchTarget {
  setScene(scene: Scene): void;
  updatePrimitive?(id: string, patch: Partial<ScenePrimitive>): void;
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
  readonly patch: Partial<ScenePrimitive>;
}

export type GltfSceneControllerDiagnosticCode =
  | 'animation-matrix-overridden'
  | 'animation-morph-target-missing'
  | 'animation-skin-joint-unreachable'
  | 'animation-skin-mesh-transform-noninvertible'
  | 'animation-pointer-material-missing'
  | 'animation-pointer-material-unmapped'
  | 'animation-pointer-unsupported'
  | 'animation-target-node-unmapped'
  | 'controller-update-primitive-failed'
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

const NODE_ID_PREFIX = 'gltf-node-';

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

function resetAfterIncrementalPrimitivePatch(target: GltfScenePatchTarget): void {
  target.reset?.();
}

function primitiveResetPatch(primitive: ScenePrimitive): Partial<ScenePrimitive> {
  const patch: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(primitive as unknown as Record<string, unknown>)) {
    if (key === 'id' || key === 'kind') continue;
    patch[key] = value;
  }
  return patch as Partial<ScenePrimitive>;
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
  readonly #basePrimitiveById: Map<string, ScenePrimitive>;
  readonly #convertedMaterials: readonly MaterialSpec[];
  readonly #materialVariantBindings: readonly GltfMaterialVariantBinding[];
  readonly #materialBindingsByMaterialIndex: Map<number, string[]>;
  readonly #materialIndexByPrimitiveId: Map<string, number>;
  readonly #instancingBindingsByPrimitiveId: ReadonlyMap<string, GltfInstancingBinding>;
  readonly #skinBindingsByPrimitiveId: ReadonlyMap<string, SkinBinding>;
  readonly #warnings: string[] = [];
  readonly #diagnostics: GltfSceneControllerDiagnostic[] = [];
  readonly #warnedMatrixOverrideNodes = new Set<number>();

  constructor(input: GltfSceneControllerInput, options: GltfSceneControllerOptions = {}) {
    this.gltf = input.gltf;
    this.animations = input.animations;
    this.animationTargets = input.animationTargets;
    this.#scene = input.scene;
    this.#sceneIndex = input.sceneIndex ?? input.gltf.scene ?? 0;
    this.#baseLocals = (input.gltf.nodes ?? []).map(baseLocalState);
    this.#baseCameras = input.cameras
      ?? collectSceneCameras(
        input.gltf,
        buildWorldTransformsForLocals(input.gltf, this.#rootNodes(), this.#baseLocals),
      );
    this.#cameras = this.#baseCameras;
    this.#nodeToPrimitiveIds = parseAnimationTargets(input.animationTargets);
    this.#basePrimitiveById = new Map(input.scene.primitives.map((p) => [String(p.id), p]));
    this.#convertedMaterials = input.convertedMaterials ?? [];
    this.#materialVariantBindings = input.materialVariantBindings ?? [];
    const materialBindings = input.materialBindings ?? [];
    this.#materialBindingsByMaterialIndex = buildMaterialBindings(materialBindings);
    this.#materialIndexByPrimitiveId = buildMaterialIndexByPrimitiveId(materialBindings);
    this.#instancingBindingsByPrimitiveId = new Map(
      (input.instancingBindings ?? []).map((binding) => [binding.primitiveId, binding]),
    );
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

  play(selector?: GltfClipSelector, options: GltfPlaybackOptions = {}): GltfAnimationApplyResult {
    if (selector !== undefined) {
      this.#activeClip = this.#resolveClip(selector);
    } else if (!this.#activeClip) {
      this.#activeClip = this.#defaultClip('play');
    }
    this.#playing = true;
    const { time, ...applyOptions } = options;
    return this.seek(time ?? this.#clock, applyOptions);
  }

  pause(): void {
    this.#playing = false;
  }

  resume(options: GltfApplyAnimationOptions = {}): GltfAnimationApplyResult {
    if (!this.#activeClip) {
      this.#activeClip = this.#defaultClip('resume');
    }
    this.#playing = true;
    return this.seek(this.#clock, options);
  }

  tick(deltaSeconds: number, options: GltfApplyAnimationOptions = {}): GltfAnimationApplyResult | undefined {
    if (!this.#playing) return undefined;
    return this.advance(deltaSeconds, options);
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
      throw new Error('[vitrum/gltf-adapter] GltfSceneController.blend: at least one clip is required.');
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
    const perClipSamples = clips.map((clip, index) => sampleAnimationClip(clip, localTimes[index] ?? 0));
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
      cameras: applied.cameras,
      scene: applied.scene,
      warnings: frameWarnings,
      diagnostics: frameDiagnostics,
      usedSetScene: applied.usedSetScene,
    };
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
    const patchMap = new Map<string, Partial<ScenePrimitive>>();
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
        if (!findPrimitive(this.#scene, binding.primitiveId)) {
          emitControllerDiagnostic(frame, {
            code: 'variant-primitive-missing-in-scene',
            path: `scene.primitives["${binding.primitiveId}"]`,
            message:
              `[vitrum/gltf-adapter] GltfSceneController.setVariant: primitive "${binding.primitiveId}" ` +
              'is no longer present in the controller scene; patch skipped.',
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
          mergePrimitivePatch(patchMap, binding.primitiveId, scenePrimitivePatchForMaterialVariant(variantPatch, material));
        } else {
          const material = materialForIndex(this.#convertedMaterials, materialIndex, frame);
          mergePrimitivePatch(patchMap, binding.primitiveId, {
            material: materialReplacementPatch(material),
          } as Partial<ScenePrimitive>);
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
      primitivePatches,
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
    if (frameWarnings.length > 0) this.#warnings.push(...frameWarnings);
    if (frameDiagnostics.length > 0) this.#diagnostics.push(...frameDiagnostics);

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
    const nextScene = {
      ...this.#scene,
      primitives: this.#scene.primitives.map((primitive) =>
        this.#basePrimitiveById.get(String(primitive.id)) ?? primitive,
      ),
    };
    this.#scene = nextScene;
    this.#cameras = this.#baseCameras;
    if (options.resetPlayback === true) {
      this.#activeClip = undefined;
      this.#clock = 0;
      this.#playing = false;
    }
    if (!target) return;
    if (options.forceSetScene !== true && target.updatePrimitive) {
      const patches = nextScene.primitives
        .map((primitive, index) => ({
          id: String(primitive.id),
          patch: primitiveResetPatch(primitive),
          changed: previousScene.primitives[index] !== primitive,
        }))
        .filter((entry) => entry.changed);
      if (patches.length === 0) return;
      const frame: GltfSceneControllerDiagnosticFrame = {
        warnings: [],
        diagnostics: [],
        caller: 'resetPose',
      };
      let attemptedPrimitiveId: string | undefined;
      try {
        for (const { id, patch } of patches) {
          attemptedPrimitiveId = id;
          target.updatePrimitive(id, patch);
        }
        resetAfterIncrementalPrimitivePatch(target);
        return;
      } catch (err) {
        const message = errorMessage(err);
        emitControllerDiagnostic(frame, {
          code: 'controller-update-primitive-failed',
          path: attemptedPrimitiveId
            ? `scene.primitives["${attemptedPrimitiveId}"]`
            : 'scene.primitives',
          message:
            `[vitrum/gltf-adapter] GltfSceneController.resetPose: ` +
            `engine.updatePrimitive failed; falling back to setScene(nextScene). ${message}`,
          ...(attemptedPrimitiveId !== undefined ? { primitiveId: attemptedPrimitiveId } : {}),
        });
        this.#warnings.push(...frame.warnings);
        this.#diagnostics.push(...frame.diagnostics);
        target.setScene(this.#scene);
        return;
      }
    }
    target.setScene(this.#scene);
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
    nextScene: Scene,
    frame: GltfSceneControllerDiagnosticFrame,
    forceSetScene: boolean,
  ): boolean {
    if (!(target && primitivePatches.length > 0)) return false;
    if (!forceSetScene && target.updatePrimitive) {
      let attemptedPrimitiveId: string | undefined;
      try {
        for (const { id, patch } of primitivePatches) {
          attemptedPrimitiveId = id;
          target.updatePrimitive(id, patch);
        }
        resetAfterIncrementalPrimitivePatch(target);
        return false;
      } catch (err) {
        const message = errorMessage(err);
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
      throw new Error(`[vitrum/gltf-adapter] GltfSceneController.${caller}: asset has no animations.`);
    }
    return clip;
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
    frame: GltfSceneControllerDiagnosticFrame,
  ): void {
    const nodeIndex = parseAnimationNodeIndex(sample.node);
    if (nodeIndex == null || !locals[nodeIndex]) {
      emitControllerDiagnostic(frame, {
        code: 'animation-target-node-unmapped',
        path: `animations.channels.target.node["${sample.node}"]`,
        message:
          `[vitrum/gltf-adapter] Animation target "${sample.node}" does not map to a glTF node; channel skipped.`,
      });
      return;
    }
    if (sample.path === 'weights') {
      morphWeightsByNode.set(nodeIndex, new Float32Array(sample.value));
      return;
    }

    const state = locals[nodeIndex]!;
    if (state.matrix && !this.#warnedMatrixOverrideNodes.has(nodeIndex)) {
      this.#warnedMatrixOverrideNodes.add(nodeIndex);
      emitControllerDiagnostic(frame, {
        code: 'animation-matrix-overridden',
        path: `nodes[${nodeIndex}].matrix`,
        message:
          `[vitrum/gltf-adapter] Animation targets node ${nodeIndex}, which imported from a matrix. ` +
          'The controller evaluates animated TRS channels over the matrix for this frame.',
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
    cameras: readonly GltfSceneCamera[];
    scene: Scene;
    usedSetScene: boolean;
  } {
    const locals = cloneLocalStates(this.#baseLocals);
    const morphWeightsByNode = new Map<number, Float32Array>();
    const materialPointerSamples: SampledChannel[] = [];

    for (const sample of samples) {
      if (sample.path === 'pointer') {
        materialPointerSamples.push(sample);
        continue;
      }
      this.#applySampleToLocals(sample, locals, morphWeightsByNode, frame);
    }

    const worldTransforms = buildWorldTransformsForLocals(this.gltf, this.#rootNodes(), locals);
    const cameras = (this.gltf.cameras?.length ?? 0) > 0
      ? collectSceneCameras(this.gltf, worldTransforms)
      : this.#baseCameras;
    const patchMap = new Map<string, Partial<ScenePrimitive>>();

    for (const [nodeIndex, primitiveIds] of this.#nodeToPrimitiveIds) {
      const world = worldTransforms.get(nodeIndex);
      if (!world) continue;
      for (const id of primitiveIds) {
        const current = findPrimitive(this.#scene, id);
        if (!current) continue;
        const instancingBinding = this.#instancingBindingsByPrimitiveId.get(id);
        if (instancingBinding && isInstancedMesh(current)) {
          const instances = buildAnimatedInstanceTransforms(world, instancingBinding.localInstanceTransforms);
          if (!instanceMatricesAlmostEqual(current.instances, instances)) {
            mergePrimitivePatch(patchMap, id, { instances } as Partial<ScenePrimitive>);
          }
          continue;
        }
        const expandedInstanceTransform = instancingBinding
          ? buildExpandedInstanceTransform(world, instancingBinding)
          : undefined;
        if (expandedInstanceTransform) {
          if (!mat4AlmostEqual(primitiveTransform(current), expandedInstanceTransform)) {
            mergePrimitivePatch(patchMap, id, { transform: expandedInstanceTransform } as Partial<ScenePrimitive>);
          }
          continue;
        }
        if (!mat4AlmostEqual(primitiveTransform(current), world)) {
          mergePrimitivePatch(patchMap, id, { transform: world } as Partial<ScenePrimitive>);
        }
      }
    }

    for (const [nodeIndex, weights] of morphWeightsByNode) {
      const primitiveIds = this.#nodeToPrimitiveIds.get(nodeIndex) ?? [];
      for (const id of primitiveIds) {
        const patch = this.#buildMorphPatch(id, weights, frame);
        if (patch) mergePrimitivePatch(patchMap, id, patch);
      }
    }

    for (const [id, binding] of this.#skinBindingsByPrimitiveId) {
      const patch = this.#buildSkinPatch(id, binding, worldTransforms, patchMap.get(id), frame);
      if (patch) mergePrimitivePatch(patchMap, id, patch);
    }
    this.#applyMaterialPointerSamples(materialPointerSamples, patchMap, frame);

    const primitivePatches = Array.from(patchMap, ([id, patch]) => ({ id, patch }));
    const target = options.engine ?? this.#engine;
    let nextScene = this.#scene;
    for (const { id, patch } of primitivePatches) {
      nextScene = patchPrimitiveInScene(nextScene, id, patch);
    }

    const usedSetScene = this.#commitSceneChange(
      target,
      primitivePatches,
      nextScene,
      frame,
      options.forceSetScene ?? false,
    );

    this.#scene = nextScene;
    this.#cameras = cameras;
    if (frame.warnings.length > 0) this.#warnings.push(...frame.warnings);
    if (frame.diagnostics.length > 0) this.#diagnostics.push(...frame.diagnostics);

    return {
      primitivePatches,
      cameras,
      scene: this.#scene,
      usedSetScene,
    };
  }

  #buildMorphPatch(
    primitiveId: string,
    weights: Float32Array,
    frame: GltfSceneControllerDiagnosticFrame,
  ): Partial<ScenePrimitive> | undefined {
    const base = this.#basePrimitiveById.get(primitiveId);
    if (!isSkinnedMesh(base) || !base.morphTargets || base.morphTargets.length === 0) {
      emitControllerDiagnostic(frame, {
        code: 'animation-morph-target-missing',
        path: `scene.primitives["${primitiveId}"].morphTargets`,
        message:
          `[vitrum/gltf-adapter] Animation weights target resolved to primitive "${primitiveId}", ` +
          'but that primitive has no morph targets; channel skipped.',
        primitiveId,
      });
      return undefined;
    }
    const morphWeights = fitMorphWeights(weights, base.morphTargets.length, primitiveId, frame);
    const solved = solveSkin({ ...base, morphWeights });
    return {
      morphWeights,
      positions: solved.positions,
      normals: solved.normals,
      ...(solved.tangents ? { tangents: solved.tangents } : {}),
      ...(solved.uvs ? { uvs: solved.uvs } : {}),
      ...(solved.uv1 ? { uv1: solved.uv1 } : {}),
    } as Partial<ScenePrimitive>;
  }

  #buildSkinPatch(
    primitiveId: string,
    binding: SkinBinding,
    worldTransforms: ReadonlyMap<number, Mat4>,
    existingPatch: Partial<ScenePrimitive> | undefined,
    frame: GltfSceneControllerDiagnosticFrame,
  ): Partial<ScenePrimitive> | undefined {
    const base = this.#basePrimitiveById.get(primitiveId);
    if (!isSkinnedMesh(base)) return undefined;

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
      ...(solved.tangents ? { tangents: solved.tangents } : {}),
      ...(solved.uvs ? { uvs: solved.uvs } : {}),
      ...(solved.uv1 ? { uv1: solved.uv1 } : {}),
    } as Partial<ScenePrimitive>;
  }

  #applyMaterialPointerSamples(
    samples: readonly SampledChannel[],
    patchMap: Map<string, Partial<ScenePrimitive>>,
    frame: GltfSceneControllerDiagnosticFrame,
  ): void {
    if (samples.length === 0) return;
    const animatedMaterials = new Map<number, MaterialSpec>();
    const touchedMaterials = new Set<number>();

    for (const sample of samples) {
      const target = resolveGltfMaterialAnimationPointer(sample.pointer);
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

      const base = animatedMaterials.get(target.materialIndex)
        ?? this.#convertedMaterials[target.materialIndex];
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
        applyGltfMaterialPointerValue(base, target, sample.value),
      );
      touchedMaterials.add(target.materialIndex);
    }

    for (const materialIndex of touchedMaterials) {
      const material = animatedMaterials.get(materialIndex);
      if (!material) continue;
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
        if (!findPrimitive(this.#scene, id)) continue;
        mergePrimitivePatch(patchMap, id, {
          material: materialReplacementPatch(material),
        } as Partial<ScenePrimitive>);
      }
    }
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

function buildMaterialBindings(
  bindings: readonly GltfMaterialBinding[],
): Map<number, string[]> {
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
  if (mapping.material < 0 || mapping.material >= (gltf.materials?.length ?? 0)) {
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

type MaterialTextureRefField = typeof MATERIAL_TEXTURE_REF_FIELDS[number];

function materialReplacementPatch(material: MaterialSpec): MaterialSpec {
  const patch: Record<string, unknown> = {};
  for (const field of MATERIAL_REPLACEMENT_CLEAR_FIELDS) patch[field] = undefined;
  return Object.assign(patch, material) as MaterialSpec;
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
  const droppedFields = new Set<MaterialTextureRefField>(
    patch.droppedTextureFields as readonly MaterialTextureRefField[] | undefined,
  );
  for (const field of MATERIAL_TEXTURE_REF_FIELDS) {
    const routedRef = patch.materialRouting[field as MaterialTextureRefField];
    const liveRef = material[field as MaterialTextureRefField];
    if (isTextureRef(routedRef)) {
      routed[field] = isTextureRef(liveRef)
        ? textureRefWithRouting(liveRef, routedRef)
        : routedRef;
    } else if (droppedFields.has(field as MaterialTextureRefField)) {
      delete routed[field];
    }
  }
  return routed as unknown as MaterialSpec;
}

function materialVariantPatchForSelection(
  binding: GltfMaterialVariantBinding,
  selectedVariantIndex: number | undefined,
): GltfMaterialVariantPrimitivePatch | undefined {
  if (selectedVariantIndex === undefined) return binding.basePatch;
  return binding.variantPatches?.find((patch) => patch.variantIndex === selectedVariantIndex)?.patch
    ?? binding.basePatch;
}

function scenePrimitivePatchForMaterialVariant(
  patch: GltfMaterialVariantPrimitivePatch,
  material: MaterialSpec,
): Partial<ScenePrimitive> {
  return {
    material: materialReplacementPatch(material),
    uvs: patch.uvs,
    uv1: patch.uv1,
    tangents: patch.tangents,
  } as unknown as Partial<ScenePrimitive>;
}

function normalizeBlendWeights(weights: readonly number[]): number[] {
  const positive = weights.map((weight) =>
    Number.isFinite(weight) && weight > 0 ? weight : 0,
  );
  const sum = positive.reduce((acc, weight) => acc + weight, 0);
  if (sum <= 0) {
    throw new Error('[vitrum/gltf-adapter] GltfSceneController.blend: at least one weight must be positive.');
  }
  return positive.map((weight) => weight / sum);
}

function blendSampledChannels(
  perClipSamples: readonly (readonly SampledChannel[])[],
  weights: readonly number[],
): SampledChannel[] {
  const accumulators = new Map<string, {
    node: SampledChannel['node'];
    path: SampledChannel['path'];
    pointer?: string;
    value: Float32Array;
    weightSum: number;
    referenceQuat?: Float32Array;
  }>();

  for (let clipIndex = 0; clipIndex < perClipSamples.length; clipIndex += 1) {
    const weight = weights[clipIndex] ?? 0;
    if (weight <= 0) continue;
    for (const sample of perClipSamples[clipIndex] ?? []) {
      const key = `${sample.node}\u0000${sample.path}\u0000${sample.pointer ?? ''}`;
      let acc = accumulators.get(key);
      if (!acc) {
        acc = {
          node: sample.node,
          path: sample.path,
          ...(sample.pointer !== undefined ? { pointer: sample.pointer } : {}),
          value: new Float32Array(sample.value.length),
          weightSum: 0,
          ...(sample.path === 'rotation' ? { referenceQuat: new Float32Array(sample.value) } : {}),
        };
        accumulators.set(key, acc);
      }
      if (sample.value.length > acc.value.length) {
        const grown = new Float32Array(sample.value.length);
        grown.set(acc.value);
        acc.value = grown;
      }

      const sign = sample.path === 'rotation' && acc.referenceQuat && quatDot(acc.referenceQuat, sample.value) < 0
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
    if (acc.path === 'rotation') {
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

function quatDot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  return (a[0] ?? 0) * (b[0] ?? 0) +
    (a[1] ?? 0) * (b[1] ?? 0) +
    (a[2] ?? 0) * (b[2] ?? 0) +
    (a[3] ?? 1) * (b[3] ?? 1);
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

function buildAnimatedInstanceTransforms(
  nodeWorld: Mat4,
  localInstances: ReadonlyArray<Mat4>,
): readonly Mat4[] {
  return localInstances.map((local) => asMat4(mat4Mul(nodeWorld, local)));
}

function buildExpandedInstanceTransform(
  nodeWorld: Mat4,
  binding: GltfInstancingBinding,
): Mat4 | undefined {
  const index = binding.expandedPrimitiveInstanceIndex;
  if (index === undefined) return undefined;
  const local = binding.localInstanceTransforms[index];
  return local ? asMat4(mat4Mul(nodeWorld, local)) : undefined;
}

function instanceMatricesAlmostEqual(
  a: ReadonlyArray<Mat4>,
  b: ReadonlyArray<Mat4>,
): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) {
    const ai = a[i];
    const bi = b[i];
    if (!ai || !bi || !matArrayAlmostEqual(ai, bi)) return false;
  }
  return true;
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

function isInstancedMesh(primitive: ScenePrimitive | undefined): primitive is InstancedMeshPrimitive {
  return primitive?.kind === 'instanced-mesh';
}

function fitMorphWeights(
  weights: Float32Array,
  targetCount: number,
  primitiveId: string,
  frame: GltfSceneControllerDiagnosticFrame,
): Float32Array {
  if (weights.length === targetCount) return new Float32Array(weights);
  emitControllerDiagnostic(frame, {
    code: 'morph-weight-count-mismatch',
    path: `scene.primitives["${primitiveId}"].morphWeights`,
    message:
      `[vitrum/gltf-adapter] Animation weights for primitive "${primitiveId}" have length ` +
      `${weights.length}, expected ${targetCount}; extra entries are dropped and missing entries default to 0.`,
    primitiveId,
  });
  const out = new Float32Array(targetCount);
  for (let i = 0; i < targetCount; i += 1) out[i] = weights[i] ?? 0;
  return out;
}
