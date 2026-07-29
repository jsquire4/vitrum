// SceneMutationRouter — the scene-mutation fast-path dispatch extracted
// verbatim from PTEngineWebGPU (Task 4.3, Theme A).
//
// This is a god-class DECOMPOSITION, not a rewrite: the engine still OWNS its
// state (#scene / #sceneBuffers / #geoPack / device / pipelines). The router
// operates on that state through the {@link MutationHost} seam — it never
// duplicates or independently owns engine state. Routing, throws, uploads, and
// return values are behavior-identical to the pre-extraction inline methods
// (addPrimitive / removePrimitive / updatePrimitive / updateEmitter /
// updateEnvironment + the 6 first-eligible-wins fast paths).
import type {
  EngineWarning,
  PrimitiveUvSets,
  Scene,
  SceneEmitter,
  ScenePrimitive,
} from '@vitrum/core';
import { asMat4 } from '@vitrum/core';
import type { ScenePackResult } from '@vitrum/shared-bvh';
import {
  BVH_NODE_FLOATS,
  fingerprintTlasBuffers,
  rebuildPrimitiveBlas,
  rebuildTlasReuseBlas,
} from '@vitrum/shared-bvh';
import { invertMat4 } from './math/mat4.js';
import {
  packFoldedMaterialEntry,
  packLightTreeForScene,
  prepareSceneBufferMutation,
  rebuildTlasForSceneTransforms,
  sceneHasMappedAnalytic,
  scenePackGeometryMutationPatch,
  scenePackTlasMutationPatch,
  type PreparedSceneBufferMutation,
  type SceneBufferMutationPatch,
  type UploadedSceneBuffers,
} from './scene/uploadSceneBuffers.js';
import {
  analyticIndexForPrimitive,
  canFastPathGeometryPatch,
  canFastPathInstancedTopologyPatch,
  canFastPathTopologyResizePatch,
  canFastPathTransformPatch,
  canReuseTlasBufferLengths,
  materialPatchRepackFields,
  materialIndexForPrimitive,
} from './scene/incrementalPatch.js';
import { patchEmitterInScene, patchPrimitiveInScene } from './scene/patchScene.js';
import { MATERIAL_FLOAT_STRIDE } from './scene/materialPacking.js';
import {
  collectMaterialTextures,
  MATERIAL_TEX_CLEARCOAT_NORMAL_VEC4_OFFSET,
  MATERIAL_TEX_FLOAT_STRIDE,
} from './scene/materialTextures.js';
import { solveSkinnedPrimitive } from './scene/solveSkinnedPrimitive.js';
import { packGpuUvSets } from './scene/gpuUvPacking.js';
import {
  hasMeshAreaEmitterForPrimitive,
  packEmitterArrays,
  type EnvSummaryForTree,
  type PackedEmitterArrays,
} from './scene/emitterPacking.js';
import { environmentParams } from './scene/environmentPacking.js';
import {
  collectUnsupportedMaterialFieldsForTraceTier,
} from './supportDetails.js';

const IDENTITY_MAT4 = asMat4(new Float32Array([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
]));

function warnHost(
  host: MutationHost,
  warning: EngineWarning,
  ...consoleArgs: readonly unknown[]
): void {
  if (host.warn != null) {
    host.warn(warning, ...consoleArgs);
  } else {
    console.warn(...(consoleArgs.length > 0 ? consoleArgs : [warning.message]));
  }
}

interface PreparedMutationSideEffect {
  commit(): void;
  rollback(): void;
  finalize(): void;
}

/**
 * The engine-state + engine-operation seam the router needs. The engine
 * implements this against its own private fields; the router holds NO state of
 * its own beyond this reference. Reading `getSceneBuffers()`/`getGeoPack()`
 * returns the LIVE objects (the fast paths mutate them in place), and the
 * setters reassign the engine's `#scene` / `#geoPack` on commit.
 */
export interface MutationHost {
  readonly device: GPUDevice;
  /** Throws if disposed or no scene yet (mirrors PTEngineWebGPU.#assertLive). */
  assertLive(method: string): void;
  getScene(): Scene | null;
  setSceneState(scene: Scene): void;
  getSceneBuffers(): UploadedSceneBuffers | null;
  /** Validate a candidate immutable scene before staging any GPU resources. */
  validateScene?(scene: Scene): void;
  getGeoPack(): ScenePackResult | null;
  setGeoPack(pack: ScenePackResult): void;
  /** Refresh derived scene bounds after a geometry-pack publication. Returns a
   * rollback closure because reset/submission can still fail transactionally. */
  refreshSceneGeometryStats?(): (() => void);
  invalidateBindGroups(): void;
  supportedAnalyticShapes(): ReadonlySet<string>;
  /** Whether camera-visible emitters (emissive fold) is enabled for this engine instance. */
  cameraVisibleEmitters(): boolean;
  /** Stage lite textures from a candidate scene-buffer preview without publication. */
  stageLiteTextures?(
    sceneBuffers: UploadedSceneBuffers,
  ): PreparedMutationSideEffect | null;
  /** True when the engine selected the single-group lite shader tier. */
  isLiteTier?(): boolean;
  /**
   * Manifold NEE's compact facet table is derived from primitive topology and
   * potential-delta material state. Preserve it transactionally by routing
   * primitive mutations through the canonical full pack while that mode is on.
   */
  requiresMneeFacetTableRepack?(): boolean;
  /** Structured warning sink owned by the engine; mirrors console.warn. */
  warn?(warning: EngineWarning, ...consoleArgs: readonly unknown[]): void;
  /** Full scene repack (engine-internal: destroys buffers + re-inits BDPT). */
  repackScene(scene: Scene, opts: { readonly warnOnEmpty: boolean }): void;
  /** Public setScene entry — the fall-through for every fast-path miss. */
  setScene(scene: Scene): void;
  reset(): void;
}

/** Candidate data returned by a first-eligible-wins fast-path handler. */
interface FastPathCommit {
  /** Complete affected-buffer patch; publication happens only in the common transaction. */
  readonly bufferPatch: SceneBufferMutationPatch;
  /** When set, becomes the new `#geoPack` before bind-group invalidation. */
  readonly geoPack?: ScenePackResult;
  /** Warnings to drain after committing (empty for the no-warning handlers). */
  readonly warnings: readonly string[];
  /**
   * H11 — when `true` the commit moved world-space vertex positions or transforms,
   * so mesh-area emitter triangles may have shifted. The common-commit code will
   * re-run `packEmitterArrays` + re-upload the emitter arrays when the patched
   * primitive is backed by a mesh-area emitter. Set by geometry (1, 2), instanced-
   * topology (4), and transform (5) fast paths; NOT set by analytic-transform (3)
   * or material-only (6) patches.
   */
  readonly reshapedWorldPositions?: boolean;
  /**
   * Item 2c — when `true` the material fast path wrote an emissive-field change
   * (emissive / emissiveIntensity). The common-commit code will re-run
   * `packEmitterArrays` + `rebuildLightTreeForScene` so the implicit `__implicit__`
   * NEE emitter (H14-A) picks up the new radiance. Without this the NEE sampling
   * distribution diverges from the camera-hit emissive value until the next full
   * repack.
   */
  readonly changedEmissiveField?: boolean;
}

/** Whether a committed buffer patch can change the world-space scene bounds. */
export function sceneGeometryStatsNeedRefresh(
  bufferPatch: SceneBufferMutationPatch,
  publishesGeoPack: boolean,
): boolean {
  return (
    publishesGeoPack ||
    bufferPatch.bvhNodes !== undefined ||
    bufferPatch.tlasNodes !== undefined ||
    bufferPatch.analyticLocalToWorld !== undefined
  );
}

const MATERIAL_DESCRIPTOR_SCALAR_OFFSETS = [
  4,  // alphaMode
  5,  // alphaCutoff
  6,  // opacity
  16, // aoMapIntensity
  17, // lightMapIntensity
  18, // bumpScale
  19, // envMapIntensity
  20, // anisotropy
  21, // anisotropyRotation
  23, // normalScale
  MATERIAL_TEX_CLEARCOAT_NORMAL_VEC4_OFFSET * 4 + 1, // clearcoatNormalScale
] as const;

function isMaterialOnlyPatch(patch: Partial<ScenePrimitive>): patch is Partial<ScenePrimitive> & {
  material: ScenePrimitive['material'];
} {
  if (patch.material == null) return false;
  return Object.keys(patch).every((key) => key === 'material' || key === 'id' || key === 'kind');
}

function materialSpecsForScene(
  scene: Scene,
  supportedAnalyticShapes: ReadonlySet<string>,
): readonly ScenePrimitive['material'][] {
  const materials: ScenePrimitive['material'][] = [];
  for (const primitive of scene.primitives) {
    if (primitive.kind === 'analytic') {
      if (supportedAnalyticShapes.has(primitive.shape)) {
        materials.push(primitive.material);
      }
      continue;
    }
    if (
      primitive.kind === 'mesh' ||
      primitive.kind === 'skinned-mesh' ||
      primitive.kind === 'instanced-mesh'
    ) {
      materials.push(primitive.material);
    }
  }
  return materials;
}

function materialDescriptorScalarReplacement(
  sceneBuffers: UploadedSceneBuffers,
  nextScene: Scene,
  materialIndex: number,
  supportedAnalyticShapes: ReadonlySet<string>,
): Float32Array | null {
  const nextDescriptors = collectMaterialTextures(
    materialSpecsForScene(nextScene, supportedAnalyticShapes),
  ).descriptors;
  const descriptorOffset = materialIndex * MATERIAL_TEX_FLOAT_STRIDE;
  if (
    descriptorOffset < 0 ||
    descriptorOffset + MATERIAL_TEX_FLOAT_STRIDE > sceneBuffers.materialTexDescriptors.length ||
    descriptorOffset + MATERIAL_TEX_FLOAT_STRIDE > nextDescriptors.length
  ) {
    return null;
  }

  const descriptorData = new Float32Array(sceneBuffers.materialTexDescriptors);
  const nextSlice = nextDescriptors.subarray(
    descriptorOffset,
    descriptorOffset + MATERIAL_TEX_FLOAT_STRIDE,
  );
  for (const relativeOffset of MATERIAL_DESCRIPTOR_SCALAR_OFFSETS) {
    const targetOffset = descriptorOffset + relativeOffset;
    descriptorData[targetOffset] = nextSlice[relativeOffset]
      ?? descriptorData[targetOffset]
      ?? 0;
  }
  return descriptorData;
}

interface DeferredWarning {
  readonly warning: EngineWarning;
  readonly consoleArgs?: readonly unknown[];
}

function emitterAndLightTreeMutationPatch(
  scene: Scene,
  packed: PackedEmitterArrays,
  envSummary: EnvSummaryForTree,
): SceneBufferMutationPatch {
  const tree = packLightTreeForScene(scene, {
    packed,
    envSummary,
  });
  return {
    directionalLightsData: packed.directionalLightsData,
    pointLightsData: packed.pointLightsData,
    spotLightsData: packed.spotLightsData,
    rectAreaLightsData: packed.rectAreaLightsData,
    meshAreaLightsData: packed.meshAreaLightsData,
    meshAreaLightSourceFactorsData: packed.meshAreaLightSourceFactorsData,
    lightTreeNodes: tree.lightTreeNodes,
    directionalLightCount: packed.directionalLightCount,
    pointLightCount: packed.pointLightCount,
    spotLightCount: packed.spotLightCount,
    rectAreaLightCount: packed.rectAreaLightCount,
    meshAreaLightCount: packed.meshAreaLightCount,
    lightTreeNodeCount: tree.lightTreeNodeCount,
    lightTreeEnabled: tree.lightTreeEnabled,
  };
}

function environmentSummaryFromSceneBuffers(
  sceneBuffers: UploadedSceneBuffers,
): EnvSummaryForTree {
  return {
    hasHdri: sceneBuffers.hasEnvironmentMap,
    sunStrength: sceneBuffers.environmentSunStrength,
    lightTreePower: sceneBuffers.environmentLightTreePower,
  };
}

function cameraVisibleEmitterMaterials(
  scene: Scene,
  sceneBuffers: UploadedSceneBuffers,
  supportedAnalyticShapes: ReadonlySet<string>,
): Float32Array {
  const materials = new Float32Array(sceneBuffers.materials);
  for (const primitive of scene.primitives) {
    const materialIndex = materialIndexForPrimitive(
      scene,
      primitive.id,
      supportedAnalyticShapes,
    );
    if (materialIndex == null) continue;
    const folded = packFoldedMaterialEntry(primitive, scene, true);
    if (folded.length !== MATERIAL_FLOAT_STRIDE) continue;
    // RGB thin-film LUT records live in a sparse tail. Emitter folding changes
    // only the fixed emissive lanes, so retain the absolute tail pointer.
    folded[28 * 4 + 2] = sceneBuffers.materials[
      materialIndex * MATERIAL_FLOAT_STRIDE + 28 * 4 + 2
    ] ?? 0;
    materials.set(folded, materialIndex * MATERIAL_FLOAT_STRIDE);
  }
  return materials;
}

function solvedSkinGeometryPatch(
  primitive: Extract<ScenePrimitive, { kind: 'skinned-mesh' }>,
): {
  readonly positions: Float32Array;
  readonly normals: Float32Array;
  readonly tangents?: Float32Array;
  readonly uvs?: Float32Array;
  readonly uv1?: Float32Array;
  readonly uvSets?: PrimitiveUvSets;
} {
  const solved = solveSkinnedPrimitive(primitive);
  return {
    positions: solved.positions,
    normals: solved.normals,
    ...(solved.tangents != null ? { tangents: solved.tangents } : {}),
    ...(solved.uvs != null ? { uvs: solved.uvs } : {}),
    ...(solved.uv1 != null ? { uv1: solved.uv1 } : {}),
    ...(solved.uvSets != null ? { uvSets: solved.uvSets } : {}),
  };
}

export class SceneMutationRouter {
  readonly #host: MutationHost;

  constructor(host: MutationHost) {
    this.#host = host;
  }

  #commitPreparedMutation(
    nextScene: Scene,
    sceneBuffers: UploadedSceneBuffers,
    bufferPatch: SceneBufferMutationPatch,
    nextGeoPack: ScenePackResult | undefined,
    deferredWarnings: readonly DeferredWarning[],
  ): void {
    const host = this.#host;
    const previousScene = host.getScene()!;
    const previousGeoPack = host.getGeoPack();
    host.validateScene?.(nextScene);
    let bufferMutation: PreparedSceneBufferMutation | null = null;
    let liteMutation: PreparedMutationSideEffect | null = null;
    let rollbackSceneGeometryStats: (() => void) | null = null;
    try {
      bufferMutation = prepareSceneBufferMutation(host.device, sceneBuffers, bufferPatch);
      liteMutation = host.stageLiteTextures?.(bufferMutation.preview) ?? null;
    } catch (error) {
      liteMutation?.rollback();
      bufferMutation?.rollback();
      throw error;
    }

    try {
      bufferMutation.commit();
      liteMutation?.commit();
      if (nextGeoPack != null) host.setGeoPack(nextGeoPack);
      host.setSceneState(nextScene);
      // Same-size mutations copy into existing handles and deliberately preserve
      // cached bind groups. Replacement mutations still require a rebuild.
      if (sceneGeometryStatsNeedRefresh(bufferPatch, nextGeoPack != null)) {
        rollbackSceneGeometryStats = host.refreshSceneGeometryStats?.() ?? null;
      }
      if (bufferMutation.replacesBufferHandles) host.invalidateBindGroups();
      host.reset();
    } catch (error) {
      // Publication is reversible because old resources remain live until finalize.
      try { host.setSceneState(previousScene); } catch { /* preserve mutation failure */ }
      try { rollbackSceneGeometryStats?.(); } catch { /* preserve mutation failure */ }
      if (nextGeoPack != null && previousGeoPack != null) {
        try { host.setGeoPack(previousGeoPack); } catch { /* preserve mutation failure */ }
      }
      try { liteMutation?.rollback(); } catch { /* preserve mutation failure */ }
      try { bufferMutation.rollback(); } catch { /* preserve mutation failure */ }
      if (bufferMutation.replacesBufferHandles) {
        try { host.invalidateBindGroups(); } catch { /* restored handles bind next frame */ }
      }
      throw error;
    }

    // Irreversible retirement and outward warning side effects happen only after
    // the scene, resource mirrors, bind groups, and temporal reset all succeeded.
    try { liteMutation?.finalize(); } catch { /* the candidate remains live */ }
    bufferMutation.finalize();
    for (const deferred of deferredWarnings) {
      warnHost(host, deferred.warning, ...(deferred.consoleArgs ?? []));
    }
  }

  /**
   * Add one whole primitive to the live scene (contract: {@link Engine.addPrimitive}).
   * TAIL-INSERTION re-packed via the host's full repack — see the original
   * PTEngineWebGPU.addPrimitive header for why a bespoke incremental append is
   * deliberately NOT taken.
   */
  addPrimitive(primitive: ScenePrimitive): void {
    this.#host.assertLive('addPrimitive');
    const currentScene = this.#host.getScene()!;
    if (currentScene.primitives.some((p) => p.id === primitive.id)) {
      throw new Error(
        `addPrimitive: a primitive with id "${primitive.id}" already exists; ` +
          'use updatePrimitive to mutate an existing primitive.',
      );
    }
    const nextScene: Scene = {
      ...currentScene,
      primitives: [...currentScene.primitives, primitive],
    };
    this.#host.repackScene(nextScene, { warnOnEmpty: false });
  }

  /**
   * Remove one whole primitive from the live scene by id (contract:
   * {@link Engine.removePrimitive}). Full repack — the dense index remap a remove
   * needs is reproduced correct-by-construction by buildPackedScene.
   */
  removePrimitive(id: ScenePrimitive['id']): void {
    this.#host.assertLive('removePrimitive');
    const currentScene = this.#host.getScene()!;
    const nextPrimitives = currentScene.primitives.filter((p) => p.id !== id);
    if (nextPrimitives.length === currentScene.primitives.length) {
      throw new Error(
        `removePrimitive: no primitive with id "${id}" in the live scene.`,
      );
    }
    const nextScene: Scene = {
      ...currentScene,
      primitives: nextPrimitives,
    };
    this.#host.repackScene(nextScene, { warnOnEmpty: false });
  }

  updatePrimitive(id: string, patch: Partial<ScenePrimitive>): void {
    this.#host.assertLive('updatePrimitive');
    const host = this.#host;
    const currentScene = host.getScene()!;
    const currentPrimitive = currentScene.primitives.find((primitive) => primitive.id === id) ?? null;
    const deferredWarnings: DeferredWarning[] = [];
    if (currentPrimitive == null) {
      throw new Error(`updatePrimitive: primitive "${id}" not found in current scene`);
    }
    if ('kind' in patch && patch.kind !== undefined && patch.kind !== currentPrimitive.kind) {
      throw new Error(
        `updatePrimitive: primitive "${currentPrimitive.id}" kind cannot change from ` +
          `"${currentPrimitive.kind}" to "${patch.kind}"`,
      );
    }
    if ('id' in patch && patch.id !== undefined && patch.id !== currentPrimitive.id) {
      throw new Error(`updatePrimitive: primitive "${currentPrimitive.id}" id cannot be changed`);
    }
    const payloadKeys = Object.keys(patch).filter((key) => key !== 'id' && key !== 'kind');
    if (payloadKeys.length === 0) {
      // A validated empty/identity-only patch is an actual no-op: preserve the
      // scene snapshot, GPU generation, temporal history, and warning surface.
      return;
    }
    // Keep the host-authored/rest-pose scene separate from the transient posed
    // geometry used for BLAS/emitter packing. Storing solved vertices in #scene
    // makes the next bone update skin the previous pose and compounds motion.
    const authoredNextScene = patchPrimitiveInScene(currentScene, id, patch);
    host.validateScene?.(authoredNextScene);
    const patchedMaterial = (
      patch as unknown as { material?: Record<string, unknown> }
    ).material;
    if (patchedMaterial != null) {
      const unsupportedMaterialFields = collectUnsupportedMaterialFieldsForTraceTier(
        patchedMaterial,
        host.isLiteTier?.() === true ? 'lite' : 'full',
      );
      if (unsupportedMaterialFields.length > 0) {
        throw new TypeError(
          `[vitrum/pt-webgpu] updatePrimitive("${id}"): material fields are supplied ` +
            `but not rendered by the selected ${host.isLiteTier?.() === true ? 'lite' : 'full'} ` +
            `tier: ${unsupportedMaterialFields.join(', ')}.`,
        );
      }
    }
    // Mapped analytics are represented by their fallback meshes in the packed
    // scene. Their BLAS/TLAS and material classification therefore differs from
    // the authored analytic snapshot. Conservatively use the canonical full
    // repack for every primitive mutation while either snapshot contains one;
    // this also handles adding/removing a map without letting an analytic fast
    // path write against a fallback-mesh buffer layout.
    if (
      sceneHasMappedAnalytic(currentScene) ||
      sceneHasMappedAnalytic(authoredNextScene)
    ) {
      host.repackScene(authoredNextScene, { warnOnEmpty: false });
      return;
    }
    if (host.requiresMneeFacetTableRepack?.() === true) {
      host.repackScene(authoredNextScene, { warnOnEmpty: false });
      return;
    }
    const authoredNextPrimitive =
      authoredNextScene.primitives.find((primitive) => primitive.id === id) ?? null;
    let renderNextScene = authoredNextScene;
    let fastPathPatch: Partial<ScenePrimitive> = patch;
    const skinGeometryMutationPatch =
      currentPrimitive.kind === 'skinned-mesh' &&
      (
        'bones' in patch ||
        'skinIndices' in patch ||
        'skinWeights' in patch ||
        'skinInfluencesPerVertex' in patch ||
        'bindMatrix' in patch ||
        'bindMatrixInverse' in patch ||
        'morphTargets' in patch ||
        'morphTargetNormals' in patch ||
        'morphTargetTangents' in patch ||
        'morphTargetUvs' in patch ||
        'morphTargetUv1s' in patch ||
        'morphTargetUvSets' in patch ||
        'uvs' in patch ||
        'uv1' in patch ||
        'uvSets' in patch ||
        'colors' in patch ||
        'colorSets' in patch ||
        'vertexColorSet' in patch ||
        'boneInverses' in patch ||
        'morphWeights' in patch ||
        'positions' in patch ||
        'normals' in patch ||
        'tangents' in patch ||
        'indices' in patch
      );
    const materialChangesEmissive = patchedMaterial != null &&
      ('emissive' in patchedMaterial || 'emissiveIntensity' in patchedMaterial);
    const meshAreaEmitterNeedsSolvedGeometry =
      ('transform' in patch || materialChangesEmissive) &&
      hasMeshAreaEmitterForPrimitive(currentScene, id);
    const requiresSolvedSkinGeometry =
      currentPrimitive.kind === 'skinned-mesh' &&
      (skinGeometryMutationPatch || meshAreaEmitterNeedsSolvedGeometry);
    let skinSolveFailed = false;

    if (
      requiresSolvedSkinGeometry &&
      currentPrimitive.kind === 'skinned-mesh' &&
      authoredNextPrimitive?.kind === 'skinned-mesh'
    ) {
      try {
        const solvedPatch = solvedSkinGeometryPatch(authoredNextPrimitive);
        renderNextScene = patchPrimitiveInScene(authoredNextScene, id, solvedPatch);
        if (skinGeometryMutationPatch) {
          const {
            bones: _bones,
            boneInverses: _boneInverses,
            morphWeights: _morphWeights,
            positions: _positions,
            skinIndices: _skinIndices,
            skinWeights: _skinWeights,
            skinInfluencesPerVertex: _skinInfluencesPerVertex,
            bindMatrix: _bindMatrix,
            bindMatrixInverse: _bindMatrixInverse,
            morphTargets: _morphTargets,
            morphTargetNormals: _morphTargetNormals,
            morphTargetTangents: _morphTargetTangents,
            morphTargetUvs: _morphTargetUvs,
            morphTargetUv1s: _morphTargetUv1s,
            morphTargetUvSets: _morphTargetUvSets,
            uvs: _legacyUvs,
            uv1: _uv1,
            uvSets: _uvSets,
            normals: _normals,
            tangents: _tangents,
            ...nonSkinPatch
          } = patch as Partial<ScenePrimitive> & Record<string, unknown>;
          fastPathPatch = {
            ...nonSkinPatch,
            positions: solvedPatch.positions,
            normals: solvedPatch.normals,
            ...(solvedPatch.tangents != null
              ? { tangents: solvedPatch.tangents }
              : {}),
            ...(solvedPatch.uvs != null
              ? { uvs: solvedPatch.uvs }
              : {}),
            ...(solvedPatch.uv1 != null
              ? { uv1: solvedPatch.uv1 }
              : {}),
            ...(solvedPatch.uvSets != null
              ? { uvSets: solvedPatch.uvSets }
              : {}),
          };
        }
      } catch (error) {
        skinSolveFailed = true;
        deferredWarnings.push({
          warning: {
            code: 'pt-webgpu.update-primitive-skin-fallback',
            backend: 'pt-webgpu',
            phase: 'mutation',
            method: 'updatePrimitive',
            message:
              `[vitrum/pt-webgpu] solveSkin failed for updatePrimitive("${id}"); ` +
              `falling back to setScene. ${String(error)}`,
            details: { id },
            raw: error,
          },
        });
      }
    }

    const drainDeferredWarnings = (): void => {
      for (const deferred of deferredWarnings) {
        warnHost(host, deferred.warning, ...(deferred.consoleArgs ?? []));
      }
    };

    if (skinSolveFailed) {
      host.setScene(authoredNextScene);
      drainDeferredWarnings();
      return;
    }

    const liteFallbackGeometryPatch =
      host.isLiteTier?.() === true &&
      currentPrimitive != null &&
      canFastPathGeometryPatch(currentPrimitive, fastPathPatch);
    const liteFallbackMeshTopologyPatch =
      host.isLiteTier?.() === true &&
      currentPrimitive != null &&
      canFastPathTopologyResizePatch(currentPrimitive, fastPathPatch);
    const liteFallbackTransformPatch =
      host.isLiteTier?.() === true &&
      currentPrimitive != null &&
      canFastPathTransformPatch(currentPrimitive, fastPathPatch);
    const liteFallbackInstancedTopologyPatch =
      host.isLiteTier?.() === true &&
      currentPrimitive != null &&
      canFastPathInstancedTopologyPatch(currentPrimitive, fastPathPatch);
    if (
      liteFallbackGeometryPatch ||
      liteFallbackMeshTopologyPatch ||
      liteFallbackTransformPatch ||
      liteFallbackInstancedTopologyPatch
    ) {
      const fallbackReason = liteFallbackInstancedTopologyPatch
        ? 'lite-merged-blas-instanced-topology-rebuild'
        : liteFallbackMeshTopologyPatch
          ? 'lite-merged-blas-mesh-topology-rebuild'
          : liteFallbackGeometryPatch
            ? 'lite-merged-blas-geometry-rebuild'
            : 'lite-merged-blas-transform-rebuild';
      host.repackScene(authoredNextScene, { warnOnEmpty: false });
      deferredWarnings.push({
        warning: {
          code: 'pt-webgpu.lite-update-primitive-fallback-rebuild',
          backend: 'pt-webgpu',
          phase: 'mutation',
          method: 'updatePrimitive',
          message:
            `[vitrum/pt-webgpu] updatePrimitive("${id}") is using a fallback scene repack ` +
            'on the lite tier because the lite shader traverses one baked merged BLAS, not TLAS instance transforms.',
          details: { id, fallbackReason },
        },
      });
      drainDeferredWarnings();
      return;
    }

    const fastPaths: ReadonlyArray<() => FastPathCommit | null> = [
      // 1) Same-count geometry update: rebuild only this primitive's BLAS.
      () => {
        const geoPack = host.getGeoPack();
        const sceneBuffers = host.getSceneBuffers();
        if (
          currentPrimitive == null ||
          geoPack == null ||
          sceneBuffers == null ||
          !canFastPathGeometryPatch(currentPrimitive, fastPathPatch)
        ) {
          return null;
        }
        const rebuilt = rebuildPrimitiveBlas(renderNextScene, id, geoPack, {
          tlas: true,
          resolveMaterialId: (primitiveId) =>
            materialIndexForPrimitive(
              authoredNextScene,
              primitiveId,
              host.supportedAnalyticShapes(),
            ) ?? 0,
        });
        if (!rebuilt.ok) return null;
        const previousTlasFingerprint = fingerprintTlasBuffers({
          tlasNodes: sceneBuffers.tlasNodes,
          tlasInstanceIndices: sceneBuffers.tlasInstanceIndices,
          tlasBlasRoots: sceneBuffers.tlasBlasRoots,
          tlasInstanceWorldToLocal: sceneBuffers.tlasInstanceWorldToLocal,
          tlasInstanceLocalToWorld: sceneBuffers.tlasInstanceLocalToWorld,
        });
        const nextTlasFingerprint = fingerprintTlasBuffers({
          tlasNodes: rebuilt.pack.tlasNodes,
          tlasInstanceIndices: rebuilt.pack.tlasInstanceIndices,
          tlasBlasRoots: rebuilt.pack.tlasBlasRoots,
          tlasInstanceWorldToLocal: rebuilt.pack.tlasInstanceWorldToLocal,
          tlasInstanceLocalToWorld: rebuilt.pack.tlasInstanceLocalToWorld,
        });
        const nextGpuUvs = packGpuUvSets(
          renderNextScene,
          rebuilt.pack.uvs,
          rebuilt.pack.primitiveTlasBindings,
          sceneBuffers.uvSetTexCoords,
        );
        return {
          bufferPatch: scenePackGeometryMutationPatch(
            sceneBuffers,
            rebuilt.pack,
            previousTlasFingerprint !== nextTlasFingerprint,
            nextGpuUvs,
          ),
          geoPack: rebuilt.pack,
          warnings: rebuilt.pack.warnings,
          reshapedWorldPositions: true,
        };
      },
      // 2) Vertex/index-count change: splice one BLAS and replace the bounded
      // BLAS/TLAS/CWBVH buffer set.
      () => {
        const geoPack = host.getGeoPack();
        const sceneBuffers = host.getSceneBuffers();
        if (
          currentPrimitive == null ||
          geoPack == null ||
          sceneBuffers == null ||
          !canFastPathTopologyResizePatch(currentPrimitive, fastPathPatch)
        ) {
          return null;
        }
        const rebuilt = rebuildPrimitiveBlas(renderNextScene, id, geoPack, {
          tlas: true,
          resolveMaterialId: (primitiveId) =>
            materialIndexForPrimitive(
              authoredNextScene,
              primitiveId,
              host.supportedAnalyticShapes(),
            ) ?? 0,
        });
        if (!rebuilt.ok) return null;
        const nextGpuUvs = packGpuUvSets(
          renderNextScene,
          rebuilt.pack.uvs,
          rebuilt.pack.primitiveTlasBindings,
          sceneBuffers.uvSetTexCoords,
        );
        return {
          bufferPatch: scenePackGeometryMutationPatch(
            sceneBuffers,
            rebuilt.pack,
            true,
            nextGpuUvs,
          ),
          geoPack: rebuilt.pack,
          warnings: rebuilt.pack.warnings,
          reshapedWorldPositions: true,
        };
      },
      // 3) Analytic transform: replace only the two analytic matrix buffers.
      () => {
        const sceneBuffers = host.getSceneBuffers();
        if (
          currentPrimitive == null ||
          currentPrimitive.kind !== 'analytic' ||
          sceneBuffers == null ||
          !canFastPathTransformPatch(currentPrimitive, fastPathPatch)
        ) {
          return null;
        }
        const analyticIndex = analyticIndexForPrimitive(
          authoredNextScene,
          id,
          host.supportedAnalyticShapes(),
        );
        const nextPrimitive = authoredNextScene.primitives.find((primitive) => primitive.id === id);
        if (
          analyticIndex == null ||
          nextPrimitive == null ||
          nextPrimitive.kind !== 'analytic'
        ) {
          return null;
        }
        const localToWorld = asMat4(nextPrimitive.transform ?? IDENTITY_MAT4);
        const maybeWorldToLocal = invertMat4(localToWorld);
        const worldToLocal = asMat4(maybeWorldToLocal ?? IDENTITY_MAT4);
        const nextLocalToWorld = new Float32Array(sceneBuffers.analyticLocalToWorld);
        const nextWorldToLocal = new Float32Array(sceneBuffers.analyticWorldToLocal);
        nextLocalToWorld.set(localToWorld, analyticIndex * 16);
        nextWorldToLocal.set(worldToLocal, analyticIndex * 16);
        if (maybeWorldToLocal == null) {
          deferredWarnings.push({
            warning: {
              code: 'pt-webgpu.noninvertible-analytic-transform',
              backend: 'pt-webgpu',
              phase: 'mutation',
              method: 'updatePrimitive',
              message:
                `[vitrum/pt-webgpu] Primitive "${nextPrimitive.id}" has non-invertible analytic transform; ` +
                'using identity fallback.',
              details: { id: nextPrimitive.id },
            },
          });
        }
        return {
          bufferPatch: {
            analyticLocalToWorld: nextLocalToWorld,
            analyticWorldToLocal: nextWorldToLocal,
          },
          warnings: [],
        };
      },
      // 4) Instance-count change: rebuild TLAS while retaining BLAS geometry.
      () => {
        const geoPack = host.getGeoPack();
        const sceneBuffers = host.getSceneBuffers();
        if (
          currentPrimitive == null ||
          geoPack == null ||
          sceneBuffers == null ||
          !canFastPathInstancedTopologyPatch(currentPrimitive, fastPathPatch)
        ) {
          return null;
        }
        const rebuilt = rebuildTlasReuseBlas(authoredNextScene, geoPack);
        if (!rebuilt.ok) return null;
        return {
          bufferPatch: scenePackTlasMutationPatch(sceneBuffers, rebuilt.pack),
          geoPack: rebuilt.pack,
          warnings: rebuilt.pack.warnings,
          reshapedWorldPositions: true,
        };
      },
      // 5) Same-count transform update: rebuild and replace TLAS buffers only.
      () => {
        const sceneBuffers = host.getSceneBuffers();
        if (
          currentPrimitive == null ||
          sceneBuffers == null ||
          !canFastPathTransformPatch(currentPrimitive, fastPathPatch)
        ) {
          return null;
        }
        const tlas = rebuildTlasForSceneTransforms(
          authoredNextScene,
          sceneBuffers.primitiveTlasBindings,
          {
            tlasNodes: sceneBuffers.tlasNodes,
            tlasInstanceIndices: sceneBuffers.tlasInstanceIndices,
            tlasBlasRoots: sceneBuffers.tlasBlasRoots,
            tlasInstanceWorldToLocal: sceneBuffers.tlasInstanceWorldToLocal,
          },
        );
        if (!tlas.ok || !canReuseTlasBufferLengths(sceneBuffers, tlas)) return null;
        return {
          bufferPatch: scenePackTlasMutationPatch(sceneBuffers, {
            tlasNodes: tlas.tlasNodes,
            tlasInstanceIndices: tlas.tlasInstanceIndices,
            tlasBlasRoots: tlas.tlasBlasRoots,
            tlasInstanceWorldToLocal: tlas.tlasInstanceWorldToLocal,
            tlasInstanceLocalToWorld: tlas.tlasInstanceLocalToWorld,
            tlasNodeCount: Math.floor(tlas.tlasNodes.length / BVH_NODE_FLOATS),
            primitiveTlasBindings: sceneBuffers.primitiveTlasBindings,
          }),
          warnings: tlas.warnings,
          reshapedWorldPositions: true,
        };
      },
      // 6) Material-only update: replace the material buffer and, when needed,
      // its descriptor buffer. Texture-array shape changes still full-repack.
      () => {
        const sceneBuffers = host.getSceneBuffers();
        if (!isMaterialOnlyPatch(fastPathPatch) || sceneBuffers == null) return null;
        const repackFields = materialPatchRepackFields(fastPathPatch);
        if (
          repackFields.textureFields.length > 0 ||
          repackFields.layerDescriptorFields.length > 0 ||
          repackFields.geometryFields.length > 0
        ) {
          return null;
        }
        if (
          repackFields.descriptorScalarFields.length > 0 &&
          host.isLiteTier?.() === true
        ) {
          return null;
        }
        const materialIndex = materialIndexForPrimitive(
          authoredNextScene,
          id,
          host.supportedAnalyticShapes(),
        );
        const primitive = authoredNextScene.primitives.find((candidate) => candidate.id === id);
        if (materialIndex == null || primitive == null) return null;
        // Thin-film RGB LUTs live in a variable-size sparse tail. Adding,
        // removing, or changing such a stack must transactionally rebuild that
        // tail and its absolute offsets; the fixed-slot fast path cannot do so.
        if (
          currentPrimitive?.material.thinFilmStack != null ||
          primitive.material.thinFilmStack != null
        ) return null;
        const packed = packFoldedMaterialEntry(
          primitive,
          authoredNextScene,
          host.cameraVisibleEmitters(),
        );
        if (packed.length !== MATERIAL_FLOAT_STRIDE) return null;
        const nextMaterials = new Float32Array(sceneBuffers.materials);
        nextMaterials.set(packed, materialIndex * MATERIAL_FLOAT_STRIDE);
        const bufferPatch: SceneBufferMutationPatch = { materials: nextMaterials };
        if (repackFields.descriptorScalarFields.length > 0) {
          const descriptors = materialDescriptorScalarReplacement(
            sceneBuffers,
            authoredNextScene,
            materialIndex,
            host.supportedAnalyticShapes(),
          );
          if (descriptors == null) return null;
          bufferPatch.materialTexDescriptors = descriptors;
        }

        const material = patchedMaterial ?? {};
        return {
          bufferPatch,
          warnings: [],
          changedEmissiveField:
            'emissive' in material || 'emissiveIntensity' in material,
        };
      },
    ];

    for (const tryFastPath of fastPaths) {
      const commit = tryFastPath();
      if (commit == null) continue;
      const sceneBuffers = host.getSceneBuffers();
      if (sceneBuffers == null) break;
      const combinedPatch: SceneBufferMutationPatch = { ...commit.bufferPatch };
      if (
        (commit.reshapedWorldPositions || commit.changedEmissiveField) &&
        (
          hasMeshAreaEmitterForPrimitive(currentScene, id) ||
          hasMeshAreaEmitterForPrimitive(renderNextScene, id)
        )
      ) {
        const emitterPacked = packEmitterArrays(renderNextScene);
        Object.assign(
          combinedPatch,
          emitterAndLightTreeMutationPatch(
            renderNextScene,
            emitterPacked,
            environmentSummaryFromSceneBuffers(sceneBuffers),
          ),
        );
        for (const warning of emitterPacked.warnings) {
          deferredWarnings.push({
            warning: {
              code: 'pt-webgpu.emitter-pack-warning',
              backend: 'pt-webgpu',
              phase: 'mutation',
              method: 'updatePrimitive',
              message: `[vitrum/pt-webgpu] ${warning}`,
              details: { warning },
            },
          });
        }
      }
      for (const warning of commit.warnings) {
        deferredWarnings.push({
          warning: {
            code: 'pt-webgpu.primitive-mutation-warning',
            backend: 'pt-webgpu',
            phase: 'mutation',
            method: 'updatePrimitive',
            message: `[vitrum/pt-webgpu] ${warning}`,
            details: { warning },
          },
        });
      }
      this.#commitPreparedMutation(
        authoredNextScene,
        sceneBuffers,
        combinedPatch,
        commit.geoPack,
        deferredWarnings,
      );
      return;
    }

    const repackFields = materialPatchRepackFields(fastPathPatch);
    if (
      repackFields.textureFields.length > 0 ||
      repackFields.descriptorScalarFields.length > 0 ||
      repackFields.layerDescriptorFields.length > 0 ||
      repackFields.geometryFields.length > 0
    ) {
      deferredWarnings.push({
        warning: {
          code: 'pt-webgpu.primitive-material-repack',
          backend: 'pt-webgpu',
          phase: 'mutation',
          method: 'updatePrimitive',
          message:
            `[vitrum/pt-webgpu] updatePrimitive("${id}") material patch touches ` +
            'texture-map, layer-normal descriptor, geometry-affecting, or lite-tier descriptor fields, so the backend is ' +
            'using a full scene repack to keep geometry, material descriptors, and texture arrays coherent.',
          details: {
            id,
            fallbackReason: 'material-texture-descriptor-repack',
            nativePatchMissing: 'targeted-material-texture-descriptor-update',
            textureFields: repackFields.textureFields,
            descriptorScalarFields: repackFields.descriptorScalarFields,
            layerDescriptorFields: repackFields.layerDescriptorFields,
            geometryFields: repackFields.geometryFields,
          },
        },
      });
    }
    host.setScene(authoredNextScene);
    drainDeferredWarnings();
  }

  updateEmitter(id: string, patch: Partial<SceneEmitter>): void {
    const host = this.#host;
    host.assertLive('updateEmitter');
    const currentScene = host.getScene()!;
    const currentEmitter = currentScene.emitters.find(
      (emitter) => String(emitter.id) === id,
    ) ?? null;
    if (currentEmitter == null) {
      throw new Error(`updateEmitter: emitter "${id}" not found in current scene`);
    }
    if ('kind' in patch && patch.kind !== undefined && patch.kind !== currentEmitter.kind) {
      throw new Error(
        `updateEmitter: emitter "${id}" kind cannot change from ` +
          `"${currentEmitter.kind}" to "${patch.kind}"`,
      );
    }
    if ('id' in patch && patch.id !== undefined && patch.id !== currentEmitter.id) {
      throw new Error(`updateEmitter: emitter "${id}" id cannot be changed`);
    }
    const payloadKeys = Object.keys(patch).filter((key) => key !== 'id' && key !== 'kind');
    if (payloadKeys.length === 0) {
      return;
    }

    const nextScene = patchEmitterInScene(currentScene, id, patch);
    const sceneBuffers = host.getSceneBuffers();
    if (sceneBuffers == null) {
      host.setScene(nextScene);
      return;
    }

    const packed = packEmitterArrays(nextScene);
    const bufferPatch = emitterAndLightTreeMutationPatch(
      nextScene,
      packed,
      environmentSummaryFromSceneBuffers(sceneBuffers),
    );

    // Camera-visible mesh emitters fold their radiance into the backing
    // primitive's material slot. Include that slot in the same candidate swap
    // as the emitter and light-tree arrays so neither side can become stale.
    if (host.cameraVisibleEmitters()) {
      const updatedEmitter = nextScene.emitters.find((emitter) => emitter.id === id);
      if (updatedEmitter?.kind === 'mesh-area') {
        const backedPrimitive = nextScene.primitives.find(
          (primitive) => primitive.id === updatedEmitter.meshId,
        );
        if (backedPrimitive != null && backedPrimitive.kind !== 'analytic') {
          const materialIndex = materialIndexForPrimitive(
            nextScene,
            backedPrimitive.id,
            host.supportedAnalyticShapes(),
          );
          if (materialIndex != null) {
            const folded = packFoldedMaterialEntry(backedPrimitive, nextScene, true);
            if (folded.length === MATERIAL_FLOAT_STRIDE) {
              // Preserve the sparse RGB thin-film LUT absolute offset; emitter
              // folding only changes emissive fields in the fixed record.
              folded[28 * 4 + 2] = sceneBuffers.materials[
                materialIndex * MATERIAL_FLOAT_STRIDE + 28 * 4 + 2] ?? 0;
              const materials = new Float32Array(sceneBuffers.materials);
              materials.set(folded, materialIndex * MATERIAL_FLOAT_STRIDE);
              bufferPatch.materials = materials;
            }
          }
        }
      }
    }

    const deferredWarnings: DeferredWarning[] = packed.warnings.map((warning) => ({
      warning: {
        code: 'pt-webgpu.emitter-mutation-warning',
        backend: 'pt-webgpu',
        phase: 'mutation',
        method: 'updateEmitter',
        message: `[vitrum/pt-webgpu] ${warning}`,
        details: { warning },
      },
    }));
    this.#commitPreparedMutation(
      nextScene,
      sceneBuffers,
      bufferPatch,
      undefined,
      deferredWarnings,
    );
  }

  updateEnvironment(env: Scene['environment'] | null): void {
    const host = this.#host;
    host.assertLive('updateEnvironment');
    const currentScene = host.getScene()!;
    const nextScene: Scene = {
      ...currentScene,
      environment: env ?? { kind: 'none' },
    };
    // Reject malformed environment payloads before any derived packing or GPU
    // candidate allocation. #commitPreparedMutation repeats this at the
    // transactional boundary to keep every caller on the same safety rail.
    host.validateScene?.(nextScene);
    const sceneBuffers = host.getSceneBuffers();
    if (sceneBuffers == null) {
      host.setScene(nextScene);
      return;
    }

    const packed = environmentParams(nextScene);
    const envSummaryForTree: EnvSummaryForTree = {
      hasHdri: packed.hasHdri,
      sunStrength: packed.sunStrength,
      lightTreePower: packed.lightTreePower,
    };
    const tree = packLightTreeForScene(nextScene, {
      envSummary: envSummaryForTree,
    });
    const bufferPatch: SceneBufferMutationPatch = {
      environmentMapTexels: packed.hdriTexels,
      environmentMapCdf: packed.hdriCdf,
      lightTreeNodes: tree.lightTreeNodes,
      environmentTint: packed.tint,
      environmentSunDirection: packed.sunDirection,
      environmentSunStrength: packed.sunStrength,
      environmentLightTreePower: packed.lightTreePower,
      environmentHdriIntensity: packed.hdriIntensity,
      environmentHdriRotationY: packed.hdriRotationY,
      environmentMapWidth: packed.hdriWidth,
      environmentMapHeight: packed.hdriHeight,
      hasEnvironmentMap: packed.hasHdri,
      lightTreeNodeCount: tree.lightTreeNodeCount,
      lightTreeEnabled: tree.lightTreeEnabled,
    };
    const deferredWarnings: DeferredWarning[] = packed.warnings.map((warning) => ({
      warning: {
        code: 'pt-webgpu.environment-mutation-warning',
        backend: 'pt-webgpu',
        phase: 'mutation',
        method: 'updateEnvironment',
        message: `[vitrum/pt-webgpu] ${warning}`,
        details: { warning },
      },
    }));
    this.#commitPreparedMutation(
      nextScene,
      sceneBuffers,
      bufferPatch,
      undefined,
      deferredWarnings,
    );
  }

  updateLighting(opts: Readonly<Record<string, unknown>>): void {
    const host = this.#host;
    host.assertLive('updateLighting');
    if (opts == null || typeof opts !== 'object' || Array.isArray(opts)) {
      throw new TypeError('updateLighting: options must be an object');
    }

    const knownKeys = new Set(['emitters', 'environment']);
    const unknownKeys = Object.keys(opts).filter((key) => !knownKeys.has(key));
    if (unknownKeys.length > 0) {
      throw new TypeError(
        `updateLighting: options contains unknown key(s): ${unknownKeys.join(', ')}`,
      );
    }
    const deferredWarnings: DeferredWarning[] = [];

    const hasEmitters = Object.prototype.hasOwnProperty.call(opts, 'emitters') &&
      opts.emitters !== undefined;
    const hasEnvironment = Object.prototype.hasOwnProperty.call(opts, 'environment') &&
      opts.environment !== undefined;
    if (hasEmitters && !Array.isArray(opts.emitters)) {
      throw new TypeError('updateLighting: emitters must be an array when supplied');
    }
    if (
      hasEnvironment &&
      opts.environment !== null &&
      (typeof opts.environment !== 'object' || Array.isArray(opts.environment))
    ) {
      throw new TypeError('updateLighting: environment must be a scene environment object or null');
    }
    if (!hasEmitters && !hasEnvironment) {
      return;
    }

    const currentScene = host.getScene()!;
    const nextScene: Scene = {
      ...currentScene,
      ...(hasEmitters
        ? { emitters: [...(opts.emitters as readonly SceneEmitter[])] }
        : {}),
      ...(hasEnvironment
        ? {
            environment: (opts.environment as Scene['environment'] | null) ??
              { kind: 'none' as const },
          }
        : {}),
    };
    // Validate the complete candidate before deriving emitter/environment
    // buffers. Shape-only checks above provide actionable method errors; core
    // scene validation enforces every emitter/environment numeric invariant.
    host.validateScene?.(nextScene);
    const sceneBuffers = host.getSceneBuffers();
    if (sceneBuffers == null) {
      host.setScene(nextScene);
      return;
    }

    const packedEmitters = packEmitterArrays(nextScene);
    const packedEnvironment = hasEnvironment ? environmentParams(nextScene) : null;
    const envSummary: EnvSummaryForTree = packedEnvironment == null
      ? environmentSummaryFromSceneBuffers(sceneBuffers)
      : {
          hasHdri: packedEnvironment.hasHdri,
          sunStrength: packedEnvironment.sunStrength,
          lightTreePower: packedEnvironment.lightTreePower,
        };
    const bufferPatch = emitterAndLightTreeMutationPatch(
      nextScene,
      packedEmitters,
      envSummary,
    );
    if (packedEnvironment != null) {
      Object.assign(bufferPatch, {
        environmentMapTexels: packedEnvironment.hdriTexels,
        environmentMapCdf: packedEnvironment.hdriCdf,
        environmentTint: packedEnvironment.tint,
        environmentSunDirection: packedEnvironment.sunDirection,
        environmentSunStrength: packedEnvironment.sunStrength,
        environmentLightTreePower: packedEnvironment.lightTreePower,
        environmentHdriIntensity: packedEnvironment.hdriIntensity,
        environmentHdriRotationY: packedEnvironment.hdriRotationY,
        environmentMapWidth: packedEnvironment.hdriWidth,
        environmentMapHeight: packedEnvironment.hdriHeight,
        hasEnvironmentMap: packedEnvironment.hasHdri,
      } satisfies SceneBufferMutationPatch);
    }
    if (hasEmitters && host.cameraVisibleEmitters()) {
      bufferPatch.materials = cameraVisibleEmitterMaterials(
        nextScene,
        sceneBuffers,
        host.supportedAnalyticShapes(),
      );
    }

    for (const warning of packedEmitters.warnings) {
      deferredWarnings.push({
        warning: {
          code: 'pt-webgpu.lighting-mutation-warning',
          backend: 'pt-webgpu',
          phase: 'mutation',
          method: 'updateLighting',
          message: `[vitrum/pt-webgpu] ${warning}`,
          details: { warning },
        },
      });
    }
    for (const warning of packedEnvironment?.warnings ?? []) {
      deferredWarnings.push({
        warning: {
          code: 'pt-webgpu.lighting-mutation-warning',
          backend: 'pt-webgpu',
          phase: 'mutation',
          method: 'updateLighting',
          message: `[vitrum/pt-webgpu] ${warning}`,
          details: { warning },
        },
      });
    }
    this.#commitPreparedMutation(
      nextScene,
      sceneBuffers,
      bufferPatch,
      undefined,
      deferredWarnings,
    );
  }
}
