/**
 * HybridEngine — WebGPU layered DDGI + ReSTIR DI engine.
 *
 * Class-based extraction of `useHybridLayeredGI.ts`. All React hooks stripped:
 *   - useRef      → private class fields
 *   - useState    → private class fields
 *   - useEffect   → initialize() + dispose() + setScene()
 *   - useFrame    → renderFrame()
 *   - useThree    → GPUDevice + canvas dimensions passed via factory opts
 *
 * Implements `@vitrum/core`'s `Engine` interface so a host can swap this
 * backend interchangeably with the native path-tracing backends.
 *
 * RC subsystem: cascade dispatch + shade-pass balance-heuristic MIS shipped
 * (W8 Phase 3). See plan/w8-rc-mis-composition.md.
 *
 * Debug globals:
 *   The original hook wrote to `window.__WGPU__.walkaround` and
 *   `window.__HYBRID_LAYERS__`. Those are host-bridge responsibilities.
 *   This class exposes `setLayerEnabled()` so the host can wire layer
 *   toggles; it calls `window.__WGPU__` only inside a debug branch
 *   guarded by `typeof window !== 'undefined'` and the `debug` option.
 *
 * Decomposition (refactor sweep 2026-05-18):
 *   - {@link PipelineInitCoordinator} (HybridEngineLifecycle.ts) owns the
 *     async pipeline-init race coordination + the multi-phase init chain.
 *   - {@link transformRefit} / {@link topologyRebuild}
 *     (HybridEnginePrimitiveUpdates.ts) implement the `updatePrimitive`
 *     fast / rebuild paths.
 *   - {@link TUNABLE_DEFINITIONS} (HybridEngineTuning.ts) is the single
 *     source of truth for audit-driven tuning knobs.
 *   - This file owns: public Engine API impl, construction-time options
 *     validation, debug surface, engine-state machine and reset
 *     coordination, scene synthesis + ownership, per-frame DDGI
 *     orchestration + frame throttle + telemetry mirror.
 */

import type {
  BackendSupportManifest,
  CapturedFrame,
  CaptureFrameOptions,
  Engine,
  EngineCapabilities,
  EngineFeatureId,
  EngineDebugSurface,
  EngineError,
  EngineFactory,
  EngineState,
  EngineWarning,
  FrameStats,
  ProgressStats,
} from '@vitrum/core';
import type {
  MaterialSpec,
  Scene,
  ScenePrimitive,
  SkinnedMeshPrimitive,
  SceneEmitter,
  SceneEnvironment,
} from '@vitrum/core';
import {
  analyticPrimitiveToMesh,
  canonicalizeFrameCamera,
  partitionSceneBySupport,
  supportSetsFromManifest,
  validateScene as validateCoreScene,
} from '@vitrum/core';
import type { FrameInput, FrameOutput } from '@vitrum/core';
import { asBackendTexture } from '@vitrum/core';
import type { BackendTexture } from '@vitrum/core';
import { DDGI } from './ddgi/DDGI.js';
import type { DDGILight } from './ddgi/types.js';
import {
  WalkaroundGPUPipeline,
  assertHybridDeviceCapableIfReported,
  assertNrcDeviceCapableIfReported,
  resolvePpgDispatchInterval,
} from './pipeline/WalkaroundGPUPipeline.js';
import {
  parseHybridEngineOptions,
  validateHybridEngineOptions,
  deriveHybridEngineConfig,
  type ParsedHybridEngineConfig,
} from './HybridEngineConfig.js';
import { createHybridEngineDebugSurface } from './HybridEngineDebug.js';
import type { PickCamera } from '@vitrum/shared-bvh';
import {
  fingerprintHybridPipelineRebuildKey,
  getPreferredSwapChainFormat,
  resolveInternalRenderSize,
  runHybridEngineFrame,
  HYBRID_FRAME_SKIP_OUTPUT,
  type HybridEngineFrameDeps,
  type HybridLightingDeps,
  type HybridDenoiserFilterDeps,
} from './HybridEngineFrameOrchestrator.js';
import {
  rebuildEmitterBuffersFromCoreScene,
  rebuildBvhEmissiveLeFromCoreScene,
  type ReSTIRBvhMode,
  type SceneBVHBuffers,
} from './restir/bvhCore.js';
import type { MaterialTextureAtlasDiagnostic } from './pipeline/materialTextureAtlas.js';
import { applyEmitterPatchToScene, applyPrimitivePatchToScene } from './scenePatch.js';
import { solveSkin } from '@vitrum/core';
import {
  needsAuthoredMorphStreamRestore,
  solvedSkinRenderPatch,
} from './skin/solvedSkinPatch.js';
import { readRgba16fWalkaround } from './util/gpuReadback.js';
import { SHADE_FLAG_DIRECT_SUN_SHADOW_DISABLED } from './pipeline/uboUpdater.js';
import {
  transformRefit,
  positionsRefit,
  topologyRebuild,
  materialPatch,
  materialPatchAffectsDisplacementGeometry,
  skinnedPosePatch,
  refitSkinnedMeshAfterGpuWrite,
  capturePrimitiveMutationUndo,
  SKIN_POSE_PATCH_FIELDS,
  SKIN_REST_STREAM_PATCH_FIELDS,
  TOPOLOGY_PATCH_FIELDS,
  TOPOLOGY_PATCH_WHOLESALE_FIELDS,
  type PrimitiveUpdateContext,
  type PrimitiveMutationUndo,
  type PrimitiveUpdateResult,
} from './HybridEnginePrimitiveUpdates.js';
import {
  CollectingBvhUpdateSink,
  type CollectedBvhMutation,
} from './pipeline/CollectingBvhUpdateSink.js';
import {
  commitSceneMutations,
  prepareSceneMutations,
  type PreparedSceneMutation,
} from './SceneMutationTransaction.js';
import {
  createHostSunWarningState,
  mergeDDGILightsDedupSun,
  PipelineInitCoordinator,
  type HostSunWarningState,
  type PipelineInitHost,
  type HybridInitStaticConfig,
} from './HybridEngineLifecycle.js';
import type { Tunables } from './HybridEngineTuning.js';
import {
  deriveScaleAwareClamps,
  type ScaleAwareHostExplicit,
} from './HybridEngineScaleAwareClamps.js';
import { FrameBudgetController } from './FrameBudgetController.js';
import type { FrameBudgetControllerConfig, FrameBudgetDecision } from './FrameBudgetController.js';
import type { HybridEngineOptions, LightingOptions } from './HybridEngineOptions.js';
import { assertKnownLightingKeys } from './HybridEngineOptions.js';
import {
  assertNoUnconsumedMaterialFields,
  type UnconsumedMaterialPrimitiveFields,
} from './restir/consumedMaterialFields.js';
import { RCSubsystem } from './HybridEngineRC.js';
import { MaterialApproximationWarner } from './HybridEngineMaterialWarner.js';
import { propagateBvhToGiSubsystems } from './HybridEngineGiPropagation.js';
import {
  coreEmittersToDDGILights,
  directionalSunMultiplier,
  orientDdgiSunLights,
  scenePrimaryLightDirection,
} from './coreEmittersToDDGILights.js';
import { GpuSkinningSubsystem, type SkinningBatchUpdate } from './skin/GpuSkinningSubsystem.js';
import type { GIStateSnapshot } from './giStateSnapshot.js';
import { makeGIStateCompatibility } from './giStateCompatibility.js';
import type {
  HybridRenderLayer,
} from './HybridEnginePublic.js';
import {
  resolveHybridEnvironment,
  type HybridEnvironmentResolverExtensions,
  type HybridResolvedEnvironment,
} from './environment/resolveHybridEnvironment.js';
import { exportGIStateImpl, importGIStateImpl } from './HybridEngineGIState.js';
import {
  buildDdgiLightingMutationInputs,
  syncDdgiFromCoreScene,
} from './HybridEngineDdgiSync.js';
import type { NrcDiagnostics } from './neural/nrc/nrcDiagnostics.js';
import { walkaroundSupportManifest } from './supportManifest.js';

const HYBRID_RENDER_LAYERS: ReadonlySet<string> = new Set(['ddgi']);

/**
 * Canonical internal-data encoder for estimator configuration fingerprints.
 * Object keys are sorted recursively so semantically identical host-created
 * light/config records cannot mismatch merely because their insertion order
 * differed.
 */
function canonicalGIStateCompatibilityText(
  value: unknown,
  active = new Set<object>(),
): string {
  if (value === null) return 'null';
  switch (typeof value) {
    case 'undefined':
      return 'undefined';
    case 'boolean':
      return value ? 'true' : 'false';
    case 'number':
      if (!Number.isFinite(value)) {
        throw new RangeError(
          'HybridEngine GI-state compatibility data must be finite.',
        );
      }
      return Object.is(value, -0) ? '-0' : String(value);
    case 'string':
      return JSON.stringify(value);
    case 'object': {
      const object = value;
      if (active.has(object)) {
        throw new TypeError(
          'HybridEngine GI-state compatibility data must be acyclic.',
        );
      }
      active.add(object);
      try {
        if (Array.isArray(value)) {
          return `[${value
            .map((entry) =>
              canonicalGIStateCompatibilityText(entry, active),
            )
            .join(',')}]`;
        }
        const record = value as Readonly<Record<string, unknown>>;
        const entries = Object.keys(record)
          .sort()
          .map(
            (key) =>
              `${JSON.stringify(key)}:${canonicalGIStateCompatibilityText(
                record[key],
                active,
              )}`,
          );
        return `{${entries.join(',')}}`;
      } finally {
        active.delete(object);
      }
    }
    default:
      throw new TypeError(
        `Unsupported GI-state compatibility value: ${typeof value}.`,
      );
  }
}

function encodeGIStateCompatibilityData(value: unknown): Uint8Array {
  return new TextEncoder().encode(canonicalGIStateCompatibilityText(value));
}

function sceneWithAnalyticMeshFallback(scene: Scene): Scene {
  let changed = false;
  const primitives = scene.primitives.map((primitive) => {
    if (primitive.kind !== 'analytic') return primitive;
    changed = true;
    return analyticPrimitiveToMesh(primitive);
  });
  return changed ? { ...scene, primitives } : scene;
}

/**
 * Consume the declared capability sets as a strict acceptance boundary. The
 * shared partitioner remains useful as the single kind/shape comparison, but a
 * professional backend must never publish its lossy warn-and-drop partition.
 */
function sceneAcceptedByManifest(
  scene: Scene,
  manifest: BackendSupportManifest,
  method: 'setScene' | 'updateEmitter',
): Scene {
  const partitioned = partitionSceneBySupport(
    scene,
    supportSetsFromManifest(manifest),
  );
  if (partitioned.warnings.length > 0) {
    throw new TypeError(
      `[vitrum/walkaround-hybrid] ${method}: scene capability mismatch; ` +
      `no authored primitive, emitter, analytic shape, or environment may be ` +
      `silently converted, skipped, or replaced. ${partitioned.warnings.join(' ')}`,
    );
  }
  return partitioned.supported;
}

const MESH_PATCH_KEYS = new Set([
  'kind', 'id', 'positions', 'normals', 'uvs', 'uv1', 'uvSets', 'tangents',
  'colors', 'colorSets', 'vertexColorSet', 'indices', 'material', 'transform',
  'castShadow',
]);
const INSTANCED_MESH_PATCH_KEYS = new Set([
  'kind', 'id', 'positions', 'normals', 'uvs', 'uv1', 'uvSets', 'tangents',
  'colors', 'colorSets', 'vertexColorSet', 'indices', 'material', 'instances',
  'castShadow',
]);
const ANALYTIC_PATCH_KEYS = new Set([
  'kind', 'id', 'shape', 'params', 'material', 'transform', 'castShadow',
  'fallbackMesh',
]);
const SKINNED_MESH_PATCH_KEYS = new Set([
  'kind', 'id', 'positions', 'normals', 'uvs', 'uv1', 'uvSets', 'tangents',
  'colors', 'colorSets', 'vertexColorSet', 'indices', 'skinIndices', 'skinWeights',
  'skinInfluencesPerVertex', 'bones', 'boneInverses', 'bindMatrix',
  'bindMatrixInverse', 'morphTargets', 'morphTargetNormals',
  'morphTargetTangents', 'morphTargetUvs', 'morphTargetUv1s',
  'morphTargetUvSets', 'morphTargetColors', 'morphTargetColorSets',
  'morphWeights', 'material', 'transform', 'castShadow',
]);

const MATERIAL_PATCH_KEY_RECORD = {
  baseColor: true,
  roughness: true,
  metallic: true,
  emissive: true,
  emissiveIntensity: true,
  shadingModel: true,
  alphaMode: true,
  alphaCutoff: true,
  opacity: true,
  doubleSided: true,
  transmission: true,
  ior: true,
  attenuationColor: true,
  attenuationDistance: true,
  thickness: true,
  baseColorMap: true,
  normalMap: true,
  normalScale: true,
  roughnessMap: true,
  metallicMap: true,
  transmissionMap: true,
  thicknessMap: true,
  emissiveMap: true,
  alphaMap: true,
  aoMap: true,
  aoMapIntensity: true,
  clearcoatMap: true,
  clearcoatRoughnessMap: true,
  clearcoatNormalMap: true,
  clearcoatNormalScale: true,
  sheenColorMap: true,
  sheenRoughnessMap: true,
  iridescenceMap: true,
  iridescenceThicknessMap: true,
  anisotropyMap: true,
  specularColorMap: true,
  specularIntensityMap: true,
  bumpMap: true,
  bumpScale: true,
  displacementMap: true,
  displacementScale: true,
  displacementBias: true,
  displacementSubdivisions: true,
  lightMap: true,
  lightMapIntensity: true,
  sheen: true,
  sheenColor: true,
  sheenRoughness: true,
  clearcoat: true,
  clearcoatRoughness: true,
  iridescence: true,
  iridescenceIor: true,
  iridescenceThicknessRange: true,
  specularIntensity: true,
  specularColor: true,
  envMapIntensity: true,
  spectralAttenuation: true,
  dispersionAbbeNumber: true,
  scatteringCoefficient: true,
  scatteringAnisotropy: true,
  scatteringCoefficientRGB: true,
  frontLayer: true,
  backLayer: true,
  thinFilmStack: true,
  anisotropy: true,
  anisotropyRotation: true,
  extensions: true,
} as const satisfies Readonly<Record<keyof MaterialSpec, true>>;
const MATERIAL_PATCH_KEYS = new Set(Object.keys(MATERIAL_PATCH_KEY_RECORD));
const SURFACE_LAYER_PATCH_KEYS = new Set([
  'transmission', 'roughness', 'normalMap', 'normalScale',
]);

const EMITTER_PATCH_KEYS: Readonly<Record<SceneEmitter['kind'], ReadonlySet<string>>> = {
  directional: new Set([
    'kind', 'id', 'color', 'intensity', 'castShadow', 'direction',
    'angularDiameter',
  ]),
  'disc-area': new Set([
    'kind', 'id', 'color', 'intensity', 'castShadow', 'position', 'normal',
    'radius',
  ]),
  'rect-area': new Set([
    'kind', 'id', 'color', 'intensity', 'castShadow', 'position', 'uAxis',
    'vAxis',
  ]),
  point: new Set([
    'kind', 'id', 'color', 'intensity', 'castShadow', 'position', 'distance',
    'decay',
  ]),
  spot: new Set([
    'kind', 'id', 'color', 'intensity', 'castShadow', 'position', 'direction',
    'angle', 'penumbra', 'distance', 'decay',
  ]),
  'mesh-area': new Set([
    'kind', 'id', 'color', 'intensity', 'castShadow', 'meshId',
  ]),
};

function assertPatchRecord(
  value: unknown,
  allowed: ReadonlySet<string>,
  path: string,
): asserts value is Record<string, unknown> {
  if (
    value == null || typeof value !== 'object' || Array.isArray(value)
    || ArrayBuffer.isView(value)
  ) {
    throw new TypeError(`${path} must be a plain object.`);
  }
  for (const key of Object.keys(value)) {
    if (!allowed.has(key)) {
      throw new TypeError(`${path}: unknown key "${key}".`);
    }
  }
}

function assertKnownPrimitivePatchKeys(
  primitive: ScenePrimitive,
  patch: Partial<ScenePrimitive>,
  id: string,
): void {
  const allowed = primitive.kind === 'mesh'
    ? MESH_PATCH_KEYS
    : primitive.kind === 'instanced-mesh'
      ? INSTANCED_MESH_PATCH_KEYS
      : primitive.kind === 'analytic'
        ? ANALYTIC_PATCH_KEYS
        : SKINNED_MESH_PATCH_KEYS;
  assertPatchRecord(patch, allowed, `HybridEngine.updatePrimitive("${id}") patch`);
  const material = (patch as { readonly material?: unknown }).material;
  if (material !== undefined) {
    assertPatchRecord(
      material,
      MATERIAL_PATCH_KEYS,
      `HybridEngine.updatePrimitive("${id}") patch.material`,
    );
    for (const layerKey of ['frontLayer', 'backLayer'] as const) {
      const layer = material[layerKey];
      if (layer !== undefined) {
        assertPatchRecord(
          layer,
          SURFACE_LAYER_PATCH_KEYS,
          `HybridEngine.updatePrimitive("${id}") patch.material.${layerKey}`,
        );
      }
    }
    // `material.extensions` is intentionally an open backend-extension map.
  }
}

function assertKnownEmitterPatchKeys(
  emitter: SceneEmitter,
  patch: Partial<SceneEmitter>,
  id: string,
): void {
  assertPatchRecord(
    patch,
    EMITTER_PATCH_KEYS[emitter.kind],
    `HybridEngine.updateEmitter("${id}") patch`,
  );
}

function assertPositiveSafeViewportDimension(value: unknown, label: string): asserts value is number {
  if (typeof value !== 'number') {
    throw new TypeError(`${label} must be a number.`);
  }
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`${label} must be a positive safe integer (got ${String(value)}).`);
  }
}

function assertFiniteFrameArray(value: unknown, length: number, label: string): void {
  if (
    value == null ||
    typeof value !== 'object' ||
    !('length' in value) ||
    (value as { readonly length?: unknown }).length !== length
  ) {
    throw new TypeError(
      `HybridEngine.renderFrame: ${label} must be an array-like value of length ${length}.`,
    );
  }
  const values = value as ArrayLike<unknown>;
  for (let index = 0; index < length; index += 1) {
    const component = values[index];
    if (typeof component !== 'number' || !Number.isFinite(component)) {
      throw new RangeError(
        `HybridEngine.renderFrame: ${label}[${index}] must be finite (got ${String(component)}).`,
      );
    }
  }
}

function assertFiniteFrameNumber(value: unknown, label: string): asserts value is number {
  if (typeof value !== 'number') {
    throw new TypeError(`HybridEngine.renderFrame: ${label} must be a number.`);
  }
  if (!Number.isFinite(value)) {
    throw new RangeError(
      `HybridEngine.renderFrame: ${label} must be finite (got ${String(value)}).`,
    );
  }
}

function assertFrameU32(value: unknown, label: string): asserts value is number {
  assertFiniteFrameNumber(value, label);
  if (!Number.isInteger(value) || value < 0 || value > 0xffffffff) {
    throw new RangeError(
      `HybridEngine.renderFrame: ${label} must be an integer in 0..4294967295 (got ${String(value)}).`,
    );
  }
}

/** Validate every host-controlled frame field before renderFrame reads or mutates engine state. */
function validateHybridFrameInput(input: FrameInput): void {
  if (input == null || typeof input !== 'object' || Array.isArray(input)) {
    throw new TypeError('HybridEngine.renderFrame: input must be a FrameInput object.');
  }
  assertFiniteFrameArray(input.viewMatrix, 16, 'viewMatrix');
  assertFiniteFrameArray(input.projMatrix, 16, 'projMatrix');
  if (input.cameraPosition !== undefined) {
    assertFiniteFrameArray(input.cameraPosition, 3, 'cameraPosition');
  }
  if (input.prevViewMatrix !== undefined) {
    assertFiniteFrameArray(input.prevViewMatrix, 16, 'prevViewMatrix');
  }
  if (input.prevProjMatrix !== undefined) {
    assertFiniteFrameArray(input.prevProjMatrix, 16, 'prevProjMatrix');
  }
  const viewport = input.viewport;
  if (viewport == null || typeof viewport !== 'object' || Array.isArray(viewport)) {
    throw new TypeError('HybridEngine.renderFrame: viewport must be an object.');
  }
  assertPositiveSafeViewportDimension(
    viewport.width,
    'HybridEngine.renderFrame: viewport.width',
  );
  assertPositiveSafeViewportDimension(
    viewport.height,
    'HybridEngine.renderFrame: viewport.height',
  );
  if (typeof viewport.devicePixelRatio !== 'number') {
    throw new TypeError('HybridEngine.renderFrame: viewport.devicePixelRatio must be a number.');
  }
  if (!Number.isFinite(viewport.devicePixelRatio) || viewport.devicePixelRatio <= 0) {
    throw new RangeError(
      `HybridEngine.renderFrame: viewport.devicePixelRatio must be finite and > 0 (got ${String(viewport.devicePixelRatio)}).`,
    );
  }
  assertFrameU32(input.frameIndex, 'frameIndex');
  assertFrameU32(input.frameSeed, 'frameSeed');

  const quality = input.quality;
  if (quality === undefined) return;
  if (quality === null || typeof quality !== 'object' || Array.isArray(quality)) {
    throw new TypeError('HybridEngine.renderFrame: quality must be an object when supplied.');
  }
  for (const field of ['samplesTarget', 'bounces'] as const) {
    const value = quality[field];
    if (value === undefined) continue;
    assertFiniteFrameNumber(value, `quality.${field}`);
    if (!Number.isInteger(value)) {
      throw new RangeError(
        `HybridEngine.renderFrame: quality.${field} must be an integer (got ${String(value)}).`,
      );
    }
    if (field === 'bounces' && (!Number.isSafeInteger(value) || value <= 0)) {
      throw new RangeError(
        `HybridEngine.renderFrame: quality.bounces must be a positive safe integer (got ${String(value)}).`,
      );
    }
  }
  if (quality.resolutionFactor !== undefined) {
    assertFiniteFrameNumber(quality.resolutionFactor, 'quality.resolutionFactor');
  }
  if (quality.exposure !== undefined) {
    assertFiniteFrameNumber(quality.exposure, 'quality.exposure');
    if (quality.exposure < 0) {
      throw new RangeError(
        `HybridEngine.renderFrame: quality.exposure must be >= 0 (got ${String(quality.exposure)}).`,
      );
    }
  }
  if (quality.filteredGlossyFactor !== undefined) {
    assertFiniteFrameNumber(quality.filteredGlossyFactor, 'quality.filteredGlossyFactor');
    if (quality.filteredGlossyFactor !== 0) {
      throw new RangeError(
        'HybridEngine.renderFrame: quality.filteredGlossyFactor is unsupported by walkaround-hybrid; use 0 or omit it.',
      );
    }
  }
  if (
    quality.tonemap !== undefined &&
    !['aces', 'agx', 'reinhard', 'linear', 'none'].includes(quality.tonemap)
  ) {
    throw new RangeError(
      `HybridEngine.renderFrame: quality.tonemap is unsupported (got ${String(quality.tonemap)}).`,
    );
  }
  if (
    quality.outputColorSpace !== undefined &&
    quality.outputColorSpace !== 'srgb' &&
    quality.outputColorSpace !== 'linear'
  ) {
    throw new RangeError(
      `HybridEngine.renderFrame: quality.outputColorSpace is unsupported (got ${String(quality.outputColorSpace)}).`,
    );
  }
  // Validate the entire quality payload before applying backend-dependency
  // rejection, matching construction-option validation ordering.
  if (quality.samplesTarget !== undefined) {
    throw new RangeError(
      'HybridEngine.renderFrame: quality.samplesTarget is unsupported by walkaround-hybrid ' +
        'because this realtime backend does not accumulate toward a sample target; ' +
        'accepting it would be a no-op, so omit it.',
    );
  }
}

// Re-export the option / lighting interfaces from their dedicated module so
// the package's public surface (`./HybridEngine.js` import path) stays
// unchanged after the type split (refactor sweep 2026-05-18).
export type { HybridEngineOptions, LightingOptions } from './HybridEngineOptions.js';

// ────────────────────────────────────────────────────────────────────────────
// Option parsing + validation
//
// `ParsedHybridEngineConfig`, `validateHybridEngineOptions`,
// `deriveHybridEngineConfig`, and `parseHybridEngineOptions` have been moved to
// `HybridEngineConfig.ts` (R3 B-chain decomposition sweep). Re-imported above.
//
// `HybridEngineOptions` + `LightingOptions` interface bodies live in
// `HybridEngineOptions.ts` (~340 LOC of pure JSDoc, extracted refactor sweep
// 2026-05-18). Re-exported above so the package surface is unchanged.
// ────────────────────────────────────────────────────────────────────────────

// Unused import guard: `validateHybridEngineOptions` and `deriveHybridEngineConfig`
// are exported by HybridEngineConfig.ts and re-exported from this file for
// back-compat (any external caller that imported them from HybridEngine.ts directly).
export { validateHybridEngineOptions, deriveHybridEngineConfig };

// `readRgba16fWalkaround` moved to `src/util/gpuReadback.ts` (R3 B-chain step 2).
// Re-imported above.

function buildWalkaroundActiveFeatures(
  opts: HybridEngineOptions,
  cfg: ParsedHybridEngineConfig,
): ReadonlySet<EngineFeatureId> {
  const features = new Set<EngineFeatureId>();
  features.add('walkaround-hybrid-gris-ddgi-proxy-reuse');
  if (cfg.ppgEnabled === 1) features.add('walkaround-hybrid-ppg-guided-gi');
  if (cfg.nrcEnabled === 1) features.add('walkaround-hybrid-nrc');
  if (opts.rcEnabled === true) features.add('walkaround-hybrid-radiance-cascades');
  if (cfg.regirConfig?.enabled === true) features.add('walkaround-hybrid-regir');
  if (opts.gpuSkinning === true) features.add('walkaround-hybrid-gpu-skinning');
  if (opts.causticStrategy === 'refractive-trace') {
    features.add('walkaround-hybrid-refractive-trace-caustics');
  } else if (opts.causticStrategy === 'manifold-nee') {
    features.add('walkaround-hybrid-manifold-nee-caustics');
  }
  if (cfg.denoiser !== 'none') {
    const denoiserFeatures: Record<
      Exclude<ParsedHybridEngineConfig['denoiser'], 'none'>,
      EngineFeatureId
    > = {
      atrous: 'walkaround-hybrid-denoiser-atrous',
      'atrous-variance': 'walkaround-hybrid-denoiser-atrous-variance',
      'svgf-real': 'walkaround-hybrid-denoiser-svgf-real',
      bmfr: 'walkaround-hybrid-denoiser-bmfr',
      neural: 'walkaround-hybrid-denoiser-neural',
      'oidn-final': 'walkaround-hybrid-denoiser-oidn-final',
    };
    features.add(denoiserFeatures[cfg.denoiser]);
  }
  return features;
}

// ────────────────────────────────────────────────────────────────────────────
// HybridEngine
// ────────────────────────────────────────────────────────────────────────────

const ERROR_THROTTLE_FRAME_WINDOW = 32;
const ERROR_THROTTLE_WALL_CLOCK_MS = 1_000;
const ERROR_THROTTLE_MAX_IDENTITIES = 256;

export class HybridEngine implements Engine {
  // ── Engine contract fields ─────────────────────────────────────────────
  private _state: EngineState = 'uninitialized';
  /** Host pause intent survives asynchronous pipeline publication/rebuilds. */
  private _pauseRequested = false;
  readonly capabilities: EngineCapabilities;

  get state(): EngineState {
    return this._state;
  }

  // ── Creation-time options (immutable after construction) ───────────────
  private readonly _device: GPUDevice;
  // Mutable since T-resize: the host calls `setSize()` whenever the
  // canvas resizes; the pipeline reallocates its FrameResources without
  // a full engine teardown. See `setSize()` for the resize contract.
  //
  // Phase-0 productization — `_width/_height` are the CANVAS (swap-chain)
  // dimensions (what the composite pass blits TO + what `setSize` sets). The
  // INTERNAL render resolution (what the compute kernels dispatch over) is
  // `_internalWidth/_internalHeight = canvas × _resolutionFactor`; it equals
  // the canvas dims when no `quality.resolutionFactor` downscale is active.
  // The two are kept in sync by `setSize` (recomputes internal from the last
  // factor) and by the per-frame `quality.resolutionFactor` path in
  // `HybridEngineFrameOrchestrator` (debounced internal resize).
  private _width: number;
  private _height: number;
  /** Monotonic authored-scene generation used by warning identity dedup. */
  private _sceneGeneration = 0;
  /** Environment warning identities already emitted for the current scene. */
  private readonly _environmentWarningKeys = new Set<string>();
  /**
   * Material-approximation / truthfulness warning subsystem (T3-A extraction).
   * Owns the 9 once-only dedup sets + every `warnX` method HybridEngine used to
   * inline. The private `_warn*` methods below delegate to it; the engine's
   * `_warn` (console + subscriber fan-out) is injected as the sink.
   */
  private readonly _materialWarner = new MaterialApproximationWarner((warning) =>
    this._warn(warning),
  );
  /** Internal render width = `_width × _resolutionFactor`. Drives compute
   *  dispatch + UBO `screenSize`; the composite upscales to `_width`. */
  private _internalWidth: number;
  /** Internal render height = `_height × _resolutionFactor`. */
  private _internalHeight: number;
  /** Last-seen `FrameInput.quality.resolutionFactor` (clamped to (0,1]).
   *  Default 1.0 (internal == canvas). */
  private _resolutionFactor: number = 1.0;
  /** `performance.now()` of the last accepted resolution-factor resize, for
   *  the debounce that prevents accumulator thrash (Risk R5). */
  private _lastResolutionResizeTs: number = 0;
  /** Optional host environment resolver extension. Used only by
   *  updateEnvironment() to reduce opaque HDRI handles into the diffuse
   *  sky-dome scalars this backend consumes. */
  private readonly _environmentResolverExtensions: HybridEnvironmentResolverExtensions | null;
  /** Last environment payload resolved and validated before scene publication.
   * Reused by async init so an opaque-handle resolver is called exactly once
   * per authored mutation and cannot fail after live scene state has changed. */
  private _resolvedEnvironment: HybridResolvedEnvironment = {
    mode: 'none',
    skyIrradiance: 0,
    warnings: [],
  };
  /** Optional host readiness predicate. Core mesh presence remains required. */
  private readonly _isSceneReady: () => boolean;
  // Lighting fields are NOT readonly — updateLighting() mutates them at runtime.
  private _primaryLightDir: [number, number, number];
  /**
   * Constructor primaryLightDir is the legacy no-scene fallback. A runtime
   * updateLighting(primaryLightDir) call is an explicit persistent host
   * override of authored scene directional directions.
   */
  private _primaryLightDirOverrideActive = false;
  private _primaryLightIntensity: number;
  private _skyTint: [number, number, number];
  private _skyIrradiance: number;
  private readonly _ctorLights: readonly DDGILight[];
  /** Resolved caustic estimator. `refractive-trace` is a bounded realtime
   * path sampler and is intentionally not labelled as manifold NEE. */
  private readonly _causticStrategy: EngineCapabilities['causticStrategy'];

  /** Rolling window of per-frame timings (newest last, cap 240 entries).
   *  Only populated when `debug === true`. Hosts that want a UI gauge
   *  should poll {@link debugTimings} instead of reaching into globals. */
  private readonly _debugTimings: Array<{ t: number; ms: number }> = [];

  /** T3.E — telemetry subscribers fired at end of each successful renderFrame. */
  private readonly _frameSubs: Array<(s: FrameStats) => void> = [];

  /** T3.E — long-running progress subscribers fired at the end of each
   *  dispatched frame, once per still-converging signal (`'ddgi-warmup'`
   *  while the probe grid warms, `'denoiser-converge'` while the temporal
   *  accumulator fills). Empty until a host calls {@link onProgress}. */
  private readonly _progressSubs: Array<(p: ProgressStats) => void> = [];

  /** GPU error subscribers (item 28). */
  private readonly _errorSubs: Array<(e: EngineError) => void> = [];

  /** Structured non-fatal warning subscribers. */
  private readonly _warningSubs: Array<(w: EngineWarning) => void> = [];
  /** Dedup-throttle: event identity → last frame + wall-clock emission. */
  private _errorThrottleMap = new Map<
    string,
    { readonly frame: number; readonly timeMs: number }
  >();
  /** Frame counter for error throttle (incremented in renderFrame). */
  private _errorFrameCount = 0;
  /** Bound uncapturederror handler — stored so it can be removed on dispose. */
  private _onUncapturedError: ((e: Event) => void) | null = null;

  /** Read-only snapshot of recent frame timings collected when the engine
   *  was constructed with `debug: true`. Returns an empty array when debug
   *  is off (no allocation cost is paid in production). */
  get debugTimings(): readonly { t: number; ms: number }[] {
    return this._debugTimings;
  }

  /** Per-pass GPU timings in milliseconds, keyed by the same `PassLabel`
   *  set the timestamp-query subsystem uses. Empty record when the active
   *  adapter doesn't expose `timestamp-query` or when the engine is not yet
   *  initialised. Useful for dev panels + telemetry harnesses. */
  get lastGpuTimings(): Record<string, number> {
    return this._pipeline?.lastGpuTimings ?? {};
  }

  /** Frame index that produced {@link lastGpuTimings}; -1 if no readback
   *  has resolved yet. */
  get lastGpuTimingsFrame(): number {
    return this._pipeline?.lastGpuTimingsFrame ?? -1;
  }

  /**
   * Diagnostic one-shot GPU timing readback (P3-Vδ). Bypasses the
   * production fire-and-forget ping-pong path and synchronously awaits a
   * fresh staging-buffer mapAsync. Use this from telemetry probes that
   * need a confirmed-fresh per-pass timing snapshot. Returns empty
   * objects when the engine isn't ready or the device lacks the
   * `timestamp-query` feature.
   */
  async readGpuTimingsOnce(): Promise<{ perPass: Record<string, number>; rawBigints: string[] }> {
    if (!this._pipeline) return { perPass: {}, rawBigints: [] };
    return this._pipeline.readGpuTimingsOnce();
  }

  /**
   * Snapshot the live Neural Radiance Cache health/cadence counters.
   * Returns `null` when NRC is disabled or the pipeline has not reached ready.
   */
  getNrcDiagnostics(): NrcDiagnostics | null {
    return this._pipeline?.getNrcDiagnostics() ?? null;
  }

  /**
   * Runtime DDGI probe-update divisor (round-robin stride) setter. The
   * construction-time value comes from the quality preset / `ddgiUpdateDivisor`
   * option; this lets a host (or the adaptive frame-budget controller) retune
   * the GI-refresh cadence per frame without recreating the engine. Clamped to
   * ≥ 1 by the DDGI subsystem. Cheap (a couple of JS field writes + a UBO field
   * the next probe-update pass picks up); no atlas teardown.
   */
  setDdgiUpdateDivisor(divisor: number): void {
    this._assertNotDisposed('setDdgiUpdateDivisor');
    this._ddgi.setProbeUpdateDivisor(divisor);
  }

  /**
   * Opt-in: turn ON the closed-loop adaptive frame-budget controller (Phase
   * IV.1 / review gap D1). Idempotent — re-calling with a new config replaces
   * the controller's config but the engine still owns no cadence: NOTHING
   * happens until the host feeds measured ms via {@link tickFrameBudget}.
   *
   * Seeds the controller's initial knobs from the engine's current state (the
   * preset/option `resolutionFactor` + `ddgiUpdateDivisor`) so the loop starts
   * from where the static path left off, then backs off / climbs from there.
   *
   * The controller defaults target ~60 fps; pass `{ targetMs: 33.3 }` for 30
   * fps, etc. See {@link FrameBudgetControllerConfig}.
   */
  enableFrameBudget(config: Partial<FrameBudgetControllerConfig> = {}): void {
    this._assertNotDisposed('enableFrameBudget');
    this._frameBudget = new FrameBudgetController(
      {
        adaptPpgDispatchInterval: this._cfg.ppgEnabled === 1,
        ...config,
      },
      {
        resolutionFactor: this._resolutionFactor,
        ddgiStride: this._cfg.ddgiUpdateDivisor,
        ppgDispatchInterval: this._ppgDispatchInterval,
      },
    );
  }

  /** Opt-out: turn OFF the adaptive frame-budget controller. The knobs are left
   *  wherever the loop last set them (the host may restore them explicitly). */
  disableFrameBudget(): void {
    this._assertNotDisposed('disableFrameBudget');
    this._frameBudget = null;
  }

  /** True when the adaptive frame-budget loop is enabled. */
  get frameBudgetEnabled(): boolean {
    return this._frameBudget !== null;
  }

  /**
   * Opt-in adaptive-quality tick. The host calls this ONCE PER FRAME with a
   * measured frame time (ms) — from the wall-clock `FrameStats.frameTimeMs` of
   * an `onFrame` subscriber, or from {@link readGpuTimingsOnce}'s confirmed GPU
   * `total` — and the controller nudges the engine's quality knobs toward the
   * configured budget.
   *
   * Consistent with "the host owns cadence", this does NOT schedule itself and
   * does NOT read frame time on its own; the host drives it. It applies the
   * engine-owned knobs (DDGI stride, and PPG train cadence when PPG is enabled)
   * directly, and returns the decision so the host can feed the PRIMARY knob
   * back as the next frame's `FrameInput.quality.resolutionFactor` (the
   * resolution lever is a per-frame host input by contract — `renderFrame`
   * consumes `quality.resolutionFactor`, debounced).
   *
   * No-op (returns `null`) when the controller is not enabled — so a host can
   * call it unconditionally in its render loop and pay nothing until it opts in
   * via {@link enableFrameBudget}.
   *
   * @returns the {@link FrameBudgetDecision} (whose `resolutionFactor` the host
   *          should apply next frame), or `null` if the loop is disabled.
   */
  tickFrameBudget(measuredMs: number): FrameBudgetDecision | null {
    this._assertNotDisposed('tickFrameBudget');
    if (!this._frameBudget) return null;
    const decision = this._frameBudget.update(measuredMs);
    // Apply engine-owned runtime knobs immediately; the primary lever
    // (resolutionFactor) is a host per-frame input by the FrameInput contract,
    // so the host applies it via the returned decision.
    this.setDdgiUpdateDivisor(decision.ddgiStride);
    if (this._cfg.ppgEnabled === 1) {
      this.setPpgDispatchInterval(decision.ppgDispatchInterval);
    }
    return decision;
  }

  /**
   * Runtime PPG train-pass dispatch interval. Requires an engine constructed
   * with `ppgEnabled:true`; a disabled PPG pipeline has no cadence to mutate.
   * Exposed mainly for the adaptive frame-budget controller and host A/B knobs.
   */
  setPpgDispatchInterval(interval: number): void {
    this._assertNotDisposed('setPpgDispatchInterval');
    if (this._cfg.ppgEnabled !== 1) {
      throw new Error(
        'HybridEngine.setPpgDispatchInterval requires construction with ppgEnabled:true.',
      );
    }
    const resolved = resolvePpgDispatchInterval(interval);
    this._ppgDispatchInterval = resolved;
    this._pipeline?.setPpgDispatchInterval(resolved);
  }

  /**
   * The construction-time-immutable derived config (Task 4.2 / Theme A).
   * `parseHybridEngineOptions(opts)` produces this once in the constructor;
   * every tunable-cluster value the engine used to splat onto an individual
   * `this._x` field is now read from `this._cfg.x`. Holding the parsed record
   * directly collapses ~25 one-by-one ctor assignments + their field
   * declarations, so a single tunable no longer hops a private-field layer.
   *
   * ONLY construction-immutable values live here. Genuinely per-instance
   * MUTABLE runtime state (lighting, size/internal-size, accumulators, the
   * rebuild-key fingerprint, BVH/pipeline/scene handles) stays in its own
   * field below because it changes after construction.
  */
  private readonly _cfg: ParsedHybridEngineConfig;
  /** Backend-owned executable support contract for this exact tier/model
   *  profile. Live capabilities and scene acceptance both derive from it. */
  private readonly _supportManifest: BackendSupportManifest;
  /** Runtime PPG cadence retained independently of pipeline lifetime. The init
   *  host reads this live value, so pre-init calls and reset/rebuild cycles
   *  preserve the host's latest setting. */
  private _ppgDispatchInterval: number;
  /** Dev A/B — mirrors `engine.debug.setDenoiserEnabled` (default on). */
  private _denoiserPassEnabled = true;

  // ── B15 — scene-scale-aware radiometric clamp defaults ──────────────────
  /** Per-knob flags: did the HOST explicitly set this clamp? Host overrides are
   *  NEVER scaled (an explicit tuning value passes through verbatim). Captured
   *  once at construction from the options. */
  private readonly _clampHostExplicit: ScaleAwareHostExplicit;
  /** Scene-scale-derived per-frame tunables (B15). Computed at `setScene` from
   *  the scene's world diagonal; `null` before the first scene (falls back to
   *  the Cornell-baseline `_cfg.tunables`). At Cornell scale this is
   *  byte-identical to `_cfg.tunables` (the law short-circuits at ratio 1). */
  private _scaledTunables: Tunables | null = null;
  /** Scene-scale-derived `indirectFireflyClamp` (B15). `null` ⇒ use the
   *  `_cfg.indirectFireflyClamp` baseline. */
  private _scaledIndirectFireflyClamp: readonly [number, number, number] | null = null;

  // ── Pipeline state ─────────────────────────────────────────────────────
  private _pipeline: WalkaroundGPUPipeline | null = null;
  private _bvhBuffers: SceneBVHBuffers | null = null;

  // ── Scene (from @vitrum/core contract) ────────────────────────────────
  /** Last authored scene accepted via `setScene()`. Incremental updates patch
   *  this snapshot so analytic primitives retain their `shape` / `params`
   *  semantics even when the renderer consumes generated mesh fallbacks. */
  private _lastScene: Scene | null = null;

  /** Render-ingestion view derived from {@link _lastScene}: analytic primitives
   *  become deterministic MeshPrimitive fallbacks with the same id/material/
   *  transform. BVH, DDGI, ReSTIR, RC, and THREE conversion consume this view. */
  private _renderScene: Scene | null = null;

  /** Last-frame camera (copied) for debug click-to-pick (`pickPrimitive`, T3.G).
   *  Captured each `renderFrame`; null until the first frame. */
  private _lastFrameCamera: PickCamera | null = null;

  // ── DDGI subsystem ─────────────────────────────────────────────────────
  private _ddgi: DDGI;
  /** Once-only conflict-warning state shared by init and incremental lighting
   *  sync for this engine instance. */
  private readonly _hostSunWarningState: HostSunWarningState =
    createHostSunWarningState();
  private _ddgiOn: boolean = true;

  // ── Adaptive frame-budget controller (opt-in; Phase IV.1 / review gap D1) ─
  /** Lazily-created on the first {@link enableFrameBudget}/{@link tickFrameBudget}
   *  call. Null ⇒ the closed loop is OFF and the static quality path is
   *  untouched (the engine never reads frame time on its own). */
  private _frameBudget: FrameBudgetController | null = null;

  // ── RC subsystem (W8 Phase 2 — opt-in via opts.rcEnabled) ───────────────
  private _rc: RCSubsystem | null = null;
  /** W8 Phase 3 — balance-heuristic MIS weight for RC in Lo_indirect.
   *  Effective only when _rc != null. Default 0.5 when rcEnabled is true. */
  private _rcWeight: number = 0;

  // ── Per-frame throttle ─────────────────────────────────────────────────
  private _lastFrameTs = 0;

  // ── Diagnostic counters (debug only) ──────────────────────────────────
  private _dbg = {
    initStart: 0,
    initCount: 0,
    disposeCount: 0,
    skipNoPipeline: 0,
    skipNoBvh: 0,
    skipNoSwapView: 0,
    skipFrameInterval: 0,
    framesDispatched: 0,
    lastReportTs: 0,
  };

  // ── Layer toggles (debug console interface) ────────────────────────────
  private _layerEnabled: Map<HybridRenderLayer, boolean> = new Map([['ddgi', true]]);

  // ── Pipeline init coordinator (see HybridEngineLifecycle.ts) ──────────
  //
  // Owns the monotonic init-sequence + dispose race coordination + multi-
  // phase async init chain. The engine delegates init/dispose flow to it
  // via the {@link PipelineInitHost} surface built in `_buildInitHost()`.
  private readonly _initCoordinator: PipelineInitCoordinator;

  // Task 4.2 / Theme A — the construction-immutable tunable-cluster values live
  // on `_cfg` (one parsed record instead of ~25 splatted `_x` fields). Consumers
  // read `this._cfg.x` directly; tests pin resolved knobs via the `_cfg` seam.

  /** Monotonic fingerprint of {@link HybridEngineOptions.pipelineRebuildKey} /
   *  {@link HybridEngineOptions.getPipelineRebuildKey} — changes trigger `reset()`. */
  private _rebuildKeyFingerprintSeen: string;

  private readonly _skinning: GpuSkinningSubsystem | null;

  readonly debug: EngineDebugSurface;

  constructor(opts: HybridEngineOptions) {
    // Pure option parsing + validation (defaults, denoiser/neural/OIDN
    // validation throws) lives in `parseHybridEngineOptions` so the
    // constructor body stays focused on `this`-dependent wiring (subsystems,
    // capabilities, init coordinator, debug surface). Behaviour-preserving:
    // same throws in the same order, same defaults. (WD decomposition sweep.)
    // Task 4.2 / Theme A — hold the parsed config in one `_cfg` field rather
    // than splatting its ~25 construction-immutable values onto individual
    // `this._x` fields. Consumers read `this._cfg.x`; only genuinely-mutable
    // runtime state (device handle, size, lighting, accumulators, the rebuild-
    // key fingerprint) gets its own field.
    const cfg = parseHybridEngineOptions(opts);
    assertHybridDeviceCapableIfReported(opts.device.limits);
    if (cfg.nrcEnabled === 1) assertNrcDeviceCapableIfReported(opts.device.limits);
    this._cfg = cfg;
    this._supportManifest = walkaroundSupportManifest({
      tier: opts.tier === 'lite' ? 'lite' : 'full',
      neuralCertification:
        cfg.neuralWeights == null
          ? 'absent'
          : cfg.neuralCheckpointAssessment.productionReady
            ? 'certified'
            : 'uncertified',
      oidnModelAvailable: cfg.oidnModelUrl != null,
    });
    this._ppgDispatchInterval = cfg.ppgDispatchInterval;
    if (opts.onWarning != null) {
      this._warningSubs.push(opts.onWarning);
    }
    if (cfg.denoiserAutoResolution != null) {
      const r = cfg.denoiserAutoResolution;
      this._warn({
        code: 'walkaround-hybrid.denoiser-auto-resolved',
        backend: 'walkaround-hybrid',
        phase: 'construction',
        method: 'createWalkaroundEngine_Hybrid',
        message:
          `[HybridEngine] denoiser:'auto' resolved to '${r.resolved}' (${r.reason}). ` +
          `The package does not ship production neural weights; neural auto-selection ` +
          `requires host neuralWeights with production checkpoint metadata, and OIDN only when the host ` +
          `provides an OIDN model URL.`,
        details: {
          requested: r.requested,
          resolved: r.resolved,
          reason: r.reason,
          packageProvidesProductionWeights: r.packageProvidesProductionWeights,
          defaultEnabled: r.defaultEnabled,
          neuralCheckpointProductionReady: r.neuralCheckpointProductionReady,
          neuralCheckpointMissing: r.neuralCheckpointMissing,
          neuralDeviceFailure: r.neuralDeviceFailure,
          neuralTensorPrecision: r.neuralTensorPrecision,
        },
      });
    }

    // B15 — capture which radiometric clamps the HOST set explicitly. These
    // bypass scene-scale scaling (an explicit override is absolute). None of
    // the three scalar clamps has a subsystem sub-object, so `opts.tuning` is
    // the only override path; `indirectFireflyClamp` is its own top-level field.
    this._clampHostExplicit = {
      restirGiIrrClamp: opts.tuning?.restirGiIrrClamp !== undefined,
      directFireflyClamp: opts.tuning?.directFireflyClamp !== undefined,
      emitterDist2Floor: opts.tuning?.emitterDist2Floor !== undefined,
      indirectFireflyClamp: opts.indirectFireflyClamp !== undefined,
    };

    // H46-A — maxBounces now drives a REAL control surface on this realtime
    // stack: the DDGI indirect-feedback gate. The walkaround engine does NOT
    // path-trace, so maxBounces is NOT a per-ray bounce cap here — it has exactly
    // two regimes: 1 ⇒ DIRECT-ONLY probes (the DDGI atlas folds in one bounce of
    // direct light per probe and the infinite-bounce diffuse EMA is disabled);
    // >= 2 ⇒ the full multi-bounce diffuse equilibrium (the default; the atlas
    // EMA converges to infinite diffuse bounces). Intermediate/large values
    // (2, 3, 4, …) are all identical to the default >= 2 regime because the EMA
    // converges to the bounce limit regardless of the integer value — only the
    // 1-vs-many distinction is meaningful. Invalid values are rejected by
    // parseHybridEngineOptions before any engine/GPU state is created.
    const requestedCausticStrategy = opts.causticStrategy ?? 'none';
    this._causticStrategy = requestedCausticStrategy === 'refractive-trace' ||
      requestedCausticStrategy === 'manifold-nee'
      ? requestedCausticStrategy
      : 'none';
    if (opts.nrcEnabled === true) {
      this._warn({
        code: 'walkaround-hybrid.nrc-biased-estimator-enabled',
        backend: 'walkaround-hybrid',
        phase: 'construction',
        method: 'createWalkaroundEngine_Hybrid',
        message:
          `[HybridEngine] nrcEnabled:true enables the stable opt-in Neural Radiance ` +
          `Cache. NRC is a biased radiance-cache estimator for realtime GI; its ` +
          `default-disabled posture is product policy, and the estimator bias is ` +
          `disclosed so hosts can make an informed scene-quality choice.`,
        details: {
          implementationStatus: 'stable',
          nrcEnabled: true,
          defaultEnabled: false,
          estimator: 'biased',
        },
      });
    }
    if (cfg.denoiser === 'neural') {
      this._warn({
        code: 'walkaround-hybrid.neural-host-weights-required',
        backend: 'walkaround-hybrid',
        phase: 'construction',
        method: 'createWalkaroundEngine_Hybrid',
        message:
          `[HybridEngine] denoiser:'neural' is a stable GPU U-Net runtime that ` +
          `requires a host-supplied certified v2 checkpoint. The package bundles ` +
          `no certified weights and does not enable neural by default; uncertified ` +
          `or legacy checkpoints are rejected.`,
        details: {
          implementationStatus: 'stable',
          denoiser: 'neural',
          weightsRequired: true,
          packageProvidesProductionWeights: false,
          defaultEnabled: false,
          checkpointProductionReady: cfg.neuralCheckpointAssessment.productionReady,
          checkpointMissing: cfg.neuralCheckpointAssessment.missing,
        },
      });
    }

    this._device = opts.device;
    this._width = opts.width;
    this._height = opts.height;
    // Phase-0 — the quality preset supplies the INITIAL internal-resolution
    // factor. A per-frame `quality.resolutionFactor` still overrides at runtime
    // (`_applyResolutionFactor`); this is just the starting point so a
    // `qualityTier:'low'` engine boots at 0.5 internal res.
    this._resolutionFactor = cfg.resolutionFactor;
    this._internalWidth = Math.max(1, Math.round(opts.width * cfg.resolutionFactor));
    this._internalHeight = Math.max(1, Math.round(opts.height * cfg.resolutionFactor));
    this._skinning = opts.gpuSkinning ? new GpuSkinningSubsystem(opts.device, true) : null;
    this._environmentResolverExtensions = opts.extensions ?? null;
    this._primaryLightDir = opts.primaryLightDir;
    this._primaryLightIntensity = opts.primaryLightIntensity;
    this._skyTint = opts.skyTint;
    this._skyIrradiance = opts.skyIrradiance;
    this._isSceneReady = opts.isSceneReady ?? (() => true);

    // `_rebuildKeyFingerprintSeen` is the one rebuild-key value that MUTATES
    // post-construction (`consumeRebuildKeyChange` rewrites it), so it stays a
    // mutable field seeded from the parsed config. The static key + getter live
    // on `_cfg` (immutable).
    this._rebuildKeyFingerprintSeen = cfg.rebuildKeyFingerprintSeen;

    this._ddgi = new DDGI({
      debug: this._cfg.debug,
      ...(opts.ddgiMaxMaterials !== undefined ? { maxMaterials: opts.ddgiMaxMaterials } : {}),
      onError: (error) => this._emitError(error),
      onWarning: (warning) => this._warn(warning),
    });
    this._ddgi.setSkyParams?.(this._skyTint, this._skyIrradiance);
    // Phase-0 — apply the quality-preset DDGI probe-update divisor (default 4).
    this._ddgi.setProbeUpdateDivisor(this._cfg.ddgiUpdateDivisor);
    // H46-A — gate the DDGI indirect-feedback (multi-bounce diffuse EMA) on the
    // engine's maxBounces. maxBounces == 1 ⇒ direct-only probes; >= 2 ⇒ the
    // infinite-bounce equilibrium (default). Construction-immutable, and the
    // ProbeUpdatePass is created once (never recreated), so one call persists.
    this._ddgi.setIndirectFeedback(this._cfg.maxBounces >= 2);
    this._ctorLights = opts.lights ?? [];
    if (this._ctorLights.length > 0) {
      this._ddgi.setLights(orientDdgiSunLights(this._ctorLights, this._primaryLightDir));
    }

    // Opt-in RC subsystem. Its cascade buffers and compact material/TLAS
    // adapter are private, while BLAS geometry is borrowed from the canonical
    // pipeline scene arena after initialization. Dispatch remains per-frame.
    if (opts.rcEnabled === true) {
      // B3b — Cornell-tuned CASCADE_DIMS default lives in walkaround-rc;
      // hosts override via opts.cascadeDims for non-Cornell aspect ratios
      // or scene scales.
      this._rc =
        opts.cascadeDims !== undefined
          ? new RCSubsystem(this._device, opts.cascadeDims, {
              onWarning: (warning) => this._warn(warning),
              transmittedInterfaceBudget: cfg.rcTransmittedInterfaceBudget,
            })
          : new RCSubsystem(this._device, undefined, {
              onWarning: (warning) => this._warn(warning),
              transmittedInterfaceBudget: cfg.rcTransmittedInterfaceBudget,
            });
      // W8 Phase 3 — host-overridable MIS weight (default 0.5 = equal
      // mix with ReSTIR-GI). When rcEnabled is false the weight stays 0
      // and pipeline.setRCInputs(null) routes the bind group to the
      // placeholder buffers (rcParams.enabled = 0u short-circuits the
      // shader's sample to vec3f(0)).
      const rcWeight = opts.rcWeight;
      this._rcWeight = Number.isFinite(rcWeight) ? Math.max(0, Math.min(1, rcWeight!)) : 0.5;
    }

    const supportSets = supportSetsFromManifest(this._supportManifest);
    const mutationDetails = this._supportManifest.mutations;
    const mutationAccepted = (
      mode: (typeof mutationDetails)[keyof typeof mutationDetails],
    ): boolean => mode !== 'unsupported';
    this.capabilities = {
      // Incremental scene patches are implemented: transform/positions fast
      // paths plus full-rebuild fallbacks for material/topology edits, and
      // emitter patching via scene-level rebuild.
      supportsIncrementalScene:
        mutationAccepted(mutationDetails.transform) ||
        mutationAccepted(mutationDetails.positions) ||
        mutationAccepted(mutationDetails.material) ||
        mutationAccepted(mutationDetails.emitter) ||
        mutationAccepted(mutationDetails.topology),
      incrementalPatchSupport: {
        transform: mutationAccepted(mutationDetails.transform),
        positions: mutationAccepted(mutationDetails.positions),
        material: mutationAccepted(mutationDetails.material),
        emitter: mutationAccepted(mutationDetails.emitter),
        topology: mutationAccepted(mutationDetails.topology),
      },
      // Explicit whole-primitive add/remove IS implemented: addPrimitive appends
      // a new primitive and removePrimitive evicts one, each by routing a fresh
      // mutated `Scene` copy through this engine's existing `setScene` spine —
      // the same `partitionSceneBySupport` → `_teardownPipeline` → init-chain
      // (full BVH / DDGI / ReSTIR rebuild + temporal-accumulator reset) the
      // initial scene build runs. A geometry change invalidates every cached GI
      // signal, so on this realtime stack the work is a rebuild either way; the
      // value is API consistency with pt-webgl / pt-webgpu, not a perf win. The
      // DDGI / ReSTIR / RC subsystems index off the packed scene, so reusing the
      // setScene packing path re-syncs them all correct-by-construction — no
      // fragile per-array index remap. Distinct from
      // incrementalPatchSupport.topology (COUNT-change patches on an EXISTING
      // primitive).
      supportsAddRemovePrimitive:
        mutationAccepted(mutationDetails.addPrimitive) &&
        mutationAccepted(mutationDetails.removePrimitive),
      // All four full-resolution auxiliary signals are surfaced on rendered
      // frames, including the freshest side of the Welford variance ping-pong.
      supportsAuxBuffers: this._supportManifest.motionVectors != null,
      // The post-denoise resolvedTexture is exposed via getProgressiveSeedTexture()
      // as the seed source for progressive walkaround→PT handoff (P8).
      supportsProgressiveSeedSource: true,
      accumulates: false,
      maxSamplesPerPixel: Infinity,
      // H46-A — echoes the authored value. SEMANTICS for this realtime stack:
      // this is NOT a path-tracer per-ray bounce cap. 1 ⇒ direct-only DDGI
      // probes; >= 2 ⇒ infinite-bounce diffuse equilibrium (the atlas EMA). All
      // values >= 2 behave identically (the EMA converges regardless of the
      // integer). See the construction-site gate `setIndirectFeedback`.
      maxBounces: this._cfg.maxBounces,
      supportedAnalyticShapes: supportSets.supportedAnalyticShapes,
      // BVH + DDGI ingest via a render-scene view. Mesh/skinned/instanced-mesh
      // flow through directly; analytic primitives are accepted in the authored
      // scene and converted to deterministic MeshPrimitive fallbacks before
      // ReSTIR / DDGI / RC consume them.
      supportedPrimitiveKinds: supportSets.supportedPrimitiveKinds,
      // Emitter kinds that genuinely reach a renderable state:
      //   - rect-area / disc-area → harvested as ReSTIR-DI direct emitter tris
      //     from core emitter data AND projected
      //     to DDGI fixture lights (coreEmittersToDDGILights) for indirect bounce.
      //   - mesh-area → folded into the referenced mesh's emissive material, so
      //     it reaches both the ReSTIR-DI emissive-triangle path and DDGI as
      //     emissive geometry.
      //   - point / spot → analytic direct-light terms in shade/OIT plus
      //     DDGI/RC fixture-light uploads via coreEmittersToDDGILights. Spots
      //     include real cone data (spotAxis + cosInner/cosOuter);
      //     evalPointLight in the probe shaders applies the smoothstep falloff
      //     when the axis is non-zero (axisLen² > 0.25). Points carry a zero
      //     axis and remain omnidirectional.
      //   - directional → projected to a DDGI `sun` light by
      //     coreEmittersToDDGILights, carrying the emitter's REAL direction
      //     (negated to a travel direction), intensity, and colour into the
      //     probe pass's sun path (replacing the packer's former hardcoded
      //     straight-down warm-white sun). Single-counted: the host sets the
      //     sun-intensity multiplier to 1 when a scene directional is present,
      //     so the emitter intensity is not double-applied. The directional
      //     still drives the shade-side Lo_emit via the WalkaroundUBO config
      //     path (primaryLightDir/Intensity) — those remain host config, not
      //     derived from the emitter, so there is no shade-side double-count.
      supportedEmitterKinds: supportSets.supportedEmitterKinds,
      // procedural-sky bakes through resolveHybridEnvironment into a finite
      // Preetham equirect + CDF. It remains approximate due model/resolution
      // limits, not because turbidity/rayleigh/mie are dropped.
      supportedEnvironmentKinds: supportSets.supportedEnvironmentKinds,
      presentationMode: 'swapchain-required',
      supportDetails: this._supportManifest,
      activeFeatures: buildWalkaroundActiveFeatures(opts, this._cfg),
      // Exact selected estimator identity. `refractive-trace` is the bounded
      // realtime path sampler in refractiveCaustics.wgsl.ts, not MNEE.
      causticStrategy: this._causticStrategy,
      // W3-D8 — this engine ships a `debug` surface (DDGI atlases, BVH nodes,
      // GI signal textures, and now estimatedGpuMemoryBytes). Hosts can
      // structurally opt-in to the dev-overlay panel without typeof-checking
      // every method.
      debugSurface: true,
    };

    // Pipeline-init coordinator: own the async init race state machine and
    // dispose coordination. The host adapter built below grants the
    // coordinator the small surface it needs without exposing private
    // engine fields directly.
    this._initCoordinator = new PipelineInitCoordinator(this._buildInitHost());

    this.debug = createHybridEngineDebugSurface({
      device: () => this._device,
      readAtlas: () => this._ddgi?.getReadAtlasGPUTextures() ?? null,
      bvhNodesCpu: () => this._bvhBuffers?.bvhNodes?.cpuData,
      debugTextures: () => this._pipeline?.getDebugTextures() ?? null,
      getMemoryBreakdown: () => this._pipeline?.getMemoryBreakdown(
        this._rc == null ? {} : { rc: this._rc.gpuMemorySection() },
      ) ?? null,
      // T3.G click-to-pick: the retained core scene + last-frame camera + canvas
      // size feed the CPU ray-cast in createHybridEngineDebugSurface.
      pickScene: () => this._lastScene,
      pickCamera: () => this._lastFrameCamera,
      pickSize: () => ({ width: this._width, height: this._height }),
      denoiserPassEnabled: () => this._denoiserPassEnabled,
      setDenoiserPassEnabled: (enabled) => {
        this._denoiserPassEnabled = enabled;
      },
      setPipelineDenoiserPassEnabled: (enabled) => {
        this._pipeline?.setDenoiserPassEnabled(enabled);
      },
    });

    // ── GPU error wiring (item 28) ─────────────────────────────────────────
    // Attach an `uncapturederror` listener on the WebGPU device to route
    // validation/internal errors to the host via onError subscribers.
    // Throttled: one report per distinct message per 32 frames to avoid spam.
    // Listener is removed on dispose (engine does not own the device).
    this._onUncapturedError = (event: Event): void => {
      if (this._state === 'disposed') return;
      const gpuEvent = event as { error?: { message?: string } };
      const rawError = gpuEvent.error;
      // eslint-disable-next-line @typescript-eslint/no-base-to-string -- fallback stringification of GPUUncapturedErrorEvent; acceptable for diagnostic messages
      const message = rawError?.message ?? String(event);
      const kind: EngineError['kind'] =
        rawError != null && rawError.constructor?.name === 'GPUInternalError'
          ? 'gpu-internal'
          : 'gpu-validation';
      const identity = `${kind}\u0000${message}`;
      const nowMs = Date.now();
      const last = this._errorThrottleMap.get(identity);
      const frameWindowElapsed =
        last == null ||
        this._errorFrameCount - last.frame >= ERROR_THROTTLE_FRAME_WINDOW;
      const wallClockWindowElapsed =
        last == null ||
        nowMs < last.timeMs ||
        nowMs - last.timeMs >= ERROR_THROTTLE_WALL_CLOCK_MS;
      if (frameWindowElapsed || wallClockWindowElapsed) {
        if (
          last == null &&
          this._errorThrottleMap.size >= ERROR_THROTTLE_MAX_IDENTITIES
        ) {
          const oldest = this._errorThrottleMap.keys().next().value;
          if (oldest !== undefined) this._errorThrottleMap.delete(oldest);
        }
        this._errorThrottleMap.set(identity, {
          frame: this._errorFrameCount,
          timeMs: nowMs,
        });
        this._emitError({ kind, message, fatal: false, raw: rawError });
      }
    };
    opts.device.addEventListener('uncapturederror', this._onUncapturedError);

    // device.lost: fatal transition to 'error' state.
    opts.device.lost
      .then((info: { reason?: string; message?: string }) => {
        if (this._state === 'disposed') return;
        this._state = 'error';
        this._emitError({
          kind: 'device-lost',
          message: info.message ?? `GPUDevice lost (reason: ${info.reason ?? 'unknown'})`,
          fatal: true,
          raw: info,
        });
      })
      .catch(() => {
        /* spec says it shouldn't reject; guard defensively */
      });
  }

  // ── Scene management ───────────────────────────────────────────────────

  // Material-approximation / truthfulness warnings are owned by
  // `_materialWarner` (T3-A extraction). These private methods delegate so the
  // call sites (setScene / updatePrimitive / updateEmitter / the mutation
  // router / lifecycle) keep their existing shape.

  private _warnUnconsumedMaterialFields(
    fields: readonly string[],
    method: 'setScene' | 'updatePrimitive',
    primitiveFields: readonly UnconsumedMaterialPrimitiveFields[] = [],
  ): void {
    this._materialWarner.warnUnconsumedMaterialFields(fields, method, primitiveFields);
  }

  private _warnMaterialTextureAtlasDiagnostics(
    diagnostics: readonly MaterialTextureAtlasDiagnostic[],
    method: 'setScene' | 'updatePrimitive',
  ): void {
    this._materialWarner.warnMaterialTextureAtlasDiagnostics(diagnostics, method);
  }

  /**
   * Replace the scene. Triggers a full pipeline reinitialisation
   * (BVH rebuild + ReSTIR pipeline re-init).
   *
   * **BVH + DDGI geometry:** ReSTIR, DDGI, and RC consume the core scene's
   * mesh/skinned/instanced primitives directly.
   *
   * **Host guidance:** pass a canonical `@vitrum/core` Scene. Host-specific
   * scene adapters live outside this package.
   *
   * **Capability validation + analytic fallback:** the scene is checked
   * against this engine's declared `supported*Kinds`; any mismatch throws.
   * Supported
   * analytic primitives stay in the authored `_lastScene`, then `_renderScene`
   * replaces them with generated MeshPrimitive fallbacks before the BVH/GI
   * ingestion path runs.
   *
   * @param inputScene - The `@vitrum/core` scene.
   */
  setScene(inputScene: Scene): void {
    if (this._state === 'disposed') {
      throw new Error('HybridEngine.setScene: engine is disposed.');
    }
    validateCoreScene(inputScene);
    const resolvedEnvironment = resolveHybridEnvironment(
      inputScene.environment ?? { kind: 'none' },
      { extensions: this._environmentResolverExtensions },
    );
    const scene = sceneAcceptedByManifest(
      inputScene,
      this._supportManifest,
      'setScene',
    );
    assertNoUnconsumedMaterialFields(
      scene.primitives as unknown as ReadonlyArray<{
        readonly id?: string;
        readonly kind: string;
        readonly material?: Record<string, unknown>;
      }>,
      'setScene',
    );

    const renderScene = sceneWithAnalyticMeshFallback(scene);
    const scaleAwareClamps = deriveScaleAwareClamps(renderScene, {
      baseTunables: this._cfg.tunables,
      baseIndirectFireflyClamp: this._cfg.indirectFireflyClamp,
      hostExplicit: this._clampHostExplicit,
    });
    this._sceneGeneration++;
    this._environmentWarningKeys.clear();
    this._emitResolvedEnvironmentWarnings(resolvedEnvironment, 'setScene');
    this._lastScene = scene;
    this._renderScene = renderScene;
    this._resolvedEnvironment = resolvedEnvironment;
    this._applyResolvedEnvironmentScalars(resolvedEnvironment);

    // B15 — derive scene-scale-aware radiometric clamp DEFAULTS from the new
    // scene's world diagonal. Uses the render-scene view (analytic primitives
    // already meshed) so the AABB is computed over real positions. At Cornell
    // scale the law short-circuits to byte-identical defaults; host-explicit
    // clamps pass through un-scaled. These feed the per-frame deps below (NOT
    // the UBO layout — the UBO stays frozen; only the host-computed VALUES move).
    this._publishScaleAwareClamps(scaleAwareClamps);

    // W8 Phase 2: rebuild the RC BVH + cascade buffers after async ReSTIR
    // BVH publish via the core-native path.

    // Tear down the existing pipeline, reinitialise asynchronously.
    this._teardownPipeline();
    this._initCoordinator.startInit();
  }

  /** Read back the retained canonical core {@link Scene} (`_lastScene` — the
   *  capability-filtered `supported` authored scene), or null before the
   *  first `setScene`. Implements the optional `Engine.getScene` contract — see
   *  its JSDoc for the no-defensive-copy / frozen-by-contract semantics. The
   *  reference survives {@link dispose}; the facade wrapper gates the
   *  post-dispose read. */
  getScene(): Scene | null {
    return this._lastScene;
  }

  /**
   * B15 — recompute the scene-scale-aware radiometric clamp defaults from the
   * current `_renderScene`'s world diagonal and cache them on
   * `_scaledTunables` / `_scaledIndirectFireflyClamp` (read by the per-frame
   * deps builders). Pure host-side arithmetic — no GPU, no UBO-layout change.
   *
   * INVARIANTS:
   *   • At Cornell scale (diagonal ≈ {@link CORNELL_DIAGONAL}) the result is
   *     byte-identical to the `_cfg` baselines (the law short-circuits at
   *     scaleRatio == 1).
   *   • Host-explicit clamps (`_clampHostExplicit[knob]`) pass through un-scaled.
   *
   * Called from `setScene` (the only diagonal-changing entry — add/remove/
   * topology-patch routes all funnel through `setScene`, so a single hook here
   * keeps the scaled defaults in sync with the live geometry).
   */
  private _publishScaleAwareClamps(
    result: ReturnType<typeof deriveScaleAwareClamps>,
  ): void {
    this._scaledTunables = result.tunables;
    this._scaledIndirectFireflyClamp = result.indirectFireflyClamp;
    if (this._cfg.verbose && Math.abs(result.scaleRatio - 1) > 1e-6) {
      console.log(
        `[HybridEngine] B15 scale-aware clamps: sceneDiagonal=${result.sceneDiagonal.toFixed(3)} ` +
          `(×${result.scaleRatio.toFixed(3)} vs Cornell) → ` +
          `restirGiIrrClamp=${result.tunables.restirGiIrrClamp.toExponential(3)}, ` +
          `directFireflyClamp=${result.tunables.directFireflyClamp.toExponential(3)}, ` +
          `emitterDist2Floor=${result.tunables.emitterDist2Floor.toExponential(3)} ` +
          `(host-explicit knobs un-scaled).`,
      );
    }
  }

  // ── updatePrimitive — geometry-change path ─────────────────────────────
  //
  // **Routing rules**:
  //  - `patch.transform` present AND no topology fields → fast-path (c):
  //     refit the BVH bounds in-place (no SAH rebuild, no pipeline
  //     recompile), rewrite the affected primitive's vertex slice in
  //     `bvhPositions` for merged BVH, reset the accumulator, and invalidate
  //     DDGI probes so cached irradiance follows the moved object.
  //  - vertex/index topology or packed-attribute field present (`positions` /
  //     `normals` / UVs / tangents / vertex-color streams or selection /
  //     `indices`) → full-rebuild path (a): re-run
  //     `buildReSTIRSceneBVH`, destroy + reupload all four BVH GPU
  //     buffers, reset the accumulator.
  //  - `instances` / `params` / `shape` / `fallbackMesh` → route through a
  //     full `setScene` rebuild (P5 contract-honesty; see the
  //     `TOPOLOGY_PATCH_WHOLESALE_FIELDS` branch below). A
  //     geometry/instance change invalidates every cached GI signal on this
  //     realtime stack anyway, so honoring `incrementalPatchSupport.topology`
  //     beats throwing "call setScene()" and matches pt-webgl/pt-webgpu.
  //  - `id` / `kind` are canonical discriminants: changing either throws in
  //     `patchPrimitiveInScene`; explicitly repeating the current value is
  //     neutral and does not trigger any rebuild.
  //  - skinned definition/pose fields (skin indices/weights, bind matrices,
  //     bones, morph targets, or morph weights) → solve through `solveSkin` and
  //     reuse the positions/normals refit path while preserving authored fields.
  //  - material-only patches → `materialPatch` fast path (A3): re-pack the
  //     affected `bvhIndex` / `bvhBeerColors` triangle slices + partial GPU
  //     upload — NO `setScene`, no pipeline recompile.
  //
  // Implementations live in `HybridEnginePrimitiveUpdates.ts`; this method
  // is the routing dispatcher.
  //
  // Implements `Engine.updatePrimitive(id, patch)` from `@vitrum/core`.
  updatePrimitive(id: string, patch: Partial<ScenePrimitive>): void {
    if (this._state === 'disposed') {
      throw new Error('HybridEngine.updatePrimitive: engine is disposed.');
    }
    if (this._state === 'initializing') {
      throw new Error(
        `HybridEngine.updatePrimitive("${id}"): engine is initializing. ` +
          `Wait for setScene init to finish before applying primitive patches.`,
      );
    }
    if (this._lastScene == null) {
      throw new Error(
        `HybridEngine.updatePrimitive("${id}"): no scene set. ` +
          `Call setScene(scene) before updatePrimitive.`,
      );
    }
    const primIndex = this._lastScene.primitives.findIndex((p) => String(p.id) === id);
    if (primIndex < 0) {
      throw new Error(
        `HybridEngine.updatePrimitive("${id}"): primitive id not found in current scene.`,
      );
    }
    assertKnownPrimitivePatchKeys(this._lastScene.primitives[primIndex]!, patch, id);
    const preflightScene = applyPrimitivePatchToScene(this._lastScene, id, patch);
    validateCoreScene(preflightScene);
    sceneAcceptedByManifest(preflightScene, this._supportManifest, 'setScene');
    assertNoUnconsumedMaterialFields(
      preflightScene.primitives as unknown as ReadonlyArray<{
        readonly id?: string;
        readonly kind: string;
        readonly material?: Record<string, unknown>;
      }>,
      'updatePrimitive',
    );

    // Canonical preflight above has already rejected changed discriminants.
    // Equal `id` / `kind` values carry no state change, so remove them before
    // route selection. This also keeps a meaningful co-patch (for example,
    // `{ kind: 'mesh', material: ... }`) on its normal fast/rebuild path.
    const routedPatch = { ...patch };
    delete (routedPatch as { id?: unknown }).id;
    delete (routedPatch as { kind?: unknown }).kind;
    if (!Object.values(routedPatch).some((value) => value !== undefined)) {
      return;
    }

    // Wholesale-replacement patches — `instances` (instanced-mesh
    // instance-COUNT change), `params` / `shape` (analytic), and
    // `fallbackMesh` can't be expressed as an in-place packed-buffer edit, so
    // route them through a full setScene rebuild (the same mutate-Scene → setScene spine addPrimitive /
    // removePrimitive use). A geometry/instance change invalidates every cached GI
    // signal anyway, so on this realtime stack the work is a rebuild either way;
    // the value is honoring incrementalPatchSupport.topology + matching
    // pt-webgl/pt-webgpu (which absorb the instance-COUNT case) instead of
    // throwing "call setScene()". P5 contract-honesty.
    if (
      TOPOLOGY_PATCH_WHOLESALE_FIELDS.some(
        (f) => (routedPatch as Record<string, unknown>)[f] !== undefined,
      )
    ) {
      this.setScene(preflightScene);
      return;
    }

    // Three.js BVH refit (fast path) preserves topology when only AABB
    // bounds need updating. Three flavours:
    //   - Transform only       → transformRefit  (~1 ms / 30k tris)
    //   - Positions only       → positionsRefit  (A3, ~1 ms / 30k tris)
    //   - True topology change → topologyRebuild (~50 ms / 30k tris)
    // "Positions only" means new vertex data on the SAME index buffer +
    // SAME vertex count. The positionsRefit path falls through to
    // topologyRebuild internally if the count doesn't match.
    const collector = new CollectingBvhUpdateSink();
    const undo = capturePrimitiveMutationUndo(this._bvhBuffers, id, routedPatch);
    let result: PrimitiveUpdateResult | null;
    try {
      result = this._routePrimitiveUpdate(
        id,
        routedPatch,
        this._buildPrimitiveUpdateContext(collector),
      );
    } catch (error) {
      undo.restore();
      throw error;
    }
    if (result == null) {
      // No recognised patch field: keep the forgiving no-op behavior, but make
      // the integration mistake visible through the structured warning channel.
      this._warnUnknownPrimitivePatchFields(
        id,
        Object.entries(routedPatch as Record<string, unknown>)
          .filter(([, value]) => value !== undefined)
          .map(([field]) => field),
      );
      undo.restore();
      return;
    }
    try {
      this._commitPrimitiveMutation(result, collector.snapshot(), undo);
    } catch (error) {
      undo.restore();
      throw error;
    }
  }

  /**
   * Select the fast/full path for an `updatePrimitive` patch and run it.
   * Returns `null` for an unrecognised patch (no-op). Branch order is
   * load-bearing — mixed material+geometry patches rebuild so the whole patch is
   * applied; otherwise skinned authored deformation beats structural topology
   * beats ordinary positions, transform, then material:
   *  - structural topology fields (`indices` / UVs / tangents / vertex-color
   *    streams or selection / `castShadow`) → full
   *    SAH `topologyRebuild` (Option (a)).
   *  - a skinned primitive's pose fields or authored rest streams →
   *    `skinnedPosePatch`, which solves exactly once before refit/rebuild.
   *  - an ordinary mesh's `positions` with optional same-count `normals` →
   *    A3/H19 `positionsRefit` (same topology, new verts/normals).
   *  - `normals` without `positions` → full rebuild until a normals-only
   *    upload path exists.
   *  - `transform` only → `transformRefit` (refit AABB bounds in place).
   *  - `material` only → `materialPatch` (re-pack slices, NO GI propagation;
   *    the result carries `applySubsystems: false`).
   */
  private _routePrimitiveUpdate(
    id: string,
    patch: Partial<ScenePrimitive>,
    ctx: PrimitiveUpdateContext = this._buildPrimitiveUpdateContext(),
  ): PrimitiveUpdateResult | null {
    const has = (f: string): boolean => (patch as Record<string, unknown>)[f] !== undefined;
    const hasMaterial = has('material');
    const hasSkinnedPose = SKIN_POSE_PATCH_FIELDS.some((f) => has(f));
    const currentPrimitive =
      ctx.lastScene.primitives.find((primitive) => String(primitive.id) === id);
    const hasSkinnedRestStream =
      currentPrimitive?.kind === 'skinned-mesh' &&
      SKIN_REST_STREAM_PATCH_FIELDS.some((field) => has(field));
    if (hasSkinnedPose || hasSkinnedRestStream) {
      return skinnedPosePatch(id, patch, ctx);
    }
    const hasStructuralTopology = TOPOLOGY_PATCH_FIELDS.some((f) => f !== 'normals' && has(f));
    if (hasStructuralTopology) return topologyRebuild(id, patch, ctx);
    if (hasMaterial && (has('positions') || has('normals') || has('transform'))) {
      return topologyRebuild(id, patch, ctx);
    }
    if (has('positions')) return positionsRefit(id, patch, ctx);
    if (has('normals')) return topologyRebuild(id, patch, ctx);
    if (has('transform')) return transformRefit(id, patch, ctx);
    if (has('material')) {
      const previousPrimitive = ctx.lastScene.primitives.find((p) => String(p.id) === id);
      const previousMaterial =
        previousPrimitive != null && 'material' in previousPrimitive
          ? previousPrimitive.material
          : undefined;
      if (
        materialPatchAffectsDisplacementGeometry(
          previousMaterial,
          patch.material as unknown as Parameters<
            typeof materialPatchAffectsDisplacementGeometry
          >[1],
        )
      ) {
        return topologyRebuild(id, patch, ctx);
      }
      return materialPatch(id, patch, ctx);
    }
    return null;
  }

  private _warnUnknownPrimitivePatchFields(id: string, fields: readonly string[]): void {
    this._materialWarner.warnUnknownPrimitivePatchFields(id, fields);
  }

  /**
   * Apply one solved skin pose through the same staged primitive transaction.
   * This public single-item seam performs a normal position-slice upload; the
   * internal GPU batch seam below supplies an unsubmitted compute command buffer.
   */
  applyGpuSkinnedRefit(
    id: string,
    localPositions?: Float32Array,
    localNormals?: Float32Array,
  ): void {
    const prim = this._lastScene?.primitives.find(
      (candidate) => String(candidate.id) === id && candidate.kind === 'skinned-mesh',
    );
    if (prim?.kind !== 'skinned-mesh') {
      throw new Error(`applyGpuSkinnedRefit("${id}"): skinned-mesh primitive not found.`);
    }
    const renderedPrim = this._renderScene?.primitives.find(
      (candidate) => String(candidate.id) === id && candidate.kind === 'skinned-mesh',
    );
    const solved = localPositions == null ? solveSkin(prim) : null;
    const patch: Partial<SkinnedMeshPrimitive> = solved == null
      ? {
          positions: localPositions!,
          ...(localNormals != null ? { normals: localNormals } : {}),
        }
      : solvedSkinRenderPatch(
          prim,
          solved,
          needsAuthoredMorphStreamRestore(
            prim,
            renderedPrim?.kind === 'skinned-mesh' ? renderedPrim : null,
          ),
        );
    this.applySkinningBatch([{ id, patch, gpuWritten: false }], null);
  }

  /**
   * Publish one animation tick as a single transaction. The optional skin
   * command buffer is already encoded against the live position/normal buffers
   * but is deliberately unsubmitted; the pipeline submits it immediately before
   * the staged BVH/learned-state copies in its final irreversible operation.
   */
  applySkinningBatch(
    updates: readonly SkinningBatchUpdate[],
    skinCommands: GPUCommandBuffer | null,
  ): void {
    if (this._state === 'disposed') {
      throw new Error('HybridEngine.applySkinningBatch: engine is disposed.');
    }
    if (this._state === 'initializing') {
      throw new Error(
        'HybridEngine.applySkinningBatch: engine is initializing; wait for setScene.',
      );
    }
    if (updates.length === 0) return;
    const ids = updates.map((update) => update.id);
    if (new Set(ids).size !== ids.length) {
      throw new Error('HybridEngine.applySkinningBatch: duplicate primitive id.');
    }

    const topology = updates.some((update) =>
      TOPOLOGY_PATCH_FIELDS.some(
        (field) =>
          field !== 'normals' &&
          (update.patch as unknown as Record<string, unknown>)[field] !== undefined,
      ),
    );
    if (topology && skinCommands != null) {
      throw new Error(
        'HybridEngine.applySkinningBatch: topology fallback cannot target live GPU buffers.',
      );
    }

    const collector = new CollectingBvhUpdateSink();
    const undoPatch: Record<string, unknown> = {};
    for (const update of updates) {
      for (const [field, value] of Object.entries(
        update.patch as unknown as Record<string, unknown>,
      )) {
        if (value !== undefined) undoPatch[field] = value;
      }
    }
    const undo = capturePrimitiveMutationUndo(
      this._bvhBuffers,
      updates[0]!.id,
      undoPatch,
      ids.slice(1),
    );
    let result: PrimitiveUpdateResult;
    try {
      const authoredScene = this._lastScene;
      const initialRenderScene = this._renderScene;
      const initialBvh = this._bvhBuffers;
      if (authoredScene == null || initialRenderScene == null || initialBvh == null) {
        throw new Error('HybridEngine.applySkinningBatch: scene/BVH not ready.');
      }
      const currentScene: Scene = authoredScene;
      let currentRenderScene: Scene = initialRenderScene;
      let currentBvh: SceneBVHBuffers = initialBvh;

      if (topology) {
        for (let i = 0; i < updates.length - 1; i += 1) {
          const update = updates[i]!;
          currentRenderScene = applyPrimitivePatchToScene(
            currentRenderScene,
            update.id,
            update.patch,
          );
        }
        const last = updates[updates.length - 1]!;
        result = topologyRebuild(last.id, last.patch, {
          ...this._buildPrimitiveUpdateContext(collector),
          bvhBuffers: currentBvh,
          lastScene: currentScene,
          renderScene: currentRenderScene,
        });
        result = {
          ...result,
          updatedScene: currentScene,
          updatedRenderScene: result.updatedRenderScene ??
            applyPrimitivePatchToScene(currentRenderScene, last.id, last.patch),
        };
      } else {
        let latest: PrimitiveUpdateResult | null = null;
        for (const update of updates) {
          const positions = update.patch.positions;
          if (positions == null) {
            throw new Error(
              `HybridEngine.applySkinningBatch("${update.id}"): positions are required.`,
            );
          }
          const context: PrimitiveUpdateContext = {
            ...this._buildPrimitiveUpdateContext(collector),
            bvhBuffers: currentBvh,
            lastScene: currentScene,
            renderScene: currentRenderScene,
          };
          latest = update.gpuWritten
            ? refitSkinnedMeshAfterGpuWrite(
                update.id,
                new Float32Array(positions),
                update.patch.normals != null
                  ? new Float32Array(update.patch.normals)
                  : undefined,
                context,
              )
            : positionsRefit(update.id, update.patch, context);
          const nextRenderScene = latest.updatedRenderScene ??
            applyPrimitivePatchToScene(currentRenderScene, update.id, update.patch);
          latest = {
            ...latest,
            updatedScene: currentScene,
            updatedRenderScene: nextRenderScene,
          };
          currentRenderScene = nextRenderScene;
          currentBvh = latest.bvhBuffers;
        }
        result = latest!;
      }
    } catch (error) {
      undo.restore();
      throw error;
    }

    try {
      this._commitPrimitiveMutation(
        result,
        collector.snapshot(),
        undo,
        skinCommands != null ? [skinCommands] : [],
      );
    } catch (error) {
      undo.restore();
      throw error;
    }
  }

  /** Merged BVH position SSBO for GPU skinning (null before pipeline init). */
  getGpuSkinningBvhBuffer(): GPUBuffer | null {
    return this._pipeline?.getBvhPositionBuffer() ?? null;
  }

  getGpuSkinningBvhBinding(): GPUBufferBinding | null {
    return this._pipeline?.getBvhPositionBinding() ?? null;
  }

  /** WS1 — merged BVH normal SSBO for GPU skinning (null before pipeline init).
   *  The skin kernel writes inverse-transpose skinned normals here at
   *  `baseVertex+vi` so the smooth shading-normal blend reads deformed normals. */
  getGpuSkinningNormalBuffer(): GPUBuffer | null {
    return this._pipeline?.getBvhNormalBuffer() ?? null;
  }

  getGpuSkinningNormalBinding(): GPUBufferBinding | null {
    return this._pipeline?.getBvhNormalBinding() ?? null;
  }

  /** Per-mesh vertex ranges in the merged BVH (for GPU skinning). */
  getMeshVertexRanges(): SceneBVHBuffers['meshVertexRanges'] | null {
    return this._bvhBuffers?.meshVertexRanges ?? null;
  }

  /** Active ReSTIR BVH layout (`merged` world positions vs `tlas` local BLAS). */
  getBvhMode(): ReSTIRBvhMode | null {
    return this._bvhBuffers?.bvhMode ?? null;
  }

  getPrimitiveTlasBindings(): SceneBVHBuffers['primitiveTlasBindings'] | null {
    return this._bvhBuffers?.primitiveTlasBindings ?? null;
  }

  /**
   * After geometry BVH updates: sync DDGI probe rays + RC cascades to the live
   * ReSTIR buffers without waiting for the next `renderFrame` tick.
   */
  private _applyPrimitiveUpdateSubsystems(result: PrimitiveUpdateResult): void {
    propagateBvhToGiSubsystems({
      ddgi: this._ddgi,
      rc: this._rc,
      bvhBuffers: this._bvhBuffers,
      lastScene: this._renderScene,
      syncDdgi: true,
      allowRcSceneRebuild: true,
      rcRefitBounds: result.rcRefitBounds,
    });
  }

  /** Build the per-call resource context the primitive-update helpers consume. */
  private _buildPrimitiveUpdateContext(
    transactionalSink?: CollectingBvhUpdateSink,
  ): PrimitiveUpdateContext {
    if (this._lastScene == null) {
      throw new Error('HybridEngine.updatePrimitive: no scene set. Call setScene(scene) first.');
    }
    if (this._renderScene == null) {
      this._renderScene = sceneWithAnalyticMeshFallback(this._lastScene);
    }
    const ctx: PrimitiveUpdateContext = {
      bvhBuffers: this._bvhBuffers,
      pipeline: transactionalSink ?? this._pipeline,
      deferSubsystemSideEffects: transactionalSink != null,
      ddgi: this._ddgi,
      primaryLightDir: this._effectivePrimaryLightDir(this._renderScene),
      primaryLightIntensity: this._primaryLightIntensity,
      lastScene: this._lastScene,
      renderScene: this._renderScene,
      coreSceneSuppliesMeshes: this._coreSceneSuppliesMeshes(),
      warnUnconsumedMaterialFields: (fields, primitiveFields) => {
        this._warnUnconsumedMaterialFields(fields, 'updatePrimitive', primitiveFields);
      },
      onWarning: (warning) => {
        this._warn(warning);
      },
    };
    if (this._cfg.restirBvhModeOverride !== undefined) {
      return { ...ctx, restirBvhModeOverride: this._cfg.restirBvhModeOverride };
    }
    return ctx;
  }

  private _commitPrimitiveMutation(
    result: PrimitiveUpdateResult,
    mutation: CollectedBvhMutation,
    undo: PrimitiveMutationUndo,
    prefixCommandBuffers: readonly GPUCommandBuffer[] = [],
  ): void {
    const previousBvh = this._bvhBuffers;
    const previousScene = this._lastScene;
    const previousRenderScene = this._renderScene;
    const nextRenderScene = result.updatedRenderScene ??
      sceneWithAnalyticMeshFallback(result.updatedScene);
    const geometryChanged = result.applySubsystems !== false;
    let stateCommitted = false;
    const stateMutation: PreparedSceneMutation = {
      commit: () => {
        if (stateCommitted) return;
        this._bvhBuffers = result.bvhBuffers;
        this._lastScene = result.updatedScene;
        this._renderScene = nextRenderScene;
        stateCommitted = true;
      },
      rollback: () => {
        undo.restore();
        if (!stateCommitted) return;
        this._bvhBuffers = previousBvh;
        this._lastScene = previousScene;
        this._renderScene = previousRenderScene;
        stateCommitted = false;
      },
      finalize: () => {
        undo.accept();
      },
    };

    const factories: Array<() => PreparedSceneMutation> = [
      () => stateMutation,
    ];
    if (geometryChanged || result.refreshDdgiMaterialSnapshot === true) {
      factories.push(() => this._ddgi.prepareSceneMutation(
        result.bvhBuffers,
        nextRenderScene,
        {
          invalidate: geometryChanged || result.refreshDdgiMaterialSnapshot === true,
          instancesDirty: geometryChanged && result.bvhBuffers.bvhMode === 'tlas',
        },
      ));
    }
    if (this._rc != null && (geometryChanged || result.refreshRcMaterials === true)) {
      factories.push(() => this._rc!.prepareSceneMutation(
        result.bvhBuffers,
        nextRenderScene,
        {
          geometryChanged,
          refreshMaterials: result.refreshRcMaterials === true,
          allowMergedRefit: mutation.replacement == null,
          ...(result.rcRefitBounds ? { rcRefitBounds: result.rcRefitBounds } : {}),
        },
      ));
    }
    if (this._pipeline != null) {
      factories.push(() => this._pipeline!.prepareSceneMutation(
        mutation,
        result.bvhBuffers,
        prefixCommandBuffers,
      ));
    }
    this._warnMaterialTextureAtlasDiagnostics(
      result.bvhBuffers.materialTextureAtlas.diagnostics,
      'updatePrimitive',
    );
    commitSceneMutations(prepareSceneMutations(factories));
  }

  /**
   * Add one whole primitive to the live scene (contract:
   * {@link Engine.addPrimitive}).
   *
   * Design choice — full `setScene`-equivalent rebuild. A whole-primitive add
   * almost always introduces NEW geometry + a NEW material, and on this realtime
   * stack the geometry change invalidates EVERY cached GI signal — the DDGI
   * irradiance/visibility atlases, the ReSTIR-DI/GI reservoir history, the RC
   * cascade buffers, and the temporal accumulator all index off the packed
   * scene and must be rebuilt/reset. Rather than bolt a brittle "incremental BVH
   * splice + per-subsystem re-sync" onto the add path, we append the primitive
   * to a fresh `Scene` copy and route it through `setScene` — the same
   * already-correct `partitionSceneBySupport` → `_teardownPipeline` →
   * `_initCoordinator.startInit()` spine the initial scene build runs. That
   * spine rebuilds the BVH, re-synthesises the DDGI traversal scene, re-derives
   * DDGI lights, rebuilds the RC BVH (via `publishBvh` →
   * `propagateBvhToGiSubsystems`), and — because the pipeline (and its temporal
   * accumulator + reservoirs + history textures) is torn down and rebuilt blank
   * — resets all temporal/accumulation state. Mirrors pt-webgl / pt-webgpu's
   * full-repack choice: correct-by-construction, no fragile per-array index
   * remap. On a realtime engine the work is a rebuild either way; the value is
   * API consistency, not a perf win.
   *
   * Contract semantics honored:
   *   • Duplicate `id` throws BEFORE any mutation — the dup check runs against
   *     the live `_lastScene`, and `nextScene` is only built (and `setScene`
   *     only called) once it passes, so the scene is unchanged on throw.
   *   • Unsupported primitive kinds (or future analytic shapes outside the
   *     capability set) are warn-skipped by the `partitionSceneBySupport` filter
   *     inside `setScene` (they do not throw).
   *   • Accumulation / temporal history resets — `setScene` tears down the
   *     pipeline (blank accumulator + reservoirs + DDGI/ReSTIR/RC rebuild) and
   *     reinitialises.
   */
  addPrimitive(primitive: ScenePrimitive): void {
    if (this._state === 'disposed') {
      throw new Error('HybridEngine.addPrimitive: engine is disposed.');
    }
    if (this._lastScene == null) {
      throw new Error(
        `HybridEngine.addPrimitive("${String(primitive.id)}"): no scene set. ` +
          `Call setScene(scene) before addPrimitive.`,
      );
    }
    if (this._lastScene.primitives.some((p) => String(p.id) === String(primitive.id))) {
      throw new Error(
        `HybridEngine.addPrimitive: a primitive with id "${String(primitive.id)}" ` +
          `already exists; use updatePrimitive to mutate an existing primitive.`,
      );
    }
    const nextScene: Scene = {
      ...this._lastScene,
      primitives: [...this._lastScene.primitives, primitive],
    };
    this.setScene(nextScene);
  }

  /**
   * Remove one whole primitive from the live scene by `id` (contract:
   * {@link Engine.removePrimitive}). The inverse of {@link addPrimitive}.
   *
   * Implementation: drop the primitive from a fresh `Scene` copy and route
   * through `setScene` (same full-rebuild approach as {@link addPrimitive}).
   * Reusing the `setScene` packing path re-packs the dense BVH / DDGI-light /
   * ReSTIR-emitter arrays correctly by construction rather than hand-rolling a
   * multi-array compaction.
   *
   * **Empty-scene behaviour (H20 / H20-A):** Removing the LAST primitive routes
   * through `setScene` with an empty primitives array. The engine transitions to
   * `'ready'` state (no pipeline / BVH allocated) and `renderFrame` now presents
   * a flat SKY-ONLY frame (`skyTint × skyIrradiance`) via a single device-level
   * clear render pass, returning a genuine `kind:'rendered'` FrameOutput rather
   * than skipping (H20-A). The walkaround sky is a scalar tint on this stack, so
   * a flat fill is the radiometrically-faithful empty-scene background. The
   * dispatched-frame counter (`window.__WALKAROUND__.dbg.framesDispatched`)
   * advances for sky-only frames; `skipNoSwapView` still increments when the host
   * provides no swap-chain view.
   *
   * Contract semantics honored:
   *   • A missing `id` throws BEFORE any mutation — the membership check runs
   *     against the live `_lastScene`; `setScene` is only called once it passes,
   *     so the scene is unchanged on throw.
   *   • Accumulation / temporal history resets exactly as for
   *     {@link addPrimitive}.
   */
  removePrimitive(id: ScenePrimitive['id']): void {
    if (this._state === 'disposed') {
      throw new Error('HybridEngine.removePrimitive: engine is disposed.');
    }
    if (this._lastScene == null) {
      throw new Error(
        `HybridEngine.removePrimitive("${String(id)}"): no scene set. ` +
          `Call setScene(scene) before removePrimitive.`,
      );
    }
    const nextPrimitives = this._lastScene.primitives.filter((p) => String(p.id) !== String(id));
    if (nextPrimitives.length === this._lastScene.primitives.length) {
      throw new Error(
        `HybridEngine.removePrimitive: no primitive with id "${String(id)}" ` +
          `in the live scene.`,
      );
    }
    const nextScene: Scene = {
      ...this._lastScene,
      primitives: nextPrimitives,
    };
    this.setScene(nextScene);
  }

  updateEmitter(id: string, patch: Partial<SceneEmitter>): void {
    if (this._state === 'disposed') {
      throw new Error('HybridEngine.updateEmitter: engine is disposed.');
    }
    if (this._state === 'initializing') {
      throw new Error(
        `HybridEngine.updateEmitter("${id}"): engine is initializing. ` +
          `Wait for setScene init to finish before applying emitter patches.`,
      );
    }
    if (this._lastScene == null) {
      throw new Error(
        `HybridEngine.updateEmitter("${id}"): no scene set. ` +
          `Call setScene(scene) before updateEmitter.`,
      );
    }
    const idx = this._lastScene.emitters.findIndex((e) => String(e.id) === id);
    if (idx < 0) {
      throw new Error(
        `HybridEngine.updateEmitter("${id}"): emitter id not found in current scene.`,
      );
    }
    assertKnownEmitterPatchKeys(this._lastScene.emitters[idx]!, patch, id);
    if (this._bvhBuffers == null) {
      throw new Error(
        `HybridEngine.updateEmitter("${id}"): BVH not ready. Wait for setScene init to finish.`,
      );
    }
    if (this._renderScene == null || !this._coreSceneSuppliesMeshes()) {
      throw new Error(
        `HybridEngine.updateEmitter("${id}"): current scene has no core mesh primitives.`,
      );
    }

    const previousScene = this._lastScene;
    const previousRenderScene = this._renderScene;
    const previousBvh = this._bvhBuffers;
    const authoredNextScene = applyEmitterPatchToScene(previousScene, id, patch);
    validateCoreScene(authoredNextScene);
    const nextScene = sceneAcceptedByManifest(
      authoredNextScene,
      this._supportManifest,
      'updateEmitter',
    );
    const nextRenderScene = sceneWithAnalyticMeshFallback(nextScene);

    const effectivePrimaryLightDir =
      this._effectivePrimaryLightDir(nextRenderScene);
    const emitterOptions = {
      primaryLightDir: {
        x: effectivePrimaryLightDir[0],
        y: effectivePrimaryLightDir[1],
        z: effectivePrimaryLightDir[2],
      },
      primaryLightIntensity: this._primaryLightIntensity,
      packSourceTriIndex: true,
      ...(previousBvh.bvhMode === 'tlas'
        ? { tlasPrimitiveBindings: previousBvh.primitiveTlasBindings }
        : {}),
      onWarning: (warning: EngineWarning) => this._warn(warning),
      warningPhase: 'mutation' as const,
      warningMethod: 'updateEmitter',
    };
    const emitterSlice = rebuildEmitterBuffersFromCoreScene(
      nextRenderScene,
      emitterOptions,
    );
    const emissiveLe = rebuildBvhEmissiveLeFromCoreScene(
      nextRenderScene,
      previousBvh,
      emitterOptions,
    );
    const nextBvh: SceneBVHBuffers = {
      ...previousBvh,
      bvhEmissiveLe: emissiveLe,
      emitters: emitterSlice.emitters,
      emitterCdf: emitterSlice.emitterCdf,
      emitterAlias: emitterSlice.emitterAlias,
      emitterCount: emitterSlice.emitterCount,
      totalEmissivePower: emitterSlice.totalEmissivePower,
      lightTree: emitterSlice.lightTree,
      lightTreeNodeCount: emitterSlice.lightTreeNodeCount,
      lightTreeEnabled: emitterSlice.lightTreeEnabled,
    };
    const lightingInputs = buildDdgiLightingMutationInputs(
      {
        ctorLights: this._ctorLights,
        hostSunWarningState: this._hostSunWarningState,
        primaryLightIntensity: this._primaryLightIntensity,
        ...(this._primaryLightDirOverrideActive
          ? { primaryLightDir: this._primaryLightDir }
          : {}),
        onWarning: (warning) => this._warn(warning),
        ...(nextBvh.bvhMode === 'tlas'
          ? { tlasPrimitiveBindings: nextBvh.primitiveTlasBindings }
          : {}),
      },
      nextRenderScene,
      {
        warningPhase: 'mutation',
        warningMethod: 'updateEmitter',
      },
    );

    let stateCommitted = false;
    const stateMutation: PreparedSceneMutation = {
      commit: () => {
        if (stateCommitted) return;
        this._lastScene = nextScene;
        this._renderScene = nextRenderScene;
        this._bvhBuffers = nextBvh;
        stateCommitted = true;
      },
      rollback: () => {
        if (!stateCommitted) return;
        this._lastScene = previousScene;
        this._renderScene = previousRenderScene;
        this._bvhBuffers = previousBvh;
        stateCommitted = false;
      },
      finalize: () => undefined,
    };
    const factories: Array<() => PreparedSceneMutation> = [
      () => stateMutation,
      () => this._ddgi.prepareLightingMutation(lightingInputs),
    ];
    if (this._rc != null) {
      factories.push(() => this._rc!.prepareBindingInvalidation());
    }
    if (this._pipeline != null) {
      factories.push(() => this._pipeline!.prepareEmitterLightingMutation(
        nextBvh,
        nextRenderScene,
      ));
    }

    commitSceneMutations(prepareSceneMutations(factories));
  }

  private _syncDdgiLightsFromCoreScene(): void {
    if (this._renderScene == null || !this._coreSceneSuppliesMeshes()) return;
    // Steps 1–4 (sun intensity, lights merge, emitter tris H18, analytic lights H41)
    // delegated to the shared helper (R3 B-chain step 4). Engine path always
    // merges lights (setLightsConditional: false = default).
    syncDdgiFromCoreScene(
      {
        ddgi: this._ddgi,
        pipeline: this._pipeline,
        ctorLights: this._ctorLights,
        hostSunWarningState: this._hostSunWarningState,
        primaryLightIntensity: this._primaryLightIntensity,
        ...(this._primaryLightDirOverrideActive
          ? { primaryLightDir: this._primaryLightDir }
          : {}),
        onWarning: (warning) => this._warn(warning),
        ...(this._bvhBuffers?.bvhMode === 'tlas'
          ? { tlasPrimitiveBindings: this._bvhBuffers.primitiveTlasBindings }
          : {}),
      },
      this._renderScene,
    );
    // B3 — push the scene's directional IBL map+CDFs to the pipeline (or reset to
    // the no-HDRI placeholder). Called here so both the initial scene load and any
    // emitter/scene fast-update re-resolve the env; no-op before pipeline init.
    this._applyDirectionalEnvironment(this._resolvedEnvironment);
    this._ddgi.invalidateProbeCache();
  }

  // ── GI state persistence ────────────────────────────────────────────────
  // Implementation bodies moved to HybridEngineGIState.ts (R3 B-chain step 3).
  // These thin delegates preserve the public API contract unchanged.

  /**
   * Export the converged DDGI global-illumination state (the "cached light
   * field") so the host can persist it (e.g. to IndexedDB via
   * {@link serializeGIState}) and restore it next session without re-converging.
   * Returns null if the probe atlases aren't allocated yet (call after the GI has
   * run at least one frame). Async (atlas readback uses mapAsync).
   */
  async exportGIState(): Promise<GIStateSnapshot | null> {
    const compatibility = this._currentGIStateCompatibility();
    if (compatibility == null) return null;
    return exportGIStateImpl({
      device: this._device,
      ddgi: this._ddgi,
      pipeline: this._pipeline,
      compatibility,
    });
  }

  /**
   * Restore a previously {@link exportGIState}-ed snapshot into the live GI state
   * (seeds the temporal blend, so rendering continues from it instead of
   * re-converging). Restores the DDGI probe atlases AND — when the snapshot
   * carries them — the ReSTIR-GI temporal reservoirs (v2+) and the PPG
   * sTree/dTree guiding distribution (v4+).
   *
   * Returns false (no-op) if the atlases aren't allocated or the snapshot's atlas
   * dims don't match the current grid. When a reservoir section is present, the
   * restore also fails (returns false) if the reservoir grid/size doesn't match
   * the live pipeline — so a partial (atlas-only) restore is never silently
   * reported as a full success. A v3 snapshot (no PPG section) restores the
   * atlases + reservoirs and returns the atlas+reservoir result unchanged; PPG
   * starts cold without error.
   */
  importGIState(snapshot: GIStateSnapshot): boolean {
    const compatibility = this._currentGIStateCompatibility();
    if (compatibility == null) return false;
    return importGIStateImpl(
      {
        device: this._device,
        ddgi: this._ddgi,
        pipeline: this._pipeline,
        compatibility,
        onWarning: (warning) => this._warn(warning),
      },
      snapshot,
    );
  }

  /**
   * Exact compatibility key for every CPU-mirrored live input consumed by the
   * persisted realtime estimators. A missing BVH means no state generation is
   * currently meaningful or importable.
   */
  private _currentGIStateCompatibility(): Uint32Array | null {
    const bvh = this._bvhBuffers;
    if (bvh == null) return null;
    return makeGIStateCompatibility({
      bvh,
      ...(this._resolvedEnvironment.directional != null
        ? {
            directionalEnvironment:
              this._resolvedEnvironment.directional,
          }
        : {}),
      environmentRotationY: this._resolvedEnvironment.rotationY ?? 0,
      environmentIntensity:
        this._resolvedEnvironment.directional == null
          ? 0
          : (this._resolvedEnvironment.directionalIntensity ?? 1),
      primaryLightDirection: this._effectivePrimaryLightDir(),
      primaryLightIntensity: this._primaryLightIntensity,
      skyTint: this._skyTint,
      skyIrradiance: this._skyIrradiance,
      estimatorConfiguration: encodeGIStateCompatibilityData({
        schema: 1,
        maxBounces: this._cfg.maxBounces,
        tunables: this._scaledTunables ?? this._cfg.tunables,
        initTunables: this._cfg.initTunables,
        stainedGlassFlags: this._cfg.stainedGlassFlags,
        // Preserve the schema-1 estimator fingerprint written by existing v8
        // snapshots. This is serialized compatibility data, not a live selector.
        grisReuse: 1,
        rcTransmittedInterfaceBudget:
          this._cfg.rcTransmittedInterfaceBudget,
        nrcEnabled: this._cfg.nrcEnabled,
        nrcConfig: this._cfg.nrcConfig,
        ppgEnabled: this._cfg.ppgEnabled,
        ppgMaxSpatialCells: this._cfg.ppgMaxSpatialCells,
        ppgMaxDTreeNodesPerCell:
          this._cfg.ppgMaxDTreeNodesPerCell,
        ppgMixAlpha: this._cfg.ppgMixAlpha,
        ppgDispatchInterval: this._ppgDispatchInterval,
        checkerboard: this._cfg.checkerboard,
        diSpatialPasses: this._cfg.diSpatialPasses,
        giSpatialPasses: this._cfg.giSpatialPasses,
        ddgiUpdateDivisor: this._cfg.ddgiUpdateDivisor,
        regirConfig: this._cfg.regirConfig,
        rcEnabled: this._rc != null,
        rcWeight: this._rcWeight,
        causticStrategy: this._causticStrategy,
        mneeMaxIterations: this._cfg.mneeMaxIterations,
        mneeMaxChainLength: this._cfg.mneeMaxChainLength,
        mneeMultiplicityTrials: this._cfg.mneeMultiplicityTrials,
        constructorLights: this._ctorLights,
      }),
    });
  }

  /**
   * Runtime update of the primary directional light + sky parameters.
   *
   * Re-uploads the WalkaroundUBO at the next frame start. Invalidates the DDGI
   * probe atlas so it re-converges over the next ~8 frames, and resets the
   * temporal accumulator so stale lighting does not bleed through history.
   *
   * No pipelines or GPU buffers are recreated; calling with an empty object is
   * a safe no-op.
   *
   * @param opts - Partial lighting overrides. Omitted fields are unchanged.
   */
  updateLighting(opts: Partial<LightingOptions>): void {
    if (this._state === 'disposed') return;
    // `Engine.updateLighting` is contractually opaque (Record<string, unknown>),
    // so enforce this backend's closed vocabulary before changing any live
    // lighting or GPU-facing state.
    assertKnownLightingKeys(opts);

    let changed = false;

    if (opts.primaryLightDir !== undefined) {
      this._primaryLightDir = opts.primaryLightDir;
      this._primaryLightDirOverrideActive = true;
      changed = true;
      // Republish DDGI sun lights so the probe-update pass follows the same
      // runtime direction that renderFrame() passes to the shade UBO. With a
      // core mesh scene this re-merges scene emitters; without one (lights-only
      // host, or before setScene) fall back to re-orienting the ctor lights —
      // mirroring the init path (line ~836) so the sun follows primaryLightDir
      // regardless of whether a mesh scene is present.
      if (this._renderScene != null && this._coreSceneSuppliesMeshes()) {
        this._syncDdgiLightsFromCoreScene();
      } else {
        this._ddgi.setLights(orientDdgiSunLights(this._ctorLights, this._primaryLightDir));
      }
    }
    if (opts.primaryLightIntensity !== undefined) {
      this._primaryLightIntensity = opts.primaryLightIntensity;
      changed = true;
      // Keep the DDGI ProbeUpdatePass sun-intensity multiplier in sync so the
      // irradiance atlas re-converges at the correct brightness. Single-count:
      // when a scene `directional` drives the sun, its `sun` DDGILight already
      // carries the emitter intensity, so the multiplier stays 1 and config
      // primaryLightIntensity does NOT additionally scale the DDGI sun (it
      // still drives the shade-side Lo_emit via the WalkaroundUBO). Absent a
      // scene directional, the config intensity is the multiplier as before.
      const sceneForSun =
        this._coreSceneSuppliesMeshes() && this._renderScene != null ? this._renderScene : null;
      this._ddgi.setSunIntensityMultiplier(
        directionalSunMultiplier(sceneForSun, opts.primaryLightIntensity),
      );
    }
    if (opts.skyTint !== undefined) {
      this._skyTint = opts.skyTint;
      changed = true;
    }
    if (opts.skyIrradiance !== undefined) {
      this._skyIrradiance = opts.skyIrradiance;
      changed = true;
    }

    if (!changed) return;

    this._ddgi.setSkyParams?.(this._skyTint, this._skyIrradiance);

    // Invalidate the DDGI probe atlas — re-converges from scratch over the
    // next STRIDE frames (~8 frames, ~133 ms at 60 FPS).
    this._ddgi.invalidateProbeCache();

    // Reset the temporal accumulator — history discarded, α=1 for next frame.
    // _pipeline may be null if the engine is still initialising; the flag is
    // applied as soon as the pipeline exists (set before any renderFrame call).
    this._pipeline?.requestAccumReset();
  }

  /**
   * Retry a sealed PPG training epoch after its bounded readback retries have
   * reached the durable failed state. This is an explicit host action so a
   * repeatedly failing device cannot spin forever in the background.
   *
   * Returns false before pipeline publication, when PPG is disabled/not
   * failed, or after disposal.
   */
  requestPpgTrainingRecovery(): boolean {
    if (this._state === 'disposed') return false;
    return this._pipeline?.requestPpgTrainingRecovery() ?? false;
  }

  /**
   * Apply an environment-only update at runtime (HDRI intensity swap, or a
   * transition to `kind: 'none'`) WITHOUT rebuilding the BVH or re-uploading
   * geometry / materials. Implements the optional `Engine.updateEnvironment`
   * contract; the sibling of {@link updateLighting} for the env-map / sky-config
   * dimension. `attachVitrum` / a host can call this for time-of-day env scrubs
   * the way `pt-webgl` does, with no engine recreation.
   *
   * **What this backend's "environment" actually is.** The walkaround-hybrid
   * realtime stack keeps a diffuse sky-dome fallback pair ({@link _skyTint} RGB
   * + {@link _skyIrradiance} scalar) for probe/shade sky misses, and also binds a
   * compact directional equirect/CDF texture when the environment supplies
   * CPU-readable HDRI pixels or a procedural-sky bake. Opaque HDRI handles remain
   * scalar-only unless the host resolver provides readable pixels.
   *
   * **What it does (the minimal correct update):**
   *  - Maps the env → sky scalars and mutates `_skyTint` / `_skyIrradiance`
   *    (same fields `updateLighting` touches — one source of truth for the
   *    per-frame {@link _lightingSnapshot}).
   *  - Caches the env on `_lastScene.environment` so the engine's scene-state
   *    read reflects the swap (parallels pt-webgl/pt-webgpu caching the env on
   *    their scene record). A `null` env collapses to `{ kind: 'none' }` (the
   *    Scene.environment field is non-nullable).
   *  - Invalidates the DDGI probe cache (re-converges the world-space irradiance
   *    atlas over the next STRIDE frames) and resets the temporal accumulator —
   *    exactly the sky-portion of `updateLighting`, because the sky-dome term
   *    feeds both the probe rays and the shade pass.
   *
   * **What it deliberately does NOT re-do:** no BVH rebuild, no pipeline
   * recompile, no FrameResources reallocation, no scene re-partition — geometry
   * and materials are untouched. That is the whole point of an env-only fast
   * path (cf. `setScene`, which tears the pipeline down).
   *
   * **Known limitation (opaque HDRI directionality).** Opaque `hdri` handles are
   * not directionally sampled unless a host resolver supplies CPU-visible data.
   * Raw numeric RGB/RGBA HDRI payloads and procedural skies do feed the same
   * directional equirect/CDF path. Procedural skies are still graded
   * 'approximate' because they use a finite Preetham bake rather than an analytic
   * infinite-resolution sky model.
   *
   * After {@link dispose} this is a safe no-op (matches the runtime-update
   * siblings + the `@vitrum/engine` facade's `'noop'` disposed-behaviour for
   * `updateEnvironment`) — the DDGI subsystem is torn down on dispose, so
   * touching it would be unsafe.
   *
   * @param env the new scene environment, or `null` to clear it (≡ `{ kind:
   *   'none' }`).
   */
  updateEnvironment(env: SceneEnvironment | null): void {
    // Disposed → no-op. The runtime-update contract for `updateEnvironment` is
    // `'noop'` after dispose (see @vitrum/engine idempotentDispose), and
    // `dispose()` already called `this._ddgi.dispose()`, so the
    // invalidateProbeCache() below would touch a torn-down subsystem. Guard
    // here so the direct (non-facade) call is also safe.
    if (this._state === 'disposed') return;

    const nextEnv: SceneEnvironment = env ?? { kind: 'none' };
    validateCoreScene({ primitives: [], emitters: [], environment: nextEnv });
    const resolved = resolveHybridEnvironment(nextEnv, {
      extensions: this._environmentResolverExtensions,
    });
    this._emitResolvedEnvironmentWarnings(resolved, 'updateEnvironment');
    // Cache on the live scene so a later scene-state read / debug surface sees
    // the current env (parallels pt-webgl/pt-webgpu). `_lastScene` may be null
    // if no setScene() ran yet — still record the sky-scalar change; the next
    // setScene() carries its own environment.
    if (this._lastScene != null) {
      this._lastScene = { ...this._lastScene, environment: nextEnv };
      this._renderScene =
        this._renderScene != null
          ? { ...this._renderScene, environment: nextEnv }
          : sceneWithAnalyticMeshFallback(this._lastScene);
    }
    this._resolvedEnvironment = resolved;

    // Map the env onto this backend's sky-dome scalars (the only env channel it
    // consumes — there is no IBL baker here). Omitted fields leave the
    // corresponding scalar unchanged.
    this._applyResolvedEnvironmentScalars(resolved);
    this._ddgi.setSkyParams?.(this._skyTint, this._skyIrradiance);

    // Re-converge the world-space DDGI irradiance atlas (the sky-dome term feeds
    // the probe rays) and discard temporal history so the new sky energy shows
    // immediately rather than bleeding in over the accumulation window. Same
    // invalidation `updateLighting` does for the sky portion; `_pipeline` may be
    // null mid-init (the accum-reset flag is applied once the pipeline exists).
    this._ddgi.invalidateProbeCache();
    this._pipeline?.requestAccumReset();

    // B3 — push the directional IBL map+CDFs to the pipeline (or reset to the
    // no-HDRI placeholder). Independent of the sky scalars above, which remain
    // the WGSL fallback when no directional data is present.
    this._applyDirectionalEnvironment(resolved);
  }

  /**
   * Map a `SceneEnvironment` onto this backend's diffuse sky-dome scalars
   * ({@link _skyTint} / {@link _skyIrradiance}). No GPU work and no engine-state
   * mutation, so
   * the mapping is straightforward to unit-test. Returns only the fields that
   * should change — an omitted field leaves the engine's current scalar in
   * place (so e.g. an `hdri` swap that carries no tint preserves a host-supplied
   * `skyTint`).
   *
   * Mapping (see {@link updateEnvironment} for the full rationale):
   *  - `none` → `skyIrradiance: 0` (sky contributes no light; matches
   *    `applyEnvironment`'s black-background `none`). Tint left unchanged.
   *  - `hdri` → raw numeric payloads / host extension resolvers provide either
   *    directional data or an explicit scalar-only contract. Opaque handles
   *    without a resolver are rejected synchronously.
   *  - `procedural-sky` → 'approximate' grade: resolveHybridEnvironment bakes
   *    turbidity/rayleigh/mie/sunDirection into a finite Preetham equirect and
   *    returns scalar skyTint/skyIrradiance as the no-directional fallback.
   */
  private _applyResolvedEnvironmentScalars(resolved: HybridResolvedEnvironment): void {
    if (resolved.skyTint !== undefined) this._skyTint = resolved.skyTint;
    if (resolved.skyIrradiance !== undefined) this._skyIrradiance = resolved.skyIrradiance;
  }

  /** Emit resolver diagnostics before any scene/lighting state is published. */
  private _emitResolvedEnvironmentWarnings(
    resolved: HybridResolvedEnvironment,
    method: 'setScene' | 'updateEnvironment',
  ): void {
    if (resolved.warnings.length === 0) return;
    for (const warning of resolved.warnings) {
      const identity =
        `${this._sceneGeneration}\u0000${resolved.mode}\u0000${warning}`;
      if (this._environmentWarningKeys.has(identity)) continue;
      this._environmentWarningKeys.add(identity);
      this._warn({
        code: 'walkaround-hybrid.environment-approximation',
        backend: 'walkaround-hybrid',
        phase: method === 'setScene' ? 'setScene' : 'mutation',
        method,
        message: `[HybridEngine] ${method}: ${warning}`,
        details: { warning },
      });
    }
  }

  /**
   * B3 — resolve the directional IBL payload (PBRT 2D distribution) from the
   * environment and push it to the pipeline's scene-group env resources. A
   * raw pixel-backed HDRI or procedural sky yields the directional map+CDFs;
   * scalar-only/none/all-black resolutions reset the pipeline to
   * the no-HDRI placeholder so the WGSL scalar-sky fallback runs (no-HDRI
   * byte-identity). No-op when the pipeline is not yet initialized — setScene's
   * init path calls this AFTER the pipeline exists.
   */
  private _applyDirectionalEnvironmentToPipeline(
    pipeline: WalkaroundGPUPipeline,
    resolved: HybridResolvedEnvironment,
  ): void {
    if (resolved.directional !== undefined) {
      pipeline.updateDirectionalEnvironment(
        resolved.directional,
        resolved.rotationY ?? 0,
        resolved.directionalIntensity ?? 1,
      );
      // Wave 4 (2026-06-10) — HDRI into DDGI probe misses: hand the equirect
      // radiance view to the probe-update pass so probe-ray misses sample the
      // real map / finite procedural-sky bake when a directional env is bound.
      const envBindings = pipeline.getEnvBindings();
      if (envBindings != null) {
        this._ddgi.setEnvironment(
          envBindings.textureView,
          envBindings.sampler,
          resolved.rotationY ?? 0,
          resolved.directionalIntensity ?? 1,
          true,
        );
      }
    } else {
      pipeline.updateDirectionalEnvironment(null, 0, 0);
      this._ddgi.setEnvironment(null, null, 0, 0, false);
    }
  }

  private _applyDirectionalEnvironment(resolved: HybridResolvedEnvironment): void {
    if (this._pipeline == null) return;
    this._applyDirectionalEnvironmentToPipeline(this._pipeline, resolved);
  }

  // ── Progressive handoff seed source ──────────────────────────────────────
  /**
   * Progressive walkaround→PT seed source (P8 increment 2). The last frame's
   * post-denoise HDR radiance (linear, pre-tonemap — same space as a PT
   * accumulator) as a `BackendTexture` + its internal render dimensions, so a host
   * coordinator can seed a converged PT engine's accumulator
   * (`engine.seedAccumulator(texture, { weight, width, height })`). Null before the
   * first rendered frame. The texture is recycled each frame — consume it
   * SYNCHRONOUSLY within the handoff frame (do not cache the handle).
   *
   * Available only when `capabilities.supportsProgressiveSeedSource === true`.
   */
  getProgressiveSeedTexture(): {
    texture: BackendTexture<'webgpu', GPUTexture>;
    width: number;
    height: number;
  } | null {
    const tex = this._pipeline?.getProgressiveSeedTexture();
    if (tex == null) return null;
    return {
      texture: asBackendTexture<'webgpu', GPUTexture>(tex),
      width: this._internalWidth,
      height: this._internalHeight,
    };
  }

  /**
   * Capture the engine's rendered output as a host-side CPU Float32 RGBA image,
   * row-major, top-left origin.
   *
   * `colorSpace:'linear'` (default) reads `resolvedTexture` — the post-denoiser,
   * pre-tonemap rgba16float output (the same texture exposed by
   * `getProgressiveSeedTexture()`). This is linear-light HDR radiance in scene
   * units, suitable for tone-mapping, EXR export, or luminance checks.
   *
   * `colorSpace:'output'` runs the SAME composite pass (tonemap + OETF +
   * exposure) into an engine-owned offscreen `rgba8unorm` texture and reads it
   * back as display-encoded, post-OETF values in [0, 1].  Unlike 'linear', this
   * path produces the display-referred image a viewer would see on screen.  The
   * composite UBO settings (tonemap operator, exposure, output color space) from
   * the most recent rendered frame are reused verbatim — there is no need to call
   * `renderFrame` again.
   *
   * Returns `null` before the first frame (no pipeline or resolvedTexture not yet
   * allocated).  Pipeline stall: submits copyTextureToBuffer + mapAsync; use for
   * debugging/export, not per-frame readback.
   */
  async captureFrame(opts?: CaptureFrameOptions): Promise<CapturedFrame | null> {
    const colorSpace = opts?.colorSpace ?? 'linear';
    if (colorSpace === 'output') {
      const rgba = (await this._pipeline?.captureOutputFrame()) ?? null;
      if (rgba == null) return null;
      return { width: this._internalWidth, height: this._internalHeight, rgba };
    }
    const seedResult = this.getProgressiveSeedTexture();
    if (seedResult == null) return null;
    const { width, height } = seedResult;
    if (width <= 0 || height <= 0) return null;
    const texture = seedResult.texture as unknown as GPUTexture;
    const rgba = await readRgba16fWalkaround(this._device, texture, width, height);
    if (rgba == null) return null;
    return { width, height, rgba };
  }

  // ── Resize ─────────────────────────────────────────────────────────────

  /**
   * Resize the render surface WITHOUT rebuilding the BVH or recompiling
   * pipelines. The host calls this whenever the canvas (or device-pixel
   * ratio) changes — much cheaper than the previous resize-storm pattern
   * (engine teardown → recreate engine → poll for ready), which on a
   * single resize tick churned every BVH buffer + every pipeline shader
   * + every DDGI atlas + ~1 GB of FrameResources textures.
   *
   * Behaviour:
   *   - Updates `_width` / `_height` on the engine.
   *   - Calls `WalkaroundGPUPipeline.resize(W, H)` if the pipeline is
   *     live, which destroys + recreates per-frame GPU resources only
   *     (FrameResources textures + reservoir buffers + variance buffers
   *     + GTAO half/full + SVGF persistent textures) at the new size.
   *     The BVH, pipeline shaders, bind-group layouts, DDGI atlases,
   *     and per-pass UBOs are preserved.
   *   - Resets the temporal accumulator + ping-pong indices on the
   *     pipeline (the new textures are blank, so reusing prior history
   *     would sample undefined memory).
   *
   * No-op when called with the current size, or when the pipeline isn't
   * yet live (the new size is stored on the engine; `_initPipeline` will
   * use it when it constructs the pipeline). `renderFrame` invokes this
   * automatically when `FrameInput.viewport` changes; direct calls remain a
   * useful eager-resize hook for hosts reacting to a ResizeObserver.
   *
   * Cost: O(W·H) GPU memory churn for the FrameResources reallocation;
   * no shader recompile, no BVH rebuild. Typical resize tick: 5-30 ms
   * for the GPU allocations on a 4K surface, vs 500-2000 ms for a full
   * engine teardown + re-init.
   */
  setSize(width: number, height: number): void {
    if (this._state === 'disposed') {
      throw new Error('HybridEngine.setSize: engine is disposed.');
    }
    assertPositiveSafeViewportDimension(width, 'HybridEngine.setSize: width');
    assertPositiveSafeViewportDimension(height, 'HybridEngine.setSize: height');
    if (width === this._width && height === this._height) return;

    // Keep every public and internal dimension unchanged until the pipeline has
    // proved that its complete replacement resource set can be published.
    const internalWidth = Math.max(1, Math.round(width * this._resolutionFactor));
    const internalHeight = Math.max(1, Math.round(height * this._resolutionFactor));
    this._pipeline?.resize(internalWidth, internalHeight);

    this._width = width;
    this._height = height;
    this._internalWidth = internalWidth;
    this._internalHeight = internalHeight;
    // No DDGI invalidation — the irradiance atlas is world-space, not
    // screen-space, so it survives a resize unchanged.
  }

  /**
   * §5.1 — apply a per-frame `quality.resolutionFactor`. Computes the target
   * internal render size (= canvas × clamped factor), and — when it changes
   * beyond a 2-px threshold and the debounce window has elapsed — resizes the
   * pipeline's per-frame resources to the internal size. The composite pass
   * upscales the internal-sized resolvedTexture to the full canvas swap-chain
   * view, so no swap-chain reconfigure is needed.
   *
   * Returns the internal dims to dispatch at this frame (unchanged when the
   * resize was debounced). Called once per frame from the orchestrator before
   * `pipeline.renderFrame`.
   */
  private _applyResolutionFactor(
    factor: number | undefined,
    nowMs: number,
  ): { width: number; height: number } {
    // An omitted per-frame override falls back to the construction-time quality
    // preset rather than silently promoting medium/low engines to full resolution.
    const requestedFactor = factor ?? this._cfg.resolutionFactor;
    // Record the clamped effective factor so a subsequent setSize() preserves it.
    const resolutionFactor =
      Number.isFinite(requestedFactor) && requestedFactor > 0
        ? Math.min(1, requestedFactor)
        : 1;

    const decision = resolveInternalRenderSize({
      swapW: this._width,
      swapH: this._height,
      factor: requestedFactor,
      currentW: this._internalWidth,
      currentH: this._internalHeight,
      nowMs,
      lastResizeTs: this._lastResolutionResizeTs,
    });

    if (decision.shouldResize) {
      this._pipeline?.resize(decision.targetW, decision.targetH);
      this._internalWidth = decision.targetW;
      this._internalHeight = decision.targetH;
      this._lastResolutionResizeTs = nowMs;
    }
    this._resolutionFactor = resolutionFactor;
    return { width: this._internalWidth, height: this._internalHeight };
  }

  // ── Frame rendering ────────────────────────────────────────────────────

  /**
   * Render one walkaround frame. Drives:
   *   1. DDGI per-frame compute (fire-and-forget) — GPU command queueing is
   *      synchronous from JS's perspective; the actual GPU work runs after the
   *      JS tick returns. The atlas DDGI writes is double-buffered: this frame
   *      reads from the previous tick's write target while next tick will read
   *      from this one.
   *   2. DDGI atlas wire into the ReSTIR shade pass.
   *   3. ReSTIR pipeline.renderFrame().
   *
   * The host calls `engine.renderFrame(input)` and receives a complete frame.
   * The host does NOT separately call `ddgi.updateFrame()`.
   *
   * Returns a FrameOutput immediately — the pipeline writes directly into the
   * swap chain texture provided via `input.swapChainView`.
   *
   * The 60 FPS internal throttle is enforced here: on high-refresh-rate
   * displays, frames arriving faster than ~16.67 ms apart are skipped and
   * this returns a "skip" FrameOutput (`kind: 'skipped'`,
   * `samplesAccumulated: 0`, `isConverged: false`).
   *
   * `input.viewport` is the physical CANVAS size. A changed width or height is
   * applied transactionally through {@link setSize} before any frame state is
   * advanced. Unchanged dimensions allocate nothing; a failed replacement
   * leaves the published canvas/internal dimensions and temporal state intact.
   *
   * However, `input.quality.resolutionFactor` IS honoured per-frame
   * (Phase-0 productization): it scales the INTERNAL render resolution
   * (= canvas × factor) via a debounced {@link _applyResolutionFactor}; the
   * composite pass upscales to the full canvas. See the `@vitrum/core`
   * FrameInput.viewport JSDoc for the cross-backend contract.
   */
  renderFrame(input: FrameInput): FrameOutput {
    validateHybridFrameInput(input);
    const canonicalInput = canonicalizeFrameCamera(
      input,
      'HybridEngine.renderFrame',
    );
    // FrameInput.viewport is the cross-backend resize contract. Keep the
    // explicit setSize method as an eager host hook, while making direct
    // renderFrame callers correct without a ResizeObserver. setSize publishes
    // dimensions only after the pipeline has published its complete replacement
    // resource set, so a thrown resize leaves this frame entirely unadvanced.
    if (
      this._state !== 'disposed' &&
      (canonicalInput.viewport.width !== this._width ||
        canonicalInput.viewport.height !== this._height)
    ) {
      this.setSize(
        canonicalInput.viewport.width,
        canonicalInput.viewport.height,
      );
    }
    // Retain the validated camera even when this frame is subsequently
    // skipped for a pipeline rebuild. `renderFrame` accepted the frame and
    // debug picking must observe its canonical camera; malformed inputs and
    // failed resize transactions still throw before this publication point.
    this._lastFrameCamera = {
      viewMatrix: new Float32Array(canonicalInput.viewMatrix),
      projMatrix: new Float32Array(canonicalInput.projMatrix),
      cameraPosition: [
        canonicalInput.cameraPosition[0],
        canonicalInput.cameraPosition[1],
        canonicalInput.cameraPosition[2],
      ],
    };
    // Advance the error-throttle frame counter (see _onUncapturedError).
    this._errorFrameCount++;
    // Rebuild-key check (D2.5 — moved out of the orchestrator dep bundle so
    // engine-state mutations don't live inside the FrameDeps closure). Must
    // run BEFORE _buildFrameDeps / runHybridEngineFrame — same position as the
    // former orchestrator-side check, so skip-output semantics are preserved.
    const fp = fingerprintHybridPipelineRebuildKey(
      this._cfg.getPipelineRebuildKey?.() ?? this._cfg.staticPipelineRebuildKey,
    );
    if (fp !== this._rebuildKeyFingerprintSeen) {
      this._rebuildKeyFingerprintSeen = fp;
      this.reset();
      return HYBRID_FRAME_SKIP_OUTPUT;
    }

    return runHybridEngineFrame(this._buildFrameDeps(), canonicalInput);
  }

  /** Live lighting snapshot — the four runtime-mutable lighting fields
   *  (`updateLighting()` mutates them). Grouped so both DI builders and any
   *  future lighting consumer share one source of truth: adding a lighting
   *  field is a single edit here, not one per builder. Read at call time
   *  (per-frame snapshot semantics — see {@link _buildFrameDeps}). */
  private _effectivePrimaryLightDir(
    scene: Scene | null = this._renderScene,
  ): [number, number, number] {
    if (this._primaryLightDirOverrideActive) return [...this._primaryLightDir];
    return scenePrimaryLightDirection(scene) ?? [...this._primaryLightDir];
  }

  private _primaryLightDirOverride():
    | readonly [number, number, number]
    | undefined {
    return this._primaryLightDirOverrideActive
      ? this._primaryLightDir
      : undefined;
  }

  private _lightingSnapshot(): HybridLightingDeps {
    const primaryLightDir = this._effectivePrimaryLightDir();
    const primaryLightDirOverride = this._primaryLightDirOverride();
    const sceneLights = this._renderScene == null
      ? []
      : coreEmittersToDDGILights(this._renderScene);
    const mergedLights = mergeDDGILightsDedupSun(
      this._ctorLights,
      sceneLights,
      {
        warningState: this._hostSunWarningState,
        onWarning: (warning) => this._warn(warning),
      },
    );
    const ddgiLights = primaryLightDirOverride == null
      ? mergedLights
      : orientDdgiSunLights(mergedLights, primaryLightDirOverride);
    return {
      primaryLightDir,
      ...(primaryLightDirOverride != null ? { primaryLightDirOverride } : {}),
      primaryLightIntensity: this._primaryLightIntensity,
      ddgiLights,
      skyTint: this._skyTint,
      skyIrradiance: this._skyIrradiance,
    };
  }

  /** Tuple-typed denoiser-filter cluster (firefly clamp + per-channel atrous
   *  sigmas). These live outside the number-only {@link Tunables} table
   *  because they are tuple-valued; grouping them keeps {@link _buildFrameDeps}
   *  compact and makes a new tuple knob a single edit. */
  private _denoiserFilterDeps(): HybridDenoiserFilterDeps {
    const directionalSunShadowDisabled =
      this._renderScene?.emitters.some(
        (e) => e.kind === 'directional' && e.castShadow === false,
      ) === true;
    return {
      // B15 — scene-scale-aware default (falls back to the Cornell baseline
      // before the first setScene). Host overrides already pass through verbatim
      // (deriveScaleAwareClamps leaves host-explicit knobs un-scaled).
      indirectFireflyClamp: this._scaledIndirectFireflyClamp ?? this._cfg.indirectFireflyClamp,
      atrousDirectSigmas: this._cfg.atrousDirectSigmas,
      atrousIndirectSigmas: this._cfg.atrousIndirectSigmas,
      stainedGlassFlags: directionalSunShadowDisabled
        ? (this._cfg.stainedGlassFlags | SHADE_FLAG_DIRECT_SUN_SHADOW_DISABLED) >>> 0
        : this._cfg.stainedGlassFlags,
      nrcEnabled: this._cfg.nrcEnabled,
    };
  }

  private _buildFrameDeps(): HybridEngineFrameDeps {
    // eslint-disable-next-line @typescript-eslint/no-this-alias -- `self` required for the `get state()` getter inside the returned object literal where `this` would refer to the object, not the class instance
    const self = this;
    return {
      subsystems: {
        pipeline: self._pipeline,
        bvhBuffers: self._bvhBuffers,
        ddgi: self._ddgi,
        rc: self._rc,
        skinning: self._skinning,
        lastScene: self._renderScene,
      },
      lighting: self._lightingSnapshot(),
      filter: self._denoiserFilterDeps(),
      telemetry: {
        frameSubs: self._frameSubs,
        progressSubs: self._progressSubs,
        verbose: self._cfg.verbose,
        debugTimings: self._debugTimings,
        debugSurface: self.debug,
        dbg: self._cfg.debug ? self._dbg : null,
        getDenoiserState: () => self._pipeline?.getActiveDenoiserState() ?? null,
      },
      dims: {
        width: self._width,
        height: self._height,
        internalWidth: self._internalWidth,
        internalHeight: self._internalHeight,
      },
      control: {
        targetFrameIntervalMs: self._cfg.targetFrameIntervalMs,
        getLastFrameTs: () => self._lastFrameTs,
        setLastFrameTs: (ts) => {
          self._lastFrameTs = ts;
        },
        applyResolutionFactor: (factor, nowMs) => self._applyResolutionFactor(factor, nowMs),
        runSkinning: () => {
          if (self._skinning != null && self._lastScene != null) {
            self._skinning.run(self, self._lastScene);
          }
        },
        presentLastFrame: (view) => {
          self._pipeline?.presentLastFrame(view);
        },
      },
      flags: {
        get state() {
          return self._state;
        },
        debug: self._cfg.debug,
        ddgiOn: self._ddgiOn,
        maxBounces: self._cfg.maxBounces,
        isLayerEnabled: (layer) => self._layerEnabled.get(layer) ?? true,
        device: self._device,
        // B15 — scene-scale-aware tunables (falls back to the Cornell baseline
        // before the first setScene; byte-identical at Cornell scale).
        tunables: self._scaledTunables ?? self._cfg.tunables,
        rcWeight: self._rcWeight,
        causticStrategy: self._causticStrategy,
        mneeMaxIterations: self._cfg.mneeMaxIterations,
        mneeMaxChainLength: self._cfg.mneeMaxChainLength,
        mneeMultiplicityTrials: self._cfg.mneeMultiplicityTrials,
      },
    };
  }

  // ── Reset ──────────────────────────────────────────────────────────────

  /**
   * Tear down the pipeline and reinitialise from scratch.
   * Hosts call this when the scene changes significantly.
   */
  reset(): void {
    if (this._state === 'disposed') {
      throw new Error('HybridEngine.reset: engine is disposed.');
    }
    this._teardownPipeline();
    this._initCoordinator.startInit();
  }

  // ── Pause / resume ─────────────────────────────────────────────────────

  /**
   * Pause per-frame compute. Engine state transitions from `'ready'` →
   * `'paused'`. During `'initializing'`, pause intent is remembered and the
   * asynchronously published generation enters `'paused'`; calls after
   * `'disposed'` throw.
   *
   * Aligns with `PTEngineWebGL2.pause()`: both throw on disposed, both
   * no-op when the state transition doesn't apply.
   */
  pause(): void {
    if (this._state === 'disposed' || this._state === 'error') {
      throw new Error('pause: engine is disposed or in error state');
    }
    this._pauseRequested = true;
    if (this._state === 'ready') {
      this._state = 'paused';
    }
    // During async initialization the intent is applied at publication.
  }

  /**
   * Resume per-frame compute. Engine state transitions from `'paused'` →
   * `'ready'`. Calls in any other live state are no-ops; calls after
   * `'disposed'` throw.
   *
   * Aligns with `PTEngineWebGL2.resume()`: both throw on disposed, both
   * no-op when the state transition doesn't apply.
   */
  resume(): void {
    if (this._state === 'disposed' || this._state === 'error') {
      throw new Error('resume: engine is disposed or in error state');
    }
    this._pauseRequested = false;
    if (this._state === 'paused') {
      this._state = 'ready';
    }
    // In 'initializing', clearing intent lets publication enter 'ready'.
  }

  // ── Layer toggles (host-accessible debug interface) ────────────────────

  /**
   * Enable or disable a named render layer. Currently recognised layers:
   *   - 'ddgi' — DDGI probe atlas wiring into the shade pass.
   *
   * Replaces `window.__HYBRID_LAYERS__` from the original host bridge.
   * The host sets up `window.__HYBRID_LAYERS__` → `engine.setLayerEnabled()`
   * forwarding if it wants console-accessible toggles.
   */
  setLayerEnabled(layer: HybridRenderLayer, enabled: boolean): void {
    this._assertNotDisposed('setLayerEnabled');
    if (!HYBRID_RENDER_LAYERS.has(layer)) {
      throw new RangeError(
        `HybridEngine.setLayerEnabled: unsupported layer "${String(layer)}"; ` +
        `supported layers: ddgi.`,
      );
    }
    if (typeof enabled !== 'boolean') {
      throw new TypeError(
        `HybridEngine.setLayerEnabled("${layer}"): enabled must be a boolean.`,
      );
    }
    this._layerEnabled.set(layer, enabled);
  }

  // ── Telemetry (T3.E) ───────────────────────────────────────────────────

  /** Subscribe to per-frame stats. Fired at the end of each successful
   *  renderFrame() call. Returns an unsubscribe function. Subscribers
   *  that throw are swallowed so the render loop stays alive. */
  onFrame(cb: (stats: FrameStats) => void): () => void {
    this._frameSubs.push(cb);
    return () => {
      const i = this._frameSubs.indexOf(cb);
      if (i >= 0) this._frameSubs.splice(i, 1);
    };
  }

  /**
   * Subscribe to long-running progress events. Returns an unsubscribe
   * function. Subscribers that throw are swallowed so the render loop stays
   * alive (mirrors {@link onFrame}).
   *
   * Walkaround engines don't accumulate samples, so there is no `'pt-spp'`
   * signal — but the two REAL warm-up signals this engine has ARE surfaced
   * (closing the contract's `'ddgi-warmup'` / `'denoiser-converge'`
   * zero-producer gap):
   *
   *   - `'ddgi-warmup'` — the DDGI probe round-robin updates `1/stride` of the
   *     grid per frame, so a freshly built / invalidated grid takes `stride`
   *     frames for every probe to receive its first update. `fraction` ramps
   *     `frame / stride` from 0→1 and emission STOPS once the grid is warm
   *     (`DDGI.ready`). Reset to 0 on `setScene()` (fresh DDGI) and
   *     `updateLighting()` (`invalidateProbeCache()`).
   *
   *   - `'denoiser-converge'` — the temporal accumulator blends `α` of the new
   *     frame with `1-α` of history (α≈0.01 ⇒ ~100-frame window). `fraction`
   *     ramps `accumFrameIndex / round(1/α)` and emission STOPS once the
   *     window is full. Reset to 0 on camera motion, `updateLighting()` /
   *     `updateEmitter()` (`requestAccumReset()`), and `setSize()` / resize.
   *
   * Both events fire at most once per dispatched frame, only while their
   * signal is still converging. A no-op when the host registered no callback.
   */
  onProgress(cb: (progress: ProgressStats) => void): () => void {
    this._progressSubs.push(cb);
    return () => {
      const i = this._progressSubs.indexOf(cb);
      if (i >= 0) this._progressSubs.splice(i, 1);
    };
  }

  /** Subscribe to GPU/runtime errors. Returns an unsubscribe function.
   *  Wired events: device `uncapturederror` (throttled, non-fatal) and
   *  `device.lost` (fatal, transitions engine to `'error'`). */
  onError(cb: (error: EngineError) => void): () => void {
    this._errorSubs.push(cb);
    return () => {
      const i = this._errorSubs.indexOf(cb);
      if (i >= 0) this._errorSubs.splice(i, 1);
    };
  }

  /** Subscribe to non-fatal contract warnings. Returns an unsubscribe function. */
  onWarning(cb: (warning: EngineWarning) => void): () => void {
    this._warningSubs.push(cb);
    return () => {
      const i = this._warningSubs.indexOf(cb);
      if (i >= 0) this._warningSubs.splice(i, 1);
    };
  }

  /** Internal: emit an error to all subscribers. Catches subscriber throws. */
  private _emitError(error: EngineError): void {
    for (const cb of this._errorSubs) {
      try {
        cb(error);
      } catch {
        /* must not break rendering */
      }
    }
  }

  /** Internal: emit a warning to all subscribers. Catches subscriber throws. */
  private _emitWarning(warning: EngineWarning): void {
    for (const cb of this._warningSubs) {
      try {
        cb(warning);
      } catch {
        /* must not break rendering */
      }
    }
  }

  private _warn(warning: EngineWarning, ...consoleArgs: readonly unknown[]): void {
    console.warn(...(consoleArgs.length > 0 ? consoleArgs : [warning.message]));
    this._emitWarning(warning);
  }

  // ── Dispose ────────────────────────────────────────────────────────────

  /**
   * Synchronous dispose — releases all engine-owned GPU resources.
   *
   * The contract intentionally remains synchronous (so hosts can call it
   * from React cleanup effects, finalizers, etc. without an async
   * paradigm shift). When the {@link PipelineInitCoordinator} has an init
   * chain in flight, the actual GPU-resource release for any work that
   * chain hasn't yet published is deferred to the chain's own `finally`
   * block — the chain checks the coordinator's `_pendingTeardown` after
   * every await boundary and, if set, disposes its locals AND finalises
   * teardown of whatever did make it to shared state.
   *
   * Idempotent: a second `dispose()` call is a no-op.
   */
  dispose(): void {
    if (this._state === 'disposed' && !this._initCoordinator.initRunning) {
      // Already disposed and no in-flight chain to coordinate with — no-op.
      return;
    }

    // Remove GPU error listener before any teardown so the handler can't fire
    // after dispose (the device is no longer ours to observe).
    if (this._onUncapturedError != null) {
      this._device.removeEventListener('uncapturederror', this._onUncapturedError);
      this._onUncapturedError = null;
    }
    this._frameSubs.length = 0;
    this._progressSubs.length = 0;
    this._errorSubs.length = 0;
    this._warningSubs.length = 0;
    this._errorThrottleMap.clear();
    this._frameBudget = null;

    // requestTeardown returns true when the coordinator has no chain in
    // flight (we tear down here and now), false when it has one in flight
    // and its finally block will tear down. The coordinator records the
    // dispose intent regardless so its phase checkpoints bail.
    const teardownNow = this._initCoordinator.requestTeardown();
    if (teardownNow) {
      // No in-flight init; tear down here and now.
      this._teardownPipeline();
      this._skinning?.dispose();
      this._ddgi.dispose();
      // W8 Phase 2 — also tear down RC subsystem when active.
      if (this._rc) {
        this._rc.dispose();
        this._rc = null;
      }
      this._state = 'disposed';
    } else {
      // An init is mid-flight. Defer teardown to that chain's finally
      // block — it will dispose its locals AND tear down whatever's
      // currently in shared state. We can't safely call
      // _teardownPipeline() here because the in-flight chain's
      // `await pipeline.initialize()` may still be holding a live
      // reference to a half-built pipeline.
      this._state = 'disposed';
      // Note: _ddgi.dispose() is deferred too; the in-flight chain may
      // still call _ddgi.setSunIntensityMultiplier() after the
      // post-pipeline checkpoint, and we don't want a torn-down DDGI
      // under it. The chain's finally calls disposeDdgi() when it sees
      // pending teardown.
    }

    if (this._cfg.debug && typeof window !== 'undefined') {
      const dbg = this._dbg;
      dbg.disposeCount++;
      const liveMs = dbg.initStart > 0 ? performance.now() - dbg.initStart : 0;
      console.log(`[hybrid:debug] dispose #${dbg.disposeCount}`, {
        ranForMs: liveMs.toFixed(1),
        framesDispatched: dbg.framesDispatched,
        deferredTeardown: !teardownNow,
        skipReasons: {
          noPipeline: dbg.skipNoPipeline,
          noBvh: dbg.skipNoBvh,
          noSwapView: dbg.skipNoSwapView,
          frameInterval: dbg.skipFrameInterval,
        },
      });
    }
  }

  // ── Private helpers ────────────────────────────────────────────────────

  private _assertNotDisposed(method: string): void {
    if (this._state === 'disposed') {
      throw new Error(`HybridEngine.${method}: engine is disposed.`);
    }
  }

  /** True when the render-ingestion scene supplies at least one triangle-backed
   *  primitive. Rest-pose skinned meshes count — host pushes deformed positions
   *  via `updatePrimitive`, but the BVH still needs a non-empty scene to build.
   *  Instanced meshes count as well because the walkaround TLAS path consumes
   *  their instance matrices directly. */
  private _coreSceneSuppliesMeshes(): boolean {
    const s = this._renderScene;
    return (
      s != null &&
      s.primitives.some(
        (p) => p.kind === 'mesh' || p.kind === 'skinned-mesh' || p.kind === 'instanced-mesh',
      )
    );
  }

  /** Scene-readiness for BVH build: core mesh payload plus optional host gate. */
  private _sceneReadyForBvh(): boolean {
    return this._coreSceneSuppliesMeshes() && this._isSceneReady();
  }

  private _teardownPipeline(): void {
    if (this._pipeline) {
      this._pipeline.dispose();
      this._pipeline = null;
    }
    if (this._bvhBuffers) {
      this._bvhBuffers = null;
    }
    if (this._state !== 'disposed') {
      this._state = 'initializing';
    }
  }

  /** Construction-time immutable config consumed by the init coordinator.
   *  Every field here is assigned once in the constructor and never mutated,
   *  so plain values are behaviorally identical to live getters — grouping
   *  them collapses ~11 one-line getters into one spread and makes adding a
   *  ctor-immutable init input a single edit. The MUTABLE fields (width,
   *  height, lastScene, lighting, current BVH/traversal-scene) stay as live
   *  getters in {@link _buildInitHost} because the coordinator reads them
   *  across async phases. */
  private _initStaticConfig(): HybridInitStaticConfig {
    return {
      device: this._device,
      restirBvhModeOverride: this._cfg.restirBvhModeOverride,
      denoiser: this._cfg.denoiser,
      neuralWeights: this._cfg.neuralWeights,
      neuralTensorStorage: this._cfg.neuralTensorStorage,
      oidnModelUrl: this._cfg.oidnModelUrl,
      oidnExecutionProviders: this._cfg.oidnExecutionProviders,
      verbose: this._cfg.verbose,
      debug: this._cfg.debug,
      cameraMoveResetThresholdSq: this._cfg.initTunables.cameraMoveResetThresholdSq,
      temporalAccumAlpha: this._cfg.initTunables.temporalAccumAlpha,
      checkerboardMotionThresholdSq: this._cfg.initTunables.checkerboardMotionThresholdSq,
      ctorLights: this._ctorLights,
      ddgi: this._ddgi,
      gtaoMode: this._cfg.gtaoMode,
      diSpatialPasses: this._cfg.diSpatialPasses,
      giSpatialPasses: this._cfg.giSpatialPasses,
      // NRC live cache — same COMPILE-TIME structural gate discipline as
      // the sole generalized-reuse GI path. `nrcEnabled` (0/1) also drives the
      // UBO flag; here we forward the boolean so the pipeline builds the
      // matching gi-ris layout (4-group DDGI default vs 5-group inline-MLP
      // variant).
      nrcEnabled: this._cfg.nrcEnabled === 1,
      nrcConfig: this._cfg.nrcConfig,
      // PPG guided sampling — builds the ppg-update pipeline + UBO gate.
      ppgEnabled: this._cfg.ppgEnabled === 1,
      // H47 — PPG max spatial cells. undefined ⇒ allocatePPGResources default (1 024).
      ppgMaxSpatialCells: this._cfg.ppgMaxSpatialCells,
      // H29 — PPG max per-cell dTree nodes. undefined ⇒ default 341-node stride.
      ppgMaxDTreeNodesPerCell: this._cfg.ppgMaxDTreeNodesPerCell,
      // PPG guide/cosine MIS mixture alpha.
      ppgMixAlpha: this._cfg.ppgMixAlpha,
      // Checkerboard half-res shading — flips the ResolvePass gate + the
      // per-frame shade UBO fields. OFF (default) is bit-identical.
      checkerboard: this._cfg.checkerboard,
      regirConfig: this._cfg.regirConfig,
    };
  }

  /** Build the back-reference the {@link PipelineInitCoordinator} consumes.
   *  Live-mutable inputs are getters closing over `this`; construction-time
   *  immutables are spread from {@link _initStaticConfig}. The coordinator
   *  never sees raw field references, only the small documented surface in
   *  `HybridEngineLifecycle.ts`. */
  private _buildInitHost(): PipelineInitHost {
    // eslint-disable-next-line @typescript-eslint/no-this-alias -- `self` required for the `get width()` / `get height()` etc. getters inside the returned object literal where `this` would refer to the object, not the class instance
    const self = this;
    return {
      ...this._initStaticConfig(),
      // Pipeline initializes at the INTERNAL render size (= canvas ×
      // resolutionFactor). Equal to the canvas size on first init (factor
      // 1.0); after a factor was applied, a reset() re-inits at the live
      // internal size so the composite upscale stays correct.
      get width() {
        return self._internalWidth;
      },
      get height() {
        return self._internalHeight;
      },
      get lastScene() {
        return self._renderScene;
      },
      get primaryLightDir() {
        return self._effectivePrimaryLightDir();
      },
      get primaryLightDirOverride() {
        return self._primaryLightDirOverride();
      },
      get primaryLightIntensity() {
        return self._primaryLightIntensity;
      },
      get ppgDispatchInterval() {
        return self._ppgDispatchInterval;
      },
      get hostSunWarningState() {
        return self._hostSunWarningState;
      },
      get preferredSwapChainFormat() {
        return getPreferredSwapChainFormat();
      },
      get currentBvhBuffers() {
        return self._bvhBuffers;
      },

      isSceneReadyForBvh: () => self._sceneReadyForBvh(),
      coreSceneSuppliesMeshes: () => self._coreSceneSuppliesMeshes(),

      publishBvh: (bvh) => {
        self._bvhBuffers = bvh;
        self._warnMaterialTextureAtlasDiagnostics(bvh.materialTextureAtlas.diagnostics, 'setScene');
        propagateBvhToGiSubsystems({
          ddgi: self._ddgi,
          // RC adopts the pipeline's canonical GPU arena after initialize();
          // avoid a second geometry BVH at this earlier CPU-only checkpoint.
          rc: null,
          bvhBuffers: bvh,
          lastScene: self._renderScene,
          syncDdgi: true,
          allowRcSceneRebuild: true,
        });
      },
      publishPipeline: (p) => {
        // A host may retune PPG while async pipeline.initialize() is in flight.
        // Re-apply the live engine value at the publication boundary so the
        // newly published generation cannot expose the stale value captured
        // when initialization began.
        p.setPpgDispatchInterval(self._ppgDispatchInterval);
        p.setDenoiserPassEnabled(self._denoiserPassEnabled);
        if (self._rc != null && self._bvhBuffers != null) {
          const sharedGeometry = p.getSceneGeometryBufferBindings();
          if (sharedGeometry == null) {
            throw new Error(
              '[HybridEngine] initialized pipeline did not publish scene geometry bindings.',
            );
          }
          self._rc.syncRestirBvhBuffers(self._bvhBuffers, sharedGeometry);
        }
        // Configure the just-built generation before publication. setScene()
        // resolves/caches its environment before async initialization starts;
        // applying it here makes initial loads and rebuilds observe that exact
        // payload instead of retaining the pipeline's no-HDRI placeholder.
        // If configuration throws, the coordinator still owns and disposes p,
        // and no partially configured generation becomes live.
        self._applyDirectionalEnvironmentToPipeline(
          p,
          self._resolvedEnvironment,
        );
        self._pipeline = p;
      },
      rollbackBvh: () => {
        self._bvhBuffers = null;
      },
      setState: (s) => {
        self._state =
          s === 'ready' && self._pauseRequested
            ? 'paused'
            : s;
      },
      reportError: (e) => {
        self._emitError(e);
      },
      reportWarning: (w) => {
        self._warn(w);
      },
      teardownPipeline: () => {
        self._teardownPipeline();
      },
      disposeDdgi: () => {
        self._ddgi.dispose();
      },
      // Mirror the synchronous dispose (HybridEngine.dispose) so a deferred
      // (dispose-races-init) teardown releases RC + skinning too, not just
      // pipeline/BVH/DDGI. RC dispose also nulls the handle, matching dispose().
      disposeRc: () => {
        if (self._rc) {
          self._rc.dispose();
          self._rc = null;
        }
      },
      disposeSkinning: () => {
        self._skinning?.dispose();
      },

      recordInitStart: () => {
        const d = self._dbg;
        d.initCount++;
        d.initStart = performance.now();
        console.log(`[hybrid:debug] init #${d.initCount} START`, {
          W: self._width,
          H: self._height,
          device: !!self._device,
          t: d.initStart.toFixed(0),
        });
      },
      recordInitComplete: (pipelineMs, totalMs) => {
        const d = self._dbg;
        console.log(`[hybrid:debug] init #${d.initCount} COMPLETE`, {
          pipelineMs: pipelineMs.toFixed(1),
          totalMs: totalMs.toFixed(1),
        });
      },
    };
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Factory
// ────────────────────────────────────────────────────────────────────────────

/**
 * The walkaround-hybrid backend's STABLE, public-facing surface BEYOND the
 * host-agnostic {@link Engine} contract: DDGI GI-state persistence (the "cached
 * light field"). {@link createWalkaroundEngine_Hybrid} returns
 * `Promise<HybridEngine>` so a host that deliberately picks
 * this backend by name gets the export/import methods typed — without the
 * universal contract baking in a backend-specific feature. (`@vitrum/engine`'s
 * `createEngine` facade forwards these same methods via its internal
 * `GIStatePersistable` shape; this is the backend-package-level peer of that.)
 */
/**
 * Create a HybridEngine instance and begin asynchronous pipeline initialisation.
 *
 * The engine is returned immediately in `'initializing'` state. The host
 * should poll `engine.state` or listen for the `'ready'` transition before
 * calling `renderFrame`.
 *
 * @param opts  Creation-time options. `opts.device` must be a live GPUDevice.
 */
export const createWalkaroundEngine_Hybrid: EngineFactory<
  HybridEngineOptions,
  HybridEngine
> = async (
  opts: HybridEngineOptions,
  // eslint-disable-next-line @typescript-eslint/require-await -- factory signature is async to match EngineFactory<…> contract; no async setup needed in this code path
): Promise<HybridEngine> => {
  // Duck-type GPUDevice validation — `instanceof GPUDevice` is not reliable
  // across realms; checking for a known required method is more robust.
  if (
    !opts.device ||
    typeof (opts.device as { createCommandEncoder?: unknown }).createCommandEncoder !== 'function'
  ) {
    throw new TypeError(
      '[createWalkaroundEngine_Hybrid] opts.device must be a live GPUDevice. ' +
        `Received value of type ${typeof opts.device}.`,
    );
  }

  const engine = new HybridEngine(opts);
  // Bootstrap setScene with an empty vitrum Scene. Two callers depend on
  // this:
  //   1. Hosts that DO call setScene afterwards (e.g. @vitrum/engine.createEngine).
  //      The host's setScene fires init-B which races init-A. The init-flight
  //      guard inside PipelineInitCoordinator (mySeq === _initSeq) ensures the
  //      loser bootstrap chain disposes its locals — no GPU resource leak.
  //      The bootstrap is wasted work but safe.
  engine.setScene({ primitives: [], emitters: [], environment: { kind: 'none' } });
  return engine;
};
