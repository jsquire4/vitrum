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
import type { EngineWarning, MaterialSpec, Scene, SceneEmitter, ScenePrimitive } from '@vitrum/core';
import { BACKEND_PROMISE_LEDGER, MATERIAL_SPEC_FIELDS, asMat4, solveSkin } from '@vitrum/core';
import type { ScenePackResult } from '@vitrum/shared-bvh';
import {
  BVH_NODE_FLOATS,
  fingerprintTlasBuffers,
  rebuildPrimitiveBlas,
  rebuildTlasReuseBlas,
} from '@vitrum/shared-bvh';
import { invertMat4 } from './math/mat4.js';
import {
  applyEnvironmentMutation,
  packFoldedMaterialEntry,
  rebuildLightTreeForScene,
  rebuildTlasForSceneTransforms,
  uploadEmitterArrays,
  uploadScenePackGeometry,
  uploadScenePackGeometryRealloc,
  uploadScenePackBlasOnly,
  uploadScenePackTlasOnly,
  uploadScenePackTlasRealloc,
  type UploadedSceneBuffers,
} from './scene/uploadSceneBuffers.js';
import {
  analyticIndexForPrimitive,
  canFastPathGeometryPatch,
  canFastPathInstancedTopologyPatch,
  canFastPathMaterialPatch,
  canFastPathTopologyResizePatch,
  canFastPathTransformPatch,
  canReuseTlasBufferLengths,
  materialIndexForPrimitive,
} from './scene/incrementalPatch.js';
import { patchEmitterInScene, patchPrimitiveInScene } from './scene/patchScene.js';
import { MATERIAL_FLOAT_STRIDE } from './scene/materialPacking.js';
import {
  defaultDirectionalAngularDiameter,
  defaultDirectionalIrradiance,
  defaultDirectionalLight,
  hasMeshAreaEmitterForPrimitive,
  packEmitterArrays,
  type EnvSummaryForTree,
} from './scene/emitterPacking.js';
import { environmentParams } from './scene/environmentPacking.js';

const IDENTITY_MAT4 = asMat4(new Float32Array([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
]));

const UNSUPPORTED_DISPLACEMENT_MATERIAL_FIELDS = [
  'displacementMap',
  'displacementScale',
  'displacementBias',
] as const;

function collectUnsupportedDisplacementPatchFields(
  material: Record<string, unknown>,
): string[] {
  const fields: string[] = [];
  for (const field of UNSUPPORTED_DISPLACEMENT_MATERIAL_FIELDS) {
    if (material[field] != null) fields.push(field);
  }
  return fields;
}

// CAP-01 — matrix-driven list of the remaining material fields pt-webgpu
// silently drops (every 'unsupported' ledger row except displacement, which has
// its own dedicated warning, and the host-discretionary `extensions` hatch).
// Mirrors UNSUPPORTED_MATERIAL_FIELDS in index.ts.
const UNSUPPORTED_PATCH_MATERIAL_FIELDS: readonly (keyof MaterialSpec)[] = MATERIAL_SPEC_FIELDS.filter(
  (field) =>
    BACKEND_PROMISE_LEDGER['pt-webgpu'].supportDetails.materials[field] === 'unsupported' &&
    !(UNSUPPORTED_DISPLACEMENT_MATERIAL_FIELDS as readonly string[]).includes(field) &&
    field !== 'extensions',
);

function collectUnsupportedPatchMaterialFields(
  material: Record<string, unknown>,
): string[] {
  const fields: string[] = [];
  for (const field of UNSUPPORTED_PATCH_MATERIAL_FIELDS) {
    if (material[field] != null) fields.push(field);
  }
  return fields;
}

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
  getGeoPack(): ScenePackResult | null;
  setGeoPack(pack: ScenePackResult): void;
  invalidateBindGroups(): void;
  supportedAnalyticShapes(): ReadonlySet<string>;
  /** Whether camera-visible emitters (emissive fold) is enabled for this engine instance. */
  cameraVisibleEmitters(): boolean;
  /** Refresh lite-tier sampled light/environment textures from the live scene buffers. */
  syncLiteTextures?(sceneBuffers: UploadedSceneBuffers): void;
  /** True when the engine selected the single-group lite shader tier. */
  isLiteTier?(): boolean;
  /** Structured warning sink owned by the engine; mirrors console.warn. */
  warn?(warning: EngineWarning, ...consoleArgs: readonly unknown[]): void;
  /** Full scene repack (engine-internal: destroys buffers + re-inits BDPT). */
  repackScene(scene: Scene, opts: { readonly warnOnEmpty: boolean }): void;
  /** Public setScene entry — the fall-through for every fast-path miss. */
  setScene(scene: Scene): void;
  reset(): void;
}

/** Commit returned by a fast-path handler, hoisted from the per-branch bodies. */
interface FastPathCommit {
  /** When set, becomes the new `#geoPack` before bind-group invalidation. */
  readonly geoPack?: ScenePackResult;
  /** Geometry/topology handlers invalidate cached bind groups; the cheap
   *  in-place writeBuffer handlers (analytic / transform / material) do not. */
  readonly invalidateBindGroups: boolean;
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

export class SceneMutationRouter {
  readonly #host: MutationHost;

  constructor(host: MutationHost) {
    this.#host = host;
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
    const device = host.device;
    // #assertLive already throws when the scene is null; the non-null assertion
    // captures that invariant for the type checker.
    const currentScene = host.getScene()!;
    const currentPrimitive = currentScene.primitives.find((p) => p.id === id) ?? null;

    // Item 1 — bones patch: when the host submits updated bone matrices (and/or
    // boneInverses/morphWeights) on a skinned-mesh, re-solve the skin and
    // rewrite the patch as a { positions, normals } geometry update so the
    // existing geometry fast paths (in-place BLAS refit or topology-resize)
    // handle the upload. This is the same pattern walkaround-hybrid uses in
    // applyGpuSkinnedRefit: solveSkin → updatePrimitive({ positions, normals }).
    //
    // A patch containing `bones` (or `boneInverses` or `morphWeights`) triggers
    // the resolution. The solved positions/normals are merged INTO the patch
    // (overriding any explicit positions/normals the caller also supplied) and
    // the bones* fields are stripped so only the geometry change is forwarded —
    // the scene-state update (patchPrimitiveInScene below) still captures the
    // new bones so future solveSkin calls use the correct matrices.
    let resolvedPatch = patch;
    const hasBonesPatch =
      currentPrimitive?.kind === 'skinned-mesh' &&
      ('bones' in patch || 'boneInverses' in patch || 'morphWeights' in patch);
    if (hasBonesPatch && currentPrimitive?.kind === 'skinned-mesh') {
      // Build the "next" state of the skinned-mesh to solve against.
      const nextPrim = {
        ...currentPrimitive,
        ...(patch),
      };
      try {
        const solved = solveSkin(nextPrim);
        // Strip bone-only keys and inject solved geometry.
        const { bones: _b, boneInverses: _bi, morphWeights: _mw, ...restPatch } = patch as {
          bones?: unknown; boneInverses?: unknown; morphWeights?: unknown;
          [k: string]: unknown;
        };
        resolvedPatch = {
          ...restPatch,
          positions: solved.positions,
          normals: solved.normals,
        };
      } catch (err) {
        warnHost(host, {
          code: 'pt-webgpu.update-primitive-skin-fallback',
          backend: 'pt-webgpu',
          phase: 'mutation',
          method: 'updatePrimitive',
          message: `[vitrum/pt-webgpu] solveSkin failed for updatePrimitive("${id}"); falling back to setScene. ${String(err)}`,
          details: { id },
          raw: err,
        });
        // Fall through to full setScene at the tail.
      }
    }

    // Build nextScene using the FULL merged patch: new bones (from `patch`) plus
    // solved positions/normals (from `resolvedPatch`). This ensures:
    //   (a) rebuildPrimitiveBlas sees solved positions, not rest-pose.
    //   (b) The committed scene state stores the new bone matrices so future
    //       solveSkin calls start from the correct bone pose.
    // For non-bones patches, resolvedPatch === patch, so this is a no-op.
    const mergedPatch: Partial<ScenePrimitive> = hasBonesPatch
      ? { ...patch, ...(resolvedPatch) }
      : patch;
    const nextScene = patchPrimitiveInScene(currentScene, id, mergedPatch);
    // The fast-path eligibility checks use only the geometry portion (solved
    // positions/normals, no bone keys) so they correctly classify the update
    // as a geometry-refit rather than an unrecognised bones-only patch.
    const fastPathPatch = resolvedPatch;

    const liteUnsupportedTransformPatch =
      host.isLiteTier?.() === true &&
      currentPrimitive != null &&
      canFastPathTransformPatch(currentPrimitive, fastPathPatch);
    const liteUnsupportedInstancedTopologyPatch =
      host.isLiteTier?.() === true &&
      currentPrimitive != null &&
      canFastPathInstancedTopologyPatch(currentPrimitive, fastPathPatch);
    if (liteUnsupportedTransformPatch || liteUnsupportedInstancedTopologyPatch) {
      throw new Error(
        '[vitrum/pt-webgpu] updatePrimitive: transform-only and instanced topology ' +
          'patches are unsupported on the lite tier because the lite shader does not ' +
          'traverse TLAS instance transforms. Use the full tier, or rebuild/bake the ' +
          'scene geometry before calling setScene().',
      );
    }

    if (host.isLiteTier?.() === true && canFastPathMaterialPatch(fastPathPatch)) {
      host.setScene(nextScene);
      return;
    }

    // The incremental fast paths, in FIRST-ELIGIBLE-WINS order (geometry →
    // topology-resize → analytic-transform → instanced-topology → transform →
    // material). Each handler does its own re-upload work and returns a
    // {@link FastPathCommit} on success, or `null` to fall through to the next
    // handler (and, if none succeed, to the full `setScene` rebuild at the
    // tail). The common commit — set `#geoPack`/invalidate-bind-groups when the
    // handler asks, set `#scene`, drain the handler's warnings, then `reset()`
    // — is hoisted out of the per-branch bodies below.
    const fastPaths: ReadonlyArray<() => FastPathCommit | null> = [
      // 1) geometry-refit: in-place BLAS rewrite (same vertex/index counts).
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
        const rebuilt = rebuildPrimitiveBlas(nextScene, id, geoPack, {
          tlas: true,
          resolveMaterialId: (pid) =>
            materialIndexForPrimitive(nextScene, pid, host.supportedAnalyticShapes()) ?? 0,
        });
        if (!rebuilt.ok) return null;
        const sb = sceneBuffers;
        const prevTlasFp = fingerprintTlasBuffers({
          tlasNodes: sb.tlasNodes,
          tlasInstanceIndices: sb.tlasInstanceIndices,
          tlasBlasRoots: sb.tlasBlasRoots,
          tlasInstanceWorldToLocal: sb.tlasInstanceWorldToLocal,
          tlasInstanceLocalToWorld: sb.tlasInstanceLocalToWorld,
        });
        const nextTlasFp = fingerprintTlasBuffers({
          tlasNodes: rebuilt.pack.tlasNodes,
          tlasInstanceIndices: rebuilt.pack.tlasInstanceIndices,
          tlasBlasRoots: rebuilt.pack.tlasBlasRoots,
          tlasInstanceWorldToLocal: rebuilt.pack.tlasInstanceWorldToLocal,
          tlasInstanceLocalToWorld: rebuilt.pack.tlasInstanceLocalToWorld,
        });
        if (prevTlasFp === nextTlasFp) {
          uploadScenePackBlasOnly(device, sb, rebuilt.pack);
        } else {
          uploadScenePackGeometry(device, sb, rebuilt.pack);
        }
        return {
          geoPack: rebuilt.pack,
          invalidateBindGroups: true,
          warnings: rebuilt.pack.warnings,
          reshapedWorldPositions: true,
        };
      },
      // 2) topology-resize: a (skinned-)mesh's vertex/index COUNT changed. Rebuild
      // ONLY this primitive's BLAS and splice it into the packed scene — growing/
      // shrinking the concat buffers and rebasing every downstream primitive's
      // offsets + the TLAS BLAS roots. The concat buffers change SIZE, so the (5)
      // BLAS + (5) TLAS GPU buffers must be reallocated (no in-place writeBuffer).
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
        const rebuilt = rebuildPrimitiveBlas(nextScene, id, geoPack, {
          tlas: true,
          resolveMaterialId: (pid) =>
            materialIndexForPrimitive(nextScene, pid, host.supportedAnalyticShapes()) ?? 0,
        });
        // rebuildPrimitiveBlas rejected (primitive missing / not mesh-like) — fall
        // through to the full setScene rebuild below.
        if (!rebuilt.ok) return null;
        uploadScenePackGeometryRealloc(device, sceneBuffers, rebuilt.pack);
        return {
          geoPack: rebuilt.pack,
          invalidateBindGroups: true,
          warnings: rebuilt.pack.warnings,
          reshapedWorldPositions: true,
        };
      },
      // 3) analytic-transform: in-place rewrite of one analytic primitive's
      // local↔world matrices (no BVH touch).
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
          nextScene,
          id,
          host.supportedAnalyticShapes(),
        );
        if (analyticIndex == null) return null;
        const nextPrimitive = nextScene.primitives.find((p) => p.id === id);
        if (nextPrimitive == null || nextPrimitive.kind !== 'analytic') return null;
        const localToWorld = asMat4(nextPrimitive.transform ?? IDENTITY_MAT4);
        const maybeWorldToLocal = invertMat4(localToWorld);
        const worldToLocal = asMat4(maybeWorldToLocal ?? IDENTITY_MAT4);
        if (maybeWorldToLocal == null) {
          warnHost(host, {
            code: 'pt-webgpu.noninvertible-analytic-transform',
            backend: 'pt-webgpu',
            phase: 'mutation',
            method: 'updatePrimitive',
            message: `[vitrum/pt-webgpu] Primitive "${nextPrimitive.id}" has non-invertible analytic transform; using identity fallback.`,
            details: { id: nextPrimitive.id },
          });
        }
        const byteOffset = analyticIndex * 16 * Float32Array.BYTES_PER_ELEMENT;
        device.queue.writeBuffer(
          sceneBuffers.analyticLocalToWorldBuffer,
          byteOffset,
          localToWorld.buffer,
          localToWorld.byteOffset,
          localToWorld.byteLength,
        );
        device.queue.writeBuffer(
          sceneBuffers.analyticWorldToLocalBuffer,
          byteOffset,
          worldToLocal.buffer,
          worldToLocal.byteOffset,
          worldToLocal.byteLength,
        );
        sceneBuffers.analyticLocalToWorld.set(localToWorld, analyticIndex * 16);
        sceneBuffers.analyticWorldToLocal.set(worldToLocal, analyticIndex * 16);
        return { invalidateBindGroups: false, warnings: [] };
      },
      // 4) instanced-topology: instanced-mesh instance COUNT changed. BLAS
      // geometry is shared across instances and byte-identical, so we rebuild
      // only the TLAS (reusing the previous pack's BLAS arrays verbatim — no
      // per-triangle buildArrayBvh) and reallocate only the 5 TLAS GPU buffers.
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
        const rebuilt = rebuildTlasReuseBlas(nextScene, geoPack);
        // rebuildTlasReuseBlas rejected (e.g. concurrent geometry change) — fall
        // through to the full setScene rebuild below.
        if (!rebuilt.ok) return null;
        uploadScenePackTlasRealloc(device, sceneBuffers, {
          tlasNodes: rebuilt.pack.tlasNodes,
          tlasInstanceIndices: rebuilt.pack.tlasInstanceIndices,
          tlasBlasRoots: rebuilt.pack.tlasBlasRoots,
          tlasInstanceWorldToLocal: rebuilt.pack.tlasInstanceWorldToLocal,
          tlasInstanceLocalToWorld: rebuilt.pack.tlasInstanceLocalToWorld,
          tlasNodeCount: rebuilt.pack.tlasNodeCount,
          primitiveTlasBindings: rebuilt.pack.primitiveTlasBindings,
        });
        return {
          geoPack: rebuilt.pack,
          invalidateBindGroups: true,
          warnings: rebuilt.pack.warnings,
          reshapedWorldPositions: true,
        };
      },
      // 5) transform-only: rebuild the TLAS from the patched world transforms and
      // upload in place (only when the TLAS buffer lengths are reusable).
      () => {
        const sceneBuffers = host.getSceneBuffers();
        if (
          currentPrimitive == null ||
          sceneBuffers == null ||
          !canFastPathTransformPatch(currentPrimitive, fastPathPatch)
        ) {
          return null;
        }
        const sb = sceneBuffers;
        const tlas = rebuildTlasForSceneTransforms(nextScene, sb.primitiveTlasBindings, {
          tlasNodes: sb.tlasNodes,
          tlasInstanceIndices: sb.tlasInstanceIndices,
          tlasBlasRoots: sb.tlasBlasRoots,
          tlasInstanceWorldToLocal: sb.tlasInstanceWorldToLocal,
        });
        if (!tlas.ok || !canReuseTlasBufferLengths(sb, tlas)) return null;
        uploadScenePackTlasOnly(device, sb, {
          tlasNodes: tlas.tlasNodes,
          tlasInstanceIndices: tlas.tlasInstanceIndices,
          tlasBlasRoots: tlas.tlasBlasRoots,
          tlasInstanceWorldToLocal: tlas.tlasInstanceWorldToLocal,
          tlasInstanceLocalToWorld: tlas.tlasInstanceLocalToWorld,
          tlasNodeCount: Math.floor(tlas.tlasNodes.length / BVH_NODE_FLOATS),
          primitiveTlasBindings: sb.primitiveTlasBindings,
        });
        return { invalidateBindGroups: false, warnings: tlas.warnings, reshapedWorldPositions: true };
      },
      // 6) material-only: in-place rewrite of one material slot.
      // H10 — also re-applies the emissive fold when cameraVisibleEmitters is on
      // and this primitive is backed by a mesh-area emitter, so a roughness/color
      // patch on the primitive doesn't lose the fold that was applied at setScene.
      // Item 2c — when the patch changes emissive/emissiveIntensity, set
      // `changedEmissiveField` so the common-commit code re-packs the implicit
      // mesh-area emitter and rebuilds the light tree.
      () => {
        const sceneBuffers = host.getSceneBuffers();
        if (!canFastPathMaterialPatch(fastPathPatch) || sceneBuffers == null) return null;
        const materialIndex = materialIndexForPrimitive(
          nextScene,
          id,
          host.supportedAnalyticShapes(),
        );
        const primitive = nextScene.primitives.find((p) => p.id === id);
        if (materialIndex == null || primitive == null) return null;
        // Use packFoldedMaterialEntry so the fold is preserved when cameraVisibleEmitters
        // is on — a plain materialToPackedVec4s call would strip the fold from the slot.
        const packed = packFoldedMaterialEntry(
          primitive,
          nextScene,
          host.cameraVisibleEmitters(),
        );
        if (packed.length !== MATERIAL_FLOAT_STRIDE) return null;
        const materialData = new Float32Array(packed);
        const floatOffset = materialIndex * MATERIAL_FLOAT_STRIDE;
        const byteOffset = floatOffset * Float32Array.BYTES_PER_ELEMENT;
        device.queue.writeBuffer(
          sceneBuffers.materialsBuffer,
          byteOffset,
          materialData.buffer,
          materialData.byteOffset,
          materialData.byteLength,
        );
        sceneBuffers.materials.set(materialData, floatOffset);
        // Item 2c — detect emissive-field changes so the implicit NEE emitter
        // (H14-A) is re-packed with the new radiance.
        const mat = (fastPathPatch as unknown as { material?: Record<string, unknown> }).material ?? {};
        const unsupportedDisplacementFields = collectUnsupportedDisplacementPatchFields(mat);
        if (unsupportedDisplacementFields.length > 0) {
          warnHost(host, {
            code: 'pt-webgpu.unsupported-displacement-material',
            backend: 'pt-webgpu',
            phase: 'mutation',
            method: 'updatePrimitive',
            message:
              `[vitrum/pt-webgpu] updatePrimitive("${id}"): displacement material fields are supplied ` +
              `but not rendered by this backend: ${unsupportedDisplacementFields.join(', ')}.`,
            details: { id, fields: unsupportedDisplacementFields },
          });
        }
        // CAP-01 — same matrix-driven warning for the remaining dropped fields.
        const unsupportedMaterialFields = collectUnsupportedPatchMaterialFields(mat);
        if (unsupportedMaterialFields.length > 0) {
          warnHost(host, {
            code: 'pt-webgpu.unsupported-material-fields',
            backend: 'pt-webgpu',
            phase: 'mutation',
            method: 'updatePrimitive',
            message:
              `[vitrum/pt-webgpu] updatePrimitive("${id}"): material fields are supplied ` +
              `but not rendered by this backend: ${unsupportedMaterialFields.join(', ')}.`,
            details: { id, fields: unsupportedMaterialFields },
          });
        }
        const changedEmissiveField =
          'emissive' in mat || 'emissiveIntensity' in mat;
        return { invalidateBindGroups: false, warnings: [], changedEmissiveField };
      },
    ];

    for (const tryFastPath of fastPaths) {
      const commit = tryFastPath();
      if (commit == null) continue;
      if (commit.geoPack != null) host.setGeoPack(commit.geoPack);
      if (commit.invalidateBindGroups) host.invalidateBindGroups();
      // H11 — when a geometry/transform fast path moved world-space vertex positions
      // or transforms, and the patched primitive is backed by a mesh-area emitter,
      // the GPU meshAreaLightsBuffer is stale (it holds pre-move world-space
      // triangle vertices). Re-run packEmitterArrays + upload to keep NEE in sync.
      // Item 2c — also re-pack when the material fast path changed emissive fields
      // (even without geometric movement) so the implicit NEE emitter (H14-A) sees
      // the new radiance and the light-tree importance weights are updated.
      const sceneBuffersForEmitters = host.getSceneBuffers();
      if (
        (commit.reshapedWorldPositions || commit.changedEmissiveField) &&
        sceneBuffersForEmitters != null &&
        (
          hasMeshAreaEmitterForPrimitive(currentScene, id) ||
          hasMeshAreaEmitterForPrimitive(nextScene, id)
        )
      ) {
        const emitterPacked = packEmitterArrays(nextScene);
        const emittersReallocated = uploadEmitterArrays(device, sceneBuffersForEmitters, emitterPacked, {
          directionalLight: defaultDirectionalLight(nextScene),
          directionalIrradiance: defaultDirectionalIrradiance(nextScene),
          directionalAngularDiameter: defaultDirectionalAngularDiameter(nextScene),
        });
        // Thread already-computed emitterPacked into rebuildLightTreeForScene so
        // it skips the redundant packEmitterArrays call inside buildLightTreeInputForScene.
        const lightTreeReallocated = rebuildLightTreeForScene(device, sceneBuffersForEmitters, nextScene, { packed: emitterPacked });
        if (emittersReallocated || lightTreeReallocated) {
          host.invalidateBindGroups();
        }
        host.syncLiteTextures?.(sceneBuffersForEmitters);
        for (const w of emitterPacked.warnings) {
          warnHost(host, {
            code: 'pt-webgpu.emitter-pack-warning',
            backend: 'pt-webgpu',
            phase: 'mutation',
            method: 'updatePrimitive',
            message: `[vitrum/pt-webgpu] ${w}`,
            details: { warning: w },
          });
        }
      }
      host.setSceneState(nextScene);
      for (const warning of commit.warnings) {
        warnHost(host, {
          code: 'pt-webgpu.primitive-mutation-warning',
          backend: 'pt-webgpu',
          phase: 'mutation',
          method: 'updatePrimitive',
          message: `[vitrum/pt-webgpu] ${warning}`,
          details: { warning },
        });
      }
      host.reset();
      return;
    }
    host.setScene(nextScene);
  }

  updateEmitter(id: string, patch: Partial<SceneEmitter>): void {
    const host = this.#host;
    const device = host.device;
    host.assertLive('updateEmitter');
    const currentScene = host.getScene()!;
    const nextScene = patchEmitterInScene(currentScene, id, patch);
    const sceneBuffers = host.getSceneBuffers();
    if (sceneBuffers != null) {
      const packed = packEmitterArrays(nextScene);
      const lightsReallocated = uploadEmitterArrays(device, sceneBuffers, packed, {
        directionalLight: defaultDirectionalLight(nextScene),
        directionalIrradiance: defaultDirectionalIrradiance(nextScene),
        directionalAngularDiameter: defaultDirectionalAngularDiameter(nextScene),
      });
      // WS2 — the light tree's powers/positions depend on the emitters, so
      // rebuild + re-upload it (reallocating + invalidating bind groups if the
      // node count changed). Without this the GPU selection would importance-
      // sample the OLD light set after an incremental emitter patch.
      // Thread already-computed `packed` so buildLightTreeInputForScene skips
      // the redundant packEmitterArrays call.
      const lightTreeReallocated = rebuildLightTreeForScene(device, sceneBuffers, nextScene, { packed });
      if (lightsReallocated || lightTreeReallocated) {
        host.invalidateBindGroups();
      }
      host.syncLiteTextures?.(sceneBuffers);
      // H10 — emissive-fold desync fix: when cameraVisibleEmitters is on and the
      // patched emitter is a mesh-area emitter, re-write the material slot of the
      // backed primitive so the kernel's emissive-on-hit term stays in sync with
      // the emitter's new color/intensity. Without this, a color patch to the
      // emitter leaves the material slot stale until the next full repack.
      if (host.cameraVisibleEmitters()) {
        const updatedEmitter = nextScene.emitters.find((e) => e.id === id);
        if (updatedEmitter?.kind === 'mesh-area') {
          const backedPrimitive = nextScene.primitives.find(
            (p) => p.id === updatedEmitter.meshId,
          );
          if (backedPrimitive != null && backedPrimitive.kind !== 'analytic') {
            const matIndex = materialIndexForPrimitive(
              nextScene,
              backedPrimitive.id,
              host.supportedAnalyticShapes(),
            );
            if (matIndex != null) {
              const foldedPacked = packFoldedMaterialEntry(
                backedPrimitive,
                nextScene,
                true,
              );
              if (foldedPacked.length === MATERIAL_FLOAT_STRIDE) {
                const materialData = new Float32Array(foldedPacked);
                const floatOffset = matIndex * MATERIAL_FLOAT_STRIDE;
                const byteOffset = floatOffset * Float32Array.BYTES_PER_ELEMENT;
                device.queue.writeBuffer(
                  sceneBuffers.materialsBuffer,
                  byteOffset,
                  materialData.buffer,
                  materialData.byteOffset,
                  materialData.byteLength,
                );
                sceneBuffers.materials.set(materialData, floatOffset);
              }
            }
          }
        }
      }
      host.setSceneState(nextScene);
      for (const warning of packed.warnings) {
        warnHost(host, {
          code: 'pt-webgpu.emitter-mutation-warning',
          backend: 'pt-webgpu',
          phase: 'mutation',
          method: 'updateEmitter',
          message: `[vitrum/pt-webgpu] ${warning}`,
          details: { warning },
        });
      }
      host.reset();
      return;
    }
    host.setScene(nextScene);
  }

  updateEnvironment(env: Scene['environment'] | null): void {
    const host = this.#host;
    const device = host.device;
    host.assertLive('updateEnvironment');
    const currentScene = host.getScene()!;
    const nextScene: Scene = {
      ...currentScene,
      environment: env ?? { kind: 'none' },
    };
    const sceneBuffers = host.getSceneBuffers();
    if (sceneBuffers != null) {
      const packed = environmentParams(nextScene);
      const texelLenMatches = packed.hdriTexels.length === sceneBuffers.environmentMapTexels.length;
      const cdfLenMatches = packed.hdriCdf.length === sceneBuffers.environmentMapCdf.length;
      if (texelLenMatches && cdfLenMatches) {
        if (packed.hdriTexels.byteLength > 0) {
          device.queue.writeBuffer(
            sceneBuffers.environmentMapTexelsBuffer,
            0,
            packed.hdriTexels.buffer,
            packed.hdriTexels.byteOffset,
            packed.hdriTexels.byteLength,
          );
        }
        if (packed.hdriCdf.byteLength > 0) {
          device.queue.writeBuffer(
            sceneBuffers.environmentMapCdfBuffer,
            0,
            packed.hdriCdf.buffer,
            packed.hdriCdf.byteOffset,
            packed.hdriCdf.byteLength,
          );
        }
        applyEnvironmentMutation(sceneBuffers, {
          environmentTint: packed.tint,
          environmentSunDirection: packed.sunDirection,
          environmentSunStrength: packed.sunStrength,
          environmentHdriIntensity: packed.hdriIntensity,
          environmentHdriRotationY: packed.hdriRotationY,
          environmentMapWidth: packed.hdriWidth,
          environmentMapHeight: packed.hdriHeight,
          hasEnvironmentMap: packed.hasHdri,
        });
        sceneBuffers.environmentMapTexels.set(packed.hdriTexels);
        sceneBuffers.environmentMapCdf.set(packed.hdriCdf);
        // WS2 — the env counts as a selectable light in the NEE walk, so an env
        // change can flip the light-tree gate / leaf count. Rebuild + re-upload
        // (reallocating + invalidating bind groups if the node count changed).
        // Thread the already-computed env summary so buildLightTreeInputForScene
        // skips the redundant environmentParams call.
        const envSummaryForTree: EnvSummaryForTree = {
          hasHdri: packed.hasHdri,
          sunStrength: packed.sunStrength,
          tint: packed.tint,
        };
        if (rebuildLightTreeForScene(device, sceneBuffers, nextScene, { envSummary: envSummaryForTree })) {
          host.invalidateBindGroups();
        }
        host.syncLiteTextures?.(sceneBuffers);
        host.setSceneState(nextScene);
        for (const warning of packed.warnings) {
          warnHost(host, {
            code: 'pt-webgpu.environment-mutation-warning',
            backend: 'pt-webgpu',
            phase: 'mutation',
            method: 'updateEnvironment',
            message: `[vitrum/pt-webgpu] ${warning}`,
            details: { warning },
          });
        }
        host.reset();
        return;
      }
    }
    host.setScene(nextScene);
  }
}
