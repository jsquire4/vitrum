// Scene description — backend-agnostic.
//
// Capability filter — the single, declared-set-driven "warn + skip" layer.
//
// `EngineCapabilities` declares which primitive kinds, emitter kinds, analytic
// shapes, and environment kinds a backend can ingest. Historically each backend
// re-encoded that kind list inline at its scene-packing site (e.g. pt-webgpu's
// `buildPackedScene` re-encoded the emitter list as a boolean chain), so the
// declared capability set and the ingestion behavior were two sources of truth
// that could drift.
//
// `partitionSceneBySupport` is the one place that consumes a backend's OWN
// declared sets to decide what to keep. It walks the scene once, bins each
// node against the relevant set, returns the supported subset plus ready-made
// warning strings, and never mutates the input. Backends pack only `supported`
// and emit `warnings`.
//
// Design: "warn + skip" (behavior-preserving). Unsupported primitives/emitters
// are dropped with a warning. The environment is NEVER dropped — an unsupported
// environment kind only produces a warning, matching how every backend already
// handles environments today (they fall back rather than refusing the scene).

import type { ScenePrimitive, AnalyticShape } from './primitives.js';
import type { SceneEmitter } from './emitters.js';
import type { SceneEnvironment } from './environment.js';
import type { Scene } from './index.js';
import { analyticPrimitiveToMesh } from './analyticToMesh.js';

/** The capability facets `partitionSceneBySupport` consults. Structurally a
 *  subset of {@link EngineCapabilities} so a backend can pass `this.capabilities`
 *  (or any object carrying these fields) directly. Omitting a `Set` means "the
 *  backend places no kind restriction on that facet" — all nodes of that facet
 *  are kept. */
export interface SupportSets {
  readonly supportedPrimitiveKinds?: ReadonlySet<ScenePrimitive['kind']>;
  readonly supportedEmitterKinds?: ReadonlySet<SceneEmitter['kind']>;
  readonly supportedAnalyticShapes?: ReadonlySet<AnalyticShape>;
  readonly supportedEnvironmentKinds?: ReadonlySet<SceneEnvironment['kind']>;
}

export interface PartitionedScene {
  /** A NEW Scene carrying only the nodes the backend can ingest. The original
   *  scene is never mutated. The environment is carried through unchanged (it
   *  is warned-about, not dropped). */
  readonly supported: Scene;
  /** Human-readable warning per dropped node (and per unsupported environment).
   *  Empty when every node is supported. */
  readonly warnings: string[];
}

/**
 * Partition `scene` into the subset a backend can ingest plus a list of
 * warnings for everything dropped. Pure — `scene` is never mutated.
 *
 * Binning rules:
 *  - A primitive is kept when its `kind` is in `supportedPrimitiveKinds`
 *    (or the set is omitted) AND, for `analytic` primitives, its `shape` is in
 *    `supportedAnalyticShapes` (or that set is omitted).
 *  - When an `analytic` primitive is unsupported (kind or shape) and `mesh` is
 *    accepted, the primitive is converted to a `MeshPrimitive` via
 *    `analyticPrimitiveToMesh` and placed in the supported partition instead of
 *    being dropped. An authored `fallbackMesh` is preferred; otherwise the
 *    canonical shape tessellator generates the mesh. A warning still documents
 *    the conversion.
 *  - An emitter is kept when its `kind` is in `supportedEmitterKinds` (or the
 *    set is omitted) AND, for `mesh-area` emitters, its `meshId` still resolves
 *    against the supported primitive partition. A `mesh-area` emitter whose
 *    target primitive was dropped is dropped too, so `supported` never carries
 *    a dangling `meshId`.
 *  - The environment is always carried through; an unsupported `kind` only
 *    appends a warning (backends fall back, they don't refuse the scene).
 */
export function partitionSceneBySupport(scene: Scene, caps: SupportSets): PartitionedScene {
  const warnings: string[] = [];

  // Whether a MeshPrimitive could be placed in the supported partition.
  const meshKindAccepted =
    caps.supportedPrimitiveKinds == null || caps.supportedPrimitiveKinds.has('mesh');

  const primitives = scene.primitives.flatMap((primitive): ScenePrimitive[] => {
    // ── Kind check ──────────────────────────────────────────────────────────
    if (
      caps.supportedPrimitiveKinds != null &&
      !caps.supportedPrimitiveKinds.has(primitive.kind)
    ) {
      // Every public analytic shape has a canonical mesh conversion. Prefer an
      // authored fallbackMesh when present, but never discard representable
      // geometry merely because the host omitted one.
      if (primitive.kind === 'analytic' && meshKindAccepted) {
        const conversion = primitive.fallbackMesh != null
          ? 'converting via fallbackMesh'
          : 'converting via canonical generated mesh';
        warnings.push(
          `Scene primitive "${primitive.id}" (analytic kind) is not supported by this backend; ` +
          `${conversion} to a MeshPrimitive.`,
        );
        return [analyticPrimitiveToMesh(primitive)];
      }
      warnings.push(
        `Scene primitive "${primitive.id}" (${primitive.kind}) is not supported by this backend; skipping.`,
      );
      return [];
    }
    // ── Shape check (analytic only) ──────────────────────────────────────────
    if (
      primitive.kind === 'analytic' &&
      caps.supportedAnalyticShapes != null &&
      !caps.supportedAnalyticShapes.has(primitive.shape)
    ) {
      if (meshKindAccepted) {
        const conversion = primitive.fallbackMesh != null
          ? 'converting via fallbackMesh'
          : 'converting via canonical generated mesh';
        warnings.push(
          `Scene primitive "${primitive.id}" (analytic shape "${primitive.shape}") is not supported by this backend; ` +
          `${conversion} to a MeshPrimitive.`,
        );
        return [analyticPrimitiveToMesh(primitive)];
      }
      warnings.push(
        `Scene primitive "${primitive.id}" (analytic shape "${primitive.shape}") is not supported by this backend; skipping.`,
      );
      return [];
    }
    return [primitive];
  });

  // Referential invariant: the supported partition must itself be a VALID
  // Scene. A `mesh-area` emitter names a primitive by id, so binning emitters
  // on `kind` alone can carry an emitter whose `meshId` points at a primitive
  // the primitive filter just dropped — a dangling reference `validateScene`
  // rejects. The survivor set is taken from the POST-filter primitive list so
  // analytic→mesh conversions (which preserve the id) correctly count as
  // survivors.
  const survivingPrimitiveIds = new Set(primitives.map((primitive) => primitive.id));

  const emitters = scene.emitters.filter((emitter) => {
    if (caps.supportedEmitterKinds != null && !caps.supportedEmitterKinds.has(emitter.kind)) {
      warnings.push(
        `Scene emitter "${emitter.id}" (${emitter.kind}) is not supported by this backend; skipping.`,
      );
      return false;
    }
    if (emitter.kind === 'mesh-area' && !survivingPrimitiveIds.has(emitter.meshId)) {
      warnings.push(
        `Scene emitter "${emitter.id}" (mesh-area) references primitive "${emitter.meshId}", ` +
        `which is not supported by this backend; skipping.`,
      );
      return false;
    }
    return true;
  });

  if (
    caps.supportedEnvironmentKinds != null &&
    !caps.supportedEnvironmentKinds.has(scene.environment.kind)
  ) {
    warnings.push(
      `Scene environment "${scene.environment.kind}" is not supported by this backend; falling back.`,
    );
  }

  return {
    supported: {
      ...scene,
      primitives,
      emitters,
    },
    warnings,
  };
}
