import type { SceneEmitter } from '../scene/emitters.js';
import type { SceneEnvironment } from '../scene/environment.js';
import type { MaterialSpec } from '../scene/material.js';
import type {
  AnalyticShape,
  ScenePrimitive,
} from '../scene/primitives.js';
import type {
  BackendSupportDetails,
  BackendSupportMode,
} from './capabilities.js';
import type { SupportSets } from '../scene/partitionSceneBySupport.js';

/**
 * Backend-owned executable support manifest.
 *
 * `BackendSupportDetails` keeps partial maps for source compatibility with
 * older hosts. A renderer implementation must use this exhaustive form:
 * adding a primitive, emitter, environment, analytic shape, or material field
 * to the core contract breaks every backend manifest at typecheck time until
 * the backend explicitly classifies it.
 */
export type BackendSupportManifest = Readonly<
  Omit<
    BackendSupportDetails,
    'primitives' | 'emitters' | 'environments' | 'analyticShapes' | 'materials'
  > & {
    readonly primitives: Readonly<
      Record<ScenePrimitive['kind'], BackendSupportMode>
    >;
    readonly emitters: Readonly<
      Record<SceneEmitter['kind'], BackendSupportMode>
    >;
    readonly environments: Readonly<
      Record<SceneEnvironment['kind'], BackendSupportMode>
    >;
    readonly analyticShapes: Readonly<
      Record<AnalyticShape, BackendSupportMode>
    >;
    readonly materials: Readonly<
      Record<keyof MaterialSpec, BackendSupportMode>
    >;
  }
>;

const IMPLEMENTED_SUPPORT_MODES: ReadonlySet<BackendSupportMode> =
  new Set<BackendSupportMode>([
    'native',
    'fallback-rebuild',
    'fallback-generated-mesh',
    'approximate',
  ]);

function isImplemented(mode: BackendSupportMode): boolean {
  return IMPLEMENTED_SUPPORT_MODES.has(mode);
}

/**
 * Convert a backend's detailed executable manifest into the exact coarse sets
 * consumed by `partitionSceneBySupport`.
 *
 * This deliberately derives coarse capability reporting and scene acceptance
 * from the same object. A row changed to `unsupported` therefore changes both
 * what a host reads and what the backend will accept.
 */
export function supportSetsFromManifest(
  manifest: BackendSupportManifest,
): Required<SupportSets> {
  return {
    supportedPrimitiveKinds: new Set(
      Object.entries(manifest.primitives)
        .filter(([, mode]) => isImplemented(mode))
        .map(([kind]) => kind as ScenePrimitive['kind']),
    ),
    supportedEmitterKinds: new Set(
      Object.entries(manifest.emitters)
        .filter(([, mode]) => isImplemented(mode))
        .map(([kind]) => kind as SceneEmitter['kind']),
    ),
    supportedEnvironmentKinds: new Set(
      Object.entries(manifest.environments)
        .filter(([, mode]) => isImplemented(mode))
        .map(([kind]) => kind as SceneEnvironment['kind']),
    ),
    supportedAnalyticShapes: new Set(
      Object.entries(manifest.analyticShapes)
        .filter(([, mode]) => isImplemented(mode))
        .map(([shape]) => shape as AnalyticShape),
    ),
  };
}

/**
 * Identity helper that preserves the exhaustive manifest type at declaration
 * sites. Its value is intentionally the very object consumed by renderer
 * validation and live capabilities; it is not copied from the static
 * cross-backend promise ledger.
 */
export function defineBackendSupportManifest(
  manifest: BackendSupportManifest,
): BackendSupportManifest {
  const seen = new WeakSet<object>();
  const freezeTree = (value: unknown): void => {
    if (value == null || typeof value !== 'object') return;
    if (seen.has(value)) return;
    seen.add(value);
    for (const key of Reflect.ownKeys(value)) {
      freezeTree((value as Record<PropertyKey, unknown>)[key]);
    }
    Object.freeze(value);
  };
  freezeTree(manifest);
  return manifest;
}
