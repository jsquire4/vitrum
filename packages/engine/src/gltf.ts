// @vitrum/engine/gltf — one-import glTF loading helpers.
//
// The glTF adapter remains the owner of asset loading, feature reporting,
// compatibility checks, texture diagnostics, and controller construction. This
// subpath only injects @vitrum/engine's createEngine facade for hosts that want
// a single import path.

import { loadGltfForEngine } from '@vitrum/gltf-adapter';
import type {
  GltfEngineSelection,
  GltfForEngineResult,
  LoadGltfForEngineOptions,
} from '@vitrum/gltf-adapter';
import { createEngine } from './createEngine.js';
import type {
  CreateEngineBackendId,
  CreateEngineOptions,
  EnginePreference,
  EngineWithBackendId,
} from './createEngine.js';

export { loadGltfForEngine } from '@vitrum/gltf-adapter';
export type {
  GltfAssetResult,
  GltfCompatibilityMode,
  GltfEngineSelection,
  GltfForEngineResult,
  LoadGltfForEngineOptions,
} from '@vitrum/gltf-adapter';

export type GltfCreateEngineOptions =
  Omit<CreateEngineOptions, 'scene' | 'prefer' | 'gltfAsset'> & {
    readonly prefer?: EnginePreference;
  };

export type LoadGltfWithEngineOptions = Omit<
  LoadGltfForEngineOptions<EngineWithBackendId, GltfCreateEngineOptions>,
  'createEngine' | 'engineOptions'
> & {
  readonly engineOptions?: GltfCreateEngineOptions;
};

export async function loadGltfWithEngine(
  input: Parameters<typeof loadGltfForEngine>[0],
  options: LoadGltfWithEngineOptions = {},
): Promise<GltfForEngineResult<EngineWithBackendId>> {
  const { engineOptions, ...adapterOptions } = options;
  return loadGltfForEngine<EngineWithBackendId, GltfCreateEngineOptions>(input, {
    ...adapterOptions,
    engineOptions: engineOptions ?? ({} as GltfCreateEngineOptions),
    createEngine: async ({ scene, backend, asset, options: createOptions }) => {
      return await createEngine({
        ...createOptions,
        scene,
        gltfAsset: asset,
        prefer: preferForSelectedBackend(backend, createOptions.prefer),
      });
    },
  });
}

function preferForSelectedBackend(
  backend: CreateEngineBackendId | GltfEngineSelection,
  fallback: EnginePreference | undefined,
): EnginePreference {
  if (backend === 'walkaround-hybrid') return 'realtime';
  if (backend === 'pt-webgpu') return 'quality-webgpu';
  if (backend === 'pt-webgl2') return 'quality';
  return fallback ?? 'auto';
}
