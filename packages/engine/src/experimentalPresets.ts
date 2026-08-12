/**
 * Named experimental niches. Each preset is a legal `createEngine` bag:
 * `prefer` plus `advancedByBackend` keys that the selected backend already
 * accepts. They do not invent lite BDPT / ReSTIR-PT / MNEE — those remain
 * construction-refused on lite adapters.
 *
 * Default `createEngine({ prefer:'auto' })` stays the viewer. These presets
 * opt into a research estimator explicitly.
 */

import type {
  CreateEngineAdvancedByBackend,
  CreateEngineOptions,
} from './createEngineInternals.js';

export const EXPERIMENTAL_PRESET_IDS = [
  'spectral-bdpt',
  'manifold-caustics',
  'photon-map-caustics',
  'one-edge-gris',
  'walkaround-nrc',
  'walkaround-ppg',
  'walkaround-rc',
] as const;

export type ExperimentalPresetId = (typeof EXPERIMENTAL_PRESET_IDS)[number];

const EXPERIMENTAL_PRESET_ID_SET: ReadonlySet<string> = new Set(EXPERIMENTAL_PRESET_IDS);

export function isExperimentalPresetId(value: unknown): value is ExperimentalPresetId {
  return typeof value === 'string' && EXPERIMENTAL_PRESET_ID_SET.has(value);
}

interface ExperimentalPresetBag {
  readonly prefer: NonNullable<CreateEngineOptions['prefer']>;
  readonly advancedByBackend: CreateEngineAdvancedByBackend;
}

const EXPERIMENTAL_PRESETS: Readonly<Record<ExperimentalPresetId, ExperimentalPresetBag>> = {
  'spectral-bdpt': {
    prefer: 'quality',
    advancedByBackend: {
      'pt-webgpu': { spectral: true, bdpt: true },
      'pt-webgl2': { spectral: true, bdpt: true },
    },
  },
  'manifold-caustics': {
    prefer: 'quality-webgpu',
    advancedByBackend: {
      'pt-webgpu': { causticStrategy: 'manifold-nee' },
    },
  },
  'photon-map-caustics': {
    prefer: 'quality-webgpu',
    advancedByBackend: {
      'pt-webgpu': { causticStrategy: 'photon-map' },
    },
  },
  'one-edge-gris': {
    prefer: 'quality-webgpu',
    advancedByBackend: {
      'pt-webgpu': { oneEdgeReconnectionReuse: true },
    },
  },
  'walkaround-nrc': {
    prefer: 'realtime',
    advancedByBackend: {
      'walkaround-hybrid': { nrcEnabled: true },
    },
  },
  'walkaround-ppg': {
    prefer: 'realtime',
    advancedByBackend: {
      'walkaround-hybrid': { ppgEnabled: true },
    },
  },
  'walkaround-rc': {
    prefer: 'realtime',
    advancedByBackend: {
      'walkaround-hybrid': { rcEnabled: true },
    },
  },
};

/**
 * Expand a named experimental preset into `prefer` + `advancedByBackend`.
 * Hosts may spread this into `createEngine` / `attachVitrum`, or pass
 * `experimentalPreset` on those options and let the factory merge it.
 */
export function experimentalPreset(id: ExperimentalPresetId): ExperimentalPresetBag {
  const preset = EXPERIMENTAL_PRESETS[id];
  if (preset == null) {
    throw new RangeError(
      `experimentalPreset: unknown id ${JSON.stringify(id)}; expected one of ` +
        EXPERIMENTAL_PRESET_IDS.map((entry) => JSON.stringify(entry)).join(', '),
    );
  }
  return {
    prefer: preset.prefer,
    advancedByBackend: {
      ...(preset.advancedByBackend['walkaround-hybrid'] != null
        ? { 'walkaround-hybrid': { ...preset.advancedByBackend['walkaround-hybrid'] } }
        : {}),
      ...(preset.advancedByBackend['pt-webgpu'] != null
        ? { 'pt-webgpu': { ...preset.advancedByBackend['pt-webgpu'] } }
        : {}),
      ...(preset.advancedByBackend['pt-webgl2'] != null
        ? { 'pt-webgl2': { ...preset.advancedByBackend['pt-webgl2'] } }
        : {}),
    },
  };
}

function mergeAdvancedByBackend(
  base: CreateEngineAdvancedByBackend | undefined,
  overlay: CreateEngineAdvancedByBackend | undefined,
): CreateEngineAdvancedByBackend | undefined {
  if (base == null) return overlay;
  if (overlay == null) return base;
  const merged: CreateEngineAdvancedByBackend = {
    ...(base['walkaround-hybrid'] != null || overlay['walkaround-hybrid'] != null
      ? {
          'walkaround-hybrid': {
            ...base['walkaround-hybrid'],
            ...overlay['walkaround-hybrid'],
          },
        }
      : {}),
    ...(base['pt-webgpu'] != null || overlay['pt-webgpu'] != null
      ? {
          'pt-webgpu': {
            ...base['pt-webgpu'],
            ...overlay['pt-webgpu'],
          },
        }
      : {}),
    ...(base['pt-webgl2'] != null || overlay['pt-webgl2'] != null
      ? {
          'pt-webgl2': {
            ...base['pt-webgl2'],
            ...overlay['pt-webgl2'],
          },
        }
      : {}),
  };
  return merged;
}

/** Apply `experimentalPreset` onto a validated createEngine payload. Host
 *  `prefer` and per-backend advanced keys win over the preset.
 *  @internal */
export function applyExperimentalPreset(opts: CreateEngineOptions): CreateEngineOptions {
  if (opts.experimentalPreset == null) return opts;
  const preset = experimentalPreset(opts.experimentalPreset);
  const { experimentalPreset: _id, ...rest } = opts;
  const advancedByBackend = mergeAdvancedByBackend(
    preset.advancedByBackend,
    rest.advancedByBackend,
  );
  return {
    ...rest,
    prefer: rest.prefer ?? preset.prefer,
    ...(advancedByBackend != null ? { advancedByBackend } : {}),
  };
}
