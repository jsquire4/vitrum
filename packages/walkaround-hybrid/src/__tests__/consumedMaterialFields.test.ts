/**
 * Tests for consumedMaterialFields — two layers:
 *
 *  Layer 1 (unit): `collectUnconsumedMaterialFields` directly — already a pure
 *    function with no GPU dependency, exercisable without a device stub.
 *
 *  Layer 2 (engine seam): `HybridEngine.setScene` wiring — verifies that the
 *    warn-once aggregated console.warn fires when unconsumed fields are present
 *    and stays silent when the scene uses only consumed fields.
 *
 *    HybridEngine requires a GPUDevice at construction time, but the warn logic
 *    is synchronous and runs before any GPU work. We use the same minimal device
 *    stub as capabilitiesPartition.test.ts; the async init chain (which touches
 *    real GPU APIs) is aborted by dispose() immediately after the synchronous
 *    setScene call under test.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { MockInstance } from 'vitest';
import type { EngineWarning, Scene, ScenePrimitive } from '@vitrum/core';
import { BACKEND_PROMISE_LEDGER, MATERIAL_SPEC_FIELDS } from '@vitrum/core';
import {
  categorizeUnconsumedMaterialFields,
  CONSUMED_MATERIAL_FIELDS,
  collectApproximateAlphaBlendPrimitiveIds,
  collectApproximateEmissiveMapTexelPdfPrimitiveIds,
  collectApproximateLightMapPrimitiveIds,
  collectApproximateRichMaterialPrimitiveFields,
  collectUnconsumedMaterialFields,
  collectUnconsumedMaterialFieldsForMaterial,
  collectUnconsumedMaterialPrimitiveFields,
  EMISSIVE_MAP_TEXEL_PDF_APPROXIMATION_DETAILS,
  LIGHT_MAP_CAMERA_VISIBLE_APPROXIMATION_DETAILS,
  RICH_MATERIAL_GI_APPROXIMATION_DETAILS,
} from '../restir/consumedMaterialFields.js';
import { HybridEngine } from '../HybridEngine.js';
import type { HybridEngineOptions } from '../HybridEngine.js';

// ── helpers ──────────────────────────────────────────────────────────────────

function makeDeviceStub(): GPUDevice {
  return {
    createCommandEncoder: vi.fn(),
    createBuffer: vi.fn(),
    createTexture: vi.fn(),
    createBindGroup: vi.fn(),
    queue: { submit: vi.fn(), writeBuffer: vi.fn() },
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    lost: new Promise<never>(() => {}),
  } as unknown as GPUDevice;
}

function makeOpts(): HybridEngineOptions {
  return {
    device: makeDeviceStub(),
    width: 64,
    height: 64,
    primaryLightDir: [0, -1, 0],
    primaryLightIntensity: 3,
    skyTint: [1, 1, 1],
    skyIrradiance: 1,
  };
}

/** Convenience alias for the parameter type of collectUnconsumedMaterialFields. */
type PrimLike = {
  readonly id?: string;
  readonly kind: string;
  readonly material?: Record<string, unknown>;
  readonly positions?: Float32Array;
  readonly colors?: Float32Array;
};

const WALKAROUND_PERMANENT_UNSUPPORTED_MATERIAL: Record<string, unknown> = {
  spectralAttenuation: {
    wavelengthStart: 380,
    wavelengthEnd: 700,
    values: new Float32Array([0.1, 0.2, 0.3]),
  },
  dispersionAbbeNumber: 42,
  scatteringCoefficient: 0.15,
  scatteringAnisotropy: 0.25,
  scatteringCoefficientRGB: [0.1, 0.2, 0.3],
  frontLayer: { transmission: [1, 0.5, 0.25] },
  backLayer: { transmission: [0.25, 0.5, 1] },
  thinFilmStack: { layers: [{ ior: 1.4, thicknessNm: 300 }] },
};

const WALKAROUND_PERMANENT_UNSUPPORTED_FIELDS = [
  'backLayer',
  'dispersionAbbeNumber',
  'frontLayer',
  'scatteringAnisotropy',
  'scatteringCoefficient',
  'scatteringCoefficientRGB',
  'spectralAttenuation',
  'thinFilmStack',
];

/** A scene with only consumed fields in its material. */
function consumedOnlyScene(): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'm1',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0 },
      } as unknown as ScenePrimitive,
    ],
    emitters: [],
    environment: { kind: 'none' },
  };
}

/** A scene whose material has `baseColorMap` (consumed) + `frontLayer` (unconsumed). */
function unconsumedFieldsScene(): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'm2',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: {
          baseColor: [1, 1, 1],
          roughness: 0.3,
          metallic: 0,
          baseColorMap: { handle: { width: 1, height: 1, data: new Uint8Array([255, 255, 255, 255]) } },
          frontLayer: { transmission: [1, 0.5, 0.25] }, // unconsumed
        },
      } as unknown as ScenePrimitive,
    ],
    emitters: [],
    environment: { kind: 'none' },
  };
}

// ── Layer 1 — unit tests for collectUnconsumedMaterialFields ─────────────────

describe('CONSUMED_MATERIAL_FIELDS allowlist', () => {
  it('contains the documented scalar fields', () => {
    for (const f of [
      'baseColor', 'roughness', 'metallic', 'emissive', 'emissiveIntensity',
      'shadingModel', 'alphaMode', 'alphaCutoff', 'opacity', 'transmission',
      'attenuationColor', 'attenuationDistance', 'thickness', 'thicknessMap', 'ior', 'extensions',
      'baseColorMap', 'normalMap', 'normalScale', 'roughnessMap', 'metallicMap', 'aoMap', 'aoMapIntensity', 'alphaMap',
      'bumpMap', 'bumpScale', 'displacementMap', 'displacementScale', 'displacementBias',
      'emissiveMap', 'transmissionMap', 'lightMap', 'lightMapIntensity', 'envMapIntensity',
      'specularColor', 'specularIntensity', 'clearcoat', 'clearcoatRoughness',
      'sheen', 'sheenColor', 'sheenRoughness',
      'specularColorMap', 'specularIntensityMap',
      'clearcoatMap', 'clearcoatRoughnessMap', 'clearcoatNormalMap', 'clearcoatNormalScale',
      'sheenColorMap', 'sheenRoughnessMap',
      'anisotropy', 'anisotropyRotation', 'anisotropyMap',
      'iridescence', 'iridescenceIor', 'iridescenceThicknessRange',
      'iridescenceMap', 'iridescenceThicknessMap',
    ]) {
      expect(CONSUMED_MATERIAL_FIELDS.has(f)).toBe(true);
    }
  });

  // CAP-01 — the ledger's per-field material support matrix and this package's
  // consumed-field allowlist are two views of the same code-derived truth.
  // Pin their equivalence: a field is in CONSUMED_MATERIAL_FIELDS exactly when
  // its walkaround-hybrid matrix row is not 'unsupported'.
  it('matches the BACKEND_PROMISE_LEDGER walkaround-hybrid material support matrix', () => {
    const matrix = BACKEND_PROMISE_LEDGER['walkaround-hybrid'].supportDetails.materials;
    for (const field of MATERIAL_SPEC_FIELDS) {
      const mode = matrix[field];
      expect(mode, `ledger row missing for ${field}`).toBeDefined();
      expect(
        CONSUMED_MATERIAL_FIELDS.has(field),
        `allowlist/ledger drift on '${field}' (ledger: ${String(mode)})`,
      ).toBe(mode !== 'unsupported');
    }
  });

  it('does NOT contain permanently unsupported walkaround fields', () => {
    for (const f of WALKAROUND_PERMANENT_UNSUPPORTED_FIELDS) {
      expect(CONSUMED_MATERIAL_FIELDS.has(f)).toBe(false);
    }
  });
});

describe('collectUnconsumedMaterialFields', () => {
  it('returns empty array when all material fields are consumed', () => {
    const scene = consumedOnlyScene();
    expect(collectUnconsumedMaterialFields(
      scene.primitives as unknown as ReadonlyArray<PrimLike>,
    )).toEqual([]);
  });

  it('returns sorted list of unconsumed fields', () => {
    const scene = unconsumedFieldsScene();
    const result = collectUnconsumedMaterialFields(
      scene.primitives as unknown as ReadonlyArray<PrimLike>,
    );
    // both fields are present, result is alphabetically sorted
    expect(result).toEqual(['frontLayer']);
  });

  it('reports material drops on analytic primitives', () => {
    const prims: ReadonlyArray<PrimLike> = [
      { kind: 'analytic', material: { baseColor: [1, 0, 0], unknownField: 42 } },
    ];
    expect(collectUnconsumedMaterialFields(prims)).toEqual(['unknownField']);
  });

  it('scans a material patch without needing a primitive wrapper', () => {
    const fields = collectUnconsumedMaterialFieldsForMaterial({
      baseColor: [1, 1, 1],
      displacementMap: { handle: 'height' },
      displacementScale: 0.25,
      displacementBias: -0.05,
    });
    expect(fields).toEqual([]);
    expect(categorizeUnconsumedMaterialFields(fields)).toEqual({});
  });

  it('reports every permanently unsupported walkaround material family from a material patch', () => {
    expect(collectUnconsumedMaterialFieldsForMaterial({
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      envMapIntensity: 0.25,
      ...WALKAROUND_PERMANENT_UNSUPPORTED_MATERIAL,
    })).toEqual(WALKAROUND_PERMANENT_UNSUPPORTED_FIELDS);
  });

  it('categorizes unconsumed fields for structured warning consumers', () => {
    expect(categorizeUnconsumedMaterialFields([
      'unknownFutureField',
      ...WALKAROUND_PERMANENT_UNSUPPORTED_FIELDS,
    ])).toEqual({
      spectral: ['dispersionAbbeNumber', 'spectralAttenuation'],
      volume: ['scatteringAnisotropy', 'scatteringCoefficient', 'scatteringCoefficientRGB'],
      layered: ['backLayer', 'frontLayer', 'thinFilmStack'],
      unknown: ['unknownFutureField'],
    });
  });

  it('unions across multiple primitives and deduplicates', () => {
    const prims: ReadonlyArray<PrimLike> = [
      { id: 'pane-a', kind: 'mesh', material: { baseColor: [1, 0, 0], frontLayer: { transmission: [1, 1, 1] } } },
      { id: 'pane-b', kind: 'mesh', material: { baseColor: [0, 1, 0], thinFilmStack: { layers: [] }, anisotropy: 0.5 } },
    ];
    expect(collectUnconsumedMaterialFields(prims)).toEqual(['frontLayer', 'thinFilmStack']);
    expect(collectUnconsumedMaterialPrimitiveFields(prims)).toEqual([
      { primitiveId: 'pane-a', fields: ['frontLayer'] },
      { primitiveId: 'pane-b', fields: ['thinFilmStack'] },
    ]);
  });

  it('surfaces representative layered and thin-film drops while iridescence is consumed', () => {
    const prims: ReadonlyArray<PrimLike> = [
      {
        kind: 'mesh',
        material: {
          baseColor: [1, 1, 1],
          roughness: 1,
          metallic: 0,
          alphaMode: 'mask',
          alphaCutoff: 0.35,
          opacity: 0.5,
          alphaMap: { handle: 'alpha' },
          normalScale: 0.5,
          specularColor: [0.8, 0.7, 0.6],
          specularIntensity: 0.4,
          specularColorMap: { handle: 'specColor' },
          specularIntensityMap: { handle: 'specIntensity' },
          clearcoatMap: { handle: 'clearcoat' },
          clearcoatRoughnessMap: { handle: 'clearcoatRoughness' },
          clearcoatNormalMap: { handle: 'clearcoatNormal' },
          clearcoatNormalScale: 0.5,
          sheenColorMap: { handle: 'sheenColor' },
          sheenRoughnessMap: { handle: 'sheenRoughness' },
          anisotropy: 0.5,
          anisotropyRotation: 0.25,
          anisotropyMap: { handle: 'anisotropy' },
          iridescence: 0.5,
          iridescenceIor: 2,
          iridescenceThicknessRange: [200, 800],
          iridescenceMap: { handle: 'iridescence' },
          iridescenceThicknessMap: { handle: 'iridescenceThickness' },
          frontLayer: { transmission: [1, 0.5, 0.25] },
          backLayer: { transmission: [0.25, 0.5, 1] },
          thinFilmStack: { layers: [{ ior: 1.4, thicknessNm: 300 }] },
        },
      },
    ];
    expect(collectUnconsumedMaterialFields(prims)).toEqual([
      'backLayer',
      'frontLayer',
      'thinFilmStack',
    ]);
  });

  it('reports only fractional blend primitives for the alpha approximation warning', () => {
    const prims: ReadonlyArray<PrimLike> = [
      { id: 'opaque', kind: 'mesh', material: { baseColor: [1, 1, 1], alphaMode: 'opaque', opacity: 0.5 } },
      { id: 'mask', kind: 'mesh', material: { baseColor: [1, 1, 1], alphaMode: 'mask', opacity: 0.25, alphaCutoff: 0.5 } },
      { id: 'transparent', kind: 'mesh', material: { baseColor: [1, 1, 1], alphaMode: 'blend', opacity: 0 } },
      { id: 'fractional', kind: 'mesh', material: { baseColor: [1, 1, 1], alphaMode: 'blend', opacity: 0.5 } },
      { id: 'base-alpha', kind: 'mesh', material: { baseColor: [1, 1, 1, 0.4], alphaMode: 'blend' } },
      {
        id: 'base-map-rgb',
        kind: 'mesh',
        material: {
          baseColor: [1, 1, 1],
          alphaMode: 'blend',
          baseColorMap: {
            handle: {
              width: 1,
              height: 1,
              data: new Uint8Array([255, 255, 255]),
              __vitrum_hint__: { channels: 3, dataType: 'uint8' },
            },
          },
        },
      },
      {
        id: 'base-map-alpha',
        kind: 'mesh',
        material: {
          baseColor: [1, 1, 1],
          alphaMode: 'blend',
          baseColorMap: {
            handle: {
              width: 1,
              height: 1,
              data: new Uint8Array([255, 255, 255, 128]),
              __vitrum_hint__: { channels: 4, dataType: 'uint8' },
            },
          },
        },
      },
      { id: 'alpha-map', kind: 'mesh', material: { baseColor: [1, 1, 1], alphaMode: 'blend', alphaMap: { handle: 'alpha' } } },
      { id: 'transparent-map', kind: 'mesh', material: { baseColor: [1, 1, 1], alphaMode: 'blend', opacity: 0, alphaMap: { handle: 'alpha' } } },
      { id: 'solid', kind: 'mesh', material: { baseColor: [1, 1, 1], alphaMode: 'blend', opacity: 1 } },
      {
        id: 'vertex-alpha',
        kind: 'mesh',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        colors: new Float32Array([1, 1, 1, 1, 1, 1, 1, 0.5, 1, 1, 1, 1]),
        material: { baseColor: [1, 1, 1], alphaMode: 'blend', opacity: 1 },
      },
      {
        id: 'vertex-rgb',
        kind: 'mesh',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        colors: new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
        material: { baseColor: [1, 1, 1], alphaMode: 'blend', opacity: 1 },
      },
    ];
    expect(collectApproximateAlphaBlendPrimitiveIds(prims)).toEqual([
      'alpha-map',
      'base-alpha',
      'base-map-alpha',
      'fractional',
      'vertex-alpha',
    ]);
  });

  it('reports emissive-map materials lit by scalar energy or mesh-area emitters for the texel-PDF warning', () => {
    const prims: ReadonlyArray<PrimLike> = [
      { id: 'non-emissive-map', kind: 'mesh', material: { emissive: [0, 0, 0], emissiveMap: { handle: 'map' } } },
      { id: 'zero-intensity', kind: 'mesh', material: { emissive: [1, 1, 1], emissiveIntensity: 0, emissiveMap: { handle: 'map' } } },
      { id: 'scalar-only', kind: 'mesh', material: { emissive: [1, 1, 1] } },
      { id: 'mapped-emitter', kind: 'mesh', material: { emissive: [0.2, 0.1, 0], emissiveIntensity: 3, emissiveMap: { handle: 'map' } } },
      { id: 'mesh-panel', kind: 'mesh', material: { emissive: [0, 0, 0], emissiveMap: { handle: 'map' } } },
      { id: 'dark-mesh-panel', kind: 'mesh', material: { emissive: [0, 0, 0], emissiveMap: { handle: 'map' } } },
      { id: 'point', kind: 'point', material: { emissive: [1, 1, 1], emissiveMap: { handle: 'map' } } },
    ];
    expect(collectApproximateEmissiveMapTexelPdfPrimitiveIds(prims, [
      { id: 'panel-light', kind: 'mesh-area', meshId: 'mesh-panel', color: [1, 1, 1], intensity: 4 },
      { id: 'dark-panel-light', kind: 'mesh-area', meshId: 'dark-mesh-panel', color: [1, 1, 1], intensity: 0 },
    ])).toEqual(['mapped-emitter', 'mesh-panel']);
  });

  it('reports only positive light-map materials for the camera-visible warning', () => {
    const prims: ReadonlyArray<PrimLike> = [
      { id: 'none', kind: 'mesh', material: { baseColor: [1, 1, 1] } },
      { id: 'zero', kind: 'mesh', material: { lightMap: { handle: 'lm' }, lightMapIntensity: 0 } },
      { id: 'mapped', kind: 'mesh', material: { lightMap: { handle: 'lm' } } },
      { id: 'scaled', kind: 'mesh', material: { lightMap: { handle: 'lm' }, lightMapIntensity: 0.25 } },
      { id: 'point', kind: 'point', material: { lightMap: { handle: 'lm' } } },
    ];

    expect(collectApproximateLightMapPrimitiveIds(prims)).toEqual(['mapped', 'scaled']);
  });

  it('documents exact direct-emitter texel support separately from residual all-path approximation', () => {
    expect(EMISSIVE_MAP_TEXEL_PDF_APPROXIMATION_DETAILS).toMatchObject({
      directEmitterPdf: 'exact-texel-cell-subtriangles-when-eligible',
      fallbackDirectEmitterPdf: 'uv-local-barycentric-micro-emitter-selection',
      giSuffixEmission: 'uv-local-emissive-texel-sampled-on-hit',
      probeHitEmission: 'uv-local-emissive-texel-sampled-on-direct-probe-hit',
      residualApproximation: 'global-texel-selection-pdf',
      missing: 'global-exact-texel-alias-pdf',
    });
    expect(EMISSIVE_MAP_TEXEL_PDF_APPROXIMATION_DETAILS.exactDirectEmitterConditions).toContain('cpu-readable-emissive-map');
    expect(EMISSIVE_MAP_TEXEL_PDF_APPROXIMATION_DETAILS.approximatePaths).toEqual([
      'ReSTIR-GI-texel-selection-pdf',
      'RC-non-direct-texel-selection-pdf',
      'DDGI-non-direct-texel-selection-pdf',
      'fallback-direct-emitter',
    ]);
  });

  it('ignores null/undefined field values', () => {
    const prims: ReadonlyArray<PrimLike> = [
      { kind: 'mesh', material: { baseColor: [1, 0, 0], baseColorMap: null, normalMap: undefined } },
    ];
    expect(collectUnconsumedMaterialFields(prims)).toEqual([]);
  });

  it('returns empty array for an empty primitives list', () => {
    expect(collectUnconsumedMaterialFields([])).toEqual([]);
  });
});

describe('collectApproximateRichMaterialPrimitiveFields', () => {
  it('reports active consumed rich-material fields without flagging default scalar metadata', () => {
    const scene = consumedOnlyScene();
    const base = scene.primitives[0]!;
    const richScene: Scene = {
      ...scene,
      primitives: [
        {
          ...base,
          id: 'rich-defaults',
          material: {
            ...base.material,
            specularColor: [1, 1, 1],
            specularIntensity: 1,
            clearcoat: 0,
            clearcoatRoughness: 0,
            sheen: 0,
            sheenColor: [0, 0, 0],
            iridescence: 0,
            iridescenceIor: 1.3,
            iridescenceThicknessRange: [100, 400],
          },
        },
        {
          ...base,
          id: 'active-rich',
          material: {
            ...base.material,
            specularColor: [0.8, 0.7, 0.6],
            specularIntensity: 0.4,
            clearcoat: 0.5,
            clearcoatRoughness: 0.2,
            clearcoatNormalMap: { handle: 'cc-normal' },
            clearcoatNormalScale: 0.5,
            sheen: 0.3,
            sheenColorMap: { handle: 'sheen' },
            anisotropy: 0.4,
            anisotropyRotation: 0.25,
            iridescence: 0.6,
            iridescenceThicknessMap: { handle: 'thin-film' },
          },
        },
      ],
    };

    expect(collectApproximateRichMaterialPrimitiveFields(
      richScene.primitives as unknown as ReadonlyArray<PrimLike>,
    )).toEqual([{
      primitiveId: 'active-rich',
      fields: [
        'anisotropy',
        'anisotropyRotation',
        'clearcoat',
        'clearcoatNormalMap',
        'clearcoatNormalScale',
        'clearcoatRoughness',
        'iridescence',
        'iridescenceThicknessMap',
        'sheen',
        'sheenColorMap',
        'specularColor',
        'specularIntensity',
      ],
    }]);
  });
});

// ── Layer 2 — engine-level setScene wiring ───────────────────────────────────

describe('HybridEngine.setScene unconsumed-field warning', () => {
  let warnSpy: MockInstance;
  let errorSpy: MockInstance;

  beforeEach(() => {
    warnSpy  = vi.spyOn(console, 'warn').mockImplementation(() => {});
    // Suppress the expected GPU-init error noise (GPUBufferUsage undefined in vitest).
    errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    warnSpy.mockRestore();
    errorSpy.mockRestore();
  });

  it('does NOT warn when the scene uses only consumed fields', () => {
    const engine = new HybridEngine(makeOpts());
    try {
      engine.setScene(consumedOnlyScene());
      const warnMessages = warnSpy.mock.calls.flat().map(String);
      const materialWarns = warnMessages.filter((m) => m.includes('not consumed'));
      expect(materialWarns).toHaveLength(0);
    } finally {
      engine.dispose();
    }
  });

  it('warns only for frontLayer when baseColorMap is also supplied', () => {
    const structured: EngineWarning[] = [];
    const engine = new HybridEngine({
      ...makeOpts(),
      onWarning: (w) => structured.push(w),
    });
    try {
      engine.setScene(unconsumedFieldsScene());
      const warnMessages = warnSpy.mock.calls.flat().map(String);
      const materialWarn = warnMessages.find((m) => m.includes('not consumed'));
      expect(materialWarn).toBeDefined();
      expect(materialWarn).not.toContain('baseColorMap');
      expect(materialWarn).toContain('frontLayer');
      expect(structured.some((w) =>
        w.code === 'walkaround-hybrid.unconsumed-material-fields' &&
        Array.isArray(w.details?.fields) &&
        !w.details.fields.includes('baseColorMap') &&
        w.details.fields.includes('frontLayer'),
      )).toBe(true);
    } finally {
      engine.dispose();
    }
  });

  it('emits structured warnings for every permanently unsupported material field', () => {
    const structured: EngineWarning[] = [];
    const engine = new HybridEngine({
      ...makeOpts(),
      onWarning: (w) => structured.push(w),
    });
    try {
      const baseScene = consumedOnlyScene();
      const basePrim = baseScene.primitives[0]!;
      const scene: Scene = {
        ...baseScene,
        primitives: [
          {
            ...basePrim,
            id: 'unsupported-material-fields',
            material: {
              ...basePrim.material,
              ...WALKAROUND_PERMANENT_UNSUPPORTED_MATERIAL,
              envMapIntensity: 0.35,
            },
          } as unknown as ScenePrimitive,
        ],
      };

      engine.setScene(scene);
      const materialWarn = warnSpy.mock.calls.flat().map(String).find((m) => m.includes('not consumed'));
      expect(materialWarn).toBeDefined();
      for (const field of WALKAROUND_PERMANENT_UNSUPPORTED_FIELDS) {
        expect(materialWarn).toContain(field);
      }
      expect(materialWarn).not.toContain('envMapIntensity');
      expect(structured.some((w) =>
        w.code === 'walkaround-hybrid.unconsumed-material-fields' &&
        JSON.stringify(w.details?.fields) === JSON.stringify(WALKAROUND_PERMANENT_UNSUPPORTED_FIELDS) &&
        JSON.stringify(w.details?.categories) === JSON.stringify({
          spectral: ['dispersionAbbeNumber', 'spectralAttenuation'],
          volume: ['scatteringAnisotropy', 'scatteringCoefficient', 'scatteringCoefficientRGB'],
          layered: ['backLayer', 'frontLayer', 'thinFilmStack'],
        }) &&
        JSON.stringify(w.details?.primitiveFields) === JSON.stringify([{
          primitiveId: 'unsupported-material-fields',
          fields: WALKAROUND_PERMANENT_UNSUPPORTED_FIELDS,
        }]),
      )).toBe(true);
    } finally {
      engine.dispose();
    }
  });

  it('emits a structured setScene warning for reserved receiveShadow:false', () => {
    const structured: EngineWarning[] = [];
    const engine = new HybridEngine({
      ...makeOpts(),
      onWarning: (w) => structured.push(w),
    });
    try {
      const baseScene = consumedOnlyScene();
      const basePrim = baseScene.primitives[0]!;
      const scene: Scene = {
        ...baseScene,
        primitives: [
          {
            ...basePrim,
            receiveShadow: false,
          } as unknown as ScenePrimitive,
        ],
      };

      engine.setScene(scene);

      const warning = structured.find((w) =>
        w.code === 'walkaround-hybrid.reserved-receive-shadow',
      );
      expect(warning).toMatchObject({
        backend: 'walkaround-hybrid',
        phase: 'setScene',
        method: 'setScene',
        details: { primitiveIds: ['m1'] },
      });
      expect(warnSpy.mock.calls.flat().map(String).some((m) =>
        m.includes('receiveShadow:false') && m.includes('m1'),
      )).toBe(true);
    } finally {
      engine.dispose();
    }
  });

  it('warns only once per distinct field set across repeated setScene calls', () => {
    const engine = new HybridEngine(makeOpts());
    try {
      engine.setScene(unconsumedFieldsScene());
      engine.setScene(unconsumedFieldsScene()); // same field set — should NOT warn again
      const materialWarns = warnSpy.mock.calls.flat().map(String).filter((m) => m.includes('not consumed'));
      expect(materialWarns).toHaveLength(1);
    } finally {
      engine.dispose();
    }
  });

  it('emits a structured warning for residual alpha blend approximation', () => {
    const structured: EngineWarning[] = [];
    const engine = new HybridEngine({
      ...makeOpts(),
      onWarning: (w) => structured.push(w),
    });
    try {
      const fractionalBlendScene: Scene = {
        ...consumedOnlyScene(),
        primitives: [
          {
            ...consumedOnlyScene().primitives[0]!,
            id: 'blend-pane',
            material: {
              ...consumedOnlyScene().primitives[0]!.material,
              alphaMode: 'blend',
              baseColorMap: { handle: { width: 1, height: 1, data: new Uint8Array([255, 255, 255, 128]) } },
            },
          },
        ],
      };
      engine.setScene(fractionalBlendScene);
      engine.setScene(fractionalBlendScene);
      const alphaWarnings = structured.filter((w) =>
        w.code === 'walkaround-hybrid.alpha-blend-approximation',
      );
      expect(alphaWarnings).toHaveLength(1);
      expect(alphaWarnings[0]?.details?.primitiveIds).toContain('blend-pane');
    } finally {
      engine.dispose();
    }
  });

  it('does not emit alpha approximation warning for RGB-only baseColorMap blend coverage', () => {
    const structured: EngineWarning[] = [];
    const engine = new HybridEngine({
      ...makeOpts(),
      onWarning: (w) => structured.push(w),
    });
    try {
      const opaqueTextureBlendScene: Scene = {
        ...consumedOnlyScene(),
        primitives: [
          {
            ...consumedOnlyScene().primitives[0]!,
            id: 'rgb-map-pane',
            material: {
              ...consumedOnlyScene().primitives[0]!.material,
              alphaMode: 'blend',
              opacity: 1,
              baseColorMap: {
                handle: {
                  width: 1,
                  height: 1,
                  data: new Uint8Array([255, 255, 255]),
                  __vitrum_hint__: { channels: 3, dataType: 'uint8' },
                },
              },
            },
          },
        ],
      };
      engine.setScene(opaqueTextureBlendScene);
      expect(structured.filter((w) =>
        w.code === 'walkaround-hybrid.alpha-blend-approximation'
      )).toHaveLength(0);
    } finally {
      engine.dispose();
    }
  });

  it('treats uint16 baseColorMap alpha as normalized coverage for blend diagnostics', () => {
    const structured: EngineWarning[] = [];
    const engine = new HybridEngine({
      ...makeOpts(),
      onWarning: (w) => structured.push(w),
    });
    try {
      const uint16AlphaBlendScene: Scene = {
        ...consumedOnlyScene(),
        primitives: [
          {
            ...consumedOnlyScene().primitives[0]!,
            id: 'uint16-alpha-pane',
            material: {
              ...consumedOnlyScene().primitives[0]!.material,
              alphaMode: 'blend',
              opacity: 1,
              baseColorMap: {
                handle: {
                  width: 1,
                  height: 1,
                  data: new Uint16Array([65535, 65535, 65535, 0x3c00]),
                  __vitrum_hint__: { channels: 4, dataType: 'uint16' },
                },
              },
            },
          },
        ],
      };
      engine.setScene(uint16AlphaBlendScene);
      const alphaWarnings = structured.filter((w) =>
        w.code === 'walkaround-hybrid.alpha-blend-approximation'
      );
      expect(alphaWarnings).toHaveLength(1);
      expect(alphaWarnings[0]?.details?.primitiveIds).toContain('uint16-alpha-pane');
    } finally {
      engine.dispose();
    }
  });

  it('emits a structured alpha approximation warning for vertex-color alpha blend', () => {
    const structured: EngineWarning[] = [];
    const engine = new HybridEngine({
      ...makeOpts(),
      onWarning: (w) => structured.push(w),
    });
    try {
      const vertexAlphaScene: Scene = {
        ...consumedOnlyScene(),
        primitives: [
          {
            ...consumedOnlyScene().primitives[0]!,
            id: 'vertex-alpha-pane',
            colors: new Float32Array([
              1, 1, 1, 1,
              1, 1, 1, 0.5,
              1, 1, 1, 1,
            ]),
            material: {
              ...consumedOnlyScene().primitives[0]!.material,
              alphaMode: 'blend',
            },
          } as unknown as ScenePrimitive,
        ],
      };
      engine.setScene(vertexAlphaScene);
      const alphaWarnings = structured.filter((w) =>
        w.code === 'walkaround-hybrid.alpha-blend-approximation',
      );
      expect(alphaWarnings).toHaveLength(1);
      expect(alphaWarnings[0]?.details?.primitiveIds).toContain('vertex-alpha-pane');
    } finally {
      engine.dispose();
    }
  });

  it('emits a structured warning for emissive-map texel-PDF approximation', () => {
    const structured: EngineWarning[] = [];
    const engine = new HybridEngine({
      ...makeOpts(),
      onWarning: (w) => structured.push(w),
    });
    try {
      const emissiveScene: Scene = {
        ...consumedOnlyScene(),
        primitives: [
          {
            ...consumedOnlyScene().primitives[0]!,
            id: 'mapped-glow',
            material: {
              ...consumedOnlyScene().primitives[0]!.material,
              emissive: [1, 0.5, 0.25],
              emissiveIntensity: 2,
              emissiveMap: { handle: { width: 1, height: 1, data: new Uint8Array([255, 128, 64, 255]) } },
            },
          },
        ],
      };
      engine.setScene(emissiveScene);
      engine.setScene(emissiveScene);
      const texelPdfWarnings = structured.filter((w) =>
        w.code === 'walkaround-hybrid.emissive-map-texel-pdf-approximation',
      );
      expect(texelPdfWarnings).toHaveLength(1);
      expect(texelPdfWarnings[0]?.details?.primitiveIds).toContain('mapped-glow');
      expect(texelPdfWarnings[0]?.details).toMatchObject({
        directEmitterPdf: 'exact-texel-cell-subtriangles-when-eligible',
        fallbackDirectEmitterPdf: 'uv-local-barycentric-micro-emitter-selection',
        giSuffixEmission: 'uv-local-emissive-texel-sampled-on-hit',
        probeHitEmission: 'uv-local-emissive-texel-sampled-on-direct-probe-hit',
        residualApproximation: 'global-texel-selection-pdf',
        missing: 'global-exact-texel-alias-pdf',
      });
      expect(texelPdfWarnings[0]?.message).toContain('exact texel-cell sub-triangles');
    } finally {
      engine.dispose();
    }
  });

  it('emits a structured warning for rich-material GI approximation', () => {
    const structured: EngineWarning[] = [];
    const engine = new HybridEngine({
      ...makeOpts(),
      onWarning: (w) => structured.push(w),
    });
    try {
      const scene = consumedOnlyScene();
      const prim = scene.primitives[0]!;
      const richScene: Scene = {
        ...scene,
        primitives: [
          {
            ...prim,
            id: 'rich-panel',
            material: {
              ...prim.material,
              specularColor: [0.8, 0.7, 0.6],
              clearcoat: 0.5,
              clearcoatNormalMap: { handle: 'cc-normal' },
              sheen: 0.3,
              anisotropy: 0.4,
              iridescenceMap: { handle: 'iridescence' },
            },
          },
        ],
      };
      engine.setScene(richScene);
      engine.setScene(richScene);
      const richWarnings = structured.filter((w) =>
        w.code === 'walkaround-hybrid.rich-material-gi-approximation',
      );
      expect(richWarnings).toHaveLength(1);
      expect(richWarnings[0]?.details?.primitiveFields).toEqual([{
        primitiveId: 'rich-panel',
        fields: ['anisotropy', 'clearcoat', 'clearcoatNormalMap', 'iridescenceMap', 'sheen', 'specularColor'],
      }]);
      expect(richWarnings[0]?.details).toMatchObject(RICH_MATERIAL_GI_APPROXIMATION_DETAILS);
    } finally {
      engine.dispose();
    }
  });

  it('emits a structured warning when setScene drops an unreadable atlas-backed material map', async () => {
    const structured: EngineWarning[] = [];
    const engine = new HybridEngine({
      ...makeOpts(),
      onWarning: (w) => structured.push(w),
    });
    try {
      const scene = consumedOnlyScene();
      const prim = scene.primitives[0]!;
      engine.setScene({
        ...scene,
        primitives: [
          {
            ...prim,
            id: 'gpu-only-map',
            material: {
              ...prim.material,
              baseColorMap: { handle: { id: 'gpu-only-texture' } },
            },
          } as unknown as ScenePrimitive,
        ],
      });

      await vi.waitFor(() => {
        const warning = structured.find((w) =>
          w.code === 'walkaround-hybrid.unreadable-material-texture-map',
        );
        expect(warning?.method).toBe('setScene');
        expect(warning?.details).toMatchObject({
          materialIndex: 0,
          field: 'baseColorMap',
          colorSpace: 'srgb',
          fallback: 'map ignored',
        });
      });
    } finally {
      engine.dispose();
    }
  });

  it('emits structured warnings for unknown updateLighting keys', () => {
    const structured: EngineWarning[] = [];
    const engine = new HybridEngine({
      ...makeOpts(),
      onWarning: (w) => structured.push(w),
    });
    try {
      engine.updateLighting({ typoIntensity: 2 } as never);
      expect(warnSpy.mock.calls.flat().map(String).some((m) => m.includes('typoIntensity'))).toBe(true);
      expect(structured.some((w) =>
        w.code === 'walkaround-hybrid.unknown-lighting-key' &&
        w.details?.key === 'typoIntensity',
      )).toBe(true);
    } finally {
      engine.dispose();
    }
  });

  it('does not warn that directional angularDiameter is missing DDGI/RC probe support', () => {
    const structured: EngineWarning[] = [];
    const engine = new HybridEngine({
      ...makeOpts(),
      onWarning: (w) => structured.push(w),
    });
    try {
      const scene: Scene = {
        ...consumedOnlyScene(),
        emitters: [{
          id: 'soft-sun',
          kind: 'directional',
          direction: [0, 1, 0],
          color: [1, 1, 1],
          intensity: 2,
          angularDiameter: 0.08,
        }],
      };
      engine.setScene(scene);
      engine.setScene(scene);
      const warnings = structured.filter((w) =>
        w.code === 'walkaround-hybrid.directional-angular-diameter-partial-support',
      );
      expect(warnings).toHaveLength(0);
    } finally {
      engine.dispose();
    }
  });

  it('emits a structured warning for camera-visible-only light maps', () => {
    const structured: EngineWarning[] = [];
    const engine = new HybridEngine({
      ...makeOpts(),
      onWarning: (w) => structured.push(w),
    });
    try {
      const lightMappedScene: Scene = {
        ...consumedOnlyScene(),
        primitives: [
          {
            ...consumedOnlyScene().primitives[0]!,
            id: 'baked-panel',
            material: {
              ...consumedOnlyScene().primitives[0]!.material,
              lightMap: { handle: { width: 1, height: 1, data: new Uint8Array([32, 64, 96, 255]) } },
              lightMapIntensity: 0.75,
            },
          },
        ],
      };

      engine.setScene(lightMappedScene);
      engine.setScene(lightMappedScene);

      const lightMapWarnings = structured.filter((w) =>
        w.code === 'walkaround-hybrid.light-map-camera-visible-approximation',
      );
      expect(lightMapWarnings).toHaveLength(1);
      expect(lightMapWarnings[0]).toMatchObject({
        backend: 'walkaround-hybrid',
        phase: 'setScene',
        method: 'setScene',
        details: {
          primitiveIds: ['baked-panel'],
          ...LIGHT_MAP_CAMERA_VISIBLE_APPROXIMATION_DETAILS,
        },
      });
      expect(warnSpy.mock.calls.flat().map(String).some((m) =>
        m.includes('camera-visible baked') && m.includes('baked-panel'),
      )).toBe(true);
    } finally {
      engine.dispose();
    }
  });
});
