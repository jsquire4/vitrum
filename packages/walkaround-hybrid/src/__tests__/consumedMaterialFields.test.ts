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
import type { EngineWarning, MaterialSpec, Scene, ScenePrimitive } from '@vitrum/core';
import { BACKEND_PROMISE_LEDGER, MATERIAL_SPEC_FIELDS } from '@vitrum/core';
import {
  categorizeUnconsumedMaterialFields,
  CONSUMED_MATERIAL_FIELDS,
  collectUnsupportedVolumeLayerPrimitiveFields,
  assertNoRcDirectSunTransmissionProfiles,
  assertNoUnsupportedLayeredTransmissionProfiles,
  assertNoUnsupportedRoughTransmissionProfiles,
  collectUnconsumedMaterialFields,
  collectUnconsumedMaterialFieldsForMaterial,
  collectUnconsumedMaterialPrimitiveFields,
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

const WALKAROUND_APPROXIMATE_OPTICAL_MATERIAL: Record<string, unknown> = {
  spectralAttenuation: {
    wavelengthStart: 380,
    wavelengthEnd: 700,
    values: new Float32Array([0.1, 0.2, 0.3]),
  },
  dispersionAbbeNumber: 42,
  thinFilmStack: { layers: [{ ior: 1.4, thicknessNm: 300 }] },
};

const WALKAROUND_APPROXIMATE_OPTICAL_FIELDS = [
  'dispersionAbbeNumber',
  'spectralAttenuation',
  'thinFilmStack',
] as const satisfies readonly (keyof MaterialSpec)[];

/** A scene with only consumed fields in its material. */
function consumedOnlyScene(): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'm1',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        uvs: new Float32Array([0, 0, 1, 0, 0, 1]),
        material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0 },
      } as unknown as ScenePrimitive,
    ],
    emitters: [],
    environment: { kind: 'none' },
  };
}

/** A scene whose material has `baseColorMap` plus a consumed layer-local normalMap. */
function layerNormalMapScene(): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'm2',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        uvs: new Float32Array([0, 0, 1, 0, 0, 1]),
        material: {
          baseColor: [1, 1, 1],
          roughness: 0.3,
          metallic: 0,
          baseColorMap: { handle: { width: 1, height: 1, data: new Uint8Array([255, 255, 255, 255]) } },
          frontLayer: {
            transmission: [1, 0.5, 0.25],
            normalMap: {
              handle: {
                width: 1,
                height: 1,
                data: new Uint8Array([128, 128, 255, 255]),
              },
            },
          },
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
      'bumpMap', 'bumpScale', 'displacementMap', 'displacementScale', 'displacementBias', 'displacementSubdivisions',
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

  it('contains every implemented approximate optical field', () => {
    for (const f of WALKAROUND_APPROXIMATE_OPTICAL_FIELDS) {
      expect(CONSUMED_MATERIAL_FIELDS.has(f)).toBe(true);
      expect(BACKEND_PROMISE_LEDGER['walkaround-hybrid'].supportDetails.materials?.[f])
        .toBe('approximate');
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

  it('consumes the implemented face-layer profile when it carries a normal map', () => {
    const scene = layerNormalMapScene();
    expect(collectUnconsumedMaterialFields(
      scene.primitives as unknown as ReadonlyArray<PrimLike>,
    )).toEqual([]);
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

  it('does not report implemented approximate optical fields from a material patch', () => {
    expect(collectUnconsumedMaterialFieldsForMaterial({
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      envMapIntensity: 0.25,
      ...WALKAROUND_APPROXIMATE_OPTICAL_MATERIAL,
    })).toEqual([]);
  });

  it('reports implemented walkaround volume scattering fields as consumed', () => {
    expect(collectUnconsumedMaterialFieldsForMaterial({
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      scatteringCoefficient: 0.15,
      scatteringAnisotropy: 0.25,
      scatteringCoefficientRGB: [0.1, 0.2, 0.3],
    })).toEqual([]);
  });

  it('does not collect implemented volume scattering and face-layer transport fields', () => {
    const prims: ReadonlyArray<PrimLike> = [
      {
        id: 'volume-layer-pane',
        kind: 'mesh',
        material: {
          baseColor: [1, 1, 1],
          roughness: 1,
          metallic: 0,
          scatteringCoefficient: 0.15,
          scatteringAnisotropy: 0.25,
          scatteringCoefficientRGB: [0.1, 0, 0.3],
          frontLayer: { transmission: [1, 0.5, 0.25] },
          backLayer: { normalMap: { handle: 'back-normal' } },
        },
      },
      {
        id: 'default-volume-fields',
        kind: 'mesh',
        material: {
          baseColor: [1, 1, 1],
          roughness: 1,
          metallic: 0,
          scatteringCoefficient: 0,
          scatteringAnisotropy: 0,
          scatteringCoefficientRGB: [0, 0, 0],
          frontLayer: {},
        },
      },
    ];

    expect(collectUnsupportedVolumeLayerPrimitiveFields(prims)).toEqual([]);
  });

  it('accepts rough, mapped, and anisotropic transmissive profiles', () => {
    const primitive = (material: Record<string, unknown>): PrimLike[] => [{
      id: 'conditional-glass',
      kind: 'mesh',
      material: { baseColor: [1, 1, 1], metallic: 0, transmission: 1, ...material },
    }];

    expect(() => assertNoUnsupportedRoughTransmissionProfiles(
      primitive({ roughness: 0.001 }), 'setScene',
    )).not.toThrow();
    expect(() => assertNoUnsupportedRoughTransmissionProfiles(
      primitive({ roughness: 0, roughnessMap: { handle: 'roughness' } }), 'setScene',
    )).not.toThrow();
    expect(() => assertNoUnsupportedRoughTransmissionProfiles(
      primitive({ roughness: 0, anisotropy: 0.25 }), 'updatePrimitive',
    )).not.toThrow();
    expect(() => assertNoUnsupportedRoughTransmissionProfiles(
      primitive({ roughness: 0, anisotropyMap: { handle: 'anisotropy' } }), 'setScene',
    )).not.toThrow();
    expect(() => assertNoUnsupportedRoughTransmissionProfiles(
      primitive({
        roughness: 0,
        normalMap: { handle: 'normal' },
        normalScale: 0.8,
        bumpMap: { handle: 'bump' },
        bumpScale: 0.2,
      }),
      'setScene',
    )).not.toThrow();
  });

  it('accepts layered opaque lobes combined with transmission', () => {
    const primitive = (material: Record<string, unknown>): PrimLike[] => [{
      id: 'layered-glass',
      kind: 'mesh',
      material: {
        baseColor: [1, 1, 1], roughness: 0, metallic: 0, transmission: 1,
        ...material,
      },
    }];
    for (const material of [
      { metallic: 0.2 },
      { clearcoat: 0.5 },
      { sheenColorMap: { handle: 'sheen' } },
      { specularIntensity: 0.5 },
      { iridescenceMap: { handle: 'film' } },
    ]) {
      expect(() => assertNoUnsupportedLayeredTransmissionProfiles(
        primitive(material), 'setScene',
      )).not.toThrow();
    }
    expect(() => assertNoUnsupportedLayeredTransmissionProfiles(
      primitive({ normalMap: { handle: 'normal' } }), 'setScene',
    )).not.toThrow();
  });

  it('accepts RC plus authored transmission through the dielectric transport path', () => {
    expect(() => assertNoRcDirectSunTransmissionProfiles([{
      id: 'rc-glass', kind: 'mesh',
      material: { baseColor: [1, 1, 1], roughness: 0, metallic: 0, transmission: 1 },
    }], 'setScene')).not.toThrow();
    expect(() => assertNoRcDirectSunTransmissionProfiles([{
      id: 'opaque', kind: 'mesh',
      material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 },
    }], 'setScene')).not.toThrow();
  });

  it('categorizes unconsumed fields for structured warning consumers', () => {
    expect(categorizeUnconsumedMaterialFields(['unknownFutureField'])).toEqual({
      unknown: ['unknownFutureField'],
    });
  });

  it('unions across multiple primitives and deduplicates', () => {
    const prims: ReadonlyArray<PrimLike> = [
      { id: 'pane-a', kind: 'mesh', material: { baseColor: [1, 0, 0], frontLayer: { transmission: [1, 1, 1], normalMap: { handle: 'normal' } } } },
      { id: 'pane-b', kind: 'mesh', material: { baseColor: [0, 1, 0], thinFilmStack: { layers: [] }, anisotropy: 0.5 } },
    ];
    expect(collectUnconsumedMaterialFields(prims)).toEqual([]);
    expect(collectUnconsumedMaterialPrimitiveFields(prims)).toEqual([]);
  });

  it('consumes representative rich, thin-film, and iridescence controls', () => {
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
          doubleSided: true,
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
          thinFilmStack: { layers: [{ ior: 1.4, thicknessNm: 300 }] },
        },
      },
    ];
    expect(collectUnconsumedMaterialFields(prims)).toEqual([]);
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

  it('publishes implemented layer-local normal maps without approximation warnings', () => {
    const structured: EngineWarning[] = [];
    const engine = new HybridEngine({
      ...makeOpts(),
      onWarning: (w) => structured.push(w),
    });
    try {
      const scene = layerNormalMapScene();
      expect(() => engine.setScene(scene)).not.toThrow();
      expect(engine.getScene()).toEqual(scene);
      expect(structured).toEqual([]);
    } finally {
      engine.dispose();
    }
  });

  it('does not emit unconsumed warnings for implemented optical material fields', () => {
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
              ...WALKAROUND_APPROXIMATE_OPTICAL_MATERIAL,
              envMapIntensity: 0.35,
            },
          },
        ],
      };

      engine.setScene(scene);
      const materialWarn = warnSpy.mock.calls.flat().map(String).find((m) => m.includes('not consumed'));
      expect(materialWarn).toBeUndefined();
      expect(structured.some((w) =>
        w.code === 'walkaround-hybrid.unconsumed-material-fields'
      )).toBe(false);
    } finally {
      engine.dispose();
    }
  });

  it('rejects unknown material keys before scene publication', () => {
    const engine = new HybridEngine(makeOpts());
    const baseScene = consumedOnlyScene();
    const basePrim = baseScene.primitives[0]!;
    const unsupportedScene: Scene = {
      ...baseScene,
      primitives: [
        {
          ...basePrim,
          material: {
            ...basePrim.material,
            unknownFutureField: { mode: 'future' },
          },
        } as unknown as ScenePrimitive,
      ],
    };
    try {
      expect(() => engine.setScene(unsupportedScene)).toThrow(/unknownFutureField.*known contract field/);
      expect(engine.getScene()).toBeNull();
      expect(warnSpy).not.toHaveBeenCalled();
    } finally {
      engine.dispose();
    }
  });

  it('does not warn for ordered/stochastic alpha blend transport', () => {
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
      expect(alphaWarnings).toHaveLength(0);
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

  it('does not warn for uint16 baseColorMap blend coverage', () => {
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
      expect(alphaWarnings).toHaveLength(0);
    } finally {
      engine.dispose();
    }
  });

  it('does not warn for vertex-color alpha blend', () => {
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
      expect(alphaWarnings).toHaveLength(0);
    } finally {
      engine.dispose();
    }
  });

  it('uses an exact whole-triangle density for a readable one-texel emissive map', () => {
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
      expect(structured).toHaveLength(0);
    } finally {
      engine.dispose();
    }
  });

  it('does not warn for implemented rich-material lobes', () => {
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
      expect(richWarnings).toHaveLength(0);
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
          },
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

  it('rejects unknown updateLighting keys before lighting mutation', () => {
    const structured: EngineWarning[] = [];
    const engine = new HybridEngine({
      ...makeOpts(),
      onWarning: (w) => structured.push(w),
    });
    try {
      expect(() => engine.updateLighting({ typoIntensity: 2 } as never))
        .toThrow(/unknown key "typoIntensity"/);
      expect(warnSpy).not.toHaveBeenCalled();
      expect(structured).toHaveLength(0);
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

  it('publishes implemented volume/layer profiles without approximation warnings', () => {
    const structured: EngineWarning[] = [];
    const engine = new HybridEngine({
      ...makeOpts(),
      onWarning: (w) => structured.push(w),
    });
    try {
      const scene = consumedOnlyScene();
      const prim = scene.primitives[0]!;
      const volumeLayerScene: Scene = {
        ...scene,
        primitives: [
          {
            ...prim,
            id: 'volume-layer-pane',
            material: {
              ...prim.material,
              scatteringCoefficient: 0.1,
              scatteringAnisotropy: -0.2,
              frontLayer: { transmission: [0.8, 0.7, 0.6] },
              backLayer: {
                transmission: [1, 1, 1],
                normalMap: {
                  handle: {
                    width: 1,
                    height: 1,
                    data: new Uint8Array([128, 128, 255, 255]),
                  },
                },
              },
            },
          } as unknown as ScenePrimitive,
        ],
      };

      expect(() => engine.setScene(volumeLayerScene)).not.toThrow();
      expect(engine.getScene()).toEqual(volumeLayerScene);
      expect(structured.some((w) =>
        w.code === 'walkaround-hybrid.volume-layer-transport-approximation',
      )).toBe(false);
    } finally {
      engine.dispose();
    }
  });

  it('does not warn for GI-propagated light maps', () => {
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
      expect(lightMapWarnings).toHaveLength(0);
      expect(warnSpy.mock.calls.flat().map(String).some((m) =>
        m.includes('camera-visible baked') && m.includes('baked-panel'),
      )).toBe(false);
    } finally {
      engine.dispose();
    }
  });
});
