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
  CONSUMED_MATERIAL_FIELDS,
  collectUnconsumedMaterialFields,
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
type PrimLike = { readonly kind: string; readonly material?: Record<string, unknown> };

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

/** A scene whose material has `baseColorMap` + `clearcoat` (unconsumed). */
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
          baseColorMap: { uri: 'tex.png' },   // unconsumed
          clearcoat: 0.8,                      // unconsumed
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
      'shadingModel', 'transmission', 'attenuationColor', 'attenuationDistance',
      'thickness', 'ior', 'extensions',
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

  it('does NOT contain texture-map fields', () => {
    for (const f of [
      'baseColorMap',
      'normalMap',
      'normalScale',
      'roughnessMap',
      'metallicMap',
      'thicknessMap',
      'alphaMap',
      'aoMap',
      'displacementMap',
      'displacementScale',
      'displacementBias',
      'specularColor',
      'specularIntensity',
      'specularColorMap',
      'specularIntensityMap',
      'frontLayer',
      'backLayer',
      'thinFilmStack',
      'anisotropy',
      'anisotropyMap',
    ]) {
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
    expect(result).toEqual(['baseColorMap', 'clearcoat']);
  });

  it('skips non-mesh kinds (analytic kind is not scanned)', () => {
    const prims: ReadonlyArray<PrimLike> = [
      { kind: 'analytic', material: { baseColor: [1, 0, 0], unknownField: 42 } },
    ];
    // analytic is NOT in the scanned kinds — should be ignored
    expect(collectUnconsumedMaterialFields(prims)).toEqual([]);
  });

  it('unions across multiple primitives and deduplicates', () => {
    const prims: ReadonlyArray<PrimLike> = [
      { kind: 'mesh', material: { baseColor: [1, 0, 0], sheenColor: [0.5, 0.5, 0.5] } },
      { kind: 'mesh', material: { baseColor: [0, 1, 0], sheenColor: [0.2, 0.2, 0.2], anisotropy: 0.5 } },
    ];
    expect(collectUnconsumedMaterialFields(prims)).toEqual(['anisotropy', 'sheenColor']);
  });

  it('surfaces representative alpha, specular, layered, thin-film, and anisotropy drops', () => {
    const prims: ReadonlyArray<PrimLike> = [
      {
        kind: 'mesh',
        material: {
          baseColor: [1, 1, 1],
          roughness: 1,
          metallic: 0,
          alphaMap: { handle: 'alpha' },
          normalScale: 0.5,
          specularColor: [0.8, 0.7, 0.6],
          specularIntensity: 0.4,
          frontLayer: { transmission: [1, 0.5, 0.25] },
          backLayer: { transmission: [0.25, 0.5, 1] },
          thinFilmStack: { layers: [{ ior: 1.4, thicknessNm: 300 }] },
          anisotropyMap: { handle: 'aniso' },
        },
      },
    ];
    expect(collectUnconsumedMaterialFields(prims)).toEqual([
      'alphaMap',
      'anisotropyMap',
      'backLayer',
      'frontLayer',
      'normalScale',
      'specularColor',
      'specularIntensity',
      'thinFilmStack',
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

  it('warns naming both baseColorMap and clearcoat when both are supplied', () => {
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
      expect(materialWarn).toContain('baseColorMap');
      expect(materialWarn).toContain('clearcoat');
      expect(structured.some((w) =>
        w.code === 'walkaround-hybrid.unconsumed-material-fields' &&
        Array.isArray(w.details?.fields) &&
        w.details.fields.includes('baseColorMap') &&
        w.details.fields.includes('clearcoat'),
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
});
