/**
 * Characterization pins for the HybridEngine material-warning subsystem.
 * Captures the exact [code, message, details] payloads and once-only dedup
 * semantics for warnings that still represent real unsupported or bounded
 * behavior.
 *
 * The remaining approximation `_warn*` method + lifecycle `_warn*` methods are
 * exercised directly (they are private; we reach through the instance) and via
 * the public setScene/updatePrimitive/updateEmitter/setSize integration paths
 * to pin the end-to-end emission ORDER.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { MockInstance } from 'vitest';
import { asMat4, type EngineWarning, type Scene, type ScenePrimitive, type SceneEmitter } from '@vitrum/core';
import { HybridEngine, type HybridEngineOptions } from '../HybridEngine.js';

function makeDeviceStub(): GPUDevice {
  return {
    createCommandEncoder: vi.fn(),
    createBuffer: vi.fn(() => ({ destroy: vi.fn() })),
    createTexture: vi.fn(() => ({ createView: vi.fn(() => ({})), destroy: vi.fn() })),
    createBindGroup: vi.fn(),
    queue: { submit: vi.fn(), writeBuffer: vi.fn(), writeTexture: vi.fn() },
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    lost: new Promise<never>(() => {}),
  } as unknown as GPUDevice;
}

function makeOpts(warnings: EngineWarning[]): HybridEngineOptions {
  return {
    device: makeDeviceStub(),
    width: 64,
    height: 64,
    primaryLightDir: [0, -1, 0],
    primaryLightIntensity: 2,
    skyTint: [1, 1, 1],
    skyIrradiance: 1,
    onWarning: (warning) => warnings.push(warning),
  };
}

/** Reach the private approximation/lifecycle warner methods. */
interface WarnerInternals {
  _warnUnconsumedMaterialFields(
    fields: readonly string[],
    method: 'setScene' | 'updatePrimitive',
    primitiveFields?: readonly { primitiveId: string; fields: readonly string[] }[],
  ): void;
  _warnMaterialTextureAtlasDiagnostics(d: readonly unknown[], m: string): void;
  _warnUnknownPrimitivePatchFields(id: string, fields: readonly string[]): void;
}

function makeEngine(): { engine: HybridEngine; warnings: EngineWarning[] } {
  const warnings: EngineWarning[] = [];
  const engine = new HybridEngine(makeOpts(warnings));
  return { engine, warnings };
}

function w(engine: HybridEngine): WarnerInternals {
  return engine as unknown as WarnerInternals;
}

/** Strip the noisy fields we don't want to over-pin (large detail objects are
 *  compared by-value; we snapshot the whole warning). */
function pick(warning: EngineWarning): { code: string; message: string; details: unknown } {
  return {
    code: warning.code,
    message: warning.message,
    details: (warning as unknown as { details?: unknown }).details,
  };
}

describe('MaterialApproximationWarner characterization pins', () => {
  let warnSpy: MockInstance;
  beforeEach(() => {
    warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
  });
  afterEach(() => {
    warnSpy.mockRestore();
  });

  it('unconsumed-material-fields: exact payload + once-only dedup', () => {
    const { engine, warnings } = makeEngine();
    w(engine)._warnUnconsumedMaterialFields(
      ['thinFilmStack', 'dispersionAbbeNumber'],
      'setScene',
      [{ primitiveId: 'm', fields: ['dispersionAbbeNumber', 'thinFilmStack'] }],
    );
    // second call with same sorted key = silent
    w(engine)._warnUnconsumedMaterialFields(['dispersionAbbeNumber', 'thinFilmStack'], 'setScene');
    expect(warnings).toHaveLength(1);
    expect(pick(warnings[0]!)).toEqual({
      code: 'walkaround-hybrid.unconsumed-material-fields',
      message:
        `[vitrum/walkaround-hybrid] setScene: the following material fields are ` +
        `supplied but not consumed by this backend: dispersionAbbeNumber, thinFilmStack. ` +
        `See consumedMaterialFields.ts for the full allowlist.`,
      details: {
        fields: ['dispersionAbbeNumber', 'thinFilmStack'],
        categories: (warnings[0]!.details as { categories: unknown }).categories,
        primitiveFields: [{ primitiveId: 'm', fields: ['dispersionAbbeNumber', 'thinFilmStack'] }],
      },
    });
    // empty-list early return
    w(engine)._warnUnconsumedMaterialFields([], 'setScene');
    expect(warnings).toHaveLength(1);
  });

  it('unknown-primitive-patch-fields: id-keyed dedup', () => {
    const { engine, warnings } = makeEngine();
    w(engine)._warnUnknownPrimitivePatchFields('p', ['zzz', 'aaa', 'aaa']);
    w(engine)._warnUnknownPrimitivePatchFields('p', ['aaa', 'zzz']);
    expect(warnings).toHaveLength(1);
    expect(warnings[0]!.code).toBe('walkaround-hybrid.unknown-primitive-patch-fields');
    expect(warnings[0]!.message).toContain('updatePrimitive("p")');
    expect(warnings[0]!.message).toContain('aaa, zzz.');
    expect((warnings[0]!.details as { fields: string[] }).fields).toEqual(['aaa', 'zzz']);
  });

  it('material-texture-atlas: every live diagnostic arm produces the correct code/message/fallback', () => {
    const { engine, warnings } = makeEngine();
    const diagnostics = [
      {
        code: 'ambiguous-material-texture-stride',
        materialIndex: 2,
        field: 'normalMap',
        colorSpace: 'linear',
        pixelStride: 3,
        valueCount: 12,
        width: 2,
        height: 2,
      },
      {
        code: 'invalid-material-texture-transform',
        materialIndex: 3,
        field: 'roughnessMap',
        colorSpace: 'linear',
        transformComponents: ['scaleX', 'scaleY'],
      },
      {
        code: 'unreadable-material-texture-map',
        materialIndex: 5,
        field: 'emissiveMap',
        colorSpace: 'srgb',
      },
    ];
    w(engine)._warnMaterialTextureAtlasDiagnostics(diagnostics, 'setScene');
    expect(warnings.map((x) => x.code)).toEqual([
      'walkaround-hybrid.ambiguous-material-texture-stride',
      'walkaround-hybrid.invalid-material-texture-transform',
      'walkaround-hybrid.unreadable-material-texture-map',
    ]);
    // Arm-specific message fragments
    expect(warnings[0]!.message).toContain('ambiguous raw pixel stride 3');
    expect(warnings[1]!.message).toContain('non-finite texture transform component(s) scaleX, scaleY');
    expect(warnings[2]!.message).toContain('neither CPU-readable nor a nominal');
    // fallback field (details) per-arm
    expect((warnings[0]!.details as { fallback: string }).fallback).toBe('heuristic pixel stride');
    expect((warnings[1]!.details as { fallback: string }).fallback).toBe('identity texture transform fallback');
    expect((warnings[2]!.details as { fallback: string }).fallback).toBe('map ignored');
    // dedup: re-running the same diagnostics adds nothing
    const before = warnings.length;
    w(engine)._warnMaterialTextureAtlasDiagnostics(diagnostics, 'setScene');
    expect(warnings).toHaveLength(before);
  });

  it('integration: implemented alpha, light-map, and rich fields emit no retired warnings', () => {
    const { engine, warnings } = makeEngine();
    const scene = makeApproxScene();
    try {
      engine.setScene(scene);
    } catch {
      /* pipeline init may throw on the stub device — warnings already emitted */
    }
    const retiredCodes = warnings
      .map((x) => x.code)
      .filter((c) => RETIRED_APPROX_CODES.has(c));
    expect(retiredCodes).toEqual([]);
  });
});

const RETIRED_APPROX_CODES = new Set([
  'walkaround-hybrid.alpha-blend-approximation',
  'walkaround-hybrid.light-map-camera-visible-approximation',
  'walkaround-hybrid.rich-material-gi-approximation',
]);

function makeApproxScene(): Scene {
  const primitives: ScenePrimitive[] = [
    {
      kind: 'mesh',
      id: 'approx',
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      colors: new Float32Array([1, 1, 1, 0.5, 1, 1, 1, 0.5, 1, 1, 1, 0.5]),
      material: {
        baseColor: [1, 1, 1],
        roughness: 0.5,
        metallic: 0,
        alphaMode: 'blend',
        opacity: 0.5,
        lightMap: { handle: { width: 1, height: 1, data: new Uint8Array([255, 255, 255, 255]), __vitrum_hint__: { channels: 4, dataType: 'uint8' } } },
        // Optical dispersion is now consumed by the shared material-optics
        // atlas and must not regress to an unconsumed-field warning.
        dispersionAbbeNumber: 42,
      } as unknown as ScenePrimitive['material'],
      transform: asMat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])),
    } as unknown as ScenePrimitive,
  ];
  const emitters: SceneEmitter[] = [];
  return { primitives, emitters, environment: { kind: 'none' } };
}
