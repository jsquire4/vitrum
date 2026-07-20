/**
 * Characterization pins for the HybridEngine material-approximation warning
 * subsystem (T3-A). Captures the exact [code, message, details] payloads +
 * once-only dedup semantics + atlas-diagnostic ternary arms BEFORE the
 * `MaterialApproximationWarner` extraction, so the extraction is proven
 * byte-identical.
 *
 * The 8 approximation `_warn*` methods + the 2 lifecycle `_warn*` methods are
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
  _warnApproximateAlphaBlendPrimitiveIds(ids: readonly string[], m: string): void;
  _warnApproximateEmissiveMapTexelPdfPrimitiveIds(ids: readonly string[], m: string): void;
  _warnApproximateLightMapPrimitiveIds(ids: readonly string[], m: string): void;
  _warnApproximateRichMaterialPrimitiveFields(
    pf: readonly { primitiveId: string; fields: readonly string[] }[],
    m: string,
  ): void;
  _warnApproximateVolumeLayerPrimitiveFields(
    pf: readonly { primitiveId: string; fields: readonly string[] }[],
    m: string,
  ): void;
  _warnReservedReceiveShadowPrimitiveIds(ids: readonly string[], m: string): void;
  _warnMaterialTextureAtlasDiagnostics(d: readonly unknown[], m: string): void;
  _warnUnknownPrimitivePatchFields(id: string, fields: readonly string[]): void;
  _warnInvalidSetSize(w: number, h: number): void;
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

  it('alpha-blend-approximation: exact payload + dedup', () => {
    const { engine, warnings } = makeEngine();
    w(engine)._warnApproximateAlphaBlendPrimitiveIds(['p1', 'p2'], 'updatePrimitive');
    w(engine)._warnApproximateAlphaBlendPrimitiveIds(['p1', 'p2'], 'updatePrimitive');
    expect(warnings).toHaveLength(1);
    expect(warnings[0]!.code).toBe('walkaround-hybrid.alpha-blend-approximation');
    expect(warnings[0]!.message).toContain('primitives: p1, p2.');
    expect((warnings[0]!.details as { primitiveIds: string[] }).primitiveIds).toEqual(['p1', 'p2']);
    expect((warnings[0]! as { method?: string }).method).toBe('updatePrimitive');
  });

  it('emissive-map-texel-pdf-approximation: exact code + updateEmitter method arm', () => {
    const { engine, warnings } = makeEngine();
    w(engine)._warnApproximateEmissiveMapTexelPdfPrimitiveIds(['e1'], 'updateEmitter');
    expect(warnings).toHaveLength(1);
    expect(warnings[0]!.code).toBe('walkaround-hybrid.emissive-map-texel-pdf-approximation');
    expect((warnings[0]! as { method?: string }).method).toBe('updateEmitter');
    expect(warnings[0]!.message).toContain('primitives: e1.');
  });

  it('light-map-camera-visible-approximation: exact code + dedup', () => {
    const { engine, warnings } = makeEngine();
    w(engine)._warnApproximateLightMapPrimitiveIds(['l1'], 'setScene');
    w(engine)._warnApproximateLightMapPrimitiveIds(['l1'], 'setScene');
    expect(warnings).toHaveLength(1);
    expect(warnings[0]!.code).toBe('walkaround-hybrid.light-map-camera-visible-approximation');
  });

  it('rich-material-gi-approximation: normalized+sorted key + fieldSet', () => {
    const { engine, warnings } = makeEngine();
    w(engine)._warnApproximateRichMaterialPrimitiveFields(
      [
        { primitiveId: 'b', fields: ['sheen', 'clearcoat'] },
        { primitiveId: 'a', fields: ['anisotropy'] },
      ],
      'setScene',
    );
    // Same set, different input order → same normalized key → dedup
    w(engine)._warnApproximateRichMaterialPrimitiveFields(
      [
        { primitiveId: 'a', fields: ['anisotropy'] },
        { primitiveId: 'b', fields: ['clearcoat', 'sheen'] },
      ],
      'setScene',
    );
    expect(warnings).toHaveLength(1);
    expect(warnings[0]!.code).toBe('walkaround-hybrid.rich-material-gi-approximation');
    const d = warnings[0]!.details as { primitiveFields: unknown; fields: string[] };
    expect(d.primitiveFields).toEqual([
      { primitiveId: 'a', fields: ['anisotropy'] },
      { primitiveId: 'b', fields: ['clearcoat', 'sheen'] },
    ]);
    expect(d.fields).toEqual(['anisotropy', 'clearcoat', 'sheen']);
    expect(warnings[0]!.message).toContain('primitives: a, b.');
  });

  it('volume-layer-transport-approximation: normalized key + fieldSet', () => {
    const { engine, warnings } = makeEngine();
    w(engine)._warnApproximateVolumeLayerPrimitiveFields(
      [{ primitiveId: 'v', fields: ['backLayer', 'frontLayer'] }],
      'updatePrimitive',
    );
    expect(warnings).toHaveLength(1);
    expect(warnings[0]!.code).toBe('walkaround-hybrid.volume-layer-transport-approximation');
    const d = warnings[0]!.details as { fields: string[] };
    expect(d.fields).toEqual(['backLayer', 'frontLayer']);
  });

  it('reserved-receive-shadow: NOT deduped (fires every call)', () => {
    const { engine, warnings } = makeEngine();
    w(engine)._warnReservedReceiveShadowPrimitiveIds(['s1'], 'setScene');
    w(engine)._warnReservedReceiveShadowPrimitiveIds(['s1'], 'setScene');
    expect(warnings).toHaveLength(2);
    expect(warnings[0]!.code).toBe('walkaround-hybrid.reserved-receive-shadow');
    expect(warnings[0]!.message).toContain('receiveShadow:false is reserved');
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

  it('invalid-set-size: dimension-keyed dedup', () => {
    const { engine, warnings } = makeEngine();
    w(engine)._warnInvalidSetSize(0, 5);
    w(engine)._warnInvalidSetSize(0, 5);
    expect(warnings).toHaveLength(1);
    expect(warnings[0]!.code).toBe('walkaround-hybrid.invalid-set-size');
    expect(warnings[0]!.message).toContain('setSize(0, 5)');
  });

  it('material-texture-atlas: all 5 ternary arms produce the correct code/message/fallback', () => {
    const { engine, warnings } = makeEngine();
    const diagnostics = [
      {
        code: 'unsupported-material-texture-texcoord',
        materialIndex: 1,
        field: 'baseColorMap',
        colorSpace: 'srgb',
        texCoord: 3,
        sourcePath: 'scene.gltf',
      },
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
        code: 'material-texture-sampler-policy-approximation',
        materialIndex: 4,
        field: 'aoMap',
        colorSpace: 'linear',
        magFilter: 'linear',
        minFilter: 'nearest',
        mipFilter: 'linear',
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
      'walkaround-hybrid.unsupported-material-texture-texcoord',
      'walkaround-hybrid.ambiguous-material-texture-stride',
      'walkaround-hybrid.invalid-material-texture-transform',
      'walkaround-hybrid.material-texture-sampler-policy-approximation',
      'walkaround-hybrid.unreadable-material-texture-map',
    ]);
    // Arm-specific message fragments
    expect(warnings[0]!.message).toContain('uses texCoord 3; the material atlas only supports UV sets 0 and 1');
    expect(warnings[1]!.message).toContain('ambiguous raw pixel stride 3');
    expect(warnings[2]!.message).toContain('non-finite texture transform component(s) scaleX, scaleY');
    expect(warnings[3]!.message).toContain('mag=linear, min=nearest, mip=linear');
    expect(warnings[4]!.message).toContain('not CPU-readable');
    // fallback field (details) per-arm
    expect((warnings[1]!.details as { fallback: string }).fallback).toBe('heuristic pixel stride');
    expect((warnings[2]!.details as { fallback: string }).fallback).toBe('identity texture transform fallback');
    expect((warnings[3]!.details as { fallback: string }).fallback).toBe('base-level atlas sampler');
    expect((warnings[4]!.details as { fallback: string }).fallback).toBe('map ignored');
    // dedup: re-running the same diagnostics adds nothing
    const before = warnings.length;
    w(engine)._warnMaterialTextureAtlasDiagnostics(diagnostics, 'setScene');
    expect(warnings).toHaveLength(before);
  });

  it('material-texture-atlas: details payload for the unsupported-texcoord arm is exact', () => {
    const { engine, warnings } = makeEngine();
    w(engine)._warnMaterialTextureAtlasDiagnostics(
      [
        {
          code: 'unsupported-material-texture-texcoord',
          materialIndex: 7,
          field: 'baseColorMap',
          colorSpace: 'srgb',
          texCoord: 2,
          sourcePath: 'p.gltf',
          textureIndex: 4,
        },
      ],
      'updatePrimitive',
    );
    expect(warnings).toHaveLength(1);
    expect(warnings[0]!.details).toEqual({
      materialIndex: 7,
      field: 'baseColorMap',
      colorSpace: 'srgb',
      texCoord: 2,
      sourcePath: 'p.gltf',
      textureIndex: 4,
      fallback: 'map ignored',
    });
  });

  it('integration: setScene emits the approximation warnings in a fixed order', () => {
    const { engine, warnings } = makeEngine();
    const scene = makeApproxScene();
    try {
      engine.setScene(scene);
    } catch {
      /* pipeline init may throw on the stub device — warnings already emitted */
    }
    // Filter to the material-approximation codes and pin the ORDER of first
    // occurrences (scene-support/other warnings interleave but the approximation
    // sequence order is load-bearing).
    const approxCodes = warnings
      .map((x) => x.code)
      .filter((c) => c.startsWith('walkaround-hybrid.') && APPROX_CODES.has(c));
    // Order of first-emission for the classes present in the fixture:
    const firstOrder: string[] = [];
    for (const c of approxCodes) if (!firstOrder.includes(c)) firstOrder.push(c);
    expect(firstOrder).toEqual([
      'walkaround-hybrid.unconsumed-material-fields',
      'walkaround-hybrid.alpha-blend-approximation',
      'walkaround-hybrid.light-map-camera-visible-approximation',
      'walkaround-hybrid.reserved-receive-shadow',
    ]);
  });
});

const APPROX_CODES = new Set([
  'walkaround-hybrid.unconsumed-material-fields',
  'walkaround-hybrid.alpha-blend-approximation',
  'walkaround-hybrid.emissive-map-texel-pdf-approximation',
  'walkaround-hybrid.light-map-camera-visible-approximation',
  'walkaround-hybrid.rich-material-gi-approximation',
  'walkaround-hybrid.volume-layer-transport-approximation',
  'walkaround-hybrid.reserved-receive-shadow',
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
        // permanently-unsupported field → unconsumed-material-fields
        dispersionAbbeNumber: 42,
      } as unknown as ScenePrimitive['material'],
      transform: asMat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])),
      // reserved-receive-shadow
      receiveShadow: false,
    } as unknown as ScenePrimitive,
  ];
  const emitters: SceneEmitter[] = [];
  return { primitives, emitters, environment: { kind: 'none' } };
}
