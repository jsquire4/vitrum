/**
 * W1-R2 — FrameResources shape contract test.
 *
 * The legacy `FrameResources` interface had 41 sibling fields. W1-R2 grouped
 * them into 8 per-algorithm sub-structs (common, restirDI, restirGI, ddgi,
 * gtao, svgf, ppg, neural). This test pins the field-path migration map so
 * that any future reshuffle is forced to update the test in lockstep —
 * making the migration auditable field-by-field for any downstream consumer
 * who reads `res.X` and needs to find the new path.
 *
 * PPG and neural are intentionally empty for now (W9 / W10 will populate).
 *
 * See plan/premium-grade-refactor-20260517.md §W1-R2 and
 * complexity-sweep-20260517 findings A3 + B6.
 */

import { describe, it, expect, vi } from 'vitest';
import { installWebGPUPolyfills } from './helpers/webgpuPolyfills.js';
import { createFrameResources } from '../src/pipeline/resourceManager.js';

// vitest runs under happy-dom; install GPUBufferUsage / GPUTextureUsage globals
// before the module-under-test reads them inside createFrameResources().
installWebGPUPolyfills();

/**
 * The canonical legacy → new mapping. Each entry is
 * `[legacyFieldName, newSubStruct]`. Every legacy field must appear here
 * exactly once and the test asserts it lives under the declared sub-struct.
 */
const FIELD_MIGRATION_TABLE = [
  // common ────────────────────────────────────────────────────────────────
  ['hdrColorTexture',                'common'],
  ['gNormalDepthTexture',            'common'],
  ['denoisedPingTexture',            'common'],
  ['denoisedPongTexture',            'common'],
  ['accumTextureA',                  'common'],
  ['accumTextureB',                  'common'],
  ['placeholderTexture',             'common'],
  ['uboBuffer',                      'common'],
  ['nearestSampler',                 'common'],
  ['compositeSampler',               'common'],
  ['motionVectorTexture',            'common'],
  ['tierTexture',                    'common'],
  ['resolvedTexture',                'common'],
  ['hdrTotalTexture',                'common'],
  ['albedoTexture',                  'common'],
  ['hdrIndirectTexture',             'common'],
  ['combinedDenoisedTexture',        'common'],
  ['indirectDenoisedPingTexture',    'common'],
  ['indirectDenoisedPongTexture',    'common'],
  ['indirectAccumPingTexture',       'common'],
  ['indirectAccumPongTexture',       'common'],
  ['varianceBuffer',                 'common'],
  ['varianceBufferAux',              'common'],
  ['atrousVarianceEstimateTexture',  'common'],

  // restirDI ──────────────────────────────────────────────────────────────
  ['reservoirCurrentBuffer',         'restirDI'],
  ['reservoirPreviousBuffer',        'restirDI'],
  ['reservoirSpatialBuffer',         'restirDI'],

  // restirGI ──────────────────────────────────────────────────────────────
  ['reservoirGiCurrentBuffer',       'restirGI'],
  ['reservoirGiPreviousBuffer',      'restirGI'],
  ['reservoirGiSpatialBuffer',       'restirGI'],

  // ddgi ──────────────────────────────────────────────────────────────────
  ['ddgiPlaceholderRgba16f',         'ddgi'],
  ['ddgiPlaceholderRg16f',           'ddgi'],
  ['ddgiUboBuffer',                  'ddgi'],

  // gtao ──────────────────────────────────────────────────────────────────
  ['aoHalfTexture',                  'gtao'],
  ['aoFullTexture',                  'gtao'],
  ['gtaoUboBuffer',                  'gtao'],

  // svgf ──────────────────────────────────────────────────────────────────
  ['svgfObjIdPlaceholderTexture',    'svgf'],
  ['svgfHistoryLengthTextureA',      'svgf'],
  ['svgfHistoryLengthTextureB',      'svgf'],
  ['svgfMomentsTextureA',            'svgf'],
  ['svgfMomentsTextureB',            'svgf'],
  ['svgfPrevRadianceTextureA',       'svgf'],
  ['svgfPrevRadianceTextureB',       'svgf'],
  ['svgfVarianceTexture',            'svgf'],
  ['svgfVarianceMomentsIntermedTexture', 'svgf'],
  ['svgfDepthTextureA',              'svgf'],
  ['svgfDepthTextureB',              'svgf'],
] as const;

function makeMockDevice() {
  const textureMock = { destroy: vi.fn(), createView: vi.fn(() => ({})) };
  const bufferMock  = { destroy: vi.fn(), size: 256 };
  const samplerMock = {};
  return {
    createBuffer:  vi.fn(() => bufferMock),
    createTexture: vi.fn(() => textureMock),
    createSampler: vi.fn(() => samplerMock),
    queue: { writeBuffer: vi.fn(), writeTexture: vi.fn() },
  } as unknown as GPUDevice;
}

describe('FrameResources shape — W1-R2 per-algorithm sub-structs', () => {
  it('returns exactly the 8 declared sub-structs', () => {
    const res = createFrameResources(makeMockDevice(), 64, 64);
    expect(Object.keys(res).sort()).toEqual(
      ['common', 'ddgi', 'gtao', 'neural', 'ppg', 'restirDI', 'restirGI', 'svgf'],
    );
  });

  it('every legacy field appears under its declared sub-struct', () => {
    const res = createFrameResources(makeMockDevice(), 64, 64) as unknown as Record<string, Record<string, unknown>>;
    for (const [legacyField, subStruct] of FIELD_MIGRATION_TABLE) {
      const bucket = res[subStruct];
      expect(bucket, `sub-struct '${subStruct}' should exist`).toBeDefined();
      expect(bucket, `legacy field '${legacyField}' should live under '${subStruct}'`).toHaveProperty(legacyField);
      expect(bucket![legacyField], `'${subStruct}.${legacyField}' must be non-null`).not.toBeUndefined();
    }
  });

  it('migration table is exhaustive — covers every legacy FrameResources sibling field', () => {
    // The legacy interface now has 47 sibling fields (the "41-field god-struct"
    // shorthand in the W1-R2 brief is approximate — actual count when
    // enumerated: common 24 + restirDI 3 + restirGI 3 + ddgi 3 + gtao 3 +
    // svgf 11 = 47). Every entry must appear exactly once.
    expect(FIELD_MIGRATION_TABLE.length).toBe(47);
    const seen = new Set<string>();
    for (const [field] of FIELD_MIGRATION_TABLE) {
      expect(seen.has(field), `legacy field '${field}' listed twice`).toBe(false);
      seen.add(field);
    }
  });

  it('ppg and neural sub-structs are present but empty (W9 / W10 placeholders)', () => {
    const res = createFrameResources(makeMockDevice(), 64, 64);
    expect(res.ppg).toBeDefined();
    expect(res.neural).toBeDefined();
    // Empty placeholder — only the optional `_empty?: never` marker; no own enumerable keys.
    expect(Object.keys(res.ppg)).toEqual([]);
    expect(Object.keys(res.neural)).toEqual([]);
  });

  it('field-set under each sub-struct exactly matches the migration table', () => {
    const res = createFrameResources(makeMockDevice(), 64, 64) as unknown as Record<string, Record<string, unknown>>;
    const buckets: Record<string, string[]> = {};
    for (const [field, sub] of FIELD_MIGRATION_TABLE) {
      (buckets[sub] ??= []).push(field);
    }
    for (const [subStruct, expected] of Object.entries(buckets)) {
      const actual = Object.keys(res[subStruct]!).sort();
      expect(actual, `'${subStruct}' field list must match migration table`).toEqual(expected.slice().sort());
    }
  });
});
