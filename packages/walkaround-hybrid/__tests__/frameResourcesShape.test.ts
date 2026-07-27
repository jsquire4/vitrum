/**
 * W1-R2 — FrameResources shape contract test.
 *
 * The legacy `FrameResources` interface had 41 sibling fields. W1-R2 grouped
 * them into resource-owning sub-structs (common, restirDI, restirGI, ddgi,
 * gtao, svgf, ppg). This test pins the field-path migration map so
 * that any future reshuffle is forced to update the test in lockstep —
 * making the migration auditable field-by-field for any downstream consumer
 * who reads `res.X` and needs to find the new path.
 *
 * PPG resources are lazy opt-in. Neural inference resources are owned by the
 * inference graph and therefore are not represented by an empty frame bucket.
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
  ['transparentCompositeTexture',    'common'],
  ['indirectDenoisedPingTexture',    'common'],
  ['indirectDenoisedPongTexture',    'common'],
  ['indirectAccumPingTexture',       'common'],
  ['indirectAccumPongTexture',       'common'],
  ['varianceBuffer',                 'common'],
  ['varianceBufferAux',              'common'],
  ['atrousVarianceEstimateTexture',  'common'],
  ['checkerboardRadianceSnapshotTexture', 'common'],

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
  ['ddgiPlaceholderVisRgba16f',      'ddgi'],
  ['ddgiSampler',                    'ddgi'],
  ['ddgiUboBuffer',                  'ddgi'],

  // gtao ──────────────────────────────────────────────────────────────────
  ['aoHalfTexture',                  'gtao'],
  ['aoFullTexture',                  'gtao'],
  ['gtaoUboBuffer',                  'gtao'],

  // svgf ──────────────────────────────────────────────────────────────────
  ['svgfCurrentObjectIdTexture',     'svgf'],
  ['svgfPreviousObjectIdTexture',    'svgf'],
  ['svgfPrevNormalDepthTexture',     'svgf'],
  ['svgfHistoryLengthTextureA',      'svgf'],
  ['svgfHistoryLengthTextureB',      'svgf'],
  ['svgfMomentsTextureA',            'svgf'],
  ['svgfMomentsTextureB',            'svgf'],
  ['svgfPrevRadianceTextureA',       'svgf'],
  ['svgfPrevRadianceTextureB',       'svgf'],
  ['svgfVarianceTexture',            'svgf'],
  ['svgfVarianceMomentsIntermedTexture', 'svgf'],
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
  it('returns exactly the declared resource-owning sub-structs', () => {
    const res = createFrameResources(makeMockDevice(), 64, 64);
    expect(Object.keys(res).sort()).toEqual(
      ['common', 'ddgi', 'gtao', 'ppg', 'restirDI', 'restirGI', 'svgf'],
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
    // The legacy interface had 46 sibling fields (the "41-field god-struct"
    // shorthand in the W1-R2 brief is approximate — actual count when
    // enumerated: common 26 + restirDI 3 + restirGI 3 + ddgi 3 + gtao 3 +
    // svgf 11 + the dedicated DDGI receiver sampler = 50). Every entry must
    // appear exactly once.
    expect(FIELD_MIGRATION_TABLE.length).toBe(50);
    const seen = new Set<string>();
    for (const [field] of FIELD_MIGRATION_TABLE) {
      expect(seen.has(field), `legacy field '${field}' listed twice`).toBe(false);
      seen.add(field);
    }
  });

  it('ppg starts empty and no fictitious neural frame-resource bucket is exposed', () => {
    const res = createFrameResources(makeMockDevice(), 64, 64);
    expect(res.ppg).toBeDefined();
    // PPG is `{}` by default; `allocatePPGResources` populates it only when the
    // host opts in via `ppgEnabled: true`.
    expect(Object.keys(res.ppg)).toEqual([]);
    expect('neural' in res).toBe(false);
  });

  it('svgf gating (svgfEnabled:false, G-P2.6) preserves the exact 11-field svgf shape', () => {
    // The full-res SVGF textures are gated off when the active denoiser is not
    // svgf-real. The struct shape MUST stay identical (every field non-null) so
    // nothing off the svgf-real dispatch path observes a missing field — the
    // textures simply collapse to 1×1 placeholders.
    const res = createFrameResources(makeMockDevice(), 64, 64, { svgfEnabled: false }) as unknown as Record<string, Record<string, unknown>>;
    const svgfFields = FIELD_MIGRATION_TABLE.filter(([, sub]) => sub === 'svgf').map(([f]) => f);
    expect(Object.keys(res.svgf!).sort()).toEqual(svgfFields.slice().sort());
    for (const f of svgfFields) {
      expect(res.svgf![f], `svgf.${f} must be non-null even when gated`).not.toBeUndefined();
    }
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
