/**
 * thinFilmLayerLimit.test.ts — D1 (Option B) pt-webgl2 thin-film capacity.
 *
 * pt-webgl2 packs at most `THIN_FILM_LAYER_LIMIT` (35) thin-film layers. This
 * suite pins: (1) a material declaring > 35 layers emits a structured
 * `thin-film-layer-limit-exceeded` warning naming the requested count + limit;
 * (2) the packer constant matches the value declared in `@vitrum/core`'s
 * `BACKEND_PROMISE_LEDGER['pt-webgl2'].supportDetails.thinFilmLayerLimit`
 * (drift-guard); (3) a ≤ 35-layer material emits NO such warning.
 */
import { describe, it, expect, vi, afterEach } from 'vitest';
import type { EngineWarning, MaterialSpec, ThinFilmStack } from '@vitrum/core';
import { BACKEND_PROMISE_LEDGER } from '@vitrum/core';
import { packMaterialsTexture, THIN_FILM_LAYER_LIMIT } from './materialsTexture.js';

afterEach(() => {
  vi.restoreAllMocks();
});

function materialWithThinFilmLayers(count: number): MaterialSpec {
  const layers = Array.from({ length: count }, (_, i) => ({
    ior: 1.3 + i * 0.001,
    thicknessNm: 100 + i,
  }));
  const thinFilmStack: ThinFilmStack = { layers };
  return { baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0, thinFilmStack };
}

describe('D1 — pt-webgl2 thin-film layer limit', () => {
  it('declares thinFilmLayerLimit === packer constant (35) in the promise ledger', () => {
    const declared = BACKEND_PROMISE_LEDGER['pt-webgl2'].supportDetails.thinFilmLayerLimit;
    expect(declared).toBe(THIN_FILM_LAYER_LIMIT);
    expect(declared).toBe(35);
  });

  it('emits thin-film-layer-limit-exceeded when a material exceeds 35 layers', () => {
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const requested = 40;
    packMaterialsTexture([materialWithThinFilmLayers(requested)], undefined, {
      onWarning: (w) => structured.push(w),
    });
    const warning = structured.find((w) => w.code === 'thin-film-layer-limit-exceeded');
    expect(warning).toBeDefined();
    expect(warning!.backend).toBe('pt-webgl2');
    expect(warning!.details).toMatchObject({
      materialIndex: 0,
      requested,
      limit: 35,
      dropped: requested - 35,
    });
    expect(warning!.message).toContain(`${requested} thin-film layers`);
    expect(warning!.message).toContain('at most 35');
  });

  it('does NOT warn for a material at exactly the 35-layer limit', () => {
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    packMaterialsTexture([materialWithThinFilmLayers(35)], undefined, {
      onWarning: (w) => structured.push(w),
    });
    expect(structured.some((w) => w.code === 'thin-film-layer-limit-exceeded')).toBe(false);
  });
});
