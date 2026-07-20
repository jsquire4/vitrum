/**
 * thinFilmLayerLimit.test.ts — D1 (Option B) pt-webgpu thin-film capacity.
 *
 * pt-webgpu packs at most `THIN_FILM_LAYER_LIMIT` (8) thin-film layers. This
 * suite pins: (1) a scene whose material declares > 8 layers emits a structured
 * `thin-film-layer-limit-exceeded` warning naming the requested count + limit;
 * (2) the packer constant matches the value declared in `@vitrum/core`'s
 * `BACKEND_PROMISE_LEDGER['pt-webgpu'].supportDetails.thinFilmLayerLimit`
 * (drift-guard); (3) a ≤ 8-layer scene emits NO such warning.
 */
import { describe, expect, it, vi, afterEach } from 'vitest';
import type { EngineWarning, Scene, ThinFilmStack } from '@vitrum/core';
import { BACKEND_PROMISE_LEDGER } from '@vitrum/core';
import { buildPackedScene } from '../scene/uploadSceneBuffers.js';
import { THIN_FILM_LAYER_LIMIT } from '../scene/materialPacking.js';

afterEach(() => {
  vi.restoreAllMocks();
});

function sceneWithThinFilmLayers(count: number): Scene {
  const layers = Array.from({ length: count }, (_, i) => ({
    ior: 1.3 + i * 0.01,
    thicknessNm: 100 + i,
  }));
  const thinFilmStack: ThinFilmStack = { layers };
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'tf',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0, thinFilmStack },
      },
    ],
    emitters: [],
    environment: { kind: 'none' },
  };
}

describe('D1 — pt-webgpu thin-film layer limit', () => {
  it('declares thinFilmLayerLimit === packer constant (8) in the promise ledger', () => {
    const declared = BACKEND_PROMISE_LEDGER['pt-webgpu'].supportDetails.thinFilmLayerLimit;
    expect(declared).toBe(THIN_FILM_LAYER_LIMIT);
    expect(declared).toBe(8);
  });

  it('emits thin-film-layer-limit-exceeded when a material exceeds 8 layers', () => {
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const requested = 12;
    // buildPackedScene surfaces structured warnings via the returned
    // `structuredWarnings` array (the same channel index.ts setScene drains).
    const { structuredWarnings } = buildPackedScene(sceneWithThinFilmLayers(requested));
    const warning = structuredWarnings.find(
      (w: EngineWarning) => w.code === 'thin-film-layer-limit-exceeded',
    );
    expect(warning).toBeDefined();
    expect(warning!.backend).toBe('pt-webgpu');
    expect(warning!.details).toMatchObject({
      materialIndex: 0,
      requested,
      limit: 8,
      dropped: requested - 8,
    });
    expect(warning!.message).toContain(`${requested} thin-film layers`);
    expect(warning!.message).toContain('at most 8');
  });

  it('does NOT warn for a scene at exactly the 8-layer limit', () => {
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const { structuredWarnings } = buildPackedScene(sceneWithThinFilmLayers(8));
    expect(
      structuredWarnings.some((w: EngineWarning) => w.code === 'thin-film-layer-limit-exceeded'),
    ).toBe(false);
  });
});
