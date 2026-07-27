/**
 * thinFilmLayerLimit.test.ts — D1 (Option B) pt-webgpu thin-film capacity.
 *
 * pt-webgpu represents at most `THIN_FILM_LAYER_LIMIT` (8) thin-film layers. This
 * suite pins: (1) a scene whose material declares > 8 layers is rejected before
 * packing rather than truncated;
 * (2) the packer constant matches the value declared in `@vitrum/core`'s
 * `BACKEND_PROMISE_LEDGER['pt-webgpu'].supportDetails.thinFilmLayerLimit`
 * (drift-guard); (3) a ≤ 8-layer scene emits NO such warning.
 */
import { describe, expect, it } from 'vitest';
import type { Scene, ThinFilmStack } from '@vitrum/core';
import { BACKEND_PROMISE_LEDGER } from '@vitrum/core';
import { buildPackedScene } from '../scene/uploadSceneBuffers.js';
import { THIN_FILM_LAYER_LIMIT } from '../scene/materialPacking.js';

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

  it('rejects a material above 8 layers instead of truncating its optical stack', () => {
    const requested = 12;
    expect(() => buildPackedScene(sceneWithThinFilmLayers(requested))).toThrow(
      new RegExp(
        `thin-film scene validation: primitive "tf" declares ${requested} coherent layers, ` +
        `but this backend's exact limit is ${THIN_FILM_LAYER_LIMIT}`,
      ),
    );
  });

  it('accepts a scene at exactly the 8-layer limit', () => {
    const { structuredWarnings } = buildPackedScene(sceneWithThinFilmLayers(8));
    expect(structuredWarnings).toEqual([]);
  });
});
