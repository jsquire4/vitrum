/**
 * thinFilmLayerLimit.test.ts — D1 (Option B) pt-webgl2 thin-film capacity.
 *
 * pt-webgl2 packs exactly up to `THIN_FILM_LAYER_LIMIT` (35) thin-film layers.
 * This suite pins: (1) a material declaring > 35 layers is rejected before
 * packing or retained-engine mutation; (2) the packer constant matches
 * the value declared in `@vitrum/core`'s
 * `BACKEND_PROMISE_LEDGER['pt-webgl2'].supportDetails.thinFilmLayerLimit`
 * (drift-guard); (3) a 35-layer material is accepted without truncation.
 */
import { describe, it, expect } from 'vitest';
import type { MaterialSpec, Scene, ThinFilmStack } from '@vitrum/core';
import { BACKEND_PROMISE_LEDGER } from '@vitrum/core';
import { createPTEngine_WebGL2 } from '../index.js';
import { createMockGl } from '../__tests__/mockGl.js';
import { packMaterialsTexture, THIN_FILM_LAYER_LIMIT } from './materialsTexture.js';

function materialWithThinFilmLayers(count: number): MaterialSpec {
  const layers = Array.from({ length: count }, (_, i) => ({
    ior: 1.3 + i * 0.001,
    thicknessNm: 100 + i,
  }));
  const thinFilmStack: ThinFilmStack = { layers };
  return { baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0, thinFilmStack };
}

function sceneWithMaterial(sceneMaterial: MaterialSpec): Scene {
  return {
    primitives: [{
      kind: 'mesh',
      id: 'film',
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      uvs: new Float32Array([0, 0, 1, 0, 0, 1]),
      material: sceneMaterial,
    }],
    emitters: [],
    environment: { kind: 'none' },
  };
}

describe('D1 — pt-webgl2 thin-film layer limit', () => {
  it('declares thinFilmLayerLimit === packer constant (35) in the promise ledger', () => {
    const declared = BACKEND_PROMISE_LEDGER['pt-webgl2'].supportDetails.thinFilmLayerLimit;
    expect(declared).toBe(THIN_FILM_LAYER_LIMIT);
    expect(declared).toBe(35);
  });

  it('rejects a material above 35 layers before allocating a packed texture', () => {
    const requested = 40;
    expect(() => packMaterialsTexture([materialWithThinFilmLayers(requested)]))
      .toThrow(`material 0 declares ${requested} thin-film layers; the exact backend limit is 35`);
  });

  it('packs a material at exactly the 35-layer limit', () => {
    expect(() => packMaterialsTexture([materialWithThinFilmLayers(35)])).not.toThrow();
  });

  it('rejects setScene and material patches before changing retained state or uploading', async () => {
    const record = new Map<string, unknown>();
    const engine = await createPTEngine_WebGL2({ device: createMockGl(record) });
    engine.setScene(sceneWithMaterial(materialWithThinFilmLayers(1)));
    const before = engine.getScene!();
    const uploadCountBefore = (record.get('__texImage2D') as unknown[] | undefined)?.length ?? 0;

    expect(() => engine.setScene(sceneWithMaterial(materialWithThinFilmLayers(36))))
      .toThrow(/setScene: primitive "film" material declares 36 thin-film layers/);
    expect(engine.getScene!()).toBe(before);
    expect((record.get('__texImage2D') as unknown[] | undefined)?.length ?? 0)
      .toBe(uploadCountBefore);

    expect(() => engine.updatePrimitive?.('film', {
      material: materialWithThinFilmLayers(40),
    })).toThrow(/updatePrimitive: primitive "film" material declares 40 thin-film layers/);
    expect(engine.getScene!()).toBe(before);
    expect((record.get('__texImage2D') as unknown[] | undefined)?.length ?? 0)
      .toBe(uploadCountBefore);
    engine.dispose();
  });
});
