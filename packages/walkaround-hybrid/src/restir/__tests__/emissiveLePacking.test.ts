/**
 * Camera-visible emitters (2026-05-30) — per-triangle emissive Le packing.
 *
 * Pins `materialEmissiveLe` (the shared emissive classifier reused by both the
 * ReSTIR-DI emitter list and the camera-glow packer, so the glow Le == the
 * NEE-sampled Le) and `packBVHEmissiveLe` (per-triangle indexing + the
 * deliberate exclusion of the transmissive "glass" branch, which `lo_emit`
 * already handles — packing it here would double-count).
 */
import { describe, it, expect } from 'vitest';
import type { MaterialSpec } from '@vitrum/core';
import {
  packBVHEmissiveLeFromCore,
} from '../packingHelpers.js';
import {
  materialEmissiveLe,
  packBVHEmissiveLe,
  type PbrMaterialLike,
} from './support/legacyPbrPackers.js';

const color = (r: number, g: number, b: number) => ({ r, g, b });

function emissiveMat(rgb: [number, number, number], intensity: number): PbrMaterialLike {
  return {
    emissive: color(rgb[0], rgb[1], rgb[2]),
    emissiveIntensity: intensity,
  };
}

describe('materialEmissiveLe — shared emissive classifier', () => {
  it('returns emissive·intensity for a self-emissive material (HDR, intensity > 1)', () => {
    const le = materialEmissiveLe(emissiveMat([0.5, 0.25, 1.0], 6));
    expect(le).not.toBeNull();
    expect(le![0]).toBeCloseTo(3.0, 6);   // 0.5 · 6
    expect(le![1]).toBeCloseTo(1.5, 6);   // 0.25 · 6
    expect(le![2]).toBeCloseTo(6.0, 6);   // 1.0 · 6
  });

  it('returns null for a non-emissive material (zero emissive)', () => {
    expect(materialEmissiveLe(emissiveMat([0, 0, 0], 5))).toBeNull();
  });

  it('returns null when emissiveIntensity is 0 (no radiance)', () => {
    expect(materialEmissiveLe(emissiveMat([1, 1, 1], 0))).toBeNull();
  });

  it('EXCLUDES a pure-transmissive glass material (no emissive) — lo_emit handles glass', () => {
    // A transmissive material with NO emissive must NOT produce a camera glow Le
    // here (that would double-count against shade.wgsl lo_emit's Beer-Lambert).
    const glass: PbrMaterialLike = {
      transmission: 0.9,
      color: color(1, 0.2, 0.2),
    };
    expect(materialEmissiveLe(glass)).toBeNull();
  });
});

describe('packBVHEmissiveLe — per-triangle HDR emissive texture data', () => {
  it('packs each triangle Le at stride 4 (rgb + 0 pad), indexed by triMaterialId', () => {
    const mats = [emissiveMat([1, 0, 0], 2), emissiveMat([0, 0, 0], 1)];
    // tri 0 → material 0 (emissive red), tri 1 → material 1 (non-emissive).
    const triMatId = new Uint32Array([0, 1]);
    const out = packBVHEmissiveLe(triMatId, mats, 2);
    expect(out.length).toBe(2 * 4);
    // tri 0: red Le = [2,0,0,0]
    expect(out[0]).toBeCloseTo(2, 6);
    expect(out[1]).toBeCloseTo(0, 6);
    expect(out[2]).toBeCloseTo(0, 6);
    expect(out[3]).toBe(0); // pad
    // tri 1: non-emissive → all zero
    expect(out[4]).toBe(0);
    expect(out[5]).toBe(0);
    expect(out[6]).toBe(0);
  });

  it('zero-fills triangles whose material is missing', () => {
    const out = packBVHEmissiveLe(new Uint32Array([7]), [], 1);
    expect(Array.from(out)).toEqual([0, 0, 0, 0]);
  });
});

describe('packBVHEmissiveLeFromCore — camera-visible core emissive Le', () => {
  it('packs scalar production Le without averaging readable emissiveMap pixels', () => {
    const material: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      emissive: [2, 2, 2],
      emissiveIntensity: 3,
      emissiveMap: {
        handle: {
          width: 2,
          height: 1,
          data: new Float32Array([
            0.25, 0.5, 1, 1,
            0.75, 0.25, 0.5, 1,
          ]),
          __vitrum_hint__: { channels: 4, dataType: 'float32', colorSpace: 'linear' },
        },
      },
    };

    const out = packBVHEmissiveLeFromCore(new Uint32Array([0]), [material], 1);

    expect(Array.from(out)).toEqual([6, 6, 6, 0]);
  });
});
