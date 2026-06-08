import { describe, expect, it } from 'vitest';
import { float16BitsToFloat32 } from '@vitrum/shared-denoisers';
import type { Scene } from '@vitrum/core';
import { FloatType, HalfFloatType } from 'three';
import { BDPT_KIND_INVALID, BDPT_KIND_LIGHT, sampleBdptBounce0FromScene } from '../legacy/three/bdpt/bdptSceneEmittersCpu.js';
import {
  encodeBdptLightPathTextureData,
  packBdptLightPathColumnsWebGL,
} from '../legacy/three/bdpt/fillBdptLightPathWebGL.js';

const cornellEmitters: Scene = {
  primitives: [],
  environment: { kind: 'none' },
  emitters: [
    {
      id: 'light',
      kind: 'rect-area',
      color: [1, 1, 1],
      intensity: 12,
      position: [0, 0.98, 0],
      uAxis: [0.5, 0, 0],
      vAxis: [0, 0, 0.5],
    },
  ],
};

describe('packBdptLightPathColumnsWebGL', () => {
  it('marks unused columns invalid and packs bounce-0 like the fork', () => {
    const sample = sampleBdptBounce0FromScene(cornellEmitters, 7);
    expect(sample).not.toBeNull();
    const width = 3;
    const data = packBdptLightPathColumnsWebGL(width, sample);
    expect(data[3]).toBe(BDPT_KIND_LIGHT);
    expect(data[width * 4 + 3]).toBeCloseTo(sample!.pdfJoint, 5);
    expect(data[width * 8 + 0]).toBeCloseTo(sample!.throughput[0], 5);
    expect(data[width * 8 + 3]).toBeCloseTo(sample!.pdfHemi, 5);
    expect(data[1 * 4 + 3]).toBe(BDPT_KIND_INVALID);
  });

  it('returns all-invalid columns when bounce0 is null', () => {
    const data = packBdptLightPathColumnsWebGL(3, null);
    for (let col = 0; col < 3; col += 1) {
      expect(data[col * 4 + 3]).toBe(BDPT_KIND_INVALID);
    }
  });

  it('encodes half-float upload payloads as binary16 Uint16Array', () => {
    const data = new Float32Array([1, 0.5, -2, 65504]);
    const floatPayload = encodeBdptLightPathTextureData(data, FloatType);
    expect(floatPayload).toBe(data);

    const halfPayload = encodeBdptLightPathTextureData(data, HalfFloatType);
    expect(halfPayload).toBeInstanceOf(Uint16Array);
    const half = halfPayload as Uint16Array;
    expect(float16BitsToFloat32(half[0]!)).toBeCloseTo(1, 6);
    expect(float16BitsToFloat32(half[1]!)).toBeCloseTo(0.5, 6);
    expect(float16BitsToFloat32(half[2]!)).toBeCloseTo(-2, 6);
    expect(float16BitsToFloat32(half[3]!)).toBeCloseTo(65504, 0);
  });
});
