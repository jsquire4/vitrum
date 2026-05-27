import { describe, expect, it } from 'vitest';
import type { Scene } from '@vitrum/core';
import { BDPT_KIND_INVALID, BDPT_KIND_LIGHT, sampleBdptBounce0FromScene } from '../bdpt/bdptSceneEmittersCpu.js';
import { packBdptLightPathColumnsWebGL } from '../bdpt/fillBdptLightPathWebGL.js';

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
});
