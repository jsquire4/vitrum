import { describe, expect, it } from 'vitest';
import type { Scene } from '@vitrum/core';
import { sampleBdptBounce0FromScene } from '../bdpt/bdptSceneEmittersCpu.js';

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

describe('sampleBdptBounce0FromScene', () => {
  it('returns a valid light vertex for rect-area emitters', () => {
    const v = sampleBdptBounce0FromScene(cornellEmitters, 42);
    expect(v).not.toBeNull();
    expect(Math.abs(v!.emitNormal[1])).toBeGreaterThan(0.5);
    expect(v!.throughput[0]).toBeGreaterThan(0);
    expect(v!.pdfJoint).toBeGreaterThan(0);
  });

  it('returns null when scene has no emitters', () => {
    expect(
      sampleBdptBounce0FromScene({ primitives: [], environment: { kind: 'none' }, emitters: [] }, 1),
    ).toBeNull();
  });
});
