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

  // Regression for the bounce-0 tangent-frame bug: the cosine hemisphere
  // direction is `wi = t*x + b*y + n*z`. The original code scaled the
  // bitangent `b` by `x` instead of `y`, leaving the `y = r*sin(phi)` local
  // dead. With frameSeed=42 the sample gives distinct x = -0.78627,
  // y = -0.61142 (x !== y), so `cosEmit = z / |t*x + b*y + n*z|` and the
  // derived hemisphere pdf differ between the buggy and corrected formulas.
  it('builds the cosine hemisphere direction with the y-scaled bitangent', () => {
    const v = sampleBdptBounce0FromScene(cornellEmitters, 42);
    expect(v).not.toBeNull();
    // emitNormal for the Cornell ceiling light is the (normalized) -Y axis.
    expect(v!.emitNormal[0]).toBeCloseTo(0, 6);
    expect(v!.emitNormal[1]).toBeCloseTo(-1, 6);
    expect(v!.emitNormal[2]).toBeCloseTo(0, 6);
    // pdfHemi = cosEmit / PI with the CORRECTED (b*y) direction.
    // The buggy (b*x) variant yields 0.0254201670 instead.
    expect(v!.pdfHemi).toBeCloseTo(0.028356709071689162, 9);
    expect(v!.pdfJoint).toBeCloseTo(0.028356709071689162, 9);
    // Guard so a regression back to b*x (which gives the lower value) fails.
    expect(v!.pdfHemi).not.toBeCloseTo(0.025420167037391334, 9);
  });

  it('returns null when scene has no emitters', () => {
    expect(
      sampleBdptBounce0FromScene({ primitives: [], environment: { kind: 'none' }, emitters: [] }, 1),
    ).toBeNull();
  });

  // Characterization golden for the deduped `buildTangentFrame` helper, which now
  // single-sources the two TBN constructions in this file: the cosine-hemisphere
  // frame in `finishBounce0` AND the disc-area in-plane frame. A disc-area emitter
  // with a tilted normal exercises BOTH frames in one sample (disc placement uses
  // the disc-plane TBN; the hemisphere direction uses the bounce-0 TBN). These
  // values were captured from the pre-dedup inline frames (Theme J, 2026-05-30).
  it('golden: disc-area sample is byte-stable across the buildTangentFrame dedup', () => {
    const discScene: Scene = {
      primitives: [],
      environment: { kind: 'none' },
      emitters: [
        {
          id: 'disc',
          kind: 'disc-area',
          color: [0.9, 0.8, 0.7],
          intensity: 5,
          position: [1, 2, 3],
          normal: [0.2, 0.3, 0.9],
          radius: 0.5,
        },
      ],
    };
    const v = sampleBdptBounce0FromScene(discScene, 7);
    expect(v).not.toBeNull();
    expect(v!.emitPos[0]).toBeCloseTo(1.495852037593665, 9);
    expect(v!.emitPos[1]).toBeCloseTo(1.548056832775566, 9);
    expect(v!.emitPos[2]).toBeCloseTo(3.0404583807206635, 9);
    expect(v!.emitNormal[0]).toBeCloseTo(0.2062842492517587, 9);
    expect(v!.emitNormal[1]).toBeCloseTo(0.30942637387763805, 9);
    expect(v!.emitNormal[2]).toBeCloseTo(0.9282791216329142, 9);
    expect(v!.throughput[0]).toBeCloseTo(14.13716694115407, 9);
    expect(v!.pdfHemi).toBeCloseTo(0.05005903659734805, 9);
    expect(v!.pdfJoint).toBeCloseTo(0.05005903659734805, 9);
  });
});
