// W3-D7 — Pins the FrameOutput discriminated-union shape.
//
// Replaces the prior `{ primaryRadiance: BackendTexture | null;
// samplesAccumulated: number }` contract where `samplesAccumulated === 0`
// was an in-band sentinel for "skipped frame". The new shape:
//
//   FrameOutput = FrameSkipped | FrameRendered
//   FrameSkipped  = { kind: 'skipped';  reason: FrameSkipReason }
//   FrameRendered = { kind: 'rendered'; primaryRadiance: BackendTexture;
//                     samplesAccumulated: number; isConverged: boolean; ... }
//
// This test lives in @vitrum/engine (not @vitrum/core) because @vitrum/core
// intentionally has no vitest harness — its public surface is exercised
// transitively through the dependent packages, of which @vitrum/engine is
// the canonical façade.

import { describe, it, expect } from 'vitest';
import type {
  BackendTexture,
  FrameOutput,
  FrameSkipped,
  FrameRendered,
  FrameSkipReason,
} from '@vitrum/core';

describe('FrameOutput discriminated union (W3-D7)', () => {
  it('a skipped output is constructible with kind+reason only', () => {
    const out: FrameOutput = { kind: 'skipped', reason: 'fps-throttle' };
    expect(out.kind).toBe('skipped');
    if (out.kind === 'skipped') {
      expect(out.reason).toBe('fps-throttle');
    }
  });

  it('each documented FrameSkipReason literal is assignable', () => {
    const reasons: FrameSkipReason[] = [
      'fps-throttle',
      'paused',
      'no-scene',
      'pending-init',
      'no-swap-view',
      'rebuild',
      'disposed',
    ];
    for (const reason of reasons) {
      const out: FrameOutput = { kind: 'skipped', reason };
      expect(out.kind).toBe('skipped');
    }
  });

  it('reason is forward-compat — arbitrary strings are accepted', () => {
    // Backends may introduce new skip reasons without bumping the contract.
    const out: FrameOutput = { kind: 'skipped', reason: 'future-backend-specific-reason' };
    if (out.kind === 'skipped') {
      expect(typeof out.reason).toBe('string');
    }
  });

  it('a rendered output requires non-null primaryRadiance + the metadata fields', () => {
    const tex: BackendTexture = { __mock: 'gpu-texture-handle' };
    const out: FrameOutput = {
      kind: 'rendered',
      primaryRadiance: tex,
      samplesAccumulated: 4,
      isConverged: false,
    };
    expect(out.kind).toBe('rendered');
    if (out.kind === 'rendered') {
      expect(out.primaryRadiance).toBe(tex);
      expect(out.samplesAccumulated).toBe(4);
      expect(out.isConverged).toBe(false);
    }
  });

  it('rendered output accepts the optional G-buffer aux textures', () => {
    const out: FrameRendered = {
      kind: 'rendered',
      primaryRadiance: { __mock: 'primary' },
      normalDepth:     { __mock: 'normalDepth' },
      albedo:          { __mock: 'albedo' },
      variance:        { __mock: 'variance' },
      motionVectors:   { __mock: 'motionVectors' },
      samplesAccumulated: 1,
      isConverged: false,
    };
    expect(out.normalDepth).toBeDefined();
    expect(out.albedo).toBeDefined();
    expect(out.variance).toBeDefined();
    expect(out.motionVectors).toBeDefined();
  });

  it('switching on kind narrows to the variant-specific shape', () => {
    // This test would fail to typecheck (not just at runtime) if narrowing
    // regressed, which is the load-bearing property of the new contract.
    const samples: FrameOutput[] = [
      { kind: 'skipped', reason: 'paused' },
      {
        kind: 'rendered',
        primaryRadiance: { __mock: 't' },
        samplesAccumulated: 8,
        isConverged: true,
      },
    ];
    const summary: string[] = [];
    for (const s of samples) {
      switch (s.kind) {
        case 'skipped':
          // s is FrameSkipped here — has reason, no primaryRadiance.
          summary.push(`skip:${s.reason}`);
          break;
        case 'rendered':
          // s is FrameRendered here — primaryRadiance is non-null.
          summary.push(`render:${s.samplesAccumulated}`);
          break;
      }
    }
    expect(summary).toEqual(['skip:paused', 'render:8']);
  });

  it('FrameSkipped and FrameRendered are exposed as standalone types', () => {
    // Compile-time only: each variant is independently importable so backend
    // authors can return them directly without re-deriving via Extract<>.
    const a: FrameSkipped  = { kind: 'skipped',  reason: 'no-scene' };
    const b: FrameRendered = {
      kind: 'rendered',
      primaryRadiance: { __mock: 't' },
      samplesAccumulated: 1,
      isConverged: false,
    };
    expect(a.kind).toBe('skipped');
    expect(b.kind).toBe('rendered');
  });
});
