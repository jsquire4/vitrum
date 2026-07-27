/**
 * H1 regression — `setProbeUpdateDivisor` must drive the ACTUAL round-robin
 * stride that builds the per-frame active-probe set, not just a dead UBO field.
 *
 * Before the fix: `DDGI.updateFrame` hardcoded `STRIDE = 8` when calling
 * `ProbeUpdatePass.runFrame(adapter, offset, stride)`, and `setProbeUpdateDivisor`
 * formerly only forwarded a value into the now-removed `probesPerFrame` UBO field
 * reads (the kernels iterate `arrayLength(&activeProbes)`). So the quality
 * preset's `ddgiUpdateDivisor` knob was a no-op: every tier ran an 8-frame
 * cadence regardless.
 *
 * These tests pin the load-bearing behaviour: the divisor set on DDGI is the
 * stride passed to `runFrame`, and that stride determines how many probes are
 * marked active per frame (`ceil(probeCount / stride)` for the first stratum).
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { Scene } from '@vitrum/core';
import { DDGI } from '../DDGI.js';
import { ProbeUpdatePass } from '../probeUpdatePass.js';

/** A minimal core scene with one cube-ish mesh so SceneBvh.updateFromCore()
 *  yields a non-degenerate boundingBox and ProbeGrid.computeFromBounds() allocates the
 *  3×3×3-minimum probe grid (27 probes) — enough to observe stratum sizes. */
function makeBoxScene(): Scene {
  return {
    primitives: [{
      kind: 'mesh',
      id: 'box',
      positions: new Float32Array([
        -1, -1, -1,
         1, -1, -1,
        -1,  1, -1,
         1,  1,  1,
      ]),
      normals: new Float32Array([
        0, 0, 1,
        0, 0, 1,
        0, 0, 1,
        0, 0, 1,
      ]),
      indices: new Uint32Array([0, 1, 2, 1, 3, 2]),
      material: { baseColor: [1, 1, 1], roughness: 1, metallic: 0 },
    }],
    emitters: [],
    environment: { kind: 'none' },
  };
}

/** The active-probe stratum the round-robin builds for (offset, stride):
 *  `{ i : offset ≤ i < probeCount, i ≡ offset (mod stride) }`. Mirrors the
 *  loop in ProbeUpdatePass.runFrame. */
function stratumSize(probeCount: number, offset: number, stride: number): number {
  let n = 0;
  for (let i = offset; i < probeCount; i += stride) n++;
  return n;
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe('H1 — probe-update divisor drives the round-robin stride', () => {
  /** Drive one DDGI frame with init/runFrame stubbed, capturing the
   *  (offset, stride) the round-robin passes to ProbeUpdatePass.runFrame. */
  async function captureRunFrameArgs(divisor: number | undefined): Promise<{
    calls: Array<{ offset: number; stride: number }>;
    probeCount: number;
  }> {
    // init() → resolve true so DDGI marks _gpuOk and reaches the round-robin.
    vi.spyOn(ProbeUpdatePass.prototype, 'init').mockResolvedValue(true);
    const calls: Array<{ offset: number; stride: number }> = [];
    vi.spyOn(ProbeUpdatePass.prototype, 'runFrame').mockImplementation(
      async (_adapter, offset: number, stride: number) => {
        calls.push({ offset, stride });
        return true;
      },
    );

    const ddgi = new DDGI();
    if (divisor !== undefined) ddgi.setProbeUpdateDivisor(divisor);
    const scene = makeBoxScene();
    // A fake device object is enough — init() is stubbed, so the device is
    // never touched; DDGI only needs a truthy `device` to build its adapter.
    await ddgi.updateFrame({ coreScene: scene, device: {} as unknown as GPUDevice, enabled: true });
    const probeCount = ddgi.probeCount;
    ddgi.dispose();
    return { calls, probeCount };
  }

  it('default (no divisor set) keeps the historical stride = 8 cadence', async () => {
    const { calls } = await captureRunFrameArgs(undefined);
    expect(calls).toHaveLength(1);
    expect(calls[0]!.stride).toBe(8);
    expect(calls[0]!.offset).toBe(0); // first frame ⇒ stratum 0
  });

  it.each([4, 8, 16])('setProbeUpdateDivisor(%i) makes runFrame use that stride', async (d) => {
    const { calls } = await captureRunFrameArgs(d);
    expect(calls).toHaveLength(1);
    expect(calls[0]!.stride).toBe(d);
  });

  it('a smaller divisor activates MORE probes per frame than a larger one', async () => {
    const small = await captureRunFrameArgs(4);
    const large = await captureRunFrameArgs(16);
    // Same scene ⇒ same probe count; the divisor is the only difference.
    expect(small.probeCount).toBe(large.probeCount);
    expect(small.probeCount).toBeGreaterThan(0);

    const smallStratum = stratumSize(small.probeCount, small.calls[0]!.offset, small.calls[0]!.stride);
    const largeStratum = stratumSize(large.probeCount, large.calls[0]!.offset, large.calls[0]!.stride);
    // divisor=4 updates ~1/4 of probes; divisor=16 updates ~1/16 — strictly
    // more active probes per frame at the smaller divisor (the GI-response
    // lever the preset is meant to expose).
    expect(smallStratum).toBeGreaterThan(largeStratum);
  });

  it('rejects a non-positive divisor instead of poisoning the cadence', async () => {
    await expect(captureRunFrameArgs(0)).rejects.toThrow(
      'DDGI probe update divisor must be a positive safe integer.',
    );
  });
});
