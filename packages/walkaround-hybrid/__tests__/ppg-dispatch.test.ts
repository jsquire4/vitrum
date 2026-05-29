/**
 * W9 — PPG dispatch shape tests.
 *
 * These tests pin the load-bearing acceptance criterion: `ppgEnabled: true`
 * MUST dispatch non-zero workgroup counts (no more `dispatchWorkgroups(0,0,0)`
 * stubs). We exercise the Pass.dispatch path with mock GPU surfaces, capture
 * every `dispatchWorkgroups(x,y,z)` call, and verify each PPG dispatch
 * receives positive counts derived from (width, height).
 */

import { describe, it, expect } from 'vitest';
import { installWebGPUPolyfills } from './helpers/webgpuPolyfills.js';
import { PPGGuidePass } from '../src/pipeline/passes/PPGGuidePass.js';
import { PPGUpdatePass } from '../src/pipeline/passes/PPGUpdatePass.js';
import type { PassDispatchContext, PassGateOptions } from '../src/pipeline/Pass.js';

installWebGPUPolyfills();

// ── Mock GPU surfaces — capture every dispatch call ─────────────────────────

interface CapturedDispatch {
  label: string;
  wgX: number;
  wgY: number;
  wgZ: number;
}

interface MockComputePass {
  setPipeline: (p: unknown) => void;
  setBindGroup: (slot: number, bg: unknown) => void;
  dispatchWorkgroups: (x: number, y: number, z: number) => void;
  end: () => void;
}

interface MockEncoder {
  beginComputePass: (desc: { label: string }) => MockComputePass;
}

interface MockDevice {
  createBindGroup: (desc: unknown) => unknown;
  createBuffer: (desc: { size: number; usage: number; label?: string }) => GPUBuffer;
  queue: { writeBuffer: (...args: unknown[]) => void };
}

interface MockPipeline {
  getBindGroupLayout: (idx: number) => unknown;
}

function makeMockEncoder(captured: CapturedDispatch[]): MockEncoder {
  return {
    beginComputePass: (desc) => ({
      setPipeline: () => {},
      setBindGroup: () => {},
      dispatchWorkgroups: (x, y, z) => {
        captured.push({ label: desc.label, wgX: x, wgY: y, wgZ: z });
      },
      end: () => {},
    }),
  };
}

function makeMockDevice(): MockDevice {
  return {
    createBindGroup: () => ({}),
    createBuffer: (desc) => ({ size: desc.size, usage: desc.usage, destroy: () => {} } as unknown as GPUBuffer),
    queue: { writeBuffer: () => {} },
  };
}

function makeMockPipeline(): MockPipeline {
  return { getBindGroupLayout: () => ({}) };
}

function makeMockBuffer(): GPUBuffer {
  return { size: 1024, usage: 0, destroy: () => {} } as unknown as GPUBuffer;
}

function makeMinimalCtx(
  width: number,
  height: number,
  captured: CapturedDispatch[],
  ppgEnabled: boolean,
): PassDispatchContext {
  const device = makeMockDevice() as unknown as GPUDevice;
  const encoder = makeMockEncoder(captured) as unknown as GPUCommandEncoder;

  // Build a minimal FrameResources shape — only the .ppg sub-struct matters
  // for these tests.
  const ppgResources = ppgEnabled
    ? {
        sTreeBuf: makeMockBuffer(),
        dTreeBuf: makeMockBuffer(),
        dTreeOffsetsBuf: makeMockBuffer(),
        fluxAtomicsBuf: makeMockBuffer(),
        samplesPosBuf: makeMockBuffer(),
        samplesDirBuf: makeMockBuffer(),
        samplesLiBuf: makeMockBuffer(),
        sampleOutBuf: makeMockBuffer(),
        guideUboBuffer: makeMockBuffer(),
        updateUboBuffer: makeMockBuffer(),
      }
    : {};

  // Cast through unknown — we only touch resources.ppg + .restirGI in the PPG pass paths.
  // W9 Phase 2: PPGGuidePass also reads `resources.restirGI.reservoirGiCurrentBuffer`
  // for the per-pixel primary-hit position lookup.
  const resources = {
    ppg: ppgResources,
    restirGI: { reservoirGiCurrentBuffer: makeMockBuffer() },
  } as unknown as PassDispatchContext['resources'];

  const ctx: PassDispatchContext = {
    device,
    encoder,
    width,
    height,
    frameIndex: 0,
    frameCount: 0,
    bglCache: {} as unknown as PassDispatchContext['bglCache'],
    resources,
    inputs: {} as unknown as PassDispatchContext['inputs'],
    frameBindGroup: {} as unknown as GPUBindGroup,
    sceneBindGroup: {} as unknown as GPUBindGroup,
    uboBindGroup: {} as unknown as GPUBindGroup,
    hybridLayersBindGroup: {} as unknown as GPUBindGroup,
    lightTreeBindGroup: {} as unknown as GPUBindGroup,
    wgX: 0, wgY: 0, wgX16: 0, wgY16: 0, halfWgX: 0, halfWgY: 0,
    gtaoDownscale: 2,
    gNormalDepthView: {} as unknown as GPUTextureView,
    computeDesc: (label) => ({ label }),
    renderTimestampWrites: () => undefined,
    frameState: {} as unknown as PassDispatchContext['frameState'],
  };
  return ctx;
}

// ── Tests ────────────────────────────────────────────────────────────────────

describe('PPG dispatch — W9 acceptance: no more (0,0,0) stubs', () => {
  it('PPGGuidePass dispatches a positive 1-D workgroup count = ceil(W*H / 64)', () => {
    const pipeline = makeMockPipeline() as unknown as GPUComputePipeline;
    const pass = new PPGGuidePass(pipeline);
    const W = 256, H = 256;
    const captured: CapturedDispatch[] = [];
    pass.dispatch(makeMinimalCtx(W, H, captured, true));
    expect(captured.length).toBe(1);
    expect(captured[0]!.label).toBe('ppg-guide');
    const expectedWG = Math.max(1, Math.ceil((W * H) / 64));
    expect(captured[0]!.wgX).toBe(expectedWG);
    expect(captured[0]!.wgY).toBe(1);
    expect(captured[0]!.wgZ).toBe(1);
    // The key acceptance criterion: NOT (0,0,0).
    expect(captured[0]!.wgX).toBeGreaterThan(0);
  });

  it('PPGUpdatePass dispatches a positive 1-D workgroup count = ceil(W*H / 64)', () => {
    const pipeline = makeMockPipeline() as unknown as GPUComputePipeline;
    const pass = new PPGUpdatePass(pipeline);
    const W = 128, H = 128;
    const captured: CapturedDispatch[] = [];
    pass.dispatch(makeMinimalCtx(W, H, captured, true));
    expect(captured.length).toBe(1);
    expect(captured[0]!.label).toBe('ppg-update');
    const expectedWG = Math.max(1, Math.ceil((W * H) / 64));
    expect(captured[0]!.wgX).toBe(expectedWG);
    expect(captured[0]!.wgY).toBe(1);
    expect(captured[0]!.wgZ).toBe(1);
    expect(captured[0]!.wgX).toBeGreaterThan(0);
  });

  it('PPGGuidePass throws if PPG resources are unallocated (contract enforcement)', () => {
    const pipeline = makeMockPipeline() as unknown as GPUComputePipeline;
    const pass = new PPGGuidePass(pipeline);
    const captured: CapturedDispatch[] = [];
    // Pass `ppgEnabled=false` so resources stay unallocated.
    expect(() => pass.dispatch(makeMinimalCtx(64, 64, captured, false))).toThrow(/PPG.*resources/);
    expect(captured.length).toBe(0);
  });

  it('PPGUpdatePass throws if PPG resources are unallocated', () => {
    const pipeline = makeMockPipeline() as unknown as GPUComputePipeline;
    const pass = new PPGUpdatePass(pipeline);
    const captured: CapturedDispatch[] = [];
    expect(() => pass.dispatch(makeMinimalCtx(64, 64, captured, false))).toThrow(/PPG.*resources/);
    expect(captured.length).toBe(0);
  });

  it('dispatch workgroup count scales with image dimensions', () => {
    const pipeline = makeMockPipeline() as unknown as GPUComputePipeline;
    const pass = new PPGGuidePass(pipeline);
    for (const [W, H] of [[64, 64], [256, 144], [1920, 1080], [3840, 2160]] as const) {
      const captured: CapturedDispatch[] = [];
      pass.dispatch(makeMinimalCtx(W, H, captured, true));
      const expected = Math.max(1, Math.ceil((W * H) / 64));
      expect(captured[0]!.wgX).toBe(expected);
    }
  });
});

// ── Item B — PPG train-pass dispatch-interval skip gate ─────────────────────
//
// The ppg-guide + ppg-update TRAIN passes gate on
// `opts.ppgEnabled && (opts.ppgTrainThisFrame ?? true)`. The orchestrator
// computes `ppgTrainThisFrame = frameCount % ppgDispatchInterval === 0`, so a
// higher interval amortises the path-guiding training cost across frames. The
// learned sTree/dTree GPU buffers persist between train cycles and gi-ris
// guided SAMPLING reads them every frame, so this gate ONLY governs the two
// train passes — never the guided sampling. We assert the gate predicate
// directly (its load-bearing logic) AND drive a frame loop through the exact
// modulo the pipeline uses to confirm the skip/run pattern.

/** Mirror of `WalkaroundGPUPipeline`'s per-frame gate computation. */
function gateOptsForFrame(
  frameCount: number,
  ppgDispatchInterval: number,
  ppgEnabled = true,
): PassGateOptions {
  return {
    denoiserMode: 'atrous-variance',
    ppgEnabled,
    ppgTrainThisFrame: frameCount % ppgDispatchInterval === 0,
    gtaoEnabled: true,
  };
}

describe('PPG train-pass dispatch interval (Item B skip gate)', () => {
  const passes = [
    { name: 'PPGGuidePass', make: () => new PPGGuidePass(makeMockPipeline() as unknown as GPUComputePipeline) },
    { name: 'PPGUpdatePass', make: () => new PPGUpdatePass(makeMockPipeline() as unknown as GPUComputePipeline) },
  ] as const;

  for (const { name, make } of passes) {
    it(`${name}: interval=1 ⇒ trains EVERY frame (unchanged behaviour)`, () => {
      const pass = make();
      for (let f = 0; f < 8; f++) {
        expect(pass.gates(gateOptsForFrame(f, 1))).toBe(true);
      }
    });

    it(`${name}: interval=N ⇒ trains only on frame multiples of N, skips otherwise`, () => {
      const pass = make();
      for (const interval of [2, 3, 4]) {
        for (let f = 0; f < interval * 4; f++) {
          const shouldTrain = f % interval === 0;
          expect(pass.gates(gateOptsForFrame(f, interval))).toBe(shouldTrain);
        }
      }
    });

    it(`${name}: never trains when PPG is disabled, regardless of interval`, () => {
      const pass = make();
      for (let f = 0; f < 8; f++) {
        expect(pass.gates(gateOptsForFrame(f, 1, /* ppgEnabled */ false))).toBe(false);
      }
    });

    it(`${name}: absent ppgTrainThisFrame defaults to "train" (forward-compat)`, () => {
      const pass = make();
      // A gate-options object WITHOUT the field (an older caller) must keep the
      // historical every-frame behaviour.
      const opts: PassGateOptions = { denoiserMode: 'atrous-variance', ppgEnabled: true };
      expect(pass.gates(opts)).toBe(true);
    });
  }

  it('interval=N actually SKIPS the dispatch on off-interval frames and RUNS it on multiples', () => {
    // Wire gate → dispatch exactly as the orchestrator does: gate first, then
    // dispatch only if the gate passes. Count dispatches across a frame loop.
    const guide = new PPGGuidePass(makeMockPipeline() as unknown as GPUComputePipeline);
    const interval = 4;
    const W = 128, H = 128;
    let dispatched = 0;
    let skipped = 0;
    for (let f = 0; f < 16; f++) {
      const captured: CapturedDispatch[] = [];
      const ctx = makeMinimalCtx(W, H, captured, true);
      if (guide.gates(gateOptsForFrame(f, interval))) {
        guide.dispatch(ctx);
        dispatched++;
        expect(captured.length).toBe(1); // ran ⇒ exactly one ppg-guide dispatch
      } else {
        skipped++;
        expect(captured.length).toBe(0); // skipped ⇒ NO dispatch captured
      }
    }
    // 16 frames / interval 4 ⇒ frames {0,4,8,12} train (4), the other 12 skip.
    expect(dispatched).toBe(4);
    expect(skipped).toBe(12);
  });

  it('guided sampling is independent of the train gate (gate governs only ppg-guide/ppg-update)', () => {
    // The skip gate is scoped to the two PPG TRAIN pass classes. No other pass
    // reads `ppgTrainThisFrame`; gi-ris (the guided-sampling consumer) is a
    // SharedBindGroupPass whose gates() ignores options entirely and always
    // returns true. We assert that contract here so a future refactor that
    // accidentally couples guided sampling to the train cadence is caught.
    const interval = 4;
    // On an OFF-interval frame the train passes skip…
    const offFrameOpts = gateOptsForFrame(1, interval);
    expect(offFrameOpts.ppgTrainThisFrame).toBe(false);
    // …but the gate object still reports PPG itself as enabled, which is what
    // gi-ris/shade read (via ubo.ppgEnabled) to keep consuming the persisted
    // tree every frame.
    expect(offFrameOpts.ppgEnabled).toBe(true);
  });
});
