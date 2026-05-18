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
import type { PassDispatchContext } from '../src/pipeline/Pass.js';

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

  // Cast through unknown — we only touch resources.ppg in the PPG pass paths.
  const resources = { ppg: ppgResources } as unknown as PassDispatchContext['resources'];

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
    wgX: 0, wgY: 0, wgX16: 0, wgY16: 0, halfWgX: 0, halfWgY: 0,
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
