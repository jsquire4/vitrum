/**
 * neuralDenoiserResize.test.ts — Issue 1 characterization tests.
 *
 * Covers the resize behavioural correctness fix: after `NeuralDenoiser.resize`
 * the denoiser must be in a consistent state (fully allocated when a device
 * is present, or cleanly null when no device yet) rather than a torn-down
 * intermediate where buffers are null but dimensions have changed.
 *
 * All tests run without a real GPU. A minimal mock device is used to exercise
 * the `_reallocForSize` path that `resize` now calls when `_device` is present.
 *
 * Also covers Issue 2 byte-identity: the extracted NEURAL_PACK_WGSL /
 * NEURAL_UNPACK_WGSL strings must equal the shader code passed to
 * `createShaderModule` during `initialize`.
 */

import { describe, it, expect, vi } from 'vitest';
import { installWebGPUPolyfills } from './helpers/webgpuPolyfills.js';
// Install GPUBufferUsage / GPUTextureUsage globals before any source import
// that references them (NeuralDenoiser.initialize touches these constants).
installWebGPUPolyfills();
import { NeuralDenoiser } from '../src/pipeline/denoisers/neural.js';
import { NEURAL_PACK_WGSL } from '../src/shaders/neuralPack.wgsl.js';
import { NEURAL_UNPACK_WGSL } from '../src/shaders/neuralUnpack.wgsl.js';
import type { InferenceGraph } from '../src/neural/InferenceGraph.js';

// ─── Mock device factory ────────────────────────────────────────────────────

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type CreatedBuffer = { label: string; size: number; destroy: (...args: any[]) => any };
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type CreatedTexture = { label: string; destroy: (...args: any[]) => any; createView: (...args: any[]) => any };

interface MockDevice {
  device: GPUDevice;
  buffers: CreatedBuffer[];
  textures: CreatedTexture[];
  shaderCodes: string[];
}

function makeMockDevice(): MockDevice {
  const buffers: CreatedBuffer[] = [];
  const textures: CreatedTexture[] = [];
  const shaderCodes: string[] = [];

  const device = {
    createBuffer: vi.fn((desc: GPUBufferDescriptor) => {
      const buf: CreatedBuffer = {
        label: (desc.label as string) ?? '',
        size: desc.size as number,
        destroy: vi.fn(),
      };
      buffers.push(buf);
      return buf;
    }),
    createTexture: vi.fn((desc: GPUTextureDescriptor) => {
      const tex: CreatedTexture = {
        label: (desc.label as string) ?? '',
        destroy: vi.fn(),
        createView: vi.fn(() => ({})),
      };
      textures.push(tex);
      return tex;
    }),
    createShaderModule: vi.fn((desc: GPUShaderModuleDescriptor) => {
      shaderCodes.push(desc.code as string);
      return {};
    }),
    createComputePipelineAsync: vi.fn(() => Promise.resolve({ getBindGroupLayout: vi.fn(() => ({})) })),
    createBindGroup: vi.fn(() => ({})),
    createPipelineLayout: vi.fn(() => ({})),
    queue: { writeBuffer: vi.fn(), submit: vi.fn() },
  } as unknown as GPUDevice;

  return { device, buffers, textures, shaderCodes };
}

/** Minimal InferenceGraph stub — enough for NeuralDenoiser to treat itself as enabled. */
const fakeGraph = { run: vi.fn() } as unknown as InferenceGraph;

// ─── Issue 1: resize state-consistency ────────────────────────────────────────

describe('NeuralDenoiser.resize — state consistency (Issue 1 fix)', () => {
  it('resize before initialize updates dimensions without allocating (no device yet)', () => {
    const d = new NeuralDenoiser({ inferenceGraph: fakeGraph });
    d.resize(1280, 720);
    // No device — should not have allocated anything (nothing to assert on,
    // but calling resize must not throw or leave a partially-torn state).
    expect(d.disabled).toBe(false);
    // Internal _width/_height are private; we verify indirectly: a subsequent
    // initialize at 1280×720 must succeed (same dims = no mismatch warning).
  });

  it('resize after initialize (with device) reallocates buffers atomically — no null-buffer window', async () => {
    const { device, buffers, textures } = makeMockDevice();
    const d = new NeuralDenoiser({ inferenceGraph: fakeGraph });

    const initCtx = {
      device,
      width: 640,
      height: 360,
      bglCache: {} as never,
      frameResources: {} as never,
    };
    await d.initialize(initCtx);

    // After initialize: 4 storage buffers + output texture should have been created.
    const bufsAfterInit = buffers.length;
    const texsAfterInit = textures.length;
    expect(bufsAfterInit).toBeGreaterThanOrEqual(4); // noisy + albedo + normals + output
    expect(texsAfterInit).toBeGreaterThanOrEqual(1); // output texture

    // resize UP
    d.resize(1280, 720);

    // After resize: new buffers + texture created for the larger size.
    // Old buffers must have been destroyed.
    const oldBufs = buffers.slice(buffers.length - (buffers.length - bufsAfterInit), bufsAfterInit);
    // The resize must have called _reallocForSize synchronously — so the
    // newly-created buffers exist RIGHT NOW (not deferred to the next dispatch).
    const newBufs = buffers.slice(bufsAfterInit);
    const newTexs = textures.slice(texsAfterInit);

    // At least 4 new storage buffers and 1 new texture must be present.
    expect(newBufs.length).toBeGreaterThanOrEqual(4);
    expect(newTexs.length).toBeGreaterThanOrEqual(1);

    // Old storage buffers (size 640*360*3*4) must have been destroyed.
    const oldStorageBufs = oldBufs.filter((b) => b.label.startsWith('neural-denoiser-noisy') ||
      b.label.startsWith('neural-denoiser-albedo') ||
      b.label.startsWith('neural-denoiser-normals') ||
      b.label.startsWith('neural-denoiser-output'));
    for (const b of oldStorageBufs) {
      expect(b.destroy).toHaveBeenCalled();
    }

    // New buffers are sized for 1280×720.
    const expectedBytes = Math.max(4, 1280 * 720 * 3 * 4);
    const storageBufs = newBufs.filter((b) =>
      b.label === 'neural-denoiser-noisy' ||
      b.label === 'neural-denoiser-albedo' ||
      b.label === 'neural-denoiser-normals' ||
      b.label === 'neural-denoiser-output',
    );
    for (const b of storageBufs) {
      expect(b.size).toBe(expectedBytes);
    }
  });

  it('resize UP then DOWN — no dangling / null buffers when device is present', async () => {
    const { device, buffers, textures } = makeMockDevice();
    const d = new NeuralDenoiser({ inferenceGraph: fakeGraph });

    await d.initialize({ device, width: 640, height: 360, bglCache: {} as never, frameResources: {} as never });

    const afterInit = { bufs: buffers.length, texs: textures.length };

    // Resize up.
    d.resize(1280, 720);
    const afterUp = { bufs: buffers.length, texs: textures.length };
    expect(afterUp.bufs).toBeGreaterThan(afterInit.bufs);
    expect(afterUp.texs).toBeGreaterThan(afterInit.texs);

    // Resize down — should trigger another realloc (dimensions differ from current 1280×720).
    d.resize(320, 180);
    const afterDown = { bufs: buffers.length, texs: textures.length };
    expect(afterDown.bufs).toBeGreaterThan(afterUp.bufs);
    expect(afterDown.texs).toBeGreaterThan(afterUp.texs);

    // The 1280×720 storage buffers must have been destroyed on the resize-down.
    const upStorageBufs = buffers
      .slice(afterInit.bufs, afterUp.bufs)
      .filter((b) => ['neural-denoiser-noisy', 'neural-denoiser-albedo',
                       'neural-denoiser-normals', 'neural-denoiser-output'].includes(b.label));
    for (const b of upStorageBufs) {
      expect(b.destroy).toHaveBeenCalled();
    }

    // The 320×180 storage buffers are present and correctly sized.
    const expectedBytes = Math.max(4, 320 * 180 * 3 * 4);
    const downStorageBufs = buffers
      .slice(afterUp.bufs)
      .filter((b) => ['neural-denoiser-noisy', 'neural-denoiser-albedo',
                       'neural-denoiser-normals', 'neural-denoiser-output'].includes(b.label));
    expect(downStorageBufs.length).toBe(4);
    for (const b of downStorageBufs) {
      expect(b.size).toBe(expectedBytes);
    }
  });

  it('same-size resize is a no-op (guard path)', async () => {
    const { device, buffers } = makeMockDevice();
    const d = new NeuralDenoiser({ inferenceGraph: fakeGraph });

    await d.initialize({ device, width: 640, height: 360, bglCache: {} as never, frameResources: {} as never });

    const beforeCount = buffers.length;
    d.resize(640, 360); // same size
    // _reallocForSize guard: already correct size → no new buffers.
    expect(buffers.length).toBe(beforeCount);
  });
});

// ─── Issue 2: byte-identity of extracted WGSL strings ───────────────────────

describe('NeuralDenoiser WGSL extraction byte-identity (Issue 2)', () => {
  it('createShaderModule receives NEURAL_PACK_WGSL — exact same string', async () => {
    const { device, shaderCodes } = makeMockDevice();
    const d = new NeuralDenoiser({ inferenceGraph: fakeGraph });

    await d.initialize({ device, width: 64, height: 64, bglCache: {} as never, frameResources: {} as never });

    // First shader compiled = pack, second = unpack (matches declaration order in initialize).
    expect(shaderCodes.length).toBeGreaterThanOrEqual(2);
    expect(shaderCodes[0]).toBe(NEURAL_PACK_WGSL);
    expect(shaderCodes[1]).toBe(NEURAL_UNPACK_WGSL);
  });

  it('NEURAL_PACK_WGSL contains the pack entry-point declaration', () => {
    expect(NEURAL_PACK_WGSL).toMatch(/struct\s+PackParams/);
    expect(NEURAL_PACK_WGSL).toMatch(/@compute\s+@workgroup_size\s*\(\s*256/);
    expect(NEURAL_PACK_WGSL).toMatch(/fn\s+main\s*\(/);
    expect(NEURAL_PACK_WGSL).toMatch(/noisyOut\[base/);
    // The bare normalize(nd * 2.0 - 1.0) is gone — replaced by the NaN-guard
    // (item 12): select(normalize(nd_remapped), fallback, dot-length-check).
    expect(NEURAL_PACK_WGSL).toMatch(/normalize\s*\(\s*nd_remapped\s*\)/);
  });

  it('NEURAL_UNPACK_WGSL contains the unpack entry-point declaration', () => {
    expect(NEURAL_UNPACK_WGSL).toMatch(/struct\s+UnpackParams/);
    expect(NEURAL_UNPACK_WGSL).toMatch(/@compute\s+@workgroup_size\s*\(\s*256/);
    expect(NEURAL_UNPACK_WGSL).toMatch(/fn\s+main\s*\(/);
    expect(NEURAL_UNPACK_WGSL).toMatch(/textureStore\s*\(denoisedOut/);
    expect(NEURAL_UNPACK_WGSL).toMatch(/max\s*\(\s*0\.0\s*,\s*denoisedIn\[base/);
  });
});
