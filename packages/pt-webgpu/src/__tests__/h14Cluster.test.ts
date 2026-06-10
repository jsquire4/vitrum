/**
 * H14-C/D/F cluster tests.
 *
 * H14-C: getRestirPtResultBuffer() is available on the Engine contract and
 *         returns null when restirPtReuse is off (or before the first frame).
 * H14-D: readOidnInputsFromTextures destroys all created buffers even when
 *         mapAsync rejects (device-loss / OOM simulation).
 * H14-F: buffer-ceiling console.warns fire at most once per engine instance
 *         regardless of how many frames hit the ceiling.
 */
import { beforeAll, describe, expect, it, vi } from 'vitest';
import { createPTEngine_WebGPU } from '../index.js';
import { readOidnInputsFromTextures } from '../denoise/rgba16fReadback.js';
import { GpuResources } from '../gpuResources.js';
import { installGpuConstStubs } from './gpuStub.js';

/** Install all WebGPU constant globals needed for these tests. */
function installAllGpuStubs() {
  installGpuConstStubs();
  const g = globalThis as unknown as { GPUMapMode?: { READ: number } };
  if (g.GPUMapMode == null) {
    g.GPUMapMode = { READ: 1 };
  }
}

function makeStubDevice(): GPUDevice {
  return {
    limits: {
      maxStorageBuffersPerShaderStage: 16,
      maxStorageTexturesPerShaderStage: 8,
    },
    createCommandEncoder: vi.fn(),
  } as unknown as GPUDevice;
}

// ── H14-C ─────────────────────────────────────────────────────────────────────
describe('H14-C: getRestirPtResultBuffer on the Engine contract', () => {
  it('method exists on the engine instance (optional Engine interface method)', async () => {
    const engine = await createPTEngine_WebGPU({ device: makeStubDevice() });
    // The method must be present — it is declared on the Engine interface.
    expect(typeof engine.getRestirPtResultBuffer).toBe('function');
    engine.dispose();
  });

  it('returns null when restirPtReuse is off (default)', async () => {
    const engine = await createPTEngine_WebGPU({ device: makeStubDevice() });
    expect(engine.getRestirPtResultBuffer!()).toBeNull();
    engine.dispose();
  });

  it('returns null before the first successful frame even when restirPtReuse:true on a full-tier device', async () => {
    const fullDevice = {
      limits: {
        // N-directional (2026-06-10): full tier needs 31 storage buffers (was 30); restirPtReuse adds 4 → 35.
        maxStorageBuffersPerShaderStage: 35,
        maxStorageTexturesPerShaderStage: 8,
      },
      createCommandEncoder: vi.fn(),
    } as unknown as GPUDevice;
    // restirPtReuse requires full tier; the buffer is null until a frame runs.
    const engine = await createPTEngine_WebGPU({ device: fullDevice, restirPtReuse: true });
    expect(engine.getRestirPtResultBuffer!()).toBeNull();
    engine.dispose();
  });
});

// ── H14-D ─────────────────────────────────────────────────────────────────────
describe('H14-D: readOidnInputsFromTextures destroys all buffers on mapAsync rejection', () => {
  beforeAll(() => {
    // GPUMapMode.READ is referenced as a global in rgba16fReadback.ts.
    installAllGpuStubs();
  });

  it('destroys all created buffers when mapAsync rejects', async () => {
    const destroyCalls: string[] = [];

    // Create mock buffers; mapAsync always rejects to simulate device loss.
    const makeBuf = (label: string) => ({
      mapAsync: vi.fn().mockRejectedValue(new Error('device lost')),
      getMappedRange: vi.fn(() => new ArrayBuffer(0)),
      unmap: vi.fn(),
      destroy: vi.fn(() => { destroyCalls.push(label); }),
    });

    let callCount = 0;
    const bufInstances = [makeBuf('color'), makeBuf('albedo'), makeBuf('normal')];

    const device = {
      createBuffer: vi.fn(() => bufInstances[callCount++] ?? bufInstances[0]),
      createCommandEncoder: vi.fn(() => ({
        copyTextureToBuffer: vi.fn(),
        finish: vi.fn(() => ({})),
      })),
      queue: { submit: vi.fn() },
    } as unknown as GPUDevice;

    const fakeTexture = () => ({}) as unknown as GPUTexture;

    await expect(
      readOidnInputsFromTextures(
        device,
        {
          color: fakeTexture(),
          albedo: fakeTexture(),
          normalDepth: fakeTexture(),
        },
        4,
        4,
      ),
    ).rejects.toThrow('device lost');

    // All three buffers must have been destroyed even though mapAsync rejected.
    expect(destroyCalls).toContain('color');
    expect(destroyCalls).toContain('albedo');
    expect(destroyCalls).toContain('normal');
  });

  it('each buffer is destroyed at most once (no double-destroy)', async () => {
    const destroyCounts = new Map<string, number>();
    let callCount = 0;

    // bytesPerRow for width=4 = 256 (aligned), height=4 → readSize=1024.
    const readSize = 256 * 4;
    const makeBuf = (label: string) => ({
      mapAsync: vi.fn().mockResolvedValue(undefined),
      getMappedRange: vi.fn(() => new ArrayBuffer(readSize)),
      unmap: vi.fn(),
      destroy: vi.fn(() => {
        destroyCounts.set(label, (destroyCounts.get(label) ?? 0) + 1);
      }),
    });

    const bufInstances = [makeBuf('color')];
    const device = {
      createBuffer: vi.fn(() => bufInstances[callCount++] ?? bufInstances[0]),
      createCommandEncoder: vi.fn(() => ({
        copyTextureToBuffer: vi.fn(),
        finish: vi.fn(() => ({})),
      })),
      queue: { submit: vi.fn() },
    } as unknown as GPUDevice;

    const fakeTexture = () => ({}) as unknown as GPUTexture;
    // Only pass color (no albedo or normalDepth) — one buffer created.
    await readOidnInputsFromTextures(device, { color: fakeTexture() }, 4, 4);

    // destroy should have been called exactly once.
    expect(destroyCounts.get('color')).toBe(1);
  });
});

// ── H14-F ─────────────────────────────────────────────────────────────────────
describe('H14-F: buffer-ceiling warns fire at most once per engine instance', () => {
  beforeAll(() => { installAllGpuStubs(); });

  it('BDPT eye-stack ceiling warn fires exactly once across 100 calls over the ceiling', () => {
    const warns: string[] = [];
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation((...args: unknown[]) => {
      warns.push(String(args[0]));
    });

    const device = {
      createBuffer: vi.fn(() => ({ destroy: vi.fn() })),
      createTexture: vi.fn(() => ({ createView: vi.fn(() => ({})), destroy: vi.fn() })),
      createCommandEncoder: vi.fn(() => ({ clearBuffer: vi.fn(), finish: vi.fn(() => ({})) })),
      queue: { submit: vi.fn() },
    } as unknown as GPUDevice;

    const gpu = new GpuResources(device, 'full', /* bdpt */ true);

    // ensureBdptEyeStack with a size beyond the 384 MiB cap triggers the warn.
    // BDPT_EYE_STACK_MAX_BYTES ~ 402,653,184 B; at 64 B/pixel a 2560x2560
    // frame uses 419,430,400 B (> cap). We use 10000x10000 to be safe.
    for (let i = 0; i < 100; i++) {
      gpu.ensureBdptEyeStack(10000, 10000, 8, /* bdptActive */ true);
    }

    const bdptWarns = warns.filter((w) => w.includes('BDPT eye-stack'));
    expect(bdptWarns.length).toBe(1);

    warnSpy.mockRestore();
  });

  it('ReSTIR-PT reservoir ceiling warn fires exactly once across 100 calls over the ceiling', () => {
    const warns: string[] = [];
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation((...args: unknown[]) => {
      warns.push(String(args[0]));
    });

    const device = {
      createBuffer: vi.fn(() => ({ destroy: vi.fn() })),
      createTexture: vi.fn(() => ({ createView: vi.fn(() => ({})), destroy: vi.fn() })),
      createCommandEncoder: vi.fn(() => ({ clearBuffer: vi.fn(), finish: vi.fn(() => ({})) })),
      queue: { submit: vi.fn(), writeBuffer: vi.fn() },
    } as unknown as GPUDevice;

    // restirPtReuse=true required for ensureReservoirBuffers to be non-trivial.
    const gpu = new GpuResources(device, 'full', /* bdpt */ false, /* restirPtReuse */ true);

    // RESTIR_PT_RESERVOIR_MAX_BYTES ~ 402 MiB; 10000x10000 exceeds it.
    for (let i = 0; i < 100; i++) {
      gpu.ensureReservoirBuffers(10000, 10000);
    }

    const rptWarns = warns.filter((w) => w.includes('ReSTIR-PT reservoir'));
    expect(rptWarns.length).toBe(1);

    warnSpy.mockRestore();
  });
});
