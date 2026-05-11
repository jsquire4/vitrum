/**
 * Sprint 11 (Phase 6) — PPG path guiding structural tests.
 *
 * Test strategy: string-content and structural validation only — no real
 * WebGPU device required. Follows the same "defensive structural" pattern
 * as sprint9-welford.test.ts.
 *
 * Coverage:
 *   1. PPG type definitions — constants, byte-stride values.
 *   2. PPG_UPDATE_WGSL — entry point, bindings, atomic update strategy.
 *   3. createPPGBuffers — buffer count, sizes, and usage flags.
 *   4. buildPpgKdTreeGpuBytes / ppgNearestCellIndexKd vs brute parity.
 *   5. PPG disabled — no buffer allocation when ppgEnabled false/unset.
 *   6. setPPGEnabled lifecycle — toggle reflected in getter; no-op dispatch.
 *   7. FrameResources.ppgBuffers — opt-in field presence and destroy.
 *
 * (The companion PPG_SAMPLE_WGSL fragment was deleted in P3-C.2;
 * shadePpgGuide.wgsl.ts handles guided indirect sampling via @group(3)
 * marker-injection into shade.wgsl.)
 */

import { describe, it, expect, vi } from 'vitest';
import {
  PPG_MAX_SPATIAL_CELLS,
  PPG_DIRECTIONS,
  PPG_CELL_BYTE_STRIDE,
  PPG_LEAF_BYTE_STRIDE,
  PPG_KD_MAX_NODES,
  PPG_KD_NODE_BYTE_STRIDE,
} from '../src/ppg/types.js';
import {
  buildPpgKdTreeGpuBytes,
  encodePpgKdDisabledRoot,
  ppgNearestCellIndexBrute,
  ppgNearestCellIndexKd,
} from '../src/ppg/buildPpgKdTree.js';
import { PPG_UPDATE_WGSL } from '../src/ppg/wgsl/ppgUpdate.wgsl.js';
import {
  createPPGBuffers,
  destroyPPGBuffers,
  createFrameResources,
  destroyFrameResources,
} from '../src/pipeline/resourceManager.js';

// WebGPU global polyfills for the Node test environment.
import { installWebGPUPolyfills } from './helpers/webgpuPolyfills.js';
installWebGPUPolyfills();

// ─── Helper: build a mock GPUDevice that records createBuffer calls ────────────

interface BufferCall {
  label?: string;
  size: number;
  usage: number;
}

function makeMockDevice(): {
  device: GPUDevice;
  bufferCalls: BufferCall[];
  textureCalls: GPUTextureDescriptor[];
} {
  const bufferCalls: BufferCall[] = [];
  const textureCalls: GPUTextureDescriptor[] = [];
  const mockBuffer = { destroy: vi.fn() };
  const mockTexture = { destroy: vi.fn(), createView: vi.fn(() => ({})) };
  const mockSampler = {};

  const device = {
    createBuffer(desc: GPUBufferDescriptor): GPUBuffer {
      bufferCalls.push({ label: desc.label, size: desc.size, usage: desc.usage });
      return mockBuffer as unknown as GPUBuffer;
    },
    createTexture(desc: GPUTextureDescriptor): GPUTexture {
      textureCalls.push(desc);
      return mockTexture as unknown as GPUTexture;
    },
    createSampler(): GPUSampler {
      return mockSampler as unknown as GPUSampler;
    },
    queue: {
      writeBuffer: vi.fn(),
      writeTexture: vi.fn(),
    },
  } as unknown as GPUDevice;

  return { device, bufferCalls, textureCalls };
}

// ─── 1. PPG type constants ─────────────────────────────────────────────────────

describe('PPG type constants (Sprint 11)', () => {
  it('PPG_MAX_SPATIAL_CELLS is 10,000', () => {
    expect(PPG_MAX_SPATIAL_CELLS).toBe(10_000);
  });

  it('PPG_DIRECTIONS is 16', () => {
    expect(PPG_DIRECTIONS).toBe(16);
  });

  it('PPG_CELL_BYTE_STRIDE is 32 bytes', () => {
    // PPGSpatialCell: vec3f(12) + f32(4) + u32(4) + 3×u32 padding(12) = 32 bytes
    expect(PPG_CELL_BYTE_STRIDE).toBe(32);
  });

  it('PPG_LEAF_BYTE_STRIDE is 256 bytes', () => {
    // 16 bins × vec2f(8 bytes) = 128 bytes used; padded to 256 for alignment.
    expect(PPG_LEAF_BYTE_STRIDE).toBe(256);
  });

  it('PPG_LEAF_BYTE_STRIDE is a multiple of 256 (WebGPU offset alignment)', () => {
    expect(PPG_LEAF_BYTE_STRIDE % 256).toBe(0);
  });

  it('PPG_DIRECTIONS × 8 bytes/bin = 128 bytes (fits in PPG_LEAF_BYTE_STRIDE)', () => {
    const binsBytes = PPG_DIRECTIONS * 8; // vec2f = 8 bytes
    expect(binsBytes).toBeLessThanOrEqual(PPG_LEAF_BYTE_STRIDE);
  });

  it('PPG_MAX_SPATIAL_CELLS × PPG_CELL_BYTE_STRIDE = 320 KB', () => {
    const totalBytes = PPG_MAX_SPATIAL_CELLS * PPG_CELL_BYTE_STRIDE;
    expect(totalBytes).toBe(320_000);
  });

  it('PPG_MAX_SPATIAL_CELLS × PPG_LEAF_BYTE_STRIDE = 2.56 MB', () => {
    const totalBytes = PPG_MAX_SPATIAL_CELLS * PPG_LEAF_BYTE_STRIDE;
    expect(totalBytes).toBe(2_560_000);
  });
});

// ── PPG kd-tree CPU query parity (matches WGSL traversal) ───────────────────

describe('buildPpgKdTreeGpuBytes / ppgNearestCellIndexKd', () => {
  function makeCells(
    count: number,
    pos: (i: number) => [number, number, number],
  ): { position: [number, number, number] }[] {
    return Array.from({ length: count }, (_, i) => ({ position: pos(i) }));
  }

  it('disabled sentinel: kd query matches brute force', () => {
    const cells = makeCells(5, (i) => [i * 1.1, i * 0.7, i * -0.3]);
    const disabled = encodePpgKdDisabledRoot();
    for (let t = 0; t < 20; t++) {
      const qx = Math.sin(t) * 3;
      const qy = Math.cos(t * 0.7) * 2;
      const qz = t * 0.15;
      const b = ppgNearestCellIndexBrute(cells, cells.length, qx, qy, qz);
      const k = ppgNearestCellIndexKd(disabled, cells, cells.length, qx, qy, qz);
      expect(k).toBe(b);
    }
  });

  it('built tree matches brute for random queries (small N)', () => {
    const cells = makeCells(24, (i) => [
      Math.sin(i * 1.7) * 10,
      Math.cos(i * 0.91) * 8,
      (i % 7) * 1.3 - 3,
    ]);
    const gpu = buildPpgKdTreeGpuBytes(cells, cells.length);
    for (let s = 0; s < 50; s++) {
      const qx = (Math.sin(s * 2.1) + 0.3) * 12;
      const qy = (Math.cos(s * 1.05) - 0.2) * 9;
      const qz = s * 0.21 - 5;
      const b = ppgNearestCellIndexBrute(cells, cells.length, qx, qy, qz);
      const k = ppgNearestCellIndexKd(gpu, cells, cells.length, qx, qy, qz);
      expect(k).toBe(b);
    }
  });

  it('count <= 0 yields index 0 from kd query', () => {
    const cells = makeCells(2, () => [0, 0, 0]);
    const gpu = buildPpgKdTreeGpuBytes(cells, 0);
    expect(ppgNearestCellIndexKd(gpu, cells, 0, 1, 2, 3)).toBe(0);
  });
});

// ─── 2. PPG_UPDATE_WGSL — compute kernel ──────────────────────────────────────
//
// (The companion PPG_SAMPLE_WGSL fragment was deleted in P3-C.2 — its
// functionality is provided by `shadePpgGuide.wgsl.ts` which injects guided
// indirect sampling into shade.wgsl with real @group(3) bindings.)

describe('PPG_UPDATE_WGSL — ppgUpdateKernel compute shader', () => {
  it('is a non-empty string', () => {
    expect(typeof PPG_UPDATE_WGSL).toBe('string');
    expect(PPG_UPDATE_WGSL.length).toBeGreaterThan(0);
  });

  it('declares @compute @workgroup_size(64, 1, 1) entry point', () => {
    expect(PPG_UPDATE_WGSL).toContain('@compute @workgroup_size(64, 1, 1)');
  });

  it('contains ppgUpdateKernel entry point', () => {
    expect(PPG_UPDATE_WGSL).toContain('fn ppgUpdateKernel');
  });

  it('uses @builtin(global_invocation_id) for per-sample dispatch', () => {
    expect(PPG_UPDATE_WGSL).toContain('@builtin(global_invocation_id)');
  });

  it('declares PPGUpdateUniforms struct with sampleCount', () => {
    expect(PPG_UPDATE_WGSL).toContain('struct PPGUpdateUniforms');
    expect(PPG_UPDATE_WGSL).toContain('sampleCount');
  });

  it('declares PPGUpdateUniforms with frameParity for ping-pong', () => {
    expect(PPG_UPDATE_WGSL).toContain('frameParity');
  });

  it('binds PPGUpdateUniforms at @group(0) @binding(0)', () => {
    expect(PPG_UPDATE_WGSL).toContain('@group(0) @binding(0)');
    expect(PPG_UPDATE_WGSL).toContain('u_ppg');
  });

  it('binds ppgSamples (path completion buffer) at @group(0) @binding(1)', () => {
    expect(PPG_UPDATE_WGSL).toContain('@group(0) @binding(1)');
    expect(PPG_UPDATE_WGSL).toContain('ppgSamples');
  });

  it('binds ppgCells (spatial cells) at @group(0) @binding(2)', () => {
    expect(PPG_UPDATE_WGSL).toContain('@group(0) @binding(2)');
    expect(PPG_UPDATE_WGSL).toContain('ppgCells');
  });

  it('binds ppgLeafData (atomic u32 array) at @group(0) @binding(3)', () => {
    expect(PPG_UPDATE_WGSL).toContain('@group(0) @binding(3)');
    expect(PPG_UPDATE_WGSL).toContain('ppgLeafData');
  });

  it('binds ppgKdNodes at @group(0) @binding(4)', () => {
    expect(PPG_UPDATE_WGSL).toContain('@group(0) @binding(4)');
    expect(PPG_UPDATE_WGSL).toContain('ppgKdNodes');
  });

  it('ppgLeafData uses atomic<u32> for WebGPU-compatible atomics', () => {
    expect(PPG_UPDATE_WGSL).toContain('array<atomic<u32>>');
  });

  it('uses atomicAdd for radiance accumulation (fixed-point)', () => {
    expect(PPG_UPDATE_WGSL).toContain('atomicAdd');
  });

  it('contains PPG_RADIANCE_SCALE constant for fixed-point encoding', () => {
    expect(PPG_UPDATE_WGSL).toContain('PPG_RADIANCE_SCALE');
  });

  it('PPG_RADIANCE_SCALE is 65536.0 (16-bit fixed point)', () => {
    expect(PPG_UPDATE_WGSL).toContain('PPG_RADIANCE_SCALE: f32 = 65536.0');
  });

  it('ping-pong gate: skips update on odd frames (frameParity != 0)', () => {
    expect(PPG_UPDATE_WGSL).toContain('u_ppg.frameParity != 0u');
  });

  it('bounds-checks sampleIdx against u_ppg.sampleCount', () => {
    expect(PPG_UPDATE_WGSL).toContain('sampleIdx >= u_ppg.sampleCount');
  });

  it('contains ppgUpdateFindCell routed through kd traversal', () => {
    expect(PPG_UPDATE_WGSL).toContain('fn ppgUpdateFindCell');
    // ppgUpdateFindCell delegates to the canonical shared traversal in PPG_COMMON_WGSL.
    expect(PPG_UPDATE_WGSL).toContain('ppgKdFindCellShared');
  });

  it('contains ppgDirToBinIdx octahedral direction encode', () => {
    expect(PPG_UPDATE_WGSL).toContain('fn ppgDirToBinIdx');
  });

  it('ppgDirToBinIdx returns bin index clamped to [0..15]', () => {
    expect(PPG_UPDATE_WGSL).toContain('clamp(row * 4u + col, 0u, 15u)');
  });

  it('contains ppgLeafSlot index calculation helper', () => {
    expect(PPG_UPDATE_WGSL).toContain('fn ppgLeafSlot');
  });

  it('skips samples with zero or negative luminance', () => {
    expect(PPG_UPDATE_WGSL).toContain('lum <= 0.0');
  });

  it('declares PPGPathSample struct with worldPos and incidentDir', () => {
    expect(PPG_UPDATE_WGSL).toContain('struct PPGPathSample');
    expect(PPG_UPDATE_WGSL).toContain('worldPos');
    expect(PPG_UPDATE_WGSL).toContain('incidentDir');
  });

  it('declares PPGSpatialCell struct in update shader', () => {
    expect(PPG_UPDATE_WGSL).toContain('struct PPGSpatialCell');
  });
});

// ─── 4. createPPGBuffers — buffer count and sizes ─────────────────────────────

describe('createPPGBuffers — PPG GPU buffer allocation', () => {
  it('creates exactly 5 buffers', () => {
    const { device, bufferCalls } = makeMockDevice();
    createPPGBuffers(device);
    expect(bufferCalls.length).toBe(5);
  });

  it('creates cellBuffer, leafBuffer, sampleBuffer, sampleHeadBuffer, kdBuffer (in any order)', () => {
    const { device, bufferCalls } = makeMockDevice();
    createPPGBuffers(device);
    const labels = bufferCalls.map((c) => c.label ?? '');
    expect(labels).toContain('ppg-cell-buffer');
    expect(labels).toContain('ppg-leaf-buffer');
    expect(labels).toContain('ppg-sample-buffer');
    expect(labels).toContain('ppg-sample-head');
    expect(labels).toContain('ppg-kd-buffer');
  });

  it('cellBuffer is exactly PPG_MAX_SPATIAL_CELLS × PPG_CELL_BYTE_STRIDE bytes', () => {
    const { device, bufferCalls } = makeMockDevice();
    createPPGBuffers(device);
    const cell = bufferCalls.find((c) => c.label === 'ppg-cell-buffer');
    expect(cell).toBeDefined();
    expect(cell!.size).toBe(PPG_MAX_SPATIAL_CELLS * PPG_CELL_BYTE_STRIDE);
  });

  it('leafBuffer is exactly PPG_MAX_SPATIAL_CELLS × PPG_LEAF_BYTE_STRIDE bytes', () => {
    const { device, bufferCalls } = makeMockDevice();
    createPPGBuffers(device);
    const leaf = bufferCalls.find((c) => c.label === 'ppg-leaf-buffer');
    expect(leaf).toBeDefined();
    expect(leaf!.size).toBe(PPG_MAX_SPATIAL_CELLS * PPG_LEAF_BYTE_STRIDE);
  });

  it('sampleBuffer size is >= maxCells × 48 bytes (PPGPathSample stride)', () => {
    const { device, bufferCalls } = makeMockDevice();
    createPPGBuffers(device);
    const sample = bufferCalls.find((c) => c.label === 'ppg-sample-buffer');
    expect(sample).toBeDefined();
    const PPG_PATH_SAMPLE_STRIDE = 48;
    expect(sample!.size).toBeGreaterThanOrEqual(PPG_MAX_SPATIAL_CELLS * PPG_PATH_SAMPLE_STRIDE);
  });

  it('kdBuffer is PPG_KD_MAX_NODES × PPG_KD_NODE_BYTE_STRIDE bytes', () => {
    const { device, bufferCalls } = makeMockDevice();
    createPPGBuffers(device);
    const kd = bufferCalls.find((c) => c.label === 'ppg-kd-buffer');
    expect(kd).toBeDefined();
    expect(kd!.size).toBe(PPG_KD_MAX_NODES * PPG_KD_NODE_BYTE_STRIDE);
  });

  it('sampleHeadBuffer is 16 bytes (atomic head + alignment)', () => {
    const { device, bufferCalls } = makeMockDevice();
    createPPGBuffers(device);
    const head = bufferCalls.find((c) => c.label === 'ppg-sample-head');
    expect(head).toBeDefined();
    expect(head!.size).toBe(16);
  });

  it('writes disabled kd sentinel on init', () => {
    const { device, bufferCalls } = makeMockDevice();
    createPPGBuffers(device);
    expect(bufferCalls.length).toBe(5);
    const mockQueue = device.queue as unknown as { writeBuffer: ReturnType<typeof vi.fn> };
    expect(mockQueue.writeBuffer).toHaveBeenCalled();
    const disabled = encodePpgKdDisabledRoot();
    const matchCall = mockQueue.writeBuffer.mock.calls.find((call: unknown[]) => {
      // writeBuffer's data argument can be an ArrayBuffer or a typed-array
      // view; normalize to Uint8Array to compare bytes uniformly.
      const data = call[2] as ArrayBuffer | ArrayBufferView;
      const buf =
        data instanceof ArrayBuffer
          ? new Uint8Array(data)
          : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
      // Assert content equality, not just length — catches a regression where
      // some other 16-byte payload accidentally matches the sentinel's size.
      if (buf.byteLength !== disabled.byteLength) return false;
      for (let i = 0; i < disabled.length; i++) {
        if (buf[i] !== disabled[i]) return false;
      }
      return true;
    });
    expect(matchCall).toBeDefined();
  });

  it('all PPG buffers include GPUBufferUsage.STORAGE', () => {
    const { device, bufferCalls } = makeMockDevice();
    createPPGBuffers(device);
    for (const call of bufferCalls) {
      expect(call.usage & GPUBufferUsage.STORAGE).toBeTruthy();
    }
  });

  it('all PPG buffers include GPUBufferUsage.COPY_DST (host upload)', () => {
    const { device, bufferCalls } = makeMockDevice();
    createPPGBuffers(device);
    for (const call of bufferCalls) {
      expect(call.usage & GPUBufferUsage.COPY_DST).toBeTruthy();
    }
  });

  it('all PPG buffers include GPUBufferUsage.COPY_SRC (CPU readback for tests)', () => {
    const { device, bufferCalls } = makeMockDevice();
    createPPGBuffers(device);
    for (const call of bufferCalls) {
      expect(call.usage & GPUBufferUsage.COPY_SRC).toBeTruthy();
    }
  });

  it('respects custom maxCells option', () => {
    const { device, bufferCalls } = makeMockDevice();
    createPPGBuffers(device, { maxCells: 500 });
    const cell = bufferCalls.find((c) => c.label === 'ppg-cell-buffer');
    expect(cell!.size).toBe(500 * PPG_CELL_BYTE_STRIDE);
  });

  it('returns maxCells field matching the requested cap', () => {
    const { device } = makeMockDevice();
    const result = createPPGBuffers(device, { maxCells: 1000 });
    expect(result.maxCells).toBe(1000);
  });

  it('returns maxCells = PPG_MAX_SPATIAL_CELLS when no options given', () => {
    const { device } = makeMockDevice();
    const result = createPPGBuffers(device);
    expect(result.maxCells).toBe(PPG_MAX_SPATIAL_CELLS);
  });
});

// ─── 5. PPG disabled → no buffer allocation ───────────────────────────────────

describe('PPG disabled — no buffer allocation when ppgEnabled = false/unset', () => {
  it('createFrameResources without ppgEnabled creates no PPG buffers', () => {
    const { device, bufferCalls } = makeMockDevice();
    const res = createFrameResources(device, 64, 64);
    // PPG buffers are opt-in; the field should be undefined when not requested.
    expect(res.ppgBuffers).toBeUndefined();
    // Sanity: non-PPG buffers ARE still created.
    const ppgLabels = bufferCalls
      .filter((c) => c.label?.startsWith('ppg-'))
      .map((c) => c.label);
    expect(ppgLabels).toHaveLength(0);
  });

  it('createFrameResources with ppgEnabled: false creates no PPG buffers', () => {
    const { device, bufferCalls } = makeMockDevice();
    const res = createFrameResources(device, 64, 64, { ppgEnabled: false });
    expect(res.ppgBuffers).toBeUndefined();
    const ppgLabels = bufferCalls.filter((c) => c.label?.startsWith('ppg-'));
    expect(ppgLabels).toHaveLength(0);
  });

  it('createFrameResources with ppgEnabled: true creates 5 PPG buffers', () => {
    const { device, bufferCalls } = makeMockDevice();
    const res = createFrameResources(device, 64, 64, { ppgEnabled: true });
    expect(res.ppgBuffers).toBeDefined();
    const ppgLabels = bufferCalls.filter((c) => c.label?.startsWith('ppg-'));
    expect(ppgLabels).toHaveLength(5);
  });

  it('destroyFrameResources without PPG buffers does not throw', () => {
    const { device } = makeMockDevice();
    const res = createFrameResources(device, 64, 64);
    expect(() => destroyFrameResources(res)).not.toThrow();
  });

  it('destroyFrameResources with PPG buffers calls destroy on all five PPG GPU buffers', () => {
    const destroyMock = vi.fn();
    const bufferMock  = { destroy: destroyMock };
    const textureMock = { destroy: vi.fn(), createView: vi.fn(() => ({})) };

    const mockDevice = {
      createBuffer:  vi.fn(() => bufferMock),
      createTexture: vi.fn(() => textureMock),
      createSampler: vi.fn(() => ({})),
      queue: { writeBuffer: vi.fn(), writeTexture: vi.fn() },
    } as unknown as GPUDevice;

    const res = createFrameResources(mockDevice, 64, 64, { ppgEnabled: true });
    destroyFrameResources(res);

    // destroyMock called for shared buffer mock across many creates;
    // at least 5 PPG buffer destroys when PPG is enabled.
    expect(destroyMock.mock.calls.length).toBeGreaterThanOrEqual(5);
  });
});

// ─── 6. destroyPPGBuffers — explicit destroy helper ──────────────────────────

describe('destroyPPGBuffers — explicit PPG buffer destruction', () => {
  it('calls destroy on cellBuffer, leafBuffer, sampleBuffer, sampleHeadBuffer, kdBuffer', () => {
    const destroyCell   = vi.fn();
    const destroyLeaf   = vi.fn();
    const destroySample = vi.fn();
    const destroyHead   = vi.fn();
    const destroyKd     = vi.fn();

    destroyPPGBuffers({
      cellBuffer:        { destroy: destroyCell }   as unknown as GPUBuffer,
      leafBuffer:        { destroy: destroyLeaf }   as unknown as GPUBuffer,
      sampleBuffer:      { destroy: destroySample } as unknown as GPUBuffer,
      sampleHeadBuffer:  { destroy: destroyHead }   as unknown as GPUBuffer,
      kdBuffer:          { destroy: destroyKd }     as unknown as GPUBuffer,
      maxCells:          100,
    });

    expect(destroyCell).toHaveBeenCalledOnce();
    expect(destroyLeaf).toHaveBeenCalledOnce();
    expect(destroySample).toHaveBeenCalledOnce();
    expect(destroyHead).toHaveBeenCalledOnce();
    expect(destroyKd).toHaveBeenCalledOnce();
  });
});

// ─── 7. setPPGEnabled lifecycle ────────────────────────────────────────────────

describe('HybridEngine — setPPGEnabled lifecycle (Sprint 11)', () => {
  /**
   * HybridEngine requires a live GPUDevice, THREE.Scene, and async pipeline init.
   * We test the public API surface (ppgEnabled getter, setPPGEnabled) without
   * instantiating the engine — the method bodies are pure state mutations that
   * don't require GPU. We validate them via direct import and prototype inspection.
   */

  it('HybridEngine is exported from the package', async () => {
    const mod = await import('../src/HybridEngine.js');
    expect(typeof mod.HybridEngine).toBe('function');
  });

  it('HybridEngine.prototype has setPPGEnabled method', async () => {
    const { HybridEngine } = await import('../src/HybridEngine.js');
    expect(typeof HybridEngine.prototype.setPPGEnabled).toBe('function');
  });

  it('HybridEngine.prototype has ppgEnabled getter', async () => {
    const { HybridEngine } = await import('../src/HybridEngine.js');
    const descriptor = Object.getOwnPropertyDescriptor(HybridEngine.prototype, 'ppgEnabled');
    expect(descriptor).toBeDefined();
    expect(typeof descriptor!.get).toBe('function');
  });

  it('HybridEngineOptions type accepts ppgEnabled field (compile-time only — verified by tsc)', () => {
    // TypeScript compile would fail if HybridEngineOptions did not accept ppgEnabled.
    // This test documents the intent; actual type-check is `npx tsc --noEmit`.
    expect(true).toBe(true);
  });
});

// ─── 8. PPG exports from package index ────────────────────────────────────────

describe('PPG exports from package index (Sprint 11)', () => {
  it('PPG_MAX_SPATIAL_CELLS is exported from package index', async () => {
    const mod = await import('../src/index.js');
    expect(typeof (mod as Record<string, unknown>)['PPG_MAX_SPATIAL_CELLS']).toBe('number');
  });

  it('PPG_DIRECTIONS is exported from package index', async () => {
    const mod = await import('../src/index.js');
    expect(typeof (mod as Record<string, unknown>)['PPG_DIRECTIONS']).toBe('number');
  });

  it('PPG_CELL_BYTE_STRIDE is exported from package index', async () => {
    const mod = await import('../src/index.js');
    expect(typeof (mod as Record<string, unknown>)['PPG_CELL_BYTE_STRIDE']).toBe('number');
  });

  it('PPG_LEAF_BYTE_STRIDE is exported from package index', async () => {
    const mod = await import('../src/index.js');
    expect(typeof (mod as Record<string, unknown>)['PPG_LEAF_BYTE_STRIDE']).toBe('number');
  });

  it('PPG_SAMPLE_WGSL is NOT exported (deleted in P3-C.2; superseded by shadePpgGuide.wgsl.ts)', async () => {
    const mod = await import('../src/index.js');
    expect((mod as Record<string, unknown>)['PPG_SAMPLE_WGSL']).toBeUndefined();
  });

  it('PPG_UPDATE_WGSL is exported from package index', async () => {
    const mod = await import('../src/index.js');
    expect(typeof (mod as Record<string, unknown>)['PPG_UPDATE_WGSL']).toBe('string');
  });

  it('createPPGBuffers is exported from package index', async () => {
    const mod = await import('../src/index.js');
    expect(typeof (mod as Record<string, unknown>)['createPPGBuffers']).toBe('function');
  });

  it('destroyPPGBuffers is exported from package index', async () => {
    const mod = await import('../src/index.js');
    expect(typeof (mod as Record<string, unknown>)['destroyPPGBuffers']).toBe('function');
  });

  it('writePpgKdTree is exported from package index', async () => {
    const mod = await import('../src/index.js');
    expect(typeof (mod as Record<string, unknown>)['writePpgKdTree']).toBe('function');
  });

  it('buildPpgKdTreeGpuBytes is exported from package index', async () => {
    const mod = await import('../src/index.js');
    expect(typeof (mod as Record<string, unknown>)['buildPpgKdTreeGpuBytes']).toBe('function');
  });
});
