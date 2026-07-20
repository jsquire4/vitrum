/**
 * r4WaveFixes.test.ts — pin tests for the R4 (pt-webgpu) complexity-sweep fixes:
 *
 *   V2-1  SPPM ceiling-miss fallback is HONORED: when the photon-cells allocation
 *         would exceed the effective ceiling, the packed FrameParams caustic mode
 *         must flip from photon-map (2u) to manifold-nee (1u) so the kernel takes
 *         the manifold path instead of gathering against placeholder photon
 *         buffers (which rendered caustics ~zero).
 *
 *   V2-5 / D2  the BDPT default light-bounce count is now 2 (was 1). At the default
 *         the packed `bdptMaxLightBounces` slot must be 2 so the kernel connection
 *         loop `for lvi=1u; lvi<maxLv` executes at least once (lvi=1) — otherwise
 *         BDPT is silently inert.
 *
 * NO real GPU (mock device, mirroring restirPtReuseWiring.test.ts). The mock
 * records every queue.writeBuffer so the test can read back the packed FrameParams
 * UBO (`vitrum.pt-webgpu.params`) and byte-check the caustic-mode / bdpt slots.
 */

import { describe, expect, it, vi } from 'vitest';
import type { Scene } from '@vitrum/core';
import { asMat4 } from '@vitrum/core';
import { createPTEngine_WebGPU } from '../index.js';
import { FrameParamsSlot } from '../scene/frameParamsLayout.js';
import { installGpuConstStubs, textureStubMethods } from './gpuStub.js';

interface Recorder {
  bufferWrites: Array<{ label: string; bytes: Uint8Array }>;
}

/**
 * A full-tier mock device. `limits` is overridable so a test can force the SPPM
 * photon-cells allocation over the ceiling by shrinking maxStorageBufferBindingSize.
 */
function makeFullTierDevice(rec: Recorder, limitsOverride: Record<string, number> = {}): GPUDevice {
  installGpuConstStubs();
  const pass = {
    setPipeline: vi.fn(),
    setBindGroup: vi.fn(),
    dispatchWorkgroups: vi.fn(),
    end: vi.fn(),
  };
  const encoder = {
    beginComputePass: vi.fn(() => pass),
    clearBuffer: vi.fn(),
    copyBufferToBuffer: vi.fn(),
    finish: vi.fn(() => ({})),
  };
  return {
    queue: {
      writeBuffer: vi.fn((buffer?: { label?: string }, _offset?: number, data?: BufferSource) => {
        let bytes = new Uint8Array();
        if (data instanceof ArrayBuffer) {
          bytes = new Uint8Array(data.slice(0));
        } else if (ArrayBuffer.isView(data)) {
          bytes = new Uint8Array(
            data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength),
          );
        }
        rec.bufferWrites.push({ label: buffer?.label ?? '', bytes });
      }),
      writeTexture: vi.fn(),
      submit: vi.fn(),
    },
    createBuffer: vi.fn((desc?: { label?: string }) => ({ label: desc?.label ?? '', destroy: vi.fn() })),
    ...textureStubMethods(),
    createShaderModule: vi.fn(() => ({ getCompilationInfo: vi.fn(async () => ({ messages: [] })) })),
    createBindGroupLayout: vi.fn(() => ({})),
    createPipelineLayout: vi.fn(() => ({})),
    createComputePipeline: vi.fn(() => ({ getBindGroupLayout: vi.fn(() => ({})) })),
    createBindGroup: vi.fn(() => ({})),
    createCommandEncoder: vi.fn(() => encoder),
    limits: {
      maxStorageBuffersPerShaderStage: 64,
      maxStorageTexturesPerShaderStage: 8,
      ...limitsOverride,
    },
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    lost: new Promise<never>(() => {}),
  } as unknown as GPUDevice;
}

function makeScene(): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'mesh-a',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: { baseColor: [0.8, 0.2, 0.1], roughness: 0.3, metallic: 0.1 },
      },
    ],
    emitters: [
      { kind: 'directional', id: 'sun', direction: [0, -1, 0], color: [1, 1, 1], intensity: 1 },
    ],
    environment: { kind: 'none' },
  };
}

function identityMat(): Float32Array {
  const m = new Float32Array(16);
  m[0] = 1; m[5] = 1; m[10] = 1; m[15] = 1;
  return m;
}

function frameInput(size: number) {
  return {
    viewMatrix: asMat4(identityMat()),
    projMatrix: asMat4(identityMat()),
    cameraPosition: [0, 0, 1] as [number, number, number],
    viewport: { width: size, height: size, devicePixelRatio: 1 },
    frameIndex: 0,
    frameSeed: 1,
    quality: { samplesTarget: 4, bounces: 2, resolutionFactor: 1 },
  };
}

/** Read back the latest packed FrameParams UBO write (`vitrum.pt-webgpu.params`). */
function latestFrameParams(rec: Recorder): Uint32Array {
  const write = rec.bufferWrites.filter((w) => w.label === 'vitrum.pt-webgpu.params').at(-1);
  expect(write, 'FrameParams UBO write').toBeDefined();
  const buf = write!.bytes.buffer.slice(
    write!.bytes.byteOffset,
    write!.bytes.byteOffset + write!.bytes.byteLength,
  );
  return new Uint32Array(buf);
}

describe('R4 / V2-1 — SPPM ceiling-miss honors the manifold-nee fallback', () => {
  it('packs causticMode=2 (photon-map) when the photon-cells allocation FITS the ceiling', async () => {
    const rec: Recorder = { bufferWrites: [] };
    // Default limits (256 MiB buffer / 128 MiB binding) — the ~96 MiB photon-cells
    // allocation fits, so the configured photon-map strategy is used.
    const engine = await createPTEngine_WebGPU({
      device: makeFullTierDevice(rec),
      causticStrategy: 'photon-map',
    });
    engine.setScene(makeScene());
    engine.renderFrame(frameInput(16));
    const params = latestFrameParams(rec);
    expect(params[FrameParamsSlot.causticStrategy]).toBe(2);
    engine.dispose();
  });

  it('flips the packed causticMode to 1 (manifold-nee) when SPPM allocation exceeds the ceiling', async () => {
    const rec: Recorder = { bufferWrites: [] };
    // Force the ceiling miss: shrink the storage-binding limit below the ~96 MiB
    // photon-cells buffer so sppmWouldExceedCeiling() returns true.
    const engine = await createPTEngine_WebGPU({
      device: makeFullTierDevice(rec, {
        maxBufferSize: 32 * 1024 * 1024,
        maxStorageBufferBindingSize: 32 * 1024 * 1024,
      }),
      causticStrategy: 'photon-map',
    });
    engine.setScene(makeScene());
    engine.renderFrame(frameInput(16));
    const params = latestFrameParams(rec);
    // Before the fix this was 2 (photon-map) — the kernel gathered against
    // placeholder photon buffers and caustics rendered ~zero. Now the fallback is
    // honored: the packed mode is manifold-nee (1u).
    expect(params[FrameParamsSlot.causticStrategy]).toBe(1);
    engine.dispose();
  });
});

describe('R4 / V2-2 — setScene is atomic (rollback on throw, no leak, old scene retained)', () => {
  it('destroys every buffer created by a failed uploadPackedScene AND keeps the previous scene', async () => {
    const rec: Recorder = { bufferWrites: [] };
    // Track every buffer the device hands out so we can assert none leak.
    interface TrackedBuffer { label: string; destroy: ReturnType<typeof vi.fn>; }
    const allBuffers: TrackedBuffer[] = [];
    let throwArmed = false;
    let createsSinceArmed = 0;

    installGpuConstStubs();
    const pass = { setPipeline: vi.fn(), setBindGroup: vi.fn(), dispatchWorkgroups: vi.fn(), end: vi.fn() };
    const encoder = {
      beginComputePass: vi.fn(() => pass),
      clearBuffer: vi.fn(),
      copyBufferToBuffer: vi.fn(),
      finish: vi.fn(() => ({})),
    };
    const device = {
      queue: {
        writeBuffer: vi.fn((buffer?: { label?: string }, _o?: number, data?: BufferSource) => {
          let bytes = new Uint8Array();
          if (data instanceof ArrayBuffer) bytes = new Uint8Array(data.slice(0));
          else if (ArrayBuffer.isView(data)) {
            bytes = new Uint8Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));
          }
          rec.bufferWrites.push({ label: buffer?.label ?? '', bytes });
        }),
        writeTexture: vi.fn(),
        submit: vi.fn(),
      },
      createBuffer: vi.fn((desc?: { label?: string }) => {
        const label = desc?.label ?? '';
        // Once armed (second setScene), throw partway through the scene upload so we
        // exercise uploadPackedScene's mid-sequence failure cleanup path.
        if (throwArmed && label.startsWith('vitrum.pt-webgpu.scene.')) {
          createsSinceArmed++;
          if (createsSinceArmed === 5) {
            throw new Error('injected device.createBuffer failure (test)');
          }
        }
        const buf: TrackedBuffer = { label, destroy: vi.fn() };
        allBuffers.push(buf);
        return buf as unknown as GPUBuffer;
      }),
      ...textureStubMethods(),
      createShaderModule: vi.fn(() => ({ getCompilationInfo: vi.fn(async () => ({ messages: [] })) })),
      createBindGroupLayout: vi.fn(() => ({})),
      createPipelineLayout: vi.fn(() => ({})),
      createComputePipeline: vi.fn(() => ({ getBindGroupLayout: vi.fn(() => ({})) })),
      createBindGroup: vi.fn(() => ({})),
      createCommandEncoder: vi.fn(() => encoder),
      limits: { maxStorageBuffersPerShaderStage: 64, maxStorageTexturesPerShaderStage: 8 },
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      lost: new Promise<never>(() => {}),
    } as unknown as GPUDevice;

    const engine = await createPTEngine_WebGPU({ device });
    // First scene uploads cleanly and renders.
    engine.setScene(makeScene());
    engine.renderFrame(frameInput(16));
    const paramsAfterFirst = latestFrameParams(rec);
    const firstTriangleCount = paramsAfterFirst[FrameParamsSlot.triangleCount];

    // Snapshot the set of scene buffers that belong to the (retained) first scene.
    const firstSceneBuffers = allBuffers.filter((b) => b.label.startsWith('vitrum.pt-webgpu.scene.'));
    const buffersBeforeSecond = allBuffers.length;

    // Arm the failure and attempt a second setScene — it must throw.
    throwArmed = true;
    expect(() => engine.setScene(makeScene())).toThrow('injected device.createBuffer failure');
    throwArmed = false;

    // No leak: every buffer created DURING the failed upload was destroyed by
    // uploadPackedScene's catch/rollback.
    const buffersCreatedByFailedUpload = allBuffers.slice(buffersBeforeSecond);
    expect(buffersCreatedByFailedUpload.length).toBeGreaterThan(0);
    for (const b of buffersCreatedByFailedUpload) {
      expect(b.destroy, `buffer ${b.label} must be destroyed on rollback`).toHaveBeenCalled();
    }

    // Old scene retained: the first scene's buffers were NOT destroyed (the swap
    // only destroys the previous scene on a SUCCESSFUL upload), and the engine can
    // still render the first scene.
    for (const b of firstSceneBuffers) {
      expect(b.destroy, `retained first-scene buffer ${b.label} must NOT be destroyed`).not.toHaveBeenCalled();
    }
    expect(() => engine.renderFrame(frameInput(16))).not.toThrow();
    const paramsAfterFailure = latestFrameParams(rec);
    expect(paramsAfterFailure[FrameParamsSlot.triangleCount]).toBe(firstTriangleCount);

    engine.dispose();
  });
});

describe('R4 / V2-5 (D2) — BDPT default light-bounces is 2 so the connection loop runs', () => {
  it('packs bdptMaxLightBounces=2 at the default so `for lvi=1u; lvi<maxLv` executes lvi=1', async () => {
    const rec: Recorder = { bufferWrites: [] };
    const engine = await createPTEngine_WebGPU({
      device: makeFullTierDevice(rec),
      bdpt: true, // no bdptOptions → default light-bounce count
    });
    engine.setScene(makeScene());
    engine.renderFrame(frameInput(16));
    const params = latestFrameParams(rec);
    const maxLv = params[FrameParamsSlot.bdptMaxLightBounces];
    expect(maxLv).toBe(2);
    // The kernel connection loop is `for (var lvi = 1u; lvi < maxLv; lvi++)`.
    // With maxLv=2 it runs once (lvi=1); with the old default maxLv=1 it never ran.
    expect(maxLv).toBeGreaterThan(1);
    engine.dispose();
  });
});
