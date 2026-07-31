/**
 * r4WaveFixes.test.ts — pin tests for the R4 (pt-webgpu) complexity-sweep fixes:
 *
 *   V2-1  SPPM ownership is strict: photon-map mode (2u) is packed only after the
 *         exact photon-cell allocation and SPPM pipelines are ready. If the
 *         allocation exceeds the effective device ceiling, renderFrame throws
 *         before publishing a FrameParams write.
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

function perspectiveMat(): Float32Array {
  const near = 0.1;
  const far = 100;
  const m = new Float32Array(16);
  m[0] = 1;
  m[5] = 1;
  m[10] = (far + near) / (near - far);
  m[11] = -1;
  m[14] = (2 * far * near) / (near - far);
  return m;
}

function frameInput(size: number) {
  return {
    viewMatrix: asMat4(identityMat()),
    projMatrix: asMat4(identityMat()),
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

describe('R4 / V2-1 — SPPM ownership is gated on exact allocation readiness', () => {
  it('packs causticMode=2 after the default device admits the photon-cell allocation', async () => {
    const rec: Recorder = { bufferWrites: [] };
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

  it('keeps causticMode=2 at a 32 MiB ceiling because the current allocation is 3 MiB', async () => {
    const rec: Recorder = { bufferWrites: [] };
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
    expect(params[FrameParamsSlot.causticStrategy]).toBe(2);
    engine.dispose();
  });

  it('rejects an undersized device before any per-frame allocation, encoding, submit, or params write', async () => {
    const rec: Recorder = { bufferWrites: [] };
    const device = makeFullTierDevice(rec, {
      maxBufferSize: 2 * 1024 * 1024,
      maxStorageBufferBindingSize: 2 * 1024 * 1024,
    });
    const engine = await createPTEngine_WebGPU({
      device,
      causticStrategy: 'photon-map',
    });
    engine.setScene(makeScene());

    const buffersBefore = vi.mocked(device.createBuffer).mock.calls.length;
    const encodersBefore = vi.mocked(device.createCommandEncoder).mock.calls.length;
    const submitsBefore = vi.mocked(device.queue.submit).mock.calls.length;
    const paramsWritesBefore = rec.bufferWrites.filter(
      (write) => write.label === 'vitrum.pt-webgpu.params',
    ).length;

    expect(() => engine.renderFrame(frameInput(16))).toThrow(
      /SPPM photon grid.*device storage-buffer limits/i,
    );
    expect(vi.mocked(device.createBuffer).mock.calls).toHaveLength(buffersBefore);
    expect(vi.mocked(device.createCommandEncoder).mock.calls).toHaveLength(encodersBefore);
    expect(vi.mocked(device.queue.submit).mock.calls).toHaveLength(submitsBefore);
    expect(
      rec.bufferWrites.filter((write) => write.label === 'vitrum.pt-webgpu.params'),
    ).toHaveLength(paramsWritesBefore);
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
    let submitThrowArmed = false;

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
        submit: vi.fn(() => {
          if (submitThrowArmed) {
            throw new Error('injected reset submit failure (test)');
          }
        }),
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

    // Also fail after the complete candidate has been published reversibly:
    // reset() submits the accumulator clear while the previous scene is still
    // alive. The catch must restore that old scene and retire only candidates.
    const buffersBeforeResetFailure = allBuffers.length;
    submitThrowArmed = true;
    expect(() => engine.setScene(makeScene())).toThrow(
      'injected reset submit failure',
    );
    submitThrowArmed = false;
    const resetFailureCandidates = allBuffers
      .slice(buffersBeforeResetFailure)
      .filter((buffer) => buffer.label.startsWith('vitrum.pt-webgpu.scene.'));
    expect(resetFailureCandidates.length).toBeGreaterThan(0);
    for (const buffer of resetFailureCandidates) {
      expect(buffer.destroy, buffer.label).toHaveBeenCalled();
    }
    for (const buffer of firstSceneBuffers) {
      expect(buffer.destroy, buffer.label).not.toHaveBeenCalled();
    }
    expect(() => engine.renderFrame(frameInput(16))).not.toThrow();

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
    engine.renderFrame({
      ...frameInput(16),
      projMatrix: asMat4(perspectiveMat()),
    });
    const params = latestFrameParams(rec);
    const maxLv = params[FrameParamsSlot.bdptMaxLightBounces];
    expect(maxLv).toBe(2);
    // The kernel connection loop is `for (var lvi = 1u; lvi < maxLv; lvi++)`.
    // With maxLv=2 it runs once (lvi=1); with the old default maxLv=1 it never ran.
    expect(maxLv).toBeGreaterThan(1);
    engine.dispose();
  });
});
