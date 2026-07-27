import { describe, expect, it } from 'vitest';
import {
  RCDispatcher,
  RC_REQUIRED_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE,
  type CascadeDim,
} from '../src/index.js';

const TEST_DIMS: readonly CascadeDim[] = [{
  probes: [1, 1, 1],
  rays: 16,
  intervalNear: 0,
  intervalFar: 1e6,
}];

async function requestRequiredWebGpuDevice(): Promise<GPUDevice> {
  const gpu = navigator.gpu;
  if (gpu == null) {
    throw new Error('RC GPU acceptance requires navigator.gpu; the browser gate must fail closed.');
  }
  const adapter = await gpu.requestAdapter({ powerPreference: 'low-power' });
  if (adapter == null) {
    throw new Error('RC GPU acceptance could not acquire a WebGPU adapter.');
  }
  const requiredStorageBuffers = RC_REQUIRED_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE;
  if (adapter.limits.maxStorageBuffersPerShaderStage < requiredStorageBuffers) {
    throw new Error(
      `RC GPU acceptance requires maxStorageBuffersPerShaderStage>=${requiredStorageBuffers}; ` +
      `adapter exposes ${adapter.limits.maxStorageBuffersPerShaderStage}.`,
    );
  }
  return adapter.requestDevice({
    requiredLimits: { maxStorageBuffersPerShaderStage: requiredStorageBuffers },
  });
}

function uploadBuffer(
  device: GPUDevice,
  label: string,
  bytes: ArrayBuffer,
  minimumSize: number,
): GPUBuffer {
  const size = Math.max(minimumSize, Math.ceil(bytes.byteLength / 4) * 4);
  const buffer = device.createBuffer({
    label,
    size,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
  });
  new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(bytes));
  buffer.unmap();
  return buffer;
}

function emptyLeafNode(): ArrayBuffer {
  const bytes = new ArrayBuffer(32);
  const floats = new Float32Array(bytes);
  const words = new Uint32Array(bytes);
  floats.set([-1e6, -1e6, -1e6, 1e6, 1e6, 1e6]);
  words[6] = 0;
  words[7] = 0xffff0000;
  return bytes;
}

async function readBuffer(
  device: GPUDevice,
  source: GPUBuffer,
  byteLength: number,
): Promise<Float32Array> {
  const staging = device.createBuffer({
    label: 'rc-behavior-readback',
    size: byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = device.createCommandEncoder({ label: 'rc-behavior-readback' });
    encoder.copyBufferToBuffer(source, 0, staging, 0, byteLength);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    return new Float32Array(staging.getMappedRange().slice(0));
  } finally {
    if (staging.mapState === 'mapped') staging.unmap();
    staging.destroy();
  }
}

describe('walkaround-rc behavior acceptance (real WebGPU)', () => {
  it('dispatches the real probe shader and produces finite environment radiance', async () => {
    const device = await requestRequiredWebGpuDevice();
    const dispatcher = new RCDispatcher(TEST_DIMS);
    const ownedBuffers: GPUBuffer[] = [];
    let environment: GPUTexture | null = null;
    try {
      const add = (buffer: GPUBuffer): GPUBuffer => {
        ownedBuffers.push(buffer);
        return buffer;
      };
      const bvhNodesBuf = add(uploadBuffer(device, 'rc-test-empty-leaf', emptyLeafNode(), 32));
      const bvhIndicesBuf = add(uploadBuffer(device, 'rc-test-indices', new ArrayBuffer(16), 16));
      const bvhPositionsBuf = add(uploadBuffer(device, 'rc-test-positions', new ArrayBuffer(16), 16));
      const bvhNormalsBuf = add(uploadBuffer(device, 'rc-test-normals', new ArrayBuffer(16), 16));
      const materialsBuf = add(uploadBuffer(device, 'rc-test-materials', new ArrayBuffer(64), 64));
      const triMaterialIdBuf = add(uploadBuffer(device, 'rc-test-tri-material', new ArrayBuffer(4), 4));
      const cascadeBuf = add(device.createBuffer({
        label: 'rc-test-cascade-c0',
        size: 16 * 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      }));
      environment = device.createTexture({
        label: 'rc-test-white-environment',
        size: [1, 1],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      device.queue.writeTexture(
        { texture: environment },
        new Uint8Array([255, 255, 255, 255]),
        { bytesPerRow: 4 },
        [1, 1],
      );

      device.pushErrorScope('validation');
      dispatcher.dispatchFrameRaw({
        device,
        bvhNodesBuf,
        bvhIndicesBuf,
        bvhPositionsBuf,
        bvhNormalsBuf,
        materialsBuf,
        triMaterialIdBuf,
        cascadeBufs: [cascadeBuf],
        probeOriginWorld: [0, 0, 0],
        roomSize: [1, 1, 1],
        sunDirection: [0, 1, 0],
        sunColor: [0, 0, 0],
        envTextureView: environment.createView(),
        envSampler: device.createSampler({ minFilter: 'linear', magFilter: 'linear' }),
        frameSeed: 7,
      });
      await device.queue.onSubmittedWorkDone();
      const validationError = await device.popErrorScope();
      expect(validationError?.message ?? null).toBeNull();

      const values = await readBuffer(device, cascadeBuf, 16 * 16);
      expect(values).toHaveLength(64);
      expect(Array.from(values).every(Number.isFinite)).toBe(true);
      let indirectEnergy = 0;
      for (let ray = 0; ray < 16; ray += 1) {
        indirectEnergy += values[ray * 4]! + values[ray * 4 + 1]! + values[ray * 4 + 2]!;
      }
      expect(indirectEnergy / (16 * 3)).toBeGreaterThan(0.99);
    } finally {
      dispatcher.dispose();
      environment?.destroy();
      for (const buffer of ownedBuffers) buffer.destroy();
      device.destroy();
    }
  }, 30_000);
});
