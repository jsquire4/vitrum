import { describe, expect, it } from 'vitest';
import type { MeshPrimitive, Scene } from '@vitrum/core';
import {
  OCTAHEDRAL_CORE_WGSL,
  PCG_HASH_TO_F32_WGSL,
} from '@vitrum/shared-samplers';
import { RC_REQUIRED_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE } from '@vitrum/walkaround-rc';
import { RCSubsystem } from '../src/HybridEngineRC.js';
import { SAMPLE_CASCADE_C0_WGSL } from '../src/shaders/sampleCascadeC0.wgsl.js';

const TEST_DIMS = [{
  probes: [1, 1, 1] as const,
  rays: 16,
  intervalNear: 0,
  intervalFar: 1e6,
}] as const;

async function requestBaselineDevice(): Promise<GPUDevice> {
  if (navigator.gpu == null) {
    throw new Error('Hybrid RC GPU acceptance requires navigator.gpu.');
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'low-power' });
  if (adapter == null) throw new Error('Hybrid RC acceptance could not acquire an adapter.');
  const required = RC_REQUIRED_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE;
  if (adapter.limits.maxStorageBuffersPerShaderStage < required) {
    throw new Error(
      `Hybrid RC acceptance requires maxStorageBuffersPerShaderStage>=${required}; ` +
      `adapter exposes ${adapter.limits.maxStorageBuffersPerShaderStage}.`,
    );
  }
  return adapter.requestDevice({
    requiredLimits: { maxStorageBuffersPerShaderStage: required },
  });
}

function triangleScene(): Scene {
  const primitive: MeshPrimitive = {
    kind: 'mesh',
    id: 'rc-gpu-triangle',
    positions: new Float32Array([
      -1, -1, 0,
      1, -1, 0,
      0, 1, 0,
    ]),
    normals: new Float32Array([
      0, 0, 1,
      0, 0, 1,
      0, 0, 1,
    ]),
    uvs: new Float32Array(6),
    indices: new Uint32Array([0, 1, 2]),
    material: { baseColor: [0.5, 0.5, 0.5], roughness: 1, metallic: 0 },
  };
  return { primitives: [primitive], emitters: [], environment: { kind: 'none' } };
}

function createUploadedBuffer(
  device: GPUDevice,
  label: string,
  bytes: ArrayBuffer,
  usage: GPUBufferUsageFlags,
): GPUBuffer {
  const buffer = device.createBuffer({ label, size: bytes.byteLength, usage });
  device.queue.writeBuffer(buffer, 0, bytes);
  return buffer;
}

async function readVec4(device: GPUDevice, source: GPUBuffer): Promise<Float32Array> {
  const staging = device.createBuffer({
    label: 'hybrid-rc-sample-readback',
    size: 16,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = device.createCommandEncoder({ label: 'hybrid-rc-sample-readback' });
    encoder.copyBufferToBuffer(source, 0, staging, 0, 16);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    return new Float32Array(staging.getMappedRange().slice(0));
  } finally {
    if (staging.mapState === 'mapped') staging.unmap();
    staging.destroy();
  }
}

describe('Radiance Cascades hybrid composition (real WebGPU)', () => {
  it('runs the real RC producer and receiver; enabled changes output without recompiling', async () => {
    const device = await requestBaselineDevice();
    const rc = new RCSubsystem(device, TEST_DIMS);
    const ownedBuffers: GPUBuffer[] = [];
    let environment: GPUTexture | null = null;
    try {
      rc.setSceneFromCore(triangleScene());
      environment = device.createTexture({
        label: 'hybrid-rc-white-environment',
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
      rc.dispatchFrame({
        sunDirection: [0, 1, 0],
        sunColor: [0, 0, 0],
        frameSeed: 7,
        triIntersectEpsilon: 1e-5,
        envTextureView: environment.createView(),
        envSampler: device.createSampler({ minFilter: 'linear', magFilter: 'linear' }),
      });
      await device.queue.onSubmittedWorkDone();
      expect(await device.popErrorScope()).toBeNull();

      const rcInputs = rc.buildRCInputs(1);
      expect(rcInputs).not.toBeNull();
      const enabledParams = rcInputs!.paramsBytes;
      const disabledParams = enabledParams.slice(0);
      new Uint32Array(disabledParams)[7] = 0;

      const enabledBuffer = createUploadedBuffer(
        device,
        'hybrid-rc-enabled-params',
        enabledParams,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      );
      const disabledBuffer = createUploadedBuffer(
        device,
        'hybrid-rc-disabled-params',
        disabledParams,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      );
      const frameBuffer = createUploadedBuffer(
        device,
        'hybrid-rc-frame-seed',
        new Uint32Array([7, 0, 0, 0]).buffer,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      );
      const outputBuffer = device.createBuffer({
        label: 'hybrid-rc-sample-output',
        size: 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      ownedBuffers.push(enabledBuffer, disabledBuffer, frameBuffer, outputBuffer);

      const shader = /* wgsl */`
${OCTAHEDRAL_CORE_WGSL}
${PCG_HASH_TO_F32_WGSL}
const INV_PI: f32 = 0.3183098861837907;
struct SampleFrame { frameSeed: u32, _pad0: u32, _pad1: u32, _pad2: u32 };
@group(0) @binding(0) var<uniform> ubo: SampleFrame;
@group(0) @binding(1) var<storage, read_write> sampleOutput: array<vec4f>;
${SAMPLE_CASCADE_C0_WGSL}
@compute @workgroup_size(1)
fn sampleMain() {
  sampleOutput[0] = vec4f(
    sampleCascadeC0(vec3f(0.0, 0.0, 0.0000005), vec3f(0.0, 1.0, 0.0)),
    1.0,
  );
}`;
      const module = device.createShaderModule({ label: 'hybrid-rc-sample', code: shader });
      const compilation = await module.getCompilationInfo();
      expect(compilation.messages.filter(message => message.type === 'error')).toEqual([]);

      const sampleLayout = device.createBindGroupLayout({
        label: 'hybrid-rc-sample-bgl',
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ],
      });
      const emptyLayout = device.createBindGroupLayout({ entries: [] });
      const rcLayout = device.createBindGroupLayout({
        label: 'hybrid-rc-receiver-bgl',
        entries: [
          { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
          { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ],
      });
      const pipeline = await device.createComputePipelineAsync({
        label: 'hybrid-rc-receiver',
        layout: device.createPipelineLayout({
          bindGroupLayouts: [sampleLayout, emptyLayout, emptyLayout, rcLayout],
        }),
        compute: { module, entryPoint: 'sampleMain' },
      });
      const sampleGroup = device.createBindGroup({
        layout: sampleLayout,
        entries: [
          { binding: 0, resource: { buffer: frameBuffer } },
          { binding: 1, resource: { buffer: outputBuffer } },
        ],
      });
      const emptyGroup = device.createBindGroup({ layout: emptyLayout, entries: [] });
      const makeRcGroup = (params: GPUBuffer): GPUBindGroup => device.createBindGroup({
        layout: rcLayout,
        entries: [
          { binding: 4, resource: { buffer: rcInputs!.cascade0Buffer } },
          { binding: 5, resource: { buffer: params } },
        ],
      });

      const runReceiver = async (params: GPUBuffer): Promise<Float32Array> => {
        const encoder = device.createCommandEncoder({ label: 'hybrid-rc-receiver' });
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, sampleGroup);
        pass.setBindGroup(1, emptyGroup);
        pass.setBindGroup(2, emptyGroup);
        pass.setBindGroup(3, makeRcGroup(params));
        pass.dispatchWorkgroups(1);
        pass.end();
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        return readVec4(device, outputBuffer);
      };

      device.pushErrorScope('validation');
      const off = await runReceiver(disabledBuffer);
      const on = await runReceiver(enabledBuffer);
      expect(await device.popErrorScope()).toBeNull();
      expect(Array.from(off.slice(0, 3))).toEqual([0, 0, 0]);
      expect(Array.from(on).every(Number.isFinite)).toBe(true);
      expect(on[0]! + on[1]! + on[2]!).toBeGreaterThan(0.01);
    } finally {
      rc.dispose();
      environment?.destroy();
      for (const buffer of ownedBuffers) buffer.destroy();
      device.destroy();
    }
  }, 30_000);
});
