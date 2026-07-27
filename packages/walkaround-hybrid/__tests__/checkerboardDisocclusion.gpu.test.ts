import { describe, expect, it } from 'vitest';

import { composeWgsl } from '../src/pipeline/wgslComposer.js';
import { WGSL_MODULES } from '../src/pipeline/wgslModules.js';
import { CB_PREFILL_MODULE } from '../src/shaders/cbPrefill.wgsl.js';
import { RESOLVE_MODULE } from '../src/shaders/resolve.wgsl.js';

const WIDTH = 4;
const HEIGHT = 4;
const ROW_BYTES = 256;

function f16ToF32(bits: number): number {
  const sign = (bits & 0x8000) === 0 ? 1 : -1;
  const exponent = (bits >>> 10) & 0x1f;
  const fraction = bits & 0x03ff;
  if (exponent === 0) return sign * Math.pow(2, -14) * (fraction / 1024);
  if (exponent === 0x1f) return fraction === 0 ? sign * Infinity : Number.NaN;
  return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}

async function requestDevice(): Promise<GPUDevice> {
  if (navigator.gpu == null) {
    throw new Error('Checkerboard GPU acceptance requires navigator.gpu.');
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'low-power' });
  if (adapter == null) throw new Error('Checkerboard GPU acceptance could not acquire an adapter.');
  return adapter.requestDevice();
}

function sampledTexture(device: GPUDevice, label: string, value: number): GPUTexture {
  const texture = device.createTexture({
    label,
    size: [WIDTH, HEIGHT],
    format: 'rgba32float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  const data = new Float32Array(WIDTH * HEIGHT * 4);
  for (let pixel = 0; pixel < WIDTH * HEIGHT; pixel += 1) {
    data[pixel * 4 + 0] = value;
    data[pixel * 4 + 1] = value;
    data[pixel * 4 + 2] = value;
    data[pixel * 4 + 3] = 1;
  }
  device.queue.writeTexture(
    { texture }, data, { bytesPerRow: WIDTH * 16 }, [WIDTH, HEIGHT],
  );
  return texture;
}

function outputTexture(device: GPUDevice, label: string): GPUTexture {
  return device.createTexture({
    label,
    size: [WIDTH, HEIGHT],
    format: 'rgba16float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
  });
}

async function readRgb(
  device: GPUDevice,
  texture: GPUTexture,
  x: number,
  y: number,
): Promise<readonly [number, number, number]> {
  const buffer = device.createBuffer({
    label: 'checkerboard-disocclusion-readback',
    size: ROW_BYTES * HEIGHT,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer(
      { texture },
      { buffer, bytesPerRow: ROW_BYTES, rowsPerImage: HEIGHT },
      [WIDTH, HEIGHT],
    );
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const words = new Uint16Array(buffer.getMappedRange());
    const wordOffset = (y * ROW_BYTES + x * 8) / 2;
    return [
      f16ToF32(words[wordOffset]!),
      f16ToF32(words[wordOffset + 1]!),
      f16ToF32(words[wordOffset + 2]!),
    ];
  } finally {
    if (buffer.mapState === 'mapped') buffer.unmap();
    buffer.destroy();
  }
}

describe('checkerboard disocclusion reconstruction (real WebGPU)', () => {
  it('rejects zero-motion stale history in both resolve and pre-denoiser paths', async () => {
    const device = await requestDevice();
    const ownedTextures: GPUTexture[] = [];
    const ownedBuffers: GPUBuffer[] = [];
    try {
      const current = sampledTexture(device, 'checkerboard-current', 0.2);
      const staleHistory = sampledTexture(device, 'checkerboard-stale-history', 20);
      const zeroMotion = sampledTexture(device, 'checkerboard-zero-motion', 0);
      const resolved = outputTexture(device, 'checkerboard-resolved');
      const prefilled = outputTexture(device, 'checkerboard-prefilled');
      ownedTextures.push(current, staleHistory, zeroMotion, resolved, prefilled);

      const uniform = device.createBuffer({
        label: 'checkerboard-test-uniform',
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      ownedBuffers.push(uniform);
      device.queue.writeBuffer(uniform, 0, new Uint32Array([WIDTH, HEIGHT, 0, 1]));

      device.pushErrorScope('validation');
      const resolveModule = device.createShaderModule({
        label: 'checkerboard-resolve-acceptance',
        code: composeWgsl(RESOLVE_MODULE, WGSL_MODULES),
      });
      const prefillModule = device.createShaderModule({
        label: 'checkerboard-prefill-acceptance',
        code: composeWgsl(CB_PREFILL_MODULE, WGSL_MODULES),
      });
      for (const module of [resolveModule, prefillModule]) {
        const info = await module.getCompilationInfo();
        expect(info.messages.filter(message => message.type === 'error')).toEqual([]);
      }
      const resolvePipeline = await device.createComputePipelineAsync({
        layout: 'auto',
        compute: { module: resolveModule, entryPoint: 'resolveKernel' },
      });
      const prefillPipeline = await device.createComputePipelineAsync({
        layout: 'auto',
        compute: { module: prefillModule, entryPoint: 'cbPrefillKernel' },
      });
      const resolveGroup = device.createBindGroup({
        layout: resolvePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniform } },
          { binding: 1, resource: current.createView() },
          { binding: 2, resource: staleHistory.createView() },
          { binding: 3, resource: zeroMotion.createView() },
          { binding: 4, resource: resolved.createView() },
        ],
      });
      const prefillGroup = device.createBindGroup({
        layout: prefillPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniform } },
          { binding: 1, resource: staleHistory.createView() },
          { binding: 2, resource: zeroMotion.createView() },
          { binding: 3, resource: current.createView() },
          { binding: 4, resource: prefilled.createView() },
        ],
      });
      const encoder = device.createCommandEncoder({ label: 'checkerboard-disocclusion' });
      for (const [pipeline, group] of [
        [resolvePipeline, resolveGroup],
        [prefillPipeline, prefillGroup],
      ] as const) {
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, group);
        pass.dispatchWorkgroups(1, 1);
        pass.end();
      }
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();

      // (1, 2) is gap parity for frameParity=0 and has four fresh neighbours.
      for (const rgb of [
        await readRgb(device, resolved, 1, 2),
        await readRgb(device, prefilled, 1, 2),
      ]) {
        expect(rgb.every(Number.isFinite)).toBe(true);
        expect(rgb[0]).toBeCloseTo(0.2, 2);
        expect(rgb[1]).toBeCloseTo(0.2, 2);
        expect(rgb[2]).toBeCloseTo(0.2, 2);
      }
      const validationError = await device.popErrorScope();
      if (validationError != null) throw new Error(validationError.message);
    } finally {
      for (const buffer of ownedBuffers) buffer.destroy();
      for (const texture of ownedTextures) texture.destroy();
      device.destroy();
    }
  }, 30_000);
});
