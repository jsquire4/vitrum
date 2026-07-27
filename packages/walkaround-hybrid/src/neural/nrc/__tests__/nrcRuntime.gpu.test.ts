import { describe, expect, it } from 'vitest';

import type { BGLCache } from '../../../bglTypes.js';
import { NrcSubsystem } from '../nrcSubsystem.js';
import { nrcEncodeHelpersWgsl } from '../wgsl/nrcEncoding.wgsl.js';
import { nrcQueryWgsl } from '../wgsl/nrcQuery.wgsl.js';

async function requestNrcDevice(): Promise<GPUDevice> {
  if (navigator.gpu == null) {
    throw new Error('NRC GPU acceptance requires navigator.gpu.');
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'low-power' });
  if (adapter == null) {
    throw new Error('NRC GPU acceptance could not acquire a WebGPU adapter.');
  }
  if (adapter.limits.maxStorageBuffersPerShaderStage < 8) {
    throw new Error(
      'NRC GPU acceptance requires the WebGPU baseline of eight storage buffers; '
      + `adapter exposes ${adapter.limits.maxStorageBuffersPerShaderStage}.`,
    );
  }
  return adapter.requestDevice({
    requiredLimits: { maxStorageBuffersPerShaderStage: 8 },
  });
}

describe('NRC query and online learning (real WebGPU)', () => {
  it('queries the published arena, gathers a record, and trains both network and hash grid', async () => {
    const device = await requestNrcDevice();
    const cache: BGLCache = {};
    const nrc = new NrcSubsystem(device, cache, {
      levels: 2,
      featuresPerEntry: 2,
      tableSize: 16,
      nMin: 2,
      growth: 2,
      oneBlobBins: 2,
      width: 16,
      hidden: 1,
      spreadC: 0.01,
      recordCap: 4,
      learningRate: 0.01,
      tableLearningRate: 0.1,
      useF16: false,
      tileB: 4,
      warmupSteps: 0,
    });
    let output: GPUBuffer | undefined;
    let readback: GPUBuffer | undefined;
    try {
      device.pushErrorScope('validation');
      await nrc.initialize([-1, -1, -1], [1, 1, 1]);
      expect(nrc.lifecycleState).toBe('ready');

      output = device.createBuffer({
        label: 'nrc-runtime-query-output',
        size: 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      readback = device.createBuffer({
        label: 'nrc-runtime-query-readback',
        size: 16,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const shader = /* wgsl */`
${nrcEncodeHelpersWgsl()}
${nrcQueryWgsl({ ...nrc.wgslConfig(), group: 0 })}
@group(0) @binding(10) var<storage, read_write> nrcAcceptanceOutput: array<vec4f>;
@compute @workgroup_size(1)
fn nrcAcceptanceMain() {
  let position = vec3f(0.25, -0.125, 0.5);
  let normal = normalize(vec3f(0.2, 0.9, 0.3));
  let viewDir = normalize(vec3f(-0.4, 0.3, 0.8));
  let albedo = vec3f(0.4, 0.6, 0.8);
  let prediction = nrcQueryRadiance(position, normal, viewDir, 0.35, albedo);
  nrcWriteRecord(0u, position, normal, viewDir, 0.35, albedo, vec3f(0.25, 0.5, 0.75));
  nrcAcceptanceOutput[0] = vec4f(prediction, 1.0);
}`;
      const module = device.createShaderModule({
        label: 'nrc-runtime-query-module',
        code: shader,
      });
      const compilation = await module.getCompilationInfo();
      expect(compilation.messages.filter(message => message.type === 'error')).toEqual([]);
      const pipeline = await device.createComputePipelineAsync({
        label: 'nrc-runtime-query-pipeline',
        layout: 'auto',
        compute: { module, entryPoint: 'nrcAcceptanceMain' },
      });
      const bindings = nrc.queryBindings();
      const bindGroup = device.createBindGroup({
        label: 'nrc-runtime-query-bind-group',
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 7, resource: { buffer: bindings.inferenceArenaBuffer } },
          { binding: 8, resource: { buffer: bindings.runtimeArenaBuffer } },
          { binding: 9, resource: { buffer: bindings.configBuffer } },
          { binding: 10, resource: { buffer: output } },
        ],
      });

      const encoder = device.createCommandEncoder({ label: 'nrc-runtime-acceptance' });
      nrc.clearSlotClaims(encoder);
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(1);
      pass.end();
      encoder.copyBufferToBuffer(output, 0, readback, 0, 16);
      nrc.recordCopyForReadback(encoder);
      device.queue.submit([encoder.finish()]);

      await readback.mapAsync(GPUMapMode.READ);
      const prediction = new Float32Array(readback.getMappedRange().slice(0));
      readback.unmap();
      expect(Array.from(prediction).every(Number.isFinite)).toBe(true);
      expect(prediction[3]).toBe(1);

      const beforeTraining = nrc.queryBindings().inferenceArenaBuffer;
      await nrc.trainFromRecords();
      await device.queue.onSubmittedWorkDone();
      const afterTraining = nrc.queryBindings().inferenceArenaBuffer;
      expect(afterTraining).not.toBe(beforeTraining);
      const diagnostics = nrc.diagnostics();
      expect(diagnostics.trainedSteps).toBe(1);
      expect(diagnostics.nonFiniteValues).toBe(0);
      expect(diagnostics.invalidPdfs).toBe(0);
      expect(diagnostics.trainingFailures).toBe(0);
      const validationError = await device.popErrorScope();
      if (validationError != null) throw new Error(validationError.message);
    } finally {
      if (readback?.mapState === 'mapped') readback.unmap();
      readback?.destroy();
      output?.destroy();
      nrc.dispose();
      device.destroy();
    }
  }, 30_000);
});
