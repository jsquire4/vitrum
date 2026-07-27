// Local executable proof for the production direct-light adjoint.
// Run: deno run --unstable-webgpu --allow-read tools/gpu-env/inverse-direct-proof-deno.ts

import { PT_WEBGPU_ADJOINT_PASS_WGSL } from
  '../../packages/pt-webgpu/src/wgsl/pathTrace/adjointPass.wgsl.ts';

const PROOF_OUTPUT_BINDING = 25;
const proofWgsl = `${PT_WEBGPU_ADJOINT_PASS_WGSL}

@group(0) @binding(${PROOF_OUTPUT_BINDING})
var<storage, read_write> proofOutput: array<f32>;

fn runDirectProof(lightPdf: f32) -> DirectLightAdjoint {
  return directLightAdjointFull(
    vec3f(0.7, -0.4, 0.55),
    vec3f(0.31, 0.46, 0.67),
    0.43,
    0.27,
    vec3f(0.0, 0.0, 1.0),
    vec3f(0.0, 0.0, 1.0),
    normalize(vec3f(0.17, -0.11, 1.0)),
    normalize(vec3f(-0.29, 0.23, 1.0)),
    vec3f(0.91, 0.83, 0.74),
    0.78,
    0.22,
    0.19,
    0.16,
    0.48,
    vec3f(0.52, 0.61, 0.72),
    0.37,
    1.42,
    180.0,
    420.0,
    180.0,
    420.0,
    -1.0,
    0.34,
    0.28,
    vec3f(1.3, 0.8, 1.1),
    lightPdf,
  );
}

@compute @workgroup_size(1)
fn inverseDirectProof() {
  let finiteMis = runDirectProof(0.37);
  let delta = runDirectProof(0.0);
  proofOutput[0] = finiteMis.baseColor.x;
  proofOutput[1] = finiteMis.baseColor.y;
  proofOutput[2] = finiteMis.baseColor.z;
  proofOutput[3] = finiteMis.roughness;
  proofOutput[4] = finiteMis.metallicGrad;
  proofOutput[5] = delta.baseColor.x;
  proofOutput[6] = delta.baseColor.y;
  proofOutput[7] = delta.baseColor.z;
  proofOutput[8] = delta.roughness;
  proofOutput[9] = delta.metallicGrad;
  let emitter = emitterRadianceAdjoint(
    vec3f(0.7, -0.4, 0.55),
    vec3f(0.8, 0.6, 1.2),
    vec3f(0.2, 0.35, 0.5),
    3.4,
    0.25,
  );
  proofOutput[10] = emitter.color.x;
  proofOutput[11] = emitter.color.y;
  proofOutput[12] = emitter.color.z;
  proofOutput[13] = emitter.intensity;
}
`;

const gpu = (navigator as unknown as { gpu: GPU }).gpu;
const adapter = await gpu.requestAdapter();
if (adapter == null) throw new Error('No WebGPU adapter is available.');
const device = await adapter.requestDevice({
  requiredLimits: {
    maxStorageBuffersPerShaderStage: 20,
  },
});

device.pushErrorScope('validation');
const module = device.createShaderModule({
  label: 'vitrum.pt-webgpu.inverse-direct-proof',
  code: proofWgsl,
});
const pipeline = device.createComputePipeline({
  label: 'vitrum.pt-webgpu.inverse-direct-proof.pipeline',
  layout: 'auto',
  compute: { module, entryPoint: 'inverseDirectProof' },
});
const byteLength = 14 * Float32Array.BYTES_PER_ELEMENT;
const output = device.createBuffer({
  label: 'vitrum.pt-webgpu.inverse-direct-proof.output',
  size: byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
});
const readback = device.createBuffer({
  label: 'vitrum.pt-webgpu.inverse-direct-proof.readback',
  size: byteLength,
  usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
});
const bindGroup = device.createBindGroup({
  layout: pipeline.getBindGroupLayout(0),
  entries: [{ binding: PROOF_OUTPUT_BINDING, resource: { buffer: output } }],
});

const encoder = device.createCommandEncoder();
const pass = encoder.beginComputePass();
pass.setPipeline(pipeline);
pass.setBindGroup(0, bindGroup);
pass.dispatchWorkgroups(1);
pass.end();
encoder.copyBufferToBuffer(output, 0, readback, 0, byteLength);
device.queue.submit([encoder.finish()]);
await readback.mapAsync(GPUMapMode.READ);
const gradients = [...new Float32Array(readback.getMappedRange().slice(0))];
readback.unmap();

const validationError = await device.popErrorScope();
if (validationError != null) throw validationError;
const finite = gradients.every(Number.isFinite);
const hasSignal = gradients.every((value) => Math.abs(value) > 1e-7);
const finiteMisDiffersFromDelta = gradients.slice(0, 5).some(
  (value, index) => Math.abs(value - gradients[index + 5]!) > 1e-5,
);
const expectedEmitter = [
  0.7 * 0.8 * 3.4 * 0.25,
  -0.4 * 0.6 * 3.4 * 0.25,
  0.55 * 1.2 * 3.4 * 0.25,
  (0.7 * 0.8 * 0.2 + -0.4 * 0.6 * 0.35 + 0.55 * 1.2 * 0.5) * 0.25,
];
const emitter = gradients.slice(10);
const emitterMatches = emitter.every(
  (value, index) => Math.abs(value - expectedEmitter[index]!) <= 1e-6,
);
const passed = finite && hasSignal && finiteMisDiffersFromDelta && emitterMatches;
console.log(JSON.stringify({
  adapter: adapter.info,
  finiteMis: gradients.slice(0, 5),
  delta: gradients.slice(5, 10),
  emitter,
  expectedEmitter,
  finite,
  hasSignal,
  finiteMisDiffersFromDelta,
  emitterMatches,
  passed,
}, null, 2));

output.destroy();
readback.destroy();
device.destroy();
if (!passed) Deno.exit(1);
