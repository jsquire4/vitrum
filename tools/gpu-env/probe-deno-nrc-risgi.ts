// Real-WebGPU compile + dispatch proof for the exact NRC-enabled ReSTIR-GI
// producer used by walkaround-hybrid.
//
// Run against a selected Vulkan ICD, for example lavapipe:
//   VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
//   VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json \
//   deno run --unstable-webgpu --sloppy-imports -A \
//     tools/gpu-env/probe-deno-nrc-risgi.ts

import {
  NRC_WEBGPU_REQUIRED_LIMITS,
} from "../../packages/walkaround-hybrid/src/pipeline/WalkaroundGPUPipeline.ts";
import {
  getHybridLayersBindGroupLayout,
  getNrcBindGroupLayout,
  getRisGiFrameBindGroupLayout,
  getSceneBindGroupLayout,
  getUboBindGroupLayout,
} from "../../packages/walkaround-hybrid/src/pipeline/bindGroupLayouts.ts";
import {
  WGSL_MODULES,
} from "../../packages/walkaround-hybrid/src/pipeline/wgslModules.ts";
import {
  composeWgsl,
} from "../../packages/walkaround-hybrid/src/pipeline/wgslComposer.ts";
import {
  buildRisGiNrcModule,
} from "../../packages/walkaround-hybrid/src/shaders/risGiNrc.wgsl.ts";
import {
  DEFAULT_NRC_CONFIG,
} from "../../packages/walkaround-hybrid/src/neural/nrc/nrcSubsystem.ts";

const gpu = (navigator as unknown as { gpu?: GPU }).gpu;
if (gpu == null) {
  console.log(JSON.stringify({ ok: false, reason: "no navigator.gpu" }, null, 2));
  Deno.exit(1);
}

const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
if (adapter == null) {
  console.log(JSON.stringify({ ok: false, reason: "requestAdapter returned null" }, null, 2));
  Deno.exit(1);
}

const adapterInfo = (adapter as unknown as { info?: GPUAdapterInfo }).info ?? {};
const requiredLimits = { ...NRC_WEBGPU_REQUIRED_LIMITS };
const unavailable = Object.entries(requiredLimits).filter(
  ([key, required]) =>
    Number((adapter.limits as unknown as Record<string, number>)[key] ?? 0) < required,
);
if (unavailable.length > 0) {
  console.log(JSON.stringify({
    ok: false,
    phase: "adapter-limits",
    vendor: adapterInfo.vendor ?? "",
    architecture: adapterInfo.architecture ?? "",
    description: adapterInfo.description ?? "",
    requiredLimits,
    unavailable: unavailable.map(([key, required]) => ({
      key,
      required,
      actual: Number((adapter.limits as unknown as Record<string, number>)[key] ?? 0),
    })),
  }, null, 2));
  Deno.exit(2);
}

const device = await adapter.requestDevice({ requiredLimits });
const resources: Array<{ destroy?: () => void }> = [];
const buffer = (usage: GPUBufferUsageFlags, size = 256): GPUBuffer => {
  const value = device.createBuffer({ size, usage });
  resources.push(value);
  return value;
};
const floatTexture = (arrayLayerCount = 1): GPUTextureView => {
  const value = device.createTexture({
    size: { width: 1, height: 1, depthOrArrayLayers: arrayLayerCount },
    format: "rgba32float",
    usage: GPUTextureUsage.TEXTURE_BINDING,
  });
  resources.push(value);
  return value.createView({
    dimension: arrayLayerCount > 1 ? "2d-array" : "2d",
    arrayLayerCount,
  });
};
const uintTexture = (): GPUTextureView => {
  const value = device.createTexture({
    size: { width: 1, height: 1 },
    format: "r32uint",
    usage: GPUTextureUsage.TEXTURE_BINDING,
  });
  resources.push(value);
  return value.createView();
};
const storage = (): GPUBuffer =>
  buffer(GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
const uniform = (size = 256): GPUBuffer =>
  buffer(GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, size);
const sampler = device.createSampler({
  minFilter: "nearest",
  magFilter: "nearest",
  mipmapFilter: "nearest",
});

const cache = {};
const risGiFrameLayout = getRisGiFrameBindGroupLayout(device, cache);
const sceneLayout = getSceneBindGroupLayout(device, cache);
const uboLayout = getUboBindGroupLayout(device, cache);
const hybridLayout = getHybridLayersBindGroupLayout(device, cache);
const nrcLayout = getNrcBindGroupLayout(device, cache);

const source = composeWgsl(buildRisGiNrcModule({
  levels: DEFAULT_NRC_CONFIG.levels,
  featuresPerEntry: DEFAULT_NRC_CONFIG.featuresPerEntry,
  oneBlobBins: DEFAULT_NRC_CONFIG.oneBlobBins,
  width: DEFAULT_NRC_CONFIG.width,
  outWidth: 3,
  hidden: DEFAULT_NRC_CONFIG.hidden,
}), WGSL_MODULES);
const shader = device.createShaderModule({ label: "risGi-nrc-real-probe", code: source });
const compilation = await shader.getCompilationInfo();
const compilationErrors = compilation.messages
  .filter((message) => message.type === "error")
  .map((message) => ({
    lineNum: message.lineNum,
    linePos: message.linePos,
    message: message.message,
  }));
if (compilationErrors.length > 0) {
  console.log(JSON.stringify({
    ok: false,
    phase: "shader-compilation",
    requiredLimits,
    compilationErrors,
  }, null, 2));
  Deno.exit(3);
}

device.pushErrorScope("validation");
const layout = device.createPipelineLayout({
  bindGroupLayouts: [
    risGiFrameLayout,
    sceneLayout,
    uboLayout,
    hybridLayout,
    nrcLayout,
  ],
});
const pipeline = await device.createComputePipelineAsync({
  label: "risGi-nrc-real-probe",
  layout,
  compute: { module: shader, entryPoint: "risGiMain" },
});

const frameGroup = device.createBindGroup({
  layout: risGiFrameLayout,
  entries: [
    { binding: 10, resource: floatTexture() },
    { binding: 11, resource: { buffer: storage() } },
  ],
});

const sceneStorageBindings = new Set([0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]);
const sceneEntries: GPUBindGroupEntry[] = [];
for (let binding = 0; binding <= 23; binding++) {
  if (sceneStorageBindings.has(binding)) {
    sceneEntries.push({ binding, resource: { buffer: storage() } });
  } else if (binding === 5 || binding === 14) {
    sceneEntries.push({ binding, resource: uintTexture() });
  } else if (binding === 18) {
    sceneEntries.push({ binding, resource: sampler });
  } else if (binding === 19) {
    sceneEntries.push({ binding, resource: { buffer: uniform(32) } });
  } else if (binding === 20) {
    sceneEntries.push({ binding, resource: floatTexture(1) });
  } else {
    sceneEntries.push({ binding, resource: floatTexture() });
  }
}
const sceneGroup = device.createBindGroup({ layout: sceneLayout, entries: sceneEntries });

const uboGroup = device.createBindGroup({
  layout: uboLayout,
  entries: [
    { binding: 0, resource: { buffer: uniform(432) } },
    { binding: 1, resource: floatTexture() },
    { binding: 2, resource: uintTexture() },
  ],
});
const hybridGroup = device.createBindGroup({
  layout: hybridLayout,
  entries: [
    { binding: 0, resource: floatTexture() },
    { binding: 1, resource: floatTexture() },
    { binding: 2, resource: sampler },
    { binding: 3, resource: { buffer: uniform(64) } },
    { binding: 4, resource: { buffer: storage() } },
    { binding: 5, resource: { buffer: uniform(64) } },
    { binding: 6, resource: { buffer: storage() } },
    { binding: 7, resource: { buffer: storage() } },
    { binding: 8, resource: { buffer: storage() } },
  ],
});
const nrcGroup = device.createBindGroup({
  layout: nrcLayout,
  entries: [
    { binding: 0, resource: { buffer: storage() } },
    { binding: 1, resource: { buffer: storage() } },
    { binding: 2, resource: { buffer: storage() } },
    { binding: 3, resource: { buffer: storage() } },
    { binding: 4, resource: { buffer: storage() } },
    { binding: 5, resource: { buffer: uniform(256) } },
    { binding: 6, resource: { buffer: storage() } },
    { binding: 7, resource: { buffer: storage() } },
  ],
});

const encoder = device.createCommandEncoder({ label: "risGi-nrc-real-probe" });
const pass = encoder.beginComputePass({ label: "risGi-nrc-real-probe" });
pass.setPipeline(pipeline);
pass.setBindGroup(0, frameGroup);
pass.setBindGroup(1, sceneGroup);
pass.setBindGroup(2, uboGroup);
pass.setBindGroup(3, hybridGroup);
pass.setBindGroup(4, nrcGroup);
// The zeroed WalkaroundUBO reports screenSize=(0,0), so this real dispatch
// takes risGiMain's bounds-return without touching the intentionally minimal
// resources. Pipeline/layout/bind-group/command validation still executes.
pass.dispatchWorkgroups(1, 1, 1);
pass.end();
device.queue.submit([encoder.finish()]);
await device.queue.onSubmittedWorkDone();
const validationError = await device.popErrorScope();

console.log(JSON.stringify({
  ok: validationError == null,
  phase: "compiled-and-dispatched",
  vendor: adapterInfo.vendor ?? "",
  architecture: adapterInfo.architecture ?? "",
  description: adapterInfo.description ?? "",
  requiredLimits,
  adapterMaxStorageBuffersPerShaderStage:
    adapter.limits.maxStorageBuffersPerShaderStage,
  deviceMaxStorageBuffersPerShaderStage:
    device.limits.maxStorageBuffersPerShaderStage,
  compilationWarnings: compilation.messages
    .filter((message) => message.type === "warning")
    .map((message) => message.message),
  validationError: validationError?.message ?? null,
}, null, 2));

for (const resource of resources) resource.destroy?.();
device.destroy();
Deno.exit(validationError == null ? 0 : 4);
