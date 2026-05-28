// Native WebGPU adapter probe (Deno built-in WebGPU = wgpu-native).
// wgpu-native uses the *system* Vulkan loader, so VK_ICD_FILENAMES /
// VK_DRIVER_FILES pointed at lavapipe (or a hardware ICD) take effect here —
// unlike Chromium's bundled SwiftShader-only Dawn.
//
// Mirrors tools/benchmark-runner/launchWebGpuBrowser.mjs tier math.
// Run: deno run --unstable-webgpu -A tools/gpu-env/probe-deno.ts

const out: Record<string, unknown> = { runtime: "deno-native-webgpu" };

const gpu = (navigator as unknown as { gpu?: GPU }).gpu;
if (!gpu) {
  console.log(JSON.stringify({ ...out, ok: false, reason: "no navigator.gpu" }, null, 2));
  Deno.exit(1);
}

const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
if (!adapter) {
  console.log(JSON.stringify({ ...out, ok: false, reason: "requestAdapter null" }, null, 2));
  Deno.exit(1);
}

let info: GPUAdapterInfo | Record<string, string> = {};
try {
  info = (adapter as unknown as { info?: GPUAdapterInfo }).info ?? {};
} catch { /* optional */ }

const limits = adapter.limits;
const msb = limits.maxStorageBuffersPerShaderStage;
const mst = limits.maxStorageTexturesPerShaderStage;

let deviceAtHybridLimits = false;
let deviceErr = "";
try {
  const dev = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBuffersPerShaderStage: 16,
      maxStorageTexturesPerShaderStage: 8,
    },
  });
  deviceAtHybridLimits = !!dev;
  dev?.destroy?.();
} catch (e) {
  deviceErr = String(e);
}

const report = {
  ...out,
  ok: true,
  vendor: (info as Record<string, string>).vendor ?? "",
  architecture: (info as Record<string, string>).architecture ?? "",
  description: (info as Record<string, string>).description ?? "",
  device: (info as Record<string, string>).device ?? "",
  maxStorageBuffersPerShaderStage: msb,
  maxStorageTexturesPerShaderStage: mst,
  ptWebgpuFullTier: msb >= 10 && mst >= 5,
  ptWebgpuLiteTier: msb >= 8 && mst >= 4,
  hybridCanRun: msb >= 16 && mst >= 8,
  deviceAtHybridLimits,
  deviceErr,
};
console.log(JSON.stringify(report, null, 2));
Deno.exit(report.ok ? 0 : 1);
