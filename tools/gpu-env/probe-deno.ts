// Native WebGPU adapter probe (Deno built-in WebGPU = wgpu-native).
// wgpu-native uses the *system* Vulkan loader, so VK_ICD_FILENAMES /
// VK_DRIVER_FILES pointed at lavapipe (or a hardware ICD) take effect here —
// unlike Chromium's bundled SwiftShader-only Dawn.
//
// Tier thresholds are sourced from the pt-webgpu authority (D17-11) instead of
// re-encoding them here — webgpuLimits.ts is a dependency-free constants module
// so this standalone Deno script can import it directly by relative .ts path.
// Walkaround exposes the same kind of dependency-free authority, so this
// map-less probe does not duplicate either backend's numeric floor.
// Run: deno run --unstable-webgpu -A tools/gpu-env/probe-deno.ts

import {
  PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  PT_WEBGPU_FULL_REQUIRED_STORAGE_TEXTURES_PER_STAGE,
  PT_WEBGPU_LITE_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  PT_WEBGPU_LITE_REQUIRED_STORAGE_TEXTURES_PER_STAGE,
} from "../../packages/pt-webgpu/src/webgpuLimits.ts";
import {
  HYBRID_WEBGPU_REQUIRED_LIMITS,
} from "../../packages/walkaround-hybrid/src/webgpuLimits.ts";

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
const msamp = limits.maxSampledTexturesPerShaderStage;

let deviceAtHybridLimits = false;
let deviceErr = "";
try {
  const dev = await adapter.requestDevice({
    requiredLimits: { ...HYBRID_WEBGPU_REQUIRED_LIMITS },
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
  maxSampledTexturesPerShaderStage: msamp,
  ptWebgpuFullTier:
    msb >= PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE &&
    mst >= PT_WEBGPU_FULL_REQUIRED_STORAGE_TEXTURES_PER_STAGE,
  ptWebgpuLiteTier:
    msb >= PT_WEBGPU_LITE_REQUIRED_STORAGE_BUFFERS_PER_STAGE &&
    mst >= PT_WEBGPU_LITE_REQUIRED_STORAGE_TEXTURES_PER_STAGE,
  hybridCanRun: Object.entries(HYBRID_WEBGPU_REQUIRED_LIMITS).every(
    ([key, required]) =>
      Number((limits as unknown as Record<string, number | undefined>)[key]) >= required,
  ),
  deviceAtHybridLimits,
  deviceErr,
};
console.log(JSON.stringify(report, null, 2));
Deno.exit(report.ok ? 0 : 1);
