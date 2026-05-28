# Headless WebGPU validation in WSL2 — findings

**Goal:** stand up a headless WebGPU adapter inside WSL2 that clears vitrum's hybrid
storage-resource limits (`maxStorageBuffersPerShaderStage >= 16`,
`maxStorageTexturesPerShaderStage >= 8` -> `hybridCanRun`; `>=10/>=5` -> pt-webgpu
full tier) so GPU validation can run locally without driving Windows Chrome by hand.

**Box:** WSL2 (kernel 6.6.87.2-microsoft-standard), Ubuntu 24.04, NVIDIA RTX 4090
host GPU exposed via /dev/dxg (NVIDIA-SMI 590.52.01 / driver 591.74). Node v24,
Playwright 1.60 with bundled Chromium present. No passwordless sudo (could not
apt install; used apt-get download + dpkg-deb -x to userspace instead).

**Required limits (verified in source):**
- walkaround-hybrid/src/pipeline/WalkaroundGPUPipeline.ts:125-132 —
  HYBRID_WEBGPU_REQUIRED_LIMITS = { maxStorageBuffersPerShaderStage: 16, maxStorageTexturesPerShaderStage: 8 }.
- tools/benchmark-runner/launchWebGpuBrowser.mjs:49-57 — tier math the repo's probe
  uses (ptWebgpuFullTier = 10/5, ptWebgpuLiteTier = 8/4, hybridCanRun = 16/8).

## TL;DR result

| Approach | Backend reached | maxStorageBuffers / Textures | hybridCanRun | ptWebgpuFull | Headless in WSL? |
|---|---|---|---|---|---|
| 1. WSLg passthrough (RTX 4090 via Vulkan) | none | - | - | - | No - no NVIDIA Vulkan ICD, no dzn |
| 2a. Chromium/Playwright Dawn -> SwiftShader (baseline) | SwiftShader | 10 / 4 | false | false | runs, fails limits |
| 2b. Chromium/Playwright Dawn -> lavapipe (forced ICD) | still SwiftShader | 10 / 4 | false | false | No - Dawn ignores ext ICD |
| 3. Mesa lavapipe (LLVMpipe Vulkan) via vulkaninfo | lavapipe | 1,000,000 / 1,000,000 | n/a | n/a | yes (installed) |
| 4a. Deno native WebGPU (wgpu-native) -> lavapipe | lavapipe | 1,000,000 / 1,000,000 | TRUE | TRUE | YES |
| 4b. Node webgpu (Dawn binding) -> lavapipe | lavapipe | 16 / 8 | TRUE | TRUE | YES |

**Bottom line:** a headless WebGPU adapter that clears the hybrid limits IS
achievable in WSL today — but only through a NATIVE WebGPU binding
(Deno/wgpu-native or the Node Dawn package) over Mesa lavapipe, NOT through the
browser/Playwright path the repo currently uses. lavapipe is a CPU rasteriser:
correct for limit-gated validation and compute, but slow — NOT a perf-benchmark
device. Hardware-speed GPU capture still needs a real adapter (native Windows
Chrome on the dGPU).

## Per-approach detail

### 1. WSLg GPU passthrough — NOT reachable for WebGPU/Vulkan
- /dev/dxg exists; /usr/lib/wsl/lib has libd3d12.so, libdxcore.so, CUDA, NVENC,
  nvidia-smi (RTX 4090, 24 GB). So compute/CUDA + D3D12 are exposed.
- But there is NO NVIDIA Vulkan ICD (find / -name 'nvidia_icd*.json' -> empty; no
  Vulkan/GLX_nvidia in /usr/lib/wsl/lib). WSL exposes the GPU to D3D12 + CUDA, not
  to a Linux Vulkan driver.
- Mesa dzn (Dozen, Vulkan-on-D3D12) — which could bridge Vulkan to the host GPU via
  libd3d12.so — is NOT packaged in Ubuntu's mesa-vulkan-drivers (dpkg-deb -c on the
  25.2.8 deb ships only libvulkan_lvp.so + lvp_icd.json; no dzn). Needs custom Mesa.
- d3d12_dri.so + zink_dri.so exist in dri/ — those are the OpenGL Gallium drivers
  (GL-on-D3D12, what WSLg uses for GL), NOT a Vulkan ICD.
- vulkaninfo --summary enumerates exactly ONE device: llvmpipe (LLVM 20.1.2),
  DRIVER_ID_MESA_LLVMPIPE, type CPU. The RTX 4090 is invisible to Vulkan.

### 2. Chromium / Playwright flags — capped at SwiftShader (10/4)
Self-contained probe on a localhost (secure) origin; replicates the repo's
readWebGpuAdapterCaps. (tools/gpu-env/probe.html + probe-adapter.mjs.)
- Baseline (--use-vulkan=swiftshader): architecture "swiftshader", 10 / 4,
  hybridCanRun false. requestDevice({maxStorageTexturesPerShaderStage:8}) ->
  OperationError: Required limit (8) is greater than the supported limit (4).
  Reproduces the reported problem exactly.
- Forced lavapipe (--use-vulkan + VK_ICD_FILENAMES/VK_DRIVER_FILES=lavapipe,
  --ignore-gpu-blocklist, --disable-software-rasterizer): STILL "swiftshader", 10/4.
  Chromium's bundled Dawn does NOT honour the external Vulkan loader / VK_ICD_FILENAMES
  and silently uses in-tree SwiftShader. chrome://gpu is blank in headless-shell.
- Conclusion: the Playwright/Chromium path CANNOT clear the hybrid limits in this WSL
  box. (The repo already knows this — playwrightWebGpu.mjs has a VITRUM_USE_WIN_CHROME=1
  escape hatch to launch Windows Chrome.)

### 3. Mesa lavapipe — generous limits, already installed
- mesa-vulkan-drivers (25.2.8) installed; libvulkan_lvp.so +
  /usr/share/vulkan/icd.d/lvp_icd.json present; libvulkan1 loader present.
- vulkaninfo raw limits: maxPerStageDescriptorStorageBuffers = 1,000,000,
  maxPerStageDescriptorStorageImages = 1,000,000 — far above 16/8.
- Functions, not just enumerates: vkcube selects "llvmpipe ... type: Cpu";
  end-to-end compute passes (approach 4).

### 4. Native WebGPU (no browser) — THE WORKING PATH
Native bindings use the SYSTEM Vulkan loader, so they pick up lavapipe directly
(with or without VK_ICD_FILENAMES, since lavapipe is the only enumerable device).

- 4a. Deno (built-in WebGPU = wgpu-native), installed userspace via
  curl -fsSL https://deno.land/install.sh | DENO_INSTALL=$HOME/.deno sh.
  deno run --unstable-webgpu -A tools/gpu-env/probe-deno.ts:
    description: "llvmpipe (LLVM 20.1.2, 256 bits)"
    maxStorageBuffersPerShaderStage: 1000000
    maxStorageTexturesPerShaderStage: 1000000
    ptWebgpuFullTier: true, ptWebgpuLiteTier: true
    hybridCanRun: true, deviceAtHybridLimits: true
  deviceAtHybridLimits:true = it actually handed out a device at 16/8, not merely
  advertised. libEGL/ZINK warnings on stderr are harmless GL-surface probing; the
  Vulkan/WebGPU path succeeds. End-to-end compute smoke (compute-smoke-deno.ts):
  WGSL compiled, storage buffer bound, dispatched, correct readback -> PASS.

- 4b. Node webgpu (Dawn binding, npm webgpu@0.4.0) over lavapipe reports 16 / 8,
  hybridCanRun:true (Dawn clamps the advertised count to 16/8 vs wgpu-native's raw
  1e6, but still exactly clears the requirement). Stays inside the repo's Node
  tooling — viable if a non-Deno path is preferred, though the package is dated.

## Limitation: lavapipe is CPU — validation, not perf
lavapipe (type CPU) runs WGSL on the CPU via LLVM. Correct and clears all limits:
good for limit-gated correctness, shader-compile validation, compute/readback
validation, small reference renders. NOT for wall-clock perf benchmarks (msPerSample
would reflect the CPU rasteriser). For real GPU perf you still need native Windows
Chrome on the RTX 4090 (the repo's VITRUM_USE_WIN_CHROME=1 / run-gpu-host-windows.mjs
path), or a Windows-side native Dawn/wgpu build.

## Setup steps (reproducible)
1. lavapipe (present here; if missing): sudo apt-get install -y mesa-vulkan-drivers.
   No-root fallback used here: apt-get download mesa-vulkan-drivers && dpkg-deb -x.
2. Deno (userspace, no root):
   curl -fsSL https://deno.land/install.sh | DENO_INSTALL=$HOME/.deno sh
3. Run validation (one command):
   tools/gpu-env/run-gpu-validation.sh          # probe + compute smoke
   tools/gpu-env/run-gpu-validation.sh probe    # adapter limits only
   The script pins VK_ICD_FILENAMES/VK_DRIVER_FILES to lavapipe and prints the
   adapter limits + hybridCanRun/ptWebgpuFullTier and the compute smoke result.

## Files in this directory
- probe.html / probe-adapter.mjs — Playwright/Chromium probe (shows SwiftShader 10/4).
- chrome-gpu-info.mjs — chrome://gpu dump helper (blank in headless-shell).
- probe-deno.ts — native WebGPU adapter probe (Deno/wgpu-native + lavapipe).
- compute-smoke-deno.ts — end-to-end WGSL compute validation on the native device.
- run-gpu-validation.sh — one-command runner (Deno + lavapipe).

## Recommendation
Cleanest path to "vitrum WebGPU validation runs in WSL with one command":
  Use Deno native WebGPU over lavapipe: tools/gpu-env/run-gpu-validation.sh.
  It clears hybridCanRun (and full pt-webgpu tier) headlessly, runs real WGSL
  compute, needs no Windows-Chrome round-trip, installs with no root.

Caveats / NOT achievable in WSL on this box:
- Hardware-GPU WebGPU is not reachable (no NVIDIA Vulkan ICD; dzn not packaged) ->
  perf benchmarks still require native Windows Chrome on the RTX 4090.
- Chromium/Playwright cannot clear the limits in WSL (Dawn locked to bundled
  SwiftShader 10/4) -> browser-driven validation needing >=16/>=8 must use the native
  path above or VITRUM_USE_WIN_CHROME=1.

Follow-up (not done — must not touch packages/*): a native-WebGPU capture entrypoint
(Deno or Node-Dawn) that benchmark-runner can target as an alternative to Playwright.
Feasible because the repo's probe only reads adapter.limits + tier booleans, which
both native paths satisfy.
