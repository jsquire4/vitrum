/**
 * Shared Playwright Chromium launch options for WebGPU capture/bench runners.
 * Mirrors packages/pt-webgpu/vitest.gpu.config.ts (SwiftShader Vulkan ICD).
 */

const jsHeapMb = Number(process.env.VITRUM_JS_HEAP_MB ?? '4096');

/** @type {import('playwright').LaunchOptions} */
export const WEBGPU_CHROMIUM_LAUNCH = {
  headless: process.env.VITRUM_BENCH_HEADLESS !== '0',
  args: [
    '--disable-dev-shm-usage',
    `--js-flags=--max-old-space-size=${jsHeapMb}`,
    '--enable-unsafe-webgpu',
    '--enable-features=Vulkan,UseSkiaRenderer',
    '--use-vulkan=swiftshader',
    '--enable-unsafe-swiftshader',
    '--use-gl=angle',
    '--use-angle=swiftshader',
  ],
};
