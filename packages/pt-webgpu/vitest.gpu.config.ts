import { defineConfig } from 'vitest/config';

// Real-WebGPU test config — runs *.gpu.test.ts in headless Chromium via
// Playwright with Vulkan/SwiftShader as the WebGPU adapter.
//
// Why these specific Chromium flags: Dawn (Chromium's WebGPU
// implementation) is built on top of Vulkan; for headless Linux without
// a real GPU, SwiftShader provides a software Vulkan ICD that Dawn
// picks up. The --use-gl/--use-angle flags handle the WebGL2 fallback
// some bridge code might use.
export default defineConfig({
  test: {
    include: ['src/__tests__/**/*.gpu.test.ts'],
    browser: {
      enabled: true,
      headless: true,
      provider: 'playwright',
      name: 'chromium',
      providerOptions: {
        launch: {
          args: [
            '--enable-unsafe-webgpu',
            '--enable-features=Vulkan,UseSkiaRenderer',
            '--use-vulkan=swiftshader',
            '--enable-unsafe-swiftshader',
            '--use-gl=angle',
            '--use-angle=swiftshader',
          ],
        },
      },
    },
  },
});
