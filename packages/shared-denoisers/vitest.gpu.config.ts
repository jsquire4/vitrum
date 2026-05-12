import { defineConfig } from 'vitest/config';

// Real-WebGPU test config — runs *.gpu.test.ts in headless Chromium via
// Playwright with SwiftShader as the WebGPU adapter. This is the same
// stack as @vitrum/shader-compile-ci uses for the cornell-box smoke;
// here we use it to give navigator.gpu real semantics inside vitest.
export default defineConfig({
  test: {
    include: ['__tests__/**/*.gpu.test.ts'],
    browser: {
      enabled: true,
      headless: true,
      provider: 'playwright',
      name: 'chromium',
      // SwiftShader is a software WebGPU adapter Chromium ships with —
      // gives us navigator.gpu without a physical GPU.
      providerOptions: {
        launch: {
          args: [
            // WebGPU-specific: enable SwiftShader as a Vulkan ICD so Dawn
            // (Chromium's WebGPU implementation) gets a software adapter.
            // --use-gl/--use-angle from the WebGL flow don't help here.
            '--enable-unsafe-webgpu',
            '--enable-features=Vulkan,UseSkiaRenderer',
            '--use-vulkan=swiftshader',
            '--enable-unsafe-swiftshader',
            // Pass through the WebGL2 fallback flags too in case the test
            // bridges to ANGLE for any reason (resource init, etc.).
            '--use-gl=angle',
            '--use-angle=swiftshader',
          ],
        },
      },
    },
  },
});
