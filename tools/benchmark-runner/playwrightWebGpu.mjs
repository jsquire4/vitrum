/**
 * Shared Playwright Chromium launch options for WebGPU capture/bench runners.
 *
 * Default (`VITRUM_WEBGPU_ADAPTER=auto`): try hardware Dawn/Vulkan first (WSL
 * `/dev/dxg`, native Linux GPU), then fall back to SwiftShader for CI.
 * Force with `VITRUM_WEBGPU_ADAPTER=hardware` or `swiftshader`.
 */

const jsHeapMb = Number(process.env.VITRUM_JS_HEAP_MB ?? '4096');
const headless = process.env.VITRUM_BENCH_HEADLESS !== '0';

const commonArgs = [
  '--disable-dev-shm-usage',
  `--js-flags=--max-old-space-size=${jsHeapMb}`,
  '--enable-unsafe-webgpu',
  '--enable-features=Vulkan,UseSkiaRenderer',
];

/** Software ICD — caps at 10 storage buffers / 4 textures (pt-webgpu lite tier). */
export const WEBGPU_SWIFTSHADER_CHROMIUM_LAUNCH = {
  headless,
  args: [
    ...commonArgs,
    '--use-vulkan=swiftshader',
    '--enable-unsafe-swiftshader',
    '--use-gl=angle',
    '--use-angle=swiftshader',
  ],
};

/** Discrete GPU path — use headed mode on Windows/WSLg when VITRUM_BENCH_HEADLESS=0. */
export const WEBGPU_HARDWARE_CHROMIUM_LAUNCH = {
  headless: process.env.VITRUM_BENCH_HEADLESS === '0' ? false : headless,
  args: [
    ...commonArgs,
    '--disable-software-rasterizer',
    ...(process.env.VITRUM_BENCH_HEADLESS === '0' ? [] : ['--headless=new']),
  ],
};

function adapterMode() {
  const mode = (process.env.VITRUM_WEBGPU_ADAPTER ?? 'auto').toLowerCase();
  if (mode === 'hardware' || mode === 'gpu') return 'hardware';
  if (mode === 'swiftshader' || mode === 'software') return 'swiftshader';
  return 'auto';
}

/** @type {import('playwright').LaunchOptions} */
export const WEBGPU_CHROMIUM_LAUNCH =
  adapterMode() === 'swiftshader'
    ? WEBGPU_SWIFTSHADER_CHROMIUM_LAUNCH
    : WEBGPU_HARDWARE_CHROMIUM_LAUNCH;

/** Launch order for `auto`: hardware first, then SwiftShader. */
export const WEBGPU_CHROMIUM_LAUNCH_CANDIDATES =
  adapterMode() === 'auto'
    ? [WEBGPU_HARDWARE_CHROMIUM_LAUNCH, WEBGPU_SWIFTSHADER_CHROMIUM_LAUNCH]
    : adapterMode() === 'hardware'
      ? [WEBGPU_HARDWARE_CHROMIUM_LAUNCH]
      : [WEBGPU_SWIFTSHADER_CHROMIUM_LAUNCH];
