/**
 * Shared Playwright Chromium launch options for WebGPU capture/bench runners.
 *
 * Default (`VITRUM_WEBGPU_ADAPTER=auto`): try hardware Dawn/Vulkan first (WSL
 * `/dev/dxg`, native Linux GPU), then fall back to SwiftShader for CI.
 * Force with `VITRUM_WEBGPU_ADAPTER=hardware` or `swiftshader`.
 *
 * `VITRUM_USE_WIN_CHROME=1` — launch Windows Chrome from WSL (`/mnt/c/...`) so
 * Playwright sees hybrid-capable WebGPU limits (16/8) instead of SwiftShader (10/4).
 */

import { existsSync } from 'node:fs';

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

function resolveWindowsChromeExecutable() {
  if (process.env.VITRUM_USE_WIN_CHROME !== '1') return undefined;
  const candidates = [
    process.env.VITRUM_WIN_CHROME_EXE,
    '/mnt/c/Program Files/Google/Chrome/Application/chrome.exe',
    '/mnt/c/Program Files (x86)/Google/Chrome/Application/chrome.exe',
  ].filter((p) => typeof p === 'string' && p.length > 0);
  return candidates.find((p) => existsSync(p));
}

/** @returns {import('playwright').LaunchOptions} */
export function withWindowsChromeIfRequested(base) {
  const exe = resolveWindowsChromeExecutable();
  if (exe == null) return base;
  return { ...base, executablePath: exe };
}

/** Launch order for `auto`: hardware first, then SwiftShader. */
export const WEBGPU_CHROMIUM_LAUNCH_CANDIDATES = (
  adapterMode() === 'auto'
    ? [WEBGPU_HARDWARE_CHROMIUM_LAUNCH, WEBGPU_SWIFTSHADER_CHROMIUM_LAUNCH]
    : adapterMode() === 'hardware'
      ? [WEBGPU_HARDWARE_CHROMIUM_LAUNCH]
      : [WEBGPU_SWIFTSHADER_CHROMIUM_LAUNCH]
).map((opts) => withWindowsChromeIfRequested(opts));
