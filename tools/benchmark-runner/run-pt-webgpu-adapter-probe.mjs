/**
 * Reports WebGPU adapter limits in headless Chromium (no vitrum render).
 */

import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { WEBGPU_CHROMIUM_LAUNCH } from './playwrightWebGpu.mjs';
import { launchDevServer, stopDevServer, waitForServerReady } from './devServer.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..', '..');

let chromium;
try {
  ({ chromium } = await import('playwright'));
} catch (error) {
  console.error(error);
  process.exit(3);
}

let devServer = null;
let probeUrl = process.env.VITRUM_PROBE_URL;
if (process.env.VITRUM_PROBE_START_SERVER === '1') {
  const cmd =
    process.env.VITRUM_PROBE_DEV_CMD ??
    'npm run dev --workspace @vitrum-examples/two-engines-one-scene -- --host 127.0.0.1 --port 5175';
  devServer = launchDevServer(cmd, repoRoot);
  const ready = await waitForServerReady(devServer, 'http://127.0.0.1:5175/', 90_000, 500);
  const base = ready.url.endsWith('/') ? ready.url : `${ready.url}/`;
  probeUrl = `${base}pt-webgpu.html`;
}

if (probeUrl == null) {
  console.error(
    'Set VITRUM_PROBE_START_SERVER=1 or VITRUM_PROBE_URL to a page with a secure origin (WebGPU needs http://localhost or https).',
  );
  process.exit(2);
}

const browser = await chromium.launch(WEBGPU_CHROMIUM_LAUNCH);
const page = await browser.newPage();
await page.goto(probeUrl, { waitUntil: 'domcontentloaded', timeout: 60_000 });
const report = await page.evaluate(async () => {
  if (!navigator.gpu) return { ok: false, reason: 'no navigator.gpu' };
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return { ok: false, reason: 'requestAdapter returned null' };
  const limits = adapter.limits;
  return {
    ok: true,
    maxStorageBuffersPerShaderStage: limits.maxStorageBuffersPerShaderStage,
    maxStorageTexturesPerShaderStage: limits.maxStorageTexturesPerShaderStage,
    ptWebgpuFullRequiredBuffers: 23,
    ptWebgpuLiteRequiredBuffers: 8,
    ptWebgpuFullTier: limits.maxStorageBuffersPerShaderStage >= 23,
    ptWebgpuLiteTier:
      limits.maxStorageBuffersPerShaderStage >= 8 &&
      limits.maxStorageTexturesPerShaderStage >= 4,
    ptWebgpuCanRun:
      limits.maxStorageBuffersPerShaderStage >= 23 ||
      (limits.maxStorageBuffersPerShaderStage >= 8 &&
        limits.maxStorageTexturesPerShaderStage >= 4),
    hybridTextureRequest: 8,
    hybridDeviceLikely:
      limits.maxStorageTexturesPerShaderStage >= 4,
  };
});
await browser.close();
if (devServer) stopDevServer(devServer);

const out = { probeUrl, ...report };
console.log(JSON.stringify(out, null, 2));
process.exit(report.ok ? 0 : 1);
