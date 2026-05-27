/**
 * Reports WebGPU adapter limits in Chromium (tries hardware GPU before SwiftShader).
 */

import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { launchWebGpuBrowser } from './launchWebGpuBrowser.mjs';
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

const probeOrigin = new URL(probeUrl).origin + '/';
const { browser, profile, caps, attempts } = await launchWebGpuBrowser(chromium, probeOrigin);
await browser.close();
if (devServer) stopDevServer(devServer);

const report = {
  probeUrl,
  launchProfile: profile,
  attempts,
  ok: caps.ok,
  vendor: caps.vendor,
  architecture: caps.architecture,
  description: caps.description,
  maxStorageBuffersPerShaderStage: caps.maxStorageBuffersPerShaderStage,
  maxStorageTexturesPerShaderStage: caps.maxStorageTexturesPerShaderStage,
  ptWebgpuFullRequiredBuffers: 10,
  ptWebgpuFullRequiredTextures: 5,
  ptWebgpuLiteRequiredBuffers: 8,
  ptWebgpuFullTier: caps.ptWebgpuFullTier,
  ptWebgpuLiteTier: caps.ptWebgpuLiteTier,
  ptWebgpuCanRun: caps.ptWebgpuFullTier || caps.ptWebgpuLiteTier,
  hybridCanRun: caps.hybridCanRun,
  hybridTextureRequest: 8,
};

console.log(JSON.stringify(report, null, 2));
process.exit(report.ok ? 0 : 1);
