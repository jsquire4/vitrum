/**
 * capturePtWebgpu.mjs — WG-0.1 headless pt-webgpu HDR capture via two-engines example.
 *
 * Writes PNG to VITRUM_OUTPUT_PNG and prints JSON telemetry on stdout.
 *
 * Prereq:
 *   npm run dev --workspace @vitrum-examples/two-engines-one-scene
 *
 * Env:
 *   VITRUM_OUTPUT_PNG          required output path
 *   VITRUM_CAPTURE_URL         default http://127.0.0.1:5175/pt-webgpu.html
 *   VITRUM_SCENARIO_ID         optional label for telemetry
 *   VITRUM_SPP                 samples target (default 64)
 *   VITRUM_CAPTURE_TIMEOUT_MS  default 120000
 */

import { mkdir, readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { dirname } from 'node:path';
import { launchWebGpuBrowser } from './launchWebGpuBrowser.mjs';

const outputPng = process.env.VITRUM_OUTPUT_PNG;
if (!outputPng) {
  console.error('VITRUM_OUTPUT_PNG is required.');
  process.exit(2);
}

const captureUrlBase = process.env.VITRUM_CAPTURE_URL ?? 'http://127.0.0.1:5175/pt-webgpu.html';
const samplesTarget = Number(process.env.VITRUM_SPP ?? '64');
const bounces = Number(process.env.VITRUM_BOUNCES ?? '8');
const width = Number(process.env.VITRUM_WIDTH ?? '1280');
const height = Number(process.env.VITRUM_HEIGHT ?? '720');
const seed = process.env.VITRUM_SEED ?? '777';
const timeoutMs = Number(process.env.VITRUM_CAPTURE_TIMEOUT_MS ?? 120_000);
const scenarioId = process.env.VITRUM_SCENARIO_ID ?? 'ptwgpu-capture';

function buildUrl() {
  const u = new URL(captureUrlBase);
  u.searchParams.set('mode', 'ptwebgpu');
  u.searchParams.set('samplesTarget', String(samplesTarget));
  u.searchParams.set('ptWebgpuBounces', String(bounces));
  u.searchParams.set('vitrumSeed', String(seed));
  if (process.env.VITRUM_SCENE) {
    u.searchParams.set('scene', process.env.VITRUM_SCENE);
  }
  return u.toString();
}

let chromium;
try {
  ({ chromium } = await import('playwright'));
} catch (error) {
  console.error('Playwright required:', error);
  process.exit(3);
}

const probeOrigin = new URL(captureUrlBase).origin + '/';
const { browser, page, profile, caps } = await launchWebGpuBrowser(chromium, probeOrigin);
console.error(
  `[capturePtWebgpu] launchProfile=${profile} ptFull=${caps.ptWebgpuFullTier} ` +
    `buffers=${caps.maxStorageBuffersPerShaderStage} vendor=${caps.vendor ?? ''}`,
);
try {
  await page.setViewportSize({ width, height });
  const url = buildUrl();
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: timeoutMs });
  const bootError = await page.evaluate(() => {
    const status = document.querySelector('#status')?.textContent ?? '';
    if (/requestDevice|OperationError|init failed|maxStorageBuffers/i.test(status)) {
      return status.slice(0, 500);
    }
    return null;
  });
  if (bootError != null) {
    throw new Error(`pt-webgpu page failed to acquire device: ${bootError}`);
  }
  await page.waitForFunction(
    (targetSpp) => {
      const p = globalThis.__vitrum?.ptWebgpu;
      if (p == null) return false;
      const converged = p.converged === true || p.isConverged === true;
      const spp = typeof p.spp === 'number' ? p.spp : 0;
      return converged || spp >= targetSpp;
    },
    samplesTarget,
    { timeout: timeoutMs, polling: 200 },
  );

  const canvas = page.locator('#c-ptgpu');
  await canvas.waitFor({ timeout: 10_000 });
  await mkdir(dirname(outputPng), { recursive: true });
  await canvas.screenshot({ path: outputPng });

  const bytes = await readFile(outputPng);
  const hash = createHash('sha256').update(bytes).digest('hex');
  const telemetry = await page.evaluate(() => {
    const p = globalThis.__vitrum?.ptWebgpu;
    return p == null
      ? null
      : {
          spp: p.spp,
          target: p.target,
          converged: p.converged ?? p.isConverged,
          state: p.state,
        };
  });

  const report = {
    scenarioId,
    url,
    pngPath: outputPng,
    hash,
    samplesTarget,
    telemetry,
  };
  console.log(JSON.stringify(report));
} finally {
  await browser.close();
}
