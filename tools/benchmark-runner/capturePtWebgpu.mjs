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
import { captureQueryForScenario } from './gapClosurePtWebgpuMap.mjs';

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
  let scenarioMeta = null;
  if (process.env.VITRUM_SCENARIO_JSON) {
    try {
      scenarioMeta = JSON.parse(process.env.VITRUM_SCENARIO_JSON);
    } catch {
      scenarioMeta = null;
    }
  }
  const q =
    scenarioMeta != null
      ? captureQueryForScenario(scenarioMeta)
      : new URLSearchParams();
  q.set('mode', 'ptwebgpu');
  q.set('samplesTarget', String(samplesTarget));
  q.set('ptWebgpuBounces', String(bounces));
  q.set('vitrumSeed', String(seed));
  for (const [key, value] of q.entries()) {
    u.searchParams.set(key, value);
  }
  if (process.env.VITRUM_SCENE) {
    u.searchParams.set('scene', process.env.VITRUM_SCENE);
  }
  const caustic = process.env.VITRUM_CAUSTIC_STRATEGY?.trim();
  if (caustic === 'manifold-nee' || caustic === 'photon-map') {
    u.searchParams.set('vitrumCaustic', caustic);
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
