/**
 * capturePtWebgl.mjs — headless pt-webgl PNG capture via two-engines example.
 *
 * Writes PNG to VITRUM_OUTPUT_PNG and prints JSON telemetry on stdout.
 *
 * Prereq:
 *   npm run dev --workspace @vitrum-examples/two-engines-one-scene
 */

import { createHash } from 'node:crypto';
import { mkdir, readFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { launchChromiumForCapture } from './playwrightWebGpu.mjs';
import { captureQueryForScenario } from './gapClosurePtWebgpuMap.mjs';

const outputPng = process.env.VITRUM_OUTPUT_PNG;
if (!outputPng) {
  console.error('VITRUM_OUTPUT_PNG is required.');
  process.exit(2);
}

const captureUrlBase = process.env.VITRUM_CAPTURE_URL ?? 'http://127.0.0.1:5175/pt-webgl.html';
const samplesTarget = Number(process.env.VITRUM_SPP ?? '64');
const bounces = Number(process.env.VITRUM_BOUNCES ?? '8');
const width = Number(process.env.VITRUM_WIDTH ?? '1280');
const height = Number(process.env.VITRUM_HEIGHT ?? '720');
const seed = process.env.VITRUM_SEED ?? '777';
const timeoutMs = Number(process.env.VITRUM_CAPTURE_TIMEOUT_MS ?? 120_000);
const scenarioId = process.env.VITRUM_SCENARIO_ID ?? 'ptwebgl-capture';

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
  q.set('mode', 'ptwebgl');
  q.set('samplesTarget', String(samplesTarget));
  q.set('vitrumSeed', String(seed));
  q.set('vitrumBounces', String(bounces));
  if (process.env.VITRUM_SCENE) q.set('scene', process.env.VITRUM_SCENE);
  const caustic = process.env.VITRUM_CAUSTIC_STRATEGY?.trim();
  if (caustic === 'manifold-nee' || caustic === 'photon-map' || caustic === 'none') {
    q.set('vitrumCaustic', caustic);
  }
  for (const [key, value] of q.entries()) {
    u.searchParams.set(key, value);
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

const browser = await launchChromiumForCapture(chromium);
try {
  const page = await browser.newPage({ viewport: { width, height } });
  const url = buildUrl();
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: timeoutMs });
  await page.waitForFunction(
    (targetSpp) => {
      const p = globalThis.__vitrum?.ptWebgl;
      if (p == null) return false;
      const converged = p.converged === true || p.isConverged === true;
      const spp = typeof p.spp === 'number' ? p.spp : 0;
      return converged || spp >= targetSpp;
    },
    samplesTarget,
    { timeout: timeoutMs, polling: 200 },
  );

  const canvas = page.locator('#c-pt');
  await canvas.waitFor({ timeout: 10_000 });
  await mkdir(dirname(outputPng), { recursive: true });
  await canvas.screenshot({ path: outputPng });

  const bytes = await readFile(outputPng);
  const hash = createHash('sha256').update(bytes).digest('hex');
  const telemetry = await page.evaluate(() => globalThis.__vitrum?.ptWebgl ?? null);
  console.log(JSON.stringify({
    scenarioId,
    url,
    pngPath: outputPng,
    hash,
    samplesTarget,
    telemetry,
  }));
} finally {
  await browser.close();
}
