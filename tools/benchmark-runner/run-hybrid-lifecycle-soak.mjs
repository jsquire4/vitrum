/**
 * PR-6 hybrid lifecycle soak — material/emitter patches during live frames.
 *
 *   VITRUM_HYBRID_SOAK_START_SERVER=1 npm run benchmark:hybrid-lifecycle-soak
 */

import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { WEBGPU_CHROMIUM_LAUNCH } from './playwrightWebGpu.mjs';
import {
  launchDevServer,
  stopDevServer,
  waitForServerReady,
} from './devServer.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..', '..');
const resultsDir = resolve(here, 'results');
const stamp = new Date().toISOString().replace(/[:.]/g, '-');
const outPath = resolve(resultsDir, `hybrid-lifecycle-soak-${stamp}.json`);

const iterations = Math.max(1, Number(process.env.VITRUM_HYBRID_SOAK_ITERATIONS ?? '8'));
const soakFrames = Math.max(16, Number(process.env.VITRUM_HYBRID_SOAK_FRAMES ?? '120'));
const materialEvery = Math.max(1, Number(process.env.VITRUM_HYBRID_SOAK_MATERIAL_EVERY ?? '10'));
const emitterEvery = Math.max(0, Number(process.env.VITRUM_HYBRID_SOAK_EMITTER_EVERY ?? '15'));
const startServer = process.env.VITRUM_HYBRID_SOAK_START_SERVER === '1';
const serverCommand =
  process.env.VITRUM_HYBRID_SOAK_DEV_CMD ??
  'npm run dev --workspace @vitrum-examples/two-engines-one-scene -- --host 127.0.0.1 --port 5175';
const navTimeoutMs = Number(process.env.VITRUM_HYBRID_SOAK_NAV_TIMEOUT_MS ?? 120_000);
const soakTimeoutMs = Number(process.env.VITRUM_HYBRID_SOAK_TIMEOUT_MS ?? 300_000);

let captureUrlBase = process.env.VITRUM_CAPTURE_URL ?? 'http://127.0.0.1:5175/walkaround.html';

function buildUrl(iteration) {
  const u = new URL(captureUrlBase);
  u.searchParams.set('mode', 'walkaround');
  u.searchParams.set('scene', iteration % 2 === 0 ? 'cornell' : 'complex');
  u.searchParams.set('hybridSoakAuto', '1');
  u.searchParams.set('hybridSoakFrames', String(soakFrames));
  u.searchParams.set('hybridSoakMaterialEvery', String(materialEvery));
  if (emitterEvery > 0) u.searchParams.set('hybridSoakEmitterEvery', String(emitterEvery));
  return u.toString();
}

let chromium;
try {
  ({ chromium } = await import('playwright'));
} catch (error) {
  console.error('Playwright required:', error);
  process.exit(3);
}

async function runIteration(browser, i) {
  const context = await browser.newContext({ viewport: { width: 1280, height: 720 } });
  const page = await context.newPage();
  const url = buildUrl(i);
  let soakResult = null;
  let error = null;

  page.on('console', (msg) => {
    const text = msg.text();
    if (text.startsWith('VITRUM_HYBRID_SOAK_RESULT=')) {
      try {
        soakResult = JSON.parse(text.slice('VITRUM_HYBRID_SOAK_RESULT='.length));
      } catch {
        /* ignore */
      }
    }
  });

  try {
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: navTimeoutMs });
    const bootError = await page.evaluate(() => {
      const status = document.querySelector('#status')?.textContent ?? '';
      if (/requestDevice|OperationError|maxStorageTextures|maxStorageBuffers/i.test(status)) {
        return status.slice(0, 400);
      }
      return null;
    });
    if (bootError != null) throw new Error(bootError);

    await page.waitForFunction(
      () => globalThis.__vitrum?.walkaround?.state === 'ready',
      null,
      { timeout: soakTimeoutMs, polling: 250 },
    );
    await page.waitForFunction(
      () => globalThis.__vitrumHybridSoakLast != null,
      null,
      { timeout: soakTimeoutMs, polling: 250 },
    );
    if (soakResult == null) {
      soakResult = await page.evaluate(() => globalThis.__vitrumHybridSoakLast ?? null);
    }
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  } finally {
    await context.close();
  }

  return {
    index: i,
    url,
    pass: error == null && soakResult?.ok === true,
    soak: soakResult,
    ...(error != null ? { error } : {}),
  };
}

async function main() {
  await mkdir(resultsDir, { recursive: true });
  let devServer = null;
  if (startServer) {
    devServer = launchDevServer(serverCommand, repoRoot);
    const ready = await waitForServerReady(devServer, captureUrlBase, 90_000, 500);
    const base = ready.url.endsWith('/') ? ready.url : `${ready.url}/`;
    captureUrlBase = `${base}walkaround.html`;
  }

  const browser = await chromium.launch(WEBGPU_CHROMIUM_LAUNCH);
  const rows = [];
  try {
    for (let i = 0; i < iterations; i += 1) {
      console.log(`[hybrid-soak] iteration ${i + 1}/${iterations}`);
      rows.push(await runIteration(browser, i));
    }
  } finally {
    await browser.close();
    if (devServer) stopDevServer(devServer);
  }

  const failures = rows.filter((r) => !r.pass).length;
  const report = {
    schemaVersion: 'hybrid-lifecycle-soak-2026-05-26',
    iterations,
    soakFrames,
    materialEvery,
    emitterEvery,
    summary: { total: rows.length, failures, passes: rows.length - failures },
    rows,
  };
  await writeFile(outPath, `${JSON.stringify(report, null, 2)}\n`);
  console.log(`Wrote ${outPath}`);
  process.exit(failures > 0 ? 1 : 0);
}

await main();
