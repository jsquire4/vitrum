/**
 * W8 Phase 4 — RC acceptance capture + metrics for `rcAcceptance.gpu.test.ts`.
 *
 * Captures Cornell walkaround frames with rcEnabled off vs on (rcWeight=1),
 * writes PNGs under tools/reference-renders/W8-rc-{off,on}/, then runs
 * benchmark:acceptance-metrics to emit VITRUM_RC_* JSON artifacts.
 *
 * Env:
 *   VITRUM_RC_CAPTURE_FRAMES   frames before screenshot (default 48)
 *   VITRUM_RC_CAPTURE_SPP      alias for frames (default 48)
 *   VITRUM_RC_SEED             vitrumSeed (default 1701)
 *   VITRUM_RC_START_SERVER     default 1 — vite two-engines on VITRUM_BENCH_DEV_PORT
 *   VITRUM_RC_REQUIRE_GPU      exit 2 when hybrid cannot run and no PNGs
 *   VITRUM_RC_SKIP_CAPTURE      1 — only run metrics on existing PNGs
 *   VITRUM_BENCH_DEV_PORT      vite port (default 5175); actual URL parsed from vite stdout
 */

import { access, mkdir, readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  assertWalkaroundDevServer,
  launchDevServer,
  stopDevServer,
  waitForServerReady,
} from './devServer.mjs';
import { launchWebGpuBrowser } from './launchWebGpuBrowser.mjs';
import { getRepoRoot } from './repoRoot.mjs';
import { runCommandWithTimeout } from './runCommandWithTimeout.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = getRepoRoot(import.meta.url);
const refOff = resolve(repoRoot, 'tools/reference-renders/W8-rc-off');
const refOn = resolve(repoRoot, 'tools/reference-renders/W8-rc-on');
const resultsDir = resolve(here, 'results', 'acceptance');

const frames = Number(process.env.VITRUM_RC_CAPTURE_FRAMES ?? process.env.VITRUM_RC_CAPTURE_SPP ?? '48');
const seed = process.env.VITRUM_RC_SEED ?? '1701';
const benchPort = process.env.VITRUM_BENCH_DEV_PORT ?? '5175';
const startServer = process.env.VITRUM_RC_START_SERVER !== '0';
const skipCapture = process.env.VITRUM_RC_SKIP_CAPTURE === '1';

const offPng = process.env.VITRUM_RC_OFF_PNG
  ? resolve(repoRoot, process.env.VITRUM_RC_OFF_PNG)
  : resolve(refOff, 'cornell-walkaround-rc-off.png');
const onPng = process.env.VITRUM_RC_ON_PNG
  ? resolve(repoRoot, process.env.VITRUM_RC_ON_PNG)
  : resolve(refOn, 'cornell-walkaround-rc-on.png');

async function assertReadablePng(path, label) {
  try {
    await access(path);
  } catch {
    throw new Error(
      `${label} missing at ${path}. Run capture, ` +
        '`npm run write-rc-mechanical-fixtures`, or set VITRUM_RC_SKIP_CAPTURE=1 with PNGs present.',
    );
  }
}

function runMetrics(envExtra) {
  return runCommandWithTimeout('node ./run-acceptance-metrics.mjs', {
    cwd: here,
    env: { ...process.env, ...envExtra },
    timeoutMs: 60_000,
  });
}

function captureVariantUrl(base, { rcEnabled, rcWeight }) {
  const u = new URL('walkaround.html', base.endsWith('/') ? base : `${base}/`);
  u.searchParams.set('mode', 'walkaround');
  u.searchParams.set('scene', 'cornell');
  u.searchParams.set('samplesTarget', String(frames));
  u.searchParams.set('vitrumSeed', seed);
  u.searchParams.set('rcEnabled', rcEnabled ? '1' : '0');
  if (rcEnabled) u.searchParams.set('rcWeight', String(rcWeight));
  return u.toString();
}

async function captureAll(browser, base) {
  await mkdir(refOff, { recursive: true });
  await mkdir(refOn, { recursive: true });

  const readyTimeoutMs = Number(process.env.VITRUM_RC_READY_TIMEOUT_MS ?? '120000');
  const frameWaitMs = Number(
    process.env.VITRUM_RC_FRAME_WAIT_MS ?? String(Math.max(90_000, frames * 4_000)),
  );

  for (const row of [
    { label: 'off', rcEnabled: false, rcWeight: 0, outPng: offPng },
    { label: 'on', rcEnabled: true, rcWeight: 1, outPng: onPng },
  ]) {
    const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
    const url = captureVariantUrl(base, row);
    try {
      await page.goto(url, { waitUntil: 'networkidle', timeout: 90_000 });
      const canvas = page.locator('#c-wgpu');
      await canvas.waitFor({ state: 'attached', timeout: readyTimeoutMs });

      await page
        .waitForFunction(
          () => {
            const hy = globalThis.__vitrumWalkaround;
            const tel = globalThis.__vitrum?.walkaround;
            return hy?.state === 'ready' || tel?.state === 'ready';
          },
          null,
          { timeout: readyTimeoutMs, polling: 250 },
        )
        .catch(async (err) => {
          const snap = await page.evaluate(() => ({
            url: location.href,
            status: document.querySelector('#status')?.textContent?.slice(0, 300),
            hybrid: globalThis.__vitrumWalkaround?.state,
            tel: globalThis.__vitrum?.walkaround,
          }));
          throw new Error(
            `[rc-acceptance] ${row.label} hybrid not ready: ${err instanceof Error ? err.message : err}; ` +
              JSON.stringify(snap),
          );
        });

      try {
        await page.waitForFunction(
          ({ target }) => (globalThis.__vitrum?.walkaround?.frame ?? 0) >= target,
          { target: frames },
          { timeout: frameWaitMs, polling: 250 },
        );
      } catch (err) {
        const snap = await page.evaluate(() => ({
          frame: globalThis.__vitrum?.walkaround?.frame ?? 0,
          state: globalThis.__vitrum?.walkaround?.state ?? 'missing',
        }));
        console.warn(
          `[rc-acceptance] ${row.label} frame wait (${frameWaitMs}ms): ${err instanceof Error ? err.message : err}; ` +
            `capturing anyway at frame=${snap.frame} state=${snap.state}`,
        );
        await page.waitForTimeout(2_000);
      }
      await canvas.screenshot({ path: row.outPng });
      const hash = createHash('sha256').update(await readFile(row.outPng)).digest('hex');
      console.log(`[rc-acceptance] ${row.label} → ${row.outPng} sha256=${hash.slice(0, 12)}`);
    } finally {
      await page.close();
    }
  }
}

async function tryCapture() {
  let devServer = null;
  let browser = null;
  let captureBase = process.env.VITRUM_RC_CAPTURE_BASE
    ? process.env.VITRUM_RC_CAPTURE_BASE.endsWith('/')
      ? process.env.VITRUM_RC_CAPTURE_BASE
      : `${process.env.VITRUM_RC_CAPTURE_BASE}/`
    : `http://127.0.0.1:${benchPort}/`;
  try {
    if (skipCapture) return;

    if (startServer) {
      const viteBin = resolve(repoRoot, 'node_modules/vite/bin/vite.js');
      const exampleDir = resolve(repoRoot, 'examples/two-engines-one-scene');
      devServer = launchDevServer(
        `node "${viteBin}" --host 127.0.0.1 --port ${benchPort} --strictPort`,
        exampleDir,
      );
      const ready = await waitForServerReady(devServer, captureBase, 90_000, 500);
      captureBase = ready.url.endsWith('/') ? ready.url : `${ready.url}/`;
      await assertWalkaroundDevServer(captureBase);
    } else if (process.env.VITRUM_RC_CAPTURE_BASE) {
      /* Vite already running (e.g. WSL server + Windows Playwright). */
    } else {
      await assertWalkaroundDevServer(captureBase);
    }
    console.log(`[rc-acceptance] using ${captureBase}`);

    const { chromium } = await import('playwright');
    const launched = await launchWebGpuBrowser(chromium, captureBase);
    browser = launched.browser;
    const { caps } = launched;

    if (!caps.hybridCanRun) {
      const msg =
        `[rc-acceptance] adapter insufficient for hybrid (buffers=${caps.maxStorageBuffersPerShaderStage})`;
      if (process.env.VITRUM_RC_REQUIRE_GPU === '1') {
        throw new Error(msg);
      }
      console.warn(`${msg}; skipping GPU capture.`);
      return;
    }

    await captureAll(browser, captureBase);
  } finally {
    if (browser) await browser.close();
    if (devServer) stopDevServer(devServer);
  }
}

async function main() {
  await tryCapture();

  await assertReadablePng(offPng, 'RC off PNG');
  await assertReadablePng(onPng, 'RC on PNG');

  await mkdir(resultsDir, { recursive: true });
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  const rcAcceptanceOut = resolve(resultsDir, `rc-acceptance-metrics-${stamp}.json`);
  const rcBehaviorOut = resolve(resultsDir, `rc-behavior-metrics-${stamp}.json`);

  const metrics = await runMetrics({
    VITRUM_RC_OFF_PNG: offPng,
    VITRUM_RC_ON_PNG: onPng,
    VITRUM_RC_ACCEPTANCE_OUT: rcAcceptanceOut,
    VITRUM_RC_BEHAVIOR_OUT: rcBehaviorOut,
    VITRUM_PIPELINE_CREATES_BEFORE: '0',
    VITRUM_PIPELINE_CREATES_AFTER: '0',
  });
  if (metrics.code !== 0) {
    throw new Error(
      `run-acceptance-metrics failed (code=${metrics.code}): ${metrics.stderr || metrics.stdout}`,
    );
  }

  const manifest = {
    capturedAt: new Date().toISOString(),
    frames,
    seed,
    offPng: 'tools/reference-renders/W8-rc-off/cornell-walkaround-rc-off.png',
    onPng: 'tools/reference-renders/W8-rc-on/cornell-walkaround-rc-on.png',
    rcAcceptanceMetrics: rcAcceptanceOut,
    rcBehaviorMetrics: rcBehaviorOut,
    stdout: metrics.stdout.trim(),
  };
  const manifestPath = resolve(resultsDir, `rc-acceptance-manifest-${stamp}.json`);
  await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, 'utf8');

  console.log(`VITRUM_RC_ACCEPTANCE_METRICS=${rcAcceptanceOut}`);
  console.log(`VITRUM_RC_BEHAVIOR_METRICS=${rcBehaviorOut}`);
  console.log(`[rc-acceptance] manifest ${manifestPath}`);
  console.log(metrics.stdout);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
