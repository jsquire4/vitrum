/**
 * run-pr-hybrid-bench.mjs — PR-6 hybrid incremental-update benchmarks.
 *
 * Drives `examples/two-engines-one-scene/walkaround.html` via `window.__vitrumPrBench`.
 *
 * Prereq: dev server running:
 *   npm run dev --workspace @vitrum-examples/two-engines-one-scene
 *
 * Usage:
 *   VITRUM_PR_SCENARIO=PR-hybrid-material-churn npm run benchmark:pr-hybrid
 *   npm run benchmark:pr-hybrid --workspace @vitrum/benchmark-runner  # all scenarios
 */

import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { PR_HYBRID_BENCHMARK_SCENARIOS } from './scenario-presets.mjs';
import {
  launchDevServer,
  stopDevServer,
  waitForServerReady,
} from './devServer.mjs';
import { launchWebGpuBrowser } from './launchWebGpuBrowser.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..', '..');
const resultsDir = resolve(here, 'results', 'pr-hybrid');
const prHybridRefRoot = resolve(repoRoot, 'tools/reference-renders/PR-hybrid');
const prHybridPerfDir = resolve(prHybridRefRoot, 'perf');
const stamp = new Date().toISOString().replace(/[:.]/g, '-');
const outPath = resolve(resultsDir, `pr-hybrid-${stamp}.json`);
const perfLatestPath = resolve(prHybridPerfDir, 'latest.json');

const benchDevPort = process.env.VITRUM_BENCH_DEV_PORT ?? '5175';
const benchDevHost = process.env.VITRUM_BENCH_DEV_HOST ?? '127.0.0.1';
const defaultCaptureBase = `http://${benchDevHost}:${benchDevPort}/walkaround.html`;
let captureUrlBase = process.env.VITRUM_CAPTURE_URL ?? defaultCaptureBase;
const startServer = process.env.VITRUM_PR_START_SERVER === '1';
const serverCommand =
  process.env.VITRUM_PR_DEV_CMD ??
  `npm run dev --workspace @vitrum-examples/two-engines-one-scene -- --host ${benchDevHost} --port ${benchDevPort}`;
const serverReadyTimeoutMs = Number(process.env.VITRUM_PR_SERVER_READY_TIMEOUT_MS ?? 90_000);
const serverPollMs = Number(process.env.VITRUM_PR_SERVER_POLL_MS ?? 500);
const headless = process.env.VITRUM_BENCH_HEADLESS !== '0';
const navTimeoutMs = Number(process.env.VITRUM_PR_NAV_TIMEOUT_MS ?? 90_000);
const benchTimeoutMs = Number(process.env.VITRUM_PR_BENCH_TIMEOUT_MS ?? 120_000);
/** @see {@link HYBRID_WEBGPU_REQUIRED_LIMITS} in @vitrum/walkaround-hybrid */
const HYBRID_MIN_STORAGE_BUFFERS = 16;
const HYBRID_MIN_STORAGE_TEXTURES = 8;
const requireGpu = process.env.VITRUM_PR_REQUIRE_GPU === '1';
const skipIfInsufficient =
  process.env.VITRUM_PR_SKIP_IF_ADAPTER_INSUFFICIENT !== '0';

const SCENARIO_QUERY = {
  'PR-hybrid-material-churn': {
    prBench: 'material-churn',
    prBenchAuto: '1',
    mode: 'walkaround',
    prBenchIters: '100',
  },
  'PR-hybrid-emitter-churn': {
    prBench: 'emitter-churn',
    prBenchAuto: '1',
    mode: 'walkaround',
    prBenchIters: '100',
  },
  'PR-hybrid-tlas-10-inst': {
    prBench: 'frame-sample',
    prBenchAuto: '1',
    mode: 'walkaround',
    scene: 'tlas10inst',
    bvhMode: 'tlas',
    prBenchFrames: '48',
    prBenchScenario: 'PR-hybrid-tlas-10-inst',
  },
  'PR-hybrid-200k-static': {
    prBench: 'frame-sample',
    prBenchAuto: '1',
    mode: 'walkaround',
    scene: 'bench200k',
    targetTriangles: '200000',
    prBenchFrames: '120',
    prBenchScenario: 'PR-hybrid-200k-static',
  },
};

function buildUrl(scenarioId) {
  const u = new URL(captureUrlBase);
  const q = SCENARIO_QUERY[scenarioId];
  if (q == null) throw new Error(`Unknown PR scenario: ${scenarioId}`);
  for (const [k, v] of Object.entries(q)) {
    u.searchParams.set(k, v);
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

async function runScenario(browser, scenarioId) {
  const context = await browser.newContext({ viewport: { width: 1280, height: 720 } });
  const page = await context.newPage();
  const url = buildUrl(scenarioId);
  const startedAt = new Date().toISOString();
  let benchResult = null;
  let error = null;

  try {
    page.on('console', (msg) => {
      const text = msg.text();
      if (text.startsWith('VITRUM_PR_BENCH_RESULT=')) {
        try {
          benchResult = JSON.parse(text.slice('VITRUM_PR_BENCH_RESULT='.length));
        } catch {
          /* ignore parse errors */
        }
      }
    });

    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: navTimeoutMs });
    const bootError = await page.evaluate(() => {
      const status = document.querySelector('#status')?.textContent ?? '';
      if (/requestDevice|OperationError|maxStorageTextures|maxStorageBuffers/i.test(status)) {
        return status.slice(0, 500);
      }
      return null;
    });
    if (bootError != null) {
      throw new Error(`walkaround page failed WebGPU init: ${bootError}`);
    }
    const scenarioBenchTimeoutMs =
      scenarioId === 'PR-hybrid-200k-static'
        ? Number(process.env.VITRUM_PR_200K_TIMEOUT_MS ?? String(Math.max(benchTimeoutMs, 300_000)))
        : scenarioId === 'PR-hybrid-tlas-10-inst'
          ? Number(process.env.VITRUM_PR_TLAS_TIMEOUT_MS ?? String(Math.max(benchTimeoutMs, 240_000)))
          : benchTimeoutMs;
    await page.waitForFunction(
      () => {
        const eng = globalThis.__vitrumWalkaround;
        const tel = globalThis.__vitrum?.walkaround;
        return (
          eng?.state === 'ready' ||
          tel?.state === 'ready' ||
          globalThis.__vitrumPrBenchLast != null
        );
      },
      null,
      { timeout: scenarioBenchTimeoutMs, polling: 250 },
    ).catch(() => {
      /* walkaround may still be initializing BVH on very large scenes */
    });
    await page.waitForFunction(
      () => globalThis.__vitrumPrBenchLast != null || globalThis.__vitrumPrBench != null,
      null,
      { timeout: scenarioBenchTimeoutMs, polling: 250 },
    );

    if (benchResult == null) {
      benchResult = await page.evaluate(() => globalThis.__vitrumPrBenchLast ?? null);
    }
    if (benchResult == null) {
      const mode = new URL(url).searchParams.get('prBench');
      if (mode != null) {
        benchResult = await page.evaluate(async (m) => {
          const api = globalThis.__vitrumPrBench;
          if (api == null) return null;
          return api.run(m);
        }, mode);
      }
    }
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  } finally {
    await context.close();
  }

  const payload = {
    scenarioId,
    startedAt,
    finishedAt: new Date().toISOString(),
    url,
    pass: error == null && benchResult?.ok === true,
    bench: benchResult,
    ...(error != null ? { error } : {}),
  };
  payload.hash = createHash('sha256').update(JSON.stringify(payload.bench ?? {})).digest('hex');
  return payload;
}

function hybridAdapterCanRun(caps) {
  if (caps == null) return false;
  return caps.hybridCanRun === true;
}

function skippedRow(scenarioId, url, skipReason, limits) {
  const startedAt = new Date().toISOString();
  const payload = {
    scenarioId,
    startedAt,
    finishedAt: startedAt,
    url,
    pass: !requireGpu,
    skipped: true,
    skipReason,
    adapterLimits: limits,
    bench: null,
  };
  payload.hash = createHash('sha256').update(JSON.stringify(payload)).digest('hex');
  return payload;
}

async function main() {
  const only = process.env.VITRUM_PR_SCENARIO?.trim();
  const scenarios = only
    ? PR_HYBRID_BENCHMARK_SCENARIOS.filter((s) => s.scenarioId === only).map((s) => s.scenarioId)
    : PR_HYBRID_BENCHMARK_SCENARIOS.map((s) => s.scenarioId);

  if (scenarios.length === 0) {
    console.error(`No scenario matched VITRUM_PR_SCENARIO=${only ?? '(empty)'}`);
    process.exit(2);
  }

  await mkdir(resultsDir, { recursive: true });

  let devServer = null;
  if (startServer) {
    console.log('[pr-hybrid] starting dev server…');
    devServer = launchDevServer(serverCommand, repoRoot);
    const ready = await waitForServerReady(
      devServer,
      captureUrlBase.replace(/\/[^/]*$/, '/'),
      serverReadyTimeoutMs,
      serverPollMs,
    );
    const base = ready.url.endsWith('/') ? ready.url : `${ready.url}/`;
    captureUrlBase = `${base}walkaround.html`;
    console.log(`[pr-hybrid] using ${captureUrlBase}`);
  }

  let browser;
  let launchProfile = 'unknown';
  let adapterCaps = null;
  try {
    const probeOrigin = new URL(captureUrlBase).origin + '/';
    const launched = await launchWebGpuBrowser(chromium, probeOrigin);
    browser = launched.browser;
    launchProfile = launched.profile;
    adapterCaps = launched.caps;
    console.log(
      `[pr-hybrid] WebGPU launchProfile=${launchProfile} ` +
        `vendor=${adapterCaps.vendor ?? ''} buffers=${adapterCaps.maxStorageBuffersPerShaderStage} ` +
        `textures=${adapterCaps.maxStorageTexturesPerShaderStage} ` +
        `ptFull=${adapterCaps.ptWebgpuFullTier} hybrid=${adapterCaps.hybridCanRun}`,
    );
  } catch (launchErr) {
    console.error(`[pr-hybrid] failed to launch WebGPU browser: ${launchErr}`);
    process.exit(3);
  }

  const rows = [];
  try {
    const canRunHybrid = hybridAdapterCanRun(adapterCaps);
    if (!canRunHybrid) {
      const msg =
        `[pr-hybrid] adapter insufficient for walkaround-hybrid ` +
        `(need ≥${HYBRID_MIN_STORAGE_BUFFERS} storage buffers and ≥${HYBRID_MIN_STORAGE_TEXTURES} storage textures; ` +
        `got ${adapterCaps?.maxStorageBuffersPerShaderStage ?? 'n/a'} / ` +
        `${adapterCaps?.maxStorageTexturesPerShaderStage ?? 'n/a'}, profile=${launchProfile}).`;
      if (skipIfInsufficient && !requireGpu) {
        console.warn(`${msg} Skipping GPU scenarios (set VITRUM_PR_REQUIRE_GPU=1 on a hardware GPU host).`);
        for (const scenarioId of scenarios) {
          rows.push(
            skippedRow(
              scenarioId,
              buildUrl(scenarioId),
              'adapter-insufficient',
              adapterCaps,
            ),
          );
        }
      } else {
        console.error(msg);
        for (const scenarioId of scenarios) {
          rows.push({
            ...skippedRow(scenarioId, buildUrl(scenarioId), 'adapter-insufficient', adapterCaps),
            pass: false,
            skipped: false,
            error: msg,
          });
        }
      }
    } else {
      for (const scenarioId of scenarios) {
        console.log(`[pr-hybrid] ${scenarioId}`);
        // eslint-disable-next-line no-await-in-loop
        rows.push(await runScenario(browser, scenarioId));
      }
    }
  } finally {
    await browser.close();
    if (devServer) stopDevServer(devServer);
  }

  const failures = rows.filter((r) => !r.pass).length;
  const report = {
    schema: 'vitrum-pr-hybrid-bench-2026-05-26',
    startedAt: rows[0]?.startedAt ?? new Date().toISOString(),
    finishedAt: new Date().toISOString(),
    launchProfile,
    adapter: adapterCaps,
    summary: { total: rows.length, failures, passes: rows.length - failures },
    rows,
  };
  const reportJson = `${JSON.stringify(report, null, 2)}\n`;
  await writeFile(outPath, reportJson);
  await mkdir(prHybridPerfDir, { recursive: true });
  await writeFile(perfLatestPath, reportJson);

  const manifestPath = resolve(prHybridRefRoot, 'manifest.json');
  let prior = { manifest: [] };
  try {
    prior = JSON.parse(await readFile(manifestPath, 'utf8'));
  } catch {
    /* first run */
  }
  const perfEntries = rows
    .filter((r) => r.pass && r.bench != null)
    .map((r) => ({
      kind: 'perf',
      scenarioId: r.scenarioId,
      reportPath: outPath,
      bench: r.bench,
    }));
  const priorManifest = Array.isArray(prior.manifest) ? prior.manifest : [];
  const withoutPerf = priorManifest.filter((e) => e?.kind !== 'perf');
  await writeFile(
    manifestPath,
    `${JSON.stringify(
      {
        generatedAt: report.finishedAt,
        note:
          'Perf: npm run benchmark:pr-hybrid (≥16 storage buffers). PNGs: npm run benchmark:pr-hybrid-refs.',
        latestPerf: perfLatestPath,
        manifest: [...withoutPerf, ...perfEntries],
      },
      null,
      2,
    )}\n`,
  );

  console.log(`Wrote ${outPath}`);
  console.log(`Wrote ${perfLatestPath}`);
  console.log(`Updated ${manifestPath}`);
  console.log(`VITRUM_PR_HYBRID_REPORT=${outPath}`);
  process.exit(failures > 0 ? 1 : 0);
}

await main();
