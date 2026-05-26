/**
 * run-quality-mode-bench.mjs
 *
 * Per-qualityMode frame-time + SPP/sec benchmark for the pt-webgl backend.
 *
 * For each (scenario, qualityMode) pair this script:
 *   1. Launches a headless Chromium via Playwright (same harness as
 *      `capture-adapter-playwright.mjs`).
 *   2. Navigates to the cornell-box dev server with
 *      `?vitrumScenario=<id>&vitrumQuality=<mode>&vitrumSpp=<target>&vitrumAutoStart=1`
 *      (the example accepts these query params; see examples/cornell-box/src/main.ts).
 *   3. Polls `window.__vitrum.ptWebgl.spp` + `window.__vitrum.ptWebgl.lastFrameMs`
 *      for `VITRUM_BENCH_DURATION_MS` (default 30000 ms) and records every
 *      unique frame.
 *   4. Emits a JSON report:
 *        tools/benchmark-runner/results/quality-modes-<timestamp>.json
 *
 * NOTE — running in a worktree:
 *   Playwright tends to be installed once in the main checkout. If the worktree
 *   does not have a usable Playwright (browser binary mismatch / missing deps),
 *   run this script in the main checkout instead:
 *       npm run benchmark:qualitymodes
 *   The dev server URL is configurable via VITRUM_CAPTURE_URL.
 *
 * Required setup before invoking:
 *   1. Start the cornell-box dev server in another terminal:
 *        npm run dev --workspace @vitrum-examples/cornell-box
 *      (default: http://127.0.0.1:5174/)
 *   2. Run this script:
 *        npm run benchmark:qualitymodes --workspace @vitrum/benchmark-runner
 *
 * Env knobs:
 *   - VITRUM_CAPTURE_URL          base URL for the dev server (default 5174)
 *   - VITRUM_BENCH_DURATION_MS    poll window per (scenario, mode) (default 30000)
 *   - VITRUM_BENCH_POLL_MS        polling interval (default 100)
 *   - VITRUM_BENCH_WARMUP_TIMEOUT_MS max wait for first live telemetry before timed sampling
 *   - VITRUM_BENCH_SAMPLES_TARGET vitrumSpp for the URL (default 128)
 *   - VITRUM_BENCH_WIDTH          render width (default 1280)
 *   - VITRUM_BENCH_HEIGHT         render height (default 720)
 *   - VITRUM_BENCH_QUALITY_MODES  comma-separated subset (default all 4)
 *   - VITRUM_BENCH_SCENARIOS      comma-separated subset (default cornell-box scenarios)
 *   - VITRUM_BENCH_HEADLESS       '0' to run headed Chromium (default '1')
 */

import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));

function parseNumberEnv(name, fallback, opts = {}) {
  const raw = process.env[name];
  if (raw == null || raw.trim() === '') return fallback;
  const value = Number(raw);
  if (!Number.isFinite(value)) {
    throw new Error(`${name} must be a finite number (got "${raw}").`);
  }
  const min = opts.min ?? -Infinity;
  const max = opts.max ?? Infinity;
  if (value < min || value > max) {
    throw new Error(`${name} must be within [${min}, ${max}] (got ${value}).`);
  }
  return opts.integer === true ? Math.trunc(value) : value;
}

const captureUrlBase = process.env.VITRUM_CAPTURE_URL ?? 'http://127.0.0.1:5174/';
const benchDurationMs = parseNumberEnv('VITRUM_BENCH_DURATION_MS', 30_000, { min: 1_000, integer: true });
const pollIntervalMs = parseNumberEnv('VITRUM_BENCH_POLL_MS', 100, { min: 10, integer: true });
const samplesTarget = parseNumberEnv('VITRUM_BENCH_SAMPLES_TARGET', 128, { min: 1, integer: true });
const renderWidth = parseNumberEnv('VITRUM_BENCH_WIDTH', 1280, { min: 1, integer: true });
const renderHeight = parseNumberEnv('VITRUM_BENCH_HEIGHT', 720, { min: 1, integer: true });
const headless = process.env.VITRUM_BENCH_HEADLESS !== '0';
const strict = process.env.VITRUM_BENCH_STRICT === '1';
const failFast = process.env.VITRUM_BENCH_FAIL_FAST === '1';
// Page settle / nav timeouts are configurable because capture mode can take
// longer than interactive/safe/final to compile on constrained machines.
const navTimeoutMs = parseNumberEnv('VITRUM_BENCH_NAV_TIMEOUT_MS', Math.max(60_000, benchDurationMs * 2), {
  min: 10_000,
  integer: true,
});
const warmupTimeoutMs = parseNumberEnv('VITRUM_BENCH_WARMUP_TIMEOUT_MS', Math.min(navTimeoutMs, 30_000), {
  min: 1_000,
  integer: true,
});

const DEFAULT_QUALITY_MODES = ['interactive', 'safe', 'final', 'capture'];
const DEFAULT_SCENARIOS = [
  // Default to the cornell-box smoke set. The example accepts any
  // `vitrumScenario` value and falls back to plain cornell-box if unknown
  // (see resolveScenarioId in examples/cornell-box/src/main.ts).
  'cornell-box',
];

const qualityModes = (process.env.VITRUM_BENCH_QUALITY_MODES ?? DEFAULT_QUALITY_MODES.join(','))
  .split(',')
  .map((s) => s.trim())
  .filter((s) => s.length > 0);
const scenarios = (process.env.VITRUM_BENCH_SCENARIOS ?? DEFAULT_SCENARIOS.join(','))
  .split(',')
  .map((s) => s.trim())
  .filter((s) => s.length > 0);

const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
const outputPath = resolve(here, `results/quality-modes-${timestamp}.json`);

let chromium;
try {
  ({ chromium } = await import('playwright'));
} catch (error) {
  console.error(
    'Playwright is not installed. Install with `npm install` in the workspace,\n' +
      'or run this script in your main checkout (not the worktree).',
  );
  console.error(String(error));
  process.exit(3);
}

function buildUrl(scenarioId, qualityMode) {
  const u = new URL(captureUrlBase);
  u.searchParams.set('vitrumScenario', scenarioId);
  u.searchParams.set('vitrumQuality', qualityMode);
  u.searchParams.set('vitrumSpp', String(samplesTarget));
  u.searchParams.set('vitrumWidth', String(renderWidth));
  u.searchParams.set('vitrumHeight', String(renderHeight));
  u.searchParams.set('vitrumAutoStart', '1');
  return u.toString();
}

/**
 * Compute frame-time percentile. Sorts a copy of the input array; safe to
 * call with the live samples array.
 */
function percentile(sortedAsc, p) {
  if (sortedAsc.length === 0) return null;
  const idx = Math.min(sortedAsc.length - 1, Math.max(0, Math.floor((p / 100) * sortedAsc.length)));
  return sortedAsc[idx];
}

function mean(values) {
  if (values.length === 0) return null;
  let sum = 0;
  for (const v of values) sum += v;
  return sum / values.length;
}

async function runOne(browser, scenario, qualityMode) {
  const startedAt = new Date().toISOString();
  const context = await browser.newContext({
    viewport: { width: renderWidth, height: renderHeight },
  });
  const page = await context.newPage();

  const url = buildUrl(scenario, qualityMode);
  const frameMsSamples = [];
  /** Track unique frame indices seen so we don't double-count when the page
   *  publishes the same frame across multiple poll ticks. */
  const seenFrameIndices = new Set();
  /** First/last spp observation define the SPP/sec rate. */
  let firstSpp = null;
  let firstSppTs = null;
  let lastSpp = 0;
  let lastSppTs = 0;
  let totalFrames = 0;
  let didConverge = false;
  let pollErrorCount = 0;
  let warmupMs = 0;
  let warmupReady = false;

  try {
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: navTimeoutMs });

    // Wait until the live telemetry global is published. The cornell-box
    // example seeds it before constructing the engine; tolerate up to 15 s
    // for slow first compiles.
    await page
      .waitForFunction(
        () => globalThis.__vitrum != null && globalThis.__vitrum.ptWebgl != null,
        null,
        { timeout: 15_000, polling: 100 },
      )
      .catch(() => {
        // Telemetry never appeared — record an empty sample for visibility.
      });

    const warmupStart = Date.now();
    while (Date.now() - warmupStart < warmupTimeoutMs) {
      const reading = await page
        .evaluate(() => {
          const p = globalThis.__vitrum?.ptWebgl;
          if (p == null) return null;
          return {
            spp: p.spp,
            lastFrameMs: p.lastFrameMs,
            frame: p.frame,
          };
        })
        .catch(() => {
          pollErrorCount++;
          return null;
        });
      if (
        reading != null &&
        ((typeof reading.frame === 'number' && reading.frame > 0) ||
          (typeof reading.spp === 'number' && reading.spp > 0) ||
          (typeof reading.lastFrameMs === 'number' && reading.lastFrameMs > 0))
      ) {
        warmupReady = true;
        break;
      }
      await new Promise((r) => setTimeout(r, pollIntervalMs));
    }
    warmupMs = Date.now() - warmupStart;

    const deadline = Date.now() + benchDurationMs;
    while (Date.now() < deadline) {
      const reading = await page
        .evaluate(() => {
          const p = globalThis.__vitrum?.ptWebgl;
          if (p == null) return null;
          return {
            spp: p.spp,
            lastFrameMs: p.lastFrameMs,
            frame: p.frame,
            isConverged: p.isConverged,
            sppPerSecond: p.sppPerSecond,
          };
        })
        .catch(() => {
          pollErrorCount++;
          return null;
        });

      if (reading != null) {
        if (firstSpp == null) {
          firstSpp = reading.spp;
          firstSppTs = Date.now();
        }
        lastSpp = reading.spp;
        lastSppTs = Date.now();
        if (reading.isConverged) didConverge = true;
        // Only count a frame once even if we poll multiple times before the
        // next render tick — the cornell-box loop publishes monotonic `frame`.
        if (typeof reading.frame === 'number' && !seenFrameIndices.has(reading.frame)) {
          seenFrameIndices.add(reading.frame);
          totalFrames = Math.max(totalFrames, reading.frame);
          // `lastFrameMs` (mapped from `batchMs` in cornell-box's loop) is
          // the engine-measured submission cost. Skip zeros from the seed
          // tick before the first render.
          if (Number.isFinite(reading.lastFrameMs) && reading.lastFrameMs > 0) {
            frameMsSamples.push(reading.lastFrameMs);
          }
        }
      }

      await new Promise((r) => setTimeout(r, pollIntervalMs));
    }
  } finally {
    await context.close();
  }

  const sorted = [...frameMsSamples].sort((a, b) => a - b);
  const elapsedSec =
    firstSppTs != null && lastSppTs > firstSppTs ? (lastSppTs - firstSppTs) / 1000 : null;
  const totalSpp = firstSpp != null ? Math.max(0, lastSpp - firstSpp) : lastSpp;
  const sppPerSec =
    elapsedSec != null && elapsedSec > 0 && totalSpp > 0 ? totalSpp / elapsedSec : null;

  return {
    scenario,
    qualityMode,
    startedAt,
    finishedAt: new Date().toISOString(),
    durationMs: benchDurationMs,
    warmupMs,
    warmupReady,
    framesRendered: totalFrames,
    totalSpp,
    sppPerSec,
    meanFrameMs: mean(frameMsSamples),
    p50FrameMs: percentile(sorted, 50),
    p99FrameMs: percentile(sorted, 99),
    minFrameMs: sorted[0] ?? null,
    maxFrameMs: sorted[sorted.length - 1] ?? null,
    converged: didConverge,
    pollErrorCount,
    config: {
      url,
      renderWidth,
      renderHeight,
      samplesTarget,
      pollIntervalMs,
    },
  };
}

async function main() {
  const jsHeapMb = parseNumberEnv('VITRUM_JS_HEAP_MB', 4096, { min: 256, integer: true });
  const browser = await chromium.launch({
    headless,
    args: ['--disable-dev-shm-usage', `--js-flags=--max-old-space-size=${jsHeapMb}`],
  });

  const entries = [];
  const failures = [];
  try {
    // Sequential — GPU contention makes parallel runs noisy.
    for (const scenario of scenarios) {
      for (const qualityMode of qualityModes) {
        console.log(`[bench] running ${scenario} × ${qualityMode}...`);
        try {
          // eslint-disable-next-line no-await-in-loop
          const result = await runOne(browser, scenario, qualityMode);
          entries.push({ ...result, status: 'ok' });
          console.log(
            `[bench]   frames=${result.framesRendered} totalSpp=${result.totalSpp}` +
              ` sppPerSec=${result.sppPerSec == null ? 'n/a' : result.sppPerSec.toFixed(2)}` +
              ` meanFrameMs=${result.meanFrameMs == null ? 'n/a' : result.meanFrameMs.toFixed(2)}`,
          );
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          failures.push({ scenario, qualityMode, message });
          entries.push({
            scenario,
            qualityMode,
            startedAt: new Date().toISOString(),
            finishedAt: new Date().toISOString(),
            durationMs: benchDurationMs,
            status: 'error',
            error: message,
            config: {
              url: buildUrl(scenario, qualityMode),
              renderWidth,
              renderHeight,
              samplesTarget,
              pollIntervalMs,
            },
          });
          console.error(`[bench]   ERROR ${scenario} × ${qualityMode}: ${message}`);
          if (failFast) {
            throw error;
          }
        }
      }
    }
  } finally {
    await browser.close();
  }

  const report = {
    generatedAt: new Date().toISOString(),
    schemaVersion: 'quality-modes-bench-2026-05-17',
    environment: {
      platform: process.platform,
      node: process.version,
      headless,
      strict,
      failFast,
      captureUrlBase,
      benchDurationMs,
      pollIntervalMs,
      warmupTimeoutMs,
    },
    qualityModes,
    scenarios,
    results: entries,
    summary: {
      total: entries.length,
      failures: failures.length,
      failedPairs: failures,
    },
  };

  await mkdir(dirname(outputPath), { recursive: true });
  await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
  console.log(`Wrote ${outputPath}`);
  if (strict && failures.length > 0) {
    console.error(`[bench] strict mode: ${failures.length} failed scenario/quality pair(s).`);
    process.exit(1);
  }
}

await main();
