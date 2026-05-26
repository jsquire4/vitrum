/**
 * run-lifecycle-soak.mjs
 *
 * Reliability soak for host/engine lifecycle churn in the cornell-box harness.
 * Runs repeated page navigations across scenario/quality/resolution combinations
 * and validates that frame telemetry continues to progress without NaN/Inf data.
 */

import { mkdir, writeFile } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..', '..');
const resultsDir = resolve(here, 'results');
const stamp = new Date().toISOString().replace(/[:.]/g, '-');
const outPath = resolve(resultsDir, `lifecycle-soak-${stamp}.json`);

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
let activeCaptureUrlBase = captureUrlBase;
const headless = process.env.VITRUM_BENCH_HEADLESS !== '0';
const strict = process.env.VITRUM_LIFECYCLE_SOAK_STRICT === '1';
const iterations = parseNumberEnv('VITRUM_LIFECYCLE_SOAK_ITERATIONS', 12, { min: 1, integer: true });
const iterationMs = parseNumberEnv('VITRUM_LIFECYCLE_SOAK_ITERATION_MS', 4_000, { min: 1_000, integer: true });
const pollMs = parseNumberEnv('VITRUM_LIFECYCLE_SOAK_POLL_MS', 100, { min: 20, integer: true });
const navTimeoutMs = parseNumberEnv('VITRUM_LIFECYCLE_SOAK_NAV_TIMEOUT_MS', 90_000, {
  min: 10_000,
  integer: true,
});
const readyTimeoutMs = parseNumberEnv('VITRUM_LIFECYCLE_SOAK_READY_TIMEOUT_MS', 60_000, {
  min: 1_000,
  integer: true,
});
const samplesTarget = parseNumberEnv('VITRUM_LIFECYCLE_SOAK_SPP', 256, { min: 8, integer: true });
const startServer = process.env.VITRUM_LIFECYCLE_SOAK_START_SERVER === '1';
const serverReadyTimeoutMs = parseNumberEnv('VITRUM_LIFECYCLE_SOAK_SERVER_READY_TIMEOUT_MS', 90_000, {
  min: 2_000,
  integer: true,
});
const serverPollMs = parseNumberEnv('VITRUM_LIFECYCLE_SOAK_SERVER_POLL_MS', 500, { min: 100, integer: true });
const serverCommand =
  process.env.VITRUM_LIFECYCLE_SOAK_DEV_CMD ??
  'npm run dev --workspace @vitrum-examples/cornell-box -- --host 127.0.0.1 --port 5174';

const qualityModes = (process.env.VITRUM_LIFECYCLE_SOAK_QUALITY_MODES ?? 'interactive,safe,final,capture')
  .split(',')
  .map((s) => s.trim())
  .filter((s) => s.length > 0);
const scenarios = (process.env.VITRUM_LIFECYCLE_SOAK_SCENARIOS ?? 'cornell-box')
  .split(',')
  .map((s) => s.trim())
  .filter((s) => s.length > 0);

const resolutions = [
  { width: 640, height: 360 },
  { width: 960, height: 540 },
  { width: 1280, height: 720 },
];

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

function pushTail(buf, line, max = 40) {
  if (line.trim().length === 0) return;
  buf.push(line);
  if (buf.length > max) {
    buf.splice(0, buf.length - max);
  }
}

function launchDevServer() {
  const stdoutTail = [];
  const stderrTail = [];
  const child = spawn(serverCommand, {
    cwd: repoRoot,
    shell: true,
    detached: process.platform !== 'win32',
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  child.stdout?.on('data', (chunk) => {
    for (const line of String(chunk).split('\n')) pushTail(stdoutTail, line);
  });
  child.stderr?.on('data', (chunk) => {
    for (const line of String(chunk).split('\n')) pushTail(stderrTail, line);
  });
  return { child, stdoutTail, stderrTail };
}

async function waitForServerReady(procInfo) {
  const started = Date.now();
  while (Date.now() - started < serverReadyTimeoutMs) {
    if (procInfo.child.exitCode != null) {
      throw new Error(
        `Dev server exited early with code ${procInfo.child.exitCode}. ` +
          `stderrTail=${procInfo.stderrTail.join(' | ')}`,
      );
    }
    try {
      const discovered = deriveServerUrlFromTail(procInfo.stdoutTail);
      if (discovered != null) {
        activeCaptureUrlBase = discovered;
        const res = await fetch(activeCaptureUrlBase, { method: 'GET' });
        if (res.status < 500) {
          return Date.now() - started;
        }
      }
    } catch {
      // Keep polling until timeout.
    }
    // eslint-disable-next-line no-await-in-loop
    await new Promise((r) => setTimeout(r, serverPollMs));
  }
  throw new Error(`Timed out waiting for dev server readiness at ${activeCaptureUrlBase}.`);
}

function deriveServerUrlFromTail(stdoutTail) {
  for (let i = stdoutTail.length - 1; i >= 0; i -= 1) {
    const line = stdoutTail[i];
    const m = line.match(/https?:\/\/[^\s]+/);
    if (m != null) return m[0];
  }
  return null;
}

function stopDevServer(procInfo) {
  const pid = procInfo?.child?.pid;
  if (pid == null) return;
  if (process.platform === 'win32') {
    procInfo.child.kill('SIGTERM');
    return;
  }
  try {
    process.kill(-pid, 'SIGTERM');
  } catch {
    procInfo.child.kill('SIGTERM');
  }
}

function buildUrl({ scenarioId, qualityMode, width, height }) {
  const u = new URL(activeCaptureUrlBase);
  u.searchParams.set('vitrumScenario', scenarioId);
  u.searchParams.set('vitrumQuality', qualityMode);
  u.searchParams.set('vitrumSpp', String(samplesTarget));
  u.searchParams.set('vitrumWidth', String(width));
  u.searchParams.set('vitrumHeight', String(height));
  u.searchParams.set('vitrumAutoStart', '1');
  return u.toString();
}

async function waitForTelemetry(page) {
  const started = Date.now();
  while (Date.now() - started < readyTimeoutMs) {
    const ready = await page
      .evaluate(() => globalThis.__vitrum?.ptWebgl != null)
      .catch(() => false);
    if (ready) return;
    // eslint-disable-next-line no-await-in-loop
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error(`Timed out waiting for __vitrum.ptWebgl after ${readyTimeoutMs}ms.`);
}

async function pollTelemetry(page) {
  return page.evaluate(() => {
    const p = globalThis.__vitrum?.ptWebgl;
    if (p == null) return null;
    return {
      frame: p.frame,
      spp: p.spp,
      lastFrameMs: p.lastFrameMs,
      isConverged: p.isConverged,
      renderWidth: p.renderWidth,
      renderHeight: p.renderHeight,
    };
  });
}

async function runIteration(page, i) {
  const scenarioId = scenarios[i % scenarios.length];
  const qualityMode = qualityModes[i % qualityModes.length];
  const res = resolutions[i % resolutions.length];
  const url = buildUrl({ scenarioId, qualityMode, width: res.width, height: res.height });
  const startedAt = new Date().toISOString();

  let pollErrors = 0;
  let nonFiniteSamples = 0;
  let frameStart = null;
  let frameEnd = null;
  let sppStart = null;
  let sppEnd = null;
  let maxFrameMs = 0;
  let convergedSeen = false;

  let error = null;
  try {
    const response = await page.goto(url, { waitUntil: 'domcontentloaded', timeout: navTimeoutMs });
    const status = response?.status() ?? 0;
    if (status >= 400) {
      throw new Error(`Navigation returned HTTP ${status}.`);
    }
    await waitForTelemetry(page);

    const deadline = Date.now() + iterationMs;
    while (Date.now() < deadline) {
      const reading = await pollTelemetry(page).catch(() => {
        pollErrors += 1;
        return null;
      });
      if (reading != null) {
        const finite =
          Number.isFinite(reading.frame) &&
          Number.isFinite(reading.spp) &&
          Number.isFinite(reading.lastFrameMs);
        if (!finite) {
          nonFiniteSamples += 1;
        } else {
          if (frameStart == null) frameStart = reading.frame;
          if (sppStart == null) sppStart = reading.spp;
          frameEnd = reading.frame;
          sppEnd = reading.spp;
          maxFrameMs = Math.max(maxFrameMs, reading.lastFrameMs);
        }
        if (reading.isConverged === true) {
          convergedSeen = true;
        }
      }
      // eslint-disable-next-line no-await-in-loop
      await new Promise((r) => setTimeout(r, pollMs));
    }
  } catch (e) {
    const title = await page.title().catch(() => '');
    const probe = await page
      .evaluate(() => ({
        hasVitrum: globalThis.__vitrum != null,
        hasPtWebgl: globalThis.__vitrum?.ptWebgl != null,
        captureReadyType: typeof globalThis.VITRUM_CAPTURE_READY,
        href: globalThis.location?.href ?? null,
      }))
      .catch(() => null);
    const baseMessage = e instanceof Error ? e.message : String(e);
    const probeSuffix =
      probe == null
        ? ''
        : ` (probe=${JSON.stringify(probe)})`;
    error = title ? `${baseMessage} (title="${title}")${probeSuffix}` : `${baseMessage}${probeSuffix}`;
  }

  const frameProgress = frameStart != null && frameEnd != null ? frameEnd - frameStart : -1;
  const sppProgress = sppStart != null && sppEnd != null ? sppEnd - sppStart : -1;
  const hasTelemetry = frameEnd != null && sppEnd != null;
  const progressed = frameProgress > 0 || sppProgress > 0;
  const pass =
    error == null &&
    hasTelemetry &&
    (progressed || convergedSeen || (frameEnd ?? 0) > 0) &&
    pollErrors < 5 &&
    nonFiniteSamples === 0;

  return {
    index: i,
    scenarioId,
    qualityMode,
    width: res.width,
    height: res.height,
    startedAt,
    finishedAt: new Date().toISOString(),
    iterationMs,
    frameStart,
    frameEnd,
    frameProgress,
    sppStart,
    sppEnd,
    sppProgress,
    maxFrameMs,
    convergedSeen,
    pollErrors,
    nonFiniteSamples,
    pass,
    ...(error != null ? { error } : {}),
    url,
  };
}

async function main() {
  let devServer = null;
  const serverInfo = {
    startServer,
    command: serverCommand,
    readyMs: null,
    launchError: null,
    stdoutTail: [],
    stderrTail: [],
  };

  if (startServer) {
    devServer = launchDevServer();
    serverInfo.stdoutTail = devServer.stdoutTail;
    serverInfo.stderrTail = devServer.stderrTail;
    try {
      serverInfo.readyMs = await waitForServerReady(devServer);
      const discovered = deriveServerUrlFromTail(devServer.stdoutTail);
      if (discovered != null) {
        activeCaptureUrlBase = discovered;
      }
    } catch (error) {
      serverInfo.launchError = error instanceof Error ? error.message : String(error);
    }
  }

  const browser = await chromium.launch({ headless });
  const context = await browser.newContext({
    viewport: { width: 1280, height: 720 },
  });
  const page = await context.newPage();

  const startedAt = new Date().toISOString();
  const rows = [];
  const failures = [];
  try {
    for (let i = 0; i < iterations; i += 1) {
      console.log(`[soak] iteration ${i + 1}/${iterations}`);
      // eslint-disable-next-line no-await-in-loop
      const row = await runIteration(page, i);
      rows.push(row);
      if (!row.pass) failures.push(row);
      console.log(
        `[soak]   ${row.scenarioId} × ${row.qualityMode} ` +
          `frameΔ=${row.frameProgress} sppΔ=${row.sppProgress} pollErrors=${row.pollErrors}`,
      );
    }
  } finally {
    await context.close();
    await browser.close();
    if (devServer != null) {
      stopDevServer(devServer);
    }
  }

  const report = {
    schemaVersion: 'lifecycle-soak-2026-05-26',
    generatedAt: new Date().toISOString(),
    startedAt,
    finishedAt: new Date().toISOString(),
    environment: {
      platform: process.platform,
      node: process.version,
      headless,
      strict,
      captureUrlBase,
      activeCaptureUrlBase,
      startServer,
      iterations,
      iterationMs,
      pollMs,
      navTimeoutMs,
      readyTimeoutMs,
      samplesTarget,
    },
    server: serverInfo,
    scenarios,
    qualityModes,
    results: rows,
    summary: {
      total: rows.length,
      failures: failures.length,
      failedIndices: failures.map((r) => r.index),
    },
  };

  await mkdir(resultsDir, { recursive: true });
  await writeFile(outPath, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
  console.log(`VITRUM_LIFECYCLE_SOAK_REPORT=${outPath}`);
  if (strict && failures.length > 0) {
    console.error(`[soak] strict mode: ${failures.length} failing iteration(s).`);
    process.exit(1);
  }
}

await main();
