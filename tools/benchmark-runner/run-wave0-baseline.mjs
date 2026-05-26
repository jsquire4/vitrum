import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { runCommandWithTimeout } from './runCommandWithTimeout.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..', '..');
const stamp = new Date().toISOString().replace(/[:.]/g, '-');
const outDir = resolve(here, 'results', 'wave0');
const outPath = resolve(outDir, `wave0-baseline-${stamp}.json`);

function parseTimeoutMs(name, fallbackMs) {
  const raw = process.env[name];
  if (raw == null || raw.trim() === '') return fallbackMs;
  const value = Number(raw);
  if (!Number.isFinite(value) || value < 1_000) {
    throw new Error(`${name} must be a finite integer >= 1000 (got "${raw}").`);
  }
  return value;
}

function nowIso() {
  return new Date().toISOString();
}

async function runStep(step) {
  const startedAt = nowIso();
  const startMs = Date.now();
  const run = await runCommandWithTimeout(step.command, {
    cwd: repoRoot,
    env: step.env ?? {},
    timeoutMs: step.timeoutMs,
  });
  const finishedAt = nowIso();
  const elapsedMs = Date.now() - startMs;
  const stdoutTail = run.stdout.split('\n').slice(-30);
  const stderrTail = run.stderr.split('\n').slice(-30);
  let status = run.code === 0 ? 'pass' : 'fail';
  let diagnostics = null;
  if (run.code === 0 && step.id === 'quality_modes_smoke') {
    const match = run.stdout.match(/Wrote\s+([^\n]*quality-modes-[^\s]+\.json)/);
    if (match != null) {
      try {
        const parsed = JSON.parse(await readFile(match[1], 'utf8'));
        const failures = Number(parsed?.summary?.failures ?? 0);
        const rows = Array.isArray(parsed?.results) ? parsed.results : [];
        const warmupNotReady = rows.filter((row) => row?.warmupReady === false).length;
        diagnostics = {
          resultPath: match[1],
          scenarioFailures: failures,
          warmupNotReady,
        };
        if (failures > 0 || warmupNotReady > 0) {
          status = 'warn';
        }
      } catch {
        diagnostics = {
          resultPath: match[1],
          scenarioFailures: null,
          warmupNotReady: null,
        };
      }
    }
  }
  if (step.required === false && status === 'fail') {
    status = 'warn';
    diagnostics = {
      ...(diagnostics ?? {}),
      optionalFailure: true,
      timedOut: run.timedOut,
      exitCode: run.code,
    };
  }
  return {
    id: step.id,
    description: step.description,
    required: step.required,
    command: step.command,
    startedAt,
    finishedAt,
    elapsedMs,
    exitCode: run.code,
    timedOut: run.timedOut,
    status,
    diagnostics,
    stdoutTail,
    stderrTail,
  };
}

async function main() {
  const includeCaptureRefs = process.env.VITRUM_WAVE0_CAPTURE_REFS === '1';
  const strict = process.env.VITRUM_WAVE0_STRICT === '1';

  const steps = [
    {
      id: 'mechanical',
      description: 'Workspace mechanical validation',
      required: true,
      command: 'npm run verify:mechanical',
      timeoutMs: parseTimeoutMs('VITRUM_WAVE0_MECHANICAL_TIMEOUT_MS', 45 * 60_000),
    },
    {
      id: 'gap_closure_smoke',
      description: 'Gap-closure benchmark smoke snapshot',
      required: true,
      command: 'npm run benchmark:gap-closure --workspace @vitrum/benchmark-runner',
      timeoutMs: parseTimeoutMs('VITRUM_WAVE0_GAP_TIMEOUT_MS', 12 * 60_000),
      env: {
        VITRUM_CAPTURE_SMOKE: '1',
      },
    },
    {
      id: 'quality_modes_smoke',
      description: 'Quality-mode benchmark smoke snapshot',
      required: false,
      command: 'npm run benchmark:qualitymodes --workspace @vitrum/benchmark-runner',
      timeoutMs: parseTimeoutMs('VITRUM_WAVE0_QUALITY_TIMEOUT_MS', 20 * 60_000),
      env: {
        VITRUM_CAPTURE_SMOKE: '1',
        VITRUM_BENCH_DURATION_MS: process.env.VITRUM_WAVE0_BENCH_DURATION_MS ?? '8000',
        VITRUM_BENCH_QUALITY_MODES: process.env.VITRUM_WAVE0_BENCH_QUALITY_MODES ?? 'interactive,safe',
        VITRUM_BENCH_SCENARIOS: process.env.VITRUM_WAVE0_BENCH_SCENARIOS ?? 'cornell-box',
      },
    },
  ];

  if (includeCaptureRefs) {
    steps.push({
      id: 'reference_capture_quick',
      description: 'Reference render quick capture set',
      required: false,
      command: 'npm run capture:refs:quick',
      timeoutMs: parseTimeoutMs('VITRUM_WAVE0_CAPTURE_TIMEOUT_MS', 30 * 60_000),
    });
  }

  const startedAt = nowIso();
  const results = [];
  for (const step of steps) {
    // Keep this loop sequential so output reflects baseline gate order.
    // eslint-disable-next-line no-await-in-loop
    const result = await runStep(step);
    results.push(result);
  }
  const finishedAt = nowIso();

  const failed = results.filter((r) => r.status === 'fail').map((r) => r.id);
  const warned = results.filter((r) => r.status === 'warn').map((r) => r.id);
  const requiredFailed = results
    .filter((r) => r.required !== false && r.status === 'fail')
    .map((r) => r.id);
  const report = {
    schemaVersion: 'vitrum-wave0-baseline-2026-05-26',
    startedAt,
    finishedAt,
    includeCaptureRefs,
    node: process.version,
    platform: process.platform,
    cwd: repoRoot,
    strict,
    results,
    summary: {
      total: results.length,
      passed: results.length - failed.length,
      failed: failed.length,
      failedSteps: failed,
      warned: warned.length,
      warnedSteps: warned,
      requiredFailedSteps: requiredFailed,
    },
  };

  await mkdir(outDir, { recursive: true });
  await writeFile(outPath, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
  console.log(`VITRUM_WAVE0_BASELINE_REPORT=${outPath}`);

  const shouldFail = strict ? failed.length > 0 : requiredFailed.length > 0;
  if (shouldFail) {
    if (strict) {
      console.error(`[wave0-baseline] strict mode failed steps: ${failed.join(', ')}`);
    } else {
      console.error(`[wave0-baseline] required failed steps: ${requiredFailed.join(', ')}`);
      if (failed.length > requiredFailed.length) {
        const softFailed = failed.filter((id) => !requiredFailed.includes(id));
        console.error(`[wave0-baseline] non-blocking failures: ${softFailed.join(', ')}`);
      }
    }
    process.exit(1);
  }
}

main().catch((error) => {
  console.error(String(error));
  process.exit(1);
});
