import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { runCommandWithTimeout } from './runCommandWithTimeout.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..', '..');
const stamp = new Date().toISOString().replace(/[:.]/g, '-');
const outDir = resolve(here, 'results', 'wave4');
const outPath = resolve(outDir, `wave4-hardening-${stamp}.json`);

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

async function readJsonIfExists(path) {
  try {
    return JSON.parse(await readFile(path, 'utf8'));
  } catch {
    return null;
  }
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

  if (step.id === 'lifecycle_soak') {
    const match = run.stdout.match(/VITRUM_LIFECYCLE_SOAK_REPORT=([^\n]+)/);
    if (match != null) {
      const reportPath = match[1].trim();
      const report = await readJsonIfExists(reportPath);
      const failures = Number(report?.summary?.failures ?? 0);
      diagnostics = {
        reportPath,
        iterations: Number(report?.summary?.total ?? 0),
        failures,
      };
      if (run.code === 0 && failures > 0) {
        status = 'warn';
      }
    }
  }

  if (step.id === 'quality_modes_smoke' && run.code === 0) {
    const match = run.stdout.match(/Wrote\s+([^\n]*quality-modes-[^\s]+\.json)/);
    if (match != null) {
      const parsed = await readJsonIfExists(match[1].trim());
      const failures = Number(parsed?.summary?.failures ?? 0);
      diagnostics = {
        ...(diagnostics ?? {}),
        resultPath: match[1].trim(),
        scenarioFailures: failures,
      };
      if (failures > 0) {
        status = 'warn';
      }
    }
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
  const strict = process.env.VITRUM_WAVE4_STRICT === '1';
  const includeMechanical = process.env.VITRUM_WAVE4_SKIP_MECHANICAL !== '1';
  const includeQualitySmoke = process.env.VITRUM_WAVE4_INCLUDE_QUALITY_SMOKE === '1';
  const includePrHybrid = process.env.VITRUM_WAVE4_INCLUDE_PR_HYBRID === '1';

  const steps = [];
  if (includeMechanical) {
    steps.push({
      id: 'mechanical',
      description: 'Workspace mechanical verification',
      required: true,
      command: 'npm run verify:mechanical',
      timeoutMs: parseTimeoutMs('VITRUM_WAVE4_MECHANICAL_TIMEOUT_MS', 45 * 60_000),
    });
  }

  steps.push({
    id: 'lifecycle_soak',
    description: 'Strict lifecycle soak benchmark',
    required: true,
    command: 'npm run benchmark:lifecycle-soak --workspace @vitrum/benchmark-runner',
    timeoutMs: parseTimeoutMs('VITRUM_WAVE4_SOAK_TIMEOUT_MS', 30 * 60_000),
    env: {
      VITRUM_LIFECYCLE_SOAK_STRICT: process.env.VITRUM_LIFECYCLE_SOAK_STRICT ?? '1',
      VITRUM_LIFECYCLE_SOAK_START_SERVER: process.env.VITRUM_LIFECYCLE_SOAK_START_SERVER ?? '1',
      VITRUM_LIFECYCLE_SOAK_ITERATIONS: process.env.VITRUM_LIFECYCLE_SOAK_ITERATIONS ?? '4',
      VITRUM_LIFECYCLE_SOAK_ITERATION_MS: process.env.VITRUM_LIFECYCLE_SOAK_ITERATION_MS ?? '3000',
    },
  });

  if (includePrHybrid) {
    steps.push({
      id: 'pr_hybrid_material_churn',
      description: 'PR-6 hybrid material-churn bench (requires two-engines dev server)',
      required: false,
      command:
        'VITRUM_PR_SCENARIO=PR-hybrid-material-churn npm run benchmark:pr-hybrid --workspace @vitrum/benchmark-runner',
      timeoutMs: parseTimeoutMs('VITRUM_WAVE4_PR_HYBRID_TIMEOUT_MS', 15 * 60_000),
    });
  }

  if (includeQualitySmoke) {
    steps.push({
      id: 'quality_modes_smoke',
      description: 'Optional quality-mode smoke benchmark',
      required: false,
      command: 'npm run benchmark:qualitymodes --workspace @vitrum/benchmark-runner',
      timeoutMs: parseTimeoutMs('VITRUM_WAVE4_QUALITY_TIMEOUT_MS', 10 * 60_000),
      env: {
        VITRUM_CAPTURE_SMOKE: '1',
      },
    });
  }

  const startedAt = nowIso();
  const results = [];
  for (const step of steps) {
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
    schemaVersion: 'vitrum-wave4-hardening-2026-05-26',
    startedAt,
    finishedAt,
    strict,
    includeMechanical,
    includeQualitySmoke,
    includePrHybrid,
    node: process.version,
    platform: process.platform,
    cwd: repoRoot,
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
  console.log(`VITRUM_WAVE4_HARDENING_REPORT=${outPath}`);

  const shouldFail = strict ? failed.length > 0 || warned.length > 0 : requiredFailed.length > 0;
  if (shouldFail) {
    if (strict) {
      console.error(`[wave4-hardening] strict-mode non-pass steps: ${[...failed, ...warned].join(', ')}`);
    } else {
      console.error(`[wave4-hardening] required failed steps: ${requiredFailed.join(', ')}`);
    }
    process.exit(1);
  }
}

main().catch((error) => {
  console.error(String(error));
  process.exit(1);
});
