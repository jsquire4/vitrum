/**
 * PT-WebGL fidelity oracle.
 *
 * Default mode is mechanical: consume the committed/reference paired PNGs,
 * compute strict metrics, then run the env-gated pt-webgl acceptance test.
 *
 * Capture mode (`VITRUM_PTWEBGL_ORACLE_CAPTURE=1`) starts the two-engines Vite
 * example, captures pt-webgpu baselines and pt-webgl candidates for the RFE
 * scenarios, then runs the same strict metrics + acceptance test against the
 * generated capture directory.
 */

import { access, mkdir } from 'node:fs/promises';
import { constants as fsConstants } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { runCommandWithTimeout } from './runCommandWithTimeout.mjs';
import { GAP_CLOSURE_SCENARIOS } from './scenario-presets.mjs';
import { getRepoRoot } from './repoRoot.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = getRepoRoot(import.meta.url);
const capture = process.env.VITRUM_PTWEBGL_ORACLE_CAPTURE === '1';
const port = Number(process.env.VITRUM_PTWEBGL_ORACLE_PORT ?? '5175');
const devBaseUrl = process.env.VITRUM_PTWEBGL_ORACLE_BASE_URL ?? `http://127.0.0.1:${port}/`;
const outputDir = resolve(
  repoRoot,
  process.env.VITRUM_PTWEBGL_ORACLE_DIR ??
    (capture
      ? `tools/benchmark-runner/results/pt-webgl-oracle-${new Date().toISOString().replace(/[:.]/g, '-')}`
      : 'tools/reference-renders/pt-webgl-fidelity'),
);
const metricsPath = resolve(
  repoRoot,
  process.env.VITRUM_PTWEBGL_FIDELITY_OUT ??
    `tools/benchmark-runner/results/pt-webgl-fidelity-oracle-${new Date().toISOString().replace(/[:.]/g, '-')}.json`,
);
const timeoutMs = Math.max(10_000, Number(process.env.VITRUM_PTWEBGL_ORACLE_TIMEOUT_MS ?? '180000'));
const strictRequired = process.env.VITRUM_PTWEBGL_FIDELITY_REQUIRED ??
  'rfe03-layered-front-back,rfe05-caustic-strategy,rfe07-11-sss-mixed-panels,rfe08-13-spectral-payload,rfe09-bridge-global-cmf,rfe14-thinfilm-angle-shift';

async function fileExists(path) {
  try {
    await access(path, fsConstants.F_OK);
    return true;
  } catch {
    return false;
  }
}

function parseResolution(resolution) {
  const [w, h] = String(resolution).split('x').map((v) => Number(v));
  return {
    width: Number.isFinite(w) && w > 0 ? w : 1280,
    height: Number.isFinite(h) && h > 0 ? h : 720,
  };
}

function smokeScenario(scenario) {
  if (process.env.VITRUM_CAPTURE_SMOKE !== '1') return scenario;
  const { width, height } = parseResolution(scenario.resolution);
  const maxW = Math.max(1, Number(process.env.VITRUM_SMOKE_MAX_WIDTH ?? '320'));
  const maxH = Math.max(1, Number(process.env.VITRUM_SMOKE_MAX_HEIGHT ?? '180'));
  const scale = Math.min(1, maxW / Math.max(width, 1), maxH / Math.max(height, 1));
  return {
    ...scenario,
    resolution: `${Math.max(1, Math.floor(width * scale))}x${Math.max(1, Math.floor(height * scale))}`,
    spp: Math.min(Number(scenario.spp ?? 64), Number(process.env.VITRUM_SMOKE_MAX_SPP ?? '8')),
    bounces: Math.min(Number(scenario.bounces ?? 8), Number(process.env.VITRUM_SMOKE_MAX_BOUNCES ?? '4')),
  };
}

function oracleRows() {
  const required = strictRequired.split(',').map((s) => s.trim()).filter(Boolean);
  const scenarios = GAP_CLOSURE_SCENARIOS.filter((s) => required.some((r) => s.scenarioId === r));
  return scenarios.map((scenario) => {
    if (scenario.scenarioId === 'rfe05-caustic-strategy') {
      return {
        id: 'rfe05-caustic-strategy.manifold-nee',
        scenario: smokeScenario(scenario),
        caustic: 'manifold-nee',
      };
    }
    return {
      id: scenario.scenarioId,
      scenario: smokeScenario(scenario),
      caustic: '',
    };
  });
}

async function waitForServer(url, ms = 45_000) {
  const deadline = Date.now() + ms;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(url);
      if (res.ok) return;
    } catch {
      // keep polling
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error(`Timed out waiting for ${url}`);
}

function startServerIfNeeded() {
  if (!capture || process.env.VITRUM_PTWEBGL_ORACLE_SKIP_SERVER === '1') return null;
  const child = spawn(
    'npm',
    [
      'run',
      'dev',
      '--workspace',
      '@vitrum-examples/two-engines-one-scene',
      '--',
      '--host',
      '127.0.0.1',
      '--port',
      String(port),
    ],
    {
      cwd: repoRoot,
      env: { ...process.env },
      stdio: ['ignore', 'pipe', 'pipe'],
    },
  );
  child.stdout?.on('data', (chunk) => process.stderr.write(String(chunk)));
  child.stderr?.on('data', (chunk) => process.stderr.write(String(chunk)));
  return child;
}

async function runChecked(command, env, label) {
  const result = await runCommandWithTimeout(command, {
    cwd: repoRoot,
    env,
    timeoutMs,
  });
  if (result.code !== 0) {
    throw new Error(
      `[pt-webgl-oracle] ${label} failed with exit ${result.code}\n` +
        `${result.stderr || result.stdout || '(no output)'}`,
    );
  }
  return result;
}

async function captureRows() {
  await mkdir(outputDir, { recursive: true });
  const rows = oracleRows();
  for (const row of rows) {
    const { width, height } = parseResolution(row.scenario.resolution);
    const commonEnv = {
      VITRUM_SCENARIO_ID: row.scenario.scenarioId,
      VITRUM_SCENARIO_JSON: JSON.stringify(row.scenario),
      VITRUM_SEED: String(row.scenario.seed ?? 777),
      VITRUM_WIDTH: String(width),
      VITRUM_HEIGHT: String(height),
      VITRUM_BOUNCES: String(row.scenario.bounces ?? 8),
      VITRUM_SPP: String(row.scenario.spp ?? 64),
      VITRUM_CAUSTIC_STRATEGY: row.caustic,
      VITRUM_CAPTURE_TIMEOUT_MS: String(timeoutMs),
    };
    await runChecked('node ./tools/benchmark-runner/capturePtWebgpu.mjs', {
      ...commonEnv,
      VITRUM_CAPTURE_URL: `${devBaseUrl}pt-webgpu.html`,
      VITRUM_OUTPUT_PNG: resolve(outputDir, `${row.id}.baseline.png`),
    }, `${row.id} pt-webgpu baseline`);
    await runChecked('node ./tools/benchmark-runner/capturePtWebgl.mjs', {
      ...commonEnv,
      VITRUM_CAPTURE_URL: `${devBaseUrl}pt-webgl.html`,
      VITRUM_OUTPUT_PNG: resolve(outputDir, `${row.id}.candidate.png`),
    }, `${row.id} pt-webgl candidate`);
  }
}

async function assertFixturePairsPresent() {
  const missing = [];
  for (const row of oracleRows()) {
    for (const suffix of ['baseline', 'candidate']) {
      const path = resolve(outputDir, `${row.id}.${suffix}.png`);
      if (!(await fileExists(path))) missing.push(path);
    }
  }
  if (missing.length > 0) {
    throw new Error(
      `[pt-webgl-oracle] missing fixture pair(s):\n${missing.join('\n')}\n` +
        'Run with VITRUM_PTWEBGL_ORACLE_CAPTURE=1 to generate them.',
    );
  }
}

async function main() {
  let server = null;
  try {
    server = startServerIfNeeded();
    if (capture) {
      await waitForServer(devBaseUrl);
      await captureRows();
    }
    await assertFixturePairsPresent();
    await runChecked('node ./tools/benchmark-runner/run-pt-webgl-fidelity-acceptance.mjs', {
      VITRUM_PTWEBGL_FIDELITY_DIR: outputDir,
      VITRUM_PTWEBGL_FIDELITY_OUT: metricsPath,
      VITRUM_PTWEBGL_FIDELITY_REQUIRED: strictRequired,
      VITRUM_PTWEBGL_FIDELITY_STRICT: '1',
      VITRUM_PTWEBGL_FIDELITY_MIN_PSNR: process.env.VITRUM_PTWEBGL_FIDELITY_MIN_PSNR ?? '28',
      VITRUM_PTWEBGL_FIDELITY_MIN_PSNR_BY_SCENARIO:
        process.env.VITRUM_PTWEBGL_FIDELITY_MIN_PSNR_BY_SCENARIO ?? '',
    }, 'strict metrics');
    await runChecked('npm test --workspace @vitrum/pt-webgl -- fidelityAcceptance', {
      VITRUM_PTWEBGL_FIDELITY_ACCEPTANCE: '1',
      VITRUM_PTWEBGL_FIDELITY_METRICS: metricsPath,
    }, 'env-gated vitest acceptance');
    console.log(`VITRUM_PTWEBGL_FIDELITY_METRICS=${metricsPath}`);
  } finally {
    if (server != null) {
      server.kill('SIGTERM');
      setTimeout(() => server.kill('SIGKILL'), 2_000).unref();
    }
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
