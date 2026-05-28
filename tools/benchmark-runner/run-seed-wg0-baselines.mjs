/**
 * run-seed-wg0-baselines.mjs — WG-0.2 baseline PNG generation.
 *
 * Starts two-engines dev server (optional), runs gap-closure with
 * VITRUM_ALLOW_BASELINE_GEN=1 for WG0_PT_WEBGPU_SCENARIOS (or VITRUM_GAP_SCENARIOS).
 */

import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { runCommandWithTimeout } from './runCommandWithTimeout.mjs';
import { WG0_PT_WEBGPU_SCENARIOS } from './scenario-presets.mjs';
import {
  launchDevServer,
  stopDevServer,
  waitForServerReady,
} from './devServer.mjs';
import { getRepoRoot } from './repoRoot.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = getRepoRoot(import.meta.url);

const benchPort = process.env.VITRUM_BENCH_DEV_PORT ?? '5175';
const startServer = process.env.VITRUM_SEED_START_SERVER !== '0';
const serverCommand =
  process.env.VITRUM_SEED_DEV_CMD ??
  `npm run dev --workspace @vitrum-examples/two-engines-one-scene -- --host 127.0.0.1 --port ${benchPort}`;
const serverReadyTimeoutMs = Number(process.env.VITRUM_SEED_SERVER_READY_TIMEOUT_MS ?? 90_000);
const serverPollMs = Number(process.env.VITRUM_SEED_SERVER_POLL_MS ?? 500);
const gapTimeoutMs = Number(process.env.VITRUM_SEED_GAP_TIMEOUT_MS ?? 20 * 60_000);

const scenarioIds =
  (process.env.VITRUM_GAP_SCENARIOS ?? '')
    .split(',')
    .map((s) => s.trim())
    .filter((s) => s.length > 0).length > 0
    ? process.env.VITRUM_GAP_SCENARIOS
    : WG0_PT_WEBGPU_SCENARIOS.join(',');

async function main() {
  let devServer = null;
  let captureUrl =
    process.env.VITRUM_CAPTURE_URL ?? `http://127.0.0.1:${benchPort}/`;

  if (startServer) {
    console.log('[seed-wg0] starting dev server…');
    devServer = launchDevServer(serverCommand, repoRoot);
    const ready = await waitForServerReady(
      devServer,
      captureUrl,
      serverReadyTimeoutMs,
      serverPollMs,
    );
    captureUrl = ready.url;
    console.log(`[seed-wg0] server ready in ${ready.readyMs}ms at ${captureUrl}`);
    const probe = await runCommandWithTimeout(
      'node ./run-pt-webgpu-adapter-probe.mjs',
      {
        cwd: here,
        env: {
          VITRUM_PROBE_URL: captureUrl.endsWith('/')
            ? `${captureUrl}pt-webgpu.html`
            : `${captureUrl}/pt-webgpu.html`,
        },
        timeoutMs: 60_000,
      },
    );
    const probeLine = probe.stdout.split('\n').find((l) => l.trim().startsWith('{'));
    if (probeLine != null) {
      try {
        const parsed = JSON.parse(probeLine);
        if (parsed.ptWebgpuCanRun === false) {
          console.error(
            '[seed-wg0] adapter cannot run pt-webgpu (need full tier ≥10 storage buffers + ≥5 textures, or lite tier ≥8 buffers + ≥4 textures). ' +
              `Host reports buffers=${parsed.maxStorageBuffersPerShaderStage}, textures=${parsed.maxStorageTexturesPerShaderStage}. ` +
              'Use a WebGPU-capable machine or set VITRUM_SEED_SKIP_PROBE=1 to attempt anyway.',
          );
          if (process.env.VITRUM_SEED_SKIP_PROBE !== '1') {
            stopDevServer(devServer);
            process.exit(2);
          }
        }
      } catch {
        /* ignore */
      }
    }
  }

  const run = await runCommandWithTimeout(`node ./tools/benchmark-runner/run-gap-closure-verification.mjs`, {
    cwd: repoRoot,
    env: {
      VITRUM_GPU_CAPTURE: '1',
      VITRUM_ALLOW_BASELINE_GEN: '1',
      VITRUM_GAP_SCENARIOS: scenarioIds,
      VITRUM_CAPTURE_URL: captureUrl.endsWith('/')
        ? `${captureUrl}pt-webgpu.html`
        : `${captureUrl}/pt-webgpu.html`,
      VITRUM_CAPTURE_SMOKE: process.env.VITRUM_CAPTURE_SMOKE ?? '1',
    },
    timeoutMs: gapTimeoutMs,
  });

  if (devServer) stopDevServer(devServer);

  if (run.code !== 0) {
    console.error(run.stderr || run.stdout);
    process.exit(run.code ?? 1);
  }
  console.log(run.stdout);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
