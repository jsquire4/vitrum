/**
 * BDPT refs via Windows Playwright (experimental). Prefer run-bdpt-layered-refs-wsl-vite-win-chrome.mjs
 * (WSL captures) until ANGLE + FEATURE_BDPT dark-frame issue is resolved.
 */

import { spawnSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { launchDevServer, stopDevServer, waitForServerReady } from './devServer.mjs';
import { getRepoRoot } from './repoRoot.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = getRepoRoot(import.meta.url);
const cornellPort = process.env.VITRUM_CORNELL_DEV_PORT ?? '5173';
const captureBase = `http://127.0.0.1:${cornellPort}/`;

const devServer = launchDevServer(
  `npm run dev --workspace @vitrum-examples/cornell-box -- --host 0.0.0.0 --port ${cornellPort} --strictPort`,
  repoRoot,
);

try {
  await waitForServerReady(devServer, captureBase, 90_000, 500);
  const label =
    process.env.VITRUM_BDPT_OUT_LABEL ?? `bdpt-layered-${new Date().toISOString().slice(0, 10)}`;

  const result = spawnSync(
    'node',
    [
      resolve(here, 'run-gpu-host-windows.mjs'),
      'run-bdpt-layered-refs.mjs',
      'VITRUM_BDPT_NODE_CAPTURE=1',
      'VITRUM_CORNELL_SKIP_VITE=1',
      `VITRUM_CAPTURE_URL=${captureBase}`,
      `VITRUM_CORNELL_DEV_PORT=${cornellPort}`,
      'VITRUM_BDPT_CPU_FILL=1',
      'VITRUM_WEBGPU_ADAPTER=hardware',
      'VITRUM_BENCH_HEADLESS=0',
      'VITRUM_BDPT_REQUIRE_GPU=1',
      `VITRUM_BDPT_QUICK=${process.env.VITRUM_BDPT_QUICK ?? '1'}`,
      `VITRUM_BDPT_OUT_LABEL=${label}`,
    ],
    { cwd: repoRoot, stdio: 'inherit', encoding: 'utf8' },
  );
  process.exit(result.status ?? 1);
} finally {
  stopDevServer(devServer);
}
