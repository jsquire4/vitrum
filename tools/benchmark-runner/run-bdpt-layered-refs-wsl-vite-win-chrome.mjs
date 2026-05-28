/**
 * BDPT + layered reference captures — cornell-box Vite in WSL, Playwright on Windows Chrome.
 *
 * WSL Playwright often sees SwiftShader WebGL (~12 KiB BDPT PNGs). Windows Chrome
 * gets hardware WebGL for the pt-webgl path.
 */

import { spawnSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { launchDevServer, stopDevServer, waitForServerReady } from './devServer.mjs';
import { getRepoRoot } from './repoRoot.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = getRepoRoot(import.meta.url);
const cornellPort = process.env.VITRUM_CORNELL_DEV_PORT ?? '5173';
const base = `http://127.0.0.1:${cornellPort}/`;

const devServer = launchDevServer(
  `npm run dev --workspace @vitrum-examples/cornell-box -- --host 0.0.0.0 --port ${cornellPort} --strictPort`,
  repoRoot,
);

try {
  const ready = await waitForServerReady(devServer, base, 90_000, 500);
  console.log(`[bdpt-wsl-win] cornell-box vite ready at ${ready.url}`);

  const winArgs = [
    'run-bdpt-layered-refs.mjs',
    'VITRUM_CORNELL_SKIP_VITE=1',
    `VITRUM_CAPTURE_URL=${base}`,
    `VITRUM_CORNELL_DEV_PORT=${cornellPort}`,
    'VITRUM_USE_WIN_CHROME=1',
    'VITRUM_BDPT_FORCE_GPU=1',
    'VITRUM_WEBGPU_ADAPTER=hardware',
    'VITRUM_BENCH_HEADLESS=0',
    `VITRUM_BDPT_OUT_LABEL=${process.env.VITRUM_BDPT_OUT_LABEL ?? `bdpt-layered-${new Date().toISOString().slice(0, 10)}`}`,
    ...process.argv.slice(2).filter((a) => !a.startsWith('VITRUM_CORNELL_')),
  ];

  const result = spawnSync('node', [resolve(here, 'run-gpu-host-windows.mjs'), ...winArgs], {
    cwd: repoRoot,
    stdio: 'inherit',
    encoding: 'utf8',
  });
  process.exit(result.status ?? 1);
} finally {
  stopDevServer(devServer);
}
