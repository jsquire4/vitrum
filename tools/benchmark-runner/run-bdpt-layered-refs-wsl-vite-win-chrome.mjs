/**
 * BDPT + layered reference captures — cornell-box Vite in WSL, Playwright uses Windows Chrome.
 *
 * WSL bundled Chromium often sees SwiftShader WebGL (~12 KiB BDPT PNGs). Set
 * Do not set VITRUM_USE_WIN_CHROME here — Playwright in WSL cannot attach to
 * Windows Chrome (remote-debugging-pipe). Use headed WSL Chromium + hardware Vulkan.
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
  `npm run dev --workspace @vitrum-examples/cornell-box -- --host 127.0.0.1 --port ${cornellPort} --strictPort`,
  repoRoot,
);

try {
  const ready = await waitForServerReady(devServer, captureBase, 90_000, 500);
  console.log(`[bdpt-wsl-win] cornell-box vite ready at ${ready.url} (capture → ${captureBase})`);

  const result = spawnSync('node', [resolve(here, 'run-bdpt-layered-refs.mjs')], {
    cwd: repoRoot,
    stdio: 'inherit',
    encoding: 'utf8',
    env: {
      ...process.env,
      VITRUM_CORNELL_SKIP_VITE: '1',
      VITRUM_CAPTURE_URL: captureBase,
      VITRUM_CORNELL_DEV_PORT: cornellPort,
      VITRUM_BDPT_FORCE_GPU: '1',
      VITRUM_WEBGPU_ADAPTER: 'hardware',
      VITRUM_BENCH_HEADLESS: '0',
      VITRUM_BDPT_OUT_LABEL:
        process.env.VITRUM_BDPT_OUT_LABEL ?? `bdpt-layered-${new Date().toISOString().slice(0, 10)}`,
    },
  });
  process.exit(result.status ?? 1);
} finally {
  stopDevServer(devServer);
}
