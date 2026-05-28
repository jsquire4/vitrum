/**
 * W8 RC capture — Vite dev server in WSL, Playwright + Chrome on Windows host.
 *
 * Windows Node cannot load WSL-installed Rollup natives; Linux Playwright often
 * sees SwiftShader (10 storage buffers). This script starts vite in WSL, then
 * runs run-rc-acceptance.mjs on Windows with VITRUM_RC_START_SERVER=0.
 */

import { spawnSync } from 'node:child_process';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { launchDevServer, stopDevServer, waitForServerReady } from './devServer.mjs';
import { assertWalkaroundDevServer } from './devServer.mjs';
import { getRepoRoot } from './repoRoot.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = getRepoRoot(import.meta.url);
const benchPort = process.env.VITRUM_BENCH_DEV_PORT ?? '5199';
const viteBin = resolve(repoRoot, 'node_modules/vite/bin/vite.js');
const exampleDir = resolve(repoRoot, 'examples/two-engines-one-scene');
const base = `http://127.0.0.1:${benchPort}/`;

const extraArgs = process.argv.slice(2);

const devServer = launchDevServer(
  `node "${viteBin}" --host 0.0.0.0 --port ${benchPort} --strictPort`,
  exampleDir,
);

try {
  await waitForServerReady(devServer, base, 90_000, 500);
  await assertWalkaroundDevServer(base);
  // WSL2 forwards localhost:port to the distro; Windows Playwright uses 127.0.0.1, not the eth0 IP.
  const captureBase = `http://127.0.0.1:${benchPort}/`;
  console.log(`[rc-wsl-vite] vite ready (WSL + Windows Playwright → ${captureBase})`);

  const winArgs = [
    'run-rc-acceptance.mjs',
    'VITRUM_RC_START_SERVER=0',
    `VITRUM_RC_CAPTURE_BASE=${captureBase}`,
    `VITRUM_BENCH_DEV_PORT=${benchPort}`,
    'VITRUM_RC_REQUIRE_GPU=1',
    `VITRUM_RC_CAPTURE_FRAMES=${process.env.VITRUM_RC_CAPTURE_FRAMES ?? '32'}`,
    ...extraArgs.filter((a) => !a.startsWith('VITRUM_RC_START_SERVER')),
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
