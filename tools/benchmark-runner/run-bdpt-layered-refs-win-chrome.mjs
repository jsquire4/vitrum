/**
 * BDPT + layered refs — cornell-box Vite in WSL, Playwright on Windows (hardware WebGL).
 *
 * Primary promotion path for `bdpt-layered-mechanical/` (~1.2 MiB PNGs). Does not
 * set vitrumBdptCpuFill — uses GPU light-subpath + eye path on ANGLE.
 */

import { execSync, spawnSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { launchDevServer, stopDevServer, waitForServerReady } from './devServer.mjs';
import { getRepoRoot } from './repoRoot.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = getRepoRoot(import.meta.url);
const cornellPort = process.env.VITRUM_CORNELL_DEV_PORT ?? '5173';

/** Windows Playwright → WSL Vite: prefer WSL eth IP when localhost forwarding fails. */
function wslCaptureBase(port) {
  if (process.env.VITRUM_CAPTURE_URL) return process.env.VITRUM_CAPTURE_URL;
  try {
    const ip = execSync("hostname -I 2>/dev/null | awk '{print $1}'", {
      encoding: 'utf8',
      shell: '/bin/bash',
    }).trim();
    if (ip.length > 0) return `http://${ip}:${port}/`;
  } catch {
    // fall through
  }
  return `http://127.0.0.1:${port}/`;
}

const captureBase = wslCaptureBase(cornellPort);
console.info(`[bdpt-win] capture base URL: ${captureBase}`);

const devServer = launchDevServer(
  `npm run dev --workspace @vitrum-examples/cornell-box -- --host 0.0.0.0 --port ${cornellPort} --strictPort`,
  repoRoot,
);

try {
  await waitForServerReady(devServer, captureBase, 90_000, 500);
  const label =
    process.env.VITRUM_BDPT_OUT_LABEL ?? `bdpt-layered-${new Date().toISOString().slice(0, 10)}`;

  const runWinCapture = (extraEnv) =>
    spawnSync(
      'node',
      [
        resolve(here, 'run-gpu-host-windows.mjs'),
        'run-bdpt-layered-refs.mjs',
      'VITRUM_BDPT_NODE_CAPTURE=1',
      'VITRUM_CORNELL_SKIP_VITE=1',
      `VITRUM_CAPTURE_URL=${captureBase}`,
      `VITRUM_CORNELL_DEV_PORT=${cornellPort}`,
      'VITRUM_WEBGPU_ADAPTER=hardware',
      'VITRUM_BENCH_HEADLESS=0',
      'VITRUM_BDPT_REQUIRE_GPU=1',
      'VITRUM_CAPTURE_TIMEOUT_MS=600000',
      'VITRUM_BDPT_MIN_PNG_BYTES=400000',
      `VITRUM_BDPT_QUICK=${process.env.VITRUM_BDPT_QUICK ?? '1'}`,
      `VITRUM_BDPT_OUT_LABEL=${label}`,
      ...(extraEnv ?? []),
    ],
    { cwd: repoRoot, stdio: 'inherit', encoding: 'utf8' },
  );

  let result = runWinCapture();
  if ((result.status ?? 1) !== 0 && process.env.VITRUM_BDPT_NO_CPU_FALLBACK !== '1') {
    console.warn(
      '[bdpt-win] hardware capture failed — retrying with vitrumBdptCpuFill=1 (WSL GPU path: benchmark:bdpt-layered-refs-gpu-wsl)',
    );
    result = runWinCapture(['VITRUM_BDPT_CPU_FILL=1', 'VITRUM_BDPT_MIN_PNG_BYTES=50000']);
  }
  process.exit(result.status ?? 1);
} finally {
  stopDevServer(devServer);
}
