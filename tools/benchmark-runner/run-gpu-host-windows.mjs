/**
 * Re-invokes a benchmark-runner script on the Windows host (WSL2 GPU path).
 * Playwright inside Linux WSL often sees SwiftShader (10/4); Windows Chrome
 * can reach the RTX 4090 with hybrid-capable limits (typically 16/8).
 *
 * Usage (from repo root):
 *   node tools/benchmark-runner/run-gpu-host-windows.mjs run-pt-webgpu-adapter-probe.mjs
 *   node tools/benchmark-runner/run-gpu-host-windows.mjs run-pr-hybrid-bench.mjs
 *
 * Starts the two-engines Vite dev server on the Windows host (port 5176 by
 * default). WSL localhost:5175 often points at a different app than WSL curl.
 */

import { spawnSync } from 'node:child_process';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const script = process.argv[2];
if (script == null || script.length === 0) {
  console.error('Usage: node run-gpu-host-windows.mjs <runner.mjs> [extra env KEY=VAL ...]');
  process.exit(2);
}

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..', '..');
const wslDistro = process.env.WSL_DISTRO_NAME ?? 'Ubuntu-24.04';
const winRepo = `\\\\wsl.localhost\\${wslDistro}${repoRoot.replace(/\//g, '\\')}`;
const winNode = 'C:\\Program Files\\nodejs\\node.exe';

const benchPort = process.env.VITRUM_BENCH_DEV_PORT ?? '5176';
const extraEnv = process.argv.slice(3);
const envLines = [
  '$env:VITRUM_WEBGPU_ADAPTER="hardware"',
  '$env:VITRUM_BENCH_HEADLESS="0"',
  `$env:VITRUM_BENCH_DEV_PORT="${benchPort}"`,
  `$env:VITRUM_PROBE_URL="http://127.0.0.1:${benchPort}/"`,
  `$env:VITRUM_CAPTURE_URL="http://127.0.0.1:${benchPort}/walkaround.html"`,
  '$env:VITRUM_PR_START_SERVER="1"',
  ...extraEnv.map((kv) => {
    const i = kv.indexOf('=');
    if (i < 0) return '';
    const k = kv.slice(0, i);
    const v = kv.slice(i + 1);
    return `$env:${k}="${v.replace(/"/g, '`"')}"`;
  }),
]
  .filter(Boolean)
  .join('; ');

const ps = `
$env:VITRUM_REPO_ROOT = '${winRepo.replace(/'/g, "''")}';
Push-Location $env:VITRUM_REPO_ROOT;
try {
  ${envLines};
  & '${winNode}' 'tools\\benchmark-runner\\${script.replace(/\//g, '\\')}';
  exit $LASTEXITCODE
} finally {
  Pop-Location
}
`;

const result = spawnSync(
  'powershell.exe',
  ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', ps],
  { stdio: 'inherit', encoding: 'utf8' },
);

process.exit(result.status ?? 1);
