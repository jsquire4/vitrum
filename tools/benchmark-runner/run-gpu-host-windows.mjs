/**
 * Re-invokes a benchmark-runner script on the Windows host (WSL2 GPU path).
 * Playwright inside Linux WSL often sees SwiftShader (10/4); Windows Chrome
 * can reach the RTX 4090 with hybrid-capable limits (typically 16/8).
 *
 * Usage (from repo root):
 *   node tools/benchmark-runner/run-gpu-host-windows.mjs run-rc-acceptance.mjs VITRUM_RC_REQUIRE_GPU=1
 *
 * Uses PowerShell `Set-Location -LiteralPath` on `\\wsl$\…` (CMD `pushd` / npm on UNC cwd fail).
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
const winRepo = `\\\\wsl$\\${wslDistro}${repoRoot.replace(/\//g, '\\')}`;
const winNode = process.env.VITRUM_WIN_NODE ?? 'C:\\Program Files\\nodejs\\node.exe';

const benchPort = process.env.VITRUM_BENCH_DEV_PORT ?? '5176';
const extraEnv = process.argv.slice(3);

/** @type {Record<string, string>} */
const envMap = {
  VITRUM_WEBGPU_ADAPTER: 'hardware',
  VITRUM_BENCH_HEADLESS: '0',
  VITRUM_BENCH_DEV_PORT: benchPort,
  VITRUM_PROBE_URL: `http://127.0.0.1:${benchPort}/`,
  VITRUM_CAPTURE_URL: `http://127.0.0.1:${benchPort}/walkaround.html`,
};

for (const kv of extraEnv) {
  const i = kv.indexOf('=');
  if (i < 0) continue;
  envMap[kv.slice(0, i)] = kv.slice(i + 1);
}

const envLines = Object.entries(envMap)
  .map(([k, v]) => `$env:${k}='${v.replace(/'/g, "''")}'`)
  .join('; ');

const scriptWin = script.replace(/\//g, '\\');
const ps = `
Set-Location -LiteralPath '${winRepo.replace(/'/g, "''")}';
${envLines};
& '${winNode.replace(/'/g, "''")}' 'tools\\benchmark-runner\\${scriptWin}';
exit $LASTEXITCODE
`;

const result = spawnSync(
  'powershell.exe',
  ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', ps],
  { stdio: 'inherit', encoding: 'utf8' },
);

process.exit(result.status ?? 1);
