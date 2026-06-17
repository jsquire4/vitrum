#!/usr/bin/env node
/**
 * Run the behavioral gate under the companion WSL dzn runtime.
 *
 * This is an opt-in local validation lane for full-tier pt-webgpu checks that
 * lavapipe cannot prove because of adapter resource limits. The default path
 * matches the repo owner's WSL GPU harness; override with WSL_GPU_DZN_ENV_SH
 * when running elsewhere.
 */
import { existsSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const dznEnv = process.env.WSL_GPU_DZN_ENV_SH ?? '/home/jsquire4/projects/wsl-gpu/dzn-runtime/env.sh';

if (!existsSync(dznEnv)) {
  console.error(`[behavioral-gate:dzn] missing dzn runtime env: ${dznEnv}`);
  console.error('[behavioral-gate:dzn] set WSL_GPU_DZN_ENV_SH=/path/to/dzn-runtime/env.sh or run the normal lavapipe gate.');
  process.exit(2);
}

const gateArgs = process.argv.slice(2);
const result = spawnSync(
  'bash',
  [
    '-lc',
    [
      'dzn_env="$1"',
      'node_bin="$2"',
      'shift 2',
      'source "$dzn_env"',
      'exec "$node_bin" tools/behavioral-gate/run-gate.mjs "$@"',
    ].join(' && '),
    'behavioral-gate-dzn',
    dznEnv,
    process.execPath,
    ...gateArgs,
  ],
  {
    cwd: repoRoot,
    env: process.env,
    encoding: 'utf8',
    maxBuffer: 128 * 1024 * 1024,
  },
);

if (result.stdout) process.stdout.write(result.stdout);
if (result.stderr) process.stderr.write(result.stderr);
if (result.error) throw result.error;

process.exit(result.status ?? 1);
