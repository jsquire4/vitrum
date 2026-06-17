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
import { writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(scriptDir, '../..');
const statusPath = resolve(scriptDir, 'behavioral-gate-dzn-host-status.json');
const dznEnv = process.env.WSL_GPU_DZN_ENV_SH ?? '/home/jsquire4/projects/wsl-gpu/dzn-runtime/env.sh';
const timeoutMs = parseTimeoutMs(process.env.VITRUM_BEHAVIORAL_GATE_DZN_TIMEOUT_MS, 180_000);
const timeoutSeconds = Math.max(1, Math.ceil(timeoutMs / 1000));

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
      'timeout_seconds="$3"',
      'shift 3',
      'source "$dzn_env"',
      'exec timeout --kill-after=10s "${timeout_seconds}s" "$node_bin" tools/behavioral-gate/run-gate.mjs "$@"',
    ].join(' && '),
    'behavioral-gate-dzn',
    dznEnv,
    process.execPath,
    String(timeoutSeconds),
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

if (result.status === 124) {
  const filter = readFlagValue(gateArgs, '--filter');
  const status = {
    generatedAt: new Date().toISOString(),
    harness: 'behavioral-gate:dzn',
    verdict: 'HOST-BLOCKED',
    command: `npm run behavioral-gate:dzn -- ${gateArgs.join(' ')}`.trim(),
    filter: filter || null,
    timeoutMs,
    dznEnv,
    exitStatus: result.status,
    signal: result.signal,
    reason: {
      code: 'dzn-behavioral-gate-timeout',
      message:
        'The dzn behavioral gate exceeded its configured timeout before the inner gate returned a structured verdict.',
    },
    nextSteps: [
      'Re-run with a narrower --filter or a larger VITRUM_BEHAVIORAL_GATE_DZN_TIMEOUT_MS when collecting full-tier evidence.',
      'Use the lavapipe behavioral gate for lite-tier/API proof when full-tier validation is not required.',
      'If the same narrow dzn lane repeatedly times out, treat it as a WSL/dzn validation-host issue until a browser/real-adapter run confirms otherwise.',
    ],
  };
  writeFileSync(statusPath, `${JSON.stringify(status, null, 2)}\n`);
  console.error(`[behavioral-gate:dzn] HOST-BLOCKED timeout status written to ${statusPath}`);
  process.exit(2);
}

process.exit(result.status ?? 1);

function parseTimeoutMs(raw, fallback) {
  if (raw == null || raw === '') return fallback;
  const n = Number(raw);
  if (!Number.isFinite(n) || n <= 0) {
    console.warn(`[behavioral-gate:dzn] ignoring invalid VITRUM_BEHAVIORAL_GATE_DZN_TIMEOUT_MS=${JSON.stringify(raw)}; using ${fallback}.`);
    return fallback;
  }
  return Math.max(1000, Math.floor(n));
}

function readFlagValue(args, name) {
  const eq = args.find((a) => a.startsWith(`${name}=`));
  if (eq) return eq.slice(name.length + 1);
  const i = args.indexOf(name);
  return i >= 0 ? (args[i + 1] ?? '') : '';
}
