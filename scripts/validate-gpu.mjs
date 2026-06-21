#!/usr/bin/env node
/**
 * validate-gpu.mjs — vitrum → wsl-gpu validation seam.
 *
 * The real-GPU validation gate lives in the sibling `wsl-gpu` repo (dzn RTX-4090
 * + lavapipe conformant oracle); it can't run in GitHub CI (runners have no GPU →
 * SwiftShader fails the hybrid tier). This script is the local invocation seam.
 * See plan/gpu-validation-gate-2026-05-29.md.
 *
 * Tiers:
 *   --smoke           T1: shader-compile on both real backends + default-render
 *                     non-regression A/B vs the stored golden. Bounded, but dzn
 *                     hybrid captures can take several minutes on WSL. Run on
 *                     every push via the warn-only pre-push hook.
 *   --items=V21,V23   T2: per-subsystem radiometric A/B. Expensive; on-demand.
 *
 * Flags:
 *   --warn-only       Always exit 0 (print a loud WARN on regression). The
 *                     pre-push hook passes this — visibility every push, never
 *                     blocks (the chosen gate hardness; see the plan §8).
 *   --timeout-ms=N    Wall-clock timeout for the delegated runner. Defaults to
 *                     300s for --smoke so the warn-only pre-push path cannot
 *                     hang indefinitely; set 0 to disable. T2 runs are unbounded
 *                     unless this flag or VITRUM_VALIDATE_GPU_TIMEOUT_MS is set.
 *
 * Graceful skip (exit 0, never an error): when wsl-gpu is absent, when its T1
 * runner isn't built yet, or when no GPU is reachable — so non-WSL contributors
 * and CI are unaffected.
 *
 * wsl-gpu location resolution order: $WSL_GPU_DIR, ../wsl-gpu, ~/projects/wsl-gpu.
 */

import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { homedir } from 'node:os';
import { fileURLToPath } from 'node:url';

const VITRUM_DIR = resolve(fileURLToPath(new URL('..', import.meta.url)));

const args = process.argv.slice(2);
const SMOKE = args.includes('--smoke');
const WARN_ONLY = args.includes('--warn-only');
const itemsArg = args.find((a) => a.startsWith('--items='));
const timeoutArg = args.find((a) => a.startsWith('--timeout-ms='));

function log(msg) { process.stderr.write(`[validate-gpu] ${msg}\n`); }

function parseTimeoutMs(raw, label) {
  if (raw === undefined || raw === '') return null;
  const value = Number(raw);
  if (!Number.isFinite(value) || value < 0 || !Number.isInteger(value)) {
    log(`invalid ${label}=${raw}; expected a non-negative integer millisecond timeout`);
    process.exit(2);
  }
  return value;
}

const explicitTimeoutMs = parseTimeoutMs(timeoutArg?.slice('--timeout-ms='.length), '--timeout-ms')
  ?? parseTimeoutMs(process.env.VITRUM_VALIDATE_GPU_TIMEOUT_MS, 'VITRUM_VALIDATE_GPU_TIMEOUT_MS');
const RUNNER_TIMEOUT_MS = explicitTimeoutMs ?? (SMOKE ? 300_000 : 0);

/** Exit per warn-only policy: code 0 always when --warn-only, else the real code. */
function finish(code, regressionMsg) {
  if (code !== 0 && regressionMsg) log(regressionMsg);
  if (WARN_ONLY && code !== 0) {
    log('──────────────────────────────────────────────────────────────');
    log('⚠️  GPU validation reported a regression (warn-only — push NOT blocked).');
    log('   Review above; re-run `npm run validate:gpu:smoke` for the full verdict.');
    log('──────────────────────────────────────────────────────────────');
    process.exit(0);
  }
  process.exit(code);
}

/** Graceful skip — never an error (so CI / non-WSL hosts pass cleanly). */
function skip(reason) {
  log(`skipped — ${reason}`);
  process.exit(0);
}

function resolveWslGpuDir() {
  const candidates = [
    process.env.WSL_GPU_DIR,
    join(VITRUM_DIR, '..', 'wsl-gpu'),
    join(homedir(), 'projects', 'wsl-gpu'),
  ].filter(Boolean);
  for (const c of candidates) {
    if (existsSync(join(c, 'package.json')) || existsSync(join(c, 'scripts'))) return resolve(c);
  }
  return null;
}

const wslGpuDir = resolveWslGpuDir();
if (!wslGpuDir) {
  skip('wsl-gpu repo not found (set $WSL_GPU_DIR, or clone github.com/jsquire4/WSL-GPU as a sibling). Real-GPU validation runs only on the WSL2 host.');
}

// T1 smoke → the dedicated cheap runner; T2 → the wave-style item runner.
const runner = SMOKE
  ? join(wslGpuDir, 'scripts', 't1-smoke.mjs')
  : join(wslGpuDir, 'scripts', 'wave7-run.mjs');

if (!existsSync(runner)) {
  skip(`wsl-gpu runner not present yet: ${runner} (build the wsl-gpu side — see plan/gpu-validation-gate-2026-05-29.md §6).`);
}

const runnerArgs = SMOKE
  ? ['--working-tree']                       // T1 validates the uncommitted working tree
  // T2: pass `--items=` through verbatim when given; otherwise pass NO flag so the
  // wave runner's filter is null → every item runs. (A literal `--items=all` is NOT
  // special-cased by the runner — its filter does token matching, so `all` would
  // match no item and silently run nothing.)
  : (itemsArg ? [itemsArg] : []);

log(`invoking ${SMOKE ? 'T1 smoke' : 'T2 radiometric'} runner: ${runner} ${runnerArgs.join(' ')}`);
if (RUNNER_TIMEOUT_MS > 0) {
  log(`runner timeout: ${RUNNER_TIMEOUT_MS}ms${WARN_ONLY ? ' (warn-only)' : ''}`);
}
const child = spawn('node', [runner, ...runnerArgs], {
  cwd: wslGpuDir,
  stdio: 'inherit',
  detached: process.platform !== 'win32',
  // The wsl-gpu runner pins/validates THIS working tree rather than vitrum's main tip.
  // The capture worker's walkaroundUbo headless shim reads the raw WGSL from
  // $VITRUM_PINNED_DIR via Deno.readTextFileSync (NOT the import map); for the
  // --working-tree smoke that must be the live tree, or the worker crashes
  // "stdout closed before ready" reading the (empty) pinned cache. T2 wave runs
  // set their own pin via the pin-vitrum workflow, so only override for --smoke.
  env: {
    ...process.env,
    VITRUM_VALIDATE_WORKTREE: VITRUM_DIR,
    ...(SMOKE ? { VITRUM_PINNED_DIR: VITRUM_DIR } : {}),
  },
});

let timedOut = false;
let timeoutHandle = null;
if (RUNNER_TIMEOUT_MS > 0) {
  timeoutHandle = setTimeout(() => {
    timedOut = true;
    log(`runner timed out after ${RUNNER_TIMEOUT_MS}ms; terminating child process tree`);
    try {
      if (process.platform !== 'win32' && child.pid !== undefined) {
        process.kill(-child.pid, 'SIGTERM');
      } else {
        child.kill('SIGTERM');
      }
    } catch (err) {
      log(`failed to terminate runner cleanly (${err instanceof Error ? err.message : String(err)})`);
    }
    setTimeout(() => {
      try {
        if (process.platform !== 'win32' && child.pid !== undefined) {
          process.kill(-child.pid, 'SIGKILL');
        } else {
          child.kill('SIGKILL');
        }
      } catch {
        // Process already exited.
      }
    }, 2_000).unref();
  }, RUNNER_TIMEOUT_MS);
  timeoutHandle.unref();
}

child.on('error', (err) => skip(`could not launch wsl-gpu runner (${err.message})`));
child.on('exit', (code, signal) => {
  if (timeoutHandle) clearTimeout(timeoutHandle);
  if (timedOut) {
    finish(124, `wsl-gpu runner timed out after ${RUNNER_TIMEOUT_MS}ms.`);
    return;
  }
  if (signal) finish(1, `runner terminated by signal ${signal}`);
  finish(code ?? 1, code !== 0 ? `wsl-gpu runner exited ${code} — GPU validation regression.` : null);
});
