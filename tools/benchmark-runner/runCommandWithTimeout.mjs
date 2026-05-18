import { spawn } from 'node:child_process';

/**
 * Run a shell command and resolve with { code, stdout, stderr, timedOut }.
 *
 * Cross-platform process-tree termination:
 *  - POSIX: child is started in its own process group (`detached: true`),
 *    so we send SIGTERM and (after 2 s) SIGKILL to the negative PID to
 *    bring down the entire tree (Playwright spawns chromium subprocesses
 *    that would otherwise leak).
 *  - Windows: spawn doesn't support detached process groups in the same
 *    way, so we just SIGTERM the child.
 *
 * Extracted from run-gap-closure-verification.mjs (was ~60 lines inline).
 */
export function runCommandWithTimeout(command, opts = {}) {
  const { cwd = process.cwd(), env = {}, timeoutMs = 30_000 } = opts;
  return new Promise((resolveResult) => {
    let settled = false;
    const child = spawn(command, {
      cwd,
      env: { ...process.env, ...env },
      shell: true,
      detached: process.platform !== 'win32',
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (chunk) => {
      stdout += String(chunk);
    });
    child.stderr.on('data', (chunk) => {
      stderr += String(chunk);
    });
    const timeout = setTimeout(() => {
      if (settled) return;
      settled = true;
      stderr += `\nCommand timed out after ${timeoutMs}ms.`;
      if (process.platform === 'win32') {
        child.kill('SIGTERM');
      } else {
        try {
          process.kill(-child.pid, 'SIGTERM');
        } catch {
          child.kill('SIGTERM');
        }
      }
      setTimeout(() => {
        if (process.platform !== 'win32') {
          try {
            process.kill(-child.pid, 'SIGKILL');
          } catch {
            // Process group already exited.
          }
        }
      }, 2_000).unref();
      resolveResult({
        code: -1,
        stdout: stdout.trim(),
        stderr: stderr.trim(),
        timedOut: true,
      });
    }, timeoutMs);
    timeout.unref();
    child.on('close', (code) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      resolveResult({
        code: code ?? -1,
        stdout: stdout.trim(),
        stderr: stderr.trim(),
        timedOut: false,
      });
    });
  });
}
