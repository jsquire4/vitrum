/**
 * Shared dev-server launcher for Playwright benchmark runners.
 */

import { spawn } from 'node:child_process';

export function pushTail(buf, line, max = 40) {
  if (line.trim().length === 0) return;
  buf.push(line);
  if (buf.length > max) {
    buf.splice(0, buf.length - max);
  }
}

export function launchDevServer(command, cwd) {
  const stdoutTail = [];
  const stderrTail = [];
  const child = spawn(command, {
    cwd,
    shell: true,
    detached: process.platform !== 'win32',
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  child.stdout?.on('data', (chunk) => {
    for (const line of String(chunk).split('\n')) pushTail(stdoutTail, line);
  });
  child.stderr?.on('data', (chunk) => {
    for (const line of String(chunk).split('\n')) pushTail(stderrTail, line);
  });
  return { child, stdoutTail, stderrTail };
}

export function deriveServerUrlFromTail(stdoutTail, fallback) {
  for (let i = stdoutTail.length - 1; i >= 0; i -= 1) {
    const line = stdoutTail[i];
    const m = line.match(/https?:\/\/[^\s]+/);
    if (m != null) return m[0];
  }
  return fallback;
}

export async function waitForServerReady(procInfo, fallbackUrl, timeoutMs, pollMs) {
  const started = Date.now();
  let activeUrl = fallbackUrl;
  while (Date.now() - started < timeoutMs) {
    if (procInfo.child.exitCode != null) {
      throw new Error(
        `Dev server exited early with code ${procInfo.child.exitCode}. ` +
          `stderrTail=${procInfo.stderrTail.join(' | ')}`,
      );
    }
    const discovered = deriveServerUrlFromTail(procInfo.stdoutTail, fallbackUrl);
    if (discovered != null) activeUrl = discovered;
    try {
      const res = await fetch(activeUrl, { method: 'GET' });
      if (res.status < 500) {
        return { readyMs: Date.now() - started, url: activeUrl };
      }
    } catch {
      /* poll */
    }
    // eslint-disable-next-line no-await-in-loop
    await new Promise((r) => setTimeout(r, pollMs));
  }
  throw new Error(`Timed out waiting for dev server at ${activeUrl}.`);
}

/** Fail fast when the port serves a different Vite app (stale dev server). */
export async function assertWalkaroundDevServer(baseUrl) {
  const base = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`;
  const res = await fetch(new URL('walkaround.html', base));
  if (!res.ok) {
    throw new Error(`walkaround.html GET ${res.status} at ${base}`);
  }
  const html = await res.text();
  if (!html.includes('id="c-wgpu"')) {
    throw new Error(
      `Dev server at ${base} is not @vitrum-examples/two-engines-one-scene ` +
        '(walkaround.html missing #c-wgpu). Free the port or set VITRUM_BENCH_DEV_PORT.',
    );
  }
}

export function stopDevServer(procInfo) {
  const pid = procInfo?.child?.pid;
  if (pid == null) return;
  if (process.platform === 'win32') {
    procInfo.child.kill('SIGTERM');
    return;
  }
  try {
    process.kill(-pid, 'SIGTERM');
  } catch {
    procInfo.child.kill('SIGTERM');
  }
}
