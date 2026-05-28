/**
 * GPU A/B workflow — layered BSDF + BDPT reference captures via cornell-box example.
 *
 * Wraps `scripts/capture-cornell-suite.sh` for the layered and BDPT scenarios and
 * records manifest paths for host A/B. Does not assert image quality (eyeball / fidelity tests).
 *
 * Env:
 *   VITRUM_BDPT_ONLY_LAYERED  1 — skip BDPT pass
 *   VITRUM_BDPT_SKIP_BDPT      1 — skip layered-only pass
 *   VITRUM_BDPT_OUT_LABEL      output subdir under tools/reference-renders/ (default bdpt-layered-YYYY-MM-DD)
 *   VITRUM_BDPT_QUICK          1 — forward --quick to capture script
 *   VITRUM_BDPT_REQUIRE_GPU    1 — exit 1 when capture script fails (default 0 logs only)
 */

import { copyFile, mkdir, stat, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { runCommandWithTimeout } from './runCommandWithTimeout.mjs';
import { getRepoRoot } from './repoRoot.mjs';

const repoRoot = getRepoRoot(import.meta.url);
const label =
  process.env.VITRUM_BDPT_OUT_LABEL ??
  `bdpt-layered-${new Date().toISOString().slice(0, 10)}`;
const outDir = resolve(repoRoot, 'tools/reference-renders', label);
const here = dirname(fileURLToPath(import.meta.url));

function bdptCaptureBaseUrl() {
  const raw = process.env.VITRUM_CAPTURE_URL ?? 'http://127.0.0.1:5173/';
  const u = new URL(raw);
  u.searchParams.set('vitrumBdpt', '1');
  if (!u.searchParams.has('vitrumSpf')) {
    u.searchParams.set('vitrumSpf', process.env.VITRUM_BDPT_QUICK === '1' ? '64' : '16');
  }
  if (process.env.VITRUM_BDPT_CPU_FILL === '1') {
    u.searchParams.set('vitrumBdptCpuFill', '1');
  } else {
    u.searchParams.delete('vitrumBdptCpuFill');
  }
  return u.toString();
}

async function runCapture(extraArgs, envExtra = {}) {
  const timeoutMs = Number(process.env.VITRUM_BDPT_TIMEOUT_MS ?? 45 * 60_000);
  const captureEnv = {
    ...process.env,
    ...envExtra,
    VITRUM_BDPT_MIN_PNG_BYTES: envExtra.VITRUM_BDPT_MIN_PNG_BYTES ?? process.env.VITRUM_BDPT_MIN_PNG_BYTES ?? '50000',
  };
  if (process.env.VITRUM_BDPT_QUICK === '1') {
    const raw = captureEnv.VITRUM_CAPTURE_URL ?? 'http://127.0.0.1:5173/';
    const u = new URL(raw);
    if (!u.searchParams.has('vitrumSpf')) u.searchParams.set('vitrumSpf', '64');
    captureEnv.VITRUM_CAPTURE_URL = u.toString();
  }
  if (process.env.VITRUM_BDPT_NODE_CAPTURE === '1') {
    const args = [
      'node',
      resolve(here, 'capture-cornell-scenarios.mjs'),
      '--out',
      outDir,
      ...extraArgs,
    ];
    if (process.env.VITRUM_BDPT_QUICK === '1') args.push('--quick');
    return runCommandWithTimeout(args.join(' '), {
      cwd: repoRoot,
      env: captureEnv,
      timeoutMs,
    });
  }
  const args = [`${repoRoot}/scripts/capture-cornell-suite.sh`, '--out', outDir, ...extraArgs];
  if (process.env.VITRUM_BDPT_QUICK === '1') args.push('--quick');
  return runCommandWithTimeout(args.join(' '), {
    cwd: repoRoot,
    env: captureEnv,
    timeoutMs,
  });
}

async function main() {
  await mkdir(outDir, { recursive: true });
  const steps = [];

  if (process.env.VITRUM_BDPT_SKIP_LAYERED !== '1') {
    console.log('[bdpt-layered-refs] capturing cornell-layered…');
    const layered = await runCapture(['--only', 'layered']);
    steps.push({ scenario: 'cornell-layered', ok: layered.code === 0, stdout: layered.stdout.slice(-500) });
  }

  if (process.env.VITRUM_BDPT_SKIP_BDPT !== '1') {
    console.log('[bdpt-layered-refs] capturing cornell-layered with vitrumBdpt=1…');
    const bdptEnv = { VITRUM_CAPTURE_URL: bdptCaptureBaseUrl() };
    if (process.env.VITRUM_BDPT_CPU_FILL === '1') bdptEnv.VITRUM_BDPT_CPU_FILL = '1';
    const bdpt = await runCapture(['--only', 'layered', '--bdpt'], bdptEnv);
    steps.push({
      scenario: 'cornell-layered-bdpt',
      ok: bdpt.code === 0,
      stdout: bdpt.stdout.slice(-500),
      png: `tools/reference-renders/${label}/cornell-layered-bdpt.png`,
    });

    if (process.env.VITRUM_BDPT_SKIP_PARITY !== '1') {
      console.log('[bdpt-layered-refs] capturing cornell-parity with vitrumBdpt=1…');
      const parityBdpt = await runCapture(['--only', 'parity', '--bdpt'], bdptEnv);
      steps.push({
        scenario: 'cornell-parity-bdpt',
        ok: parityBdpt.code === 0,
        stdout: parityBdpt.stdout.slice(-500),
        png: `tools/reference-renders/${label}/cornell-parity-bdpt.png`,
      });
    }
  }

  const layeredGpu = resolve(outDir, 'cornell-layered.png');
  const bdptGpu = resolve(outDir, 'cornell-layered-bdpt.png');
  const parityBdptGpu = resolve(outDir, 'cornell-parity-bdpt.png');
  const mechDir = resolve(repoRoot, 'tools/reference-renders/bdpt-layered-mechanical');
  const minPromoteBytes = Number(process.env.VITRUM_BDPT_MIN_PROMOTE_BYTES ?? 50_000);
  async function promoteIfGpu(path, dest, label) {
    const st = await stat(path);
    if (st.size < minPromoteBytes) {
      console.warn(
        `[bdpt-layered-refs] skip promote ${label}: ${st.size} bytes (< ${minPromoteBytes}); keeping existing mechanical fixture`,
      );
      return false;
    }
    await copyFile(path, dest);
    return true;
  }
  const failed = steps.some((s) => !s.ok);
  if (!failed) {
    try {
      const layeredStep = steps.find((s) => s.scenario === 'cornell-layered');
      if (layeredStep?.ok) {
        await promoteIfGpu(layeredGpu, resolve(mechDir, 'cornell-layered.png'), 'cornell-layered');
      }
      if (process.env.VITRUM_BDPT_SKIP_BDPT !== '1') {
        const bdptStep = steps.find((s) => s.scenario === 'cornell-layered-bdpt');
        if (bdptStep?.ok) {
          await promoteIfGpu(bdptGpu, resolve(mechDir, 'cornell-layered-bdpt.png'), 'cornell-layered-bdpt');
        }
        if (process.env.VITRUM_BDPT_SKIP_PARITY !== '1') {
          const parityStep = steps.find((s) => s.scenario === 'cornell-parity-bdpt');
          if (parityStep?.ok) {
            await promoteIfGpu(
              parityBdptGpu,
              resolve(mechDir, 'cornell-parity-bdpt.png'),
              'cornell-parity-bdpt',
            );
          }
        }
      }
      console.log(`[bdpt-layered-refs] promoted GPU PNGs (when ≥${minPromoteBytes} B) → ${mechDir}`);
    } catch (e) {
      console.warn(`[bdpt-layered-refs] could not promote to mechanical dir: ${e}`);
    }
  } else {
    console.warn('[bdpt-layered-refs] skipping mechanical promotion — one or more captures failed');
  }

  const manifest = {
    capturedAt: new Date().toISOString(),
    outDir: `tools/reference-renders/${label}`,
    steps,
    note:
      'Layered BSDF fork patch (Sprint 14) + BDPT dispatch (Sprint 10c). Compare cornell-layered.png vs BDPT variant visually; pt-webgpu parity in gap-closure rfe03.',
  };
  const manifestPath = resolve(outDir, 'manifest.json');
  await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, 'utf8');
  console.log(`[bdpt-layered-refs] wrote ${manifestPath}`);
  if (failed) {
    if (process.env.VITRUM_BDPT_REQUIRE_GPU === '1') process.exit(1);
    console.warn('[bdpt-layered-refs] capture failed (GPU/Playwright required); manifest written for audit.');
    process.exit(0);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
