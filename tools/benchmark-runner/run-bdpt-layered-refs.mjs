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

import { copyFile, mkdir, writeFile } from 'node:fs/promises';
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

async function runCapture(extraArgs) {
  const args = [`${repoRoot}/scripts/capture-cornell-suite.sh`, '--out', outDir, ...extraArgs];
  if (process.env.VITRUM_BDPT_QUICK === '1') args.push('--quick');
  return runCommandWithTimeout(args.join(' '), {
    cwd: repoRoot,
    env: process.env,
    timeoutMs: Number(process.env.VITRUM_BDPT_TIMEOUT_MS ?? 45 * 60_000),
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
    const bdpt = await runCapture(['--only', 'layered', '--bdpt']);
    steps.push({
      scenario: 'cornell-layered-bdpt',
      ok: bdpt.code === 0,
      stdout: bdpt.stdout.slice(-500),
      png: `tools/reference-renders/${label}/cornell-layered-bdpt.png`,
    });

    if (process.env.VITRUM_BDPT_SKIP_PARITY !== '1') {
      console.log('[bdpt-layered-refs] capturing cornell-parity with vitrumBdpt=1…');
      const parityBdpt = await runCapture(['--only', 'parity', '--bdpt']);
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
  try {
    await copyFile(layeredGpu, resolve(mechDir, 'cornell-layered.png'));
    if (process.env.VITRUM_BDPT_SKIP_BDPT !== '1') {
      await copyFile(bdptGpu, resolve(mechDir, 'cornell-layered-bdpt.png'));
      if (process.env.VITRUM_BDPT_SKIP_PARITY !== '1') {
        await copyFile(parityBdptGpu, resolve(mechDir, 'cornell-parity-bdpt.png'));
      }
    }
    console.log(`[bdpt-layered-refs] promoted GPU PNGs → ${mechDir}`);
  } catch (e) {
    console.warn(`[bdpt-layered-refs] could not promote to mechanical dir: ${e}`);
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
  const failed = steps.some((s) => !s.ok);
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
