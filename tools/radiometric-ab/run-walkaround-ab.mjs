#!/usr/bin/env node
/**
 * Wrapper for the walkaround-hybrid radiometric A/B harness.
 *
 * The harness itself runs under Deno native WebGPU. In the current WSL
 * validation host, Deno 2.8.1 can panic inside wgpu-hal before the harness is
 * able to write a normal verdict. This wrapper keeps that failure explicit and
 * machine-readable instead of leaving future runs as an unclassified crash.
 */
import { spawnSync } from 'node:child_process';
import { writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(scriptDir, '../..');
const statusPath = resolve(scriptDir, 'walkaround-ab-host-status.json');
const denoArgs = [
  'run',
  '--unstable-webgpu',
  '--sloppy-imports',
  '--allow-read',
  '--allow-env',
  '--allow-write',
  'tools/radiometric-ab/walkaround-ab.mjs',
];

const result = spawnSync('deno', denoArgs, {
  cwd: repoRoot,
  env: process.env,
  encoding: 'utf8',
  maxBuffer: 64 * 1024 * 1024,
});

if (result.stdout) process.stdout.write(result.stdout);
if (result.stderr) process.stderr.write(result.stderr);

if (result.error) {
  throw result.error;
}
if (result.status === 0) {
  process.exit(0);
}

const combined = `${result.stdout ?? ''}\n${result.stderr ?? ''}`;
const knownDenoWgpuPanic =
  combined.includes('Deno has panicked') &&
  combined.includes('wgpu-hal-28.0.0/src/gles/command.rs:771:21');

if (knownDenoWgpuPanic) {
  const status = {
    generatedAt: new Date().toISOString(),
    harness: 'walkaround-ab',
    verdict: 'HOST-BLOCKED',
    command: `deno ${denoArgs.join(' ')}`,
    icd: process.env.VK_ICD_FILENAMES ?? null,
    exitStatus: result.status,
    signal: result.signal,
    reason: {
      code: 'deno-wgpu-hal-gles-index-oob',
      location: 'wgpu-hal-28.0.0/src/gles/command.rs:771:21',
      message:
        'Deno native WebGPU panicked before the walkaround radiometric A/B harness could produce a verdict.',
    },
    preservedResultFile: 'tools/radiometric-ab/walkaround-ab-results.json',
    nextSteps: [
      'Run the same harness in the browser/real-adapter validation lane.',
      'Re-run this wrapper after the Deno native WebGPU panic is fixed or the host path changes.',
      'Do not promote GRIS, rich-material GI, glass, or glossy walkaround rows from this blocked native-Deno run.',
    ],
  };
  writeFileSync(statusPath, `${JSON.stringify(status, null, 2)}\n`);
  console.error(`[walkaround-ab] HOST-BLOCKED status written to ${statusPath}`);
  process.exit(2);
}

process.exit(result.status ?? 1);
