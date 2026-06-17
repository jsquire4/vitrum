#!/usr/bin/env node
/**
 * Wrapper for tools/behavioral-gate/gate.mjs.
 *
 * The gate itself is a Deno native-WebGPU program. Some WSL/Deno 2.8.1
 * walkaround lanes currently panic inside wgpu-hal before gate.mjs can return a
 * structured result. Keep normal runs byte-for-byte visible on stdout/stderr,
 * but classify the known host panic into a checked-in JSON status so the
 * validation queue can distinguish "engine failed" from "host crashed".
 */
import { spawnSync } from 'node:child_process';
import { writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(scriptDir, '../..');
const statusPath = resolve(scriptDir, 'behavioral-gate-host-status.json');

const gateArgs = process.argv.slice(2);
const denoArgs = [
  'run',
  '--unstable-webgpu',
  '--sloppy-imports',
  '--allow-read',
  '--allow-env',
  '--allow-net',
  '--allow-write=tools/reference-renders',
  'tools/behavioral-gate/gate.mjs',
  ...gateArgs,
];

const result = spawnSync('deno', denoArgs, {
  cwd: repoRoot,
  env: process.env,
  encoding: 'utf8',
  maxBuffer: 128 * 1024 * 1024,
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
  const filter = readFlagValue(gateArgs, '--filter');
  const status = {
    generatedAt: new Date().toISOString(),
    harness: 'behavioral-gate',
    verdict: 'HOST-BLOCKED',
    command: `deno ${denoArgs.join(' ')}`,
    filter: filter || null,
    icd: process.env.VK_ICD_FILENAMES ?? null,
    exitStatus: result.status,
    signal: result.signal,
    reason: {
      code: 'deno-wgpu-hal-gles-index-oob',
      location: 'wgpu-hal-28.0.0/src/gles/command.rs:771:21',
      message:
        'Deno native WebGPU panicked before behavioral-gate could classify the selected lanes.',
    },
    nextSteps: [
      'Use pt-webgpu-focused filters such as --filter gltf on the current WSL native-Deno lane.',
      'Run walkaround behavioral lanes in the browser/real-adapter validation lane.',
      'Re-run this wrapper after the Deno native WebGPU panic is fixed or the host path changes.',
    ],
  };
  writeFileSync(statusPath, `${JSON.stringify(status, null, 2)}\n`);
  console.error(`[behavioral-gate] HOST-BLOCKED status written to ${statusPath}`);
  process.exit(2);
}

process.exit(result.status ?? 1);

function readFlagValue(args, name) {
  const eq = args.find((a) => a.startsWith(`${name}=`));
  if (eq) return eq.slice(name.length + 1);
  const i = args.indexOf(name);
  return i >= 0 ? (args[i + 1] ?? '') : '';
}
