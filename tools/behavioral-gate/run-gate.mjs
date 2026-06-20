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
const hostStatusPath = resolve(scriptDir, 'behavioral-gate-host-status.json');
const requestedStatusPath = process.env.VITRUM_BEHAVIORAL_GATE_STATUS_PATH
  ? resolve(repoRoot, process.env.VITRUM_BEHAVIORAL_GATE_STATUS_PATH)
  : '';

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
if (requestedStatusPath) {
  const passed = result.status === 0;
  const filter = readFlagValue(gateArgs, '--filter');
  const status = {
    generatedAt: new Date().toISOString(),
    harness: 'behavioral-gate',
    verdict: passed ? 'PASS' : 'FAIL',
    command: `npm run behavioral-gate -- ${gateArgs.join(' ')}`.trim(),
    filter: filter || null,
    icd: process.env.VK_ICD_FILENAMES ?? null,
    exitStatus: result.status,
    signal: result.signal,
    summary: parseSummary(result.stdout ?? ''),
    configs: parseConfigRows(result.stdout ?? ''),
  };
  writeFileSync(requestedStatusPath, `${JSON.stringify(status, null, 2)}\n`);
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
  writeFileSync(hostStatusPath, `${JSON.stringify(status, null, 2)}\n`);
  console.error(`[behavioral-gate] HOST-BLOCKED status written to ${hostStatusPath}`);
  process.exit(2);
}

process.exit(result.status ?? 1);

function readFlagValue(args, name) {
  const eq = args.find((a) => a.startsWith(`${name}=`));
  if (eq) return eq.slice(name.length + 1);
  const i = args.indexOf(name);
  return i >= 0 ? (args[i + 1] ?? '') : '';
}

function parseSummary(stdout) {
  const match = stdout.match(/=== summary: (\d+) configs total, (\d+) failures, (\d+) known-residuals ===/);
  if (!match) return null;
  return {
    totalConfigs: Number(match[1]),
    failures: Number(match[2]),
    knownResiduals: Number(match[3]),
  };
}

function parseConfigRows(stdout) {
  const rows = [];
  for (const line of stdout.split(/\r?\n/)) {
    const match = line.match(/^\s*(PASS|FAIL|KNOWN-RESIDUAL)\s+\|\s+([^|]+?)\s+\|\s+([^|]+?)\s+\|\s+([^|]+?)(?:\s+\|\s+(.*))?$/);
    if (!match) continue;
    const details = match[4].trim();
    const extras = (match[5] ?? '').split('|').map((part) => part.trim()).filter(Boolean);
    const mutation = extras.find((part) => part.startsWith('mutation=')) ?? '';
    const cwbvhParity = extras.find((part) => part.startsWith('cwbvhParity=')) ?? '';
    const cwbvhPerf = extras.find((part) => part.startsWith('cwbvhPerf=')) ?? '';
    const golden = extras.find((part) => part.startsWith('golden=')) ?? '';
    rows.push({
      verdict: match[1],
      label: match[2].trim(),
      rawStatus: match[3].trim(),
      tier: readToken(details, 'tier'),
      luminance: readNumberToken(details, 'lum'),
      gpuErrors: readNumberToken(details, 'gpuErrs'),
      nan: readBoolToken(details, 'nan'),
      mutation: mutation || null,
      mutationKind: readToken(mutation, 'mutation'),
      mutationMeanAbs: readNumberToken(mutation, 'meanAbs'),
      mutationMaxAbs: readNumberToken(mutation, 'maxAbs'),
      cwbvhParity: cwbvhParity || null,
      cwbvhParityKind: readToken(cwbvhParity, 'cwbvhParity'),
      cwbvhParityRmse: readNumberToken(cwbvhParity, 'rmse'),
      cwbvhParityMeanAbs: readNumberToken(cwbvhParity, 'meanAbs'),
      cwbvhParityMaxAbs: readNumberToken(cwbvhParity, 'maxAbs'),
      cwbvhParityThresholds: readThresholds(cwbvhParity),
      cwbvhPerf: cwbvhPerf || null,
      cwbvhPerfKind: readToken(cwbvhPerf, 'cwbvhPerf'),
      cwbvhBinaryRenderMs: readNumberToken(cwbvhPerf, 'binaryMs'),
      cwbvhRenderMs: readNumberToken(cwbvhPerf, 'cwbvhMs'),
      cwbvhRenderMsRatio: readNumberToken(cwbvhPerf, 'ratio'),
      cwbvhBinaryMemoryBytes: readNumberToken(cwbvhPerf, 'binaryMem'),
      cwbvhMemoryBytes: readNumberToken(cwbvhPerf, 'cwbvhMem'),
      cwbvhMemoryBytesDelta: readNumberToken(cwbvhPerf, 'memDelta'),
      cwbvhBinarySceneBytes: readNumberToken(cwbvhPerf, 'binaryScene'),
      cwbvhSceneBytes: readNumberToken(cwbvhPerf, 'cwbvhScene'),
      cwbvhSceneBytesDelta: readNumberToken(cwbvhPerf, 'sceneDelta'),
      golden: golden || null,
      goldenStatus: readToken(golden, 'golden'),
      goldenVariant: readToken(golden, 'variant'),
      rmse: readNumberToken(golden, 'rmse'),
      meanAbs: readNumberToken(golden, 'meanAbs'),
      maxAbs: readNumberToken(golden, 'maxAbs'),
      thresholds: readThresholds(golden),
      rawLine: line.trim(),
    });
  }
  return rows;
}

function readToken(text, key) {
  const match = text.match(new RegExp(`${key}=([^\\s]+)`));
  return match ? match[1] : null;
}

function readNumberToken(text, key) {
  const value = readToken(text, key);
  if (value == null) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function readBoolToken(text, key) {
  const value = readToken(text, key);
  if (value === 'true') return true;
  if (value === 'false') return false;
  return null;
}

function readThresholds(text) {
  const match = text.match(/<=\(([^,]+),([^,]+),([^)]+)\)/);
  if (!match) return null;
  const values = match.slice(1).map((v) => Number(v.trim()));
  if (values.some((v) => !Number.isFinite(v))) return null;
  return {
    maxRmse: values[0],
    maxMeanAbs: values[1],
    maxAbs: values[2],
  };
}
