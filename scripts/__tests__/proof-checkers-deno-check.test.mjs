import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');

const proofCheckers = [
  join(repoRoot, 'tools', 'behavioral-gate', 'check-dzn-status.mjs'),
  join(repoRoot, 'tools', 'behavioral-gate', 'check-cwbvh-parity-status.mjs'),
  join(repoRoot, 'tools', 'behavioral-gate', 'check-cwbvh-renderer-parity-status.mjs'),
  join(repoRoot, 'tools', 'radiometric-ab', 'check-results.mjs'),
  join(repoRoot, 'tools', 'renderer-fidelity-proof', 'check-proofs.mjs'),
  join(repoRoot, 'tools', 'learned-systems', 'check-status.mjs'),
];

test('proof checkers pass deno check', async () => {
  const result = await runDenoCheck(proofCheckers);
  assert.equal(
    result.status,
    0,
    `deno check failed\nstdout:\n${result.stdout}\nstderr:\n${result.stderr}`,
  );
});

function runDenoCheck(paths) {
  return new Promise((resolveResult) => {
    const child = spawn('deno', ['check', '--sloppy-imports', ...paths], {
      cwd: repoRoot,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    child.stdout.setEncoding('utf8');
    child.stderr.setEncoding('utf8');
    child.stdout.on('data', (chunk) => { stdout += chunk; });
    child.stderr.on('data', (chunk) => { stderr += chunk; });
    child.on('close', (status, signal) => {
      resolveResult({ status, signal, stdout, stderr });
    });
  });
}
