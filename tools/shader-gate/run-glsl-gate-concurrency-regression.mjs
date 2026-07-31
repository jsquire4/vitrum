#!/usr/bin/env -S deno run --allow-run
/**
 * Regression exercise for process-isolated GLSL gate scratch files.
 *
 * Runs the production gate and its failure-detection self-test at the same time.
 * Before each invocation received a private temporary directory, these two
 * processes deterministically overwrote and deleted one another's shader files.
 */

const gateUrl = new URL('./glslGate.mjs', import.meta.url).href;
const commonArgs = [
  'run',
  '--sloppy-imports',
  '--allow-read',
  '--allow-env',
  '--allow-run',
  '--allow-write=/tmp',
  gateUrl,
];
const decoder = new TextDecoder();

async function runGate(label, extraArgs) {
  const command = new Deno.Command(Deno.execPath(), {
    args: [...commonArgs, ...extraArgs],
    stdout: 'piped',
    stderr: 'piped',
  });
  const result = await command.output();
  return {
    label,
    ...result,
    stdoutText: decoder.decode(result.stdout),
    stderrText: decoder.decode(result.stderr),
  };
}

const startedAt = Date.now();
const results = await Promise.all([
  runGate('production', []),
  runGate('self-test', ['--self-test']),
]);

let failed = false;
for (const result of results) {
  if (result.success) {
    console.log(`[glsl-gate-concurrency] ${result.label} PASS (exit ${result.code})`);
    continue;
  }
  failed = true;
  console.error(`[glsl-gate-concurrency] ${result.label} FAIL (exit ${result.code})`);
  if (result.stdoutText.trim()) console.error(result.stdoutText.trimEnd());
  if (result.stderrText.trim()) console.error(result.stderrText.trimEnd());
}

console.log(
  `[glsl-gate-concurrency] completed both overlapping gates in ${Date.now() - startedAt} ms`,
);
Deno.exit(failed ? 1 : 0);
