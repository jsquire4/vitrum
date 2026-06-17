/**
 * Orchestrator-side verification for a single task. Does NOT trust sub-agent reports.
 */
import { execSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');

/**
 * @param {string} cmd
 * @returns {{ ok: boolean, exitCode: number, output: string }}
 */
export function runCmd(cmd) {
  try {
    const output = execSync(cmd, {
      cwd: REPO_ROOT,
      encoding: 'utf8',
      stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...process.env, FORCE_COLOR: '0' },
    });
    return { ok: true, exitCode: 0, output: output.slice(0, 8000) };
  } catch (e) {
    const output = (e.stdout || '') + (e.stderr || '') + (e.message || '');
    return { ok: false, exitCode: e.status ?? 1, output: output.slice(0, 8000) };
  }
}

/**
 * @param {string} baseSha
 * @param {string[]} files
 */
export function gitDiffFiles(baseSha, files) {
  if (!baseSha || baseSha === '__WAVE_BASE_SHA__') {
    return { ok: false, changed: [], error: 'invalid baseSha' };
  }
  try {
    const changed = [];
    for (const f of files) {
      const rel = f.replace(/^\//, '');
      const out = execSync(`git diff --name-only ${baseSha} HEAD -- "${rel}"`, {
        cwd: REPO_ROOT,
        encoding: 'utf8',
      }).trim();
      if (out) changed.push(...out.split('\n').filter(Boolean));
    }
    return { ok: true, changed: [...new Set(changed)] };
  } catch (e) {
    return { ok: false, changed: [], error: String(e.message) };
  }
}

/**
 * @param {object} spec — from verify-spec.json entry with real baseSha
 * @param {object} task — full task from jsonl
 * @returns {object}
 */
export function verifyTask(spec, task) {
  const failures = [];
  const checks = [];

  // 1. Files exist
  for (const f of spec.files) {
    const abs = path.join(REPO_ROOT, f);
    if (!fs.existsSync(abs)) {
      failures.push(`Declared file missing on disk: ${f}`);
    }
  }
  checks.push({ name: 'filesExist', ok: failures.length === 0 });

  // 2. Git diff (orchestrator truth — not agent claims)
  const diff = gitDiffFiles(spec.baseSha, spec.files);
  if (!diff.ok) {
    failures.push(`git diff failed: ${diff.error}`);
  } else if (spec.checks?.requireGitDiff && diff.changed.length === 0) {
    failures.push(
      `No git diff vs wave base for task files: ${spec.files.join(', ')}. Agent did not land edits.`,
    );
  }
  checks.push({ name: 'gitDiff', ok: diff.ok && !(spec.checks?.requireGitDiff && diff.changed.length === 0), changed: diff.changed });

  // 3. Run tests (orchestrator runs — not agent claims)
  const testResults = [];
  if (spec.checks?.requireTestsPass && spec.tests?.length) {
    for (const cmd of spec.tests) {
      const r = runCmd(cmd);
      testResults.push({ cmd, ...r });
      if (!r.ok) failures.push(`Test failed (exit ${r.exitCode}): ${cmd}`);
    }
  }
  checks.push({
    name: 'tests',
    ok: testResults.every((t) => t.ok) || !spec.tests?.length,
    results: testResults.map((t) => ({ cmd: t.cmd, ok: t.ok, exitCode: t.exitCode })),
  });

  return {
    taskId: spec.taskId,
    passed: failures.length === 0,
    failures,
    checks,
    remediationHint: failures.join('\n'),
  };
}
