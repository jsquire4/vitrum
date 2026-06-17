#!/usr/bin/env node
/**
 * Wave-level smoke tests — orchestrator runs between verify and commit.
 *
 *   node tools/gap-scan/smoke-wave.mjs W001
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { runCmd } from './verify-task.mjs';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');

function main() {
  const waveId = process.argv[2];
  if (!waveId) {
    console.error('Usage: smoke-wave.mjs <Wnnn>');
    process.exit(1);
  }

  const manifest = JSON.parse(
    fs.readFileSync(path.join(REPO_ROOT, 'plan/waves', waveId, 'manifest.json'), 'utf8'),
  );

  const failures = [];
  const results = [];

  // Always typecheck
  const tc = runCmd('cd /home/jsquire4/projects/vitrum && npm run typecheck');
  results.push({ name: 'typecheck', ...tc });
  if (!tc.ok) failures.push('typecheck failed');

  // Lane vitest (dedupe lanes)
  const lanes = [...new Set(manifest.lanes.filter((l) => l !== 'repo-root' && l !== 'tools'))];
  for (const lane of lanes) {
    const pkg = path.join(REPO_ROOT, 'packages', lane, 'package.json');
    if (!fs.existsSync(pkg)) continue;
    const pkgJson = JSON.parse(fs.readFileSync(pkg, 'utf8'));
    if (!pkgJson.scripts?.test) continue;
    const cmd = `cd /home/jsquire4/projects/vitrum/packages/${lane} && npx vitest run`;
    const r = runCmd(cmd);
    results.push({ name: `vitest:${lane}`, cmd, ok: r.ok, exitCode: r.exitCode });
    if (!r.ok) failures.push(`vitest failed: ${lane}`);
  }

  // Bootstrap wave: run task-level tests from manifest agents
  if (manifest.bootstrap) {
    for (const a of manifest.agents) {
      const spec = JSON.parse(
        fs.readFileSync(path.join(REPO_ROOT, 'plan/waves', waveId, 'verify-spec.json'), 'utf8'),
      ).find((s) => s.taskId === a.taskId);
      if (!spec?.tests) continue;
      for (const cmd of spec.tests) {
        const r = runCmd(cmd);
        results.push({ name: `task-test:${a.taskId}`, cmd, ok: r.ok });
        if (!r.ok) failures.push(`bootstrap task test failed: ${a.taskId}`);
      }
    }
  }

  const report = {
    waveId,
    smokedAt: new Date().toISOString(),
    passed: failures.length === 0,
    failures,
    results: results.map((r) => ({ name: r.name, ok: r.ok, exitCode: r.exitCode })),
  };

  const out = path.join(REPO_ROOT, 'plan/waves', waveId, 'smoke-report.json');
  fs.writeFileSync(out, JSON.stringify(report, null, 2) + '\n');

  console.log(JSON.stringify({ passed: report.passed, failures: report.failures }, null, 2));
  console.log(`Report: plan/waves/${waveId}/smoke-report.json`);

  if (!report.passed) process.exit(1);
}

main();
