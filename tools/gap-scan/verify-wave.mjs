#!/usr/bin/env node
/**
 * Orchestrator verifies every task in a wave. Does NOT trust sub-agent self-reports.
 *
 *   node tools/gap-scan/verify-wave.mjs W001 --base-sha abc123
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { verifyTask } from './verify-task.mjs';
import { buildRemediationPrompt } from './agent-prompt-template.mjs';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const TASKS_JSONL = path.join(REPO_ROOT, 'plan/code-gap-tasks.jsonl');

function loadTasks() {
  const map = new Map();
  for (const line of fs.readFileSync(TASKS_JSONL, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    const t = JSON.parse(line);
    map.set(t.id, t);
  }
  return map;
}

function resolveBaseSha(flag) {
  if (flag) return flag;
  const statePath = path.join(REPO_ROOT, 'plan/.gap-orchestrator-state.json');
  if (fs.existsSync(statePath)) {
    const state = JSON.parse(fs.readFileSync(statePath, 'utf8'));
    if (state.waveBaseSha) return state.waveBaseSha;
  }
  return null;
}

function main() {
  const waveId = process.argv[2];
  const baseIdx = process.argv.indexOf('--base-sha');
  const baseSha = resolveBaseSha(baseIdx >= 0 ? process.argv[baseIdx + 1] : null);

  if (!waveId) {
    console.error('Usage: verify-wave.mjs <Wnnn> [--base-sha <sha>]');
    process.exit(1);
  }
  if (!baseSha) {
    console.error('Missing --base-sha or plan/.gap-orchestrator-state.json waveBaseSha');
    process.exit(1);
  }

  const waveDir = path.join(REPO_ROOT, 'plan/waves', waveId);
  const specs = JSON.parse(fs.readFileSync(path.join(waveDir, 'verify-spec.json'), 'utf8'));
  const tasks = loadTasks();

  const results = [];
  for (const spec of specs) {
    const task = tasks.get(spec.taskId);
    const fullSpec = { ...spec, baseSha };
    results.push(verifyTask(fullSpec, task));
  }

  const passed = results.filter((r) => r.passed);
  const failed = results.filter((r) => !r.passed);

  const report = {
    waveId,
    baseSha,
    verifiedAt: new Date().toISOString(),
    summary: { total: results.length, passed: passed.length, failed: failed.length },
    results,
  };

  const reportPath = path.join(waveDir, 'verify-report.json');
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2) + '\n');

  // Write remediation prompts for failures
  if (failed.length > 0) {
    const remDir = path.join(waveDir, 'remediation');
    fs.mkdirSync(remDir, { recursive: true });
    for (const f of failed) {
      const spec = specs.find((s) => s.taskId === f.taskId);
      const task = tasks.get(f.taskId);
      const prompt = buildRemediationPrompt({
        waveId,
        slot: spec.slot,
        task,
        failureReason: f.remediationHint,
      });
      fs.writeFileSync(path.join(remDir, `agent-${String(spec.slot).padStart(2, '0')}-${f.taskId}.md`), prompt);
    }
    fs.writeFileSync(
      path.join(remDir, 'DISPATCH.md'),
      [
        `# Remediation dispatch — ${waveId}`,
        '',
        `**${failed.length}** task(s) failed orchestrator verification.`,
        '',
        'Re-dispatch ONLY these agents with prompts in this folder:',
        '',
        ...failed.map((f) => {
          const spec = specs.find((s) => s.taskId === f.taskId);
          return `- agent-${String(spec.slot).padStart(2, '0')} → \`${f.taskId}\`: plan/waves/${waveId}/remediation/agent-${String(spec.slot).padStart(2, '0')}-${f.taskId}.md`;
        }),
        '',
        'Then re-run: `node tools/gap-scan/orchestrator-run.mjs --verify ' + waveId + '`',
        '',
      ].join('\n'),
    );
  }

  console.log(JSON.stringify(report.summary, null, 2));
  console.log(`Report: plan/waves/${waveId}/verify-report.json`);
  if (failed.length) {
    console.error(`FAILED: ${failed.map((f) => f.taskId).join(', ')}`);
    console.error(`Remediation prompts: plan/waves/${waveId}/remediation/`);
    process.exit(1);
  }
  console.log('All tasks passed orchestrator verification.');
}

main();
