#!/usr/bin/env node
/**
 * Generates per-wave agent prompts + manifests for orchestrator dispatch.
 * Run: node tools/gap-scan/generate-wave-manifests.mjs
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { MAX_AGENTS_PER_WAVE } from './parallel-schedule.mjs';
import { buildAgentPrompt, buildVerifySpec, ROOT } from './agent-prompt-template.mjs';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const SCHEDULE = path.join(REPO_ROOT, 'plan/code-gap-parallel-schedule.json');
const TASKS_JSONL = path.join(REPO_ROOT, 'plan/code-gap-tasks.jsonl');
const WAVES_DIR = path.join(REPO_ROOT, 'plan/waves');

function loadTasks() {
  /** @type {Map<string, object>} */
  const map = new Map();
  for (const line of fs.readFileSync(TASKS_JSONL, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    const t = JSON.parse(line);
    map.set(t.id, t);
  }
  return map;
}

function smokeSpecForWave(wave) {
  const lanes = wave.lanes.filter((l) => l !== 'repo-root');
  return {
    required: ['typecheck'],
    laneVitest: lanes.map((lane) => ({
      lane,
      cmd:
        lane === 'tools'
          ? null
          : `cd ${ROOT}/packages/${lane} && npx vitest run`,
    })),
    bootstrap: !!wave.bootstrap,
    behavioralGate:
      wave.bootstrap || wave.waveId === 'W001'
        ? { optional: true, cmd: `cd ${ROOT} && npm run behavioral-gate` }
        : null,
  };
}

function commitMessage(wave, taskIds) {
  const preview = taskIds.slice(0, 8).join(', ');
  const more = taskIds.length > 8 ? ` +${taskIds.length - 8} more` : '';
  const tag = wave.bootstrap ? 'bootstrap' : 'wave';
  return `chore(gap-remediation): ${wave.waveId} ${tag} — ${taskIds.length} tasks (${preview}${more})`;
}

function generate() {
  const schedule = JSON.parse(fs.readFileSync(SCHEDULE, 'utf8'));
  const tasks = loadTasks();

  fs.mkdirSync(WAVES_DIR, { recursive: true });

  /** @type {object[]} */
  const index = [];

  for (const wave of schedule.waves) {
    const waveDir = path.join(WAVES_DIR, wave.waveId);
    const agentsDir = path.join(waveDir, 'agents');
    fs.mkdirSync(agentsDir, { recursive: true });

    /** @type {object[]} */
    const agents = [];
    /** @type {object[]} */
    const verifySpecs = [];

    wave.taskIds.forEach((taskId, i) => {
      const task = tasks.get(taskId);
      if (!task) throw new Error(`Missing task ${taskId} for ${wave.waveId}`);
      const slot = i + 1;
      const prompt = buildAgentPrompt({ waveId: wave.waveId, slot, task });
      const promptFile = `agents/agent-${String(slot).padStart(2, '0')}.md`;
      fs.writeFileSync(path.join(waveDir, promptFile), prompt);

      agents.push({
        slot,
        agentId: `agent-${String(slot).padStart(2, '0')}`,
        taskId,
        lane: task.lane,
        disposition: task.disposition,
        promptFile,
        promptAbsPath: path.join(waveDir, promptFile),
      });

      verifySpecs.push({
        slot,
        taskId,
        ...buildVerifySpec(task, '__WAVE_BASE_SHA__'),
      });
    });

    const manifest = {
      waveId: wave.waveId,
      waveIndex: wave.wave,
      bootstrap: !!wave.bootstrap,
      agentCap: MAX_AGENTS_PER_WAVE,
      taskCount: wave.taskCount,
      lanes: wave.lanes,
      agents,
      dispatch: {
        instruction: `Spawn ${wave.taskCount} sub-agent(s). Each agent reads its prompt file and executes exactly one task.`,
        maxConcurrent: Math.min(wave.taskCount, MAX_AGENTS_PER_WAVE),
      },
      phases: ['build', 'verify', 'remediate', 'smoke', 'commit'],
      smoke: smokeSpecForWave(wave),
      commit: {
        message: commitMessage(wave, wave.taskIds),
        files: 'all staged changes from verified tasks',
      },
    };

    fs.writeFileSync(path.join(waveDir, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
    fs.writeFileSync(path.join(waveDir, 'verify-spec.json'), JSON.stringify(verifySpecs, null, 2) + '\n');

    // Orchestrator dispatch script for parent agent
    const dispatchMd = [
      `# Dispatch ${wave.waveId}`,
      '',
      `**Tasks:** ${wave.taskCount} | **Max agents:** ${Math.min(wave.taskCount, MAX_AGENTS_PER_WAVE)}`,
      '',
      '## Sub-agent prompts (copy each to a Task subagent)',
      '',
      ...agents.map(
        (a) =>
          `### ${a.agentId} → \`${a.taskId}\` (${a.lane})\n\nRead and execute: \`plan/waves/${wave.waveId}/${a.promptFile}\``,
      ),
      '',
      '## After all agents return',
      '',
      '```bash',
      `node tools/gap-scan/orchestrator-run.mjs --verify ${wave.waveId}`,
      `node tools/gap-scan/orchestrator-run.mjs --smoke ${wave.waveId}`,
      `node tools/gap-scan/orchestrator-run.mjs --commit ${wave.waveId}`,
      `node tools/gap-scan/orchestrator-run.mjs --advance`,
      '```',
      '',
    ].join('\n');
    fs.writeFileSync(path.join(waveDir, 'DISPATCH.md'), dispatchMd);

    index.push({
      waveId: wave.waveId,
      waveIndex: wave.wave,
      taskCount: wave.taskCount,
      dir: `plan/waves/${wave.waveId}`,
    });
  }

  fs.writeFileSync(path.join(WAVES_DIR, 'index.json'), JSON.stringify(index, null, 2) + '\n');
  console.log(`Generated ${index.length} wave manifests under plan/waves/`);
}

generate();
