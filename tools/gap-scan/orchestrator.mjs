#!/usr/bin/env node
/**
 * Orchestrator helper for parallel code-gap remediation.
 *
 *   node tools/gap-scan/orchestrator.mjs status
 *   node tools/gap-scan/orchestrator.mjs wave 0
 *   node tools/gap-scan/orchestrator.mjs claim WH-012 agent-3
 *   node tools/gap-scan/orchestrator.mjs integration W001
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const SCHEDULE = path.join(ROOT, 'plan/code-gap-parallel-schedule.json');
const PROGRESS = path.join(ROOT, 'plan/code-gap-task-progress.md');

function loadSchedule() {
  return JSON.parse(fs.readFileSync(SCHEDULE, 'utf8'));
}

function cmdStatus() {
  const s = loadSchedule();
  const cap = s.stats.maxAgentsPerWave ?? 25;
  console.log(`Waves: ${s.stats.waveCount} | Tasks: ${s.stats.totalTasks} | Agent cap: ${cap}/wave | Largest wave: ${s.stats.maxParallelism} tasks`);
  console.log(`Bootstrap W000: ${s.waves[0].taskCount} tasks`);
  console.log('');
  const full = s.waves.filter((w) => w.taskCount === cap).length;
  console.log(`Waves at cap (${cap}): ${full} | Single-task waves: ${s.waves.filter((w) => w.taskCount === 1).length}`);
  console.log('');
  for (const w of s.waves.slice(0, 15)) {
    const tag = w.bootstrap ? ' [BOOTSTRAP]' : w.taskCount === cap ? ' [FULL]' : '';
    console.log(`${w.waveId}${tag}: ${w.taskCount} tasks, lanes=[${w.lanes.slice(0, 5).join(', ')}${w.lanes.length > 5 ? ',…' : ''}]`);
  }
  if (s.waves.length > 15) console.log(`… +${s.waves.length - 15} waves`);
}

function cmdDispatch(waveIndex) {
  const s = loadSchedule();
  const cap = s.stats.maxAgentsPerWave ?? 25;
  const w = s.waves.find((x) => x.wave === Number(waveIndex) || x.waveId === `W${String(waveIndex).padStart(3, '0')}`);
  if (!w) {
    console.error('Wave not found');
    process.exit(1);
  }
  const slots = Math.min(w.taskCount, cap);
  console.log(`# Dispatch ${w.waveId}: spawn ${slots} agent(s) for ${w.taskCount} task(s)`);
  console.log('');
  for (let i = 0; i < slots; i++) {
    const tid = w.taskIds[i];
    console.log(`agent-${i + 1}: ${tid}`);
  }
  if (w.taskCount > cap) {
    console.error(`ERROR: wave has ${w.taskCount} tasks > cap ${cap} — regenerate schedule`);
    process.exit(1);
  }
}

function cmdWave(n) {
  const s = loadSchedule();
  const w = s.waves.find((x) => x.wave === Number(n) || x.waveId === `W${String(n).padStart(3, '0')}`);
  if (!w) {
    console.error('Wave not found');
    process.exit(1);
  }
  console.log(JSON.stringify(w, null, 2));
}

function cmdIntegration(waveId) {
  const s = loadSchedule();
  const w = s.waves.find((x) => x.waveId === waveId);
  const lanes = w?.lanes ?? [];
  console.log('# Integration after', waveId);
  console.log('cd /home/jsquire4/projects/vitrum && npm run typecheck');
  for (const lane of lanes) {
    if (lane === 'repo-root') continue;
    const pkg = lane === 'tools' ? 'tools' : `packages/${lane}`;
    console.log(`# ${lane}`);
    if (lane === 'tools') {
      console.log('node tools/behavioral-gate/gate.mjs --filter bootstrap  # if GPU');
    } else if (fs.existsSync(path.join(ROOT, pkg, 'package.json'))) {
      console.log(`cd /home/jsquire4/projects/vitrum/${pkg} && npx vitest run`);
    }
  }
}

const [,, sub, ...rest] = process.argv;
switch (sub) {
  case 'status':
    cmdStatus();
    break;
  case 'dispatch':
    cmdDispatch(rest[0] ?? '0');
    break;
  case 'wave':
    cmdWave(rest[0] ?? '0');
    break;
  case 'integration':
    cmdIntegration(rest[0] ?? 'W000');
    break;
  default:
    console.log(`Usage:
  orchestrator.mjs status
  orchestrator.mjs dispatch <n|Wnnn>   # print agent→task assignments (≤25)
  orchestrator.mjs wave <n|Wnnn>
  orchestrator.mjs integration <Wnnn>
`);
}
