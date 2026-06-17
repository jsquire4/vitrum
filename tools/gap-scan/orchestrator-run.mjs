#!/usr/bin/env node
/**
 * Stateful orchestrator loop for gap remediation waves.
 *
 * Lifecycle per wave: begin → build → verify → [remediate → verify]* → smoke → commit → advance
 *
 * Commands:
 *   node tools/gap-scan/orchestrator-run.mjs status
 *   node tools/gap-scan/orchestrator-run.mjs begin          # snapshot base sha, open current wave
 *   node tools/gap-scan/orchestrator-run.mjs dispatch       # print what to run (parent agent)
 *   node tools/gap-scan/orchestrator-run.mjs --verify       # verify current wave
 *   node tools/gap-scan/orchestrator-run.mjs --smoke        # smoke current wave
 *   node tools/gap-scan/orchestrator-run.mjs --commit       # git commit current wave (requires --yes)
 *   node tools/gap-scan/orchestrator-run.mjs advance        # move to next wave after commit
 *   node tools/gap-scan/orchestrator-run.mjs resume-md      # regenerate ORCHESTRATOR_RESUME.md
 */
import { execSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const STATE_PATH = path.join(REPO_ROOT, 'plan/.gap-orchestrator-state.json');
const SCHEDULE_PATH = path.join(REPO_ROOT, 'plan/code-gap-parallel-schedule.json');
const RESUME_PATH = path.join(REPO_ROOT, 'plan/ORCHESTRATOR_RESUME.md');

const PHASES = ['ready', 'building', 'verifying', 'remediating', 'smoking', 'committing', 'done'];

function loadSchedule() {
  return JSON.parse(fs.readFileSync(SCHEDULE_PATH, 'utf8'));
}

function defaultState(schedule) {
  return {
    version: 1,
    startedAt: null,
    currentWaveIndex: 0,
    currentWaveId: schedule.waves[0]?.waveId ?? 'W000',
    phase: 'ready',
    waveBaseSha: null,
    remediationRound: 0,
    completedWaves: [],
    lastCommitSha: null,
    totalWaves: schedule.waves.length,
  };
}

function loadState() {
  const schedule = loadSchedule();
  if (!fs.existsSync(STATE_PATH)) {
    return defaultState(schedule);
  }
  const state = JSON.parse(fs.readFileSync(STATE_PATH, 'utf8'));
  state.totalWaves = schedule.waves.length;
  return state;
}

function saveState(state) {
  fs.writeFileSync(STATE_PATH, JSON.stringify(state, null, 2) + '\n');
  writeResumeMd(state);
}

function gitSha() {
  return execSync('git rev-parse HEAD', { cwd: REPO_ROOT, encoding: 'utf8' }).trim();
}

function waveById(schedule, waveId) {
  return schedule.waves.find((w) => w.waveId === waveId);
}

function writeResumeMd(state) {
  const schedule = loadSchedule();
  const wave = waveById(schedule, state.currentWaveId);
  const waveDir = `plan/waves/${state.currentWaveId}`;

  const md = [
    '# ORCHESTRATOR RESUME — gap remediation',
    '',
    '> **Parent orchestrator agent:** read this file at the start of every session. Do not stop until `phase: done`.',
    '',
    '## State snapshot',
    '',
    '```json',
    JSON.stringify(
      {
        currentWaveId: state.currentWaveId,
        currentWaveIndex: state.currentWaveIndex,
        phase: state.phase,
        remediationRound: state.remediationRound,
        completedWaves: state.completedWaves.length,
        totalWaves: state.totalWaves,
        waveBaseSha: state.waveBaseSha,
      },
      null,
      2,
    ),
    '```',
    '',
    '## What to do NOW',
    '',
  ];

  switch (state.phase) {
    case 'ready':
      md.push(
        `1. Run: \`node tools/gap-scan/orchestrator-run.mjs begin\``,
        `2. Run: \`node tools/gap-scan/orchestrator-run.mjs dispatch\``,
        `3. Launch **${wave?.taskCount ?? '?'}** sub-agents (max 25) using prompts in \`${waveDir}/agents/\``,
        `4. When all return: \`node tools/gap-scan/orchestrator-run.mjs --verify\``,
      );
      break;
    case 'building':
      md.push(
        `1. Sub-agents are working on **${state.currentWaveId}** (${wave?.taskCount} tasks).`,
        `2. When all return: \`node tools/gap-scan/orchestrator-run.mjs --verify\``,
        `3. **Do not trust** agent completion claims — verification is orchestrator-only.`,
      );
      break;
    case 'verifying':
      md.push(`1. Run: \`node tools/gap-scan/orchestrator-run.mjs --verify\``);
      break;
    case 'remediating':
      md.push(
        `1. Re-dispatch failed agents from \`${waveDir}/remediation/DISPATCH.md\``,
        `2. Re-run: \`node tools/gap-scan/orchestrator-run.mjs --verify\``,
      );
      break;
    case 'smoking':
      md.push(`1. Run: \`node tools/gap-scan/orchestrator-run.mjs --smoke\``);
      break;
    case 'committing':
      md.push(
        `1. Run: \`node tools/gap-scan/orchestrator-run.mjs --commit --yes\``,
        `2. Run: \`node tools/gap-scan/orchestrator-run.mjs advance\``,
        `3. If more waves remain, loop back to \`begin\` + \`dispatch\` for next wave.`,
      );
      break;
    case 'done':
      md.push('**All waves complete.** Remediation build finished.');
      break;
    default:
      md.push('Unknown phase — run `orchestrator-run.mjs status`');
  }

  md.push(
    '',
    '## Wave loop (every wave)',
    '',
    '```',
    'begin → dispatch N sub-agents → verify → [remediate → verify]* → smoke → commit → advance',
    '```',
    '',
    `## Progress: ${state.completedWaves.length} / ${state.totalWaves} waves committed`,
    '',
    state.completedWaves.length
      ? `Last committed: ${state.completedWaves[state.completedWaves.length - 1]}`
      : 'No waves committed yet.',
    '',
  );

  fs.writeFileSync(RESUME_PATH, md.join('\n'));
}

function cmdStatus() {
  const state = loadState();
  const schedule = loadSchedule();
  const wave = waveById(schedule, state.currentWaveId);
  console.log(JSON.stringify({ ...state, currentTaskCount: wave?.taskCount }, null, 2));
  console.log(`Resume: plan/ORCHESTRATOR_RESUME.md`);
}

function cmdBegin() {
  const state = loadState();
  if (state.phase === 'done') {
    console.log('All waves complete.');
    return;
  }
  state.waveBaseSha = gitSha();
  state.phase = 'building';
  state.remediationRound = 0;
  if (!state.startedAt) state.startedAt = new Date().toISOString();
  saveState(state);
  console.log(`Began ${state.currentWaveId} at base ${state.waveBaseSha}`);
  console.log(`Dispatch: plan/waves/${state.currentWaveId}/DISPATCH.md`);
}

function cmdDispatch() {
  const state = loadState();
  const dispatchPath = path.join(REPO_ROOT, 'plan/waves', state.currentWaveId, 'DISPATCH.md');
  if (state.phase === 'remediating') {
    const rem = path.join(REPO_ROOT, 'plan/waves', state.currentWaveId, 'remediation/DISPATCH.md');
    if (fs.existsSync(rem)) {
      console.log(fs.readFileSync(rem, 'utf8'));
      return;
    }
  }
  if (!fs.existsSync(dispatchPath)) {
    console.error(`Missing ${dispatchPath} — run generate-wave-manifests.mjs`);
    process.exit(1);
  }
  console.log(fs.readFileSync(dispatchPath, 'utf8'));
}

function runScript(script, waveId) {
  execSync(`node tools/gap-scan/${script} ${waveId}`, { cwd: REPO_ROOT, stdio: 'inherit' });
}

function cmdVerify() {
  const state = loadState();
  state.phase = 'verifying';
  saveState(state);
  try {
    runScript('verify-wave.mjs', state.currentWaveId);
    state.phase = 'smoking';
    saveState(state);
    console.log('Verify PASSED → run --smoke');
  } catch {
    state.phase = 'remediating';
    state.remediationRound += 1;
    saveState(state);
    console.error('Verify FAILED → dispatch remediation agents, then --verify again');
    process.exit(1);
  }
}

function cmdSmoke() {
  const state = loadState();
  state.phase = 'smoking';
  saveState(state);
  try {
    runScript('smoke-wave.mjs', state.currentWaveId);
    state.phase = 'committing';
    saveState(state);
    console.log('Smoke PASSED → run --commit --yes');
  } catch {
    console.error('Smoke FAILED — fix before commit');
    process.exit(1);
  }
}

function cmdCommit(yes) {
  const state = loadState();
  if (!yes) {
    console.error('Pass --yes to commit. Orchestrator commits one wave at a time for rollback.');
    process.exit(1);
  }
  const manifest = JSON.parse(
    fs.readFileSync(path.join(REPO_ROOT, 'plan/waves', state.currentWaveId, 'manifest.json'), 'utf8'),
  );
  const msg = manifest.commit.message;
  execSync('git add -A', { cwd: REPO_ROOT, stdio: 'inherit' });
  const status = execSync('git status --porcelain', { cwd: REPO_ROOT, encoding: 'utf8' });
  if (!status.trim()) {
    console.error('Nothing to commit — verification may have failed silently');
    process.exit(1);
  }
  execSync('git commit -F -', {
    cwd: REPO_ROOT,
    stdio: ['pipe', 'inherit', 'inherit'],
    input: msg,
  });
  state.lastCommitSha = gitSha();
  console.log(`Committed ${state.currentWaveId} at ${state.lastCommitSha}`);
}

function cmdAdvance() {
  const state = loadState();
  const schedule = loadSchedule();
  state.completedWaves.push(state.currentWaveId);
  const nextIndex = state.currentWaveIndex + 1;
  if (nextIndex >= schedule.waves.length) {
    state.phase = 'done';
    state.currentWaveIndex = nextIndex;
    saveState(state);
    console.log('ALL WAVES COMPLETE');
    return;
  }
  state.currentWaveIndex = nextIndex;
  state.currentWaveId = schedule.waves[nextIndex].waveId;
  state.phase = 'ready';
  state.waveBaseSha = null;
  state.remediationRound = 0;
  saveState(state);
  console.log(`Advanced to ${state.currentWaveId} (${nextIndex + 1}/${schedule.waves.length})`);
  console.log('Next: orchestrator-run.mjs begin && dispatch');
}

// CLI
const args = process.argv.slice(2);
if (args.includes('status')) cmdStatus();
else if (args.includes('begin')) cmdBegin();
else if (args.includes('dispatch')) cmdDispatch();
else if (args.includes('--verify')) cmdVerify();
else if (args.includes('--smoke')) cmdSmoke();
else if (args.includes('--commit')) cmdCommit(args.includes('--yes'));
else if (args.includes('advance')) cmdAdvance();
else if (args.includes('resume-md')) writeResumeMd(loadState());
else {
  console.log(`Gap remediation orchestrator

  node tools/gap-scan/orchestrator-run.mjs status
  node tools/gap-scan/orchestrator-run.mjs begin
  node tools/gap-scan/orchestrator-run.mjs dispatch
  node tools/gap-scan/orchestrator-run.mjs --verify
  node tools/gap-scan/orchestrator-run.mjs --smoke
  node tools/gap-scan/orchestrator-run.mjs --commit --yes
  node tools/gap-scan/orchestrator-run.mjs advance

Parent agent: read plan/ORCHESTRATOR_AGENT.md
`);
}
