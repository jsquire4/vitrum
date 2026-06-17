#!/usr/bin/env node
/**
 * Generates plan/code-gap-implementation-plan.md from task-registry.mjs
 * Run: node tools/gap-scan/generate-implementation-plan.mjs
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { ALL_TASKS, PHASE_0 } from './task-registry.mjs';
import { RT100_TASKS } from './road-to-100-tasks.mjs';
import {
  computeParallelSchedule,
  computeSuperBatches,
  renderParallelMarkdown,
  SKIP_TASK_IDS,
  MAX_AGENTS_PER_WAVE,
} from './parallel-schedule.mjs';
import { listDeferredValidationTasks } from './validation-deferral.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const OUT = path.join(ROOT, 'plan/code-gap-implementation-plan.md');
const JSONL = path.join(ROOT, 'plan/code-gap-tasks.jsonl');
const PROGRESS = path.join(ROOT, 'plan/code-gap-task-progress.md');
const PARALLEL_JSON = path.join(ROOT, 'plan/code-gap-parallel-schedule.json');
const PARALLEL_MD = path.join(ROOT, 'plan/code-gap-parallel-schedule.md');
const LANES_MD = path.join(ROOT, 'plan/code-gap-lane-workers.md');

const schedule = computeParallelSchedule(ALL_TASKS);
const batches = computeSuperBatches(schedule.waves);
const deferredValidation = listDeferredValidationTasks(ALL_TASKS);

/** @type {Map<string, object>} */
const enrichedById = new Map(schedule.tasks.map((t) => [t.id, t]));

function renderTask(t) {
  const e = enrichedById.get(t.id);
  const lines = [];
  lines.push(`## ${t.id}`);
  lines.push('');
  lines.push('| Field | Value |');
  lines.push('|-------|-------|');
  lines.push(`| Phase | ${t.phase} |`);
  lines.push(`| Priority | ${t.priority} |`);
  lines.push(`| Disposition | ${t.disposition} |`);
  lines.push(`| Lane | ${e?.lane ?? '—'} |`);
  lines.push(`| Wave | ${e ? `${e.waveId} (#${e.wave})` : '—'} |`);
  lines.push(`| Parallel wave | ${e?.parallelSafe ? 'yes' : 'no'} |`);
  lines.push(`| Agent cap | ${MAX_AGENTS_PER_WAVE} |`);
  if (SKIP_TASK_IDS.has(t.id)) {
    lines.push(`| **SKIP** | Duplicate — see Phase-0 alias |`);
  }
  lines.push(`| Depends on | ${(e?.depends ?? t.depends)?.length ? (e?.depends ?? t.depends).join(', ') : '(none)'} |`);
  lines.push(`| Blocks | ${t.blocks?.length ? t.blocks.join(', ') : '(none)'} |`);
  lines.push('');
  lines.push('### Problem');
  lines.push(t.problem);
  lines.push('');
  lines.push('### Files to modify (exact paths)');
  t.files.forEach((f, i) => lines.push(`${i + 1}. \`${f}\``));
  lines.push('');
  lines.push('### Steps (execute in order — do not skip, do not improvise)');
  t.steps.forEach((s, i) => lines.push(`${i + 1}. ${s}`));
  lines.push('');
  lines.push('### Tests (run exactly in this order)');
  lines.push('```bash');
  t.tests.forEach((x) => lines.push(x));
  lines.push('```');
  lines.push('');
  lines.push('### Done when (ALL boxes required before next task)');
  t.done.forEach((d) => lines.push(`- [ ] ${d}`));
  lines.push('');
  lines.push('---');
  lines.push('');
  return lines.join('\n');
}

const counts = ALL_TASKS.reduce(
  (acc, t) => {
    acc.phases[t.phase] = (acc.phases[t.phase] || 0) + 1;
    return acc;
  },
  { phases: {} },
);

const PROTOCOL = `# Code Gap Implementation Plan — vitrum

> **Parallel-first execution.** See \`plan/code-gap-parallel-schedule.md\` for wave table.  
> **Orchestrator:** \`plan/ORCHESTRATOR_AGENT.md\` + \`node tools/gap-scan/orchestrator-run.mjs\`  
> **Per-wave prompts:** \`plan/waves/Wnnn/agents/agent-NN.md\` (generated)  
> **Machine-readable:** \`plan/code-gap-tasks.jsonl\` + \`plan/code-gap-parallel-schedule.json\`

> **Worktree:** \`/home/jsquire4/projects/vitrum-gap-remediation\` on branch \`feat/gap-remediation\`  
> **Validation:** ${deferredValidation.length} GPU/render tasks **deferred** — \`plan/VALIDATION-DEFERRED.md\`  
> **Active scheduled tasks:** ${schedule.stats.totalTasks} (after dedup + deferral)  
**Parallel waves:** ${schedule.stats.waveCount} · **Agent cap:** ${MAX_AGENTS_PER_WAVE} per wave  
**Generated:** ${new Date().toISOString().slice(0, 10)}

---

## 1. Parallel execution protocol (MANDATORY)

### 1.1 Orchestrator model
- Work proceeds in **waves** (\`W000\`, \`W001\`, …). Each wave has **at most ${MAX_AGENTS_PER_WAVE} tasks** — dispatch **≤${MAX_AGENTS_PER_WAVE} agents**, one task per agent.
- Tasks within a wave are **file-disjoint** — safe to run concurrently up to the cap.
- **Wave gate:** after each wave, run integration (§1.5). Do not start wave N+1 until wave N is done + integrated.
- **Super-batches** (\`B00\`, …): optional deeper integration every ~4 waves (see parallel-schedule.md).

### 1.2 Agent pool sizing
| Setting | Value |
|---------|------:|
| **Max concurrent agents** | **${MAX_AGENTS_PER_WAVE}** |
| Bootstrap W000 | 8 agents (all P0) |
| Typical wave | up to ${MAX_AGENTS_PER_WAVE} agents |
| Mutex waves | 1 agent (promiseLedger, materialAtlas, etc.) |

### 1.3 Agent assignment

| Model | When | Rule |
|-------|------|------|
| **Wave pool (recommended)** | Fixed pool of ≤${MAX_AGENTS_PER_WAVE} agents | Orchestrator publishes current \`waveId\`; each agent claims one unclaimed \`taskId\` from that wave only. |
| **Lane-pinned** | Long-running workers | Agent owns a **lane**; claims only tasks in current wave matching its lane. Still ≤${MAX_AGENTS_PER_WAVE} total active. |

### 1.4 Claiming work
1. Read \`plan/code-gap-parallel-schedule.json\` → \`waves[current]\`.
2. Pick an unclaimed \`taskId\` from that wave whose \`lane\` you own (or any if wave pool).
3. Record claim in \`plan/code-gap-lane-workers.md\` → \`claims\` table: \`taskId | agent | started | status\`.
4. **Hard rule:** do not edit any file not listed in your task's **Files to modify**.
5. **Hard rule:** if two tasks share a file, they are in different waves — never start early.

### 1.5 Skipped duplicates
Do not execute: ${[...SKIP_TASK_IDS].map((id) => `\`${id}\``).join(', ')}. Phase-0 aliases cover these.

### 1.6 Integration commands (run after EVERY wave)
\`\`\`bash
cd /home/jsquire4/projects/vitrum && npm run typecheck
# Plus package tests for lanes touched in the wave (orchestrator lists lanes in wave JSON)
\`\`\`

### 1.7 Per-task loop (within your claim)
1. Read **Problem**, **Files**, **Steps**.
2. Execute steps only — no drive-by refactors.
3. Run task **Tests**.
4. Mark done in \`plan/code-gap-task-progress.md\`.
5. **Commits:** orchestrator commits each wave (\`--commit --yes\`). Workers do not commit. **Never push** unless user explicitly requests.

### 1.8 Code-first policy (validation deferred)

GPU A/B, golden PNG, reference-render capture, behavioral-gate **render** proof, and
wsl-gpu harness tasks are **excluded from the schedule**. See \`plan/VALIDATION-DEFERRED.md\`
(${deferredValidation.length} tasks). Priority is **landing code**; validation sprint follows merge.

### 1.9 Disposition codes

| Code | Meaning | Required outcome |
|------|---------|------------------|
| \`BUG\` | Incorrect runtime behavior | Fix + regression test |
| \`IMP\` | Missing feature / native path | Implement + test |
| \`DOC\` | Contract/JSDoc/ledger text wrong | Text fix only |
| \`TEST\` | Missing test or gate | Add test/gate only |
| \`ACC\` | Permanent unsupported | EngineWarning + ledger; do not implement |
| \`RT\` | Route to other backend | gltf planner only |
| \`TOG\` | Fidelity toggle | Default preserves current behavior |
| \`VERIFY\` | Unit-test / code-read proof | Close or file follow-up (render VERIFY deferred) |
| \`SKIP\` | Deferred validation or duplicate | No agent dispatch |
| \`DECIDE\` | Architecture/product call | Record decision; align ledger |

### 1.10 Road-to-100 overlay (phases 8–12)

See \`plan/code-gap-road-to-100-crosswalk.md\` for mapping from \`plan/road-to-100.md\`
buckets A–D, Phases 2–5, and F1–F6 to \`RT100-*\` task IDs.

- **Phase 8–11 validation/decision tasks:** **SKIP** (deferred) in code-first mode
- **Phase 9:** Implementation tails (\`RT100-ADJ-*\`, \`RT100-WA-*\`, \`RT100-PT*\`) — **active**
- **Phase 10:** \`RT100-5D-DOC\` only (doc sync); render proof skipped
- **Phase 12:** SOTA perf IMP (\`RT100-LD-SAMPLING-01\`, \`RT100-WBVH-01\`, F3)

Generic \`MAT-WH-*\` tasks cover per-field promotion; \`RT100-WA-3D/3E\` are the
authoritative walkaround atlas epics per road-to-100 Phase 3 scope decision.

### 1.11 Mutex hotspots (serialize — never parallelize)
| File / area | Max concurrent editors |
|-------------|------------------------|
| \`packages/core/src/engine/promiseLedger.ts\` | **1** globally |
| \`packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts\` | **1** |
| \`packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts\` | **1** |
| \`packages/pt-webgpu/src/inverse/inverseSession.ts\` | **1** |
| \`packages/walkaround-hybrid/src/HybridEngine.ts\` | **1** |

### 1.12 Bootstrap wave W000 (8 agents, mandatory first)
All 8 \`P0-*\` tasks run in parallel — **no other work until W000 integration passes.**

${PHASE_0.map((t) => `- \`${t.id}\` lane \`${enrichedById.get(t.id)?.lane ?? '?'}\``).join('\n')}

### 1.13 Dispatch loop (W001 onward)
\`\`\`
repeat until all waves done:
  wave = read plan/code-gap-parallel-schedule.json → waves[current]
  spawn min(wave.taskCount, ${MAX_AGENTS_PER_WAVE}) agents
  each agent: claim taskId → execute → mark done
  integration: npm run typecheck + lane tests
  current += 1
\`\`\`
Typical wave size: **${MAX_AGENTS_PER_WAVE} tasks** (file-disjoint batch). Total waves: **${schedule.stats.waveCount}**.

---

## 2. Phase overview (logical grouping — waves may cross phases)

| Phase | Name | Tasks | Notes |
|-------|------|------:|-------|
| 0 | P0 correctness | ${counts.phases[0] ?? 0} | **Wave W000:** all ${PHASE_0.length} run in parallel |
| 1 | Contract + engine | ${counts.phases[1] ?? 0} | Interleaved with other lanes |
| 2 | pt-webgl2 + FP | ${counts.phases[2] ?? 0} | FP-01→06 chain is sequential |
| 3 | pt-webgpu | ${counts.phases[3] ?? 0} | Independent of walkaround lane |
| 4 | walkaround | ${counts.phases[4] ?? 0} | Largest lane; internal file mutex |
| 5 | glTF | ${counts.phases[5] ?? 0} | Fully parallel with other lanes |
| 6 | Inverse | ${counts.phases[6] ?? 0} | inverseSession/adjoint mutex |
| 7 | Tools + shared | ${counts.phases[7] ?? 0} | Behavioral gate, radiometric-ab |
| 8 | RT100 V28 validation | ${counts.phases[8] ?? 0} | **Deferred** — not scheduled |
| 9 | RT100 impl/promotion | ${counts.phases[9] ?? 0} | Adjoint, atlas tail, PT mutations |
| 10 | RT100 proof | ${counts.phases[10] ?? 0} | Golden PNG, doc sync, full-tier gate |
| 11 | RT100 decisions | ${counts.phases[11] ?? 0} | NRC tier, neural weights |
| 12 | RT100 SOTA perf | ${counts.phases[12] ?? 0} | LD sampling, WBVH, denoiser:auto |

**Scheduling truth:** \`${schedule.stats.waveCount} waves\`, not phase order. Phases are taxonomy only.

---

## 3. Task register (full)

`;

let body = PROTOCOL;
for (const t of ALL_TASKS) {
  body += renderTask(t);
}

fs.writeFileSync(OUT, body);

fs.writeFileSync(
  JSONL,
  schedule.tasks.map((t) => JSON.stringify(t)).join('\n') + '\n',
);

fs.writeFileSync(
  PARALLEL_JSON,
  JSON.stringify({ generated: new Date().toISOString(), stats: schedule.stats, batches, waves: schedule.waves }, null, 2) + '\n',
);

fs.writeFileSync(PARALLEL_MD, renderParallelMarkdown(schedule, batches) + '\n');

const laneWorkers = [
  '# Lane Worker Registry',
  '',
  `> **Agent pool cap: ${MAX_AGENTS_PER_WAVE} concurrent agents.** One task per agent per wave.`,
  '> Orchestrator advances `currentWave` in parallel-schedule.json after each integration.',
  '',
  `**Current wave:** W000`,
  '',
  '## Active agents (max 25)',
  '',
  '| slot | agent | taskId | lane | wave | status |',
  '|------|-------|--------|------|------|--------|',
  ...Array.from({ length: MAX_AGENTS_PER_WAVE }, (_, i) => `| ${i + 1} | _empty_ | | | | |`),
  '',
  '## Lane roster',
  '',
  '| lane | worker agent | notes |',
  '|------|--------------|-------|',
  ...Object.entries(schedule.stats.laneStats)
    .sort((a, b) => b[1] - a[1])
    .map(([lane, n]) => `| \`${lane}\` | _unassigned_ | ${n} tasks |`),
  '',
];
fs.writeFileSync(LANES_MD, laneWorkers.join('\n'));

const progressLines = [
  '# Code Gap Task Progress',
  '',
  `> Parallel schedule: ${schedule.stats.waveCount} waves. Check wave column before claiming.`,
  '',
  '| Task ID | Wave | Lane | Phase | Priority | Status |',
  '|---------|------|------|-------|----------|--------|',
  ...schedule.tasks.map((t) => `| ${t.id} | ${t.waveId} | ${t.lane} | ${t.phase} | ${t.priority} | ⬜ pending |`),
  '',
];
fs.writeFileSync(PROGRESS, progressLines.join('\n'));

// Wave manifests + orchestrator resume
const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const manifestResult = spawnSync(process.execPath, [path.join(scriptDir, 'generate-wave-manifests.mjs')], {
  cwd: ROOT,
  stdio: 'inherit',
});
if (manifestResult.status !== 0) {
  console.error('generate-wave-manifests.mjs failed');
  process.exit(manifestResult.status ?? 1);
}
spawnSync(process.execPath, [path.join(scriptDir, 'orchestrator-run.mjs'), 'resume-md'], {
  cwd: ROOT,
  stdio: 'inherit',
});

console.log(`Wrote ${OUT} (${ALL_TASKS.length} tasks, ${body.split('\n').length} lines)`);
console.log(`Wrote ${JSONL} (${schedule.tasks.length} scheduled)`);
console.log(`Wrote ${PARALLEL_MD} (${schedule.stats.waveCount} waves, cap ${MAX_AGENTS_PER_WAVE}/wave, largest wave ${schedule.stats.maxParallelism})`);
console.log(`Wrote ${PARALLEL_JSON}`);
console.log(`Wrote ${LANES_MD}`);
console.log(`Wrote ${PROGRESS}`);
