/**
 * Computes maximum-parallelism execution waves from task registry.
 * Two tasks share a wave iff: (1) all explicit deps satisfied, (2) no file overlap.
 *
 * Run standalone: node tools/gap-scan/parallel-schedule.mjs
 */

/** @typedef {import('./task-registry.mjs').ALL_TASKS extends infer T ? T extends readonly (infer U)[] ? U : never : never} Task */

import {
  isDeferredValidationTask,
  DEFERRED_VALIDATION_TASK_IDS,
  listDeferredValidationTasks,
} from './validation-deferral.mjs';

/** Tasks superseded by Phase-0 aliases — skip in schedule. */
export const SKIP_TASK_IDS = new Set([
  'PTGL-003',
  'PTWG-037',
  'WH-034',
  'CORE-041',
]);

/**
 * @param {{ id: string }} t
 * @returns {boolean}
 */
export function isSkippedTask(t) {
  return SKIP_TASK_IDS.has(t.id) || isDeferredValidationTask(t);
}

/** Synthetic gate — all non-bootstrap tasks wait for Phase-0 P0 wave. */
export const BOOTSTRAP_GATE = '__BOOTSTRAP_P0__';

/** P0 correctness tasks — always wave W000, exclusively. */
export const BOOTSTRAP_TASK_IDS = new Set([
  'P0-001-PTWG-037',
  'P0-002-PTGL-003',
  'P0-003-WH-034',
  'P0-004-DENO-001',
  'P0-005-LEDGER-01',
  'P0-006-LEDGER-02',
  'P0-007-LEDGER-03',
  'P0-008-TOOL-001',
]);

/** Extra implicit dependencies not in registry. */
export const IMPLICIT_DEPENDS = {
  'FP-02': ['FP-01'],
  'FP-03': ['FP-02'],
  'FP-04': ['FP-01'],
  'FP-05': ['FP-01'],
  'FP-06': ['FP-01', 'FP-03'],
  'TOOL-002': ['P0-008-TOOL-001'],
  'TOOL-003': ['P0-008-TOOL-001'],
};

/** Ledger-only edits serialize through one mutex owner at a time. */
export const GLOBAL_MUTEX_FILE_PREFIXES = [
  'packages/core/src/engine/promiseLedger.ts',
];

/** Hard cap on concurrent agents per wave (orchestrator pool size). */
export const MAX_AGENTS_PER_WAVE = 25;

/**
 * @param {string} filePath
 * @returns {string}
 */
export function laneFromFile(filePath) {
  const m = filePath.match(/^packages\/([^/]+)/);
  if (m) return m[1];
  if (filePath.startsWith('tools/')) return 'tools';
  return 'repo-root';
}

/**
 * @param {Task} t
 * @returns {string}
 */
export function primaryLane(t) {
  const lanes = [...new Set(t.files.map(laneFromFile))];
  if (lanes.length === 1) return lanes[0];
  if (lanes.includes('core')) return 'core';
  return lanes[0] ?? 'unknown';
}

/**
 * @param {Task} t
 * @returns {string[]}
 */
export function allDepends(t) {
  const explicit = t.depends ?? [];
  const implicit = IMPLICIT_DEPENDS[t.id] ?? [];
  const bootstrap =
    !BOOTSTRAP_TASK_IDS.has(t.id) && t.phase !== 0 ? [BOOTSTRAP_GATE] : [];
  return [...new Set([...explicit, ...implicit, ...bootstrap])];
}

/**
 * Files that are touched incidentally (validation sync) — do not participate in mutex.
 * @param {Task} t
 * @param {string} file
 */
function isMutexFile(t, file) {
  if (!GLOBAL_MUTEX_FILE_PREFIXES.includes(file)) return false;
  // DENO-001 lists promiseLedger for row sync — not a structural ledger rewrite
  if (t.id === 'P0-004-DENO-001' && file.includes('promiseLedger')) return false;
  return true;
}

/**
 * @param {Task} t
 * @returns {boolean}
 */
function touchesGlobalMutex(t) {
  return t.files.some((f) => isMutexFile(t, f));
}

/**
 * @param {Task} a
 * @param {Task} b
 * @returns {boolean}
 */
function fileConflict(a, b) {
  for (const fa of a.files) {
    for (const fb of b.files) {
      if (fa === fb) return true;
      // Directory-level conflict for heavily shared WGSL modules (same folder file edits)
      if (fa.includes('/shaders/') && fb.includes('/shaders/')) {
        const da = fa.split('/').slice(0, -1).join('/');
        const db = fb.split('/').slice(0, -1).join('/');
        if (da === db && (fa.endsWith('.wgsl.ts') || fb.endsWith('.wgsl.ts'))) {
          // Only exact file match for wgsl — directory is too coarse
        }
      }
    }
  }
  return false;
}

/**
 * @param {Task[]} tasks
 * @returns {{ waves: object[], tasks: object[], stats: object }}
 */
export function computeParallelSchedule(tasks) {
  /** @type {Set<string>} */
  const deferredValidationIds = new Set(
    tasks.filter((t) => isDeferredValidationTask(t)).map((t) => t.id),
  );
  /** @param {string} id */
  const isScheduledOut = (id) => SKIP_TASK_IDS.has(id) || deferredValidationIds.has(id);

  const active = tasks.filter((t) => !isScheduledOut(t.id));

  /** @type {Set<string>} */
  const done = new Set();
  /** @type {object[]} */
  const waves = [];
  /** @type {object[]} */
  const enriched = [];

  const priorityRank = { P0: 0, P1: 1, P2: 2 };

  // ── Wave W000: bootstrap (P0 only, all parallel) ─────────────────────
  const bootstrap = active.filter((t) => BOOTSTRAP_TASK_IDS.has(t.id));
  if (bootstrap.length > 0) {
    waves.push({
      wave: 0,
      waveId: 'W000',
      parallel: bootstrap.length > 1,
      bootstrap: true,
      agentCap: MAX_AGENTS_PER_WAVE,
      taskCount: bootstrap.length,
      lanes: [...new Set(bootstrap.map(primaryLane))],
      taskIds: bootstrap.map((t) => t.id),
    });
    for (const t of bootstrap) {
      done.add(t.id);
      enriched.push({
        ...t,
        wave: 0,
        waveId: 'W000',
        lane: primaryLane(t),
        lanes: [...new Set(t.files.map(laneFromFile))],
        depends: allDepends(t).filter((d) => d !== BOOTSTRAP_GATE),
        parallelSafe: bootstrap.length > 1,
        bootstrap: true,
        mutexFiles: t.files.filter((f) => isMutexFile(t, f)),
      });
    }
  }
  done.add(BOOTSTRAP_GATE);

  let waveIndex = 1;
  let remaining = active.filter((t) => !BOOTSTRAP_TASK_IDS.has(t.id));

  while (remaining.length > 0) {
    const candidates = remaining
      .filter((t) => allDepends(t).every((d) => done.has(d) || isScheduledOut(d)))
      .sort((a, b) => {
        const pr = (priorityRank[a.priority] ?? 9) - (priorityRank[b.priority] ?? 9);
        if (pr !== 0) return pr;
        return a.phase - b.phase;
      });

    if (candidates.length === 0) {
      throw new Error(
        `Scheduler deadlock: ${remaining.length} tasks left, 0 ready. Remaining: ${remaining
          .slice(0, 5)
          .map((t) => t.id)
          .join(', ')}…`,
      );
    }

    /** @type {Task[]} */
    const picked = [];
    /** @type {Set<string>} */
    const usedFiles = new Set();
    let ledgerMutexTaken = false;

    for (const c of candidates) {
      const cLedger = touchesGlobalMutex(c);
      if (cLedger && ledgerMutexTaken) continue;

      let conflict = false;
      for (const f of c.files) {
        if (usedFiles.has(f)) {
          conflict = true;
          break;
        }
      }
      if (conflict) continue;

      for (const p of picked) {
        if (fileConflict(c, p)) {
          conflict = true;
          break;
        }
      }
      if (conflict) continue;

      picked.push(c);
      for (const f of c.files) usedFiles.add(f);
      if (cLedger) ledgerMutexTaken = true;

      if (picked.length >= MAX_AGENTS_PER_WAVE) break;
    }

    if (picked.length === 0) {
      picked.push(candidates[0]);
    }

    const waveId = `W${String(waveIndex).padStart(3, '0')}`;
    const lanes = [...new Set(picked.map(primaryLane))];

    waves.push({
      wave: waveIndex,
      waveId,
      parallel: picked.length > 1,
      bootstrap: false,
      agentCap: MAX_AGENTS_PER_WAVE,
      taskCount: picked.length,
      lanes,
      taskIds: picked.map((t) => t.id),
    });

    for (const t of picked) {
      done.add(t.id);
      enriched.push({
        ...t,
        wave: waveIndex,
        waveId,
        lane: primaryLane(t),
        lanes: [...new Set(t.files.map(laneFromFile))],
        depends: allDepends(t),
        parallelSafe: picked.length > 1,
        bootstrap: false,
        mutexFiles: t.files.filter((f) => isMutexFile(t, f)),
      });
    }

    const pickedIds = new Set(picked.map((t) => t.id));
    remaining = remaining.filter((t) => !pickedIds.has(t.id));
    waveIndex += 1;
  }

  const laneStats = {};
  for (const t of enriched) {
    laneStats[t.lane] = (laneStats[t.lane] ?? 0) + 1;
  }

  const maxParallelism = Math.max(...waves.map((w) => w.taskCount));
  const parallelWaves = waves.filter((w) => w.parallel).length;

  return {
    waves,
    tasks: enriched,
    stats: {
      totalTasks: active.length,
      bootstrapTasks: bootstrap.length,
      skippedDuplicates: [...SKIP_TASK_IDS].filter((id) => tasks.some((t) => t.id === id)).length,
      deferredValidation: deferredValidationIds.size,
      waveCount: waves.length,
      maxAgentsPerWave: MAX_AGENTS_PER_WAVE,
      maxParallelism,
      parallelWaves,
      avgParallelism: (active.length / waves.length).toFixed(2),
      laneStats,
    },
  };
}

/**
 * Group waves into super-batches for multi-agent orchestration.
 * @param {object[]} waves
 * @returns {object[]}
 */
export function computeSuperBatches(waves) {
  const batches = [];
  let i = 0;
  while (i < waves.length) {
    const start = i;
    const batchWaves = [];
    /** @type {Set<string>} */
    const batchLanes = new Set();

    // Accumulate consecutive waves while lane-disjoint OR high parallelism
    while (i < waves.length) {
      const w = waves[i];
      const overlap = [...w.lanes].some((l) => batchLanes.has(l));
      if (batchWaves.length > 0 && overlap && w.taskCount > 1) break;
      batchWaves.push(w);
      for (const l of w.lanes) batchLanes.add(l);
      i += 1;
      if (batchWaves.length >= 4) break; // ~4 waves × 25 agents before extended integration
    }

    batches.push({
      batchId: `B${String(batches.length).padStart(2, '0')}`,
      waveRange: [batchWaves[0].wave, batchWaves[batchWaves.length - 1].wave],
      waveIds: batchWaves.map((w) => w.waveId),
      totalTasks: batchWaves.reduce((s, w) => s + w.taskCount, 0),
      maxParallelInBatch: Math.max(...batchWaves.map((w) => w.taskCount)),
      lanes: [...batchLanes],
    });
  }
  return batches;
}

/**
 * @param {object} schedule
 * @returns {string}
 */
export function renderParallelMarkdown(schedule, batches) {
  const { waves, stats } = schedule;
  const lines = [
    '# Code Gap Parallel Execution Schedule',
    '',
    `> **${stats.totalTasks} tasks** in **${stats.waveCount} waves** (${stats.bootstrapTasks} bootstrap + ${stats.totalTasks - stats.bootstrapTasks} scheduled).`,
    `> **Agent cap: ${stats.maxAgentsPerWave} per wave** — dispatch at most ${stats.maxAgentsPerWave} agents; each wave lists ≤${stats.maxAgentsPerWave} tasks.`,
    `> Skipped duplicates: ${[...SKIP_TASK_IDS].join(', ')}`,
    `> Deferred validation (code-first): see plan/VALIDATION-DEFERRED.md (${stats.deferredValidation ?? 0} tasks)`,
    '',
    '## How to run in parallel',
    '',
    `1. **Pool size:** maintain **≤${stats.maxAgentsPerWave} concurrent agents**. Never exceed the task count in the current wave.`,
    '2. **Wave W000 (bootstrap):** 8 `P0-*` tasks — all parallel, then integrate.',
    '3. **W001+:** spawn up to **25 agents** (or fewer if the wave has fewer tasks). Each agent claims one `taskId` from the current wave JSON.',
    '4. After **every** wave: run integration (`npm run typecheck` + lane tests). Proceed only when green.',
    '5. **Never** edit a file another active agent owns. **Never** start tasks from the next wave early.',
    '',
    '## Stats',
    '',
    '| Metric | Value |',
    '|--------|------:|',
    `| Active tasks | ${stats.totalTasks} |`,
    `| Waves | ${stats.waveCount} |`,
    `| Agent cap (per wave) | ${stats.maxAgentsPerWave} |`,
    `| Largest wave (tasks) | ${stats.maxParallelism} |`,
    `| Waves with parallelism >1 | ${stats.parallelWaves} |`,
    `| Avg tasks/wave | ${stats.avgParallelism} |`,
    '',
    '### Tasks per lane (worker pool sizing)',
    '',
    '| Lane | Tasks | Suggested workers |',
    '|------|------:|------------------:|',
  ];

  for (const [lane, count] of Object.entries(stats.laneStats).sort((a, b) => b[1] - a[1])) {
    const workers = Math.min(count, stats.maxAgentsPerWave);
    lines.push(`| \`${lane}\` | ${count} | ${workers} |`);
  }

  lines.push('', '## Super-batches (integration checkpoints)', '');
  lines.push('| Batch | Waves | Tasks | Max parallel | Lanes |');
  lines.push('|-------|-------|------:|-------------:|-------|');
  for (const b of batches) {
    lines.push(
      `| ${b.batchId} | ${b.waveIds[0]}–${b.waveIds[b.waveIds.length - 1]} | ${b.totalTasks} | ${b.maxParallelInBatch} | ${b.lanes.join(', ')} |`,
    );
  }

  lines.push('', '## Wave table (first 40 waves)', '');
  lines.push('| Wave | Parallel | Tasks | Lanes | Task IDs |');
  lines.push('|------|:--------:|------:|-------|----------|');
  for (const w of waves.slice(0, 40)) {
    const ids =
      w.taskIds.length <= 6
        ? w.taskIds.join(', ')
        : `${w.taskIds.slice(0, 5).join(', ')} +${w.taskIds.length - 5}`;
    lines.push(`| ${w.waveId} | ${w.parallel ? '✓' : '·'} | ${w.taskCount} | ${w.lanes.join(', ')} | ${ids} |`);
  }
  if (waves.length > 40) {
    lines.push(`| … | | | | _${waves.length - 40} more waves — see JSON_ |`);
  }

  lines.push('', '## Lane worker charters', '');
  const charter = {
    core: 'Contract, promiseLedger, capabilities, inverse types. **Mutex:** promiseLedger.ts — max 1 agent globally.',
    engine: 'createEngine, VitrumCanvas, progressive handoff, backends.',
    'pt-webgl2': 'WebGL2 PT backend, GLSL, scene mutations.',
    'pt-webgpu': 'WebGPU PT, inverse/adjoint, specialty integrators.',
    'walkaround-hybrid': 'Realtime GI — largest lane. Sub-lanes: restir/, ddgi/, shaders/, pipeline/.',
    'gltf-adapter': 'glTF ingest, featureReport, compression.',
    'shared-bvh': 'BVH shared package — independent.',
    'shared-samplers': 'Samplers shared — independent.',
    'shared-denoisers': 'Denoisers shared — independent.',
    'walkaround-rc': 'RC subsystem — independent.',
    tools: 'behavioral-gate, shader-gate, benchmarks.',
    dev: 'Debug overlays — independent.',
  };
  for (const [lane, text] of Object.entries(charter)) {
    if (stats.laneStats[lane]) lines.push(`### \`${lane}\` (${stats.laneStats[lane]} tasks)`, '', text, '');
  }

  return lines.join('\n');
}
