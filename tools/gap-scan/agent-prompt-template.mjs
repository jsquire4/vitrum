/**
 * Builds the exact prompt text for one sub-agent slot.
 */

const ROOT = '/home/jsquire4/projects/vitrum';

/**
 * @param {object} opts
 * @param {string} opts.waveId
 * @param {number} opts.slot
 * @param {object} opts.task
 * @returns {string}
 */
export function buildAgentPrompt({ waveId, slot, task }) {
  const files = task.files.map((f) => `- \`${f}\``).join('\n');
  const steps = task.steps.map((s, i) => `${i + 1}. ${s}`).join('\n');
  const tests = task.tests.map((t) => `- \`${t}\``).join('\n');
  const done = task.done.map((d) => `- ${d}`).join('\n');

  return `# Gap remediation worker — ${waveId} / agent-${String(slot).padStart(2, '0')}

You are **worker agent-${String(slot).padStart(2, '0')}** in wave **${waveId}**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | \`${task.id}\` |
| Lane | \`${task.lane}\` |
| Disposition | \`${task.disposition}\` |
| Priority | \`${task.priority}\` |

## Problem
${task.problem}

## Files you may edit (ONLY these)
${files}

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
${steps}

## Tests you must run locally
${tests}

## Definition of done
${done}

## Hard rules
1. Implement **only** task \`${task.id}\`.
2. Run every test command; fix until green.
3. **Do not** \`git commit\` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
\`\`\`
TASK_COMPLETE: ${task.id}
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
\`\`\`
`;
}

/**
 * @param {object} opts
 * @param {string} opts.waveId
 * @param {number} opts.slot
 * @param {object} opts.task
 * @param {string} opts.failureReason
 * @returns {string}
 */
export function buildRemediationPrompt({ waveId, slot, task, failureReason }) {
  return `${buildAgentPrompt({ waveId, slot, task })}

---

## REMEDIATION (orchestrator verification failed)

Previous attempt for \`${task.id}\` did **not** pass orchestrator verification.

**Failure reason:**
${failureReason}

Fix the issue, re-run all tests, and do not repeat the same mistake.
`;
}

/**
 * Orchestrator-only verification spec for one task.
 * @param {object} task
 * @param {string} baseSha
 * @returns {object}
 */
export function buildVerifySpec(task, baseSha) {
  return {
    taskId: task.id,
    disposition: task.disposition,
    files: task.files,
    tests: task.tests,
    baseSha,
    checks: {
      requireGitDiff: !['VERIFY'].includes(task.disposition),
      requireTestsPass: task.tests.length > 0,
      requireTestAdded:
        task.disposition === 'BUG' &&
        task.steps.some((s) => /add.*test|regression test/i.test(s)),
    },
  };
}

export { ROOT };
