#!/usr/bin/env node
// @ts-check
// Human-readable Road-to-100 queue triage. The validation checker is the
// pass/fail guard; this reporter is the work-order view so code, proof, and
// future-contract tails do not get mixed together during long closure runs.

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(scriptDir, "../..");
const queuePath = resolve(repoRoot, "tools/road-to-100/validation-queue.json");

const PROOF_STATUSES = new Set(["partial-proof-green", "host-blocked", "evidence-needed"]);
const PROVISIONING_STATUSES = new Set(["provisioning-needed"]);
const DECISION_STATUSES = new Set(["decision-needed"]);
const RESEARCH_PROMOTION_CLASSES = new Set(["research-promotion"]);

/**
 * @typedef {{
 *   id: string,
 *   title?: string,
 *   status: string,
 *   kind?: string,
 *   workClass?: string,
 *   remaining?: string,
 *   command?: string,
 *   promotionCommand?: string,
 *   currentContract?: string,
 *   decisionBlockers?: string[]
 * }} QueueRow
 */

/**
 * @typedef {{
 *   currentAsOf: string,
 *   implementationQueue: QueueRow[],
 *   validationQueue: QueueRow[],
 *   futureContractRows: QueueRow[]
 * }} Queue
 */

/**
 * @typedef {{
 *   currentAsOf: string,
 *   implementation: QueueRow[],
 *   validationByStatus: Map<string, QueueRow[]>,
 *   proofRows: QueueRow[],
 *   researchPromotionRows: QueueRow[],
 *   provisioningRows: QueueRow[],
 *   decisionRows: QueueRow[],
 *   futureRows: QueueRow[],
 *   activeCodeBlocked: boolean
 * }} QueueSummary
 */

/**
 * @param {unknown} value
 * @param {string} label
 * @returns {QueueRow[]}
 */
function requireRows(value, label) {
  if (!Array.isArray(value)) throw new Error(`${label} must be an array`);
  return value.map((row, index) => {
    if (row == null || typeof row !== "object") throw new Error(`${label}[${index}] must be an object`);
    const id = "id" in row ? String(row.id) : "";
    const status = "status" in row ? String(row.status) : "";
    if (id.length === 0) throw new Error(`${label}[${index}].id must be present`);
    if (status.length === 0) throw new Error(`${label}[${index}].status must be present`);
    return /** @type {QueueRow} */ (row);
  });
}

/**
 * @param {unknown} raw
 * @returns {Queue}
 */
export function parseQueue(raw) {
  if (raw == null || typeof raw !== "object") throw new Error("queue must be an object");
  const queue = /** @type {Record<string, unknown>} */ (raw);
  const currentAsOf = String(queue.currentAsOf ?? "");
  if (!/^\d{4}-\d{2}-\d{2}$/.test(currentAsOf)) {
    throw new Error("queue.currentAsOf must be YYYY-MM-DD");
  }
  return {
    currentAsOf,
    implementationQueue: requireRows(queue.implementationQueue, "implementationQueue"),
    validationQueue: requireRows(queue.validationQueue, "validationQueue"),
    futureContractRows: requireRows(queue.futureContractRows, "futureContractRows"),
  };
}

/**
 * @param {Queue} queue
 * @returns {QueueSummary}
 */
export function summarizeQueue(queue) {
  /** @type {Map<string, QueueRow[]>} */
  const validationByStatus = new Map();
  for (const row of queue.validationQueue) {
    const rows = validationByStatus.get(row.status) ?? [];
    rows.push(row);
    validationByStatus.set(row.status, rows);
  }

  const researchPromotionRows = queue.validationQueue.filter((row) =>
    RESEARCH_PROMOTION_CLASSES.has(row.workClass ?? "")
  );
  const researchPromotionIds = new Set(researchPromotionRows.map((row) => row.id));
  const proofRows = queue.validationQueue.filter((row) =>
    PROOF_STATUSES.has(row.status) && !researchPromotionIds.has(row.id)
  );
  const provisioningRows = queue.validationQueue.filter((row) => PROVISIONING_STATUSES.has(row.status));
  const decisionRows = queue.validationQueue.filter((row) => DECISION_STATUSES.has(row.status));

  return {
    currentAsOf: queue.currentAsOf,
    implementation: queue.implementationQueue,
    validationByStatus,
    proofRows,
    researchPromotionRows,
    provisioningRows,
    decisionRows,
    futureRows: queue.futureContractRows,
    activeCodeBlocked: queue.implementationQueue.length > 0,
  };
}

/**
 * @param {QueueRow[]} rows
 */
function ids(rows) {
  return rows.map((row) => row.id).join(", ") || "none";
}

/**
 * @param {QueueSummary} summary
 * @returns {string}
 */
export function formatSummary(summary) {
  const lines = [
    "[road-to-100-next-actions]",
    `currentAsOf: ${summary.currentAsOf}`,
    `implementationQueue: ${summary.implementation.length}`,
  ];

  if (summary.implementation.length > 0) {
    lines.push(`code-now: ${ids(summary.implementation)}`);
  } else {
    lines.push("code-now: none (do not reopen source work unless implementationQueue gains a source-verified row)");
  }

  lines.push("validation:");
  for (const [status, rows] of [...summary.validationByStatus.entries()].sort(([a], [b]) => a.localeCompare(b))) {
    lines.push(`  ${status}: ${rows.length} (${ids(rows)})`);
  }

  lines.push(`proof-or-adapter-work: ${summary.proofRows.length} (${ids(summary.proofRows)})`);
  lines.push(`research-promotion-work: ${summary.researchPromotionRows.length} (${ids(summary.researchPromotionRows)})`);
  lines.push(`provisioning-work: ${summary.provisioningRows.length} (${ids(summary.provisioningRows)})`);
  lines.push(`decision-work: ${summary.decisionRows.length} (${ids(summary.decisionRows)})`);
  lines.push(`future-contract: ${summary.futureRows.length} (${ids(summary.futureRows)})`);

  lines.push("next:");
  if (summary.activeCodeBlocked) {
    lines.push("  1. Finish implementationQueue rows before spending time on validation recaptures.");
  } else {
    lines.push("  1. Keep code frozen except for proof harness fixes required by a failing validation lane.");
  }
  if (summary.proofRows.length > 0) {
    lines.push("  2. Run or provision the proof/adapter rows listed above; do not promote partial rows from source-only evidence.");
  }
  if (summary.researchPromotionRows.length > 0) {
    lines.push("  3. Treat research-promotion rows as estimator/default-tier work, not forgotten implementationQueue bugs.");
  }
  if (summary.provisioningRows.length > 0) {
    lines.push("  4. Provision learned-system assets/quality manifests before claiming production neural/NRC status.");
  }
  if (summary.futureRows.length > 0) {
    lines.push("  5. Treat future-contract rows as product/API design work, not active bugs.");
  }

  return lines.join("\n");
}

/**
 * @param {QueueRow} row
 * @returns {string[]}
 */
function formatRowDetails(row) {
  const lines = [`  - ${row.id}: ${row.title ?? "(untitled)"} [${row.status}]`];
  if (row.kind) lines.push(`    kind: ${row.kind}`);
  if (row.workClass) lines.push(`    workClass: ${row.workClass}`);
  if (row.command) lines.push(`    command: ${row.command}`);
  if (row.promotionCommand) lines.push(`    promotionCommand: ${row.promotionCommand}`);
  if (row.remaining) lines.push(`    remaining: ${row.remaining}`);
  if (row.currentContract) lines.push(`    currentContract: ${row.currentContract}`);
  if (Array.isArray(row.decisionBlockers) && row.decisionBlockers.length > 0) {
    lines.push("    decisionBlockers:");
    for (const blocker of row.decisionBlockers) lines.push(`      * ${blocker}`);
  }
  return lines;
}

/**
 * @param {string[]} lines
 * @param {string} title
 * @param {QueueRow[]} rows
 */
function appendDetailedSection(lines, title, rows) {
  lines.push(`${title}: ${rows.length}`);
  if (rows.length === 0) {
    lines.push("  - none");
    return;
  }
  for (const row of rows) lines.push(...formatRowDetails(row));
}

/**
 * @param {QueueSummary} summary
 * @returns {string}
 */
export function formatDetailedSummary(summary) {
  const lines = [
    formatSummary(summary),
    "",
    "[road-to-100-next-actions:details]",
  ];

  appendDetailedSection(lines, "implementationQueue", summary.implementation);
  appendDetailedSection(lines, "proofOrAdapterWork", summary.proofRows);
  appendDetailedSection(lines, "researchPromotionWork", summary.researchPromotionRows);
  appendDetailedSection(lines, "provisioningWork", summary.provisioningRows);
  appendDetailedSection(lines, "decisionWork", summary.decisionRows);
  appendDetailedSection(lines, "futureContract", summary.futureRows);

  return lines.join("\n");
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const raw = JSON.parse(readFileSync(queuePath, "utf8"));
  const summary = summarizeQueue(parseQueue(raw));
  if (process.argv.includes("--json")) {
    console.log(JSON.stringify({
      currentAsOf: summary.currentAsOf,
      implementationQueue: summary.implementation.map((row) => row.id),
      validationByStatus: Object.fromEntries(
        [...summary.validationByStatus.entries()].map(([status, rows]) => [status, rows.map((row) => row.id)]),
      ),
      proofOrAdapterWork: summary.proofRows.map((row) => row.id),
      researchPromotionWork: summary.researchPromotionRows.map((row) => row.id),
      provisioningWork: summary.provisioningRows.map((row) => row.id),
      decisionWork: summary.decisionRows.map((row) => row.id),
      futureContract: summary.futureRows.map((row) => row.id),
    }, null, 2));
  } else if (process.argv.includes("--details")) {
    console.log(formatDetailedSummary(summary));
  } else {
    console.log(formatSummary(summary));
  }
}
