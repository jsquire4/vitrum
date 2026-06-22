#!/usr/bin/env -S deno run --sloppy-imports --allow-read
// @ts-check
// Verifies the Road-to-100 work queue is explicit, machine-readable, and not
// silently turning validation/provisioning/future-contract tails into code gaps.

const QUEUE_PATH = "tools/road-to-100/validation-queue.json";
const PACKAGE_PATH = "package.json";
const EXECUTION_PLAN_PATH = "plan/gap-closure-execution-plan.md";
const LEDGER_PATH = "plan/road-to-100-gap-ledger-2026-06-11.md";
const LEARNED_CHECKPOINT_MANIFEST_PATH = "tools/neural-denoiser-training/checkpoints/manifest.json";

const ALLOWED_STATUSES = new Set([
  "committed-proof-green",
  "partial-proof-green",
  "host-blocked",
  "evidence-needed",
  "provisioning-needed",
  "decision-needed",
  "future-contract",
]);

const REQUIRED_VALIDATION_IDS = [
  "VQ-PT-WEBGPU-RUNTIME-GOLDENS",
  "VQ-WALKAROUND-BEHAVIORAL-MATRIX",
  "VQ-MUTATION-MATRIX",
  "VQ-GLTF-REAL-WEBGPU",
  "VQ-GLTF-BROWSER-PTWEBGL2",
  "VQ-RADIOMETRIC-PT",
  "VQ-WALKAROUND-RADIOMETRIC-AB",
  "VQ-RENDERER-FIDELITY-PROOF",
  "VQ-ADJOINT-SCOPED-PATH-REPLAY",
  "VQ-LEARNED-SYSTEMS",
  "VQ-GLTF-MATERIAL-TOPOLOGY",
];

const REQUIRED_FUTURE_IDS = [
  "FC-DISPLACEMENT-MICROTESSELLATION",
  "FC-TRANSPARENT-GI-TRANSPORT",
  "FC-NATIVE-POINT-LINE",
  "FC-ARBITRARY-UV-ARRAYS",
  "FC-NATIVE-INSTANCED-SKINNED-MORPHED",
  "FC-ADJOINT-FULL-PATH-PARITY",
];

/** @param {string} path */
function repoUrl(path) {
  return new URL(`../../${path}`, import.meta.url);
}

/** @param {string} message @returns {never} */
function fail(message) {
  throw new Error(`[road-to-100-validation-status] ${message}`);
}

/** @param {string} path */
async function readText(path) {
  return await Deno.readTextFile(repoUrl(path));
}

/** @param {string} path */
async function readJson(path) {
  return JSON.parse(await readText(path));
}

/** @param {string} path */
async function assertFile(path) {
  const stat = await Deno.stat(repoUrl(path));
  if (!stat.isFile) fail(`${path} is missing or not a file`);
  if (stat.size <= 0) fail(`${path} is empty`);
}

/**
 * @param {unknown} value
 * @param {string} label
 */
function assertNonEmptyString(value, label) {
  if (typeof value !== "string" || value.length === 0) fail(`${label} must be a non-empty string`);
}

/**
 * @param {readonly unknown[]} rows
 * @param {readonly string[]} requiredIds
 * @param {string} label
 */
function assertRequiredIds(rows, requiredIds, label) {
  const ids = new Set(rows.map((row) => row && typeof row === "object" ? String(row.id ?? "") : ""));
  for (const id of requiredIds) {
    if (!ids.has(id)) fail(`${label} missing ${id}`);
  }
  if (ids.size !== rows.length) fail(`${label} contains duplicate or invalid ids`);
}

/**
 * @param {string} command
 * @param {Record<string, string>} scripts
 */
function assertCommandScriptsExist(command, scripts) {
  for (const part of command.split(/\s*&&\s*/)) {
    const match = part.trim().match(/^npm run ([^\s]+)/);
    if (!match) fail(`command must be an npm run script chain: ${command}`);
    const script = match[1];
    if (scripts[script] == null) fail(`command references missing package script "${script}"`);
  }
}

/**
 * @param {unknown} object
 * @param {string} path
 */
function getJsonPath(object, path) {
  let cur = object;
  for (const part of path.split(".")) {
    if (cur == null || typeof cur !== "object" || !(part in cur)) return undefined;
    cur = cur[part];
  }
  return cur;
}

/**
 * @param {unknown} artifact
 * @param {string} ownerId
 */
async function assertArtifact(artifact, ownerId) {
  if (artifact == null || typeof artifact !== "object") fail(`${ownerId}: artifact must be an object`);
  const path = artifact.path;
  assertNonEmptyString(path, `${ownerId}: artifact.path`);
  await assertFile(path);

  if (artifact.type === "png" || String(path).endsWith(".png")) {
    const bytes = await Deno.readFile(repoUrl(path));
    if (bytes[0] !== 0x89 || bytes[1] !== 0x50 || bytes[2] !== 0x4e || bytes[3] !== 0x47) {
      fail(`${ownerId}: ${path} is not a PNG`);
    }
  }

  if (artifact.json != null) {
    if (typeof artifact.json !== "object" || Array.isArray(artifact.json)) {
      fail(`${ownerId}: ${path} json expectations must be an object`);
    }
    const json = await readJson(path);
    for (const [jsonPath, expected] of Object.entries(artifact.json)) {
      const actual = getJsonPath(json, jsonPath);
      if (actual !== expected) {
        fail(`${ownerId}: ${path} ${jsonPath} expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
      }
    }
  }
}

const [queue, packageJson, executionPlan, ledger, road] = await Promise.all([
  readJson(QUEUE_PATH),
  readJson(PACKAGE_PATH),
  readText(EXECUTION_PLAN_PATH),
  readText(LEDGER_PATH),
  readText("plan/road-to-100.md"),
]);
const learnedCheckpointManifest = await readJson(LEARNED_CHECKPOINT_MANIFEST_PATH);

if (queue.schema !== "vitrum.road-to-100.validation-queue.v1") fail("queue schema mismatch");
if (typeof queue.currentAsOf !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(queue.currentAsOf)) {
  fail("queue currentAsOf must be YYYY-MM-DD");
}
if (queue.currentAsOf < "2026-06-22") fail("queue currentAsOf predates the latest code-closure wave");

if (!Array.isArray(queue.implementationQueue)) fail("implementationQueue must be an array");
if (queue.implementationQueue.length !== 0) {
  fail("implementationQueue must stay empty unless a source-verified code bug is promoted");
}
if (!executionPlan.includes("implementation queue is empty")) {
  fail("gap execution plan must retain the implementation-queue-empty source verdict");
}
if (!ledger.includes("not the active implementation queue")) {
  fail("compact ledger must keep items_to_fix out of the active implementation queue");
}

if (!Array.isArray(queue.validationQueue)) fail("validationQueue must be an array");
if (!Array.isArray(queue.futureContractRows)) fail("futureContractRows must be an array");
assertRequiredIds(queue.validationQueue, REQUIRED_VALIDATION_IDS, "validationQueue");
assertRequiredIds(queue.futureContractRows, REQUIRED_FUTURE_IDS, "futureContractRows");

const scripts = packageJson.scripts ?? {};
if (typeof scripts !== "object") fail("package.json scripts must be an object");

for (const row of queue.validationQueue) {
  if (row == null || typeof row !== "object") fail("validation row must be an object");
  assertNonEmptyString(row.id, "validation row id");
  assertNonEmptyString(row.title, `${row.id}: title`);
  assertNonEmptyString(row.kind, `${row.id}: kind`);
  assertNonEmptyString(row.status, `${row.id}: status`);
  assertNonEmptyString(row.remaining, `${row.id}: remaining`);
  if (!ALLOWED_STATUSES.has(row.status)) fail(`${row.id}: invalid status ${row.status}`);
  if (row.status === "future-contract") fail(`${row.id}: future-contract rows belong in futureContractRows`);
  if (row.command != null) {
    assertNonEmptyString(row.command, `${row.id}: command`);
    assertCommandScriptsExist(row.command, scripts);
  } else if (row.status !== "evidence-needed" && row.status !== "decision-needed") {
    fail(`${row.id}: non-decision validation rows need a command`);
  }
  if (!Array.isArray(row.proofArtifacts)) fail(`${row.id}: proofArtifacts must be an array`);
  for (const artifact of row.proofArtifacts) await assertArtifact(artifact, row.id);
}

const learnedRow = queue.validationQueue.find((row) => row.id === "VQ-LEARNED-SYSTEMS");
if (learnedRow == null) fail("validationQueue missing VQ-LEARNED-SYSTEMS");
if (learnedCheckpointManifest.schema !== "vitrum.neural-denoiser.checkpoints.v1") {
  fail("learned checkpoint manifest schema mismatch");
}
const productionCheckpoint = learnedCheckpointManifest.productionCheckpoint ?? null;
if (productionCheckpoint === null) {
  if (learnedRow.status !== "provisioning-needed") {
    fail("VQ-LEARNED-SYSTEMS must remain provisioning-needed while productionCheckpoint is null");
  }
  if (!String(learnedRow.remaining).includes("Production neural checkpoint")) {
    fail("VQ-LEARNED-SYSTEMS remaining text must keep the production-checkpoint tail explicit");
  }
} else {
  if (typeof productionCheckpoint !== "string") {
    fail("learned checkpoint manifest productionCheckpoint must be null or a string");
  }
  const checkpoints = Array.isArray(learnedCheckpointManifest.checkpoints)
    ? learnedCheckpointManifest.checkpoints
    : [];
  const productionEntry = checkpoints.find((checkpoint) => checkpoint?.name === productionCheckpoint);
  if (productionEntry == null || productionEntry.role !== "production" || productionEntry.productionDefaultEligible !== true) {
    fail(`productionCheckpoint ${productionCheckpoint} must name a productionDefaultEligible production entry`);
  }
  if (learnedRow.status === "provisioning-needed") {
    fail("VQ-LEARNED-SYSTEMS must move off provisioning-needed once productionCheckpoint is populated");
  }
  if (!learnedRow.proofArtifacts.some((artifact) =>
    artifact?.path === "tools/neural-denoiser-training/quality-ab-production.json"
  )) {
    fail("VQ-LEARNED-SYSTEMS must cite quality-ab-production.json when a production checkpoint exists");
  }
}

const adjointRow = queue.validationQueue.find((row) => row.id === "VQ-ADJOINT-SCOPED-PATH-REPLAY");
if (adjointRow == null) fail("validationQueue missing VQ-ADJOINT-SCOPED-PATH-REPLAY");
for (const needle of [
  "path replay",
  "finite-difference",
  "transport",
  "visibility",
  "displacement geometry",
]) {
  if (!String(adjointRow.remaining).includes(needle)) {
    fail(`VQ-ADJOINT-SCOPED-PATH-REPLAY remaining text must include ${needle}`);
  }
}
if (!adjointRow.proofArtifacts.some((artifact) =>
  artifact?.path === "packages/pt-webgpu/src/inverse/inverseSession.ts"
)) {
  fail("VQ-ADJOINT-SCOPED-PATH-REPLAY must cite inverseSession.ts");
}
if (!adjointRow.proofArtifacts.some((artifact) =>
  artifact?.path === "packages/pt-webgpu/src/__tests__/inverseSession.test.ts"
)) {
  fail("VQ-ADJOINT-SCOPED-PATH-REPLAY must cite inverseSession.test.ts");
}

for (const row of queue.futureContractRows) {
  if (row == null || typeof row !== "object") fail("future-contract row must be an object");
  assertNonEmptyString(row.id, "future-contract row id");
  assertNonEmptyString(row.title, `${row.id}: title`);
  assertNonEmptyString(row.currentContract, `${row.id}: currentContract`);
  if (row.status !== "future-contract") fail(`${row.id}: status must be future-contract`);
}

const adjointFutureRow = queue.futureContractRows.find((row) => row.id === "FC-ADJOINT-FULL-PATH-PARITY");
if (adjointFutureRow == null) fail("futureContractRows missing FC-ADJOINT-FULL-PATH-PARITY");
if (!road.includes("Still OPEN for full-path parity")) {
  fail("Road adjoint full-path parity boundary disappeared without queue reconciliation");
}
for (const needle of [
  "scoped direct-light path replay",
  "finite-difference fallback diagnostics",
  "unsupported render regimes",
  "indirect paths",
]) {
  if (!String(adjointFutureRow.currentContract).includes(needle)) {
    fail(`FC-ADJOINT-FULL-PATH-PARITY currentContract must include ${needle}`);
  }
}

const counts = queue.validationQueue.reduce((acc, row) => {
  acc[row.status] = (acc[row.status] ?? 0) + 1;
  return acc;
}, /** @type {Record<string, number>} */ ({}));
console.log(
  `[road-to-100-validation-status] PASS (${queue.validationQueue.length} validation rows, ` +
  `${queue.futureContractRows.length} future-contract rows, statuses=${JSON.stringify(counts)})`,
);
