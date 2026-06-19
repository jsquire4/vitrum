#!/usr/bin/env -S deno run --sloppy-imports --allow-read
// @ts-check

const STATUS_PATH = "tools/behavioral-gate/cwbvh-parity-status.json";
const SCRIPT_PATH = "tools/behavioral-gate/cwbvh-parity-oracle.mjs";

function fail(message) {
  throw new Error(`[cwbvh-parity-proof-check] ${message}`);
}

const [statusText, scriptStat] = await Promise.all([
  Deno.readTextFile(STATUS_PATH),
  Deno.stat(SCRIPT_PATH),
]);
if (!scriptStat.isFile) fail(`${SCRIPT_PATH} is missing`);

const status = JSON.parse(statusText);
if (status.harness !== "cwbvh-parity-oracle") fail(`unexpected harness ${status.harness}`);
if (status.verdict !== "PASS") fail(`committed verdict is ${status.verdict}`);
if (status.command !== "npm run behavioral-gate:cwbvh -- --write-status") {
  fail(`unexpected command ${status.command}`);
}
if (!Number.isInteger(status.rayCount) || status.rayCount < 5) {
  fail(`rayCount ${status.rayCount} is too small`);
}
if (status.rootCount !== 2) {
  fail(`rootCount ${status.rootCount} does not prove multi-root CWBVH traversal`);
}
if (!Number.isInteger(status.nonzeroRoot) || status.nonzeroRoot <= 0) {
  fail(`nonzeroRoot ${status.nonzeroRoot} does not prove root remapping`);
}
if (!Number.isInteger(status.cwbvhNodeCount) || status.cwbvhNodeCount <= 1) {
  fail(`cwbvhNodeCount ${status.cwbvhNodeCount} does not prove multi-node traversal`);
}
if (!Number.isInteger(status.triangleCount) || status.triangleCount < 50) {
  fail(`triangleCount ${status.triangleCount} is too small`);
}
for (const key of [
  "closestNoSkip",
  "closestSkipGlass",
  "anyNoSkip",
  "anySkipGlass",
  "nonzeroRootClosest",
  "nonzeroRootAny",
]) {
  if (status.checks?.[key] !== true) fail(`missing check flag ${key}`);
}
if (!Array.isArray(status.mismatches) || status.mismatches.length !== 0) {
  fail(`mismatches are present: ${JSON.stringify(status.mismatches)}`);
}

console.log(
  `[cwbvh-parity-proof-check] PASS (${status.rayCount} rays, ${status.rootCount} roots, ${status.cwbvhNodeCount} CWBVH nodes, ${status.triangleCount} triangles)`,
);
