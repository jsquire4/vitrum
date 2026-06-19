#!/usr/bin/env -S deno run --sloppy-imports --allow-read
// @ts-check

const STATUS_PATH = "tools/behavioral-gate/behavioral-gate-cwbvh-status.json";

function fail(message) {
  throw new Error(`[cwbvh-renderer-parity-proof-check] ${message}`);
}

function sameJson(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

const status = JSON.parse(await Deno.readTextFile(STATUS_PATH));
if (status.harness !== "behavioral-gate") fail(`unexpected harness ${status.harness}`);
if (status.verdict !== "PASS") fail(`committed verdict is ${status.verdict}`);
if (status.command !== "npm run behavioral-gate -- --filter cwbvh --require-full-tier") {
  fail(`unexpected command ${status.command}`);
}
if (status.filter !== "cwbvh") fail(`unexpected filter ${status.filter}`);
if (status.exitStatus !== 0) fail(`exitStatus must be 0, got ${status.exitStatus}`);
if (status.summary?.totalConfigs !== 1 || status.summary?.failures !== 0) {
  fail(`unexpected summary ${JSON.stringify(status.summary)}`);
}

const config = status.configs?.find((entry) => entry.label === "pt/cwbvh-binary-parity");
if (config == null) fail("missing pt/cwbvh-binary-parity config");
if (config.verdict !== "PASS") fail(`config verdict is ${config.verdict}`);
if (config.rawStatus !== "OK") fail(`config rawStatus is ${config.rawStatus}`);
if (config.tier !== "full") fail(`config tier is ${config.tier}`);
if (config.gpuErrors !== 0) fail(`gpuErrors must be 0, got ${config.gpuErrors}`);
if (config.nan !== false) fail(`nan must be false, got ${config.nan}`);
if (!(config.luminance >= 0.005)) fail(`luminance below bound: ${config.luminance}`);
if (config.cwbvhParityKind !== "binary") fail(`parity kind is ${config.cwbvhParityKind}`);
if (config.cwbvhParityRmse > 1) fail(`RMSE exceeds bound: ${config.cwbvhParityRmse}`);
if (config.cwbvhParityMeanAbs > 0.5) fail(`meanAbs exceeds bound: ${config.cwbvhParityMeanAbs}`);
if (config.cwbvhParityMaxAbs > 8) fail(`maxAbs exceeds bound: ${config.cwbvhParityMaxAbs}`);
if (!sameJson(config.cwbvhParityThresholds, { maxRmse: 1, maxMeanAbs: 0.5, maxAbs: 8 })) {
  fail(`thresholds mismatch: ${JSON.stringify(config.cwbvhParityThresholds)}`);
}

console.log(
  `[cwbvh-renderer-parity-proof-check] PASS (rmse=${config.cwbvhParityRmse}, meanAbs=${config.cwbvhParityMeanAbs}, maxAbs=${config.cwbvhParityMaxAbs})`,
);
