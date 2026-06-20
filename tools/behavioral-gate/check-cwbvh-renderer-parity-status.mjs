#!/usr/bin/env -S deno run --sloppy-imports --allow-read
// @ts-check

const STATUS_PATH = "tools/behavioral-gate/behavioral-gate-cwbvh-status.json";

function fail(message) {
  throw new Error(`[cwbvh-renderer-parity-proof-check] ${message}`);
}

function sameJson(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function requireFinite(config, key) {
  const value = config[key];
  if (!Number.isFinite(value)) fail(`${key} must be finite, got ${value}`);
  return value;
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
if (config.cwbvhPerfKind !== "same-scene") fail(`perf kind is ${config.cwbvhPerfKind}`);
const binaryMs = requireFinite(config, "cwbvhBinaryRenderMs");
const cwbvhMs = requireFinite(config, "cwbvhRenderMs");
const ratio = requireFinite(config, "cwbvhRenderMsRatio");
const binaryMemory = requireFinite(config, "cwbvhBinaryMemoryBytes");
const cwbvhMemory = requireFinite(config, "cwbvhMemoryBytes");
const memoryDelta = requireFinite(config, "cwbvhMemoryBytesDelta");
const binaryScene = requireFinite(config, "cwbvhBinarySceneBytes");
const cwbvhScene = requireFinite(config, "cwbvhSceneBytes");
const sceneDelta = requireFinite(config, "cwbvhSceneBytesDelta");
if (binaryMs <= 0 || cwbvhMs <= 0 || ratio <= 0) {
  fail(`render timings must be positive, got binaryMs=${binaryMs}, cwbvhMs=${cwbvhMs}, ratio=${ratio}`);
}
if (binaryMemory <= 0 || cwbvhMemory <= 0 || binaryScene <= 0 || cwbvhScene <= 0) {
  fail(`memory figures must be positive, got binary=${binaryMemory}, cwbvh=${cwbvhMemory}, binaryScene=${binaryScene}, cwbvhScene=${cwbvhScene}`);
}
if (memoryDelta !== 0 || sceneDelta !== 0) {
  fail(`expected no opt-in memory delta for the current uploaded mirror layout, got total=${memoryDelta}, scene=${sceneDelta}`);
}

console.log(
  `[cwbvh-renderer-parity-proof-check] PASS (rmse=${config.cwbvhParityRmse}, meanAbs=${config.cwbvhParityMeanAbs}, maxAbs=${config.cwbvhParityMaxAbs}, binaryMs=${binaryMs}, cwbvhMs=${cwbvhMs}, memoryDelta=${memoryDelta})`,
);
