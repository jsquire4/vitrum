#!/usr/bin/env -S deno run --sloppy-imports --allow-read
// @ts-check
// Verifies that renderer-fidelity supported pt-webgpu rows still cite committed
// runtime proof artifacts instead of drifting back into stale "queued" wording.

const MATRIX_PATH = "plan/renderer-fidelity-matrix.md";
const PLAYBOOK_PATH = "plan/fidelity-promotion-playbook.md";
const PT_WEBGL2_BROWSER_STATUS_PATH = "tools/gltf-browser-proof/pt-webgl2-real-status.json";

const PT_WEBGPU_SUPPORTED_ROWS = [
  {
    feature: "Hero-wavelength + CMF accumulation",
    matrixNeedles: [
      "pt-webgpu dzn (RTX 4090) spectral ON/OFF A/B",
      "tools/reference-renders/baseline/ptwgpu-spectral-hero.png",
    ],
    goldenPaths: ["tools/reference-renders/baseline/ptwgpu-spectral-hero.png"],
    dznStatus: {
      path: "tools/behavioral-gate/behavioral-gate-dzn-spectral-status.json",
      labels: ["pt/spectral"],
    },
  },
  {
    feature: "Spectral Beer–Lambert (packed μ)",
    matrixNeedles: [
      "pt-webgpu dzn (RTX 4090) μ-curve present-vs-absent A/B",
      "green-peaked packed μ shifts transmitted light magenta",
    ],
    dznStatus: {
      path: "tools/behavioral-gate/behavioral-gate-dzn-spectral-status.json",
      labels: ["pt/spectral"],
    },
  },
  {
    feature: "Multi-layer thin film TMM",
    matrixNeedles: [
      "pt-webgpu dzn (RTX 4090) hue-vs-angle A/B",
      "tools/reference-renders/baseline/ptwgpu-thinfilm-angle.png",
    ],
    goldenPaths: ["tools/reference-renders/baseline/ptwgpu-thinfilm-angle.png"],
  },
  {
    feature: "Cauchy dispersion",
    matrixNeedles: [
      "pt-webgpu dzn (RTX 4090) Abbe-set-vs-absent A/B",
      "tools/reference-renders/baseline/ptwgpu-cauchy-dispersion.png",
    ],
    goldenPaths: ["tools/reference-renders/baseline/ptwgpu-cauchy-dispersion.png"],
    dznStatus: {
      path: "tools/behavioral-gate/behavioral-gate-dzn-spectral-status.json",
      labels: ["pt/spectral"],
    },
  },
  {
    feature: "Layered front/back + transmission MIS",
    matrixNeedles: [
      "pt-webgpu dzn (RTX 4090) front/back A/B",
      "tools/reference-renders/baseline/ptwgpu-layered-front.png",
    ],
    goldenPaths: ["tools/reference-renders/baseline/ptwgpu-layered-front.png"],
  },
  {
    feature: "SSS / translucent panels",
    matrixNeedles: [
      "pt-webgpu dzn (RTX 4090) mixed-panel toggle A/B",
      "tools/reference-renders/baseline/ptwgpu-sss-mixed-panels.png",
    ],
    goldenPaths: ["tools/reference-renders/baseline/ptwgpu-sss-mixed-panels.png"],
  },
  {
    feature: "Multi emitter direct lighting",
    matrixNeedles: [
      "dzn (RTX 4090) baseline `tools/reference-renders/baseline/cornell-manylights.png`",
    ],
    goldenPaths: ["tools/reference-renders/baseline/cornell-manylights.png"],
    dznStatus: {
      path: "tools/behavioral-gate/behavioral-gate-dzn-light-status.json",
      labels: ["pt/point-light", "pt/disc-light", "pt/spot-light"],
    },
  },
  {
    feature: "Material fields parity (cornell)",
    matrixNeedles: [
      "strict-hash re-capture == committed `tools/reference-renders/baseline/ptwgpu-parity-material-fields.png`",
      "PSNR 999 dB",
    ],
    goldenPaths: ["tools/reference-renders/baseline/ptwgpu-parity-material-fields.png"],
  },
  {
    feature: "Caustic strategies",
    matrixNeedles: [
      "MNEE GPU-validated vs DETERMINISTIC references",
      "tools/reference-renders/baseline/mnee-glass-slab.png",
    ],
    goldenPaths: ["tools/reference-renders/baseline/mnee-glass-slab.png"],
    dznStatus: {
      path: "tools/behavioral-gate/behavioral-gate-dzn-caustic-status.json",
      labels: ["pt/caustic-manifold", "pt/caustic-photon"],
    },
  },
  {
    feature: "BDPT (eye↔light connections)",
    matrixNeedles: [
      "pt-webgpu GPU-validated (V18/V25)",
      "tools/reference-renders/baseline/cornell-bdpt-on.png",
    ],
    goldenPaths: ["tools/reference-renders/baseline/cornell-bdpt-on.png"],
    dznStatus: {
      path: "tools/behavioral-gate/behavioral-gate-dzn-bdpt-status.json",
      labels: ["pt/bdpt", "pt/spectral+bdpt"],
    },
  },
];

const PLAYBOOK_FORBIDDEN_STALE_NEEDLES = [
  "Native lavapipe capture adapter not yet wired (§1.0)",
  "strict-hash re-capture on full-tier adapter is the only step",
  "no emitter-count-only baseline committed yet",
  "no dedicated BDPT gap-closure scenario exists",
];

/** @param {string} path */
function repoUrl(path) {
  return new URL(`../../${path}`, import.meta.url);
}

/** @param {string} message */
function fail(message) {
  throw new Error(`[renderer-fidelity-proof-check] ${message}`);
}

/** @param {string} path */
async function readText(path) {
  return await Deno.readTextFile(repoUrl(path));
}

/** @param {string} path */
async function assertPng(path) {
  const url = repoUrl(path);
  const stat = await Deno.stat(url);
  if (!stat.isFile || stat.size <= 8) fail(`${path} is missing or empty`);
  const header = await Deno.readFile(url);
  if (header[0] !== 0x89 || header[1] !== 0x50 || header[2] !== 0x4e || header[3] !== 0x47) {
    fail(`${path} is not a PNG`);
  }
}

/**
 * @param {string} matrix
 * @param {string} feature
 */
function findMatrixRow(matrix, feature) {
  const row = matrix.split("\n").find((line) => line.startsWith(`| ${feature} |`));
  if (!row) fail(`renderer fidelity matrix missing row for ${feature}`);
  return row;
}

/**
 * @param {string} row
 * @param {string} feature
 */
function assertPtWebgpuSupported(row, feature) {
  const columns = row.split(" | ");
  const ptWebgpuColumn = columns[2];
  if (ptWebgpuColumn !== "supported") {
    fail(`${feature}: pt-webgpu matrix grade must be supported, got ${ptWebgpuColumn ?? "<missing>"}`);
  }
}

/** @param {string} matrix */
function featureRows(matrix) {
  return matrix
    .split("\n")
    .filter((line) => {
      if (!line.startsWith("| ")) return false;
      if (line.startsWith("| Feature |")) return false;
      if (line.startsWith("|---------")) return false;
      return line.split(" | ").length >= 6;
    });
}

/**
 * @param {{ path: string, labels: string[] }} status
 * @param {string} feature
 */
async function assertDznStatus(status, feature) {
  const payload = JSON.parse(await readText(status.path));
  if (payload.harness !== "behavioral-gate:dzn") fail(`${feature}: ${status.path} harness mismatch`);
  if (payload.verdict !== "PASS") fail(`${feature}: ${status.path} verdict must be PASS`);
  if (payload.summary?.failures !== 0) fail(`${feature}: ${status.path} must have zero failures`);
  const configs = new Map((payload.configs ?? []).map((config) => [config.label, config]));
  for (const label of status.labels) {
    const config = configs.get(label);
    if (!config) fail(`${feature}: ${status.path} missing config ${label}`);
    if (config.verdict !== "PASS") fail(`${feature}: ${label} verdict must be PASS`);
    if (config.rawStatus !== "OK") fail(`${feature}: ${label} rawStatus must be OK`);
    if (config.tier !== "full") fail(`${feature}: ${label} must run on pt-webgpu full tier`);
    if (config.gpuErrors !== 0) fail(`${feature}: ${label} must have zero GPU errors`);
    if (config.nan !== false) fail(`${feature}: ${label} must report nan=false`);
  }
}

const matrix = await readText(MATRIX_PATH);
const playbook = await readText(PLAYBOOK_PATH);
const ptWebgl2BrowserStatus = JSON.parse(await readText(PT_WEBGL2_BROWSER_STATUS_PATH));

for (const proof of PT_WEBGPU_SUPPORTED_ROWS) {
  const row = findMatrixRow(matrix, proof.feature);
  assertPtWebgpuSupported(row, proof.feature);
  for (const needle of proof.matrixNeedles) {
    if (!row.includes(needle)) fail(`${proof.feature}: matrix row missing runtime proof text: ${needle}`);
  }
  for (const path of proof.goldenPaths ?? []) await assertPng(path);
  if (proof.dznStatus) await assertDznStatus(proof.dznStatus, proof.feature);
}

if (ptWebgl2BrowserStatus.backend !== "pt-webgl2") {
  fail(`${PT_WEBGL2_BROWSER_STATUS_PATH} backend mismatch`);
}
if (ptWebgl2BrowserStatus.verdict === "HOST-BLOCKED") {
  for (const row of featureRows(matrix)) {
    const columns = row.split(" | ");
    const feature = columns[0].slice(2);
    const ptWebgl2Grade = columns[1];
    if (ptWebgl2Grade === "supported") {
      fail(
        `${feature}: pt-webgl2 must not be marked supported while ` +
        `${PT_WEBGL2_BROWSER_STATUS_PATH} is HOST-BLOCKED`,
      );
    }
  }
} else if (ptWebgl2BrowserStatus.verdict !== "PASS") {
  fail(`${PT_WEBGL2_BROWSER_STATUS_PATH} verdict must be PASS or HOST-BLOCKED`);
}

for (const staleNeedle of PLAYBOOK_FORBIDDEN_STALE_NEEDLES) {
  if (playbook.includes(staleNeedle)) {
    fail(`fidelity promotion playbook still contains stale blocker: ${staleNeedle}`);
  }
}

console.log(
  `[renderer-fidelity-proof-check] PASS (${PT_WEBGPU_SUPPORTED_ROWS.length} pt-webgpu supported rows verified; pt-webgl2 browser-promotion guard checked)`,
);
