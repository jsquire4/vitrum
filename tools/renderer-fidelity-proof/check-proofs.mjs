#!/usr/bin/env -S deno run --sloppy-imports --allow-read
// @ts-check
// Verifies that renderer-fidelity supported pt-webgpu rows still cite committed
// runtime proof artifacts instead of drifting back into stale "queued" wording.

const MATRIX_PATH = "plan/renderer-fidelity-matrix.md";
const PLAYBOOK_PATH = "plan/fidelity-promotion-playbook.md";
const README_PATH = "README.md";
const ARCHITECTURE_PATH = "plan/library-architecture.md";
const HARDWARE_VALIDATION_PATH = "HARDWARE-VALIDATION-NEEDS.md";
const GAP_EXECUTION_PLAN_PATH = "plan/gap-closure-execution-plan.md";
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
    feature: "Cornell/core material fixture parity",
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

const PT_WEBGL2_MATERIAL_FURNACE_PROOFS = [
  {
    feature: "GGX multiscatter white furnace",
    files: [
      {
        path: "packages/pt-webgl2/src/glsl/shader/bsdf/__tests__/b9Multiscatter.test.ts",
        needles: [
          "WHITE-FURNACE",
          "without compensation, a rough conductor is dark",
          "with compensation, energy is recovered",
          "expect(Math.abs(r)).toBeLessThan(0.06)",
        ],
      },
      {
        path: "packages/pt-webgl2/src/glsl/shader/bsdf/ggx_functions.glsl.js",
        needles: [
          "ggxDirectionalAlbedo",
          "ggxAverageAlbedo",
          "ggxMultiscatter",
        ],
      },
      {
        path: "packages/pt-webgl2/src/glsl/shader/bsdf/bsdf_functions.glsl.js",
        needles: [
          "color += ggxMultiscatter(",
          "1.0 / 21.0",
        ],
      },
    ],
  },
  {
    feature: "thickness and SSS transport",
    files: [
      {
        path: "packages/pt-webgl2/src/glsl/composeTraceGlsl.test.ts",
        needles: [
          "D10: SSS free-flight helper is defined before the SSS sample path uses it",
          "D10: SSS consumes packed sigmaS and derives albedo in shader",
          "sampleExponentialDistance( rand( 17 ), sigmaTMajorant, 1e6 )",
          "vec3 sigmaS = max( surf.sssSigmaS, vec3( 0.0 ) );",
        ],
      },
      {
        path: "packages/pt-webgl2/src/scene/materialsTexture.test.ts",
        needles: [
          "scatteringCoefficientRGB packs per-channel sigmaS override and majorant sigmaT",
          "scatteringCoefficientRGB alone activates translucent SSS and packs sigmaS",
          "packs thicknessMap layer, thickness scalar, UV1 bit, transform, and wrap mode",
        ],
      },
      {
        path: "packages/pt-webgl2/src/glsl/render/get_surface_record_function.glsl.js",
        needles: [
          "material.thicknessMap != - 1",
          "material.thicknessMapTransform * vec3( MAP_UV( 20u ), 1 )",
          "surf.sssSigmaS = material.sssSigmaS;",
        ],
      },
      {
        path: "packages/pt-webgl2/src/glsl/render/attenuate_hit_function.glsl.js",
        needles: [
          "material.thickness > 0.0 || material.thicknessMap != - 1",
          "material.thicknessMapTransform * vec3( ATTENUATE_MAP_UV( 20u ), 1 )",
          "attenuationDist = min( attenuationDist, max( attenuationThickness, 0.0 ) );",
          "transmissionAttenuationThroughput(",
        ],
      },
    ],
  },
  {
    feature: "procedural sky bake and sampling",
    files: [
      {
        path: "packages/pt-webgl2/src/capabilities.ts",
        needles: [
          "supportedEnvironmentKinds: new Set<SceneEnvironment['kind']>(['none', 'hdri', 'procedural-sky'])",
        ],
      },
      {
        path: "packages/pt-webgl2/src/scene/equirectHdrInfo.ts",
        needles: [
          "if (env.kind === 'procedural-sky')",
          "bakePreethamSkyEquirect({",
          "source = { width: baked.width, height: baked.height, data: baked.texels };",
          "const marginalData = new Float32Array(height * 4);",
          "const conditionalData = new Float32Array(pixelCount * 4);",
          "conditional: { data: conditionalData, width, height },",
        ],
      },
      {
        path: "packages/pt-webgl2/src/scene/equirectHdrInfo.test.ts",
        needles: [
          "bakes procedural-sky environments into the equirect HDRI path",
          "honors zero procedural-sky intensity as a black environment",
          "places the procedural-sky maximum near the authored sun direction",
        ],
      },
    ],
  },
  {
    feature: "emissive-map panels and mesh-area MIS",
    files: [
      {
        path: "packages/pt-webgl2/src/scene/meshAreaLights.test.ts",
        needles: [
          "subdivides CPU-readable emissiveMap implicit triangle lights with UV-local radiance",
          "clips CPU-readable emissiveMap UV footprints to exact texel cells",
          "subdivides explicit mesh-area triangle lights through the referenced material emissiveMap",
        ],
      },
      {
        path: "packages/pt-webgl2/src/scene/meshAreaMis.test.ts",
        needles: [
          "forward pdf equals the sample pdf for the same emitted-power density",
          "uses the textured implicit-emitter pack total power for forward/sample pdf parity",
          "neeForwardPdf(distSq, cosLight, out.totalEmissivePower",
        ],
      },
      {
        path: "packages/pt-webgl2/src/glsl/shader/sampling/light_sampling_functions.glsl.js",
        needles: [
          "LightRecord sampleMeshAreaLight(",
          "float meshAreaLightForwardPdf(",
          "float uPick = ruv.x * totalEmissivePower;",
          "float areaDensity = max( tri.power, 0.0 ) / ( max( tri.area, EPSILON ) * totalEmissivePower );",
        ],
      },
      {
        path: "packages/pt-webgl2/src/glsl/render/get_surface_record_function.glsl.js",
        needles: [
          "if ( useTextures && material.emissiveMap != - 1 )",
          "material.emissiveMapTransform * vec3( MAP_UV( 4u ), 1 )",
          "emission *= sampleMaterialTexture(",
        ],
      },
    ],
  },
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

/**
 * @param {{
 *   feature: string,
 *   files: Array<{ path: string, needles: string[] }>,
 * }} proof
 */
async function assertPtWebgl2MaterialFurnaceProof(proof) {
  for (const file of proof.files) {
    const text = await readText(file.path);
    for (const needle of file.needles) {
      if (!text.includes(needle)) {
        fail(`${proof.feature}: ${file.path} missing proof needle: ${needle}`);
      }
    }
  }
}

const matrix = await readText(MATRIX_PATH);
const playbook = await readText(PLAYBOOK_PATH);
const readme = await readText(README_PATH);
const architecture = await readText(ARCHITECTURE_PATH);
const hardwareValidation = await readText(HARDWARE_VALIDATION_PATH);
const gapExecutionPlan = await readText(GAP_EXECUTION_PLAN_PATH);
const ptWebgl2BrowserStatus = JSON.parse(await readText(PT_WEBGL2_BROWSER_STATUS_PATH));

if (!matrix.includes("| Feature | pt-webgl2 (WebGL2) | pt-webgpu full tier (WebGPU) |")) {
  fail("renderer fidelity matrix must label the pt-webgpu column as full-tier proof");
}
if (matrix.includes("| Feature | pt-webgl2 (WebGL2) | pt-webgpu (WebGPU) |")) {
  fail("renderer fidelity matrix must not use the stale unqualified pt-webgpu column heading");
}

for (const proof of PT_WEBGPU_SUPPORTED_ROWS) {
  const row = findMatrixRow(matrix, proof.feature);
  assertPtWebgpuSupported(row, proof.feature);
  for (const needle of proof.matrixNeedles) {
    if (!row.includes(needle)) fail(`${proof.feature}: matrix row missing runtime proof text: ${needle}`);
  }
  for (const path of proof.goldenPaths ?? []) await assertPng(path);
  if (proof.dznStatus) await assertDznStatus(proof.dznStatus, proof.feature);
}

for (const proof of PT_WEBGL2_MATERIAL_FURNACE_PROOFS) {
  await assertPtWebgl2MaterialFurnaceProof(proof);
}
if (!gapExecutionPlan.includes("pt-webgl2 material furnace | pt-webgl2 | Source/oracle proof is guarded by `npm run renderer-fidelity-proof-check`")) {
  fail("gap closure execution plan must classify pt-webgl2 material furnace as source/oracle-proof guarded, with remaining A/B called out separately");
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

if (readme.includes("clearcoat/sheen unsupported")) {
  fail("README still contains the stale pt-webgl2 clearcoat/sheen unsupported claim");
}
if (!readme.includes("pt-webgpu (WebGPU)")) {
  fail("README capability matrix must include the pt-webgpu backend column");
}
if (architecture.includes("experimental backend, evolving toward Phase 7 goals")) {
  fail("library architecture still labels the whole pt-webgpu package as the stale experimental backend");
}
if (!architecture.includes("peer WebGPU PT backend; row-level fidelity tiers")) {
  fail("library architecture must describe pt-webgpu with row-level fidelity tiers");
}
if (hardwareValidation.includes("WSL2 with SwiftShader only") ||
    hardwareValidation.includes("cannot validate any of the items below")) {
  fail("hardware validation doc still contains the stale SwiftShader-only all-blocked premise");
}

console.log(
  `[renderer-fidelity-proof-check] PASS (${PT_WEBGPU_SUPPORTED_ROWS.length} pt-webgpu supported rows verified; ${PT_WEBGL2_MATERIAL_FURNACE_PROOFS.length} pt-webgl2 material-furnace source/oracle proof groups verified; pt-webgl2 browser-promotion guard checked)`,
);
