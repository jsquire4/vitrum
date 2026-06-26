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
const PT_WEBGL2_BROWSER_CANVAS_FIRST_STATUS_PATH = "tools/gltf-browser-proof/pt-webgl2-real-canvas-first-status.json";
const PT_WEBGL2_BROWSER_MANIFEST_PATH = "tools/reference-renders/gltf-real-browser-pt-webgl2/manifest.json";
const CHECKER_PATH = "tools/renderer-fidelity-proof/check-proofs.mjs";
const PROMOTION_STATUS_PATH = "tools/renderer-fidelity-proof/promotion-status.json";
const PT_RADIOMETRIC_PROMOTION_STATUS_PATH = "tools/radiometric-ab/pt-promotion-status.json";
const BDPT_RADIOMETRIC_RESULT_PATH = "tools/radiometric-ab/results-bdpt.json";

/**
 * @typedef {{
 *   path: string,
 *   sha256: string,
 *   width: number,
 *   height: number,
 * }} GoldenPngProof
 */

const PT_WEBGPU_SUPPORTED_ROWS = [
  {
    feature: "Hero-wavelength + CMF accumulation",
    matrixNeedles: [
      "pt-webgpu dzn (RTX 4090) spectral ON/OFF A/B",
      "tools/reference-renders/baseline/ptwgpu-spectral-hero.png",
    ],
    goldenPaths: [
      goldenPng(
        "tools/reference-renders/baseline/ptwgpu-spectral-hero.png",
        "db9854d670168a4528d08a532acda03b07b6368389cd106227ea45237f137655",
        512,
        512,
      ),
    ],
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
    goldenPaths: [
      goldenPng(
        "tools/reference-renders/baseline/ptwgpu-thinfilm-angle.png",
        "5a33d8322c677830b5698fb6fa41b602a2764ca94babd74a782d3d034d3e46f7",
        640,
        640,
      ),
    ],
  },
  {
    feature: "Cauchy dispersion",
    matrixNeedles: [
      "pt-webgpu dzn (RTX 4090) Abbe-set-vs-absent A/B",
      "tools/reference-renders/baseline/ptwgpu-cauchy-dispersion.png",
    ],
    goldenPaths: [
      goldenPng(
        "tools/reference-renders/baseline/ptwgpu-cauchy-dispersion.png",
        "8f1c1e99503d76797ce3cbe32da825bb8f110df416c9311ab568f5669f71f0a7",
        512,
        512,
      ),
    ],
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
    goldenPaths: [
      goldenPng(
        "tools/reference-renders/baseline/ptwgpu-layered-front.png",
        "e14e99ef16fe1ca4f4c18793be0c454434542a941c2bfec74f867752bc1d4f1c",
        512,
        512,
      ),
    ],
  },
  {
    feature: "SSS / translucent panels",
    matrixNeedles: [
      "pt-webgpu dzn (RTX 4090) mixed-panel toggle A/B",
      "tools/reference-renders/baseline/ptwgpu-sss-mixed-panels.png",
    ],
    goldenPaths: [
      goldenPng(
        "tools/reference-renders/baseline/ptwgpu-sss-mixed-panels.png",
        "8bebc9a1c83e922a1b1c811f4e175db35b8a5d1e980b70df71706483227003c2",
        512,
        512,
      ),
    ],
  },
  {
    feature: "Multi emitter direct lighting",
    matrixNeedles: [
      "dzn (RTX 4090) baseline `tools/reference-renders/baseline/cornell-manylights.png`",
    ],
    goldenPaths: [
      goldenPng(
        "tools/reference-renders/baseline/cornell-manylights.png",
        "c857ba59494da9db08601e1e1faac764a8c9f4c37203e9d513aabed03baa9bf1",
        512,
        512,
      ),
    ],
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
    goldenPaths: [
      goldenPng(
        "tools/reference-renders/baseline/ptwgpu-parity-material-fields.png",
        "df76229af4aab72dddc00567d21a05a50203292b0a16922085d97bdc8c2cf721",
        1280,
        720,
      ),
    ],
  },
  {
    feature: "Caustic strategies",
    matrixNeedles: [
      "MNEE GPU-validated vs DETERMINISTIC references",
      "tools/reference-renders/baseline/mnee-glass-slab.png",
    ],
    goldenPaths: [
      goldenPng(
        "tools/reference-renders/baseline/mnee-glass-slab.png",
        "60117577855fc849b9c2c4be276012990c90a58d59a156770747f69df13f95a1",
        1280,
        720,
      ),
    ],
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
      "`bdpt:true` safe default is endpoint-only",
      "opt-in multi-vertex (`maxLightBounces>1`) remains research-only",
      "not-weighted-against-regular-eye-path-strategy",
    ],
    goldenPaths: [
      goldenPng(
        "tools/reference-renders/baseline/cornell-bdpt-on.png",
        "229cc7ebb31ec1dfe9f9d5d6564147406941ba8fc51ab1b30069f772bf1c6a19",
        512,
        512,
      ),
    ],
    dznStatus: {
      path: "tools/behavioral-gate/behavioral-gate-dzn-bdpt-status.json",
      labels: ["pt/bdpt", "pt/spectral+bdpt"],
    },
  },
];

const PT_WEBGL2_EXPECTED_ROWS = [
  {
    feature: "Hero-wavelength + CMF accumulation",
    grade: "experimental",
    rowNeedles: [
      "pt-webgl2: runtime A/B capture pending",
      "row stays `experimental` until pt-webgl2 spectral runtime A/B has a committed reference",
    ],
  },
  {
    feature: "Spectral Beer–Lambert (packed μ)",
    grade: "experimental",
    rowNeedles: [
      "pt-webgl2: runtime A/B capture pending",
      "remains unpromoted without a visual Beer-Lambert reference",
    ],
  },
  {
    feature: "Multi-layer thin film TMM",
    grade: "experimental",
    rowNeedles: [
      "pt-webgl2: runtime A/B capture pending",
      "row remains `experimental` pending angle/hue A/B",
    ],
  },
  {
    feature: "Cauchy dispersion",
    grade: "experimental",
    rowNeedles: [
      "pt-webgl2: runtime A/B capture pending",
      "needs dispersion visual promotion",
    ],
  },
  {
    feature: "Layered front/back + transmission MIS",
    grade: "approximate",
    rowNeedles: [
      "pt-webgl2: runtime A/B capture pending",
      "row remains `approximate` until runtime visual A/B promotes pt-webgl2",
    ],
  },
  {
    feature: "SSS / translucent panels",
    grade: "approximate",
    rowNeedles: [
      "pt-webgl2: runtime A/B capture pending",
      "still `approximate` until the scalar-majorant WebGL single-scatter model has visual promotion",
    ],
  },
  {
    feature: "Multi emitter direct lighting",
    grade: "approximate",
    rowNeedles: [
      "Grade remains `approximate` until unequal-power/mixed-emitter visual A/B promotion",
    ],
  },
  {
    feature: "Cornell/core material fixture parity",
    grade: "experimental",
    rowNeedles: [
      "pt-webgl2: browser/runtime material-fidelity A/B capture pending",
      "row stays `experimental` until a real browser/WebGL2 runtime capture promotes it",
    ],
  },
  {
    feature: "Caustic strategies",
    grade: "approximate",
    rowNeedles: [
      "grade 'approximate'",
      "phenomenological GLSL path, NOT Newton-solve MNEE",
    ],
  },
  {
    feature: "SVGF-real denoiser",
    grade: "unsupported",
    rowNeedles: [
      "Converged tracer → `oidn-final`; SVGF is real-time-only",
      "Both converged backends warn on `'svgf-real'`",
    ],
  },
  {
    feature: "BDPT (eye↔light connections)",
    grade: "approximate",
    rowNeedles: [
      "Grade remains `approximate` until pt-webgl2 BDPT has visual A/B promotion",
    ],
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

/**
 * @param {string} path
 * @param {string} sha256
 * @param {number} width
 * @param {number} height
 * @returns {GoldenPngProof}
 */
function goldenPng(path, sha256, width, height) {
  return { path, sha256, width, height };
}

/**
 * @param {string} message
 * @returns {never}
 */
function fail(message) {
  throw new Error(`[renderer-fidelity-proof-check] ${message}`);
}

/** @param {string} path */
async function readText(path) {
  return await Deno.readTextFile(repoUrl(path));
}

/** @param {Uint8Array} bytes */
async function sha256Hex(bytes) {
  const owned = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(owned).set(bytes);
  const digest = await crypto.subtle.digest("SHA-256", owned);
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

/** @param {string} path */
async function sha256RepoPath(path) {
  return await sha256Hex(await Deno.readFile(repoUrl(path)));
}

/** @param {string[]} sourceStatuses */
async function rendererFidelityPromotionProvenance(sourceStatuses) {
  return {
    schema: "vitrum.renderer-fidelity.promotion-provenance.v1",
    checkerPath: CHECKER_PATH,
    checkerSha256: await sha256RepoPath(CHECKER_PATH),
    statusPath: PROMOTION_STATUS_PATH,
    sourceStatuses,
    sourceStatusSha256: await Promise.all(sourceStatuses.map(async (path) => ({
      path,
      sha256: await sha256RepoPath(path),
    }))),
    generatedBy: "vitrum renderer fidelity proof ledger",
  };
}

/** @param {GoldenPngProof} proof */
async function assertPng(proof) {
  const url = repoUrl(proof.path);
  const stat = await Deno.stat(url);
  if (!stat.isFile || stat.size <= 24) fail(`${proof.path} is missing or empty`);
  const bytes = await Deno.readFile(url);
  if (bytes[0] !== 0x89 || bytes[1] !== 0x50 || bytes[2] !== 0x4e || bytes[3] !== 0x47) {
    fail(`${proof.path} is not a PNG`);
  }
  const width =
    bytes[16] * 0x1000000 +
    bytes[17] * 0x10000 +
    bytes[18] * 0x100 +
    bytes[19];
  const height =
    bytes[20] * 0x1000000 +
    bytes[21] * 0x10000 +
    bytes[22] * 0x100 +
    bytes[23];
  if (width !== proof.width || height !== proof.height) {
    fail(`${proof.path} size ${width}x${height} differs from expected ${proof.width}x${proof.height}`);
  }
  const hash = await sha256Hex(bytes);
  if (hash !== proof.sha256) {
    fail(`${proof.path} sha256 ${hash} differs from expected ${proof.sha256}`);
  }
}

/**
 * @param {string} matrix
 * @param {string} feature
 * @returns {string}
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

/**
 * @param {string} matrix
 */
function assertEveryPtWebgpuSupportedRowHasProof(matrix) {
  /** @type {Set<string>} */
  const proofFeatures = new Set();
  for (const proof of PT_WEBGPU_SUPPORTED_ROWS) {
    if (proofFeatures.has(proof.feature)) {
      fail(`duplicate pt-webgpu supported proof entry for ${proof.feature}`);
    }
    proofFeatures.add(proof.feature);
  }

  for (const row of featureRows(matrix)) {
    const columns = row.split(" | ");
    const feature = columns[0].slice(2);
    const ptWebgpuGrade = columns[2];
    if (ptWebgpuGrade === "supported" && !proofFeatures.has(feature)) {
      fail(
        `${feature}: pt-webgpu is marked supported in ${MATRIX_PATH} but has no ` +
        "runtime proof entry in PT_WEBGPU_SUPPORTED_ROWS",
      );
    }
  }
}

/**
 * @param {string} matrix
 */
function assertPtWebgl2ExpectedRows(matrix) {
  /** @type {Set<string>} */
  const expectedFeatures = new Set();
  for (const proof of PT_WEBGL2_EXPECTED_ROWS) {
    if (expectedFeatures.has(proof.feature)) {
      fail(`duplicate pt-webgl2 expected-grade entry for ${proof.feature}`);
    }
    expectedFeatures.add(proof.feature);

    const row = findMatrixRow(matrix, proof.feature);
    const columns = row.split(" | ");
    const ptWebgl2Grade = columns[1];
    if (ptWebgl2Grade !== proof.grade) {
      fail(`${proof.feature}: pt-webgl2 matrix grade must be ${proof.grade}, got ${ptWebgl2Grade ?? "<missing>"}`);
    }
    for (const needle of proof.rowNeedles) {
      if (!row.includes(needle)) {
        fail(`${proof.feature}: pt-webgl2 row missing proof/truthfulness text: ${needle}`);
      }
    }
  }
}

/**
 * @param {string} matrix
 * @returns {string[]}
 */
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
  if (!Array.isArray(payload.configs)) fail(`${feature}: ${status.path} configs must be an array`);
  /** @type {Map<string, Record<string, any>>} */
  const configs = new Map();
  for (const config of /** @type {Record<string, any>[]} */ (payload.configs)) {
    configs.set(String(config.label), config);
  }
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
 * @param {readonly Record<string, any>[] | undefined} items
 * @param {string} key
 * @param {string} owner
 */
function byKey(items, key, owner) {
  /** @type {Map<string, Record<string, any>>} */
  const map = new Map();
  for (const item of items ?? []) {
    const value = item[key];
    if (typeof value !== "string" || value.length === 0) fail(`${owner}: invalid ${key}`);
    if (map.has(value)) fail(`${owner}: duplicate ${key}: ${value}`);
    map.set(value, item);
  }
  return map;
}

/**
 * @param {Record<string, any>} row
 * @param {Record<string, any>} manifestAsset
 */
async function assertPtWebgl2BrowserPassRow(row, manifestAsset) {
  if (row.verdict !== "PASS") fail(`${row.assetId}: browser row must be PASS when top-level browser status is PASS`);
  if (row.harness !== "gltf-browser-proof:pt-webgl2-real") fail(`${row.assetId}: browser row harness mismatch`);
  if (row.backend !== "pt-webgl2") fail(`${row.assetId}: browser row backend mismatch`);
  if (row.kind !== manifestAsset.kind) fail(`${row.assetId}: browser row kind mismatch`);
  if (row.telemetry?.backend !== "pt-webgl2") fail(`${row.assetId}: browser telemetry backend mismatch`);
  if (row.telemetry?.assetId !== manifestAsset.assetId) fail(`${row.assetId}: browser telemetry assetId mismatch`);
  if (row.telemetry?.realAssetReady !== true) fail(`${row.assetId}: browser telemetry must prove realAssetReady=true`);
  if ((row.telemetry?.textureDecodeReport?.mapCount ?? 0) < (manifestAsset.minTextures ?? 0)) {
    fail(`${row.assetId}: browser textureDecodeReport.mapCount below manifest expectation`);
  }
  for (const ext of manifestAsset.requiredExtensions ?? []) {
    if (!(row.telemetry?.extensionsUsed ?? []).includes(ext)) {
      fail(`${row.assetId}: browser telemetry missing required extension ${ext}`);
    }
  }
  for (const hook of manifestAsset.requiredHooks ?? []) {
    if (row.telemetry?.browserDecodeHooks?.[hook] !== true) {
      fail(`${row.assetId}: browser telemetry missing decode hook ${hook}`);
    }
  }
  if (!(row.luminance > 0.005)) fail(`${row.assetId}: browser PASS row must be non-black`);
  const structure = row.structure;
  if (structure == null || typeof structure !== "object") fail(`${row.assetId}: browser PASS row must include visual structure`);
  const thresholds = structure.thresholds ?? {};
  if (!(structure.lumaRange >= (thresholds.minLumaRange ?? 12))) fail(`${row.assetId}: browser lumaRange below bound`);
  if (!(structure.uniqueColorCount >= (thresholds.minUniqueColorCount ?? 16))) fail(`${row.assetId}: browser uniqueColorCount below bound`);
  if (!(structure.nonDominantFraction >= (thresholds.minNonDominantFraction ?? 0.05))) {
    fail(`${row.assetId}: browser nonDominantFraction below bound`);
  }
  if (row.golden?.pass !== true) fail(`${row.assetId}: browser golden comparison must pass`);
  if (row.golden?.path !== manifestAsset.goldenPath) fail(`${row.assetId}: browser golden path mismatch`);
  if (row.golden?.thresholds?.maxRmse !== 8 || row.golden?.thresholds?.maxMeanAbs !== 4 || row.golden?.thresholds?.maxAbs !== 48) {
    fail(`${row.assetId}: browser golden thresholds mismatch`);
  }
  await assertPngHeader(row.golden.path, `${row.assetId}: browser golden`);
}

/** @param {string} path @param {string} label */
async function assertPngHeader(path, label) {
  const url = repoUrl(path);
  const stat = await Deno.stat(url);
  if (!stat.isFile || stat.size <= 8) fail(`${label}: PNG is missing or empty`);
  const bytes = await Deno.readFile(url);
  if (bytes[0] !== 0x89 || bytes[1] !== 0x50 || bytes[2] !== 0x4e || bytes[3] !== 0x47) {
    fail(`${label}: file is not a PNG`);
  }
}

/**
 * @param {Record<string, any>} status
 * @param {Record<string, any>} manifest
 */
async function assertPtWebgl2BrowserPassStatus(status, manifest) {
  if (status.harness !== "gltf-browser-proof:pt-webgl2-real") fail("pt-webgl2 browser status harness mismatch");
  if (status.backend !== "pt-webgl2") fail("pt-webgl2 browser status backend mismatch");
  if (manifest.kind !== "vitrum-browser-gltf-pt-webgl2-goldens") fail("pt-webgl2 browser manifest kind mismatch");
  if (manifest.backend !== "pt-webgl2") fail("pt-webgl2 browser manifest backend mismatch");
  if (!Array.isArray(manifest.assets) || manifest.assets.length !== 3) {
    fail("pt-webgl2 browser manifest must contain textured, Draco, and meshopt rows");
  }
  if (status.assetCount != null && status.assetCount !== manifest.assets.length) {
    fail("pt-webgl2 browser status assetCount differs from manifest assets");
  }
  const assetsById = byKey(manifest.assets, "assetId", "pt-webgl2 browser manifest");
  const statusAssets = Array.isArray(status.assets) ? status.assets : [status];
  const statusById = byKey(statusAssets, "assetId", "pt-webgl2 browser status");
  for (const [assetId, asset] of assetsById) {
    const row = statusById.get(assetId);
    if (!row) fail(`pt-webgl2 browser status missing ${assetId}`);
    await assertPtWebgl2BrowserPassRow(row, asset);
  }
  for (const assetId of statusById.keys()) {
    if (!assetsById.has(assetId)) fail(`pt-webgl2 browser status has unexpected asset ${assetId}`);
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

/**
 * @param {Array<Record<string, any>>} ptWebgl2BrowserStatuses
 */
async function assertPromotionStatus(ptWebgl2BrowserStatuses) {
  const status = JSON.parse(await readText(PROMOTION_STATUS_PATH));
  const ptRadiometricPromotion = JSON.parse(await readText(PT_RADIOMETRIC_PROMOTION_STATUS_PATH));
  const bdptRadiometricResult = JSON.parse(await readText(BDPT_RADIOMETRIC_RESULT_PATH));
  const primaryBrowserStatus = ptWebgl2BrowserStatuses[0];
  if (status.harness !== "renderer-fidelity-promotion-proof") fail(`${PROMOTION_STATUS_PATH} harness mismatch`);
  if (status.verdict !== "PASS-PARTIAL") {
    fail(`${PROMOTION_STATUS_PATH} must stay PASS-PARTIAL until pt-webgl2 browser promotion evidence lands`);
  }
  if (status.ptWebgpuFullTier?.supportedRowCount !== PT_WEBGPU_SUPPORTED_ROWS.length) {
    fail(`${PROMOTION_STATUS_PATH} ptWebgpuFullTier.supportedRowCount drifted`);
  }
  if (status.ptWebgpuFullTier?.runtimeProofClass !== "committed-dzn-full-tier-and-golden-artifacts") {
    fail(`${PROMOTION_STATUS_PATH} ptWebgpuFullTier.runtimeProofClass mismatch`);
  }
  if (status.ptWebgpuFullTier?.promotionBoundary !== "row-level-supported") {
    fail(`${PROMOTION_STATUS_PATH} ptWebgpuFullTier.promotionBoundary mismatch`);
  }
  const expectedBdptBoundary = {
    safeDefault: "endpoint-only",
    endpointOnlyMatchesUni: bdptRadiometricResult.controls?.endpointOnlyMatchesUni,
    multiVertexDefaultReady: ptRadiometricPromotion.researchFindings?.bdptMultiVertex?.defaultReady,
    multiVertexWarningCode: ptRadiometricPromotion.researchFindings?.bdptMultiVertex?.warningCode,
    multiVertexCurrentEstimator: ptRadiometricPromotion.researchFindings?.bdptMultiVertex?.currentEstimator,
    multiVertexBlocker: ptRadiometricPromotion.researchFindings?.bdptMultiVertex?.blocker,
    multiVertexSafeAlternative: ptRadiometricPromotion.researchFindings?.bdptMultiVertex?.safeAlternative,
    evidencePath: BDPT_RADIOMETRIC_RESULT_PATH,
  };
  if (JSON.stringify(status.ptWebgpuFullTier?.bdptBoundary) !== JSON.stringify(expectedBdptBoundary)) {
    fail(`${PROMOTION_STATUS_PATH} ptWebgpuFullTier.bdptBoundary must match the radiometric BDPT research boundary`);
  }
  if (
    bdptRadiometricResult.controls?.endpointOnlyMatchesUni !== true ||
    bdptRadiometricResult.controls?.multiVertexPromotion?.defaultReady !== false ||
    bdptRadiometricResult.controls?.multiVertexPromotion?.currentEstimator !== "additive-sidecar-not-weighted-against-eye-path" ||
    bdptRadiometricResult.controls?.multiVertexPromotion?.safeAlternative !== "omit bdptOptions.maxLightBounces or set maxLightBounces:1" ||
    ptRadiometricPromotion.researchFindings?.bdptMultiVertex?.defaultReady !== false ||
    ptRadiometricPromotion.researchFindings?.bdptMultiVertex?.currentEstimator !== "additive-sidecar-not-weighted-against-eye-path" ||
    ptRadiometricPromotion.researchFindings?.bdptMultiVertex?.blocker !== "not-weighted-against-regular-eye-path-strategy"
  ) {
    fail(`${PROMOTION_STATUS_PATH} BDPT boundary artifacts no longer preserve endpoint-only safe default and multi-vertex non-promotion`);
  }
  if (status.ptWebgl2?.browserPromotionReady !== false) {
    fail(`${PROMOTION_STATUS_PATH} ptWebgl2.browserPromotionReady must remain false while browser capture is blocked`);
  }
  if (!ptWebgl2BrowserStatuses.every((browserStatus) => browserStatus.verdict === primaryBrowserStatus.verdict)) {
    fail(`${PROMOTION_STATUS_PATH} ptWebgl2 browser status artifacts must agree before promotion accounting`);
  }
  if (status.ptWebgl2?.browserStatus !== primaryBrowserStatus.verdict) {
    fail(`${PROMOTION_STATUS_PATH} ptWebgl2.browserStatus must match ${PT_WEBGL2_BROWSER_STATUS_PATH}`);
  }
  if (primaryBrowserStatus.verdict === "HOST-BLOCKED") {
    const expectedClasses = Array.from(new Set(
      ptWebgl2BrowserStatuses.flatMap((browserStatus) => browserStatus.hostBlockClasses ?? []),
    )).sort();
    if (JSON.stringify(status.ptWebgl2?.hostBlockClasses) !== JSON.stringify(expectedClasses)) {
      fail(`${PROMOTION_STATUS_PATH} ptWebgl2.hostBlockClasses must match browser proof statuses`);
    }
    const expectedPreflightByMode = Object.fromEntries(
      ptWebgl2BrowserStatuses.map((browserStatus) => [
        browserStatus.captureMode,
        browserStatus.hostReadbackProbe?.status,
      ]),
    );
    if (JSON.stringify(status.ptWebgl2?.browserReadbackPreflight?.statusByCaptureMode) !== JSON.stringify(expectedPreflightByMode)) {
      fail(`${PROMOTION_STATUS_PATH} ptWebgl2.browserReadbackPreflight.statusByCaptureMode must match browser proof probes`);
    }
    for (const browserStatus of ptWebgl2BrowserStatuses) {
      const probe = browserStatus.hostReadbackProbe;
      if (
        probe?.status !== "PASS" ||
        probe.webgl2 !== true ||
        probe.unsignedByteReadback?.status !== "PASS" ||
        probe.floatReadback?.status !== "PASS" ||
        probe.dataUrl?.status !== "PASS"
      ) {
        fail(`${browserStatus.captureMode}: browser proof must preserve passing WebGL2/readPixels/toDataURL preflight diagnostics`);
      }
    }
    if (
      status.ptWebgl2?.browserReadbackPreflight?.webgl2 !== true ||
      status.ptWebgl2?.browserReadbackPreflight?.unsignedByteReadback !== "PASS" ||
      status.ptWebgl2?.browserReadbackPreflight?.floatReadback !== "PASS" ||
      status.ptWebgl2?.browserReadbackPreflight?.dataUrl !== "PASS" ||
      !String(status.ptWebgl2?.browserReadbackPreflight?.scope ?? "").includes("real glTF page browser capture remains HOST-BLOCKED") ||
      JSON.stringify(status.ptWebgl2?.browserReadbackPreflight?.sourceStatuses) !== JSON.stringify([
        PT_WEBGL2_BROWSER_STATUS_PATH,
        PT_WEBGL2_BROWSER_CANVAS_FIRST_STATUS_PATH,
      ])
    ) {
      fail(`${PROMOTION_STATUS_PATH} ptWebgl2.browserReadbackPreflight must summarize passing simple readback and blocked real glTF capture`);
    }
  }
  if (status.ptWebgl2?.nonPromotionGradeCount !== PT_WEBGL2_EXPECTED_ROWS.length) {
    fail(`${PROMOTION_STATUS_PATH} ptWebgl2.nonPromotionGradeCount drifted`);
  }
  if (status.ptWebgl2?.materialFurnaceSourceOracleGroupCount !== PT_WEBGL2_MATERIAL_FURNACE_PROOFS.length) {
    fail(`${PROMOTION_STATUS_PATH} ptWebgl2.materialFurnaceSourceOracleGroupCount drifted`);
  }
  if (!String(status.ptWebgl2?.requiredEvidence ?? "").includes("browser/real-adapter reference A/B")) {
    fail(`${PROMOTION_STATUS_PATH} ptWebgl2.requiredEvidence must name browser/real-adapter reference A/B`);
  }
  const expectedSourceStatuses = [
    PT_WEBGL2_BROWSER_STATUS_PATH,
    PT_WEBGL2_BROWSER_CANVAS_FIRST_STATUS_PATH,
    PT_WEBGL2_BROWSER_MANIFEST_PATH,
    "tools/behavioral-gate/behavioral-gate-dzn-spectral-status.json",
    "tools/behavioral-gate/behavioral-gate-dzn-light-status.json",
    "tools/behavioral-gate/behavioral-gate-dzn-caustic-status.json",
    "tools/behavioral-gate/behavioral-gate-dzn-bdpt-status.json",
    PT_RADIOMETRIC_PROMOTION_STATUS_PATH,
    BDPT_RADIOMETRIC_RESULT_PATH,
  ];
  if (JSON.stringify(status.sourceStatuses) !== JSON.stringify(expectedSourceStatuses)) {
    fail(`${PROMOTION_STATUS_PATH} sourceStatuses drifted`);
  }
  const expectedProvenance = await rendererFidelityPromotionProvenance(expectedSourceStatuses);
  if (JSON.stringify(status.provenance) !== JSON.stringify(expectedProvenance)) {
    fail(`${PROMOTION_STATUS_PATH} provenance differs from current checker/source artifact identity`);
  }
}

const matrix = await readText(MATRIX_PATH);
const playbook = await readText(PLAYBOOK_PATH);
const readme = await readText(README_PATH);
const architecture = await readText(ARCHITECTURE_PATH);
const hardwareValidation = await readText(HARDWARE_VALIDATION_PATH);
const gapExecutionPlan = await readText(GAP_EXECUTION_PLAN_PATH);
const ptWebgl2BrowserStatus = JSON.parse(await readText(PT_WEBGL2_BROWSER_STATUS_PATH));
const ptWebgl2BrowserCanvasFirstStatus = JSON.parse(await readText(PT_WEBGL2_BROWSER_CANVAS_FIRST_STATUS_PATH));
const ptWebgl2BrowserManifest = JSON.parse(await readText(PT_WEBGL2_BROWSER_MANIFEST_PATH));

if (!matrix.includes("| Feature | pt-webgl2 (WebGL2) | pt-webgpu full tier (WebGPU) |")) {
  fail("renderer fidelity matrix must label the pt-webgpu column as full-tier proof");
}
if (matrix.includes("| Feature | pt-webgl2 (WebGL2) | pt-webgpu (WebGPU) |")) {
  fail("renderer fidelity matrix must not use the stale unqualified pt-webgpu column heading");
}

assertEveryPtWebgpuSupportedRowHasProof(matrix);
assertPtWebgl2ExpectedRows(matrix);

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

for (const [path, status] of [
  [PT_WEBGL2_BROWSER_STATUS_PATH, ptWebgl2BrowserStatus],
  [PT_WEBGL2_BROWSER_CANVAS_FIRST_STATUS_PATH, ptWebgl2BrowserCanvasFirstStatus],
]) {
  if (status.backend !== "pt-webgl2") fail(`${path} backend mismatch`);
}
const browserStatuses = [ptWebgl2BrowserStatus, ptWebgl2BrowserCanvasFirstStatus];
if (browserStatuses.some((status) => status.verdict === "HOST-BLOCKED")) {
  if (!browserStatuses.every((status) => status.verdict === "HOST-BLOCKED")) {
    fail("pt-webgl2 browser status artifacts must all pass before any promotion");
  }
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
} else if (browserStatuses.some((status) => status.verdict !== "PASS")) {
  fail("pt-webgl2 browser status artifact verdicts must be PASS or HOST-BLOCKED");
} else {
  for (const browserStatus of browserStatuses) {
    await assertPtWebgl2BrowserPassStatus(browserStatus, ptWebgl2BrowserManifest);
  }
}
await assertPromotionStatus(browserStatuses);

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
  `[renderer-fidelity-proof-check] PASS (${PT_WEBGPU_SUPPORTED_ROWS.length} pt-webgpu supported rows verified; ${PT_WEBGL2_EXPECTED_ROWS.length} pt-webgl2 non-promotion grades pinned; ${PT_WEBGL2_MATERIAL_FURNACE_PROOFS.length} pt-webgl2 material-furnace source/oracle proof groups verified; pt-webgl2 browser-promotion guard checked; renderer promotion status pinned)`,
);
