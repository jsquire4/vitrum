#!/usr/bin/env -S deno run --sloppy-imports --allow-read
// @ts-check

// Source-facing renderer maturity guard. Grades are derived from the public
// contract and the code that packs, uploads, consumes, owns, and validates each
// estimator. Host-specific captures are regression artifacts, not maturity
// prerequisites, and are intentionally not read by this checker.

const MATRIX_PATH = 'plan/renderer-fidelity-matrix.md';
const PT_WEBGL2_README_PATH = 'packages/pt-webgl2/README.md';

/** @type {ReadonlyArray<readonly [string, string, string]>} */
const EXPECTED_ROWS = [
  ['Hero-wavelength + CMF accumulation', 'supported', 'supported'],
  ['Spectral Beer–Lambert (packed μ)', 'supported', 'supported'],
  ['Multi-layer thin-film TMM', 'supported', 'supported'],
  ['Cauchy / Abbe dispersion', 'supported', 'supported'],
  ['Layered front/back + transmission MIS', 'supported', 'supported'],
  ['SSS / translucent panels', 'approximate', 'supported'],
  ['Multi-emitter direct lighting', 'supported', 'supported'],
  ['Cornell/core material fixture parity', 'supported', 'supported'],
  ['Manifold next-event estimation (MNEE)', 'unsupported', 'supported'],
  ['Progressive photon mapping (SPPM; `photon-map`)', 'unsupported', 'supported'],
  ['SVGF-real denoiser', 'unsupported', 'unsupported'],
  ['BDPT (eye↔light connections)', 'supported', 'supported'],
];

/**
 * Direct implementation tripwires for every substantive classification family.
 * These are deliberately implementation/test symbols, not comments in planning
 * documents or hashes of external captures.
 * @type {ReadonlyArray<{label: string, path: string, needles: readonly string[]}>}
 */
const SOURCE_PROOFS = [
  {
    label: 'core layered-material envelope',
    path: 'packages/core/src/scene/material.ts',
    needles: [
      'export interface SurfaceAbsorptionLayer',
      'infinitesimally thin absorber',
      'simplified (non-multiple-',
      'scattering) form of Belcour',
    ],
  },
  {
    label: 'core emitter contract',
    path: 'packages/core/src/scene/emitters.ts',
    needles: [
      'Delta-position source origin.',
      'Angular cone falloff, not a finite-source/soft-shadow radius.',
    ],
  },
  {
    label: 'host-readable backend promises',
    path: 'packages/core/src/engine/promiseLedger.ts',
    needles: [
      "'manifold-nee': {",
      "'photon-map': {",
      'thinFilmLayerLimit: 35',
      'thinFilmLayerLimit: 8',
      'bounded general BDPT:',
      'one-to-eight planar geometric-normal',
    ],
  },
  {
    label: 'pt-webgl2 strict estimator options',
    path: 'packages/pt-webgl2/src/options.ts',
    needles: [
      'bounded general-BDPT walk accepts 1..8 vertices; default 4',
      "readonly causticStrategy?: 'bdpt';",
      'not MNEE or',
      'photon mapping.',
    ],
  },
  {
    label: 'pt-webgl2 strict denoiser validation',
    path: 'packages/pt-webgl2/src/options.validate.ts',
    needles: [
      "const WEBGL2_DENOISERS = new Set(['none', 'auto', 'oidn-final']);",
      "const CAUSTIC_STRATEGIES = new Set(['bdpt']);",
      'createPTEngine_WebGL2: causticStrategy',
      'the supported range 1..${BDPT_MAX_LIGHT_BOUNCES}',
    ],
  },
  {
    label: 'pt-webgl2 spectral upload',
    path: 'packages/pt-webgl2/src/gl/uploadFrameUniforms.ts',
    needles: [
      "prog.setFloatArray('uCmfX', CMF_X_F32);",
      "prog.setFloatArray('uCmfY', CMF_Y_F32);",
      "prog.setFloatArray('uCmfZ', CMF_Z_F32);",
    ],
  },
  {
    label: 'pt-webgl2 spectral/BDPT trace ownership',
    path: 'packages/pt-webgl2/src/glsl/renderMain.glsl.ts',
    needles: [
      'sampleHeroWavelengthMIS(',
      'evaluateBdptConnection(',
      '#if FEATURE_BDPT',
      '#if ! FEATURE_BDPT',
    ],
  },
  {
    label: 'pt-webgl2 material transport',
    path: 'packages/pt-webgl2/src/glsl/shader/bsdf/bsdf_functions.glsl.js',
    needles: [
      'thinFilmTMM(',
      'spectralAttenuationMuHero(',
      'transmissionEtaAtHero(',
      'float sigmaTMajorant',
      'sampleExponentialDistance( rand( 17 ), sigmaTMajorant, 1e6 )',
    ],
  },
  {
    label: 'pt-webgl2 bounded thin-film packing',
    path: 'packages/pt-webgl2/src/scene/materialsTexture.ts',
    needles: [
      'export const THIN_FILM_LAYER_LIMIT = 35;',
      'the exact backend limit is ${THIN_FILM_LAYER_LIMIT}',
      'spectralAttenuation',
    ],
  },
  {
    label: 'pt-webgl2 analytic/triangle light estimators',
    path: 'packages/pt-webgl2/src/glsl/shader/sampling/light_sampling_functions.glsl.js',
    needles: [
      'LightRecord sampleMeshAreaLight(',
      'float meshAreaLightForwardPdf(',
      'LightRecord randomLightSample(',
    ],
  },
  {
    label: 'pt-webgl2 general-BDPT executable oracle',
    path: 'packages/pt-webgl2/src/__tests__/bdptProductionEstimator.test.ts',
    needles: [
      'connects finite c=0 plus every stored c>=1 vertex and evaluates Veach MIS in log space',
      'uses one power-heuristic denominator for distant s=0, s=1, and bounded s>=2',
    ],
  },
  {
    label: 'pt-webgpu public estimator options',
    path: 'packages/pt-webgpu/src/index.ts',
    needles: [
      "readonly denoiser?: 'none' | 'auto' | 'oidn-final';",
      "readonly causticStrategy?: 'none' | 'manifold-nee' | 'photon-map';",
      'Integer 1..8; default 2.',
      'SPPM photon grid could not be allocated',
    ],
  },
  {
    label: 'pt-webgpu spectral/BDPT trace ownership',
    path: 'packages/pt-webgpu/src/wgsl/pathTrace/kernel.wgsl.ts',
    needles: [
      'sampleHeroWavelengthMIS(rand_f32(&rng), rand_f32(&rng))',
      'bdptOwnsFiniteLightFamily',
      'directFamilyCount',
      'evaluateBdptConnection(',
      'fn bdptResolveCameraSplats(',
      'var frameEstimatorHash =',
      'let rptMixtureSelected =',
      'let advancedEstimatorSelected =',
      'advancedPeerEnabled && !rptMixtureSelected',
    ],
  },
  {
    label: 'pt-webgpu bounded MNEE Newton core',
    path: 'packages/pt-webgpu/src/wgsl/pathTrace/mneeNewton.wgsl.ts',
    needles: [
      'mneeBoundedChainResidualAt',
      'mneeNewtonSolveChainBounded',
      'mneeBoundedChainFocusingDet',
      'mneeBoundedChainAreaPdfDet',
    ],
  },
  {
    label: 'pt-webgpu MNEE fail-closed domain',
    path: 'packages/pt-webgpu/src/scene/mneeFacetCandidates.ts',
    needles: [
      'export function assertMneeInterfaceDomainSupported',
      'Fail closed outside the implemented planar/geometric-normal manifold domain.',
      'normal/bump/layer-normal maps require a varying-normal manifold Jacobian',
    ],
  },
  {
    label: 'pt-webgpu progressive SPPM implementation',
    path: 'packages/pt-webgpu/src/wgsl/pathTrace/caustic.wgsl.ts',
    needles: [
      'true Hachisuka & Jensen 2009 SPPM with per-pixel progressive',
      'sppmUpdateSurfaceProgressive(',
    ],
  },
  {
    label: 'pt-webgpu BDPT executable ownership oracle',
    path: 'packages/pt-webgpu/src/__tests__/bdptEstimatorOwnership.test.ts',
    needles: [
      'assigns every finite direct-light branch to BDPT without changing the off path',
      'samples the primary eye vertex and emitter endpoint as explicit strategies',
      'bdptOwnsFiniteLightFamily',
      'PT_WEBGPU_BDPT_CONNECTION_WGSL',
    ],
  },
  {
    label: 'pt-webgpu native BDPT camera-splat implementation',
    path: 'packages/pt-webgpu/src/wgsl/bdpt/bdptCameraSplat.wgsl.ts',
    needles: [
      'export const PT_WEBGPU_BDPT_CAMERA_SPLAT_WGSL',
      'atomicCompareExchangeWeak(',
      'fn bdptLoadCameraRgb(',
      'fn bdptAccumulateCameraSplatStrategies(',
    ],
  },
  {
    label: 'pt-webgpu native BDPT camera-splat executable oracle',
    path: 'packages/pt-webgpu/src/__tests__/bdptCameraSplatWiring.test.ts',
    needles: [
      'keeps the BDPT-off shader/resource surface unchanged',
      'composes the atomic t=1 strategy and resolver only for bdpt:true',
      'allocates, binds, clears, dispatches, and disposes the camera-splat cohort',
      'admits s=n-1 in the canonical CPU+WGSL bounded-strategy mask',
      'reports the now-executable native camera-splat strategy',
    ],
  },
  {
    label: 'pt-webgpu frame-global advanced-estimator composition oracle',
    path: 'packages/pt-webgpu/src/__tests__/advancedEstimatorComposition.test.ts',
    needles: [
      'uses one backend-independent frame coin and keeps producer drops out of BDPT',
      'selectionBlock).not.toContain(\'gid.\')',
      'selectedAtArbitrarySplatTarget',
      'camera splats merely because this target pixel had no RPT contribution',
    ],
  },
  {
    label: 'cross-backend emitter stream decision',
    path: 'packages/engine/src/__tests__/emitterCanonicalParity.test.ts',
    needles: [
      'pins the core spot emitter as a delta-position source in both backend packers',
      'dedicated triangle-light stream',
    ],
  },
];

/** @param {string} path */
function repoUrl(path) {
  return new URL(`../../${path}`, import.meta.url);
}

/** @param {string} message @returns {never} */
function fail(message) {
  throw new Error(`[renderer-fidelity-proof-check] ${message}`);
}

/** @param {string} path */
async function readText(path) {
  return await Deno.readTextFile(repoUrl(path));
}

/** @param {string} matrix @param {string} feature */
function findMatrixRow(matrix, feature) {
  const row = matrix.split('\n').find((line) => line.startsWith(`| ${feature} |`));
  if (!row) fail(`${MATRIX_PATH} is missing the ${feature} row`);
  return row;
}

const matrix = await readText(MATRIX_PATH);
if (!matrix.includes('| Feature | pt-webgl2 (WebGL2) | pt-webgpu full tier (WebGPU) |')) {
  fail('matrix backend headings drifted');
}
if (/\|\s*experimental\s*\|/i.test(matrix)) {
  fail('matrix contains an unfinished maturity grade');
}
if (matrix.includes('runtime A/B capture pending')) {
  fail('matrix makes host capture a maturity prerequisite');
}

const featureTable = matrix.split('## Feature rows')[1]?.split('## Maintainer gate')[0] ?? '';
const matrixRows = featureTable.split('\n').filter((line) =>
  line.startsWith('| ') &&
  !line.startsWith('| Feature |') &&
  !line.startsWith('|---------')
);
if (matrixRows.length !== EXPECTED_ROWS.length) {
  fail(`matrix has ${matrixRows.length} feature rows; expected ${EXPECTED_ROWS.length}`);
}

for (const [feature, expectedWebgl2, expectedWebgpu] of EXPECTED_ROWS) {
  const columns = findMatrixRow(matrix, feature).split('|').map((value) => value.trim());
  if (columns[2] !== expectedWebgl2 || columns[3] !== expectedWebgpu) {
    fail(
      `${feature}: expected pt-webgl2=${expectedWebgl2}, pt-webgpu=${expectedWebgpu}; ` +
      `got ${columns[2] ?? '<missing>'}/${columns[3] ?? '<missing>'}`,
    );
  }
}

for (const proof of SOURCE_PROOFS) {
  const source = await readText(proof.path);
  for (const needle of proof.needles) {
    if (!source.includes(needle)) {
      fail(`${proof.label}: ${proof.path} is missing source proof: ${needle}`);
    }
  }
}

const ptWebgl2Readme = await readText(PT_WEBGL2_README_PATH);
for (const needle of [
  'Supported bounded general BDPT',
  "Caustic strategy `'bdpt'`",
  "MNEE / SPPM (`'manifold-nee'`, `'photon-map'`) | Unsupported",
  'one back-face single-scatter event',
]) {
  if (!ptWebgl2Readme.includes(needle)) {
    fail(`${PT_WEBGL2_README_PATH} is missing current boundary text: ${needle}`);
  }
}

for await (const entry of Deno.readDir(repoUrl('packages'))) {
  if (!entry.isDirectory) continue;
  const path = `packages/${entry.name}/README.md`;
  let text;
  try {
    text = await readText(path);
  } catch (error) {
    if (error instanceof Deno.errors.NotFound) continue;
    throw error;
  }
  if (/\bexperimental\b/i.test(text)) {
    fail(`${path} still uses an unfinished maturity label`);
  }
}

console.log(
  `[renderer-fidelity-proof-check] PASS (${EXPECTED_ROWS.length} source-graded rows; ` +
  `${SOURCE_PROOFS.length} implementation proof groups; package README maturity labels clean)`,
);
