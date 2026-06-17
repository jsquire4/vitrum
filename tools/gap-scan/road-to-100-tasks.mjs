/**
 * Road-to-100 open items NOT fully covered by code-gap line-scan tasks.
 * Source: plan/road-to-100.md (2026-06-16 HEAD reconciliation).
 *
 * Disposition guide:
 * - VERIFY = implementation landed; GPU/radiometric proof pending (V28-B) — **SKIP in code-first mode**
 * - SKIP = excluded from schedule (deferred validation or duplicate)
 * - IMP = still requires code
 * - DECIDE = product/architecture call before code
 * - DOC = ledger/matrix/README sync
 */

function rt(id, phase, priority, disposition, problem, files, steps, tests, done, depends = []) {
  return {
    id,
    phase,
    priority,
    disposition,
    depends,
    blocks: [],
    problem,
    files,
    steps,
    tests,
    done,
  };
}

/** Mark validation-only tasks as SKIP for code-first scheduling. */
function skipVal(id, phase, priority, problem, files, steps, done, depends = []) {
  return rt(
    id,
    phase,
    priority,
    'SKIP',
    `[DEFERRED — code-first] ${problem} See plan/VALIDATION-DEFERRED.md.`,
    files,
    [...steps, 'No work in code-first campaign — task deferred to validation sprint.'],
    ['# deferred'],
    [...done, 'Deferred — no agent dispatch.'],
    depends,
  );
}

/** @type {import('./task-registry.mjs').task[]} */
export const RT100_VALIDATION = [
  skipVal(
    'RT100-V28-000',
    8,
    'P1',
    'V28-B baseline recapture on wsl-gpu.',
    ['HARDWARE-VALIDATION-NEEDS.md', 'tools/reference-renders/'],
    ['Recapture reference renders for all render-changing landings.'],
    ['V28-B baseline set captured.'],
  ),
  ...['A1', 'A2', 'A3', 'A4', 'A5', 'A7', 'B1', 'B2', 'B4', 'B8', 'B15', 'B16'].map((bucket) =>
    skipVal(
      `RT100-V28-${bucket}`,
      8,
      'P2',
      `${bucket} radiometric/variance GPU A/B (road-to-100 Bucket ${bucket[0]}).`,
      ['plan/road-to-100.md', 'HARDWARE-VALIDATION-NEEDS.md'],
      [`Run ${bucket} A/B scenario on wsl-gpu.`],
      [`${bucket} V28 evidence captured.`],
      ['RT100-V28-000'],
    ),
  ),
  skipVal(
    'RT100-V19-GRIS',
    8,
    'P1',
    'GRIS-on unbiasedness + biased-default GPU quantification (A8/F6).',
    ['tools/benchmark-runner/'],
    ['Run converged-unbiasedness harness on wsl-gpu.'],
    ['V19 GRIS A/B report written.'],
    ['RT100-V28-000'],
  ),
];

/** @type {import('./task-registry.mjs').task[]} */
export const RT100_IMPLEMENTATION = [
  rt(
    'RT100-ADJ-001',
    9,
    'P1',
    'IMP',
    'pt-webgpu inverse adjoint path replay OPEN: alpha-map adjoint, normal/bump/transmission adjoints, env light terms, indirect paths (road-to-100 §2C Adjoint row).',
    [
      'packages/pt-webgpu/src/inverse/adjointPass.wgsl.ts',
      'packages/pt-webgpu/src/inverse/pathTraceAdjoint.wgsl.ts',
      'packages/pt-webgpu/src/inverse/inverseSession.ts',
    ],
    [
      'Read road-to-100.md §2C integrator audit Adjoint row Still OPEN list.',
      'Implement or permanently downgrade each OPEN adjoint domain with structured diagnostic.',
      'Extend brdfAdjoint.test.ts + inverseSession.test.ts per domain.',
      'Update promiseLedger inverse downgrade matrix comments.',
    ],
    [
      'cd packages/pt-webgpu && npx vitest run src/inverse/',
      'npm run typecheck',
    ],
    ['Each OPEN adjoint row closed or downgraded with test.'],
    ['P0-008-TOOL-001'],
  ),
  rt(
    'RT100-WA-3D',
    9,
    'P1',
    'IMP',
    'Walkaround Phase 3D tail: remaining atlas gaps — bump/displacement policy rows, morph-target UV deform refresh, narrower atlas refresh optimization (road-to-100 §3D footguns).',
    [
      'packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts',
      'packages/walkaround-hybrid/src/HybridEnginePrimitiveUpdates.ts',
      'packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts',
    ],
    [
      'Read road-to-100.md Phase 3D — atlas slices largely landed; identify remaining gaps vs Master checklist.',
      'Implement bump map consumption if still missing from shade/ReSTIR paths.',
      'Add morph-target UV deform detection → full atlas refresh or documented limitation.',
      'Optional: narrower atlas refresh for map-handle-only edits (cost footgun).',
    ],
    [
      'cd packages/walkaround-hybrid && npx vitest run',
      'cd packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts',
    ],
    ['3D footguns closed or documented with tests.', 'Ledger rows match CONSUMED_MATERIAL_FIELDS.'],
    ['P0-003-WH-034'],
  ),
  skipVal(
    'RT100-WA-3E',
    9,
    'P1',
    'Walkaround Phase 3E promotion via material-furnace/reference A/B.',
    ['tools/reference-renders/'],
    ['Run promotion A/B per extension lobe.'],
    ['3E promotion evidence captured.'],
    ['RT100-WA-3D'],
  ),
  rt(
    'RT100-WA-ALPHA',
    9,
    'P1',
    'IMP',
    'Walkaround Phase 3C alpha: transparent ReSTIR/GI promotion + layered transport (road-to-100 Master checklist alpha rows).',
    [
      'packages/walkaround-hybrid/src/shaders/transparentOit.wgsl.ts',
      'packages/walkaround-hybrid/src/shaders/shade.wgsl.ts',
      'packages/walkaround-hybrid/src/restir/consumedMaterialFields.ts',
    ],
    [
      'Read road-to-100 alpha row: OIT direct sun done; ReSTIR/GI transport still approximate.',
      'Implement stochastic alpha GI transport OR document permanent OIT-split with ledger ACC.',
      'Add behavioral gate wh/alpha-blend if not present.',
    ],
    ['cd packages/walkaround-hybrid && npx vitest run'],
    ['Alpha GI policy implemented or permanently documented.'],
    ['RT100-WA-3D'],
  ),
  rt(
    'RT100-PTGL-MUT',
    9,
    'P1',
    'IMP',
    'pt-webgl2 geometry mutations: port TLAS/refit/splice from pt-webgpu sceneMutationRouter (road-to-100 §2D — transform/positions/topology still fallback-rebuild).',
    [
      'packages/pt-webgl2/src/scene/mutateSceneTextures.ts',
      'packages/pt-webgpu/src/sceneMutationRouter.ts',
    ],
    [
      'Study pt-webgpu fast paths: transform, positions, topology-resize, instanced-topology.',
      'Port applicable paths to pt-webgl2 WebGL2 buffer upload model.',
      'Promote promiseLedger mutation rows from fallback-rebuild to native where implemented.',
      'Extend updatePrimitiveIncremental tests on pt-webgl2.',
    ],
    [
      'cd packages/pt-webgl2 && npx vitest run',
      'cd packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts',
    ],
    ['transform/positions/topology native or explicitly documented fallback.'],
    ['P0-002-PTGL-003'],
  ),
  rt(
    'RT100-PTWG-MAT',
    9,
    'P1',
    'IMP',
    'PTWG-MAT-01 integrator audit: extension lobes must match across BDPT/SPPM/ReSTIR-PT/MNEE paths — material-furnace promotion tail.',
    [
      'packages/pt-webgpu/src/wgsl/pathTrace/',
      'packages/pt-webgpu/src/__tests__/extensionLobeReference.test.ts',
    ],
    [
      'Read road-to-100.md §2C integrator audit table.',
      'For each path row not ✅: wire evaluateBrdf/sampleNextBounceDirection parity.',
      'Run extensionLobeReference + bdptGlossyLightSubpath + restirPtSpecialtyReference tests.',
      'Promote renderer-fidelity-matrix rows when unit/oracle tests pass (GPU furnace deferred).',
    ],
    [
      'cd packages/pt-webgpu && npx vitest run',
      'npm run shader-gate',
    ],
    ['All integrator audit rows ✅ or documented approximate.'],
    ['P0-001-PTWG-037'],
  ),
  rt(
    'RT100-GLTF-PICK',
    9,
    'P1',
    'IMP',
    'Arbitrary glTF Phase 4: pickBackend must use feature report not triangle-count alone (road-to-100 §4 + createEngineScale footgun).',
    [
      'packages/engine/src/createEngineScale.ts',
      'packages/gltf-adapter/src/featureReport.ts',
      'packages/engine/src/gltf.ts',
    ],
    [
      'Read road-to-100.md Phase 4 and trap table pickBackend row.',
      'Wire rankGltfBackends / evaluateGltfBackendCompatibility into createEngine preference path.',
      'Add test: textured hero asset must not auto-route to walkaround when PT supports features.',
    ],
    ['cd packages/engine && npx vitest run', 'cd packages/gltf-adapter && npx vitest run'],
    ['pickBackend uses compatibility report for glTF assets.'],
    ['FP-04'],
  ),
  rt(
    'RT100-EMISSIVE-PDF',
    9,
    'P2',
    'IMP',
    'B4 tail: full energy-weighted emissive texel alias/PDF with forward-hit MIS parity (road-to-100 — not just area-PDF).',
    [
      'packages/shared-samplers/src/meshAreaLights.ts',
      'packages/pt-webgpu/src/scene/meshAreaLights.ts',
      'packages/pt-webgl2/src/scene/meshAreaLights.ts',
      'packages/walkaround-hybrid/src/restir/emitterList.ts',
    ],
    [
      'Read road-to-100 B4 Done= tail about texel alias/PDF.',
      'Implement alias table for emissive texel importance on CPU packers.',
      'Ensure forward NEE PDF matches forward-hit MIS weight.',
      'Extend meshAreaMis.test.ts across backends.',
    ],
    ['npm test --workspaces --if-present'],
    ['Alias/PDF path live on all three backends.', 'meshAreaMis parity tests green.'],
    ['WH-012'],
  ),
  skipVal(
    'RT100-A9-BDPT',
    9,
    'P2',
    'BDPT radiometric/material-furnace oracle for light-subpath connections.',
    ['tools/radiometric-ab/'],
    ['Run radiometric oracle vs forward-traced reference.'],
    ['BDPT radiometric oracle green.'],
    ['RT100-PTWG-MAT'],
  ),
  skipVal(
    'RT100-PTWG-FURNACE',
    9,
    'P1',
    'PTWG approximate ledger rows: material-furnace + reference-render promotion.',
    ['tools/reference-renders/'],
    ['Furnace promotion for approximate PTWG material rows.'],
    ['Furnace promotion report.'],
    ['RT100-PTWG-MAT'],
  ),
];

/** @type {import('./task-registry.mjs').task[]} */
export const RT100_PROOF = [
  skipVal(
    'RT100-5A-GOLDEN',
    10,
    'P1',
    'Phase 5A: golden PNG + real Khronos asset render sweep.',
    ['tools/reference-assets/gltf/'],
    ['Render 64spp per fixture; compare golden PNG.'],
    ['Golden renders for Khronos set.'],
    ['RT100-V28-000'],
  ),
  skipVal(
    'RT100-5C-GPU-MUT',
    10,
    'P2',
    'Phase 5C: GPU mutation matrix observability on wsl-gpu.',
    ['tools/benchmark-runner/'],
    ['Observe real GPU buffers on mutation scenarios.'],
    ['GPU mutation matrix report.'],
    ['RT100-V28-000'],
  ),
  rt(
    'RT100-5D-DOC',
    10,
    'P2',
    'DOC',
    'Phase 5D documentation sync: fidelity matrix, items_to_fix §H, road-to-100 stale addenda (H1–H5), READMEs cite ledger (road-to-100 C5).',
    [
      'plan/renderer-fidelity-matrix.md',
      'plan/road-to-100.md',
      'items_to_fix.md',
      'README.md',
      'packages/*/README.md',
    ],
    [
      'Remove deleted pt-webgl column from fidelity matrix; ensure pt-webgl2 column accurate.',
      'Strike/reconcile stale road-to-100 addendum bullets (e.g. H1–H5 inert — closed in items_to_fix).',
      'Close or strike items_to_fix §H entries verified fixed.',
      'README maturity claims cite BACKEND_PROMISE_LEDGER not prose.',
    ],
    ['npm run typecheck'],
    ['Doc sync checklist complete.', 'No stale OPEN claims contradicting items_to_fix.'],
  ),
  skipVal(
    'RT100-GATE-FULL',
    10,
    'P2',
    'Phase 5E: full-tier behavioral gate + walkaround glTF render lanes.',
    ['tools/behavioral-gate/gate.mjs'],
    ['Run glTF fixtures at full tier when adapter allows.'],
    ['Full-tier gate path documented or implemented.'],
    ['RT100-5A-GOLDEN'],
  ),
];

/** @type {import('./task-registry.mjs').task[]} */
export const RT100_DECIDE = [
  skipVal(
    'RT100-A6-DECIDE',
    11,
    'P1',
    'NRC default-on tier decision (requires GPU quality A/B).',
    ['plan/road-to-100.md'],
    ['Run NRC quality A/B vs DDGI reference.'],
    ['Decision recorded.'],
    ['RT100-V19-GRIS'],
  ),
  skipVal(
    'RT100-A10-WEIGHTS',
    11,
    'P1',
    'Production neural checkpoint decision (requires quality A/B).',
    ['tools/neural-denoiser-training/'],
    ['Quality A/B vs starter-v1 checkpoint.'],
    ['Production checkpoint decision made.'],
  ),
];

/** @type {import('./task-registry.mjs').task[]} */
export const RT100_SOTA = [
  rt(
    'RT100-LD-SAMPLING-01',
    12,
    'P2',
    'IMP',
    'F1/LD-SAMPLING-01: Owen-Sobol or PMJ02 + blue-noise screen scramble in shared-samplers; pt-webgpu + pt-webgl2 integration.',
    [
      'packages/shared-samplers/src/',
      'packages/pt-webgpu/src/wgsl/pathTrace/kernel.wgsl.ts',
      'packages/pt-webgl2/src/glsl/',
    ],
    [
      'Read road-to-100.md F1 section.',
      'Generate LD tables CPU-side; upload as textures/buffers.',
      'Per-dimension assignment audit (bounce/lobe/light).',
      'Revive or replace dead pt-webgl2 RANDOM_TYPE branches.',
      'GPU RMSE A/B deferred — see plan/VALIDATION-DEFERRED.md.',
    ],
    ['cd packages/shared-samplers && npx vitest run', 'npm run typecheck'],
    ['LD sampling integrated both PT backends.'],
    ['RT100-PTWG-MAT'],
  ),
  rt(
    'RT100-WBVH-01',
    12,
    'P2',
    'IMP',
    'F2/WBVH-01: compressed wide BVH (CWBVH) opt-in builder + traversal behind capability flag.',
    [
      'packages/shared-bvh/src/',
      'packages/shared-bvh/wgsl/',
    ],
    [
      'Read road-to-100.md F2 section.',
      'Implement CWBVH build + WGSL traversal in shared-bvh.',
      'CPU brute-force oracle vs binary BVH in package vitest (wsl-gpu oracle deferred).',
      'Per-backend opt-in until parity proven.',
    ],
    ['cd packages/shared-bvh && npx vitest run'],
    ['CWBVH behind capability flag.', 'Oracle parity on test scenes.'],
    ['RT100-PTGL-MUT'],
  ),
  rt(
    'RT100-F3-DENO-AUTO',
    12,
    'P3',
    'IMP',
    'F3: denoiser:auto default when weights resolve; turnkey OIDN/neural without host wiring.',
    [
      'packages/engine/src/createEngine.ts',
      'packages/walkaround-hybrid/src/HybridEngineOptions.ts',
      'packages/shared-denoisers/',
    ],
    [
      'Read road-to-100 F3 section.',
      'Add denoiser:auto union value.',
      'Resolve bundled or downloadable weights at engine construction.',
      'Clear error when assets missing.',
    ],
    ['cd packages/engine && npx vitest run'],
    ['denoiser:auto documented and functional when weights present.'],
    ['RT100-A10-WEIGHTS'],
  ),
  rt(
    'RT100-F4-WAVEFRONT',
    12,
    'P3',
    'ACC',
    'F4: wavefront PT rearchitecture — profile-gated research item, not arbitrary-glTF blocker (road-to-100 post-100).',
    ['plan/road-to-100.md', 'packages/pt-webgpu/src/wgsl/pathTrace/kernel.wgsl.ts'],
    [
      'Run divergence profiling on reference scenes before scheduling.',
      'Document gate criteria in plan/road-to-100.md or archive.',
      'Do not implement unless profiling justifies; mark ACC in roadmap.',
    ],
    ['npm run typecheck'],
    ['F4 decision documented as deferred or scoped.'],
    ['RT100-LD-SAMPLING-01'],
  ),
  rt(
    'RT100-F5-VOLUMES',
    12,
    'P3',
    'ACC',
    'F5: heterogeneous volumes (null-collision delta tracking) — product-gated; stained-glass may never need (road-to-100 post-100).',
    ['plan/road-to-100.md', 'packages/core/src/scene/'],
    [
      'Confirm product scope with user before contract extension.',
      'If in scope: add AnalyticShape/Material.extensions volume primitive first.',
      'Else: document permanent unsupported + planner routing.',
    ],
    ['npm run typecheck'],
    ['F5 scope decision recorded.'],
  ),
  rt(
    'RT100-F-BRIDGE',
    12,
    'P3',
    'DOC',
    'F-BRIDGE: experimental no-hardware-RT bridge levers — track as research backlog, not 100% blockers (road-to-100 §F-BRIDGE table).',
    ['plan/road-to-100.md', 'plan/roadmap.md'],
    [
      'Ensure F-BRIDGE table remains in road-to-100 with feasibility notes.',
      'Cross-link active performance track (LD-SAMPLING, WBVH) vs bridge items.',
      'No implementation unless promoted by user.',
    ],
    ['npm run typecheck'],
    ['F-BRIDGE backlog visible and separated from Phase 0–6 closure.'],
  ),
];

export const RT100_TASKS = [
  ...RT100_VALIDATION,
  ...RT100_IMPLEMENTATION,
  ...RT100_PROOF,
  ...RT100_DECIDE,
  ...RT100_SOTA,
];
