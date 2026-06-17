# Code Gap Parallel Execution Schedule

> **473 tasks** in **70 waves** (8 bootstrap + 465 scheduled).
> **Agent cap: 25 per wave** — dispatch at most 25 agents; each wave lists ≤25 tasks.
> Skipped duplicates: PTGL-003, PTWG-037, WH-034, CORE-041
> Deferred validation (code-first): see plan/VALIDATION-DEFERRED.md (28 tasks)

## How to run in parallel

1. **Pool size:** maintain **≤25 concurrent agents**. Never exceed the task count in the current wave.
2. **Wave W000 (bootstrap):** 8 `P0-*` tasks — all parallel, then integrate.
3. **W001+:** spawn up to **25 agents** (or fewer if the wave has fewer tasks). Each agent claims one `taskId` from the current wave JSON.
4. After **every** wave: run integration (`npm run typecheck` + lane tests). Proceed only when green.
5. **Never** edit a file another active agent owns. **Never** start tasks from the next wave early.

## Stats

| Metric | Value |
|--------|------:|
| Active tasks | 473 |
| Waves | 70 |
| Agent cap (per wave) | 25 |
| Largest wave (tasks) | 25 |
| Waves with parallelism >1 | 26 |
| Avg tasks/wave | 6.76 |

### Tasks per lane (worker pool sizing)

| Lane | Tasks | Suggested workers |
|------|------:|------------------:|
| `walkaround-hybrid` | 134 | 25 |
| `core` | 108 | 25 |
| `pt-webgpu` | 96 | 25 |
| `pt-webgl2` | 33 | 25 |
| `engine` | 33 | 25 |
| `gltf-adapter` | 23 | 23 |
| `shared-bvh` | 8 | 8 |
| `shared-samplers` | 8 | 8 |
| `shared-denoisers` | 6 | 6 |
| `walkaround-rc` | 6 | 6 |
| `dev` | 6 | 6 |
| `tools` | 4 | 4 |
| `scene-lighting` | 3 | 3 |
| `repo-root` | 3 | 3 |
| `stained-glass-extensions` | 2 | 2 |

## Super-batches (integration checkpoints)

| Batch | Waves | Tasks | Max parallel | Lanes |
|-------|-------|------:|-------------:|-------|
| B00 | W000–W000 | 8 | 8 | pt-webgpu, pt-webgl2, walkaround-hybrid, core, tools |
| B01 | W001–W001 | 25 | 25 | core, engine, pt-webgl2, pt-webgpu, walkaround-hybrid |
| B02 | W002–W002 | 25 | 25 | core, engine, pt-webgl2, pt-webgpu, walkaround-hybrid |
| B03 | W003–W003 | 25 | 25 | pt-webgl2, core, pt-webgpu, walkaround-hybrid, gltf-adapter |
| B04 | W004–W004 | 25 | 25 | walkaround-hybrid, pt-webgl2, pt-webgpu, gltf-adapter, core |
| B05 | W005–W005 | 25 | 25 | pt-webgl2, pt-webgpu, walkaround-hybrid, gltf-adapter, core, engine |
| B06 | W006–W006 | 25 | 25 | pt-webgl2, walkaround-hybrid, pt-webgpu, gltf-adapter, core, engine, tools |
| B07 | W007–W007 | 25 | 25 | pt-webgl2, pt-webgpu, walkaround-hybrid, core, engine, gltf-adapter |
| B08 | W008–W008 | 25 | 25 | walkaround-hybrid, pt-webgl2, engine, core, pt-webgpu |
| B09 | W009–W009 | 25 | 25 | pt-webgpu, walkaround-hybrid, core, pt-webgl2 |
| B10 | W010–W010 | 25 | 25 | walkaround-hybrid, core, pt-webgpu |
| B11 | W011–W011 | 25 | 25 | walkaround-hybrid, core, pt-webgpu |
| B12 | W012–W012 | 25 | 25 | core, pt-webgpu, walkaround-hybrid |
| B13 | W013–W013 | 25 | 25 | walkaround-hybrid, core, pt-webgpu, gltf-adapter |
| B14 | W014–W014 | 25 | 25 | core, pt-webgpu, walkaround-hybrid, gltf-adapter, shared-bvh, shared-samplers |
| B15 | W015–W015 | 25 | 25 | core, pt-webgpu, walkaround-hybrid, gltf-adapter, shared-bvh, shared-samplers, shared-denoisers, walkaround-rc, scene-lighting, stained-glass-extensions, dev |
| B16 | W016–W016 | 17 | 17 | core, walkaround-hybrid, gltf-adapter, shared-bvh, walkaround-rc, dev, pt-webgpu, shared-samplers, repo-root, engine |
| B17 | W017–W017 | 6 | 6 | core, gltf-adapter, walkaround-rc, pt-webgpu, repo-root |
| B18 | W018–W018 | 5 | 5 | core, walkaround-rc, pt-webgpu |
| B19 | W019–W019 | 4 | 4 | core, pt-webgpu, repo-root |
| B20 | W020–W020 | 3 | 3 | core, pt-webgpu |
| B21 | W021–W021 | 3 | 3 | core, pt-webgpu |
| B22 | W022–W022 | 2 | 2 | core, pt-webgpu |
| B23 | W023–W023 | 2 | 2 | core, pt-webgpu |
| B24 | W024–W024 | 2 | 2 | core, pt-webgpu |
| B25 | W025–W028 | 5 | 2 | core, pt-webgpu |
| B26 | W029–W032 | 4 | 1 | core |
| B27 | W033–W036 | 4 | 1 | core |
| B28 | W037–W040 | 4 | 1 | core |
| B29 | W041–W044 | 4 | 1 | core |
| B30 | W045–W048 | 4 | 1 | core |
| B31 | W049–W052 | 4 | 1 | core |
| B32 | W053–W056 | 4 | 1 | core |
| B33 | W057–W060 | 4 | 1 | core |
| B34 | W061–W064 | 4 | 1 | core |
| B35 | W065–W068 | 4 | 1 | core |
| B36 | W069–W069 | 1 | 1 | core |

## Wave table (first 40 waves)

| Wave | Parallel | Tasks | Lanes | Task IDs |
|------|:--------:|------:|-------|----------|
| W000 | ✓ | 8 | pt-webgpu, pt-webgl2, walkaround-hybrid, core, tools | P0-001-PTWG-037, P0-002-PTGL-003, P0-003-WH-034, P0-004-DENO-001, P0-005-LEDGER-01 +3 |
| W001 | ✓ | 25 | core, engine, pt-webgl2, pt-webgpu, walkaround-hybrid | CORE-001, CORE-002, CORE-003, CORE-006, CORE-008 +20 |
| W002 | ✓ | 25 | core, engine, pt-webgl2, pt-webgpu, walkaround-hybrid | CORE-004, CORE-005, CORE-007, CORE-009, ENG-006 +20 |
| W003 | ✓ | 25 | pt-webgl2, core, pt-webgpu, walkaround-hybrid, gltf-adapter | PTGL-004, PTGL-008, FP-03, MUT-11, MUT-12 +20 |
| W004 | ✓ | 25 | walkaround-hybrid, pt-webgl2, pt-webgpu, gltf-adapter, core | FP-06, MUT-01, PTWG-003, WH-003, WH-010 +20 |
| W005 | ✓ | 25 | pt-webgl2, pt-webgpu, walkaround-hybrid, gltf-adapter, core, engine | MUT-02, PTWG-006, WH-004, WH-023, WH-037 +20 |
| W006 | ✓ | 25 | pt-webgl2, walkaround-hybrid, pt-webgpu, gltf-adapter, core, engine, tools | MUT-03, MUT-08, PTWG-007, WH-005, WH-024 +20 |
| W007 | ✓ | 25 | pt-webgl2, pt-webgpu, walkaround-hybrid, core, engine, gltf-adapter | MUT-04, PTWG-012, WH-006, WH-030, WH-038 +20 |
| W008 | ✓ | 25 | walkaround-hybrid, pt-webgl2, engine, core, pt-webgpu | WH-007, WH-031, WH-039, RT100-PTGL-MUT, RT100-GLTF-PICK +20 |
| W009 | ✓ | 25 | pt-webgpu, walkaround-hybrid, core, pt-webgl2 | MUT-06, WH-019, WH-040, WH-045, CORE-017 +20 |
| W010 | ✓ | 25 | walkaround-hybrid, core, pt-webgpu | RT100-WA-3D, CORE-018, PTWG-023, PTWG-029, PTWG-052 +20 |
| W011 | ✓ | 25 | walkaround-hybrid, core, pt-webgpu | RT100-WA-ALPHA, CORE-019, PTWG-030, PTWG-060, PTWG-067 +20 |
| W012 | ✓ | 25 | core, pt-webgpu, walkaround-hybrid | CORE-020, PTWG-031, PTWG-062, PTWG-071, WH-049 +20 |
| W013 | ✓ | 25 | walkaround-hybrid, core, pt-webgpu, gltf-adapter | MUT-10, CORE-021, PTWG-032, PTWG-074, WH-053 +20 |
| W014 | ✓ | 25 | core, pt-webgpu, walkaround-hybrid, gltf-adapter, shared-bvh, shared-samplers | CORE-022, PTWG-033, PTWG-075, WH-056, WH-088 +20 |
| W015 | ✓ | 25 | core, pt-webgpu, walkaround-hybrid, gltf-adapter, shared-bvh, shared-samplers, shared-denoisers, walkaround-rc, scene-lighting, stained-glass-extensions, dev | CORE-023, PTWG-064, PTWG-076, WH-107, WH-115 +20 |
| W016 | ✓ | 17 | core, walkaround-hybrid, gltf-adapter, shared-bvh, walkaround-rc, dev, pt-webgpu, shared-samplers, repo-root, engine | CORE-029, WH-128, GLTF-014, GLTF-019, SBVH-008 +12 |
| W017 | ✓ | 6 | core, gltf-adapter, walkaround-rc, pt-webgpu, repo-root | CORE-030, GLTF-015, RC-003, INV-002, INV-005, RT100-F4-WAVEFRONT |
| W018 | ✓ | 5 | core, walkaround-rc, pt-webgpu | CORE-031, RC-006, INV-003, INV-006, RT100-F5-VOLUMES |
| W019 | ✓ | 4 | core, pt-webgpu, repo-root | PTWG-072, INV-007, INV-011, RT100-F-BRIDGE |
| W020 | ✓ | 3 | core, pt-webgpu | MAT-WH-baseColor, INV-008, INV-012 |
| W021 | ✓ | 3 | core, pt-webgpu | MAT-WH-roughness, INV-009, INV-013 |
| W022 | ✓ | 2 | core, pt-webgpu | MAT-WH-metallic, INV-010 |
| W023 | ✓ | 2 | core, pt-webgpu | MAT-WH-shadingModel, INV-014 |
| W024 | ✓ | 2 | core, pt-webgpu | MAT-WH-alphaMode, INV-015 |
| W025 | ✓ | 2 | core, pt-webgpu | MAT-WH-alphaCutoff, INV-016 |
| W026 | · | 1 | core | MAT-WH-opacity |
| W027 | · | 1 | core | MAT-WH-transmission |
| W028 | · | 1 | core | MAT-WH-ior |
| W029 | · | 1 | core | MAT-WH-attenuationColor |
| W030 | · | 1 | core | MAT-WH-attenuationDistance |
| W031 | · | 1 | core | MAT-WH-thickness |
| W032 | · | 1 | core | MAT-WH-baseColorMap |
| W033 | · | 1 | core | MAT-WH-normalMap |
| W034 | · | 1 | core | MAT-WH-normalScale |
| W035 | · | 1 | core | MAT-WH-roughnessMap |
| W036 | · | 1 | core | MAT-WH-metallicMap |
| W037 | · | 1 | core | MAT-WH-transmissionMap |
| W038 | · | 1 | core | MAT-WH-thicknessMap |
| W039 | · | 1 | core | MAT-WH-emissiveMap |
| … | | | | _30 more waves — see JSON_ |

## Lane worker charters

### `core` (108 tasks)

Contract, promiseLedger, capabilities, inverse types. **Mutex:** promiseLedger.ts — max 1 agent globally.

### `engine` (33 tasks)

createEngine, VitrumCanvas, progressive handoff, backends.

### `pt-webgl2` (33 tasks)

WebGL2 PT backend, GLSL, scene mutations.

### `pt-webgpu` (96 tasks)

WebGPU PT, inverse/adjoint, specialty integrators.

### `walkaround-hybrid` (134 tasks)

Realtime GI — largest lane. Sub-lanes: restir/, ddgi/, shaders/, pipeline/.

### `gltf-adapter` (23 tasks)

glTF ingest, featureReport, compression.

### `shared-bvh` (8 tasks)

BVH shared package — independent.

### `shared-samplers` (8 tasks)

Samplers shared — independent.

### `shared-denoisers` (6 tasks)

Denoisers shared — independent.

### `walkaround-rc` (6 tasks)

RC subsystem — independent.

### `tools` (4 tasks)

behavioral-gate, shader-gate, benchmarks.

### `dev` (6 tasks)

Debug overlays — independent.

