# GPU-gated follow-up — complexity-remediation sweep (2026-05-28)

> **SUPERSEDED / HISTORICAL (2026-06).** The deferred items below (A3, T16, T5,
> directional→DDGI, Möller, DDGI border-texel, pt-webgl instanced-mesh, …) have
> since shipped and been GPU-validated — the authoritative live record is
> `HARDWARE-VALIDATION-NEEDS.md` (V1–V27) + `git log` on `main`. Kept for history;
> do not treat the "PENDING" framing below as current.

The 2026-05-28 complexity sweep (`feat/w0-complexity-remediation-foundation`, 11 commits, all 5 bugs + 13 themes + Wave-4 capability reconciliation + GpuResources) is **code-complete and green** (typecheck + `npm test`, all per-wave + whole-branch audits clean). This doc tracks the work that was deliberately **deferred because it requires a hardware-WebGPU machine** (the dev WSL2 box has only SwiftShader-class limits — `hybridCanRun: false`, pt-webgpu lite-tier only — so radiometric/GPU-compile validation cannot run there) or is a follow-on feature.

## 1. Reference-render validation (PENDING — needs hardware GPU)
The testing protocol's before/after A/B was not run for the sweep's radiometric changes (their math is pinned by unit tests + audits, but the visual confirm is outstanding):
- **A3** BDPT bounce-0 tangent fix — render a BDPT scene on `main` vs branch, both pt-webgl + pt-webgpu (full tier).
- **T16** DDGI emitter radiometry (chroma + `4·|u×v|` area) — Cornell-style colour-bleed scene, hybrid backend.
- **T5** stained-glass opt-in module — Cornell-SG with `stainedGlass:{sunCaustic,skyAperture}` ON must match pre-sweep bit-for-bit (flag-ON WGSL is already byte-identical, so this is the least necessary); generic scene flags-OFF must zero the terms.
Harness: `tools/benchmark-runner/` `capture-adapter-playwright.mjs` + `benchmark:gap-closure` (checkout main → baseline, branch → candidate, runner diffs/PSNR). On WSL, route to Windows Chrome via the `*-win-chrome.mjs` variants.

## 2. directional → DDGI sun (decision: track)
A scene `directional` emitter currently produces no DDGI light — the sun is config-driven (`primaryLightDir`/`primaryLightIntensity` via constructor/`updateLighting`), and the packer's sun path hardcodes direction `(0,-1,0)`. Wire it properly: (a) carry the real `primaryLightDir` into the DDGI sun path, (b) single-count directional (resolve the setSunIntensityMultiplier-vs-fixture double-count), then re-add `directional` to walkaround's `supportedEmitterKinds` (+ ledger). Radiometric → needs reference render.

## 3. Deferred structural refactors (decision: GPU follow-up)
- **T9-stepC** per-pass WGSL `requires`-narrowing: the W1 split made `common.wgsl` 11 modules but all passes still pull the full `common` aggregate (byte-identity preserved). Narrowing each pass to its minimum module set needs either per-pass `createShaderModule` (GPU) or a static WGSL ident-resolution check, plus changing the `wgslCompose` byte-identity test contract. A wrongly-narrowed pass only fails at GPU compile.
- **Möller-Trumbore unify** (pt-webgpu): pt-webgpu keeps its own `intersectTriangle` (different barycentric formula + `f32` return vs the canonical `IntersectionResult`); unifying changes edge-case intersection numerics and must update the `cpuTracer.ts` oracle → needs a render A/B. Requires unifying the `IntersectionResult`/`SceneHit` hit contract across backends first.

## 4. Pre-existing bug (decision: GPU follow-up)
- **DDGI IRR border-texel fill**: the irradiance border pass covers positions `[0,96)` of a 10×10 cell, leaving the 4 bottom-edge texels (`lx∈{6,7,8,9}, ly=9`) unfilled — pre-existing, faithfully preserved through the W1 border-factory refactor (`probeUpdateBorder.wgsl.ts`). Radiometric (probe-atlas border) → needs reference render to validate the fix.

## 5. Feature follow-up (surfaced by the Wave-4 audit)
- **pt-webgl instanced-mesh**: pt-webgl declares `instanced-mesh` UNSUPPORTED (warn-skipped) because the fork's `StaticGeometryGenerator`/`convertToStaticGeometry` ignores `instanceMatrix`. To genuinely support it, expand the InstancedMesh into N baked meshes on the pt-webgl side (NOT in the shared `instancedMeshPrimitiveToThree` — walkaround needs the single InstancedMesh for its TLAS path), then re-declare it.

## Watch (not bugs)
HybridEngine.ts ~1371 LOC + HybridEnginePrimitiveUpdates.ts ~994 LOC are large-but-cohesive (meaningfully reduced this sweep, no longer mixed-concern). `struct GBufferSample` is dead in `reservoirDi.wgsl` (dead pre-split; left under the byte-identity contract).
