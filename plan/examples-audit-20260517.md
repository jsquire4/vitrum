# `examples/` completeness audit — 2026-05-17

Read-only audit of every demo in `examples/` for completeness, consistency, and
buildability. Trivial fixes (README typos, missing manifest fields) applied
in-place; non-trivial findings recorded here only.

## Status update — 2026-05-18

Per-example findings rechecked against current `main`:

- **`cornell-box` typecheck NO** → **YES** as of `ac1b593` (`fix(examples,pt-webgpu): typecheck cornell-box + two-engines-one-scene + pt-webgpu process ref`). `npm run typecheck --workspace @vitrum-examples/cornell-box` runs clean. Typecheck script is present in `package.json`.
- **`two-engines-one-scene` typecheck NO** → **YES** (same `ac1b593` fix; the `pt-webgpu/src/scene/buildCpuBvh.ts` `process.env` reference was reconciled). Typecheck script is present.
- **`hero-product-viz` README typo (50ms vs 80ms)** → confirmed fixed (audit-time in-place fix held).
- **New example added 2026-05-18:** `examples/neural-denoiser` (W10 — neural denoiser example + acceptance test + README, commit `452a0a6`). Not in the audit table below; typechecks clean.
- **Shared-helper hoists** landed after the audit: `mat4FromThree` + `resizeCanvasToDisplaySize` moved to `examples/shared` (`77245ea`); `parsePositiveInt` canonicalised across `hero-product-viz` + `hero-viewer` (`9e77566`).

Coverage-gaps section below is still accurate for items not called out above.

---


## Scope

Six subdirectories audited:

- `examples/cornell-box`
- `examples/hero-lighting-designer`
- `examples/hero-product-viz`
- `examples/hero-viewer`
- `examples/shared` (utility — not a runnable demo)
- `examples/two-engines-one-scene`

## Per-example summary

| Example | Demonstrates | README OK? | Typecheck OK? | Notes |
| --- | --- | --- | --- | --- |
| `cornell-box` | `pt-webgl` + `three-bindings` + (URL-param) `shared-denoisers` (OIDN / WGSL bilateral / SVGF) | drifted — see findings | **NO** (8 errors in `src/main.ts` + `src/denoiseDisplay.ts`) | No `typecheck` script in `package.json`; depends on `shared-denoisers` but README doesn't mention it. Mixes capture-harness + URL-param flags + multiple denoise paths — heaviest example by far. |
| `hero-lighting-designer` | `attachVitrum({ prefer: 'realtime' })` + slider-driven `setScene` / `reset` debounce + `onFrame` telemetry | yes | yes | Clean, single-concern. Procedural scene. |
| `hero-product-viz` | `attachVitrum({ prefer: 'quality' })` + `onProgress({ kind: 'pt-spp' })` + offscreen high-res render via `createEngine()` | **had typo (50ms vs 80ms)** — fixed | yes | Open TODO in README: "swap in a polished CC0 hero asset from Sketchfab and cite source." |
| `hero-viewer` | `attachVitrum()` + drag-drop `loadGltfScene()` + `prefer: 'realtime' \| 'quality'` toggle | yes | yes | Open TODO in README: "ship a small CC0 procedural `.glb` as a default scene." `getCurrentScene()` helper indirection (lines 119–146) only exists because the engine handle doesn't expose its current scene — minor architectural smell, not a bug. |
| `shared` | THREE.js fixture scenes (Cornell + complex) used by `two-engines-one-scene` | no README | yes | Utility package, not a runnable demo. Not listed in root README's examples section (correctly). |
| `two-engines-one-scene` | One `vitrumScene` driving `pt-webgl` + `walkaround-hybrid` + `pt-webgpu` simultaneously (G2 demo) | yes | **NO** (2 errors in `src/main.ts` + 2 leak-through errors from `pt-webgpu/src/scene/buildCpuBvh.ts` `process.env` reference with no `node` types) | No `typecheck` script in `package.json`. 4 HTML entrypoints (`index.html`, `pt-webgl.html`, `walkaround.html`, `walkaround-webgl2.html`, `pt-webgpu.html`) selected via `body[data-engine-mode]` + URL params. Heaviest of the demos (~490 LoC `main.ts`). |

## Coverage gaps

Public packages with **no example demonstrating them** as a primary feature:

- **`@vitrum/dev`** — debug overlay components (FrameTimeHUD, MaterialInspector, …) per root README. Zero examples import from this package. The README sells it as the dev-overlay surface; without an example demo a consumer has no reference for wiring it up.
- **`@vitrum/shared-bvh`** — internal utility used by `walkaround-hybrid`, `pt-webgpu`. No example consumes it directly. Likely intentional (the contract claim is that it stays internal), but worth flagging if direct CPU-BVH usage is ever a public API surface.
- **`@vitrum/shared-samplers`** — internal utility used by backends. No example exercises Sobol / Hammersley / light-tree / hero-MIS at the public-API level. The hero-wavelength MIS path lives behind a `vitrum.ptWebgl.spectralRendering` extension flag (cornell-box exposes it via URL param `?vitrumScenario=spectral`) — that's the closest thing to a demo.
- **`@vitrum/shared-denoisers`** — *partially* covered. `cornell-box` dynamic-imports `denoiseFinal` (OIDN), `runHdrLuminanceBilateralWebGPU`, and `runAtrousVarianceWebGPU` under URL params (`vitrumDisplay=oidn|wgsl|svgf`). No standalone "denoiser comparison" demo exists.
- **`@vitrum/pt-webgpu`** — listed by the project brief as "pre-alpha prototype WebGPU PT — internal, not production". `two-engines-one-scene` drives it headlessly (no swap-chain present; accumulates to internal HDR texture). No standalone interactive example. Consistent with its internal status.

The four currently-shipping runnable demos cleanly cover `@vitrum/engine` (drop-in facade), `@vitrum/core` (types/contract), `@vitrum/three-bindings` (scene conversion + glTF loader), `@vitrum/pt-webgl` (WebGL2 PT), and `@vitrum/walkaround-hybrid` (real-time GI). That's the prime-time surface — the gaps above are either dev tooling (`dev`), internal utilities (`shared-bvh`, `shared-samplers`), or pre-alpha (`pt-webgpu`).

## Non-trivial findings

### F1 — `cornell-box` and `two-engines-one-scene` have no `typecheck` script

The other example packages (`hero-*`, `shared`) all declare `"typecheck": "tsc --noEmit"`. The two missing it are the two whose sources don't currently typecheck:

**`cornell-box` (8 errors):**

```
src/denoiseDisplay.ts(100,5): error TS2532: Object is possibly 'undefined'.
src/denoiseDisplay.ts(101,5): error TS2532: Object is possibly 'undefined'.
src/main.ts(480,7): error TS18047: 'canvas' is possibly 'null'.
src/main.ts(481,7): error TS18047: 'canvas' is possibly 'null'.
src/main.ts(576,7): error TS18047: 'canvas' is possibly 'null'.
src/main.ts(580,7): error TS18047: 'canvas' is possibly 'null'.
src/main.ts(607,11): error TS18047: 'canvas' is possibly 'null'.
src/main.ts(615,11): error TS18047: 'canvas' is possibly 'null'.
```

Root cause: `denoiseDisplay.ts` indexes a uniforms record without re-narrowing; `main.ts` captures `canvas` from `document.querySelector(...)` after a null check but the closure-capture pattern (used inside `resize()` / `triggerDenoise(...)`) loses the narrowing. A localized fix would be a non-null assertion on a captured `const canvasRef = canvas!` at the top of the function, or proper hoisted narrowing.

**`two-engines-one-scene` (2 own errors + 2 leak-through from `pt-webgpu`):**

```
src/main.ts(458,20): error TS18047: 'canvasPtGpu' is possibly 'null'.
src/main.ts(459,21): error TS18047: 'canvasPtGpu' is possibly 'null'.
../../packages/pt-webgpu/src/scene/buildCpuBvh.ts(373,14): error TS2591: Cannot find name 'process'.
../../packages/pt-webgpu/src/scene/buildCpuBvh.ts(373,41): error TS2591: Cannot find name 'process'.
```

The `canvasPtGpu` failures are similar narrowing-loss patterns. The `process.env` reference inside `packages/pt-webgpu/src/scene/buildCpuBvh.ts` is a runtime `typeof process !== 'undefined'` guard but the type isn't declared — when typechecked via the example's tsconfig path-alias (which pulls pt-webgpu's source files in directly), the example inherits the missing-`@types/node` issue. The example's tsconfig doesn't include `"types": ["@webgpu/types"]` either, so the error count would grow if the pt-webgpu typecheck didn't already filter.

Adding `"typecheck": "tsc --noEmit"` is a one-line manifest change but would break workspace-wide `npm run typecheck` (currently green). Fix is **not trivial**: the source must first compile cleanly, ideally through proper narrowing rather than `!` non-null assertions, and the pt-webgpu `process` reference needs to be guarded with a typed shim or moved behind a `@vitrum/dev` runtime flag.

### F2 — `cornell-box` README content drift

The README declares the example "Validates `@vitrum/core → @vitrum/three-bindings → @vitrum/pt-webgl`". The `package.json` now also lists `@vitrum/shared-denoisers`, and the source dynamic-imports three denoiser entrypoints (`denoiseFinal`, `runHdrLuminanceBilateralWebGPU`, `runAtrousVarianceWebGPU`) under URL parameters (`?vitrumDisplay=oidn|wgsl|svgf`).

The example is also the de-facto regression-test scene (capture-harness hooks `VITRUM_CAPTURE_READY` / `VITRUM_MS_PER_SAMPLE` / `VITRUM_CAPTURE_TELEMETRY` are read by `tools/benchmark-runner/capture-adapter-playwright.mjs`). The README mentions none of this.

Recommend splitting into either:
1. A two-section README that documents both the "what it shows" and the "what the capture harness reads" responsibilities, or
2. Splitting the file into a minimal `examples/cornell-box` (the README's stated contract) and a separate `examples/cornell-capture-harness` (or moving the capture-mode plumbing to `tools/benchmark-runner/`). The capture-harness path is heavy (URL flags, preset table, multiple async denoiser flows, telemetry exposure) and dilutes the "minimal pt-webgl consumer" framing.

### F3 — `hero-viewer` and `hero-product-viz` open TODOs in README

Both Hero examples explicitly admit a missing default asset:

- `hero-viewer/README.md` line 34: "TODO: ship a small CC0 procedural `.glb` as a default scene so the viewer has something to show without user action."
- `hero-product-viz/README.md` line 49: "TODO: swap in a polished CC0 hero asset from Sketchfab and cite source."

For demos sold as the "public face of how do I use vitrum?", first-load-with-nothing-to-show is a real onboarding friction. These are explicitly tagged TODO so they're not a finding so much as an open commitment.

### F4 — `examples/shared` has no README

It's a utility package — fine — but a one-line README ("Shared THREE.js demo scenes used by `two-engines-one-scene`; not a runnable demo") would close the discovery gap for a new agent landing here from a `ls examples/`.

### F5 — `getCurrentScene` indirection in `hero-viewer`

`hero-viewer/src/main.ts` lines 119–146 store the last-loaded vitrum Scene in a module-level `lastVitrumScene` and define a helper `getCurrentScene` that takes the engine handle but ignores it and returns the stored scene. The helper exists because `AttachVitrumHandle` doesn't expose its current scene.

Either:
- The engine handle should expose `handle.scene` (read-only) so consumers can switch backends without bookkeeping, or
- The helper should be inlined and the dead `_h` parameter removed.

Not a bug — minor smell that documents an API gap.

### F6 — Capture-harness coupling in `cornell-box` is undocumented at the example level

`cornell-box/src/main.ts` exposes three globals (`VITRUM_CAPTURE_READY`, `VITRUM_MS_PER_SAMPLE`, `VITRUM_CAPTURE_TELEMETRY`) read by `tools/benchmark-runner/capture-adapter-playwright.mjs`. The example itself doesn't link to the harness, and the harness lives in a sibling tree. Cross-tree contracts of this kind should have at least a forward-link in the README ("the capture-harness lives at `tools/benchmark-runner/`; see that package for the side of the contract this example fulfills").

## Trivial fixes applied

| File | Change |
| --- | --- |
| `examples/hero-product-viz/README.md` | Fixed self-contradiction: "(50ms debounce)" → "(80ms debounce)" to match the code (`hero-product-viz/src/main.ts:149` uses `80` ms) and the "What it demonstrates" section, which already said "(debounced 80ms)". |

Workspace `npm run typecheck` re-run post-fix: green (no regression).

## Process check (verified)

- `npm run typecheck` at workspace root **passes** — `cornell-box` and `two-engines-one-scene` are silently skipped via `--if-present` because they don't declare a `typecheck` script.
- All four runnable examples have matching `dev` / `build` / `preview` scripts, `private: true`, and `description` set in `package.json`.
- All `file:../../packages/*` deps resolve to existing directories.
- Vite `dedupe: ['three', 'three-mesh-bvh', 'three-gpu-pathtracer']` is consistent across all examples (necessary because `three-gpu-pathtracer` is consumed via `file:../../../three-gpu-pathtracer`).
- Port assignments per `vite.config.ts`: cornell-box 5174, two-engines 5175, hero-viewer 5176, hero-lighting-designer 5177, hero-product-viz 5178. Each README's "open http://localhost:NNNN" matches its vite config.

## Recommendations summary

1. **Highest impact** — Fix `cornell-box` and `two-engines-one-scene` source narrowing so they can declare `typecheck` scripts. The workspace currently green-lights two examples that don't compile in strict mode — that's a silent gap that lets contract drift slip in.
2. **Medium impact** — Split `cornell-box` capture-harness plumbing from the "minimal pt-webgl consumer" demo, or document the harness coupling in the README.
3. **Medium impact** — Ship default CC0 assets for `hero-viewer` and `hero-product-viz` (both have open README TODOs).
4. **Low impact** — Add a README to `examples/shared`; expose `handle.scene` on `AttachVitrumHandle` and remove the `getCurrentScene` indirection in `hero-viewer`; consider whether `@vitrum/dev` warrants a dedicated example or remains demonstrated only via in-app dev tooling.

## Inventory: package coverage matrix

| Package | Example primary | Example secondary (URL-param / inline) |
| --- | --- | --- |
| `@vitrum/core` | all | — |
| `@vitrum/engine` | hero-viewer, hero-lighting-designer, hero-product-viz | — |
| `@vitrum/three-bindings` | cornell-box, hero-viewer, hero-* | two-engines-one-scene |
| `@vitrum/pt-webgl` | cornell-box | hero-* (via `prefer: 'quality'`), two-engines-one-scene |
| `@vitrum/pt-webgpu` | — | two-engines-one-scene (headless) |
| `@vitrum/walkaround-hybrid` | hero-lighting-designer | hero-viewer (via `prefer: 'realtime'`), two-engines-one-scene |
| `@vitrum/shared-denoisers` | — | cornell-box (URL-param OIDN / WGSL bilateral / SVGF) |
| `@vitrum/shared-bvh` | — | — (internal) |
| `@vitrum/shared-samplers` | — | — (internal; hero-MIS via `vitrumScenario=spectral` indirect) |
| `@vitrum/dev` | — | — |
