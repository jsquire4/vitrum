# GPU Validation Gate — Scoping (2026-05-29)

> Goal: turn the one-off "we got lucky catching those 4 GPU bugs" into a **standing gate** —
> so the bug class (format/layout/naga conformance, default-render regressions, radiometric
> drift) **can't merge**. Scoped against the *actual* `wsl-gpu` harness, not assumptions.

## 1. What already exists (verified by reading both repos)

- **`wsl-gpu`** (`github.com/jsquire4/WSL-GPU`, sibling repo) is a mature, autonomous headless
  capture + A/B harness running entirely in WSL2:
  - **dzn** (RTX 4090 via Mesa-on-D3D12 + a ~250-line conformance-spoof Vulkan layer) — hardware,
    ~20–26× faster than software, clears the hybrid + pt-webgpu-full tier. The fast A/B path.
  - **lavapipe** (CPU software Vulkan, *conformant*) — the verification oracle. **Byte-identical
    to dzn for pt-webgpu (999 dB)**, so dzn is provably not miscompiling.
  - **headless-Chromium WebGL2** (software GL) — for pt-webgl rows (V6/V9). Correctness-valid, not perf.
  - **MCP server** (`probe`/`capture`/`compare`/`run_validation`) + a warm Deno wgpu worker.
  - **vitrum pin**: `scripts/pin-vitrum.sh` freezes a vitrum commit into `~/.cache/wsl-gpu/vitrum-pinned`
    (auto-tracks main tip, parse-guarded).
- It has **already validated** V2/V4/V7/V8/V13/V16/V18(BDPT-ON)/V19(GRIS-ON)/V20(NRC) across Waves 8–12,
  and cross-checked HYBRID transforms vs conformant Windows-Chrome/Dawn (46.38 dB). The earlier
  "blocked on conformant Dawn" notes are **superseded** — BDPT-ON and GRIS-ON pass on the harness now.
- **vitrum CI** (`.github/workflows/ci.yml`) already runs the no-GPU tier: typecheck, vitest,
  `fork-shader-smoke`, `shader-compile-ci` (Playwright Chromium), release dry-run.

## 2. The constraint that shapes the whole design

The GPU lives on the **local WSL2 box**. GitHub `ubuntu-latest` runners have **no real GPU** —
Chromium there falls back to SwiftShader (10 buffers / 4 textures) and **fails the hybrid gate**.
So a real-GPU radiometric gate **cannot run in GitHub CI**. The real-GPU gate is inherently a
**local pre-push hook**; GitHub CI keeps the no-GPU mechanical+compile tier it already has.

## 3. Architecture decision: standalone-referenced (confirming the earlier lean)

`wsl-gpu` **stays a standalone sibling repo** that pins+validates vitrum — NOT consumed into vitrum.
Rationale: (a) keeps vitrum's dep graph + npm-publish surface clean; (b) it *can't* run in CI anyway
(needs the local GPU, the dzn conformance-spoof layer, WSLg `/dev/dxg`, the Deno worker); (c) it
versions independently; (d) it already works this way. The work is the **invocation seam** (vitrum →
wsl-gpu) + closing capability gaps, not a dependency.

## 4. Tiered gate

| Tier | Trigger | Where | What it catches | Cost |
|---|---|---|---|---|
| **T0 mechanical** (exists) | every push/PR, auto | GitHub CI | type errors, unit regressions, shader-compile-on-SwiftShader, release health | ~min |
| **T1 GPU smoke** (NEW — cheap, always) | pre-push hook, local | wsl-gpu dzn + lavapipe | (a) **shader-compile on BOTH real backends** — the format/layout/naga-conformance class SwiftShader misses (this is the GRIS black-frame + BDPT `rgba32float read_write` + `layout:'auto'` bug class — all 4 of this wave's GPU bugs); (b) **default-render non-regression A/B** (Cornell default vs a pinned golden, both backends, PSNR ≥ threshold + 0 NaN/Inf) | seconds–1 min on dzn |
| **T2 radiometric A/B** (NEW — expensive, gated) | on-demand / nightly | wsl-gpu (dzn+lavapipe; WebGL2 for pt-webgl rows) | per-subsystem V-item A/Bs (V21–V25 + the fidelity matrix) | minutes |

The split answers your "always is annoying": **T1 is fast enough for every push** and catches the
*catastrophic* class (default render breaks, shader won't compile on a conformant/real device). **T2
is opt-in / nightly** — slow, and only relevant when you touched that subsystem.

## 5. The invocation seam (what to build, vitrum side)

- `npm run validate:gpu:smoke` — pins the **working tree** (not main tip, so it validates the
  *uncommitted pre-push* state), invokes wsl-gpu's runner for T1, exits nonzero on regression.
- `npm run validate:gpu -- --items=V21,V23,…` — T2, explicit items.
- A **git `pre-push` hook** that calls `validate:gpu:smoke` in **WARN-ONLY mode** (DECIDED — §8 option B):
  it runs T1, prints any regression loudly, and **never blocks the push** (always exits 0). Still
  auto-skips when no GPU / no wsl-gpu present (silent no-op for non-WSL contributors + CI). The point
  is *visibility every push* without friction; enforcement stays the human reading the warning.

## 6. Capability gaps to build (wsl-gpu side — new scenarios for this wave's V-items)

- **Default-render golden**: a pinned per-backend Cornell-default capture for the T1 diff + a
  `refresh-baselines` flow (partial: `scripts/refresh-baselines.mjs` exists).
- **Scenarios** (`scenario-builder/` + scenes): V21 low-poly curved mesh (faceted→smooth + no-acne);
  V22 8-light 10:1 spread (variance ↓ + equal converged mean); V23 translucent slab (exit-face
  chrominance + Beer's-law + BDPT-on-unchanged); V24 1-sphere-1-light diff-RT (loss-descent +
  GPU-adjoint==FD); **V25 BDPT race (the headline first catch)**.
- **V25 BDPT-race nuance (real design risk):** a plain output-diff gate may NOT catch a works-by-luck
  workgroup-ordering race (dzn happens to serialize small dispatches — which is *why* V18 passed).
  The fix must be validated against a **CPU-BDPT reference** (the `shared-samplers` Veach oracle) and/or
  a **deep-subpath stress config** where mis-ordering corrupts the chain — not just "renders non-black."

## 7. Sequencing

1. **Settle the pre-push-hook decision** (§8).
2. Build the vitrum-side `validate:gpu[:smoke]` seam + the `pre-push` hook (+ auto-skip).
3. Stand up **T1** (compile-both-backends + default-render golden) — the standing gate.
4. Implement the **BDPT race fix** + validate it as T2's **first real catch (V25)** — the proof the gate works.
5. Author V21–V24 scenarios; backfill the radiometric matrix to `supported`.
6. (Optional, later) wire T2 nightly.

## 8. Decision — DECIDED: B (soft warn-only)

**Pre-push hook hardness:** the `pre-push` hook runs T1 and prints regressions **loudly but never
blocks** (always exits 0; auto-skips with no GPU). Rationale (user's call): visibility on every push
without the friction of a hard block; the human reading the warning is the enforcement. If the
warn-only signal turns out to be ignored in practice, hardening to a block later is a one-line change
in the hook. (Options A hard-block / C manual-only were considered and not chosen.)
