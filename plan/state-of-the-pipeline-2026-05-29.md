# State of the vitrum rendering pipeline — 2026-05-29

A full snapshot: where the pipeline is, everything that landed in this work wave, what
remains, and an honest estimate of the gap to a *fully professional* rendering pipeline.
Authoritative live state is always `git log` on `main` + `CLAUDE.md` + `HARDWARE-VALIDATION-NEEDS.md`;
this is a point-in-time consolidation.

---

## 1. Where the pipeline is

vitrum is a host-agnostic WebGPU/WebGL2 path-tracing + global-illumination engine with **two
rendering stacks** behind one `@vitrum/core` contract:

- **Real-time stack — `@vitrum/walkaround-hybrid` (WebGPU).** DDGI (coherent physical model) +
  ReSTIR-DI + ReSTIR-GI + Radiance Cascades + PPG (guided) + GTAO + SVGF + OIDN, and — as of this
  wave — **ReGIR** (grid light reservoirs), **GRIS/ReSTIR-PT** reconnection-shift reuse, and **NRC**
  (neural radiance cache, live query+train). Opt-in subsystems gate cleanly (full vs lite tier).
- **Hero/converged stack.** `@vitrum/pt-webgl` (WebGL2, the absorbed three-gpu-pathtracer fork) +
  `@vitrum/pt-webgpu` (native WebGPU peer). BDPT (full Veach §10.3 MIS), spectral (Jakob-Hanika),
  thin-film, SSS, caustics, BMFR/OIDN denoise.

Maturity: **release-candidate track** for `@vitrum/engine`, `walkaround-hybrid`, `pt-webgl`;
`pt-webgpu` is a peer PT backend with several fidelity rows still tagged `experimental`. Mechanical
health: workspace `tsc --noEmit` clean; full `npm test` green across packages.

**The single most important shift this wave:** correctness moved from "unit-pinned, GPU-unverified"
to **"GPU-validated on real hardware (RTX 4090 via Mesa dzn) cross-checked against a software
conformance oracle (lavapipe)."** That campaign caught four real bugs that *every* unit/oracle test
was blind to (see §2). The validation harness (`../wsl-gpu`) is now a genuine capability, not a stub.

---

## 2. What landed in this wave (≈40 commits)

### New algorithms / features
- **BDPT full Veach §10.3 connection MIS on GPU** (`b21439e`) — WGSL/GLSL mirror the
  `shared-samplers` oracle to ≤1e-12; fixed the biased oracle first (`6360b8b`, PBRT MISWeight);
  per-pixel eye-vertex scratch stack; latent primary-hit double-count fixed.
- **ReGIR** grid-based many-light reservoirs for ReSTIR-DI (`5929f92`) — co-located in the
  light-tree buffer to stay within the 16-storage-buffer floor; unbiased `pSel` into the RIS divisor.
- **GRIS / ReSTIR-PT** — Phase 0 reservoir widening + reconnection-shift oracle (`5e7b8bb`),
  Phases 1+2 GPU shift + visibility + pairwise-MIS (`f8df9a4`), opt-in `restirPtReuse`.
- **NRC (Müller 2021)** — fused/tiled MLP train kernel (`2e82c52`), staged foundation (`92fcff6`),
  live query+train passes with hash-grid encoding (`fcca54f`), device-capability gate (`29daab6`).
- **Topology + incremental** — explicit `addPrimitive`/`removePrimitive` on all 3 backends
  (`34b82f4`/`43561b4`/`163eaee`); incremental vertex/index/instance-count topology
  (`968a27b`/`54fdfd0`/`b864f00`/`8b819d8`); `onProgress` stats (`f3453c2`).
- **Fidelity-combination fixes** — true quarter-res GTAO (`dfb0ec0`); light-tree ReSTIR-DI +
  PPG scheduling (`adee4c7`); RC trilinear cascade-0 (`1915a0c`); Disney sheen/clearcoat/iridescence
  (`8ccb0f5`); Jakob-Hanika spectral now CONSUMED in pt-webgl (`c873ecc`); SVGF dropped from the
  converged tracer (`ad6b753`); RC⊕ReSTIR-GI per-pixel confidence MIS (`5953139`); resolutionFactor
  upscale + load-bearing DDGI cadence (`c5e90f8`); PPG bounded readback + sTree growth fix (`70465dd`).
- **Differentiable-RT roadmap** captured as a phased, gated plan (`988d9e6`, `plan/differentiable-rt.md`).

### The GPU-validation campaign (WAVE8 → WAVE12) — and the 4 bugs it caught
Built real harness capabilities (WebGL2 headless capture; a naga-gap WGSL shim with faithful
alpha-rename; Dawn-conformance reasoning) and validated **V2/V4/V5/V6/V12/V13/V14/V15/V16/V20 PASS**
(+ V9 consumption-confirmed). It then surfaced and we fixed **four real shipped-but-broken paths
that unit tests could not see**:

1. **GRIS default-render black-frame regression** (`b2ed2e3`) — `f8df9a4` added a `@group(1)` bind
   group *unconditionally*, structurally altering the default pipeline → all-black on both backends
   even gate-OFF. Fix: compile-time-conditional gating + a structural guard test.
2. **BDPT light path non-conformant** (`fdb0b95`) — `rgba32float read_write` storage texture is not
   core WebGPU (gpuweb #4651); rejected at bind-group creation everywhere. Fix: → storage buffer.
3. **BDPT pipeline-layout exclusivity** (`6bb11f7`) — both compute pipelines used `layout:'auto'`
   (pipeline-exclusive), so BDPT couldn't share bind groups. Fix: explicit shared `GPUPipelineLayout`.
   With 2+3, BDPT-ON now **dispatches + renders** on real hardware (it never had).
4. **GRIS-ON biased + divergent** (`cc86c39`) — indirect mean climbed *unbounded* (the earlier
   "+13%" was an LDR-clamp artifact). Two over-energization bugs: a `/pSrc` double-discount in the
   reused-reservoir weight, and a non-unity streaming-pairwise MIS. Fix: drop `/pSrc` + exact
   generalized balance heuristic. GPU-confirmed: mean stable, temporal-only converges to ground
   truth, variance ≤ GRIS-OFF, both backends.

**Meta-lesson (the load-bearing takeaway):** oracle-parity, partition-of-unity, and structural unit
tests are *necessary but not sufficient*. Pipeline-structure faults, format conformance, and
temporal-recursion divergence only manifest on a real device. A standing GPU-validation gate is the
structural fix (see §4).

---

## 3. What work remains — specifically

### Correctness long-tail (bounded, known)
- **GRIS spatial residual ~16% darkness** — stable + comparable to GRIS-OFF; a reconnection-
  visibility-rejection / ping-pong double-filter artifact in the *spatial* pass (temporal-only is
  exact). A refinement, not a defect.
- **V9 Jakob-Hanika out-of-gamut convergence** — consumption confirmed; the negative-green
  spectral-hero estimator doesn't converge at software-GL SPP. Needs a real-GPU high-SPP capture.
- **BDPT §10.3 variance A/B** — BDPT-ON renders correctly; the full vs 2-strategy variance win
  needs a small/hidden-emitter *caustic* scene (large-area-light Cornell under-exercises it).
- **GPU skinning normals** (task #24) — skin compute transforms positions but not normals;
  inverse-transpose needed for non-uniform bone scale.

### Performance (the biggest unvalidated dimension)
- **Nothing has been perf-validated on target hardware.** lavapipe is a CPU rasterizer (correctness
  only); dzn confirmed *correctness*, not tuned frame-time. There are no per-tier frame-time budgets.
- **NRC recompute-in-backward kernel** (task #23, ~46× memory-traffic reduction) — the current fused
  kernel still stages activations to global memory.
- Wavefront PT, register-blocked GEMM, dispatch-cadence tuning — all unprofiled.

### Fidelity promotion (`plan/renderer-fidelity-matrix.md`)
- pt-webgpu spectral / thin-film / SSS / caustics / multi-emitter rows are implemented + mechanically
  tested but tagged `experimental` until gap-closure scenes promote them to `supported`.

### Frontier (tracked, de-prioritized — roadmap §8)
- Differentiable RT (Phase 0 finite-difference proof unstarted); cached light field; MNEE caustics;
  deferred-GI material-swap. (ReGIR + NRC are now done.)

---

## 4. Closing the gap to a *fully professional* pipeline — honest estimate

vitrum's **feature breadth is ahead of its validation + performance depth.** The algorithms are
broad and, after this wave, the *correctness* foundation is genuinely strong. The gap to
"fully professional" is **not more features** — it's depth, in this priority order:

1. **Make GPU validation a standing gate (highest leverage).** This wave found four real bugs that
   unit tests missed; that will keep happening until a per-PR GPU smoke + radiometric A/B on the
   `wsl-gpu` harness runs continuously (shader-compile on Dawn-class + dzn, default-render
   non-regression, opt-in-feature convergence A/B). This converts "we got lucky catching these" into
   "these can't merge." Without it, the bug class recurs. **Estimate: the single most valuable
   investment.**
2. **Real-hardware performance pass.** Profile + optimize on target GPUs; establish frame-time
   budgets per quality tier (the §4 hardware-class matrix is designed but unmeasured). Land the NRC
   recompute-in-backward kernel. Until perf is measured, the real-time stack's "real-time" claim is
   unverified on anything but a 4090.
3. **Promote the fidelity matrix.** Author the gap-closure scenes that move pt-webgpu's
   `experimental` rows to `supported`, each with a perf + reference-image sign-off.
4. **Close the correctness long-tail** (§3) — GRIS spatial residual, V9 high-SPP, BDPT variance
   regime, skinning normals. Each is bounded and scoped.
5. **Harden via a real host integration.** A real stainedGlass (or other) embedding surfaces the
   workflow/robustness/error-boundary gaps a synthetic test scene never will — the tier-4
   "usage questions" the roadmap defers to a host.
6. **Then frontier** (differentiable RT, etc.) — only after 1–5.

**Bottom line by my estimation:** the engine is at a strong release-candidate point with a now-solid
correctness spine. The distance to *fully professional* is dominated by **continuous GPU validation
(1)** and **real-hardware perf (2)** — both infrastructure/depth investments rather than new
rendering math. Items 3–5 are finite and scoped. The frontier (6) is genuinely optional polish on top.
If I had to pick one thing: **stand up the GPU-validation CI gate** — it is what separates "a deep
feature set that mostly works" from "a pipeline you can trust to ship."

---

## 5. Pointers
- Live state: `git log` on `main`, `CLAUDE.md`, `HARDWARE-VALIDATION-NEEDS.md` (V1–V20).
- Validation evidence: `../wsl-gpu/WAVE8…WAVE12-RESULTS.md` + `FINDINGS.md`.
- Roadmap: `plan/roadmap.md` (§0.5 locked decisions), `plan/renderer-fidelity-matrix.md`,
  `plan/differentiable-rt.md`, `plan/tier4-vision-not-yet.md`.
- Open task threads: #23 (NRC recompute-in-backward), #24 (GPU skinning normals).
