# Overnight autonomous run — P0–P8 campaign status (2026-06-03)

Autonomous session against the P0–P8 "address every gap" campaign
(`memory/campaign-p0-p8-2026-06-02.md`). The user asked to push P0+P1 and keep
working through P8 overnight. This doc records exactly what shipped, what is
blocked and why, and the precise resume path.

## GPU validation — initially blocked, now RESOLVED (commit `d997610`)

> **RESOLVED.** The "render worker won't start" was a harness bug, not a missing
> env. The `--working-tree` smoke pointed the import map at the live tree but
> never set `VITRUM_PINNED_DIR`, so the walkaroundUbo headless shim
> (`Deno.readTextFileSync($VITRUM_PINNED_DIR/.../walkaroundUbo.wgsl.ts`)) fell
> back to the empty `~/.cache/wsl-gpu/vitrum-pinned` and crashed "stdout closed
> before ready". Fixed in `scripts/validate-gpu.mjs` (passes
> `VITRUM_PINNED_DIR=<worktree>` for `--smoke`). **T1 smoke now PASSES on both
> backends** — lavapipe 81.15 dB / dzn 89.03 dB vs golden, cross-check 34.87 dB —
> which also GPU-confirms P0–P4 render non-regressing. **The GPU phases
> (P2/P5/P7/P8) are UNBLOCKED for validated work.**

**Original finding (kept for context):** the wsl-gpu render worker would not
start (`worker stdout closed before ready` on lavapipe + dzn); under that
constraint, committing unvalidated GPU code to delicate areas (e.g. P2's
material-stride) was judged unsafe — hence the contract/CPU-first ordering below.

## SHIPPED — committed + pushed to `origin/main`

| Phase | Commit | What | Tests |
|------|--------|------|-------|
| **P0** docs | `d934f4f` | ~45 behavioral-verified fixes to stale comments/docs | typecheck + all touched-pkg vitest green |
| **P1** contract | `98cc933` | structured `TextureRef` + `UvTransform` + alpha-mode + AO/clearcoat/sheen/iridescence/anisotropy map slots; `HdriEnvironment.hdri`→`EnvironmentMapRef`; importer `toTextureRef`/`fromTextureRef` | core 50 / three-bindings 86 |
| **P1** mesh/anim | `cd233f4` | `MeshPrimitive`/`InstancedMeshPrimitive` `uv1`+`colors`; `AnimationClip` data type; `convertMesh` reads them | three-bindings 89 |
| **P3** import | `3f81080` | `loadGltfScene` → `AnimationClip[]` (`convertAnimations`); `LoadedGltf.animations` | three-bindings 92 |
| **P4** tonemap | `a4e7aba` | `FrameQualitySettings` exposure/tonemap/outputColorSpace; shared `applyTonemap`/`vitrumTonemap` operators (ACES/AgX/Reinhard/linear/none) | shared-samplers 281 |
| **P3** sampler | `038414a` | `sampleAnimationClip` (LINEAR+slerp / STEP / CUBICSPLINE) — animation now usable end-to-end on CPU | core 55 |

These are the campaign's **contract + CPU-logic** layers — fully verifiable
without a GPU. They are the foundation the GPU phases consume (P2 consumes the
P1 texture contract; backends consume the P4 operators).

## BLOCKED on the GPU validation env

- **P2 — pt-webgpu textures (the #1 ordered phase).** pt-webgpu has *zero*
  sampled-texture infra (even the HDRI env is a storage *buffer*). Adding it
  requires (a) bumping `MATERIAL_VEC4_STRIDE` — the delicate "misaligns every
  read" constant — or a parallel index buffer, AND (b) bind slots beyond the
  explicitly-tight group budget (likely `maxBindGroups ≥ 5`, gated like NRC).
  Both the stride/packing alignment and the bind-group creation are exactly what
  fails only at GPU runtime. **Needs the GPU env.** (P1 already delivers the host
  contract it will consume + the UV-transform extraction.)
- **P5 — contract honesty.** Capability-honesty is done (P0). **Aux-buffer emission
  SHIPPED 2026-06-03:** `HybridEngine` now surfaces `FrameRendered.{normalDepth,
  albedo, motionVectors}` from the always-allocated G-buffer via
  `WalkaroundGPUPipeline.getAuxBufferTextures()`; `supportsAuxBuffers` flipped to
  true in both the engine capability and `BACKEND_PROMISE_LEDGER`. (Variance is
  the RG32F Welford buffer ≠ the contract's RGBA32F, so not exposed.)
  **Instance-COUNT topology rebuild SHIPPED 2026-06-03:** `updatePrimitive`
  intercepts the wholesale-replacement fields (`instances`/`params`/`shape`/
  `fallbackMesh`/`kind`) and routes them through a full setScene rebuild (no
  longer throws "call setScene"), matching pt-webgl/pt-webgpu's contract surface;
  `BACKEND_PROMISE_LEDGER` doc updated. **Spotlight cone in GI SHIPPED
  2026-06-03:** the DDGI probe pass now confines a spot emitter's GI to its cone
  (was a point-like omnidirectional flood — `coreEmittersToDDGILights` dropped the
  cone). `coreEmittersToDDGILights` packs spotAxis + cos(inner/outer) (from the
  core SpotEmitter angle/penumbra); `packDDGIProbeLights` writes them into the
  reserved DDGILight WGSL slots (direction/innerCone/outerCone); `evalPointLight`
  applies `smoothstep(cosOuter, cosInner, dot(toLight, axis))`. Unit-pinned
  (conversion + packing); no-spot scenes byte-identical (axis=0 → falloff 1).
  GPU cone-confinement A/B pending a spotlight harness scene (V-item). Remaining
  P5: analytic-shape expansion (lower-priority).
- **P7 — shallow-algo depth.** Real MNEE Newton solve, wiring the dead GPU
  differentiable adjoint, NRC debias — all radiometric, GPU-A/B-critical. Shipping
  a default neural U-Net checkpoint needs a PyTorch + training-dataset env.
- **P8 — frontier (4 moonshots).** All GPU-heavy / multi-week. The verified
  research dossiers from the planning session are the build specs.

## LARGE REFACTOR (non-GPU, but not a safe blind slice)

- **P6 — THREE decoupling.** `shared-bvh` (21 `THREE.Mesh`, 18 `BufferAttribute`
  usages) + walkaround-hybrid (24 THREE imports) are load-bearing — they *ingest*
  THREE geometry to build the BVH. A real decouple rewrites the geometry-ingestion
  path onto the `@vitrum/core` `Scene` contract; large, cross-package, and
  validated by typecheck+tests. Best done as a dedicated refactor, not blind on a
  shrinking context.

## Resume path

1. **Get the wsl-gpu render worker running** (`npm run validate:gpu:smoke` →
   `worker stdout closed before ready` must clear; needs `ptWebgpuFullTier: true`
   + `hybridCanRun: true` per `HARDWARE-VALIDATION-NEEDS.md` §0).
2. **P2** — build the pt-webgpu texture subsystem (parallel index buffer to avoid
   the stride landmine; texture-array + sampler in a gated full-tier bind group;
   alpha-test/blend), A/B vs `bbd32c8`-style baseline.
3. **P5 / P7** radiometric items — implement + GPU A/B (new V-items).
4. **P6** — dedicated THREE-decouple refactor (typecheck+test gated).
5. **P8** — pick a frontier headline per the dossiers (recommend NRC-default).

All P0–P4 work above is on `origin/main`, typecheck-clean, tests green.
