# Overnight autonomous run — P0–P8 campaign status (2026-06-03)

Autonomous session against the P0–P8 "address every gap" campaign
(`memory/campaign-p0-p8-2026-06-02.md`). The user asked to push P0+P1 and keep
working through P8 overnight. This doc records exactly what shipped, what is
blocked and why, and the precise resume path.

## Hard constraint discovered this session

**The GPU validation env (wsl-gpu render worker) will not start in this shell.**
The pre-push T1 GPU smoke and `npm run validate:gpu:smoke` both fail identically
on **lavapipe and dzn** with `worker stdout closed before ready` — i.e. the
render worker dies *before any shader compiles* (an environment failure, not a
code regression; comments+types can't cause a pre-ready worker crash, and all
typecheck+vitest pass). So **no radiometric / WGSL change could be GPU-validated
this session.** Per the repo's discipline, radiometric/WGSL work must be A/B'd on
a real GPU before it's trusted; the 2026-06-02 lesson is that byte-identity
goldens do NOT catch naga regressions — only the GPU smoke does. Committing large
unvalidated GPU code to delicate areas (e.g. P2's material-stride, "changing it
silently misaligns every material read") was therefore judged unsafe.

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
- **P5 — contract honesty.** Capability-honesty is done (P0). The remaining items
  are radiometric/pipeline: walkaround instance-count topology rebuild, spotlight
  cone in GI, analytic-shape expansion, aux-buffer (variance/motion) emission.
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
