# WS7 — Phasing, cutover, and the THREE deletion

> The build order (vertical slices, each a shippable A/B gate) and the final cutover that deletes THREE from the runtime graph.

## 1. Slice sequencing (each gate = typecheck + unit suite + the slice's A/B)

### S0 — diffuse spine (the feasibility-validating spike)
Build: WS1 (package + contract skeleton + factory + capabilities) → WS2 (GlResources + GlProgram + FBO + fullscreen quad + Regime 3) → WS3 (`packSceneFromCore` + BVH texture adapter + a minimal positions/normals upload) → WS4 (a Lambertian-only `composeTraceGlsl`: `trace_scene` + `get_surface_record` + `bsdf_functions` diffuse branch + the BVH GLSL) → WS5 (renderFrame accumulation loop + frameParams UBO).
**Gate:** a core Cornell renders + converges to a `WebGLTexture`; silhouette + diffuse GI A/B vs a fork diffuse capture (SwiftShader OK for geometry); BVH adapter passes the brute-force oracle at 100%. *This slice de-risks the entire architecture and validates the 6–8 wk estimate against reality.*

### S1 — material parity
Add: WS3 `materialsTexture.ts` (85px) + the attribute array (tangent derivation) + the material texture atlas; WS4 full BSDF (GGX/metal/transmission/clearcoat/sheen/iridescence/thin-film), the `material_struct` decoder (kept).
**Gate:** A/B vs fork on a multi-material scene (PSNR ≥ 28 on the relevant `rfe` baselines).

### S2 — lights + IBL
Add: WS3 `lightsTexture.ts` (6px) + `equirectHdrInfo.ts` (CDF); WS4 light/equirect/shape sampling, `FEATURE_MIS`.
**Gate:** A/B vs fork on the emitter + HDRI scenes.

### S3 — fork extensions (each its own A/B)
Add per feature (WS5 §5): spectral hero-λ → MNEE caustics (swap in pt-webgpu's validated math) → BDPT (the light-subpath scratch-RT pass) → Jakob-Hanika → additive accumulation (Regime 1).
**Gate:** all six `pt-webgl-fidelity/*.baseline.png` at PSNR ≥ 28 on the real-GPU capture host.

### S4 — cutover (§2–§4 below).

## 2. The facade switch

`@vitrum/engine`'s backend selection (`createEngine` / `createProgressiveEngine` / `threeSceneBridge`) currently routes WebGL2 requests to `@vitrum/pt-webgl`. Add `pt-webgl2` as the WebGL2 backend:
1. Add the `'pt-webgl2'` `BackendId` + ledger row to `@vitrum/core` (master §5).
2. Point the facade's WebGL2 branch at `createPTEngine_WebGL2`.
3. Keep `pt-webgl` importable behind a deprecated flag for one release if any host pins it (optional); otherwise delete in §3.
4. Update `plan/renderer-fidelity-matrix.md`: replace the pt-webgl `experimental` rows with `pt-webgl2` rows promoted to `supported` (gated on the WS6 real-GPU A/B).

## 3. The deletions (what THREE removal actually removes)

On S4 green:
- **Delete `packages/three-gpu-pathtracer/`** (the fork — 13,237 LOC). Its GLSL kernels now live in `pt-webgl2/src/glsl/` (copied); its packers are ported; its render framework is replaced.
- **Delete `packages/pt-webgl/`** (the wrapper — its THREE-free logic that's still wanted, `adaptiveScheduler`/`sceneTlasAudit`/`bdptSceneEmittersCpu`/the readback math, moves into `pt-webgl2` during S0–S3; see WS port notes).
- **Drop the `three-mesh-bvh` dependency** (replaced by shared-bvh + the ported `BVHShaderGLSL`).
- **Drop the `three` dependency** from the WebGL2 runtime graph (~600 KB min bundle cut).
- Remove the `@vitrum-pathtracer` / `three-gpu-pathtracer` vitest aliases + the `legacy/three/` quarantine in pt-webgl (gone with the package).

## 4. `three-bindings` slim + the remaining optional THREE

`three-bindings` (the THREE→core ingestion adapter) is still optional (dynamically loaded by `engine/threeSceneBridge.ts` only when a host passes a raw `THREE.Scene`). After S4, decide (host-API call):
- **Keep it** as the optional THREE-host on-ramp (THREE stays a *peer dep* of `three-bindings` only, not the engine/backends), OR
- **Replace it with a glTF→core loader** (`@vitrum/gltf-loader`) so THREE is gone entirely and THREE-host on-ramp is via glTF export. This is a separate, smaller project — not required for "THREE removed from vitrum's own packages."
The walkaround `./three` bridges (`applyDDGIShading`, `nodeMaterialUpgrade`, host scene types) + the `legacy/three/` test oracles are unaffected by this workstream (they're walkaround-hybrid's, already quarantined). They can be trimmed in a follow-up once no host consumes them.

## 5. Low-hanging-fruit checklist (grab during S0–S4)

- [ ] Generated FrameParams layout codegen (WS5 §2) — packer↔GLSL can't drift.
- [ ] Capability `supported*Kinds` derived from `PT_WEBGL2_SUPPORT` (WS1 §6) — capability can't diverge from ingestion.
- [ ] `EXT_float_blend`-absent → Regime 2 fallback (robustness).
- [ ] `rgba16f` accumulation tier when `rgba32f` not renderable (lite tier).
- [ ] Context-loss recovery (`webglcontextlost`/`restored` handlers → rebuild GlResources, host-owns-lifecycle) — cheap at greenfield.
- [ ] Honest software-GL tier exposed via capabilities (renders, flagged non-fidelity).
- [ ] Skip the Looking-Glass quilt renderer (don't port `QuiltPathTracingRenderer.js`).
- [ ] Reuse `shared-bvh`/`shared-samplers`/`shared-denoisers` from line one (no fork copies).
- [ ] Re-capture + promote the 8 fork fidelity rows out of `experimental` (free with the WS6 capture host).

## 6. WS7 done-when
- The facade routes WebGL2 to `pt-webgl2`; `createEngine` with a `WebGL2RenderingContext` returns a working `pt-webgl2` engine.
- `three` is absent from the WebGL2 runtime dependency graph (`npm ls three` shows it only under `three-bindings`, if kept).
- Full `npm test` green; the real-GPU fidelity gate green; the fidelity matrix updated.
- `three-gpu-pathtracer`, `pt-webgl`, `three-mesh-bvh` removed (or `pt-webgl` deprecated-for-one-release per §2.3).
