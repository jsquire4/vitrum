# WS8 — Robustness wins, greenfield freebies, and the horizons it unlocks

> The second-order payoff. These are why owning the WebGL2 backend is worth more than just "no THREE." Woven into WS1/WS2 at greenfield where marked (cheap now, costly to retrofit).

## 1. Robustness wins (the WebGL2 fallback gets materially better)

The fork's WebGL2 path is "works or it doesn't" — its degradation + context handling are THREE's. Owning it:

- **Graceful tiered degradation** (WS5 §3 `traceTier`). Probe WebGL2 limits directly and step down through vitrum's quality machinery: `rgba16f` accumulation when `rgba32f`-render is unsupported; smaller BVH/material square textures when `MAX_TEXTURE_SIZE` is small; fewer bounces; drop HDRI/MRT on `lite`. Today the fork hard-fails.
- **Context-loss recovery** [greenfield, WS1/WS2]. Register `webglcontextlost`/`webglcontextrestored`; on restore, rebuild `GlResources` (FBOs/textures/programs) + re-upload `#sceneTextures` from the retained `#scene`. Aligns with the host-owns-lifecycle principle (survive canvas remounts / route changes). Cheap to build in, expensive to retrofit.
- **No THREE-version coupling.** The fork is pinned to a THREE version; THREE updates can silently break it. Gone.
- **Honest software-GL tier.** SwiftShader renders correctly — expose it as a flagged capability tier (renders, marked non-fidelity) instead of a black canvas.
- **Bundle size.** `three` (~600 KB min) + `three-mesh-bvh` + the fork removed from the WebGL2 path.

## 2. Greenfield discipline freebies (do them at WS1/WS2 — painful later)

- **Generated FrameParams layout codegen** (WS5 §2) — one source generates the GLSL std140 `FrameParams` struct + the TS packer offsets, so the shader block and the packer can't drift (the pt-webgpu `frameParamsLayout.generated.ts` pattern via `tools/generate-wgsl-layouts.mjs`, extended to emit GLSL).
- **Capability single-source** (WS1 §6) — `capabilities.supported*Kinds` derive from the same `PT_WEBGL2_SUPPORT` object the scene packer filters against; advertised capability can't diverge from ingestion (the verified pt-webgpu invariant).
- **Boundary test from line one** — a `packageBoundary.test.ts` asserting no `from 'three'`, so THREE can never creep back in.
- **The BVH brute-force oracle gate** (WS3 §2 / WS6 §3) wired into the per-push smoke — closes the F-TLAS1/F-RC1 stride-bug class for WebGL2 from the start.

## 3. Newly enabled — the big architectural payoff (real follow-on work, not freebies)

- **One source of truth for the path-tracing math across both backends.** Today the BSDF/sampling math is duplicated — WGSL (`shared-samplers`/pt-webgpu) and GLSL (the fork). That duplication is exactly what bred the audit's cross-backend divergences (fork scalar-throughput crush, glossy-BSDF drift). Once we author the WebGL2 kernels, both targets can be driven from one shared source via a small macro/transpile layer (WGSL and GLSL differ mostly in type names + texture/buffer access — a templating pass can target both). Result: cross-backend parity *by construction*. This is the highest-value follow-on; scope it after the cutover.
- **Bring pt-webgpu's validated physics to WebGL2.** S3 already swaps the fork's phenomenological `pow(dot,10)` caustics for pt-webgpu's validated manifold-NEE (the audit's P1.7). The same applies to spectral/MNEE: pt-webgl stops being "experimental phenomenological" and becomes a real peer with shared, validated math.
- **Consistent introspection** [freebie if wired in WS5]. `engine.debug` (the backend-agnostic `pickPrimitive`/`estimatedGpuMemoryBytes`/`isDenoiserEnabled`) + `dev` overlays wire into WebGL2 the same way as pt-webgpu.
- **First-class `Material.extensions` / `AnalyticShape`** — honored in a vitrum-authored kernel instead of shoehorned through THREE's material system (design principle #3).

## 4. Long-horizon unlocks (speculative + large — flagged, not scoped)

- **WebGL2 realtime GI.** Owning the WebGL2 GL framework + BVH textures + `shared-samplers` is the foundation a WebGL2 realtime-GI path needs (DDGI/ReSTIR via fragment shaders + MRT, since WebGL2 has no compute). `walkaround-hybrid` is WebGPU-only today, so WebGL2 devices get zero realtime GI. This rewrite is the prerequisite that makes it *possible* — and extends the progressive `walkaround→PT` handoff (`createProgressiveEngine`) to WebGL2.
- **A unified `shared-gl` framework** that both a future WebGL2 realtime engine and `pt-webgl2` share (the WebGPU side already shares `shared-bvh`/`shared-samplers`/`shared-denoisers`) — the symmetric WebGL2 story.

## 5. What NOT to do (scope discipline)

- Don't build the shared WGSL↔GLSL source layer *during* the cutover — it's a follow-on; the cutover copies the GLSL as-is.
- Don't port the Looking-Glass quilt renderer.
- Don't implement `createInverseSession` (pt-webgpu-only).
- Don't block the cutover on the `three-bindings` glTF-loader replacement (separate, optional).
