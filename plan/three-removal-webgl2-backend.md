# Path B — THREE-free native WebGL2 path tracer (scoping)

> Status: **SCOPING** (no implementation). Produced 2026-06-08 to let the team decide
> Path A (drop WebGL2) vs Path B (own the WebGL2 backend) with grounded numbers.
> Every structural claim below was verified by code-read, not commit messages.

## Goal

Remove THREE entirely. The host-agnostic engine core is **already** THREE-free
(verified: `core`, `engine`, `pt-webgpu`, all shared-* packages, and the
`walkaround-hybrid`/`walkaround-rc` live engine paths — THREE is quarantined there
to opt-in `./three` host-bridges + test-only `legacy/three/` oracles, enforced by
`packageRootBoundary.test.ts`). THREE survives in exactly one load-bearing place:

- **`pt-webgl`** (the WebGL2 path-tracing backend), which wraps
- **`three-gpu-pathtracer`** (the absorbed fork), which depends on
- **`three-mesh-bvh`** (BVH build + GLSL traversal) and **`three`** itself.

Path B replaces all three with a vitrum-owned WebGL2 backend (`@vitrum/pt-webgl2`)
that **keeps the path-tracing algorithms** and removes the THREE render framework.

## The existence proof

`pt-webgpu` is already a complete THREE-free converged path tracer (spectral, BDPT,
MNEE, ReSTIR-PT, light-tree NEE; 0 THREE imports). It is the architectural template
for the WebGL2 backend — same `@vitrum/core` `Engine` contract, same
`presentationMode: 'offscreen-texture'`, same "host owns the device" model
(`device: WebGL2RenderingContext` instead of `GPUDevice`).

## Verified keep / rewrite / port inventory (fork = 13,237 LOC)

| Category | LOC | Disposition | Why |
|---|---|---|---|
| **C — GLSL kernels** (33 `*.glsl.js`) | **4,663** | **KEEP, port as-is** | Verified THREE-free (0 imports/`from 'three'`/`#include`). They read only uniforms/`#define`s named by the host material. The path-tracing math (BSDF 1,472 · main loop 1,432 · structs 448 · sampling 435 · common 450 · RNG 363) ports verbatim. |
| **B — driver + CPU packers** (~30 files) | **5,777** | **PORT** (re-target carrier) | THREE coupling is shallow: `DataTexture`/`DataArrayTexture` used only as a typed-array → GPU-texture carrier, plus `Vector*`/`Matrix4` math. Re-point the carrier at a vitrum GL texture; the byte-packing math (`MaterialsTexture` 603, `EquirectHdrInfoUniform` 312, `LightsInfoUniformStruct` 238, the geometry mergers) is unchanged. **Pivotal glue:** `PhysicalPathTracingMaterial.js` (1,109) owns the uniform/`#define` contract and string-assembles the kernel — it must be reworked to feed a vitrum program builder, but its *schema is the contract the kernels depend on* and must be preserved. |
| **A — THREE render framework** | **2,764** | **REWRITE** (raw WebGL2) | The `WebGLRenderer`-bound layer: `PathTracingRenderer.js` (584 — the ping-pong accumulation loop, additive-blend RTs, tiling/scissor), `MaterialBase`/`ShaderMaterial` `#define`-recompile + program-cache, all `FullScreenQuad` blit materials (Blend/Display/Denoise/…), `RenderTarget2DArray`, the env/sobol generators, the precision/compat detectors. **This is the only genuinely new infrastructure.** |

Note the fork's `PathTracingRenderer` is *already* a fragment-shader accumulation
design (it is WebGL2) — Path B does not invent the approach, it re-expresses the
same FBO-ping-pong + fullscreen-quad + additive-blend operations in raw WebGL2
(framebuffers, programs, uniform/sampler binding) without `WebGLRenderer`.

## BVH: shared-bvh already matches three-mesh-bvh (the big de-risk)

The fork's ray-vs-BVH traversal GLSL is `three-mesh-bvh`'s `BVHShaderGLSL`
(`common_functions` + `bvh_struct_definitions` + `bvh_ray_functions`, ~545 LOC GLSL),
injected as strings (`PhysicalPathTracingMaterial.js:248-250`) — render-framework-free.
The CPU side packs the BVH into **4 WebGL2 data textures** via `MeshBVHUniformStruct`:
`bvhBounds` (RGBA32F, 2 texels/node), `bvhContents` (RG32UI, 1/node), `position`
(RGBA32F square), `index` (RGBA32UI square).

**Verified:** `@vitrum/shared-bvh`'s 32-byte node layout is byte-identical to
three-mesh-bvh's (`buildArrayBvh.ts:13-27` + `bvhIntersect.wgsl.ts:180-181` —
"exactly matches three-mesh-bvh's raw node layout, `BYTES_PER_NODE = 32`"). Same
relative-right-child + implicit `nodeIndex+1` left + `0xFFFF0000 | count` leaf word
(the form `bvhToTextures` already normalizes to). shared-bvh is *direct* (physically
reorders indices) so it skips three-mesh-bvh's indirection step.

→ Replacing three-mesh-bvh is **(a)** porting the ~545 LOC of `BVHShaderGLSL` into
vitrum as string modules (MIT, render-framework-free) + **(b)** a ~200 LOC
texture-packing adapter (re-stride shared-bvh's 3 flat arrays into the 4 data
textures — the inverse of `bvhToTextures`). **No node-format translation.** This
fully removes three-mesh-bvh and its transitive THREE.

## Architecture (mirror pt-webgpu)

- `class PTEngineWebGL2 implements Engine` (non-exported) + async
  `createPTEngine_WebGL2: EngineFactory` validating `device: WebGL2RenderingContext`.
- `GlResources` analogue of pt-webgpu's `GpuResources`: owns FBOs / textures / UBOs /
  programs (instead of bind groups + storage buffers). **The defining divergence:**
  WebGL2 has no compute/SSBOs, so geometry+BVH live in *float textures* and
  accumulation is *FBO ping-pong* (which the fork already does) rather than an
  RW storage buffer.
- `composeTraceGlsl(features)` mirroring `composePtWebgpuTraceWgsl` + a generated
  uniform-layout file (the `frameParamsLayout.generated.ts` discipline) so the
  packer and the shader struct can't drift.
- `SceneMutationRouter`-style state seam for the incremental patch fast paths
  (transform/material/emitter — pt-webgl already has these in `scenePatch.ts`).
- `renderFrame` accumulation loop; `presentationMode: 'offscreen-texture'` returning
  a `WebGLTexture` as `FrameOutput.primaryRadiance` (the `BackendTexture` brand is
  already backend-parameterized for exactly this).
- Reuse as-is: `@vitrum/shared-bvh` `packSceneFromCore`, `@vitrum/shared-samplers`
  (Hammersley/octahedral/luminance/hero-wavelength), `@vitrum/shared-denoisers`
  `OIDNDispatcherCore`, and pt-webgl's THREE-free logic (`adaptiveScheduler`,
  `sceneTlasAudit`, `bdptSceneEmittersCpu`, the `readbackHdr` math half).

## Work streams + effort bands

Honest engineering-week bands (single experienced dev; the algorithms are kept, so
this is infrastructure + porting + validation, not research):

1. **WebGL2 render framework** (rewrite Cat A, ~2.7k LOC) — FBO/program/uniform
   abstraction, ping-pong accumulation, blit passes, the `#define`-recompile program
   builder. **~2–3 wk.** Well-trodden WebGL2; the fork is the exact spec.
2. **Driver + CPU packers** (port Cat B, ~5.8k LOC) — re-target `DataTexture`→GL
   texture; rework `PhysicalPathTracingMaterial` glue to feed the program builder.
   **~2–3 wk.**
3. **BVH** (port `BVHShaderGLSL` + shared-bvh texture adapter, ~750 LOC) — **~1 wk.**
4. **GLSL kernels** (port Cat C as-is + adapt binding conventions, ~4.7k LOC) —
   **~1 wk** to wire, but GPU output debugging is slow.
5. **Validation infra** — see below; the real wildcard.

**Implementation total: ~6–8 engineering-weeks**, then a feature-parity validation
tail (each fork feature — BDPT, MNEE, spectral, caustics, IBL — needs its own A/B).

**De-risked by:** algorithms kept (Cat C), BVH already format-compatible, pt-webgpu
as template, the fork as the exact behavioral reference.

## Validation strategy — and the one real blocker

To prove "no fidelity regression," A/B the native backend against the **current fork
output as oracle**: the `tools/reference-renders/pt-webgl-fidelity/*.baseline.png`
set already IS current-fork output across six scenarios (layered, caustic, SSS,
spectral, CMF bridge, thin-film), and `run-pt-webgl-fidelity-acceptance.mjs`
(`minPsnr=28`) + `diff-baselines.mjs` is a reusable gate — only the producer changes.

**The blocker (honest):** pt-webgl has **no automated hardware-GPU gate today.** The
pre-push WGSL T1 smoke is WebGPU-only (lavapipe/dzn) and cannot run WebGL2. The
WSL WebGL2 capture path (`wsl-gpu/webgl2-capture/`) runs only on **SwiftShader
(software GL)** — valid for geometry/transform/consume-confirmation, but the
fidelity matrix explicitly needs a real GPU for variance/out-of-gamut/perf. This is
the **same "GL capture path" gap that keeps all 8 fork fidelity rows `experimental`.**
Path B therefore also requires standing up a real-GPU WebGL2 capture host (Windows
Chrome or a self-hosted GPU runner) — days if a host is available, a hard dependency
if not. **This validation gap exists regardless of Path B** (it already blocks fork
promotion); Path B just makes closing it mandatory.

## Phasing (thin vertical slice first)

1. **Slice 0 — diffuse Cornell end-to-end.** `createPTEngine_WebGL2` + `GlResources`
   + the BVH texture adapter + a minimal Lambertian kernel rendering one core Scene
   to a `WebGLTexture`, A/B'd vs a fork diffuse capture. Proves the whole spine
   (contract, render loop, BVH, packers) before any feature porting.
2. **Slice 1 — material parity** (GGX/metal/transmission, MaterialsTexture packer).
3. **Slice 2 — IBL/env + emitters + light sampling.**
4. **Slice 3 — the fork's vitrum extensions** (spectral hero-λ, MNEE, BDPT,
   Jakob-Hanika, additive accumulation) — each with its fidelity A/B.
5. **Cutover** — `engine` facade selects `pt-webgl2`; delete `pt-webgl`, the fork,
   three-mesh-bvh, three. Trim the optional `three-bindings` adapter + walkaround
   `./three` bridges (host-API decision: require core Scenes / ship a glTF→core
   loader instead of THREE→core).

## A-vs-B recommendation framing (the decision)

- **Path B is feasible and bounded** — ~6–8 wk + a validation-infra build, NOT
  "recreate THREE." Most of THREE is irrelevant; the fork uses a thin slice, the
  algorithms are kept, and the BVH already matches.
- **The pivotal question is whether the WebGL2 fallback earns it.** `pt-webgpu`
  already delivers THREE-free converged PT, and WebGPU support is now broad. If
  WebGL2 coverage (older Safari / no-WebGPU devices) matters, Path B is the way and
  this plan is the route. If it doesn't, **Path A** (delete pt-webgl + fork +
  three-mesh-bvh, rely on pt-webgpu) reaches zero-THREE in days.
- Either way, the optional `three-bindings` ingestion adapter + walkaround `./three`
  host-bridges are a small, separate cleanup that follows the main decision.

## Second-order benefits, robustness wins, and ride-along low-hanging fruit

### WebGL2 fallback robustness (yes — materially)

The fork's WebGL2 path is "works or it doesn't" — its degradation + context handling
are THREE's, not ours. Owning it makes the fallback genuinely robust:

- **Graceful tiered degradation.** Probe WebGL2 limits directly (float-render
  support, `MAX_TEXTURE_SIZE`, uniform vectors) and degrade through vitrum's existing
  quality-preset machinery — `rgba16f` accumulation when `rgba32f`-render is
  unsupported, smaller BVH textures, fewer bounces, lower res — instead of a hard fail.
- **Context-loss recovery** aligned with the host-owns-lifecycle principle (survive
  canvas remounts / route changes). Today the fork owns this; owning the backend lets
  us make it a first-class vitrum behavior.
- **No THREE version coupling.** The fork is pinned to a THREE version; THREE updates
  can silently break it. Gone.
- **Honest software-GL tier.** SwiftShader (software WebGL2) renders correctly; expose
  it as an explicit capability tier (flagged non-fidelity) instead of a black canvas.
- **Bundle size.** Dropping `three` (~600 KB min) + `three-mesh-bvh` + the fork is a
  real footprint cut for WebGL2 hosts.

### Newly enabled (grounded; freebie vs. real-work flagged)

- **Single source of truth for path-tracing math across BOTH backends** *(real
  sub-project, the big payoff).* Today the BSDF/sampling math is duplicated — WGSL
  (shared-samplers/pt-webgpu) and GLSL (fork). That duplication is exactly what bred
  the audit's cross-backend divergences (fork scalar-throughput crush, glossy-BSDF
  drift). Once we author the WebGL2 kernels, both targets can be driven from one
  shared source (macro/transpile layer) → cross-backend parity by construction.
- **Bring pt-webgpu's validated physics to WebGL2** *(newly possible, per-feature
  work).* pt-webgpu's manifold-NEE caustics (validated, physically-based) replace the
  fork's phenomenological `pow(dot,10)`; same for spectral/MNEE. pt-webgl stops being
  "experimental phenomenological" and becomes a real peer.
- **Consistent introspection** *(freebie).* `engine.debug` (pickPrimitive,
  getDebugTextures, estimatedGpuMemoryBytes) + `dev` overlays wire into WebGL2 the
  same way as pt-webgpu.
- **First-class `Material.extensions` / `AnalyticShape`** *(freebie).* Honored in a
  vitrum-authored kernel instead of shoehorned through THREE's material system.

### Long-horizon unlock (honest: speculative + large)

Owning the WebGL2 GL framework + BVH textures + shared-samplers is the foundation a
**WebGL2 realtime-GI path** would need (DDGI/ReSTIR via fragment shaders + MRT, since
WebGL2 has no compute). walkaround-hybrid is WebGPU-only today, so WebGL2 devices get
no realtime GI — this rewrite is the prerequisite that opens that door (and extends
the progressive walkaround→PT handoff to WebGL2). Not a deliverable of Path B, but
Path B is what makes it possible.

### Low-hanging fruit to grab while ripping out the old

1. **Stand up the real-GPU WebGL2 capture host FIRST — two-for-one.** Path B needs it
   anyway; the same host promotes the 8 existing fork fidelity rows out of
   `experimental` (the long-standing blocker). Pays double.
2. **Greenfield discipline freebies:** generated uniform-layout codegen (the
   `frameParamsLayout.generated` pattern — no packer/shader drift) and capability
   `supported*Kinds` derived from the same support-set the scene packer filters
   against (the `PT_WEBGPU_SUPPORT` pattern). Cheap now, painful to retrofit.
3. **Half-float accumulation tier + context-loss recovery from day one** — both
   expensive to retrofit, near-free at greenfield.
4. **Reuse shared-\* from line one** — shared-bvh / shared-samplers / shared-denoisers
   instead of the fork's copies → shrinks the port surface and auto-dedupes.
5. **Drop what we don't need:** the Looking-Glass quilt renderer (229 LOC); on
   cutover, delete three-mesh-bvh + the `./three` bridges + slim `three-bindings` to a
   glTF→core loader.
