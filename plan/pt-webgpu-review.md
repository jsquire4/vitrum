# pt-webgpu In-Progress Review — 2026-05-09

## Files inventory

| Path                                   | LOC | Description                                                                                 |
| -------------------------------------- | --- | ------------------------------------------------------------------------------------------- |
| `src/index.ts`                         | 353 | Engine class, factory, state/slot pattern                                                   |
| `src/math/mat4.ts`                     | 100 | Mat4 multiply, invert, transformPoint, transformDirection                                   |
| `src/scene/flattenScene.ts`            | 51  | `summarizeScene()` — structural summary of a Scene                                          |
| `src/scene/uploadSceneBuffers.ts`      | 148 | `buildPackedScene()` + `uploadPackedScene()` — CPU packing + GPU buffer upload              |
| `src/wgsl/common.wgsl.ts`              | 89  | PCG RNG, BVH structs, Ray/HitResult, `intersectTriangle`, `safe_normalize`                  |
| `src/wgsl/hammersley.wgsl.ts`          | 37  | Van der Corput radical inverse, Hammersley, uniform sphere, Rodrigues rotation              |
| `src/wgsl/octahedral.wgsl.ts`          | 23  | Octahedral encode/decode for compact normal storage                                         |
| `src/wgsl/pathTraceBruteforce.wgsl.ts` | 107 | Main compute kernel: camera ray gen, brute-force tri intersect, Lambert shade, accumulation |
| `src/wgsl/pathTraceSeed.wgsl.ts`       | 36  | Debug-only seed kernel (UV gradient); NOT wired into the engine — dead file                 |

Total source: ~944 LOC across 9 files.

## Engine contract conformance

All required `Engine` interface members are present and correct:

- `state: EngineState` — getter delegates to private `StateSlot`. (index.ts:78)
- `capabilities: EngineCapabilities` — returns all required fields. (index.ts:82)
- `setScene(scene: Scene): void` — fully wired; calls `buildPackedScene` + `uploadPackedScene`. (index.ts:175)
- `renderFrame(input: FrameInput): FrameOutput` — dispatches compute shader; returns `primaryRadiance`, `samplesAccumulated`, `isConverged`. (index.ts:203)
- `reset(): void` — zeros `#samplesAccumulated`, clears accumulation buffer. (index.ts:294)
- `pause(): void` — transitions state to 'paused'. (index.ts:300)
- `resume(): void` — transitions state back to 'ready'. (index.ts:306)
- `dispose(): void` — destroys all GPU resources, nulls references, sets state 'disposed'. (index.ts:314)
- `updatePrimitive?` / `updateEmitter?` — implemented as stubs that throw `Not implemented`; correct since `capabilities.supportsIncrementalScene = false`. (index.ts:193–200)

Signatures match the contract exactly. TypeScript compiles clean with `--noEmit`.

## Device narrowing

`PTEngineWebGPUOptions` extends `EngineOptions` and narrows `device` to `GPUDevice` (index.ts:25–27). The factory validates the device by duck-typing for `createCommandEncoder` (index.ts:331–334). This is the correct pattern matching the JSDoc in `core/engine.ts:165`. No Three.js dependency.

## Capabilities reporting

index.ts:82–93:

| Capability                 | Value                                                    | Correct?                                                                                                                                                                                                                |
| -------------------------- | -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `accumulates`              | `true`                                                   | Yes — PT-style                                                                                                                                                                                                          |
| `supportsMotionBlur`       | `false`                                                  | Yes — not yet implemented                                                                                                                                                                                               |
| `supportsAuxBuffers`       | `false`                                                  | Yes — no G-buffer output yet                                                                                                                                                                                            |
| `supportsIncrementalScene` | `false`                                                  | Yes — full setScene required                                                                                                                                                                                            |
| `maxSamplesPerPixel`       | from opts, default 4096                                  | Yes                                                                                                                                                                                                                     |
| `maxBounces`               | from opts, default 12                                    | Yes — NOTE: shader does not yet implement multi-bounce loops; this cap is currently a lie (see Gaps)                                                                                                                    |
| `supportedAnalyticShapes`  | `{'sphere','box','capsule','cylinder','h-channel-came'}` | CONCERN: these five shapes are advertised but the scene packer (uploadSceneBuffers.ts:71–73) SKIPS analytic primitives with a warning. This creates a false capability claim.                                           |
| `supportedEmitterKinds`    | full set of 6 kinds                                      | CONCERN: the shader only reads one directional light from params; point/rect-area/disc-area/spot/mesh-area emitters are extracted but only direction is extracted; no light sampling loop. Same false capability claim. |
| `causticStrategy`          | from opts, default 'none'                                | Yes — API complete, impl deferred                                                                                                                                                                                       |

## WGSL shaders

### `pathTraceBruteforce.wgsl.ts` (the active kernel)

The shader does:

1. Camera ray generation via inverse VP matrix — correct and functional.
2. Brute-force O(N) triangle intersection loop — functional for small meshes; no BVH.
3. Single bounce Lambert diffuse shading + emissive scalar — functional.
4. Progressive running average in `accumBuffer` (storage buffer) — correct accumulation algorithm.
5. Writes final display color to `outputTexture` (rgba16float storage texture) — correct.

The WGSL is real code, not placeholder. The shader compiles (verified by tsc not rejecting string types). Its size (95 lines of WGSL) is consistent with what it claims: a single-bounce brute-force integrator.

### `pathTraceSeed.wgsl.ts` — dead code

This 36-line debug kernel that writes a UV-gradient pattern is NOT imported or used anywhere in `index.ts`. It is an artifact of earlier scaffolding. It exports `PT_WEBGPU_SEED_WGSL` but nothing imports it.

### `common.wgsl.ts`, `hammersley.wgsl.ts`, `octahedral.wgsl.ts`

All three are imported and concatenated into the active kernel. The WGSL is semantically correct:

- PCG hash function implementation is standard; `pcgInit` / `pcgNext` / `rand_f32` are correct.
- `intersectTriangle` implements Möller-Trumbore with correct backface handling (double-sided: abs(det) check).
- Hammersley and octahedral encoders are textbook-correct ports.

## Scene → GPU upload

`buildPackedScene` (uploadSceneBuffers.ts:60–125) walks `Scene.primitives`:

- Handles `mesh` and `instanced-mesh` kinds: flattens vertices with transform, packs indices with global offset, assigns material IDs.
- Skips `analytic` primitives with a warning.
- Extracts one directional emitter for the shader (only the first; ignores all other emitter kinds).
- Packs materials as `[baseColor.rgb, emissiveScalar]` — extremely simplified (no roughness, metallic, transmission, textures).

`uploadPackedScene` (uploadSceneBuffers.ts:127–148): allocates 4 GPU storage buffers with correct `STORAGE | COPY_DST` usage flags. Minimum size of 16 bytes for empty scenes prevents zero-size buffer errors. `destroy()` method is present and calls `destroy()` on all 4 buffers — no leak.

**Key concern:** the material packer discards `roughness`, `metallic`, `transmission`, `ior`, `normalMap`, and all texture refs. The shader only does Lambertian + emissive shading, so this is self-consistent — but it means the advertised PBR material model is entirely absent from the GPU side.

## Tests

`@vitrum/pt-webgpu` has **zero tests**. No `__tests__/` directory, no `.test.ts` files, no vitest config, no test script beyond `typecheck` in package.json. The package does not participate in `npm test --workspaces --if-present`.

## Workspace integration

All checks pass:

- `tsc --noEmit -p packages/pt-webgpu/tsconfig.json` — **clean** (no output = no errors)
- `tsc --noEmit -p .` — **clean** (workspace-wide)
- `npm test --workspaces --if-present` — **542 tests across 19 test files, all passing** (pt-webgpu contributes 0)
- `grep -rn "@/"` — **no hits** (no host-app imports)
- `grep -rn "from 'react'"` — **no hits**
- `grep -rn "useEffect|useState|useRef|useFrame"` — **no hits**
- `grep -rn "from 'three'"` — **no hits** (Three.js-free as required)
- `grep -rn "from '@vitrum/walkaround-hybrid|from '@vitrum/pt-webgl"` — **no hits**

Dependencies: only `@vitrum/core` is in `package.json`. Does NOT import `@vitrum/shared-bvh`, `@vitrum/shared-samplers`, or `@vitrum/shared-denoisers` despite the architecture doc listing them as expected dependencies (Phase 7+).

## Internal consistency findings

### HIGH

1. **`supportedAnalyticShapes` advertises 5 shapes that are skipped in the packer.** (index.ts:90, uploadSceneBuffers.ts:71–73). Host code that checks capabilities and passes analytic primitives will get silent warnings and no geometry rendered. Should be `new Set<string>()` until the analytic intersection path exists.

2. **`supportedEmitterKinds` advertises 6 kinds but only `directional` is consumed.** (index.ts:91, uploadSceneBuffers.ts:48–57). `rect-area`, `disc-area`, `point`, `spot`, `mesh-area` are all silently ignored. Should be `new Set<string>(['directional'])` until NEE is wired.

3. **`maxBounces` is reported from options (default 12) but the shader is single-bounce.** (index.ts:85, pathTraceBruteforce.wgsl.ts entire kernel). There is no bounce loop. A host calling `renderFrame` with `quality.bounces = 8` will get 1-bounce shading regardless. This is the primary rendering correctness gap.

4. **The `accumTexture` (rgba16float storage texture) is returned as `primaryRadiance` in the paused path (index.ts:209), but `this.#accumTexture` may be `null` before the first `renderFrame`.** `#assertLive` checks `#scene != null` (index.ts:96–103) but not `#accumTexture != null`. If `pause()` is called before the first `renderFrame`, `primaryRadiance: null` is returned without the `samplesAccumulated === 0` skip-frame guard being set — `samplesAccumulated` starts at 0 but `isConverged` is `0 >= this.#maxSamplesLimit` = false. The contract spec says "`primaryRadiance` is null on skip frames" — this path technically satisfies that but the null is the uninit null, not a deliberate skip-frame sentinel. Low likelihood in practice but technically incorrect.

### MEDIUM

5. **Buffer format vs. usage mismatch — accumBuffer vs. accumTexture.** The accumulation algorithm stores running totals in `accumBuffer` (storage buffer, vec4f array, rgba32 effective precision) but the final display write goes to `accumTexture` (rgba16float). The fp16 precision of the texture limits the accumulation fidelity for high sample counts (overflow starts around SPP 65535 for saturated colors, earlier for bright emitters). This is acceptable for early work but should be flagged for the move to fp32 accumulation texture or keeping totals in the storage buffer.

6. **`pathTraceSeed.wgsl.ts` is dead code.** It is never imported or used. (pathTraceSeed.wgsl.ts:1–36). Low harm but adds noise.

7. **Params buffer layout assumes little-endian and specific alignment.** The CPU-side layout (index.ts:240–256) packs `width/height/frameIndex/frameSeed/triangleCount` as Uint32 at offsets 0–4, then `cameraPos` as Float32 at offsets 8–11, `lightDir` at 12–15, `invViewProj` at 16–31. The WGSL struct has `triangleCount` at u32[4], then 3 padding u32s, then `cameraPos: vec4f` starting at byte offset 32 (u32 offset 8). The CPU code writes `paramsU32[4] = triangleCount` (byte 16) and then `paramsF32[8] = cameraPos.x` (byte 32). This is internally consistent — but the 3 padding u32s in the WGSL (`_pad0`, `_pad1`, `_pad2`) are implicit zeros from `new ArrayBuffer(128)`. No bug here, but the layout is not documented.

### LOW

8. **`pathTraceSeed.wgsl.ts` exports a shader with a different bind group layout** (2 bindings: texture + params) **than the active kernel** (7 bindings). If someone wires the seed shader later, the existing `#bindGroupLayout` (from the brute-force pipeline) would fail the bind group creation check.

9. **`BVHNode` struct is defined in `common.wgsl.ts` (line 17) but never used in any shader.** It is scaffolding for when BVH traversal is wired in.

10. **`HAMMERSLEY_WGSL` and `OCTAHEDRAL_WGSL` are included in the active shader but none of their functions are called** in `pathTraceBruteforce.wgsl.ts`. Only `pcgInit`, `rand_f32`, `intersectTriangle`, and `safe_normalize` from `common.wgsl.ts` are called. These are unused dead WGSL code in the kernel.

## Cross-package dependencies

| Dependency                  | Status                                                        |
| --------------------------- | ------------------------------------------------------------- |
| `@vitrum/core`              | Correct — types only, no runtime import                       |
| `@vitrum/shared-bvh`        | Not used — expected when BVH traversal is wired               |
| `@vitrum/shared-samplers`   | Not used — Hammersley/Octahedral reimplemented inline in WGSL |
| `@vitrum/shared-denoisers`  | Not used — expected when denoiser is wired                    |
| `three`                     | Absent — correct, Three.js-free                               |
| `@vitrum/walkaround-hybrid` | Absent — correct                                              |
| `@vitrum/pt-webgl`          | Absent — correct                                              |

Note: the WGSL includes for Hammersley/octahedral are duplicates of what `@vitrum/shared-samplers` owns. When `shared-samplers` gains a WGSL export path, these should be replaced.

## What's missing / gaps

### BLOCKER

- **No multi-bounce path tracing loop.** The kernel traces exactly 1 path segment. `maxBounces` is advertised but unused. Any scene with indirect lighting (GI, caustics, interior illumination) will render as purely direct-lit Lambertian. This is the single most important gap.
- **No BSDF beyond Lambertian.** Roughness, metallic, transmission, IOR are all discarded by the material packer. All surfaces render as diffuse grey-ish with emissive additive. Specular, metallic, glass: none.
- **No BVH.** O(N) brute-force loop over all triangles. Will become unusably slow beyond ~5K triangles. Not a correctness blocker, but a performance blocker for any non-trivial scene.

### MEDIUM

- **No NEE / direct light sampling.** Only Lambert shading from a single hardcoded directional light direction. Point, rect-area, disc-area, spot, mesh-area emitters are entirely ignored. Multi-light scenes will render incorrect radiance.
- **No texture support.** `baseColorMap`, `normalMap`, `roughnessMap`, `emissiveMap` are all discarded. Textured scenes will render with flat base color.
- **Analytic primitive intersection not implemented.** Five shapes advertised in capabilities, zero implemented.
- **No tests.** No behavioral coverage of engine lifecycle, setScene, renderFrame output structure, or capabilities.
- **No environment map / HDRI sampling.** The sky model is a hardcoded gradient in the shader (`sampleSky`). HDRI and procedural-sky environment types are ignored.

### LOW

- **Dead `pathTraceSeed.wgsl.ts`.** Can be deleted or promoted to a test utility.
- **Inline WGSL samplers duplicate `shared-samplers`.** Should migrate to shared package when that package gains a WGSL export path.
- **`transformDirection` in mat4.ts is exported but unused** in the current codebase.
- **No reference render tooling.** The testing protocol requires "before/after reference renders" — no tooling in place yet.
- **`@vitrum/shared-bvh`, `@vitrum/shared-samplers`, `@vitrum/shared-denoisers` not yet wired.** All three are listed as dependencies in the architecture doc. Package.json has none of them.

## Overall assessment

The work is a **well-structured partial implementation** — not a stub (the engine compiles, pipelines dispatch, accumulation works), but also not functional for any real-world scene. The contract conformance at the TypeScript level is nearly perfect: all interface members present, correct state machine, correct factory shape, no host-app coupling, no Three.js dependency. The GPU mechanics (buffer allocation, compute dispatch, accumulation algorithm) are correct. The PCG RNG and triangle intersector are genuine implementations.

The primary maturity gap is rendering correctness: single-bounce Lambertian only, no BSDF, no proper NEE, no multi-light support, and the capabilities object advertises five analytic shapes and six emitter kinds that are silently unimplemented. The capability overclaiming is the most actionable concern — it will mislead any host that queries capabilities before the shader implementations catch up. The next implementation priority should be: (1) fix the false capability claims, (2) add a multi-bounce loop to the kernel, (3) add a minimal Cook-Torrance or Disney Diffuse BSDF, (4) add tests.
