# stainedGlass gap audit — 2026-05-12

## Integration shape

### Vitrum packages consumed
Source: `stainedGlass/packages/app/package.json` lines 34–40.

| Package | Role |
|---|---|
| `@vitrum/core` | `Engine`, `FrameInput`, `FrameOutput`, `FrameQualitySettings`, `detectGpu` |
| `@vitrum/three-bindings` | `sceneFromThreeJS` (PT path) — THREE.Scene → vitrum Scene conversion |
| `@vitrum/pt-webgl` | `createPTEngine_WebGL2`, `debounceMsForEditRate`, `bakeSkyEquirect`, `computeLightingState`, `SkyParams`, `PTEngineWebGL2FrameOutput` |
| `@vitrum/walkaround-hybrid` | `createWalkaroundEngine_Hybrid` |
| `@vitrum/shared-bvh` | Listed in package.json but not directly imported in the rendering code reviewed |
| `@vitrum/shared-denoisers` | Listed in package.json but not directly imported in the rendering code reviewed |
| `@vitrum/shared-samplers` | Listed in package.json but not directly imported in the rendering code reviewed |

Three.js fork (`three-gpu-pathtracer: file:../../../three-gpu-pathtracer`) is still consumed directly by `pt-webgl`; the app itself consumes it only transitively.

### Engines used and selection logic

**PT (final + preview mode):** `createPTEngine_WebGL2` — initialised by `useVitrumPTEngine` (`src/rendering/vitrum-bridge/useVitrumPTEngine.ts`). Active whenever `cameraMode === '3d'` and `pathTracerEnabled`.

**Walkaround (GI explorer mode):** `createWalkaroundEngine_Hybrid` — initialised by `useVitrumWalkaroundEngine` (`src/rendering/vitrum-bridge/useVitrumWalkaroundEngine.ts`). Active when the user switches to walkaround canvas.

**Raster (editor default):** Three.js / R3F direct — no vitrum engine involved.

### Bridge architecture

**PT path:** `useVitrumPTEngine` → `VitrumSceneSync` → `sceneFromThreeJS(threeScene)` → `engine.setScene(vitrumScene)`. The THREE scene carries baked `MeshPhysicalMaterial` instances; `sceneFromThreeJS` in `@vitrum/three-bindings/src/material.ts` reads standard PBR fields **plus** all `userData.vitrum*` stamps. `VitrumBlitPass` drives `engine.renderFrame()` each rAF and blits `output.primaryRadiance` to the canvas.

**Walkaround path:** `useVitrumWalkaroundEngine` calls `createWalkaroundEngine_Hybrid` with lighting state at construction time. The engine receives the raw THREE.Scene (`threeScene` option) and internally reads BVH geometry. Per-frame: `VitrumWalkaroundStage.tsx` calls `engine.renderFrame(input)`.

**Material baking:** `@stained-glass/physics/baking` (local package) builds `THREE.MeshPhysicalMaterial` from `FaceProperties`. Physics-rich fields (spectral curves, scatter params, thin-film stacks) are stamped on `material.userData` with `vitrum*` keys. `sceneFromThreeJS` reads those keys and projects them into `@vitrum/core` `Material` fields. Vitrum-extended fields flow: baker → userData → `convertMaterial()` → `Material` → `vitrumSceneToThree()` → THREE `MeshPhysicalMaterial.userData` on the engine-owned scene → fork GLSL uniforms (where the fork reads them).

---

## Workflow

### Real-time design loop
Raster mode. No vitrum engine active. Frame budget: ~16 ms (60 FPS target). PT preview quality mode: `interactive` (40 ms batch target per `ptEngineWebGL2.ts` line 189).

### Final-render mode
PT mode. Target: 2112 samples / 10 bounces (`pathtracerConstants.ts` lines 66–70). Budget: 900 s (15 min hero render). Timing gate: 60–120 s for preview convergence (192 samples, `PT_HONEYCOMB/LIGHTBOX/FULL_SCENE_TIMING_BUDGET_MS`).

### Material edit
`VitrumSceneSync` detects Redux selector changes and issues a debounced `engine.setScene()`. Debounce is adaptive via `debounceMsForEditRate()` from `@vitrum/pt-webgl`. Scalar-only changes (density, thickness, baseColor) go through `updateBakedGlassScalars()` which mutates the live THREE material in-place. There is no `engine.updatePrimitive()` path — pt-webgl throws on `updatePrimitive` (`ptEngineWebGL2.ts` line 581).

### Sun tracking / time-of-day
Walkaround: lighting is baked into `createWalkaroundEngine_Hybrid` creation-time options. `useVitrumWalkaroundEngine` quantizes `timeOfDay` to 8 slots and **recreates the engine** on slot boundaries (`useVitrumWalkaroundEngine.ts` lines 59–62). Comment at line 34: `"Sprint N: extend HybridEngine with an updateLighting() method to eliminate the recreation step for real-time scrubbing."` PT: sun is a `RectAreaLight` managed by `SunPathTraced`, re-synced to the engine via `setScene` on IBL changes.

---

## Concrete gaps found

### 🔴 Hard-blocked (host can't ship without vitrum change)

**Gap 1 — No `updateLighting()` on `HybridEngine` (walkaround time-of-day scrubbing)** ✅ RESOLVED

- **What stainedGlass needs:** Real-time scrubbing of the time-of-day slider in walkaround mode without a full engine teardown + rebuild cycle. Users expect the sun to move continuously like a real dial.
- **What vitrum delivers today:** `HybridEngineOptions.primaryLightDir/Intensity/skyTint/skyIrradiance` are creation-time-only options with no runtime update method. `HybridEngine.updatePrimitive` and `updateEmitter` are explicitly `never` (`HybridEngine.ts` lines 656–657).
- **What's missing:** A `HybridEngine.updateLighting(opts: Partial<LightingOptions>)` method (or equivalent) that re-uploads the sun UBO and resets the temporal accumulator without tearing down pipelines.
- **Fix scope:** Medium — UBO path exists; adding the method means defining the API surface, re-uploading affected uniform buffers per-frame, and invalidating DDGI probe cache.
- **File pointers:** `vitrum/packages/walkaround-hybrid/src/HybridEngine.ts:656`, `stainedGlass/packages/app/src/rendering/vitrum-bridge/useVitrumWalkaroundEngine.ts:34` (comment documents the gap explicitly).
- **Resolution:** Implemented in `feat/sweep-2026-05-12-followup` (commit after 4f5975d). `HybridEngine.updateLighting(opts: Partial<LightingOptions>)` added. `DDGI.invalidateProbeCache()` and `WalkaroundGPUPipeline.requestAccumReset()` added as internal plumbing. 9 tests in `packages/walkaround-hybrid/__tests__/hybridEngineLighting.test.ts`. stainedGlass host can replace the 8-slot engine-recreation dance with a single `engine.updateLighting(computeLightingState(...))` call on every timeOfDay selector change.

**Gap 2 — `spectralAttenuation`, `thinFilmStack`, `scatteringCoefficient` flow through the contract but are NOT consumed by the three-gpu-pathtracer fork (pt-webgl path)**

- **What stainedGlass needs:** Per-wavelength Beer-Lambert attenuation (cobalt, iron, Se/Cd, gold ruby, etc.) and TMM dichroic evaluation to drive true spectral color in final PT renders.
- **What vitrum delivers today:** `@vitrum/three-bindings` `convertMaterial()` reads all `userData.vitrum*` stamps and maps them to `Material.spectralAttenuation`, `Material.thinFilmStack`, `Material.scatteringCoefficient`, `Material.dispersionAbbeNumber`. `vitrumSceneToThree()` stamps them back onto the THREE scene's `userData`. But `forkUniformBridge.ts` only uploads the **global** CMF tables (`uCmfX/Y/Z`, `uYCmfCdf`) to the fork material uniforms — it does NOT upload per-material spectral curves, thin-film stacks, or volume-scatter coefficients. Comment at `forkUniformBridge.ts` line 115: `"per-material scalar drives now come from the fork MaterialsTexture packing path"` — implying the fork is expected to read from the materials texture, but the host has no confirmed evidence this path is implemented in the fork.
- **What's missing:** Either (a) per-material spectral data uploaded as uniforms/textures to the fork shader and consumed in the fork's BSDF, or (b) a hero-wavelength spectral path in pt-webgl that replaces RGB Beer-Lambert when `spectralAttenuation` is present. The host bakes and stamps the data correctly; vitrum's pt-webgl backend drops it on the floor.
- **Fix scope:** Large — requires fork shader work (`three-gpu-pathtracer` fork) and per-material uniform/texture upload in pt-webgl.
- **File pointers:** `vitrum/packages/pt-webgl/src/forkUniformBridge.ts:115`, `stainedGlass/packages/stained-glass-physics/src/baking/spectralAbsorption.ts`, `stainedGlass/packages/stained-glass-physics/src/baking/createBakedGlassMaterial.ts:355–358`.

**Gap 3 — `scatterTint` / Mie forward-scatter RGB modulation has no vitrum receiver**

- **What stainedGlass needs:** Opalescent, wispy, ringMottled glass types have a `scatterTint` field (`FaceProperties.scatterTint`) intended to modulate the scattered color (slightly cool blue bias, `[0.85, 0.92, 1.0]` default). The field comment says "interim RGB-tint while vitrum volume-scatter ships" (`properties.ts` line 133).
- **What vitrum delivers today:** `@vitrum/core` `Material` has `scatteringCoefficientRGB: Vec3` for chromatic scattering. `convertMaterial()` reads `userData.vitrumScatteringCoefficientRGB`. But `createBakedGlassMaterial.ts` stamps `vitrumScatteringCoefficientRGB` as the attenuation color `[attenuationColor.r/g/b]` (line 376), NOT as `scatterTint`. The `scatterTint` field on `FaceProperties` is never stamped on `userData` and is never read by any vitrum bridge path.
- **What's missing:** Either wire `scatterTint` → `vitrumScatteringCoefficientRGB` in the bake pipeline (minor baking change), or document that `scatterTint` is host-only and the Mie-bias effect is approximated via `attenuationColor`.
- **Fix scope:** Small — if wiring is desired, it's a 3-line change in `createBakedGlassMaterial.ts`. But the larger issue (vitrum pt-webgl doesn't consume `scatteringCoefficientRGB` in the fork anyway — see Gap 2) means this is blocked by Gap 2.
- **File pointers:** `stainedGlass/packages/stained-glass-physics/src/types/properties.ts:133`, `stainedGlass/packages/stained-glass-physics/src/baking/createBakedGlassMaterial.ts:369–378`.

---

### 🟡 Worked around (host has a hack; native vitrum support would be better)

**Gap 4 — THREE.js version skew (0.171 app vs 0.184 vitrum): all bridge calls cast via `as any`**

- **Workaround:** `useVitrumPTEngine.ts:31`, `VitrumSceneSync.tsx:70`, `useVitrumWalkaroundEngine.ts:94`, `ptEnvironment.ts:58,92` — all cast the THREE.js scene/renderer to `any` with a comment explaining the 0.171 vs 0.184 structural skew.
- **Risk:** TypeScript safety is fully suspended at these call sites. If vitrum adds a call to a 0.180+ API internally (e.g., `transmissionResolutionScale`, `setEffects`, `static`, `pivot`) the cast will succeed at TS compile time but throw at runtime.
- **Fix scope:** Small — upgrade `three` in `stainedGlass` from 0.171 to 0.184. Requires verifying no 0.171→0.184 breaking changes affect the rest of the app's THREE usage.
- **File pointers:** `stainedGlass/packages/app/package.json:21` (`three: 0.171.0`), `vitrum/packages/three-bindings/package.json:21` (`three: ^0.184.0`).

**Gap 5 — Anisotropy/anisotropyRotation from baked materials not round-tripped through vitrum Material contract**

- **Workaround:** `createBakedGlassMaterial.ts:275–278` sets `anisotropy` and `anisotropyRotation` directly on the `MeshPhysicalMaterial` for directional rolled textures (ripple, waterglass). `convertMaterial()` in `vitrum/packages/three-bindings/src/material.ts` does NOT read these fields — they are absent from `@vitrum/core`'s `Material` type and absent from the `convertMaterial` function. `vitrumSceneToThree()` similarly doesn't stamp them.
- **Effect:** For the PT path (`sceneFromThreeJS → vitrumSceneToThree → fork setScene`), the anisotropy baked on the original THREE material is LOST in the vitrum round-trip. The fork receives a `MeshPhysicalMaterial` with `anisotropy = 0` (the default) regardless of what the host baked. Ripple and waterglass cells render as isotropic in PT mode.
- **Fix scope:** Small — add `anisotropy?: number` and `anisotropyRotation?: number` to `@vitrum/core` `Material`; read them in `convertMaterial()`; write them in `vitrumSceneToThree()`. Then the fork already handles anisotropy via its existing `MeshPhysicalMaterial` plumbing.
- **File pointers:** `vitrum/packages/three-bindings/src/material.ts:42–162` (no anisotropy read), `vitrum/packages/core/src/scene.ts:173–313` (no anisotropy field), `stainedGlass/packages/stained-glass-physics/src/baking/createBakedGlassMaterial.ts:275–278`.

**Gap 6 — `foilBackingTint` on `EdgeProperties.foilBacking` — unwired in edge material**

- **Workaround:** `useEdgeMaterial.ts:19` contains a `TODO Phase 4 follow-up: read material.userData.foilBackingTint here and apply as edge-face transmission tint when cameType === 'copperFoil'`. The `FaceProperties`/`EdgeProperties` contract has `foilBacking?: FoilBacking` (`properties.ts:224`), but the edge material factory never stamps `userData.foilBackingTint`.
- **Effect:** The copper foil backing color (copper / black / silver) that modulates edge-face transmission tint in PT mode is never applied. The effect is subtle (edge-seam tinting) but authored in the physics package and documented as a gap.
- **Fix scope:** Small — create the `userData.foilBackingTint` stamp in the edge material factory, then read it in `useEdgeMaterial.ts`. Vitrum itself doesn't need to change; this is host-side wiring.
- **File pointers:** `stainedGlass/packages/app/src/rendering/edges/useEdgeMaterial.ts:19`.

**Gap 7 — `vitrumFrontLayer` / `vitrumBackLayer` (RFE-03) intentionally deferred (silver stain / painted trace)**

- **Workaround:** `createBakedGlassMaterial.ts:388–392` has an explicit comment: `"vitrumFrontLayer / vitrumBackLayer (RFE-03 layered BSDF): intentionally NOT stamped. … When painting unfreezes, this becomes the silver-stain integration point."` The vitrum contract has `frontLayer`/`backLayer` on `Material` and `convertMaterial()` would read them, but the host never stamps them.
- **Effect:** Silver stain on the back face and painted trace on the front face cannot be rendered correctly in PT mode until painting is un-deferred. Vitrum's contract side is ready; the host-side authoring flow is blocked on the Decision C milestone.
- **Fix scope:** N/A for vitrum itself — the contract exists. Host-side painting flow needed.
- **File pointers:** `stainedGlass/packages/stained-glass-physics/src/baking/createBakedGlassMaterial.ts:388–392`.

---

### 🟢 Working as expected (no gap)

- **Base PBR fields (transmission, IOR, attenuationColor, attenuationDistance, thickness, roughness, metalness):** Correctly stamped on `MeshPhysicalMaterial`, read by `convertMaterial()`, passed through the round-trip. Verified in `material.ts:70–78`.
- **Iridescence (single-layer, iridescent-lustre, iridescent-peacock):** `iridescence`, `iridescenceIOR`, `iridescenceThicknessRange` all pass through. Per-cell `iridescenceThicknessNm` correctly collapses range to `[d,d]`. Verified in `createBakedGlassMaterial.ts:305–316`.
- **`dispersionAbbeNumber` (bevels):** Stamped as `vitrumDispersionAbbeNumber`, read by `convertMaterial()` → `Material.dispersionAbbeNumber`. Verified in `material.ts:106–109`.
- **Strike modulation (seCd, goldRuby):** Correctly shifts `attenuationColor` in `strikeModulatedColor()`. Flows through RGB Beer-Lambert path as expected.
- **Solarization (Mn-decolorized glass):** `solarizedColor()` applies amethyst shift to `attenuationColor`. Same path.
- **Volume scatter params for opalescent/wispy/ringMottled/dalleDeVerre:** `vitrumScatteringCoefficient`, `vitrumScatteringAnisotropy`, `vitrumScatteringCoefficientRGB` all stamped and read. The contract plumbing is complete; actual volumetric rendering in the fork is the open question (see Gap 2).
- **`spectralAttenuation` data (baking side):** All 7 colorant curves (cobalt, iron, copper, Mn, SeCd, goldRuby, dalle amber) are correctly computed and stamped. The issue is pt-webgl not consuming them (Gap 2), not the data being wrong.
- **`thinFilmStack` for dichroic:** DICHROIC_STACK_DEFAULT (35-layer quarter-wave TiO₂/SiO₂ stack) correctly computed and stamped. Same consumption gap as Gap 2.
- **`surfaceTextureId` for walkaround procedural dispatch:** Stamped in `createBakedGlassMaterial.ts:343`, read in `walkaround-hybrid/src/restir/packingHelpers.ts:127`. This round-trip works.
- **Adaptive PT debounce on material edits:** `debounceMsForEditRate` from pt-webgl correctly throttles `setScene` calls on rapid slider drags. Verified in `VitrumSceneSync.tsx:59`.
- **Engine lifecycle (mount/unmount/Canvas-remount):** `useVitrumPTEngine` and `useVitrumWalkaroundEngine` correctly cancel creation if unmounted before resolution, dispose on `gl`/`device` change. Verified in both hooks.
- **Camera per-frame update:** `VitrumBlitPass` correctly passes `viewMatrix`, `projMatrix`, `cameraPosition` from the live R3F camera on every rAF. Works correctly.
- **IBL / sky / HDRI:** `bakeSkyEquirect` + `usePTEnvironment` pattern correctly handles all four backdrop modes (sky, night, studio, sunset) and outdoor HDRIs via `RGBELoader`. `setScene` fires after each change.
- **Sun in PT mode (RectAreaLight):** `SunPathTraced` mounts a `RectAreaLight` replacing the directional sun; NEE samples it for soft shadows. Correctly suppressed when outdoor HDRI bakes in a sun lobe.
- **SwiftShader gate (walkaround):** `detectGpu()` from `@vitrum/core` and `refuseToMount` gate work correctly. Verified in `VitrumWalkaroundStage.tsx:66`.

---

## Add-to-in-flight recommendations

### Fold into current sweep (sweep-2026-05-12-followup.md)

**Gap 4 (three.js version upgrade):** This is a pre-requisite for structural type safety at all bridge call sites. Small scope; can be done in the Tier 1 closeout branch without touching rendering logic. Zero vitrum changes needed.

**Gap 5 (anisotropy round-trip):** Small contract addition in `@vitrum/core/scene.ts` + two-file change in `@vitrum/three-bindings`. Ripple/waterglass cells would gain correct anisotropic highlight in PT final render. Natural fit for the Tier 1 closeout.

**Gap 6 (foilBackingTint wiring):** Host-only. One-function change in the edge material factory and one-line read in `useEdgeMaterial.ts`. No vitrum change needed.

### Defer (Tier 2 / separate sprint)

**Gap 1 (updateLighting for walkaround):** ✅ Implemented in `feat/sweep-2026-05-12-followup`. See Gap 1 resolution note above.

**Gap 2 (spectral/TMM/scatter consumption in pt-webgl):** This is effectively Tier 2.H5 (fork patches needed) + additional fork shader work for per-material spectral curves. Large scope. Prerequisite for scientifically accurate stained-glass color rendering in final PT mode. Block it on the fork-patch sprint chain (H6.2–H6.5). The host-side data is already correctly produced; this is purely a backend (fork) implementation gap.

**Gap 3 (scatterTint wiring):** Trivial host change, but only meaningful after Gap 2 lands (pt-webgl needs to consume scatteringCoefficientRGB before the wire-up has any effect).

**Gap 7 (vitrumFrontLayer/backLayer):** Blocked on painting Decision C. Defer until painting milestone is scheduled.

---

## Surprising findings

1. **The vitrum userData round-trip is a two-way loop, not a one-way pipe.** The host bakes `userData.vitrum*` onto THREE materials → `sceneFromThreeJS` reads them into `vitrum.Material` → `vitrumSceneToThree` stamps them BACK onto a new `MeshPhysicalMaterial.userData` for the fork. The fork is then expected to read these from its own materials texture packing path. If the fork's `MaterialsTexture` packing does not include these keys, the data vanishes silently. This is the root of Gap 2.

2. **`sceneFromThreeJS` is called TWICE per scene change in the PT path** — once in `VitrumSceneSync.tsx` and once in `ptEnvironment.ts`'s `notifyEngine()`. Both call `engine.setScene(sceneFromThreeJS(scene))`. This means every IBL change (e.g., sky → studio) triggers a redundant full BVH rebuild. Low priority (each rebuild is cheap in PT mode relative to convergence time) but worth knowing.

3. **Walkaround engine does NOT go through `sceneFromThreeJS` / the vitrum Scene contract.** It receives the raw `THREE.Scene` directly via `HybridEngineOptions.threeScene`. All the `userData.vitrum*` stamps that the baking pipeline sets are therefore available to the walkaround engine's internal material readers — but `walkaround-hybrid/src/restir/packingHelpers.ts` only reads `surfaceTextureId` from `userData`. All other vitrum-extended fields (spectralAttenuation, scatteringCoefficient, thinFilmStack) are silently ignored in walkaround mode. For walkaround this is expected (GI explorer quality, not spectral PT), but it means the contract assumption "stamp once, all backends consume" is only fully true for pt-webgl.

4. **PT convergence target is 192 samples (preview) / 2112 samples (final)** at 5 bounces/10 bounces respectively (`pathtracerConstants.ts`). The 60 s honeycomb gate, 90 s lightbox gate, and 15 min final budget are explicitly coded. This confirms the 4090 "screams" claim — 192 samples converges in under 60 s and 2112 in under 15 min.
