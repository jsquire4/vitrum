# Road to 100% — vitrum

> Authored 2026-06-09 from a full code-truth re-audit (15 packages read line-by-line by
> deep-reader agents; every load-bearing claim re-verified by lead code-read/grep).
> Scope per the user's request: **frontier-feature completion, fidelity, provisioning,
> hygiene.** Deliberately EXCLUDES public-distribution posture, release governance, and
> cross-host GPU-validation evidence (tracked separately in `HARDWARE-VALIDATION-NEEDS.md`).

## Where we actually are

> **Updated 2026-06-16 after the latest code-first closure wave (receiver-lobe GI target, glTF/OIT predictable-tail closures, point/line fallback geometry, and pt-webgl2 analytic fallback parity).**
> **2026-06-17 strictness/proof follow-up:** walkaround learned/research opt-ins now surface through `capabilities.experimentalFeatures`
> (`walkaround-hybrid-gris-unbiased-reuse`, `walkaround-hybrid-ppg-guided-gi`,
> `walkaround-hybrid-nrc-biased-cache`, `walkaround-hybrid-neural-denoiser-host-weights`) and
> `denoiser:'neural'` emits a structured host-weights/no-production-checkpoint warning; `loadGltfForEngine(...,
> compatibilityMode:'reject-degraded')` now rejects generated tangents, missing tangent texcoord, tangent-generation
> failures, and ignored high-UV texture fields as approximate import degradations instead of allowing construction;
> `ppgGuidedSampling.test.ts` now mirrors the production equal-area cylindrical PPG direction map rather than the
> retired octahedral map.
> For this ledger, "100%" = everything fully implemented.
> **R7a-R7d campaign additions:** behavioral gate coverage (43 lanes today: 33
> pt-webgpu + 10 walkaround-hybrid; permanent CI); anisotropic
> GGX (A-item closed — `materialAnisotropy` now renders); engine error surface (`onError` —
> silent-GPU-error class dead); `@vitrum/gltf-adapter` new package (glTF 2.0 → core Scene);
> `captureFrame` pixel-readback API + `pickPrimitive` real on all 3 backends;
> `CameraLike`/`QualityTier`/presets public; examples/ + debugging runbook docs; IES dead
> chain removed; spectral×photon-map gather spectralized; giState v4 (PPG warm restore);
> SPPM streaming-window corrected (non-progressive — see A4-progressive below).
> **A4-progressive DONE (2026-06-10; radius/capacity follow-ups 2026-06-16):** true Hachisuka per-pixel SPPM — `SppmPixelStats`
> binding(9), `sppmGatherProgressive` update rule (N′=N+α·M; R′²=R²·ratio; τ′=(τ+Φ_M)·ratio),
> stable r₀ insertion-grid lookup plus shrunk per-pixel physical gather disk, bounded reservoir
> replacement and totalInserted/storedCount compensation for over-capacity cells, buffer reset on
> camera/scene/reset, Cesàro accumulator argument, TS-mirror tests.
> **Latest closure wave:** pt-webgpu inverse path replay now only advertises the
> direct-light domains it actually supports (delta directional, point, spot, and
> stochastic area-measure rect/disc/mesh-area) and falls back to finite difference for most
> maps except scoped camera-direct emissiveMap replay and baseColorMap/COLOR_0
> plus roughnessMap/metallicMap, specularColorMap/specularIntensityMap,
> clearcoatMap/clearcoatRoughnessMap, sheenColorMap/sheenRoughnessMap,
> iridescenceMap/iridescenceThicknessMap, anisotropyMap, and aoMap local chain/state
> factors, plus transmission, layered, spectral/volume, environment escape,
> indirect, and other unsupported light/material cases; inverse sessions now
> expose structured downgrade diagnostics when requested path replay falls back
> to finite difference; walkaround material truth was tightened so textured
> alpha blend emits structured approximation warnings and the ledger no longer
> claims native rows where GI reuse still used stored-Lo/proxy targeting; neural
> weights are validated before allocation, the tracked research checkpoints are
> covered by loader/spec tests, dispatch failures fall back to raw HDR,
> NRC opt-in emits a structured experimental/biased warning, and glTF
> `decodeSceneTextures(target:"webgpu")` now actually resolves raw image handles
> to backend-ready texture payloads. Latest pt-webgpu inverse follow-up: transformed
> mesh/skinned scenes now bake a transient world-space flat replay stream for the
> adjoint pass (positions, normals, uv0/uv1, tangents, colors, vec4 indices, and
> live material IDs), so non-identity mesh transforms no longer force a path-replay
> downgrade. Follow-up: non-empty `instanced-mesh` primitives now use that same
> transient world-space stream, baking every instance into the adjoint replay geometry;
> zero-instance targets remain structured finite-difference downgrades. **2026-06-19
> follow-up:** full-tier supported analytic primitives now feed the same replay stream
> through deterministic core analytic-to-mesh tessellation, with lite/unsupported
> analytic shapes still failing closed to finite difference instead of pretending to
> native analytic-intersection adjoint parity.
> **2026-06-18 source-verified follow-up:** pt-webgpu primary-hit path replay now keeps
> unlit emissive fits, emissive/light-map primary terms with normal-only maps, and
> unlit AO map intensity with irrelevant BRDF lobe flags on the analytic route;
> glTF `reject-degraded` rejects generated flat normals; walkaround atlas-backed
> maps warn on unsupported `texCoord` values instead of silently disabling; DDGI
> probe rays consume readable `transmissionMap`/`thicknessMap` payloads for glass
> visibility and direct probe-hit glass mixing.
> into backend-readable CPU texture payloads instead of returning a no-op. glTF
> required-extension, generated-tangent, tangent-failure, missing-UV, and ignored-camera
> degradations now surface as source-pathed `GltfImportDiagnostic` rows instead of only
> free-form warning strings, and glTF-origin texture decode/readiness diagnostics now point
> at their original `materials[i]...Texture` slots instead of only scene material slots.
> Same-day DDGI vertex-alpha follow-up: probe alpha visibility now carries the
> shared-BVH `COLOR_0` rgba stream into a DDGI-local vertex-color texture, keys
> probe bind-group reuse on that view, and multiplies vertex alpha into
> `opacity * baseColorMap.a * alphaMap.r` coverage just like the main material
> atlas path.
> **2026-06-19 RC vertex-alpha follow-up:** RC probe shadow/transmittance now
> binds the same per-vertex `COLOR_0` texture exposed by the main walkaround
> scene bind group, keys RC bind-group reuse on that view, and multiplies
> vertex alpha into the existing `opacity * baseColorMap.a * alphaMap.r`
> coverage approximation. Raw RC callers get an opaque-white placeholder, so
> the old scalar/material-map behavior is preserved when no vertex-color stream
> is supplied. This closes the RC/DDGI/main-material alpha-coverage parity slice;
> it does **not** promote transparent layers to full path-space GI transport
> vertices.
> **2026-06-19 DDGI finite-light glass follow-up:** DDGI point/spot direct
> lighting and mesh-area emitter NEE now use `ddgiTraceShadowVisibility()`, an
> RGB visibility walk that applies atlas alpha, readable `transmissionMap`,
> readable `thicknessMap`, and Beer-Lambert attenuation for glass blockers.
> Fully transparent alpha-glass skips Beer, while glass continuation uses the
> same scale-aware exit/advance policy as sun visibility. This closes the
> DDGI sun-vs-finite-light transparent-shadow parity slice; it does **not**
> make transparent surfaces ReSTIR/GI transport vertices.
> pt-webgpu extension-lobe CPU reference tests now pin clearcoat, sheen,
> iridescence zero-default, and normalized sampled-PDF behavior; this closes the
> lobe-specific unit-proof tail but does **not** close GPU material-furnace /
> reference-render promotion. Follow-up code in this wave added map-free
> `shadingModel:'unlit'` baseColor and map-free clearcoat scalars to the safe
> pt-webgpu path-replay adjoint slice, added map-free scalar `iridescence` to
> that same direct-light adjoint slice, added map-free scalar `iridescenceIor`
> via a local thin-film-F0 derivative, added map-free scalar `anisotropy` /
> `anisotropyRotation` through a local anisotropic-GGX derivative, added
> baseColorMap/COLOR_0, roughnessMap/metallicMap,
> specularColorMap/specularIntensityMap, clearcoatMap/clearcoatRoughnessMap,
> sheenColorMap/sheenRoughnessMap, iridescenceMap/iridescenceThicknessMap,
> and anisotropyMap local chain/state factors for scoped pt-webgpu
> path-replay fits, and follow-up adjoint partials keep baseColor,
> roughness, metallic, specularColor, and specularIntensity on path replay
> for anisotropic direct-light BRDF materials instead of downgrading the
> whole fit to finite difference merely because KHR_materials_anisotropy is
> present, made transparent-OIT direct sun
> cast-shadow-aware, rejected adjoint path replay for primitive targets the
> triangle-only replay pass cannot actually hit, added material-lobe analytic point/spot and
> camera-visible finite-emitter direct lighting to transparent OIT, and
> warm-up gated NRC substitution so spread-fired records
> train from frame 0 while cold MLP predictions cannot replace DDGI suffix
> radiance. Same-day follow-ups on `main`: `loadGltfWithEngine()` reports the
> negotiated pt-webgpu runtime profile even in best-effort/existing-engine paths,
> `attachVitrum`/`VitrumCanvas` can opt into RAF-driven glTF controller playback
> with real delta seconds, DDGI probe direct-light visibility samples
> atlas-backed baseColor/alpha-map coverage for blend/mask shadow transmittance,
> DDGI mesh-area emitter NEE samples mapped source texels when TLAS metadata is
> available, and pt-webgpu inverse sessions finite-difference scalar transmission,
> thickness, attenuation, opacity, alphaCutoff, dispersion, and scattering controls
> while keeping requested path replay on a
> structured finite-difference downgrade until those visibility/transport terms
> are mirrored in the adjoint pass. Latest follow-up: pt-webgpu path replay now
> differentiates deterministic direct emitter `color` / `intensity` for delta
> directional, point, spot, rect-area, disc-area, and uncapped explicit mesh-area
> lights by scattering through the scoped direct-light BRDF and the matching
> light attenuation/geometric factors; the finite-area replay now samples
> rect/disc/mesh surfaces with area PDFs instead of center points. Soft-sun,
> capped/reordered or unmapped mesh-area emitter targets, exact texel-PDF mesh
> emission, forward light-selection MIS parity, environment/indirect transport,
> and unreplayed receiver material domains still downgrade with structured
> diagnostics. Latest diagnostic cleanup: finite-
> difference-only inverse fields now distinguish transport tails
> (`ior`/`transmission`/`thickness`/`attenuationColor`/`attenuationDistance`/
> `dispersionAbbeNumber`/`scatteringCoefficient`/`scatteringAnisotropy`/
> `scatteringCoefficientRGB`) from visibility tails
> (`opacity`/`alphaCutoff`) via contract-level diagnostic codes. Latest RC
> emitter follow-up: RC probe-cast emitter NEE now receives the main material
> atlas and samples UV0/UV1 material-backed emissive texels through the packed
> `EmitterTri` source-triangle/subdivision lanes, falling back to scalar `Le`
> only for unmapped emitters and omitted atlas bindings.
> Latest emissive-map emitter follow-up: CPU-readable emissive maps now split
> mesh-area emitter support against exact transformed texel cells in shared CPU
> packing, and pt-webgpu, pt-webgl2, and walkaround consume those clipped
> constant-radiance sub-triangles before falling back to bounded barycentric
> quadrature. This removes the old fixed-subdivision estimator for readable
> simple wrap/clamp maps. 2026-06-18 pt-webgl2 follow-up: pt-webgl2 mesh-area
> NEE now stores per-sub-triangle `selectionPower = luminance(radiance)·area`,
> samples mapped emissive texel cells by emitted power, uploads
> `uTotalEmissivePower`, and computes the forward emissive-hit MIS PDF from
> `luminance(surf.emission) / totalPower`, closing the pt-webgl2 forward/sample
> PDF parity slice. Cross-backend energy-alias promotion and render A/B evidence
> remain validation/promotion tails. Same-day pt-webgpu cap follow-up:
> pt-webgpu mesh-area NEE cap ranking now keeps the highest
> `luminance(radiance)·area` triangles instead of the largest geometric-area
> triangles, so UV-local emissive texel sub-triangles are not discarded ahead of
> darker but larger triangles when the cap is reached.
> Latest inverse API follow-up: pt-webgpu path replay now mirrors top-level
> `normalMap` / `normalScale`, `bumpMap` / `bumpScale`, and
> `clearcoatNormalMap` / `clearcoatNormalScale` in the scoped direct-light
> adjoint path. The normal paths use the forward shader's authored/generated
> tangent.xyzw handedness before UV-gradient fallback, and the clearcoat path
> samples its independent tangent-frame normal for the additive clearcoat lobe.
> Current follow-up: the adjoint pass now evaluates replay-local full normal-stack
> central differences, so `normalScale` remains on path replay through bump and
> clearcoat-normal tangent frames, and `bumpScale` remains on path replay through
> clearcoat-normal tangent frames.
> `aoMapIntensity` stays on scoped path replay as the local derivative of glTF
> AO's baseColor multiplier, `lightMapIntensity` path-replays the primary-hit
> baked-radiance partial, and `envMapIntensity` path-replays the direct
> HDRI/procedural-sky NEE multiplier while environment escape/indirect transport
> remains outside scope.
> Latest transport follow-up: `attenuationColor`, `attenuationDistance`,
> `dispersionAbbeNumber`, `scatteringCoefficient`, `scatteringAnisotropy`, and
> `scatteringCoefficientRGB` now join `transmission` and `thickness` as
> finite-difference inverse parameters with a structured transport downgrade for
> requested path replay.
> Latest vec2 inverse follow-up: `iridescenceThicknessRange` now fits through
> `kind:'vec2'` inverse sessions and stays on scoped path replay for both
> map-free ranges and readable `iridescenceThicknessMap` ranges by
> differentiating sampled thin-film thickness and chaining it to min/max
> endpoints through `V·H` or the sampled G-channel texel.
> Latest API truthfulness follow-up: glTF texture decode reports now include
> decoded dimensions, original/resize metadata, texture/image/sampler provenance,
> and selected texture-source extension provenance, and decoded spec-gloss alpha
> roughness bakes remove the now-satisfied glossiness-alpha compatibility issue
> for direct asset callers. pt-webgpu inverse path replay also gates on the real
> forward render regime: multi-bounce or spectral baselines now downgrade with a
> structured `path-replay-unsupported-render-regime` diagnostic because the
> current adjoint is a single-bounce RGB direct-light replay.
> Latest arbitrary-glTF hardening follow-up: authored `TANGENT` and `COLOR_0`
> accessors are now validated against the core buffer contract before import,
> all-degenerate UV tangent generation fails with a tangent-generation diagnostic
> instead of fabricating a clean tangent frame, and texture decode diagnostics /
> hooks carry selected KTX2/DDS/WebP source provenance.
> Latest walkaround point/spot follow-up: analytic shade + transparent OIT +
> DDGI probe-light + RC point/spot paths now all consume authored
> `distance`/`decay`, DDGI/RC spot cones use the same forward-axis convention
> as shade/OIT (`dot(-axis, receiverToLight)`), and hard-edge
> `penumbra:0` avoids `smoothstep(edge, edge, x)` undefined behavior. GPU
> recapture remains a V28-B proof item because this is render-changing.
> Same-day analytic point/spot radiometry follow-up: opaque shade and
> transparent OIT no longer multiply receiver `NdotL` after the shared
> extension-aware `evalGGX*` helpers, whose return value is already
> cosine-weighted. `nDotL` remains a hemisphere gate only; V28-B recapture
> remains required because grazing point/spot brightness changes visibly.
> Latest walkaround soft-sun truthfulness follow-up: authored
> `DirectionalEmitter.angularDiameter` now threads into the shared
> WalkaroundUBO as a direct sun cone radius and is consumed by opaque direct
> sun NEE, transparent OIT sun lighting, and stained-glass caustics; the old
> hard-coded 0.00436 rad radius remains only as the no-authored-diameter
> default. **2026-06-19 follow-up:** DDGI now carries the same authored sun
> cone radius in the packed sun `innerCone` lane and samples deterministic
> per-hit soft-sun directions for probe direct lighting; RC repurposes the
> existing cascade-uniform sun padding lane as `sunAngularRadius` and applies
> the same deterministic cone sample to direct probe hits and glass
> continuation sun terms. The old
> `walkaround-hybrid.directional-angular-diameter-partial-support` warning is
> removed. V28-B recapture remains required because this is render-changing.
> Same-day pt-webgl2 soft-sun follow-up: directional emitters now pack
> positive finite `angularDiameter` into the lights texture, GLSL decodes it,
> and `randomLightSample()` samples a finite cone with a solid-angle PDF instead
> of warning that the field is ignored. Hard directional lights keep the legacy
> delta shortcut; finite cones enter the regular MIS path. GPU recapture remains
> a promotion/proof item because this is render-changing.
> **2026-06-18 walkaround mutation follow-up:** material-only scalar edits now
> refresh DDGI's `RestirBvhSnapshot` material payload without RC geometry
> propagation, and roughness/metallic scalar edits invalidate DDGI probe cache
> because the DDGI glossy probe bounce consumes those fields. Same-day alpha
> metadata follow-up: scalar `alphaMode` / `opacity` / `alphaCutoff` material
> patches now rebuild the material texture atlas even without `alphaMap`, so
> transparent OIT and alpha visibility paths cannot keep stale opaque metadata
> after a material-only update.
> **2026-06-18 pt-webgpu BDPT shadow follow-up:** BDPT bounce-0 emitter vertices
> now mirror authored `castShadow:false` into the light-subpath payload, and
> eye↔light connection visibility skips the occlusion ray for that emitter
> endpoint.
> **2026-06-18 pt-webgpu MNEE shadow follow-up:** point-light MNEE reflection,
> refraction, and glass-slab caustics now decode the packed point-light
> `castShadow:false` lane and skip only the point-emitter light-leg visibility
> test, matching the finite-area MNEE behavior while preserving receiver/interface
> validity checks.
> Same-day SPPM shadow follow-up: `causticStrategy:"photon-map"` now excludes
> `castShadow:false` emitters from photon-source selection and renormalizes the
> source PDF over the remaining shadow-casting sources, so no-shadow emitters
> remain direct/camera/specular-visible without seeding photon-map caustic/shadow
> transport. The pt-webgpu `emitterCastShadow` promise row is now native.
> Same-day API/update truthfulness follow-up: `@vitrum/engine/gltf` re-exports
> the texture decode report/diagnostic types it already forwards at runtime, and
> walkaround `updatePrimitive()` now warns for `receiveShadow:false` plus routes
> mixed material+geometry patches through a full rebuild so the material patch
> is applied and unsupported material fields cannot be hidden by geometry fast
> paths.
> Same-day walkaround no-op diagnostic follow-up: invalid `setSize(width,height)`
> calls with non-positive dimensions still no-op to avoid zero-sized WebGPU
> allocations, but now emit structured `walkaround-hybrid.invalid-set-size`
> warnings with the rejected dimensions; non-empty `updatePrimitive()` patches
> whose fields are not recognized still no-op for host pass-through compatibility
> but now emit `walkaround-hybrid.unknown-primitive-patch-fields` with exact
> primitive id and field list.
> Same-day lite-tier diagnostic follow-up: `tier:'lite'` overriding
> `extensions['walkaround-hybrid'].bvhMode:'tlas'` now preserves the historical
> console warning and also emits structured
> `walkaround-hybrid.lite-bvh-mode-overridden` through the host `onWarning`
> callback with requested/effective BVH mode details.
> Same-day PT runtime-warning follow-up: pt-webgl2 and pt-webgpu `setScene()`
> unsupported-material/displacement warnings now retain the aggregate `fields`
> list while also reporting exact `primitiveIds` plus per-primitive
> `primitiveFields`, matching the mutation-path diagnostic specificity.
> Same-day pt-webgl2 skinning-warning follow-up: empty-bones skinned meshes
> still render in rest pose, but scene ingestion now routes that fallback through
> structured `pt-webgl2.skinned-mesh-empty-bones` warnings with the primitive id
> and `fallback:"rest-pose"` instead of being console-only.
> Same-day pt-webgpu skinning-warning follow-up: initial `setScene()` skin-solve
> failures now mirror the mutation path by emitting structured
> `pt-webgpu.set-scene-skin-fallback` warnings with the primitive id and
> `fallback:"rest-pose"`, while direct packer calls keep their console fallback.
> Same-day DDGI runtime-warning follow-up: missing-device skips, GPU-init
> disabled fallback, and missing-core-scene BVH skips now route through DDGI's
> structured warning sink (`walkaround-hybrid.ddgi-*`) while standalone DDGI
> keeps console fallback and existing non-fatal `onError` reports.
> Same-day DDGI WebGPU-acquisition follow-up: `ProbeUpdatePass` now routes
> `navigator.gpu.requestAdapter` failures and no-WebGPU probe-update init
> failures through structured `walkaround-hybrid.ddgi-*` warnings when an
> `onWarning` sink is present, preserving console fallback for standalone tests.
> **2026-06-19 walkaround proof-harness follow-up:** the radiometric A/B host
> harness now uses stricter visible material cases for glass/glossy probes and
> treats a glossy-material `FINDING` as partial evidence only: enough to prove
> the scene changes materially, not enough to promote rich-material GI. The
> proof check therefore accepts that case without counting it as a full
> promotion result.
> Same-day DDGI sub-pass hardening: `DDGI` now routes `ProbeUpdatePass`
> construction warnings through its guarded `_warn` wrapper, so a throwing
> standalone host warning callback cannot break DDGI construction.
> Same-day React attach-error follow-up: `<VitrumCanvas>` initial async attach
> failures now route through guarded `onAttachError` and structured `onError`
> with phase `attach:initial`, so host callback throws cannot leak out of the
> React helper or turn attach/load failures into console-only diagnostics.
> Same-day WebGPU orchestration callback follow-up: walkaround adapter-profile
> telemetry plus the WebGPU canvas configure / swap-chain acquisition error
> helper callbacks are now guarded, matching the factory's error/warning
> callback policy and preventing host callback throws from breaking best-effort
> setup paths.
> Same-day glTF predictable-API follow-up: `<VitrumCanvas>` now forwards
> `advancedBackend`, `advancedByBackend`, `onWarning`, and `onAdapterProfile`
> through both the glTF engine bridge and `attachVitrum`; `loadGltfWithEngine()`
> honors explicit `runtimeProfile`, revalidates actual pt-webgpu fallback engines
> against the runtime full/lite profile before attachment, disposes rejected
> fallback engines, and strict compatibility failures are structured
> `GltfCompatibilityError`s while preserving existing messages.
> Same-day pt-webgpu ReSTIR-PT follow-up: the producer suffix/reconnection
> environment paths now mirror the megakernel by spectralizing environment
> radiance in hero-wavelength mode and applying the current surface's
> `envMapIntensity` to both direct environment NEE and BSDF-escape/synthetic
> reconnection env terms; lite-tier BDPT construction no longer emits the
> full-tier multi-vertex research warning when BDPT is disabled by tier limits.
> Same-day GI-state import follow-up: `importGIState()` grid-layout mismatch
> rejection now emits structured
> `walkaround-hybrid.import-gi-state-grid-mismatch` warnings with snapshot/current
> layout details before returning `false`, instead of being console-only.
> Same-day DDGI sun-dedup follow-up: when a host `opts.lights` sun is overridden
> by a scene `directional` emitter, DDGI light sync now emits structured
> `walkaround-hybrid.ddgi-host-sun-overridden` with the drop reason while still
> keeping exactly one scene-authored DDGI sun.
> Same-day walkaround atlas-diagnostic follow-up: ambiguous raw material texture
> payload strides now surface through `MaterialTextureAtlasDiagnostic` and
> `walkaround-hybrid.ambiguous-material-texture-stride` host warnings with
> glTF source provenance and the heuristic pixel-stride fallback, instead of
> writing directly to `console.warn`.
> Same-day PPG diagnostic follow-up: PPG snapshot-restore compatibility
> rejections (`maxSpatialCells`, `maxDTreeNodesPerCell`, scene bounds) now emit
> structured `walkaround-hybrid.ppg-import-*` warnings through the engine
> `onWarning` surface, and async PPG training-readback failures now report
> deduped non-fatal render errors instead of writing directly to `console.warn`.
> Same-day mesh-area diagnostic follow-up: mesh-area emitters whose `meshId`
> matches no scene primitive now report
> `walkaround-hybrid.mesh-area-emitter-missing-mesh` through BVH construction,
> emitter refresh, and DDGI sync warning sinks with source/fallback details,
> instead of console-only helper warnings.
> Same-day neural diagnostic follow-up: neural denoiser size-mismatch,
> inference-dispatch, and graph-resize fallback paths now emit structured
> `walkaround-hybrid.neural-*` warnings through the engine `onWarning` surface,
> preserving direct `console.warn` only for standalone/no-sink use.
> Same-day RC diagnostic follow-up: RC fixture/teaLight list truncation at the
> 16-light probe cap now emits structured
> `walkaround-hybrid.rc-light-cap-exceeded` warnings through engine-owned RC,
> preserving direct `console.warn` only for standalone helper use.
> Same-day shader-compiler diagnostic follow-up: walkaround pipeline WGSL
> compilation warnings, including optional PPG/ReGIR shader warnings, now emit
> structured `walkaround-hybrid.shader-compilation-warning` host warnings with
> shader label and source-location details, preserving console fallback only
> for standalone compiler use.
> Same-day pt-webgpu resource-warning follow-up: `GpuResources` buffer-ceiling
> downgrades (BDPT eye-stack, ReSTIR-PT reservoirs, SPPM photon cells, and SPPM
> pixel stats) now route through the engine's structured `onWarning` channel
> when engine-owned, while preserving direct `console.warn` fallback for
> standalone resource tests.
> Same-day mapped-emitter adjoint follow-up: explicit uncapped mesh-area emitter
> `color` / `intensity` path replay now stays analytic when the source material
> has a readable `emissiveMap` and the authored color/intensity denominators are
> nonzero; the adjoint shader chains through the packed per-triangle radiance so
> local emissive-map multipliers are included. 2026-06-19 follow-up:
> `meshAreaLightSourceFactors` now carries per-triangle readable emissive-map
> multipliers into the adjoint pass, so zero authored color channels stay on
> path replay. Current follow-up: mesh-area emitter inverse fits now build an
> adjoint-only replay stream that preserves zero-power explicit mesh triangles,
> so zero-intensity mapped emitter `intensity` targets stay on path replay
> without perturbing the forward renderer's light-selection stream.
> Same-day glTF report follow-up: high `TEXCOORD_N` remappability analysis now
> includes `KHR_materials_variants` primitive mappings, so variant-only materials
> no longer produce stale unsupported UV rows when the mapped primitive can
> losslessly remap the single high UV set into `uv1`.
> **2026-06-18 pt-webgl2 emitter shadow follow-up:** folded mesh-area emitter
> materials now carry a dedicated shadow-disabled flag into the GLSL material
> payload, and the ordinary forward emissive-hit MIS estimator skips those
> emitters while preserving camera/specular-visible emission. The pt-webgl2
> `emitterCastShadow` promise row is now native.
> **Implementation distance remaining:** full analytic adjoint replay beyond the
> current scoped single-bounce RGB direct-light/unlit-primary/environment-NEE slice; walkaround transparent
> ReSTIR/GI promotion plus validation of first-hit light-map/emissive
> approximations, finite-emitter/light-map promotion
> decisions, and rich-material GI GPU A/B evidence;
> real
> production neural checkpoints and NRC/neural quality/default-tier decisions;
> validation-backed promotion decisions for GRIS, BDPT/pt-webgpu fidelity rows,
> and rich-material GI. Explicitly unsupported rows such as displacement,
> volumetric/spectral scattering, thin-film-stack parity in walkaround GI, and
> front/back-layer specialty handling are now contract/truthfulness rows, not
> silent implementation promises.
> **GLTF-01 skinned-node double-transform CLOSED 2026-06-15:** glTF joint matrices are now converted into mesh-node-local skinning space and animation patches use the same convention.
> **Big validation tail: V28-B** — GPU A/B recapture for every render-changing landing
> (improvement confirmations, not regression suspects).
> **BDPT A/B status updated 2026-06-17:** the pt-webgpu Cornell radiometric fixture
> now keeps the box open toward the fixed +Z camera, eliminating the old black/black
> false-pass lane. Focused behavioral gates for `pt/bdpt`, `pt/spectral*`,
> `pt/restirPtReuse`, and `pt/caustic*` now render finite non-black signal on the
> full lavapipe adapter. A source bug was also fixed: secondary BDPT connections skip
> `lvi=0`, the emitter endpoint already covered by per-bounce NEE. `bdpt:true` now
> defaults to endpoint-only light-subpath depth (`maxLightBounces:1`), so the default
> A/B agrees exactly with `bdpt:false` and `results-bdpt.json` records
> `"verdict":"PASS"`. **2026-06-18 dzn follow-up:** `behavioral-gate:dzn -- --filter bdpt --require-full-tier`
> now has a committed status artifact proving `pt/bdpt` and `pt/spectral+bdpt`
> boot/render finite non-black on the dzn full-tier adapter with zero GPU errors.
> A second dzn status artifact for
> `behavioral-gate:dzn -- --filter restirPtReuse --require-full-tier` proves
> off-default `pt/restirPtReuse` also boots/renders finite non-black full-tier on
> the dzn adapter with zero GPU errors; equal-spp variance and specialty
> radiometric promotion remain separate proof work.
> **2026-06-18 caustic/SPPM dzn follow-up:** focused dzn status artifacts for
> `--filter caustic` and `--filter photon` prove `pt/caustic-manifold`,
> `pt/caustic-photon`, and `pt/spectral+photon` boot/render finite non-black
> full-tier with zero GPU errors. This closes a specialty-path execution proof
> slice; caustic radiometric convergence / forward-traced oracle A/B remains the
> A4 proof tail.
> **2026-06-18 emitter/environment dzn follow-up:** focused dzn status artifacts
> for `--filter light`, `--filter directional`, `--filter hdri`, and
> `--filter procedural-sky` prove pt-webgpu point/disc/spot/directional,
> HDRI/procedural-sky, lite point/HDRI, and walkaround directional-sun/HDRI lanes
> boot/render finite non-black with zero GPU errors. This is execution proof, not
> a replacement for reference-quality radiometric sweeps.
> **2026-06-18 walkaround dzn follow-up:** the broad
> `behavioral-gate:dzn -- --filter wh/ --require-full-tier` lane now passes all
> ten walkaround behavioral rows with zero GPU errors. That run exposed and
> closed an RC validation bug: `rc_materialTextureAtlas` / `rc_materialMapMeta`
> are `rgba32float` textures read with `textureLoad`, so their bind-group layout
> entries must be `unfilterable-float`, not filterable `float`.
> **2026-06-18 baseline/lite/spectral/skinned/analytic dzn follow-up:** focused
> dzn status artifacts for `--filter default`, `--filter lite-tier`,
> `--filter spectral`, `--filter skinned`, and `--filter analytic` now prove
> baseline pt/walkaround rows, explicit pt-webgpu lite fallback, pt spectral
> combinations, pt/walkaround skinned rows, glTF skinned animation, and full-tier
> pt-webgpu analytic sphere execution all boot/render finite non-black with zero
> GPU errors. The committed dzn status checker now derives the behavioral-gate
> label inventory from `gate.mjs` and fails if any real gate label lacks status
> coverage; only the synthetic `__self-test/always-black` row is excluded. This
> also closed a validation-harness truthfulness issue: the analytic fixture now
> uses the core `AnalyticPrimitive` contract (`shape:'sphere'`,
> `params:[cx,cy,cz,radius]`), and pt-webgpu suppresses the shared-bvh
> triangle-stream "skipped" warning for analytic primitives that it actually
> consumes through full-tier analytic buffers.
> The remaining proof/implementation tail is explicit
> multi-vertex BDPT: `controls.byMaxLightBounces` still shows the finding starting
> at `maxLightBounces:2` (+13.21% global luminance) and reaching +17.08% at
> `maxLightBounces:3`. **2026-06-19 API guard:** requesting that mode now requires
> `bdptOptions.experimentalMultiVertex:true`; otherwise pt-webgpu/full-tier and
> pt-webgl2 construction reject the request instead of silently enabling the known
> research path. When explicitly opted in, both backends emit structured
> `*.bdpt-multivertex-research-mode` warnings; pt-webgl2 still caps to its current
> 3-column light-path texture.

- **Foundations + default render paths: solid, advancing toward 100%.** The `@vitrum/core`
  contract, each backend's default integrator, shared-bvh/samplers/denoisers are real,
  correct, type-clean (typecheck green across 12 packages), and test-backed (~3,300+
  assertions). CI rewritten; in-repo shader compile gate (48 WGSL shaders, naga-validated).
  The P0 default-path correctness issues from prior audits are resolved.
- **The real distance to "fully implemented + professional" is four buckets:**
  **A** frontier features that are implemented but still need fidelity promotion/default-tier
  decisions, **B** deliberate fidelity ceilings or approximation semantics in realtime paths,
  **C** provisioning for optional learned systems, **D** hygiene/ledger consistency. That's
  what this doc lists.

**Legend:** `✅` = verified by lead code-read/grep this session · `◻` = reported by a
deep-reader agent with file:line, not independently re-verified (verify before acting).
**Effort:** S = hours · M = 1–3 days · L = ~1 week · XL = multi-week / research-grade.

---

## Bucket A — Frontier feature completion (largest gap to "fully implemented")

These are the features the README/feature-list advertises that a *default* engine does not
actually deliver. Per the project north-star (fidelity-paramount, implement-don't-remove),
the target is to make each one real and consumed — not to demote it.

| ID | Item | Where | State now | Done = | Effort |
|----|------|-------|-----------|--------|--------|
| **A1** | ✅ **DONE (Campaign 2 Wave A; hybrid-shift wiring 2026-06-16) — V28 radiometric pending** | `pt-webgpu/src/index.ts:844-918,1026-1036`; `wgsl/pathTrace/restirPt{Spatial,Compose,Resolve,Temporal,Producer}` | **Implemented:** producer→temporal→**spatial (5-neighbour GRIS, no /p_src)**→resolve, then a COMPOSITE megakernel folds the reconnection-indirect into the BEAUTY accumulator via an E0-direct / indirect estimator split (producer-dropped specular pixels fall through to the full path). The formerly harness-only hybrid shift is now live for prefix-1 reuse: reservoirs store a visible-domain replay PDF, temporal/spatial reuse use half-G × source/target BSDF replay-pdf ratio via `restirPtHybridShiftJacobianForPair`, and selected samples refresh the cache for downstream reuse. OFF-path byte-identical. `restirPtReuse:true`+full-tier only. | Equal-spp variance-reduction A/B vs megakernel on real GPU (V28-B), including glossy/moderately-specular reuse scenes. | done (impl) |
| **A2** | ✅ **DONE (Wave A + v1-closure Wave 4 + 2026-06-17 public mix control) — sTree splits + runaway fixed; atomics saturate** | `walkaround-hybrid/src/ppg/sTree.ts:179`, `pipeline/PPGCoordinator.ts:157-455`, `wgsl/ppgUpdate.wgsl.ts:241` | **Implemented:** per-cell atomic sample counters → GPU readback → `splitOverflowLeaves`; children seeded via `cloneDTree`; PPG_FLUX_DECAY=0.5 Müller per-window decay. **Wave 4 fix (06910e2):** flux atomics now saturate instead of u32-wrapping (clamp before atomic add); stale "single global cell" JSDoc deleted (code-verified: grep returns 0 hits). **2026-06-17:** `HybridEngineOptions.ppgMixAlpha` now threads through derived config and `PPGCoordinator` into the GI UBO, clamped to `[0,1]`, so the guide/cosine MIS blend is no longer a hidden fixed 0.5. | Multi-region guiding-localization A/B on real GPU (V28-B). | done (impl) |
| **A3** | ✅ **DONE (Wave B + 2026-06-15 pt-webgl2 parity) — true spectral reflectance** | `wgsl/pathTrace/shadePrologue.wgsl.ts:127-147`, `material.wgsl.ts:732-769`; `pt-webgl2/src/scene/materialsTexture.ts`, `glsl/render/get_surface_record_function.glsl.js` | **Implemented:** in spectral mode the RGB albedo is replaced by a SCALAR spectral reflectance S(λ) at the hero λ via Jakob-Hanika per-material coeffs (solved at pack time), broadcast to all channels so throughput·brdf·NEE·MIS carries a genuine single-wavelength quantity. Material stride 26→27 (stale adjoint-stride latent bug fixed). Flat-spectrum invariant harness pins RGB-mode byte-identity. **2026-06-15 pt-webgl2 parity:** the WebGL2 material texture now packs a per-material Jakob-Hanika reflectance texel and `getSurfaceRecord()` evaluates it at `state.wavelength` before BSDF/NEE use. **Documented approximation:** emitter/env chroma is reconstructed as a D65-relative tristimulus SPD (not authored spectra); materials lacking packed coeffs fall back to RGB-luminance. | Dispersive-scene A/B vs RGB (V28-B). | done (impl) |
| **A4** | ✅ **DONE (A4-progressive: 2026-06-10; radius/capacity follow-ups 2026-06-16) — true Hachisuka progressive SPPM** | `pt-webgpu/src/wgsl/pathTrace/sppmBindings.wgsl.ts`, `caustic.wgsl.ts`, `kernel.wgsl.ts`, `gpuResources.ts`, `index.ts` | **Implemented (A4-progressive):** per-pixel `SppmPixelStats` buffer (tau.rgb, radius2, N, _pad×3 = 32 bytes/pixel, group-3 binding 9, full-tier only, 64-byte placeholder when off). Hachisuka & Jensen 2009 §4 / Knaus-Zwicker update rule runs in `sppmGatherProgressive`: N′=N+α·M; ratio=N′/(N+M) [guarded M=0]; R′²=R²·ratio; τ′=(τ+Φ_M)·ratio; L=τ′/(Ne·π·R′²) where Ne=frameAccumulated×photonCount. Buffer GPU-cleared on `reset()`/`setScene()`/camera move (static-eye-point invariant). **2026-06-16 follow-ups:** gather now queries the same stable r₀ hash grid used by `sppmInsertPhoton` while retaining the shrunk per-pixel `dist2 <= R²` physical disk, closing the post-shrink missed-cell tail; over-capacity hash cells now keep a bounded reservoir instead of newest-photon modulo overwrite, and gather multiplies stored hits by `totalInserted / storedCount` so dense caustic cells are not silently underweighted. Accumulator interaction is a Cesàro mean (L_caustic(k)→L_true ⟹ running mean converges). TS-mirror recurrence/radius-coherence/capacity tests pin N(k)=k·α·M closed form, R² monotone-shrink, M=0 stability, first-frame seeding, stable-grid lookup after radius shrink, and overflow compensation. Supersedes the R7a streaming-window form; off-path byte-identical. | Radiometric A/B vs forward-traced oracle (V28-B) — caustic photon-map convergence test. | done |
| **A5** | ✅ **DONE for safe-default BDPT (v1-closure Wave 3, 1d31f0b; 2026-06-17 endpoint fix/proof; 2026-06-19 research-flag guard) — multi-vertex remains research-mode** | `pt-webgl2/src/glsl/composeTraceGlsl.ts:275-383`, `pt-webgpu/src/wgsl/bdpt/bdptLightSubpath.wgsl.ts:351-431`, `pt-webgpu/src/wgsl/pathTrace/kernel.wgsl.ts:900-916`, `tools/radiometric-ab/{README.md,results-bdpt.json,proofs.mjs}` | **Implemented/proven:** (A) pt-webgl2 light-subpath RNG re-seeds row-independently (`vec2(gl_FragCoord.x, 0)`) so the three rows of one vertex trace the SAME path (were three independent paths = garbage vertices); confined to FEATURE_BDPT subpath branch — off-path byte-identical. (B) pt-webgpu BDPT estimator coherence: light-subpath extension now samples ONE real-BSDF direction at the previous vertex used for BOTH the trace and the stored throughput/pdfFwd (was a sampled-then-discarded direction); pdfRev patched per PBRT §16.3 reciprocal convention. (C) 2026-06-17: the Cornell radiometric fixture now places the back wall at `z=-1` for the fixed +Z camera, so local-light proof lanes are finite instead of black/false-pass; the BDPT connection loop starts at `lvi=1` because the emitter endpoint is already covered by per-bounce NEE. `results-bdpt.json` records default `bdpt:true` safe mode as endpoint-only (`maxLightBounces:1`) and exactly matching UNI at the 60-frame checkpoint (global/ROI relErr 0, variance ratio 1.0). The previous +13%/+17% mismatch starts only when a host explicitly opts into `maxLightBounces>=2`, and `proofs.mjs` pins that as `multiVertexFindingStartsAt:2`. **2026-06-19:** normal constructors now require `bdptOptions.experimentalMultiVertex:true` before accepting `maxLightBounces>1`; the A/B harness opts in explicitly so the research finding remains reproducible without leaking into ordinary API use. | Full all-strategy/multi-vertex BDPT radiometric closure remains a research-mode promotion tail; default BDPT proof is closed and checked by `npm run radiometric-ab:proof-check`. | done (safe default) / research tail |
> **2026-06-18/19 pt-webgl2 safe-default alignment:** pt-webgl2 `bdpt:true` now
> defaults to endpoint-only `bdptOptions.maxLightBounces: 1`, and the eye
> connection loop starts at light vertex 1 so the stored emitter endpoint is not
> double-counted against per-bounce NEE. Explicit
> `bdptOptions.maxLightBounces > 1` now requires
> `bdptOptions.experimentalMultiVertex:true`; opted-in requests are capped to the
> backend's current 3-column light-path texture and emit structured
> clamp/round/research-mode warnings.

| **A6** | ✅ **NRC: from opt-in/biased to a validated consumable** | `walkaround-hybrid/src/neural/nrc/*` | Online training (forward+backprop+Adam) is genuinely live on GPU and inference is consumed. H26/H27 structural defects are closed: camera-pdf footprint, spread seeding, post-loop `r.Lo` training target, zero-radiance-vs-empty-slot semantics, and atomic slot claims are all wired and tested. **2026-06-15 follow-up:** CPU record unpacking now treats a slot as empty only when the whole encoded-input prefix is zero, so valid records with `input[0] === 0` survive dense repack. **2026-06-16 follow-up:** opt-in NRC now has a UBO-level warm-up gate (`trainedSteps`/`warmupSteps`): spread-fired candidates still gather training records immediately, but visible reservoir substitution keeps DDGI until enough completed trainer windows exist. **2026-06-17 truthfulness/control follow-up:** when enabled, NRC is now advertised in `capabilities.experimentalFeatures` as `walkaround-hybrid-nrc-biased-cache` in addition to the existing structured biased/experimental warning, and `HybridEngineOptions.nrcWarmupSteps` threads the warm-up gate from public options through pipeline initialization into `NrcSubsystem` with integer `>=0` clamping. **2026-06-17 capability-gate reconciliation:** `assertNrcDeviceCapable()` fails fast when an opt-in NRC device lacks `maxBindGroups >= 5` or `maxComputeWorkgroupStorageSize >= 24576`, so the old hardware-note follow-up is code-closed. Still OFF by default and acknowledged biased. | Converge/quality A/B vs reference; decide default-on tier; keep the documented bias bound honest. | L |
| **A7** | ✅ **DONE (v1-closure Wave 5, caab499) — RC finished (user decision: keep+finish)** | `walkaround-rc/src/`, `shade.wgsl.ts`, `HybridEngineRC.ts` | **Implemented:** RC receiver replaced with correct MC irradiance estimator `E=(4π/N)·ΣL·cos` (was `Le/Wsum·N·0.5` — ray-count-dependent; N=16/N=64 now agree ≈π in tests); real env map bound into the last cascade (was permanently 1×1 black); point/spot lights added to the RC light model (binding 15, DDGI conventions, fingerprint-gated upload); chromatic sun from scene's directional emitter (was achromatic); scene-scale shadow bias. | Real-GPU cascade A/B at N=16/N=64 (V28-B). | done (impl) |
| **A8** | ✅ **DECIDED (2026-06-10) — biased default retained for realtime; unbiased GRIS documented as first-class opt-in** | `HybridEngineOptions.restirPtReuse`, `temporalGi.wgsl.ts`, `spatialGi.wgsl.ts`, `jacobianShift.wgsl.ts`, `restirPHat.wgsl.ts`, README bias docs, `tools/radiometric-ab/walkaround-ab-results.json` | **Architecture decision:** The default (`restirPtReuse: false`) retains the pre-GRIS Sprint-17 clamped-Jacobian reuse for the realtime frame budget (the unbiased path adds one visibility ray + full-GBH O(K²) MIS cross-evaluation per accepted neighbour — the dominant cost in the GI reuse passes). Remaining documented default-OFF bias sources are B1 Jacobian clamp [0.1,10] (`jacobianShift.wgsl.ts`), B2 no reconnection-visibility ray (OFF variants of `spatialGi`/`temporalGi`), and B3 no full GBH MIS (OFF combine weights). The old B4 centroid-p̂ note is stale: `restirPHat.wgsl.ts` now evaluates `restir_di_compute_phat_xi(lid, xi, surf)`, RIS finalization uses stored `r.xi`, temporal/spatial reuse call the xi-aware helper, and `lo_direct` shades the selected xi. The unbiased GRIS path (`restirPtReuse: true`) is first-class, compile-time gated, fully functional (Phase-1 shift + Phase-2 full-GBH spatial, pairwise-MIS temporal), and the JSDoc specifies exactly when to enable it. H24 follow-up: the default path no longer pays the widened GRIS reservoir memory cost — `restirPtReuse:false` compiles/allocates the compact 20-u32 ReSTIR-GI layout, while the 30-u32 GRIS cache is opt-in only and PPG bakes the matching stride. A compile-time variant-selection pin test added (`__tests__/grisVariantPin.test.ts`), now complemented by `giStructuralGate.test.ts` shader-source stride checks. **2026-06-18 V19 partial proof:** `npm run radiometric-ab:walkaround` now completes on the native WSL host and records `PASS-PARTIAL`; its A8 case bounds the convex-Cornell biased-vs-GRIS delta as `NEGLIGIBLE` (`overall = -0.000020`, about 0.03% of mean, with per-region deltas proof-checked), while the SUN analytic case remains partial and preserves the do-not-promote warning. The default is NOT being flipped now; this gathers real data for F6 without pretending the broader validation matrix is complete. | Remaining V19 work: higher-quality/browser/real-adapter converged-unbiasedness validation and biased-default error quantification on harder scenes with occlusion/emitter/M-count gradients. The current committed A8 snapshot is evidence for one convex scene, not a default-policy promotion gate. | done (decision); partial A/B evidence / validation ACTIVE |
| **A9** | ✅ **DONE for default/off-default safety (Wave A + v1-closure Wave 3 + 2026-06-15 parity/proof waves + 2026-06-17 A/B finding + 2026-06-19 research-flag guard) — serial build retained; multi-vertex research open** | `wgsl/bdpt/bdptLightSubpath.wgsl.ts:140-255,575-728`, `wgsl/bdpt/bdptConnection.wgsl.ts:369-443`, `wgsl/pathTrace/kernel.wgsl.ts:900-916`, `pt-webgpu/src/__tests__/bdptGlossyLightSubpath.test.ts`, `pt-webgpu/src/__tests__/bdptPlumbing.test.ts`, `tools/radiometric-ab/results-bdpt.json` | **Implemented:** light subpath carries a REAL glossy/specular BSDF (VNDF) at each vertex, bounce cap 3->8, and a 5-row light-vertex record whose row 4 stores hit-local material payload plus a front-face side bit. **Wave 3 fix (1d31f0b):** estimator coherence — extension now samples ONE real-BSDF direction used for BOTH the trace and stored throughput/pdfFwd (was a sampled-then-discarded direction — biased MIS densities); pdfRev patched per PBRT §16.3. **2026-06-15 parity wave:** BDPT light-side material decode now mirrors the shade prologue for mapped base/vertexColor/AO/ORM/transmission/normal/bump/clearcoat-normal/extension/specular/anisotropy plus layer tint/roughness, thin-film reflect tint, Cauchy IOR, and spectral reflectance scalar. **2026-06-15 proof wave:** A9 now has independent numeric TS oracles for row-4 tri/front-face packing, mapped payload transforms, front/back layer selection, thin-film tint mixing, Cauchy IOR, and Jakob-Hanika spectral override; this removes the pure-substring proof residue for that lane. **2026-06-17 proof wave:** the Cornell proof fixture was repaired, secondary emitter-endpoint connection double-count was removed/pinned, and the committed safe-default A/B shows `bdpt:true` endpoint-only mode matches UNI exactly. **2026-06-19 guard:** `maxLightBounces>1` is now an explicit `experimentalMultiVertex:true` opt-in on pt-webgpu/full-tier and pt-webgl2. **Honest remaining:** serial dispatch (one workgroup; the per-column variant was a spec-undefined cross-workgroup race — documented), multi-vertex research-mode radiometry for explicitly opted-in `maxLightBounces>=2`, and independent material-furnace/radiometric A/B before any full-tier/default promotion. | Keep multi-vertex BDPT behind the research warning and explicit flag until the mean mismatch is fixed/proven; default/off-default safe mode is no longer an implementation blocker. | done (safe default) / research tail |
| **A10** | ✅ **Pipeline E2E (Wave A) — package weights not production-default** | `tools/neural-denoiser-training/{train.py,capture-dataset.mjs,export_weights.py}` | **Implemented:** capture→train→export→load CLOSES — `train.py --dry-run/--smoke` exports a valid 535,107-param `.vitrum-model` binary (CANONICAL_PARAM_COUNT pinned to the engine loader; the "vi-neural-weights.json" reference was stale — binary is the real format), round-trip test green; `capture-dataset.mjs` CPU smoke. **2026-06-15 posture fix:** repo research/starter checkpoints are shape/value-validated before allocation, and `neuralWeightsRoundTrip.test.ts` now loads the actual tracked `starter-v1.vitrum-model` and `v2-random.vitrum-model` files through the runtime loader/spec validator. **2026-06-17 truthfulness follow-up:** selecting `denoiser:'neural'` with host weights emits `walkaround-hybrid.neural-host-weights-required`, `capabilities.experimentalFeatures` includes `walkaround-hybrid-neural-denoiser-host-weights`, and `learned-systems-proof-check` pins both claims. The package still requires host-supplied weights and does not advertise a production default checkpoint. `neural` remains opt-in/experimental until a validated production checkpoint and quality A/B exist. | Vendor/ship or otherwise bless a production checkpoint; quality A/B. | pipeline done / weights remaining |

---

## Bucket B — Fidelity ceilings in the default path

Deliberate approximations a discerning user hits immediately. Closing these is what makes
the *default* render "hero-fidelity" rather than "good real-time."

| ID | Item | Where | State now | Done = | Effort |
|----|------|-------|-----------|--------|--------|
| **B1** | ✅ **MOSTLY DONE (Wave A + R8-B tail + B1-ior-per-tri + GI material-payload + receiver-lobe target wave)** | `shaders/risGi.wgsl.ts`, `shaders/risGiNrc.wgsl.ts`, `shaders/restirGiMaterial.wgsl.ts`, `materialDecode.wgsl.ts`, `shade.wgsl.ts`, `restir/packingHelpers.ts` | **Wave A:** per-tri `bvh_material` r32uint texture; metals get DI + analytic NEE + specular indirect. **R8-B tail (2026-06-10):** glass primaries get a refracted GI reservoir via 1-interface Snell walk. **B1-ior-per-tri (2026-06-10):** `bvh_material` bits[15:8] now carry IOR quantized [1.0, 3.0] → step ≈ 0.0078; risGi glass walk uses `decodeIor()` per-tri (no more fixed 1.5 constant); shade `lo_transmittedGI` derives Schlick F0 from per-tri IOR via `((ior−1)/(ior+1))²`; rough-glass GI: for roughness > 0.1 the Snell refracted direction is perturbed by a GGX-distributed offset (one sample), giving frosted glass blurred GI. Default IOR=1.5 glass: byte 64, decodes to 1.502, F0=0.04004 (error < 0.003). Mutation path `packBVHRoughMetalFromCore` / `repackBVHMaterialRange` updated. **2026-06-15 GI material-payload wave:** default/NRC GI-RIS suffix vertices now apply smooth/normal/bump material normals, mapped base/vertex/roughness/metallic/specular/clearcoat/sheen/anisotropy/iridescence payloads, and extension-aware GGX/clearcoat/sheen suffix `Lo`; NRC query/training records now use the same mapped albedo/roughness payload. **2026-06-16 receiver-lobe target wave:** GI RIS/NRC producers and default+GRIS temporal/spatial reuse now evaluate candidate/final `pHat` through `restir_gi_receiver_phat_from_payload()` / `restir_gi_receiver_phat_from_surface_or_geometry()`. Diffuse receivers reduce to the old luminance/cosine target; rich receivers add specular/clearcoat/sheen receiver BRDF terms without widening the compact reservoir. **2026-06-17 NRC glass parity:** NRC GI-RIS now mirrors the default bounded refracted-GI glass walk and stores the post-glass reservoir instead of immediately writing `emptyReservoirGI()`; cache substitution/training remains bypassed for glass primaries. Temporal previous-domain material recast is best-effort and falls back to the geometric target when the current UBO cannot reconstruct the previous camera ray exactly. Structural pins: `roughMetalPacking.test.ts`, `b1GlossyMetalGi.test.ts`, `restirGiMaterialParity.test.ts`, `grisReuseUbo.test.ts`, `giStructuralGate.test.ts`, `dispatchEquivalence.test.ts`, and `wgslCompose.test.ts`. | GPU A/B for rich-material/glass GI, temporal previous-domain fallback quantification if needed, and V28-B/browser recaptures. | impl mostly done / validation tail |
| **B2** | ✅ **DONE/approx (R8-B, 2026-06-10; atlas/lobe probe-material follow-ups 2026-06-18) — DDGI glossy-aware probe bounce; specular complement** | `ddgi/wgsl/probeUpdateRays.wgsl.ts`, `ddgiGlossyProbeBounce.test.ts`, `walkaround-rc/src/wgsl/probeRayCast.wgsl.ts`, `rcLightEvalWgsl.test.ts` | **Implemented:** specular complement via reflected previous-frame SH atlas lookup. The original metal/roughness scalar path blends Lambertian indirect toward specular indirect (reflected-direction atlas sample, metal/base-F0 tint). Blend-not-add — energy-conserving lerp; no double-counting. Gated on `indirectFeedback != 0u` (direct-only probes stay Lambertian). **2026-06-18 atlas follow-ups:** DDGI ordinary probe hits now sample readable `baseColorMap`, `roughnessMap`, `metallicMap`, `normalMap`, `bumpMap`, specular color/intensity maps, clearcoat factor/roughness/normal maps, sheen color/roughness maps, anisotropy controls/maps, and iridescence factor/thickness maps; clearcoat-normal gets its own reflected SH query, and extension lobes feed bounded lobe weights/tints. RC probe-cast direct sun/emitter/point/spot terms likewise consume roughness/metallic/specular/normal/bump plus clearcoat, clearcoat-normal, sheen, anisotropy, and iridescence atlas payloads in a compact direct BRDF response. Scalar `MaterialEntry` fields remain the fallback when no readable map/UV is present. Approximation documented in-code: DDGI stores cosine-weighted irradiance, not GGX-filtered radiance; RC still evaluates a compact direct response, not full stochastic transport. **Cite:** Karis (2013) UE4 §4.4; McGuire et al. (2017) probe specular. | V28-B A/B on metallic/textured/extension-lobe probe scenes and material-furnace reference sweeps. Remaining work is validation/promotion of the approximation, not unconsumed probe-material fields. | done (impl) / lobe-validation tail |
| **B3** | ✅ **DONE (Wave B + v1-closure Wave 4/5, caab499 + follow-up verified 2026-06-14) — env pillar COMPLETE; hdri → native** | `walkaround-hybrid/src/shaders/ris.wgsl.ts:354-376`, `shaders/risGiNrc.wgsl.ts:300-388`, `ddgi/wgsl/probeUpdateRays.wgsl.ts`, `WalkaroundGPUPipeline.ts:1425`, `HybridEngine.ts:updateEnvironment`, `HybridEngineFrameOrchestrator.ts` | **Implemented:** (Wave B) equirect CDFs built at scene-load (bindings 15-19), directional samples + scalar-tint fallback. (Wave 4/5) `envImportanceSample` is now a live DI NEE candidate in the RIS loop (M_ENV=1 sentinel, measure-consistent source pdf, phat_xi spatial reuse); `risGiNrc` GI-escape reads `envRadiance` (NRC no longer downgrades IBL); DDGI probe misses sample the real HDRI (group-2 bindings 6/7, rotationY identical convention, procedural fallback intact); `updateEnvironment` rebuilds directional CDFs at runtime. Follow-up verification: `updateEnvironment()` is an env-only fast path that updates sky scalars, invalidates DDGI, resets accumulation, and calls `_applyDirectionalEnvironment()` without geometry/BVH rebuild; RC now receives `pipeline.getEnvBindings()` each frame. Walkaround `hdri` ledger grade promoted to `'native'` — code-verified `promiseLedger.ts:254`. | GPU A/B evidence remains in validation matrix; implementation is closed. | done (impl) |
| **B4** | ✅ **DONE (Wave A + 2026-06-16 emissive-material parity + 2026-06-18 power-PDF parity) — pt-webgl2 mesh-area NEE** | `scene/meshAreaLights.ts`, `glsl/composeTraceGlsl.ts`, `glsl/shader/sampling/light_sampling_functions.glsl.js`, `scene/foldEmissiveEmitters.ts`, `scene/mutateSceneTextures.ts` | **Implemented:** a dedicated `uMeshLights` triangle-light texture (6 texels/tri) is NEE-sampled by emitted-power mass after the 2026-06-18 power-PDF follow-up, with triangle-independent density recovered from `luminance(radiance)·area` and forward-hit MIS from `uTotalEmissivePower`. Explicit `mesh-area` emitters keep the emissive-fold as the BSDF strategy (exactly-one-MIS-estimate algebra documented). 2026-06-16: ordinary emissive mesh materials now synthesize implicit triangle lights too, with CPU-readable `emissiveMap` UV-local quadrature folded into per-triangle radiance and scalar material mutations repacking the mesh-light texture; same-day follow-up splits readable mapped implicit and explicit mesh-area triangle lights into bounded barycentric micro-triangles on both PT backends, preserving total emissive area while localizing textured emission. Later same-day follow-up replaces that fixed subdivision path for CPU-readable simple wrap/clamp maps with shared exact transformed texel-cell clipping, so pt-webgpu, pt-webgl2, and walkaround pack constant-radiance texel-footprint sub-triangles before falling back to bounded quadrature for unsupported map topology. Same-day finite-light follow-up: rect/disc analytic area lights are camera-visible path terminals and ordinary BSDF hits remain MIS-weighted against analytic-light NEE. 2026-06-18 pt-webgl2 power-PDF follow-up: packed mesh lights now carry `selectionPower = luminance(radiance)·area`; GL uploads `uTotalEmissivePower`; `sampleMeshAreaLight()` selects mapped texel-cell sub-triangles by emitted power; and `meshAreaLightForwardPdf()` reconstructs the matching area density from `surf.emission`, with `meshAreaMis.test.ts` pinning forward/sample PDF parity. Same-day pt-webgpu cap follow-up: pt-webgpu's mesh-area NEE cap now ranks capped texel sub-triangles by emitted power instead of geometric area. 2026-06-19 decode-reconciliation follow-up: JSON-only glTF analysis stays conservative, but `loadGltfAndDecodeTextures()` now clears the `emissiveMap.texelPdf` degradation for `pt-webgl2` and full `pt-webgpu` only once every emissive map is decoded to CPU-linear pixel/data texture handles; sRGB `textureTarget:"webgpu"` payloads stay degraded for exact CPU-built emitted-power PDFs. Same-day adapter-report follow-up: plain `loadGltfAsset()` now includes inactive material variant textures in `textureDecodeReport`, and `loadGltfForEngine()` rechecks strict compatibility against an engine-exposed `backendProfileId`/`profileId` before attaching. Walkaround still reports the GI/RC/DDGI texel-PDF approximation. The analytic `lightsTexture` still excludes `mesh-area` by design — NEE comes from the separate mesh-light texture. | Variance A/B on Cornell and glTF emissive panels (V28-B); cross-backend energy-alias promotion for walkaround and render-proof promotion remain tails. | done (pt-webgl2 impl + pt-webgpu cap) / validation tail |
| **B5** | ✅ **DONE (Wave A) — Beer-Lambert DDGI probes** | `ddgi/wgsl/probeUpdateRays.wgsl.ts:276-290` | **Implemented:** real `transmission · exp(−attenuationColor · t/attenuationDistance)` over path length, with `t` thickness-clamped (`clamp(distToExit,0,thickness)`); reduces to Beer-Lambert exactly. | — | done |
| **B6** | ✅ **DONE (Wave B) — GTAO per-pixel view axis** | `shaders/gtao.wgsl.ts:120-188` | **Implemented:** per-pixel view axis reconstructed from the inverse perspective projection (was the constant `(0,0,-1)` central-pixel approximation); correct at wide FOV / frame edges. | — | done |
| **B7** | ✅ **DONE (Wave B) — planar-SAH half-area fix** | `shared-bvh/src/buildArrayBvh.ts:127-147` | **Implemented:** `surfaceArea` now returns a nonzero half-perimeter term for planar boxes (one extent 0) so flat geometry ranks splits — a 2000-tri coplanar floor builds depth 45→9. **Remaining (out of scope):** no SBVH; recursive builder retained. | Optional SBVH; iterative build. | done (planar) / SBVH remaining |
| **B8** | ✅ **DONE (Wave B) — light-tree orientation cones** | `shared-samplers/src/lightTree.ts:48-74,387` | **Implemented:** Conty-Estévez orientation cone (axis + thetaO + thetaE) per node, stride 12→16; spot/area producers wired; full-sphere sentinel keeps the cone term ≡1 (byte-identical when unoriented). | A/B on directional-emitter scenes (V28-B). | done (impl) |
| **B9** | ✅ **DONE (Wave B) — GGX multiscatter (all 3 backends)** | pt-webgpu `material.wgsl.ts`; pt-webgl2 `glsl/render/get_surface_record_function.glsl.js`; walkaround `ggxBrdf.wgsl.ts` | **Implemented:** Kulla-Conty multiscatter energy compensation in all three GGX evals (LUT + furnace test on pt-webgpu; furnace-pinned on pt-webgl2, lite-mode skipped). | — | done |
| **B10** | ✅ **DONE (Wave B) — physical refraction transmittance** | `wgsl/pathTrace/bsdf.wgsl.ts` | **Implemented:** physical Fresnel-consistent transmittance replaces the phenomenological `mix(vec3(1),baseColor,0.15)` tint. | — | done |
| **B11** | ✅ **DONE (source-verified 2026-06-13) — pt-webgl2 + pt-webgpu disc-area NATIVE** | `pt-webgl2/src/scene/lightsTexture.ts`, `pt-webgpu/src/scene/emitterPacking.ts:54-67,149-224`, `promiseLedger.ts:220-229`, `scenePack.emitters.test.ts:141-191` | **pt-webgl2** packs `disc-area` emitters as `CIRC_AREA_LIGHT = 1` with concentric-disc sampling and `intersectsCircle` — geometrically exact. **pt-webgpu** now packs `disc-area` emitters natively into the rect-area stream with `shape = 1.0`, radius-scaled tangent/bitangent axes, concentric-disc sampling, and π·r² light-tree power; the old 32-triangle fan is gone. The pt-webgpu ledger grade is `'native'`. `procedural-sky` now grades `'approximate'` on both PT backends via the shared Preetham equirect bake; lite-tier pt-webgpu still explicitly gates disc/mesh area support as a profile limit (B12), not a full-backend disc-area gap. | GPU A/B evidence stays in the validation matrix; implementation is complete. | done |
| **B12** | ✅ **DONE (source-verified 2026-06-12; disc policy reconciled 2026-06-17; lite multi-directional closed 2026-06-18) — lite-tier texture packing shipped** | `webgpuLimits.ts:45-73`, `wgsl/pathTrace/material.wgsl.ts:112-142`, `wgsl/pathTrace/kernelLite.wgsl.ts:143-304`, `scene/litePackedTextures.ts`, `gpuResources.ts:711-728,1686-1755`, `liteTierCapabilities.test.ts`, `webgpuLimits.test.ts` | **Implemented:** the lite tier packs HDRI radiance/pdf + CDF into sampled textures (`liteEnvTex`, `liteEnvCdfTex`) and directional/point/spot/rect-area/disc-area light data into `liteLightTex`, avoiding the storage-buffer cliff while staying inside the baseline sampled-texture budget. Capabilities/tests now advertise directional/point/spot/rect-area/disc-area + HDRI support on lite; explicit mesh-area emitters remain unsupported. 2026-06-17 behavioral-gate follow-up replaced the inward-facing Cornell pt smoke fixture with a front-facing quad, restoring adapter-backed `pt/default`, `pt/point-light`, `pt/lite+point-light`, `pt/spot-light`, `pt/directional-2`, and `pt/disc-light` to finite non-black renders on the WSL lite adapter. 2026-06-18 predictability follow-up first added exact downgrade warnings; the same-day implementation follow-up removed that downgrade by prepending the full-tier 2-vec4 directional records to `liteLightTex`, looping `params.directionalLightCount` in `kernelLite`, and shifting rect/disc connection offsets accordingly. | GPU A/B promotion evidence still belongs to the validation matrix, not this binding-budget gap. | done |

---

## Bucket C — Provisioning (turnkey usability)

The code is done; what's missing are shipped assets / managed deps so a consumer gets a
working feature out of the box.

| ID | Item | Where | State now | Done = | Effort |
|----|------|-------|-----------|--------|--------|
| **C1** | ✅ DONE (capabilities/error approach) | `pt-webgpu/index.ts`, `pt-webgl2/index.ts`, `oidnBridge.ts` | Real ONNX inference; needs a host-supplied `.onnx` model **and** the `onnxruntime-web` peer dep. | Both backend factories now throw a clear **two-asset** error naming the model URL AND the `onnxruntime-web` peer dep up front (was modelUrl-only + a late first-frame runtime throw); the `oidn` option JSDoc states it is NOT turnkey + lists both assets; the bridge's missing-runtime error already says `npm install onnxruntime-web`. No binary vendored (per "set aside distribution"). | done |
| **C2** | — **Neural denoiser checkpoint** | (see A10) | Overlaps A10. | — | XL |
| **C3** | ✅ DONE — **pt-webgl2 OIDN final-pass aux readback** | `packages/pt-webgl2/src/denoise/*`, `packages/pt-webgl2/src/gl/glResources.ts`, `packages/pt-webgl2/src/index.ts` | The retired fork/MRT blocker is stale. Native pt-webgl2 now reads the linear HDR accumulator plus full-tier albedo + normal MRT aux into OIDN RGB tensors, runs the shared async OIDN dispatcher once converged, reports `FrameStats.denoiserState`, exposes `getLatestDenoised()`, and invalidates on reset/dispose. Lite tier still supplies color-only because aux MRTs are unavailable by tier. | Targeted tests: `oidnFinal.test.ts`, `rgba32fReadback.test.ts`, updated `engineContract.test.ts`; core ledger-vs-capabilities gate passes with `pt-webgl2.supportDetails.denoisers['oidn-final']='native'`. | done |

---

## Bucket D — Hygiene / maintainability (the "obvious gaps, just close them")

Mostly S-effort. These don't change rendering but they mislead readers, ship dead weight,
or silently drop user data — exactly the rot that has made the maturity picture hard to read.

> **Status 2026-06-12:** D2 (silent drops), D4 (memory accounting), D5 (stale comments),
> D7 (SVGF allocation), D8 (fork lint), D9 (traceTier dedup), D3 (contract + ingestion),
> and C1 (OIDN clear-error/capabilities) are closed in HEAD. D1 is no longer a single
> "delete all names" task: source verification found stale names, public/test helpers, and
> one real pt-webgl2 residue; the D1 row below now records the explicit keep/remove policy
> decision and is closed. D6 bind-group churn is now closed in HEAD; remaining:
> D3 per-backend BSDF consumption (B-bucket fidelity), and **three new lead-verified rows added 2026-06-13 from the maturity
> audit: D10 (pt-webgpu full-tier storage-buffer limit constant undercounts 31→33),
> D11 (pt-webgl2 declared-but-never-uploaded `u_volumeDensity`/`materialLodDepth`),
> D12 (shared-bvh `worldSpaceMerge` uv1 zero-triangle desync) — all three are now
> closed in HEAD.**

- **D1 — Dead code removal / source reconciliation** ✅ POLICY CLOSED 2026-06-16: pt-webgl2 `frameParamsPacker.ts`/`uploadFrameParams`/`#paramsUbo`/`#bindParamsUbo` are stale Road references (no active implementation remains; the misleading `glResources` FrameParams-UBO prose was removed); pt-webgl2 `'additive'` accumulation is now fully removed from the compile contract and emitted GLSL (`FEATURE_ADDITIVE_ACCUM` blocks pruned; `composeTraceGlsl.test.ts` gates it); retired `pt-webgl` debounce symbols are absent from active packages (legacy staging/docs only); `PPGCoordinator.resetTrainingAccumulators`, `heroStrategy`, old `GPU_SKIN_BVH_WGSL`, `ownsEnvSampler`, and `cleanupAfterSubmit` are stale names/renames rather than live deletion targets. The policy call is now explicit: keep `packCameUBO` as an exported stained-glass host utility with tests/README but no in-repo runtime shader consumer; keep `sTreeAccumulate`/`resetAccumulators` as CPU oracle/test helpers while production PPG uses GPU readback counts; keep `probesPerFrame` as ABI layout/packing even when a shader path does not branch on it; keep `expandIndicesToStride4` as an exported shared-bvh convenience with tests/README; keep `RESTIR_PT_HYBRID_SHIFT_HARNESS_WGSL` as a test/GPU-oracle harness; keep walkaround-rc TSL references as historical mapping docs/comments with raw-WebGPU boundary tests. Do not remove these as dead runtime code without a future API-breaking cleanup decision. **Effort: closed.**
- **D2 — Silent data drops** ✅ DONE / reconciled in HEAD: the old `three-bindings/src/index.ts` drop claim is stale (no `packages/three-bindings`, `@vitrum/three-bindings`, `sceneFromThree`, or `vitrumSceneToThree` remains after the glTF-adapter path replaced it); pt-webgpu `MaterialTextureArray.warnings` route through `UploadedSceneBuffers.warnings` / the engine structured warning path; heterogeneous texture-array layers now expose per-layer UV-fit scales (`MaterialTextureArray.layerUvScales`) and the full-tier material samplers address the copied source rect per map instead of sampling padded black texels. 2026-06-15 follow-up: glTF sampler `magFilter`/`minFilter`/mip intent now survives into core `TextureRef` and `textureDecodeReport`; exact backend filter/mipmap non-enforcement is now a structured `*.samplerPolicy` compatibility issue and `reject-degraded` gate rather than import-time data loss. **Effort: closed.**
- **D3 — Contract material gaps** ✅ DONE (contract + ingestion) / ◑ consumption tracked: added `specularIntensity`/`specularColor` (+ their maps), `bumpMap`/`bumpScale`, `displacementMap`/`displacementScale`/`displacementBias`, `lightMap`/`lightMapIntensity`, `envMapIntensity` as first-class optional fields on core `MaterialSpec` (+ `MaterialMapFields` slice); `three-bindings.convertMaterial` now extracts them so the THREE→core data loss is closed (+4 tests). **2026-06-15 follow-up:** walkaround now packs `envMapIntensity` in material-atlas metadata and applies it coherently to HDRI ReSTIR-DI candidate scoring, canonical temporal/spatial p-hat reuse, and shade resolve; the core ledger row is native with atlas/oracle/mutation tests. **2026-06-15 DI material-payload wave:** walkaround ReSTIR-DI `PrimarySurface` now carries the material-atlas payload used by shade-owned direct lighting; RIS candidate scoring, finalization, temporal reuse, and spatial reuse call `evalGGXWithSpecularClearcoatSheen` via `restir_di_eval_surface_brdf`, consuming base/vertex/roughness/metallic maps, normal/bump/clearcoat-normal perturbations, specular scalar/color maps, clearcoat, sheen, anisotropy, iridescence, and env-map intensity. `restirDiMaterialParity.test.ts` + shader-gate pin the path. **2026-06-15 GI material-payload wave:** default/NRC ReSTIR-GI suffix shading now consumes the same mapped material payload through `restirGiMaterial.wgsl.ts`; NRC records use mapped albedo/roughness. **2026-06-16 GI receiver-lobe target wave:** ReSTIR-GI producer and reuse `pHat` now evaluate the receiver material/lobes through `restir_gi_receiver_phat_from_payload()` / `restir_gi_receiver_phat_from_surface_or_geometry()`; the compact reservoir still stores geometry+`Lo`, with temporal previous-domain material recast falling back to geometry when exact previous-camera reconstruction is unavailable. **REMAINING — per-backend BSDF consumption (these require golden-breaking material-layout changes, so they're real B-bucket fidelity work, not ingestion): displacement-map geometry plus specialty rows ledgered elsewhere, and GPU A/B evidence for the newly material-aware GI receiver targeting.** Scalar `specularColor`/`specularIntensity` now have PT backend coverage and walkaround approximate shade-owned/ReSTIR-DI/GI suffix/receiver modulation; `bumpMap`/`bumpScale` now have PT native coverage plus walkaround approximate visible-normal perturbation through shade-owned, ReSTIR-DI pHat, and ReSTIR-GI suffix/receiver paths. **Effort: ingestion M (done); consumption S–L per field.**
- **D4 — Memory accounting** ✅ VERIFIED closed in HEAD: `UploadedSceneBuffers.gpuMemoryBytes()` sums live scene buffers + material texture arrays (`packages/pt-webgpu/src/scene/uploadSceneBuffers.ts:1408`), and `debug.estimatedGpuMemoryBytes()` includes those bytes under `byCategory.scene`, `byTextureFormat`, and `byBufferUsage.storage` before telemetry emits the scalar total (`packages/pt-webgpu/src/index.ts:787`, `:950`). **Effort: closed.**
- **D5 — Stale comments contradicting code** ✅ CLOSED 2026-06-12: source-read verified the RC light-model and current `packages/walkaround-rc/src/cascadeDispatch.ts` verification-status citations were already corrected; `createRestirGIFrameResources.ts` now says the GRIS reconnection cache is read by the reuse variants today, and `atrousVariance.wgsl.ts` now calls `svgfVarianceMain`/`svgfAtrousMain` legacy entry-point names rather than evidence that the module is Schied SVGF. Focused stale-comment gates pin those statements. **Effort: closed.**
- **D6 — Per-frame bind-group churn** ✅ CLOSED 2026-06-13: `PipelineResourceCache` now memoizes texture views and a bounded set of bind-group key variants per id (so ping-pong groups reuse both hot variants instead of missing every other frame). Central frame/scene/ubo/risGi/composite/hybrid/light-tree/GTAO groups were already cached; this wave added cache keys for ReGIR, sample-budget, GTAO upsample, motion vectors, indirect combine, indirect temporal accum, temporal/ spatial GI, checkerboard prefill, resolve, PPG update, temporal accum, indirect à-trous, and built-in denoisers (`atrous`, default `atrous-variance`, `svgf-real`, `bmfr`, `neural`). UBO-writing builders were split so alpha/sigma/uniform writes still execute every dispatch while the bind group object is reused. `passBindGroupCache.test.ts` pins cache reuse, ping-pong variants, invalidation after identity changes, tuple-valued PPG cache entries, and live per-frame UBO writes. Remaining `createBindGroup` source hits are setup/lifecycle/harness paths or cached builder internals rather than default per-frame pass churn. **Effort: closed.**
- **D7 — SVGF texture allocation** ✅ CLOSED 2026-06-12: `createSvgfFrameResources.ts` now collapses the heavy SVGF history textures and the current/previous object-id textures to 1×1 whenever the active denoiser is not `svgf-real`; shade writes object IDs through a dimension-guarded helper so the inactive 1×1 storage texture remains a legal frame-layout placeholder. `gpuMemoryEstimate.test.ts`, `svgfObjectIdResources.test.ts`, and `svgfObjectId.test.ts` pin the inactive placeholder sizes, active full-res sizes, and guarded shader store. **Effort: closed.**
- **D8 — fork lint red** ✅ FIXED (bumped to ESLint 9): the red was an `eslint@8` vs `@typescript-eslint@8` plugin crash (`no-unused-expressions` reading `allowShortCircuit`) — NOT the audit's stale-SSS gate (`tsc` + `shader-smoke` always passed). Fix: bumped the fork to `eslint@^9.39.4` (deduped to root; had to prune an orphan nested `eslint@8.57.1` the lockfile kept reinstalling — uninstall→reinstall on the workspace cleared it), kept `.eslintrc.json` via `ESLINT_USE_FLAT_CONFIG=false` in the lint script, and made its `extends` hoist-proof (`"mdcs"` shareable name instead of a relative `./node_modules/...` path that broke when mdcs hoisted to root). `npm run lint` is now green (0 errors, 1 pre-existing `no-unused-vars` warning in `example/`; tsc + shader-smoke pass). **Future:** eslintrc is deprecated in eslint v10 → a flat-config migration when the repo moves to v10.
- **D9 — traceTier dedup** ✅ CLOSED 2026-06-12: source-read verified `WebGl2TraceTier` is owned by `packages/pt-webgl2/src/traceTier.ts` and re-exported from `options.ts` instead of being duplicated; `traceTier.ts` and the package README now describe `lite` as aux-buffer-only degradation with the trace kernel unchanged rather than promising hidden bounce/texture caps. `traceTier.test.ts` pins both the single-source type surface and the lite-tier policy wording. **Effort: closed.**
- **D10 — pt-webgpu full-tier storage-buffer limit constant undercounts (ADDED 2026-06-13 — lead-verified audit)** ✅ CLOSED 2026-06-13 / prose reconciled 2026-06-15: corrected `PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE` from the stale 31 floor to the current 34-binding full-tier floor (`webgpuLimits.ts`), and the ReSTIR-PT reuse floor now derives to 38. `webgpuLimits.test.ts` counts distinct `var<storage>` declarations from the composed full-tier WGSL (`PT_WEBGPU_TRACE_WGSL`) so the exported floor cannot drift from the actual layout. Updated the lite-tier warning text, full-tier test fixtures, H14 ReSTIR-PT fixture, and `uploadSceneBuffers.ts` group-3 binding note to remove stale 31/35 and "bindings 0–9" assumptions. Focused pt-webgpu tests + typecheck passed; later group-3 additions remain covered by the derive-from-WGSL test.
- **D11 — pt-webgl2 declared-but-never-uploaded uniforms (inert features) (ADDED 2026-06-13 — lead-verified audit)** ✅ CLOSED 2026-06-13: source classification found two different cases. The old scene-global homogeneous medium (`u_volumeDensity`/`u_scatterAlbedo`/`u_anisotropyG` + `volumeMarch`) had no core contract and no TS setter, while `FEATURE_FOG` is already pinned false for future fog-volume primitives; the dead global-medium declarations, active branch, and unused `volume_march` GLSL module were removed, and `featureTypes.ts` now documents that omission honestly. `materialLodDepth` is a real optional pt-webgl2 performance knob: `PTEngineWebGL2Options.materialLodDepth` defaults to `0` (historical full-fidelity texture sampling at every bounce), validates finite `>=0`, flows through `FrameUniforms`, and is uploaded via `prog.setInt('materialLodDepth', ...)`; opting into a positive value activates the existing depth-based texture-LOD branch. Guards: `composeTraceGlsl.test.ts` asserts the removed global-medium uniforms/branch are absent from the active shader; `uploadGapGuard.test.ts` asserts default and opt-in `materialLodDepth` uploads. Focused pt-webgl2 composer/upload tests passed.
- **D12 — shared-bvh `worldSpaceMerge` uv1 range desync on zero-triangle primitives (ADDED 2026-06-13 — lead-verified audit)** ✅ CLOSED 2026-06-13: `mergeUv1FromCore()` now mirrors the `localTriCount === 0` skip used by `mergeWorldSpaceFromCore()`, keeping `meshVertexRanges` indexing aligned when a ≥3-vertex primitive has an empty/zero-triangle index buffer. `scenePack.test.ts` adds the degenerate-before-valid uv1 regression; focused shared-bvh scenePack tests + typecheck passed.

---

## Suggested sequencing

1. **Hygiene + provisioning first (Bucket D + C1).** Mostly S-effort, removes the noise that
   makes maturity hard to read, stops silent data loss, and makes denoising turnkey. A few
   days, high signal-to-effort.
2. **Finish the cheap frontier wins (A2 PPG sTree, A8 GRIS-default, A1 ReSTIR-PT composite).**
   These convert "wired-but-inert" into "actually delivers."
3. **Close the default-path fidelity ceilings users see first (B4 mesh-area NEE, B3 IBL,
   B1 glossy GI, B9 multiscatter).** This is the biggest perceived-quality lift.
4. **Then the research-grade items (A3 true spectral, A10/C2 neural weights, A9 BDPT-prod,
   A4 caustic decision, B7 SBVH).** Multi-week each; schedule deliberately.

## Addendum — 2026-06-09 second-wave deep-read (11 agents, lead-verified)

A full line-by-line re-read of all 12 packages found that several bucket entries UNDERSTATE
the gap, plus new fidelity items. **Bugs/broken-surface findings went to `items_to_fix.md`
Section H** (H1–H38, ✅/◻ legend there); this addendum only adjusts THIS doc's picture:

- **"Foundations + default render paths ~90%" needs a caveat: pt-webgl2 is NOT at RC level.**
  ✅ Its entire analytic-light system is inert (`lights.count` never uploaded — items H1),
  `spectral` renders black (H2), directly-visible env never accumulates (H3), and `bdpt` is
  never driven (H5). As shipped it is an emissive-geometry-lit RGB tracer (~60-70%); the
  fork-vs-native A/Bs passed because they exercised exactly the paths that work
  (emissive-fold Cornell, glass, IBL-indirect, textures). Treat pt-webgl2's
  release-candidate label as suspended until H1–H5 land.
- **B1 is DONE (Wave A):** the pre-Wave-A claim that metals were excluded from direct
  light is stale. Current `shade.wgsl.ts:252` (`lo_analyticNEE`) gates only on `isGlass`;
  `shade.wgsl.ts:333` (`lo_direct`) gates only on `isGlass` — metals receive both analytic
  NEE and ReSTIR-DI direct light. `lo_indirectSpecular` (line 580) reflects the GI
  reservoir via the GGX specular lobe for metals/glossy. DDGI's diffuse `lo_indirect`
  (line 429) still exits early for `isGlass || isMetal`, but that is intentional — metals
  consume `lo_indirectSpecular` instead (the DDGI cache is Lambertian-targeted; applying it
  directly to a mirror-like metal would be physically wrong). Glass direct-light terms
  are still shade-owned/transparent-transport-scoped, but primary glass no longer loses
  GI: default ReSTIR-GI uses the bounded refracted walk, and the 2026-06-17 NRC parity
  follow-up mirrors that walk before bypassing cache substitution/training.
- **A2/H25 (PPG) is now reflected in the closed P0 table:** sTree splitting/runaway
  and dTree interior-flux propagation are implemented; `PPGCoordinator` propagates
  subtree sums bottom-up before refinement, `ppgPdf.wgsl.ts` samples child flux
  proportionally, and `dTreeInteriorFlux.test.ts` pins CPU/GPU sampling/pdf parity.
- **A6/H26-H27 (NRC) no longer carries the filed structural defects:** the GI-RIS
  NRC variant seeds spread accumulation from `0.0`, derives the primary footprint
  from `nrcCfg.cameraPixelPdf`, tracks the first fired candidate, writes one
  post-loop training record with `r.Lo`, and uses atomic slot claims. 2026-06-15
  follow-up: CPU record unpacking now scans the full encoded-input prefix instead
  of using `input[0]` as a brittle sentinel. The remaining A6 work is
  consumability/quality validation and default-tier policy, not those stale
  spread/target/torn-record bugs.
- **A10/H28 (neural denoiser) is no longer blocked on ReLU bind aliasing:** in-place
  ReLU layers allocate distinct output buffers and remap downstream tensor reads;
  `reluPingPong.test.ts` pins the no read/read_write alias invariant. The remaining
  A10 gap is a production-quality checkpoint plus quality A/B; 2026-06-15 follow-up
  added graph-weight validation, actual tracked-checkpoint loader/spec tests, and
  raw-HDR fallback on inference dispatch failure.
- **NEW B13 — ✅ CLOSED / SOURCE-VERIFIED 2026-06-13 — walkaround texture seam UVs**
  `restir/bvhCore.ts` now packs real UVs at both build seams: scene-pack UVs are
  extracted to stride-2 before `packUVIntoPositionW(...)`, and merged geometry uses
  `merged.uvs`. `hRemediationItems.test.ts` pins the old all-zero UV failure.
- **NEW B14 — ✅ DONE (v1-closure Wave 1/2, 0dbaff5 + 2026-06-16 explicit-map follow-up) — DDGI emitter NEE complete**: rect/disc-area fixture point-proxy REMOVED (was double-counted against H18 NEE triangles — `coreEmittersToDDGILights.ts:155` map deleted; code-verified 0 hits for fixture-rect pattern); mesh-area emitter triangles now expand into the probe NEE list (`HybridEngineLifecycle.ts:545`, `collectMeshAreaEmitterTrisFromCore` + `setEmitterTris`); emissive-mesh scenes get nonzero DDGI indirect. 2026-06-16 follow-up: explicit mesh-area emitters whose referenced mesh material has a CPU-readable `emissiveMap` now split into bounded barycentric micro-triangles, carry `sourceSubdiv*` metadata for material-atlas sampling when TLAS bindings are available, and keep constant scalar behavior for unmapped emitters. Code-verified both fix sites.
- **NEW B15 — ✅ DONE (Wave B) scene-scale-aware radiometric clamp defaults**:
  `HybridEngineScaleAwareClamps.ts` + `HybridEngine.ts:719-1145` derive the clamp DEFAULTS
  from the scene diagonal at setScene (1/s² law on the radiometric knobs: irradiance/GI-W/
  firefly clamps), so the Cornell-tuned absolutes no longer cap GI energy in larger scenes;
  hosts that set a clamp explicitly keep their absolute value (override flags captured).
  Root-caused the size-200 estimator instability to this bimodal clipping (the 1/dist²
  suspect was REFUTED — the base estimator is scale-invariant). V28 clamp-sweep scenario
  specced. **Remaining:** real-GPU clamp-sweep A/B (V28-B).
- **NEW B16 — ✅ DONE (Wave B + 2026-06-15 material-payload wave) DI BRDF candidate + rich pHat**: `ris.wgsl.ts:83,255-279` now SAMPLES
  the `M_BRDF=1` GGX-VNDF candidate (measure-converted solid-angle→same RIS measure),
  contributing to glossy DI. **2026-06-15:** ReSTIR-DI pHat now uses the same atlas-backed
  extension-aware BRDF surface as shade-owned direct lighting across RIS, finalization,
  temporal reuse, and spatial reuse (`restir_di_eval_surface_brdf`, `castPrimary`,
  `sampleRestirDIMaterialPayloadForHit`). Render-changing (NOT byte-identity-preserving). **Remaining:**
  glossy/rich-material DI A/B (V28-B).
- **C-bucket correction:** H35 is now closed/source-verified. OIDN runtime failures flow
  through structured backend error/state surfaces, walkaround consumes denoiser `state()`
  in `FrameStats.denoiserState`, and the one-shot shared-denoiser WebGPU dispatchers clean
  up transient textures/buffers from `finally` on readback failure. Remaining C-bucket work
  should focus on still-open renderer/API gaps rather than the stale OIDN console-warning
  claim.

**Second-wave claims-surface audit (same session, items H39–H59) added three structural
buckets that the A–D framing was missing:**

- **NEW C4 — examples / DX surface** ✅ SOURCE-RECONCILED 2026-06-12:
  the old "zero examples" claim is stale. The repo now has runnable Vite examples for
  `createEngine`, `attachVitrum`, `VitrumCanvas`, `createProgressiveEngine`, both direct
  PT factories, and **`loadGltfWithEngine()`** (`examples/gltf-viewer`). H57 is closed
  as a provisioning gap; future examples are product polish, not a Road blocker.
- **NEW C5 — contract-truth reconciliation** ✅: `promiseLedger` rows contradict shipped
  runtime capabilities (pt-webgl2 analytic/mutations/aux); the fidelity matrix's `pt-webgl`
  column describes a deleted package and omits pt-webgl2; CHANGELOG `[Unreleased]` has no
  Removed entry for e14000c; ~6 tool READMEs document dead workflows; 2 packages have no
  README (items H39–H45, H59). S–M effort, zero rendering risk, large honesty payoff.
- **NEW D10 — test-infrastructure gates** ✅/◻: the GL uniform-upload completeness
  sub-gate is now landed in `pt-webgl2` (`uploadGapGuard.test.ts` extracts declared
  shader uniforms, exercises default/spectral/DOF/BDPT frames, and requires every
  non-sampler uniform to be uploaded or explicitly classified). That gate immediately
  closed one additional inert residue: `backgroundBlur` is now a validated
  `PTEngineWebGL2Options.backgroundBlur` knob, default `0`, uploaded every frame.
  Follow-up D10 gate wave: `tools/shader-gate` is now a real workspace package, so
  ordinary `npm test` runs the CPU GLSL production-variant compile gate and its
  injected-error self-test (`@vitrum/shader-gate`); the first strict run exposed and
  fixed pt-webgl2's SSS GLSL helper holes (`sampleExponential` / `sampleHG_glsl` /
  `hg_phase`) that unit string tests had missed. The size-validating GPU test stub is
  also landed and hardened in `pt-webgpu` (`gpuStub.test.ts`, `gpuResourcesUsage.test.ts`)
  so mock devices now reject impossible buffer/texture descriptors, invalid buffer
  usage flags, bind-group range overflows, missing `UNIFORM`/`STORAGE` usage bits, and
  `minBindingSize` violations; the SPPM resource guard proves it does not allocate past
  an artificial `maxBufferSize`. The walkaround-specific sizing stub is now hardened
  too: `dummyBufferSizing.test.ts` proves it rejects invalid buffer usage, layout-derived
  `minBindingSize` failures, missing/duplicate/unknown layout entries, buffer range
  overflows, texture-slot buffer resources, and missing `UNIFORM`/`STORAGE` usage bits.
  The H55 proof-gate residue is narrowed as well:
  `frameParamsSlotCrossCheck.test.ts` derives TS slot offsets from the WGSL struct, and
  `cpuTracerDriftTripwire.test.ts` now uses literal frozen WGSL function-body hashes
  rather than live-computed "frozen" values. `oidnFinalDenoiser.test.ts` now pins
  dispatch-time OIDN failure degradation on walkaround: raw HDR fallback remains visible,
  state becomes retryable `failed`, and the next dispatch retries successfully.
  **2026-06-18 follow-up:** dispatch-time OIDN failures now also route through
  `HybridEngine.onWarning` as `walkaround-hybrid.oidn-final-inference-failed`
  with model URL, dimensions, fallback target, and retryability; when no host warning
  sink is installed, the prior console error fallback remains.
  `sprint9-10a-welford.test.ts` now adds an independent two-pass arithmetic oracle for
  Welford mean/M2 variance plus sample-budget first-frame and threshold tier behavior,
  so that formerly string-only adaptive-sampling path has executable numeric coverage.
  `gtaoQuarterRes.test.ts` now adds an independent CPU oracle for the GTAO
  per-channel bilateral upsample: equal-surface taps average per channel, depth
  edges prefer matching low-res cells, and zero bilateral weight falls back to the
  per-channel 2x2 average. `volumetricSss.test.ts` now adds an independent
  pt-webgpu oracle for volume-thickness vec4 #28 packing,
  `materialAttenuationDistance`, negative-segment handling, and finite
  Beer-Lambert absorption for an infinite ray segment through an authored slab.
  `mneeNewton.test.ts` now adds a flat-mirror analytic CPU oracle proving the
  mirror-image intersection zeroes the MNEE half-vector residual while a shifted
  vertex remains nonzero. `restirPtReuseContract.test.ts` now adds numeric
  pairwise-GRIS temporal weight oracles, including reused-reservoir
  `pdfSrc`-independence and final `W = w_sum / pHat` without `M` normalization.
  `materialTextures.test.ts` now adds CPU coverage for shader-side wrap modes,
  KHR/THREE UV transforms, and post-wrap UV-fit scaling.
  `oracle.concentricDiscSample.test.ts` now independently pins the Shirley-Chiu
  concentric-disc map's quadrant anchors, unit-disc bound, radial area law, and
  signed WGSL divisions; this fixed the full/lite/adjoint disc-area two-quadrant
  sign bug that the old SHA/string pins could not see.
  `spectralTransportInvariant.test.ts` now independently pins
  `activeLayerWeightRgb` pass-through, spectral luminance-collapse, and
  negative-luminance clamp behavior, replacing the old material-layer
  string-only proof with executable oracle coverage.
  `oracle.environmentRotation.test.ts` now independently pins the HDRI
  `rotationY` helper behavior: zero-rotation identity, +/-90 degree lookup
  convention, inverse relationship, direction-length preservation, and
  full/lite call-site linkage.
  `bdptEmitterPickCpu.test.ts` now independently pins the BDPT selectable-emitter
  power rule: Rec.709 luminance equivalence for differently-coloured point-like
  emitters, the `1e-20` positive floor for non-positive light colours, finite
  rect/disc/mesh area multiplication, and the WGSL helper's linkage to canonical
  `luminance(c)` with the same floor.
  `pt-webgl2`'s `uploadGapGuard.test.ts` now pins the environment upload path:
  `environment:'none'` drives both `envMapInfo.totalSum` and
  `environmentIntensity` to zero, while a raw HDRI scene drives them positive.
  `adjointEmitterGradientOracle.test.ts` now independently pins pt-webgpu
  path-replay emitter `color` / `intensity` gradients against finite
  differences of the direct-light radiance law across directional, point, spot,
  finite-area, and mapped mesh-area quotient cases, replacing that lane's old
  string-only proof with executable behavior coverage.
  `oracle.directionalConeSample.test.ts` now independently pins the shared
  soft-directional/sun cone sampler used by full-tier forward tracing and the
  adjoint replay pass: sign-encoded angular-diameter decode, rim/centre
  endpoints, unit-length directions, alternate-basis handling, and the
  solid-angle-uniform `cos(theta)` law.
  `extensionLobeReference.test.ts` now independently pins the
  base/clearcoat/sheen source-lobe mixture probabilities and inverse-probability
  throughput relationship for opaque plus transmissive source sampling, so the
  sampled-density / `brdfExtensionLobeWeightSum` lane is no longer protected only
  by `wgslContract` string slices.
  `oracle.analyticShapes.test.ts` now independently pins pt-webgpu full-tier
  analytic-shape local intersectors for sphere, box, capsule, cylinder, and
  H-channel came plus the public shape-id ↔ WGSL dispatch linkage, converting
  that advertised capability from capability/string coverage into behavior
  coverage.
  `oracle.normalMapTangentFrame.test.ts` now independently pins authored
  tangent.xyzw handedness, UV-gradient tangent fallback, `normalScale`,
  front/back layer normal descriptor selection, and clearcoat-normal scaling,
  converting the normal-map tangent-frame lane from string/linkage coverage into
  executable behavior coverage.
  Source reconciliation on 2026-06-15 verified that the WebGPU WGSL/PASS_ORDER
  parse gate itself is already present and CI-backed: `npm run shader-gate`
  compiles 51 production WGSL modules (pt-webgpu full/lite/ReSTIR-PT/SPPM,
  walkaround PASS_ORDER roots including NRC when the adapter supports it,
  shared-denoisers, and walkaround-rc), and `--self-test` catches an injected
  broken shader. Follow-up reconciliation on 2026-06-15 strengthened the WGSL
  gate from parse-only to adapter-backed pipeline creation: the same command now
  creates 28 compute/production pipeline variants, including pt-webgpu full,
  lite, ReSTIR-PT, and SPPM entries; shared-denoiser and walkaround-rc kernels;
  and walkaround production default, GRIS, PPG, ReGIR, and NRC-capable layouts
  via `compilePipelines()`. Remaining D10/H55 proof work is now the non-mirrored
  WGSL behavior-oracle class, not missing shader or pipeline creation gates.
  M effort total; this is what stops the next H1 from shipping green.
- **MaterialSpec consumption matrix** (items H46–H52): the original "~60 fields
  advertised / walkaround consumes ~8" audit is now stale. Later waves closed
  pt-webgpu AO/bump/light maps, shadow flags, authored/generated tangent.xyzw
  consumption, and walkaround atlas/rich-lobe consumption for the current
	  map-backed material rows. Remaining broad-residual rows are displacement,
	  receiveShadow, transparent light-map/emissive/GI promotion, UV-varying emissive/light-map promotion,
  unsupported specialty fields (spectral/scattering/layered/thin-film), and
  backend-specific approximation/proof rows ledgered in `BACKEND_PROMISE_LEDGER`.
  The former pt-webgl2 `TextureRef.texCoord`/`alphaMap.transform` warning is closed:
  pt-webgl2 now packs per-map UV bits, KHR texture transforms, and wrap modes for its
  atlas-backed material maps, including alpha sampling in both surface and attenuation
  paths. `denoiser` is still an honest unsupported/degrade path on pt-webgl2 except for
  explicit `oidn-final` bridge behavior (H48).

## Open decisions (need a call before building)

- **A4 DECIDED (2026-06-10):** build real SPPM — DONE (v1-closure Wave 4, 06910e2).
- **A7 DECIDED (2026-06-10):** keep RC + finish — DONE (v1-closure Wave 5, caab499). See B3 update.
- **A6/A10 DECIDED (source-verified 2026-06-18):** keep NRC/neural opt-in + experimental until quality/default-tier evidence and a production neural checkpoint exist. Source truth: `HybridEngine` advertises `walkaround-hybrid-nrc-biased-cache` / `walkaround-hybrid-neural-denoiser-host-weights` only when opted in, emits `walkaround-hybrid.nrc-experimental-biased` and `walkaround-hybrid.neural-host-weights-required`, and `capabilitiesPartition.test.ts` pins both capability flags and structured warnings. `neuralWeightsRoundTrip.test.ts` proves loader/allocation shape compatibility for tracked research checkpoints, not production default quality.
- **A8 DECIDED (2026-06-10) + A/B ELEVATED (2026-06-13):** biased default retained for the realtime budget; bias quantified+documented; unbiased GRIS first-class opt-in. The default-flip question (F6) is NOT closed — it is now gated on the **active** A8 GPU A/B task (converged-unbiasedness of GRIS-on + measured error of the biased default on the wsl-gpu rig). Revisit the flip with those numbers.
- **sun-NEE default: DECIDED + DONE (2026-06-10, item 4 R8-A).** `lo_sunNEE` wired in `shade.wgsl.ts` — deterministic shadow ray + evalGGX BRDF, default-ON for opaque surfaces, no flag required. `lo_sg_caustic` (stainedGlass flag) unchanged — tinted-glass transmittance path. No double-count vs DDGI indirect: DDGI stores sun→wall radiance at the PROBE bounce surface; lo_sunNEE is sun→receiver DIRECT, disjoint paths. Behavioral gate: `wh/directional-sun` (LUM_THRESHOLD 0.005, intensity 3.0). Render-changing for directional-lit scenes → V28-B recapture in R8-C.
- **B1 receiver-lobe target wave (2026-06-16):** default/NRC GI-RIS suffix `Lo` uses mapped material payloads and the extension-aware GGX/clearcoat/sheen proxy, and producer/default/GRIS reuse `pHat` now evaluates the receiver-side material lobes instead of the old pure `luminance(Lo)·cos·INV_PI` target for rich materials. The compact reservoir remains geometry+`Lo`; temporal previous-domain material recast falls back to geometry when exact previous-camera reconstruction is unavailable. Remaining work is GPU A/B promotion and V28-B recapture, not an unimplemented receiver-lobe target.
- **B1 tail + B1-ior-per-tri: DONE (2026-06-10; NRC parity 2026-06-17).** R8-B tail: 1-interface refraction walk in `risGi.wgsl.ts`; 2026-06-17 follow-up mirrors that bounded walk in `risGiNrc.wgsl.ts` and bypasses NRC substitution/training for glass instead of storing an immediate empty reservoir. Shade `lo_transmittedGI` weights by Fresnel-T × Beer tint; `wh/glass-gi` behavioral gate PASS. **B1-ior-per-tri (2026-06-10 follow-up, DONE):** `bvh_material` bits[15:8] carry IOR quantized over [1.0, 3.0] (step ≈ 0.008); `materialDecode.wgsl` exposes `decodeIor()`; GI glass walks decode per-tri IOR via `decodeIor(glassPrimaryPacked)` (no more hardcoded 1.5); shade `lo_transmittedGI` computes Schlick F0 from per-tri IOR (`((ior−1)/(ior+1))²`); rough-glass GI: roughness > 0.1 perturbs the Snell refracted direction by a GGX-distributed offset (one sample, per-tri roughness) so frosted glass receives blurred GI; smooth glass stays exact-Snell-byte-identical. Default IOR=1.5 → byte 64 → decodes to 1.502 (error < 0.003, within glass dispersion); `packBVHRoughMetalFromCore` / structural packer / `repackBVHMaterialRange` all updated. Structural tests live in `roughMetalPacking.test.ts` + `b1GlossyMetalGi.test.ts`. Render-changing for non-1.5-IOR/NRC glass → V28-B recapture in R8-C.

---

## Addendum — 2026-06-11 condensed implementation spec (glTF + walkaround + arbitrary glTF)

> Authored 2026-06-11 from a code-truth audit of `@vitrum/gltf-adapter` (working tree),
> `promiseLedger.ts`, engine entry points, walkaround consumption, and pt-backend material
> gaps. **No timelines or effort estimates.** Supersedes the conversational plan from the
> same session for execution ordering; does not retract the A–D bucket items above — this
> addendum is the **closure checklist** for the three user-facing 100% targets.
>
> **Baseline:** `main` @ `309fdebe` (oracle math fixes). ~2,148 LOC uncommitted across
> gltf-adapter, engine bridge, unlit all backends, H30 canvas sizing, progressive handoff
> fallback, ledger fixes. **Land Phase 0 first** before treating any claim below as shipped.
>
> **Companion docs:** use `plan/gap-closure-execution-plan.md` for the curated
> closeout plan and `items_to_fix.md` §H for legacy implementation notes.
>
> **REVIEW RECONCILIATION (2026-06-12, lead-verified against `main@309fdebe`):**
> 1. **Several P0 rows below were already landed and pushed before this addendum
>    was written** — they are now marked `✅ LANDED` in place. Do NOT re-execute
>    them; verify-on-read if in doubt. Affected: Phase 2A (PTWG-01..05, WEBGL2-01,
>    H49), Phase 3A (W-HYB-01/02/03), Phase 1C (GLTF-01, CORE-01).
> 2. **H49 was wrong, not just stale**: pt-webgl2 ALREADY packs
>    `specularColor`/`specularIntensity` (+ both maps) at `materialsTexture.ts:168-232`;
>    the ledger correctly grades them native. Row struck.
> 3. **H33's fix target was the wrong file**: `worldSpaceMerge.ts` `materialSig`
>    already includes attenuation fields. The real drift is
>    `sceneBvh.ts` `materialSetHashFloats` (~:173-191), which carries
>    `attenuationColor` only and omits `attenuationDistance`/`thickness` —
>    mutating only those fields skips the DDGI rebuild. Row corrected in 3G.
> 4. **V28-B baseline recapture was dropped from this addendum** — restored as
>    Phase 0.3. `309fdebe` is heavily render-changing (env-DI fix alone should
>    visibly brighten HDRI scenes; ALL pre-309fdebe baselines embed the ~66× bias).
> 5. **DECISION LOCKED (user, 2026-06-12): walkaround gets the FULL texture atlas**
>    — Phases 3C/3D/3E as written are committed scope, not optional. The apparent
>    conflict with the acceptance table (ledger-truth is already satisfied by
>    `unsupported` grades post-CAP-01) is resolved in favor of implementation:
>    walkaround material rows are expected to PROMOTE to native/approximate via
>    the atlas, not rest at honest-unsupported.
> 6. **COMPLETENESS PASS (2026-06-12):** three verified holes INSIDE the plan's own
>    acceptance criteria were added (texture wrap modes + mipmaps in Phase 1A;
>    `environment:'none'` phantom skylight + grayscale-directional shortcut class
>    widened into 2C), and **Phase 6** carried the gap-ledger residue outside
>    the three targets until the rows below were implemented or downgraded
>    (onError unification, lite single-BLAS, RC footguns,
>    sampled fingerprint, core contract additions; the pt-webgl2 NEE,
>    pt-webgpu trace-lite shader-gate mismatch, attachVitrum recreate scene-loss,
>    and morph-normal skip rows are closed by Wave 1). With Phases 0–6
>    executed, this plan IS the categorical close.
> 7. **FOLLOW-UP CROSSOUT PASS (2026-06-12):** verified and struck/narrowed the
>    rows that landed after this addendum: glTF required `KHR_materials_unlit` /
>    `KHR_materials_pbrSpecularGlossiness`, scalar spec-gloss compatibility
>    scoring, morph-tangent/point-line unsupported policy tests, bridge hook
>    pass-through, walkaround + PT unlit branches (`shadingModel` remains
>    intentionally `approximate`), and the pt-webgpu local BRDF-helper propagation
>    wave. Remaining payload/sampler/schema gaps stay open in place.

### What 100% means (acceptance, not aspiration)

| Target | Done when |
|--------|-----------|
| **glTF** | `loadGltfAsset` → `loadGltfForEngine` handles URL/GLB/JSON+external resources; `analyzeGltfAsset` + `rankGltfBackends` are complete; `GltfSceneController` drives skin/morph/TRS; every extension in `REQUIRED_EXTENSION_SUPPORT` (`featureReport.ts:183-207`) has import + compatibility + test; zero silent `console.warn` in adapter (string warnings in return values, plus structured import diagnostics for converter-owned degradations). |
| **Walkaround** | Every `MaterialSpec` key graded `native`/`approximate` in `WALKAROUND_MATERIALS` is consumed in GPU shaders; `CONSUMED_MATERIAL_FIELDS` (`consumedMaterialFields.ts`) matches ledger exactly; emitter/environment/shadow grades match runtime; P0 walkaround bugs (W-HYB-01..03, H25-H29) closed. |
| **Arbitrary glTF** | For any asset in Khronos sample set + internal hero fixtures: `loadGltfAsset` succeeds or throws structurally; `evaluateGltfBackendCompatibility(selectedBackend).unsupportedCount === 0` for used features OR `compatibilityMode` rejects before render; rendered output passes material-furnace + reference gate on **recommended** backend; `prefer:'auto'` uses feature report, not triangle count alone. |

**Explicit non-goals** (otherwise "100%" is undefined): displacement tessellation, production neural weights, true Hachisuka SPPM (A4-progressive shipped — separate from this campaign), cross-host GPU certification. glTF point/line topology is no longer a non-goal: it imports as explicit fallback-generated mesh geometry.

### Why plans stall at ~85% (gaps this spec closes)

| Trap | Symptom | This spec addresses |
|------|---------|---------------------|
| glTF API without **texture decode bridge** | Scene loads, everything flat gray | Phase 1 § Texture pipeline |
| Compatibility planner without **engine wiring** | Correct backend known, wrong one created | Phase 4 § `createEngine` + `VitrumCanvas` |
| Walkaround **scalar-only** + ledger drift | Textured glTF "works" with warnings | Phase 3 § Texture atlas (non-optional for WA 100%) |
| **PTWG-MAT-01** only on megakernel | glTF clearcoat wrong in BDPT/SPPM/ReSTIR | Phase 2 § Integrator audit matrix |
| **Lite tier** used for arbitrary glTF | Missing material maps/alpha silently | Phase 4 § Lite policy + Phase 2 |
| Animation **without** skin+morph+material sync | Controller patches positions, normals wrong | Phase 1 § Controller + Phase 4 |
| **Tangent** bugs on pt-webgl2 | Normal maps wrong on glTF | Phase 2 § WEBGL2-01 |
| Oracle fixes **without** reference renders | Tests green, images wrong | Phase 5 |
| Uncommitted work | "Plan done", `main` unchanged | Phase 0 |
| `pickBackend` triangle budget (`createEngineScale.ts:46-47`) | Textured asset → walkaround | Phase 4 |
| PPG/NRC/neural **enabled but broken** | Optional features crash bind groups | Phase 3 § Subsystems |
| **items_to_fix §H** residue | Silent wrong behavior | Woven into phases below |

---

### Phase 0 — Land baseline + mechanical gates

#### 0.1 Commit working tree

**Files:** 55 modified + 7 new under `packages/gltf-adapter/`, `packages/engine/`, `packages/core/`, `packages/walkaround-hybrid/`, `packages/pt-webgpu/`.

**Footgun:** ~~`textures.ts` header still says "adapter does not fetch external image URIs" while `assetLoader.ts` does~~ ✅ fixed — the file header now describes the pluggable image-byte handoff. The low-level `getImageBytes` warning remains intentional for callers that bypass `loadGltfAsset`.

#### 0.2 Gate infrastructure (blocks false 100%)

| Gate | Plug-in | Footgun |
|------|---------|---------|
| **GATE-01** | ✅ CLOSED — `core/src/__tests__/ledgerVsCapabilities.test.ts` imports live pt-webgl2 support/capability data and pins full-tier aux buffers, lite-tier downgrade, primitive/emitter/env/support-detail parity, and analytic fallback-generated-mesh rows against `BACKEND_PROMISE_LEDGER`. | Historical footgun resolved; keep this gate as the regression guard. |
| **GATE-02** | ✅ CLOSED — `core/src/__tests__/materialNativeEvidence.test.ts` enumerates every `native` material row from `BACKEND_PROMISE_LEDGER` and fails unless that backend/field has a named packer+shader/shared-classifier/readback evidence record with existing test/source file paths. | This is the ledger-evidence gate; renderer A/B and material-furnace proof still live in Phase 5 where required. |
| **GATE-06** | CPU GLSL gate now runs under ordinary `npm test` via `@vitrum/shader-gate`; WGSL/PASS_ORDER parse gate is source-verified present as root `npm run shader-gate` and CI-backed with lavapipe (51 production modules + self-test). Keep it explicit rather than default `npm test` because that path needs a WebGPU adapter. | WGSL string tests don't compile shaders; pipeline-layout creation remains a stronger future proof gate |
| **GATE-GLTF** | ✅ CLOSED — `gltfKhronosSweep.test.ts` exercises representative Khronos-style JSON fixtures through `analyzeGltfAsset` + compatibility ranking only: scalar mesh, textured PBR, extension glass, skin/morph/animation, compression hooks, source-path diagnostics, and full-vs-lite WebGPU profile differences. 2026-06-18 verification: `npm run gltf-material-sweep` passes (`maps=18`, `recommended=pt-webgl2/pt-webgl2`, `proof=pt/gltf-material-sweep`), and `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json npm run behavioral-gate -- --filter gltf` passes 11 adapter-backed render lanes including unlit, textured PBR, transmission, skinned animation, Draco/mock, point/line fallback, strip/fan, material sweep, real Box textured, real Draco, and real meshopt. | Live URL tests stay out of CI; broader material-furnace/reference-render promotion remains a later proof gate, not missing bridge/planner code. |

#### 0.3 V28-B baseline recapture (restored 2026-06-12 — was dropped from this addendum)

`309fdebe` (oracle math fixes) is render-changing on: walkaround DI (env ~66×
brighter where the bias applied, selected-xi −30% correction), DDGI visibility
in open scenes, pt-webgpu BDPT connections (~5×), lite rect NEE. **Every
pre-309fdebe reference render and benchmark baseline embeds the old biases**
(same class as the within-leaf-hit precedent: improvement confirmations, not
regression suspects). Recapture via `~/projects/wsl-gpu` before any later
render-changing wave lands, or A/B attribution becomes impossible.

---

### Phase 1 — glTF pipeline 100%

#### 1A — Asset loading (`assetLoader.ts`)

**Already in working tree:** `loadGltfAsset`, external buffer/image fetch, `extensionsRequired` enforcement, `imageBytes` map.

**Still required:**

| Task | Code | Plug-in | Footgun |
|------|------|---------|---------|
| ~~Typed errors~~ ✅ DONE | `packages/gltf-adapter/src/errors.ts`, `assetLoader.ts`, `gltfAssetApi.test.ts` | `GltfFetchFailed` / `GltfResourceNotFound` now carry `{ url, kind }` plus HTTP status when applicable; loader tests assert typed rejection. | Generic fetch/resource errors are gone from the high-level loader path. |
| ~~Cache hooks~~ ✅ DONE | `LoadGltfAssetOptions.cache`, `assetLoader.ts`, `gltfAssetApi.test.ts` | `fetchArrayBuffer` now checks/sets a host cache for resolved asset/buffer/image URLs, keyed by final URL + resource kind. | Relative assets with different `baseUri`s resolve to different absolute keys. |
| ~~`loadGltfAndDecodeTextures()` helper + decode report~~ ✅ API DONE | `assetLoader.ts`, `texturePipeline.ts`, `engineBridge.ts`, `index.ts`, `gltfAssetApi.test.ts` | High-level loading now returns `textureDecodeReport` and `loadGltfAndDecodeTextures()` invokes `decodeSceneTextures()` directly when a host `decodePixels` hook is supplied, returning decoded/unchanged counts plus structured diagnostics. The report walks the converted `Scene` and classifies each material `TextureRef` by field/path/UV/transform/handle-kind plus backend readiness (`pt-webgl2`, `pt-webgpu`, `walkaround-hybrid`). When decode is requested, inactive `KHR_materials_variants` material-table entries are decoded too and appear as synthetic `gltf-material-N` rows so variant switches cannot reintroduce raw handles. | This closes the adapter diagnostics/API surface; backend atlas/upload completion remains in 2B/3D. |
| ~~sRGB → linear~~ ✅ DONE for glTF diagnostics + PT backends; ✅ walkaround atlas color/data policy slices | `texturePipeline.ts`, `texturesArray.ts`, `materialTextureArray.ts`, `materialTextureAtlas.ts` | `textureDecodeReport` now reports each map's policy (`srgb` for baseColor/emissive/sheenColor/specularColor tint maps, `linear` for scalar/data maps). pt-webgpu already keeps separate `rgba8unorm-srgb` and `rgba8unorm` arrays; pt-webgl2 converts color-map payloads into its linear RGBA32F atlas exactly once, while role-aware layer maps keep a shared handle distinct when it is used as both a color and data map. Walkaround inverse-sRGB decodes readable uv0/uv1 `baseColorMap`, `emissiveMap`, specular-color, and sheen-color handles, and packs readable normal/scalar/data maps, extension-lobe maps, light maps, and `thicknessMap` as linear atlas layers; bump/displacement policy rows remain separate. |
| ~~NPOT / max dim~~ ✅ DECODE/API DONE | `texturePipeline.ts`, PT/WA atlas builders | `decodeSceneTextures(target:'cpu-linear', { maxTextureSize })` now resizes oversized decoded raw-image payloads before backend upload and reports the original/resized dimensions in structured diagnostics. pt-webgl2 and walkaround atlas builders already resample readable layers to a common max source dimension; pt-webgpu uses per-layer UV-fit scales and full-tier mip generation. | NPOT repeat-wrap remains an explicit diagnostic because exact border/mip parity is backend policy, not a silent import step. |
| Basis/WebP/DDS source policy | ✅ CODE CLOSED (`featureReport.ts`, `textures.ts`, `gltfExtensionPolicy.test.ts`, `gltfAssetApi.test.ts`): required/no-base-fallback texture-source extensions report `requires-hook`; optional alternates with a base `texture.source` fallback are compatibility-clean until the host opts into `textureSourceExtensions`. | `reject-degraded` no longer rejects deterministic PNG/JPEG fallback assets just because they also advertise KTX2/WebP/DDS alternates. | Default browser transcoder/decode path remains a future 4F polish item. |
| **Texture wrap modes (ADDED 2026-06-12 — verified hole)** | ~~`TextureRef.wrapS/wrapT` in core (`repeat`/`clamp`/`mirror`) + adapter reads `gltf.samplers`~~ ✅ API DONE (`material.ts`, `textures.ts`, `gltfTextureSweep.test.ts`, `gltfAssetApi.test.ts`, `texturePipeline.ts`). ✅ pt-webgpu full-tier material textures now pack per-map wrap pairs (`materialTextures.ts`) and WGSL applies repeat/clamp/mirrored-repeat before UV-fit scaling (`material.wgsl.ts`). ✅ pt-webgl2 now packs per-map wrap pairs at material texels 100..110 (after alphaMap transform at 93/94, anisotropyMap transform at 95/96, thickness payload at 97, and thicknessMap transform at 98/99) and routes every material texture sample in surface + attenuation shaders through manual repeat/clamp/mirror wrapping (`materialsTexture.ts`, `material_struct.glsl.js`, `get_surface_record_function.glsl.js`, `attenuate_hit_function.glsl.js`). ✅ THIRD SLICE walkaround atlas-backed material maps, including `baseColorMap`, `normalMap`, `roughnessMap`, `metallicMap`, `aoMap`, `alphaMap`, `emissiveMap`, `transmissionMap`, `lightMap`, extension-lobe maps, and `thicknessMap`, apply wrapS/wrapT in shade/traversal paths. | API no longer drops sampler semantics and `textureDecodeReport` exposes wrap modes. Runtime parity for clamp/mirror/repeat is closed on pt-webgpu, pt-webgl2, and walkaround's current atlas-backed map slices; unsupported bump/displacement policy rows remain separate. |
| ~~Mipmaps (ADDED 2026-06-12 — verified hole)~~ ✅ pt-webgpu CLOSED (2026-06-13) | pt-webgpu raw-data upload normalizes 1/2/3/4-channel 8-bit, Float32/Float64, and 16/32-bit numeric `{data,width,height}` payloads to explicit RGBA8 rows and warns/leaves the layer black for unsupported typed-array shapes. Full-tier material texture arrays now allocate a complete mip chain, generate lower levels with a WebGPU render pass for every sRGB/linear array layer, and the WGSL material sampler uses an explicit geometric LOD estimate (`textureNumLevels`, triangle UV density, projected world area, camera distance) instead of hard-coding `textureSampleLevel(..., 0.0)`. Tests: `materialTextureArray.test.ts`, `materialTextures.test.ts`, `wgslContract.test.ts`; shader-gate compiles `pt-webgpu/trace-full-*`. | pt-webgpu minification no longer has the level-0-only hole. Remaining texture sampler parity belongs to walkaround's atlas path and renderer-specific A/B validation. |

#### 1B — Feature report & planner (`featureReport.ts`)

**Already:** `analyzeGltfAsset`, `evaluateGltfBackendCompatibility`, `rankGltfBackends`, per-field ledger crosswalk.

**Still required:**

| Task | Code | Footgun |
|------|------|---------|
| ~~Source paths on every issue~~ ✅ DONE | `featureReport.ts`, `gltfAssetApi.test.ts` | `GltfCompatibilityIssue.path` is now required, analyzer source-path maps cover extensions/primitives/materials/scene cameras, and the test asserts every issue has a non-empty path including `materials[0].normalTexture`-style material paths. |
| ~~Scalar `KHR_materials_pbrSpecularGlossiness` scoring~~ ✅ DONE | `specularGlossinessMaterialCount` + compatibility issue `KHR_materials_pbrSpecularGlossiness` | Texture-alpha glossiness remains the next row |
| ~~Spec-gloss glossiness-alpha~~ ✅ DONE (explicit pre-decode downgrade + decoded roughness bake) | `materials.ts` imports `specularGlossinessTexture` RGB as `specularColorMap`; `featureReport.ts` emits `KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture.glossinessAlpha=approximate` with source path for planner/strict modes; `decodeSceneTextures()` / `loadGltfAndDecodeTextures()` now bake alpha glossiness into a generated linear `roughnessMap` for both `cpu-linear` and `webgpu` texture targets; `loadGltfForEngine(..., compatibilityMode:'reject-degraded')` no longer keeps this specific alpha issue active after that bake succeeds, and best-effort bridge tests verify the engine receives the decoded scene with the generated `roughnessMap`. Tests: `gltfExtensionPolicy.test.ts`, `gltfAssetApi.test.ts`. | No silent roughness-map lie: scalar `glossinessFactor` drives roughness before decode; host-supplied pixel decode closes the per-pixel legacy glossiness path. The broader archived spec-gloss model conversion remains approximate unless callers accept it. |
| ~~Morph `TANGENT` contract policy~~ ✅ DONE/approx | `SkinnedMeshPrimitive.morphTargetTangents`, `skinSolver.ts`, `gltfToScene.ts`, `sceneController.ts`, `featureReport.ts`, `gltfModesMorphsAnimations.test.ts`, `gltfAssetApi.test.ts`, `sceneController.test.ts`, `skinnedMeshIngestion.test.ts`, `pt-webgl2/src/scene/solveSkinPrimitives.ts`, `skinSolve.test.ts` | glTF morph-target TANGENT VEC3 deltas are preserved on the core primitive contract and the shared CPU skin solver now applies them to solved tangent streams when rest tangents exist. `GltfSceneController` now carries solved tangents in runtime morph and skin animation patches, and pt-webgl2 preserves posed tangents during skinned ingestion instead of dropping them before attribute packing. Compatibility remains `approximate` because GPU-native tangent skinning and broader backend evidence are still validation/promotion work rather than full native compute-kernel coverage. |
| ~~Morph `TEXCOORD_0/1` contract policy~~ ✅ DONE (high-UV remap follow-up 2026-06-19) | `SkinnedMeshPrimitive.morphTargetUvs` / `.morphTargetUv1s`, `skinSolver.ts`, `gltfToScene.ts`, `featureReport.ts`, `sceneController.ts`, `gltfModesMorphsAnimations.test.ts`, `gltfAssetApi.test.ts`, `skinnedMeshIngestion.test.ts`, `pt-webgl2/src/scene/solveSkinPrimitives.ts`, `skinSolve.test.ts`, `mutationMatrix.test.ts` | glTF morph-target UV deltas for the two UV lanes the core Scene contract actually carries now import, preflight as supported when the matching base UV stream exists, blend through the shared CPU skin solver, propagate through animation patches, and reach pt-webgpu/pt-webgl2/walkaround attribute uploads. The 2026-06-19 follow-up makes this source-truthful for high static UV remaps too: when a primitive's material losslessly remaps a single high `TEXCOORD_N` into core `uv1`, morph deltas for that same `TEXCOORD_N` feed `.morphTargetUv1s`; stale `TEXCOORD_1` deltas are not paired with a `uv1` stream sourced from `TEXCOORD_N`. Remaining unsupported cases are explicit: missing base UV streams, conflicting/multiple high UV sets, or high morph UV lanes that are not the semantic currently assigned to core `uv1` still emit `ignored-morph-target-texcoord` diagnostics plus `morphTargetTexcoords=unsupported` compatibility issues. |
| ~~Cameras~~ ✅ DONE (2026-06-16 strict-mode alignment) | `featureReport.ts`, `gltfAssetApi.test.ts` | `sceneGraph.cameraPaths` records `cameras[n]`; compatibility now emits `scene:cameras=approximate` to match the importer’s `ignored-camera` diagnostic. `reject-unsupported` accepts otherwise renderable assets with glTF cameras because Vitrum cameras are supplied through `FrameInput`; `reject-degraded` still rejects before engine construction when callers require zero ignored scene features. |
| ~~Double-sided~~ ✅ DONE | `featureReport.ts`, `gltfAssetApi.test.ts` | Compatibility emits `material:doubleSided=approximate` at `materials[n].doubleSided`; raw data remains preserved in material extensions, but planner now surfaces the lack of first-class double-sided/backface-normal semantics. |

#### 1C — Import (`gltfToScene.ts`, `materials.ts`, `accessors.ts`)

**Closed:** strip/fan triangulation, morph POSITION/NORMAL, animations, bound skins, punctual lights, KHR material extensions, `resolveTextureRef` UV/transform. Skin attributes now match importer semantics in compatibility preflight: `JOINTS_0` / `WEIGHTS_0` only predict `skinned-mesh` when a node binds `skin` and both streams are present; unbound or incomplete skin streams are structured unsupported issues and best-effort static-mesh fallbacks. 2026-06-17 proof follow-up: `pt/gltf-triangle-strip-fan` now imports `TRIANGLE_STRIP` and `TRIANGLE_FAN` through `loadGltfForEngine()`, asserts the exact generated triangle-list indices, rejects unexpected topology diagnostics, renders on the adapter, and compares against `tools/reference-renders/gltf-triangle-topology-behavioral/pt-gltf-triangle-strip-fan.png`; `npm run gltf-topology-proof-check` now verifies the lane's manifest, proof metadata, thresholds, and PNG presence.

**Still required:**

| Task | Code | Footgun |
|------|------|---------|
| ~~**GLTF-01** skinned-node bind space~~ ✅ CLOSED 2026-06-15 | `gltfToScene.ts` now stores glTF bones as mesh-node-local matrices (`inverse(meshWorld) · jointWorld`) and no longer emits the incorrect `bindMatrix=worldMat` / `bindMatrixInverse` sandwich. `sceneController.ts` applies the same conversion for animated joint patches, so rest pose and runtime animation share one skinning-space convention. `SkinnedMeshPrimitive` docs now describe skinning-space bones instead of overclaiming world-space-only bones. | `gltfAdapter.test.ts` now imports translated and translated+scaled skinned nodes, runs `solveSkin`, applies the primitive transform exactly as backends do, and asserts the world output is not double-translated. `sceneController.test.ts` now animates a joint under a translated skinned mesh node and pins mesh-local patch positions plus the final world sanity check. Focused/full glTF adapter tests and typecheck passed. |
| ~~**CORE-01** CUBICSPLINE quats~~ ✅ LANDED | `sampleAnimationClip` normalizes LINEAR/STEP/clamped/CUBICSPLINE rotations | — |
| ~~Generate tangents when missing~~ ✅ DONE | `gltfToScene.ts`, `tangents.ts`, `gltfAdapter.test.ts` | Normal/bump/clearcoat-normal mapped primitives now synthesize xyzw tangents from POSITION/NORMAL/TEXCOORD_0 when authored TANGENT is absent, and preserve authored tangents unchanged. 2026-06-15 hygiene pass: `tangents.ts` no longer calls the fallback MikkTSpace; the header now states the actual Gram-Schmidt accumulate + orthonormalize algorithm. 2026-06-19 hardening: malformed authored TANGENT accessors are dropped with `invalid-primitive-attribute` and tangent generation now returns `tangent-generation-failed` when every UV triangle is degenerate instead of reporting a clean generated tangent frame. |
| ~~`COLOR_0` vertex colors~~ ✅ DONE for adapter + planner + pt-webgl2 + pt-webgpu full + walkaround approximate; pt-webgpu-lite constant-RGB bake | `gltfToScene.ts` imports `COLOR_0`; `featureReport.ts` records source paths and compatibility; pt-webgl2 threads merged vertex colors into `attributesArray` layer 3 and enables the GLSL `material.vertexColors` multiply for affected material slots. pt-webgpu full now packs rgba vertex colors through shared-bvh, binds them at group(3)/binding(11), and multiplies baseColor plus alpha pass-through in the full-tier material paths. walkaround-hybrid now packs merged RGBA vertex colors through shared-bvh/world-space merge, uploads them as a scene texture, and multiplies visible baseColor plus traversal alpha coverage in the atlas-backed material path. pt-webgpu-lite now preserves the common primitive-constant RGB `COLOR_0` case by baking the multiplier into the merged material `baseColor` without adding a ninth storage buffer; non-constant or alpha-bearing `COLOR_0` remains a structured `pt-webgpu.lite-unsupported-vertex-colors` warning / compatibility issue because the lite shader layout intentionally omits the full-tier per-vertex color binding. 2026-06-16 follow-up: secondary vertex color sets (`COLOR_1+`) now emit `ignored-vertex-color-set` import diagnostics and unsupported planner issues instead of being silently ignored. 2026-06-19 hardening: malformed `COLOR_0` accessors that are not VEC3/VEC4 or do not match the primitive vertex count are dropped with `invalid-primitive-attribute` instead of forwarding invalid color buffers into core. | glTF vertex color × baseColor is native on pt-webgl2 and pt-webgpu full; pt-webgpu-lite is exact for primitive-constant RGB tints and explicit fallback for arbitrary per-vertex/alpha color; walkaround is `approximate` because realtime GI reservoirs are not fully path-tracer-equivalent secondary material transport. Multiple vertex-color sets remain unsupported but are now machine-readable compatibility issues. |
| ~~Direct pt-webgpu-lite bridge targeting~~ ✅ DONE | `gltf-adapter/src/assetLoader.ts`, `gltf-adapter/src/engineBridge.ts`; `gltfAssetApi.test.ts` | Direct adapter callers can pass `backend:"pt-webgpu-lite"` to validate against the lite profile while the injected engine factory still receives the real backend id `pt-webgpu`. Adapter-only hosts that select `backend:"pt-webgpu"` can now also pass `runtimeProfile:"pt-webgpu-lite"` after device negotiation. Strict loads reject lite-unsupported material-map rows before constructing an engine; for `COLOR_0`, the generic JSON-only planner stays conservative, then `loadGltfAsset()` clears the lite `vertexColors` blocker only after import proves every colored primitive is primitive-constant RGB with alpha 1, matching the renderer-side material bake. Nonconstant or alpha-bearing `COLOR_0` remains rejected under strict lite modes. Cross-backend runtime profiles throw instead of silently validating the wrong row. |
| ~~Required extension import errors~~ ✅ DONE | `gltfToScene.ts`; `gltfAdapter.test.ts` | Unknown `extensionsRequired[]` entries now throw `GltfImportError` with an `unsupported-required-extension` diagnostic and exact `extensionsRequired[i]` path, preserving the hard fail while making the reason machine-readable. |
| ~~Sparse accessors~~ ✅ DONE | `accessors.ts`, `accessors.test.ts`, `gltfAdapter.test.ts`, `gltfModesMorphsAnimations.test.ts` | Sparse patches now have focused coverage for base+pure-sparse accessors, unsigned-byte sparse indices, byte offsets, strided base data, normalized values, integer index accessors, and out-of-range/invalid sparse-index diagnostics. |
| ~~Point/line modes~~ ✅ DONE | `gltfToScene.ts` imports `POINTS`, `LINES`, `LINE_LOOP`, and `LINE_STRIP` as fallback-generated triangle mesh geometry; `featureReport.ts` reports `fallbackGeneratedModes`; `gltfPointLinePrimitivePolicy.test.ts` pins compatibility + import diagnostics | Approximation is explicit: `reject-unsupported` accepts, `reject-degraded` rejects; native point/line primitive kinds remain optional future promotion |

#### 1D — Runtime controller (`sceneController.ts`)

**Already:** `seek`, `advance`, skin bones via `solveSkin`, morph weights, `updatePrimitive` with `setScene` fallback.

**Still required:**

| Task | Code | Plug-in | Footgun |
|------|------|---------|---------|
| ~~Multi-clip blend~~ ✅ DONE | `GltfSceneController.blend(clips, weights, time, { times?, loop?, engine? })` samples each clip, normalizes positive weights, blends channels per node/path (including hemisphere-corrected rotation nlerp), then runs the same transform/morph/skin patch path as single-clip animation. Tests: `sceneController.test.ts` transform blend + morph-before-skin-solve blend. | Morph blending now happens before `solveSkin`; sparse clips blend per authored channel so unrelated channels are not damped. |
| ~~`KHR_materials_variants` at runtime~~ ✅ DONE | `GltfSceneController.setVariant(name/index/undefined)` uses importer-emitted `materialVariantBindings` + `convertedMaterials` to patch only affected primitive materials; `loadGltfForEngine()` now forwards that metadata into bridge-created controllers; decoded loads now decode the inactive material table before controller creation; `updatePrimitive` fast path falls back to `setScene(nextScene)` on rejection; `resetPose()` preserves the active material variant. Tests: `sceneController.test.ts` runtime switch + fallback, `gltfAssetApi.test.ts` bridge-created controller switch + decoded inactive variant texture, `gltfExtensionPolicy.test.ts` import-time selection. | Variant switch now invalidates material fast-path caches by issuing material patches through the same engine patch channel as animation updates, without reintroducing raw texture handles after a decoded load. |
| ~~Engine attach API~~ ✅ DONE | `controller.attachEngine(engine, { setScene })` exists and `loadGltfForEngine` attaches after load | Use `attachScene:false` / `setScene:false` when the host already set the scene |
| ~~Patch routing per backend~~ ✅ DONE | `sceneController.ts` applies `patchPrimitiveInScene` first, tries `updatePrimitive`, and falls back to one `setScene(nextScene)` with a controller warning if an incremental patch throws; `sceneController.test.ts` pins the throw→fallback path. | — | Partial patch desync is closed for the glTF controller path |

#### 1E — Engine bridge (`engineBridge.ts`)

**Already:** `loadGltfForEngine`, `compatibilityMode`, factory injection.

**Still required:**

| Task | Code | Footgun |
|------|------|---------|
| ~~`@vitrum/engine/gltf` re-export~~ ✅ DONE | `packages/engine/src/gltf.ts` exports the adapter bridge and adds `loadGltfWithEngine()`, which injects `createEngine` and maps the glTF planner's selected backend to the matching engine preference. `packages/engine/package.json` exposes `./gltf` and declares the adapter dependency. Test: `gltfSubpathExport.test.ts`. | Adapter still owns loading/planning/controller logic; the engine subpath is a thin one-import DX wrapper. |
| ~~Pass `decodeImage` + `dracoDecode` + `meshoptDecode` through bridge~~ ✅ DONE | `LoadGltfForEngineOptions` extends `LoadGltfAssetOptions`; `loadGltfForEngine` passes options through `loadGltfAsset` | Bridge without hooks still fails required compressed assets, but now through the intended hook contract |
| ~~Return `textureDecodeReport`~~ ✅ DONE | `GltfAssetResult` + `GltfForEngineResult` expose the scene-level report; tests pin raw-image fallback entries. | Host knows before first frame which maps are raw/opaque/CPU-readable/ignored. |

---

### Phase 2 — PT backends (pt-webgl2 + pt-webgpu full) — material & integrator 100%

Required for **arbitrary glTF** on fidelity backends. Walkaround is Phase 3.

#### 2A — P0 correctness (do before fidelity promotion)

> **2026-06-12 reconciliation: ALL rows in this table are ✅ LANDED on `main`**
> (waves `00047313`..`f1b1dd79` + verified on disk). Kept for audit history only
> — do not re-execute.

| ID | Status | Evidence |
|----|--------|----------|
| PTWG-01 | ✅ LANDED | `state==='error'` blocks at `pt-webgpu/src/index.ts:783` |
| PTWG-02 | ✅ LANDED | old-OR-new implicit-emitter repack in `sceneMutationRouter.ts` |
| PTWG-03–04 | ✅ LANDED | `lightSelectInvPdf` all source kinds (`sppmBindings.wgsl.ts:394+`); flux oracle PASSES (energy conserved ±3%) |
| PTWG-05 | ✅ LANDED | Abbe/spectralMinMu lanes split |
| WEBGL2-01 | ✅ LANDED | authored tangent XYZW + nonzero fallback handedness (`attributesTextureArray.ts:252`); CPU-solved skinned tangents are preserved through `solveSkinPrimitives()` |
| H49 | ✅ STRUCK — was WRONG, not stale | `specularColor`/`specularIntensity` + maps ALREADY packed at `materialsTexture.ts:168-232`; ledger grades native |

#### 2B — Material packing gaps → ledger `native`

**pt-webgl2** (`materialsTexture.ts` + GLSL): scalar `anisotropy` / `anisotropyRotation` plus `anisotropyMap` are now native; `thicknessMap` is approximate via a KHR volume thickness-texture clamp; `displacement*` remains unsupported.

| Field | Work |
|-------|------|
| ~~`thicknessMap`~~ ✅ DONE/approx (2026-06-13) | Packed into the material atlas/stride with scalar `thickness`, per-map UV channel, KHR texture transform, and wrap modes. GLSL samples the KHR volume G channel and clamps closed-surface Beer-Lambert attenuation distance to `thicknessFactor * thicknessTexture.g`. Ledger grade is `approximate`, not `native`, because pt-webgl2 still uses closed-surface traversal rather than exact thin-shell volume integration. |
| ~~`anisotropy` / `anisotropyRotation` / `anisotropyMap`~~ ✅ DONE (2026-06-13) | Scalar strength/rotation pack into reserved lanes `s11.a` / `s17.b`; `anisotropyMap` packs into `s6.b`, transform texels `95..96`, UV bit 19, and the final wrap-mode pair. GLSL decodes the map and samples RG/B according to KHR_materials_anisotropy (`B` strength, `RG` rotation offset), feeding anisotropic GGX sampling/eval/PDF in `bsdf_functions.glsl.js`. Tests: `materialsTexture.test.ts`, `materialStrideParity.test.ts`, `engineContract.test.ts`, core ledger contract. |

**pt-webgpu** (`materialPacking.ts`, `materialTextures.ts`, `material.wgsl.ts`):

| Field | Work | Footgun |
|-------|------|---------|
| ~~`normalScale` / `normalMap` tangent basis~~ ✅ DONE | `normalScale` is packed in the group-3 material texture descriptor and applied in `applyNormalMap`; full-tier pt-webgpu now uploads per-vertex authored/generated tangent.xyzw at group(3)/binding(10), interpolates handedness, and falls back to UV-gradient derivation only when tangent data is absent. The `normalMap` ledger row is promoted to `native`. Tests: `materialTextures.test.ts`, `scenePack.test.ts`, `sharedPipelineLayout.test.ts`, `engineContract.test.ts`. | Legacy tangentless scenes still use the fallback; walkaround tangent-space maps are separate Phase 3D work. |
| ~~`roughnessMap` / `metallicMap` distinct handles~~ ✅ DONE (2026-06-13) | `materialTextures.ts` expands the full-tier descriptor stride to 75 vec4s, preserving canonical combined glTF metallicRoughness textures by pointing both slots at one layer while also packing distinct authored roughness and metallic handles into separate linear-array slots. `material.wgsl.ts` samples roughness from the G channel with its own UV/wrap metadata and metallic from the B channel with its own UV/wrap metadata. The promise ledger now marks both rows `native`. Tests: `materialTextures.test.ts`, `h51WarnCoercions.test.ts`, `wgslContract.test.ts`. | Full-tier megakernel coverage; lite tier still rejects group-3 material texture features through its tier-specific capability row. |
| ~~`transmissionMap`, `alphaMap`~~ ✅ DONE | `materialTextures.ts` packs both maps into the linear texture array; `material.wgsl.ts` samples alpha coverage in alphaMode paths and transmission R in the full-tier prologue. Tests: `materialTextures.test.ts`, `wgslContract.test.ts`, `wgslLiteContract.test.ts`. | Texture-map handle changes remain full-rebuild material patches; that is mutation performance, not a missing render path |
| ~~`clearcoatMap`, `clearcoatRoughnessMap`, `clearcoatNormalMap`, `sheen*Map`, `iridescence*Map`, `specular*Map`~~ ✅ DONE/approx | `materialTextures.ts` packs these extension-lobe maps into the correct sRGB/linear arrays inside the 75-vec4 full-tier descriptor stride, records per-map texCoord / KHR_texture_transform / wrap / UV-fit data, and `material.wgsl.ts` samples the glTF channels. `shadePrologue.wgsl.ts` now also samples `clearcoatNormalMap` / `clearcoatNormalScale` and the main megakernel threads the sampled clearcoat normal through clearcoat BRDF/PDF/source-sampler paths. Follow-up: BDPT light-path vertices now have a fifth scratch row carrying hit-local `triIndex`/`baryVW`/`instanceIndex` plus a front-face bit, so surface light vertices sample baseColor/vertexColor/AO/ORM/transmission/normal/bump/clearcoat-normal/extension/specular/anisotropy maps and apply layer tint/roughness, thin-film reflect tint, Cauchy IOR, and spectral reflectance scalar before scatter/connection BRDF/PDF evaluation. ReSTIR-PT now stores `clearcoatNormalV` in `ReservoirPTHero` and uses clearcoat-normal-aware target/resolve/source-PDF/suffix-Lo BRDF paths. 2026-06-16 follow-ups: map-free scalar `clearcoat` / `clearcoatRoughness`, map-free `sheen` / `sheenColor` / `sheenRoughness`, map-free scalar `iridescence`, and path-replay `vec2` `iridescenceThicknessRange` inverse fitting through map-free or readable `iridescenceThicknessMap` ranges are now covered in the scoped inverse domain. 2026-06-19 clearcoat-normal adjoint follow-up: `adjointPass.wgsl.ts` now replays `clearcoatNormalMap` with its own descriptor UV/wrap/scale metadata, evaluates clearcoat scalar gradients against that lobe normal, and exposes `clearcoatNormalScale` path replay for frozen direct lighting. Tests: `materialTextures.test.ts`, `wgslContract.test.ts`, `liteTierCapabilities.test.ts`, `bdptGlossyLightSubpath.test.ts`, `reservoirPtHeroLayout.test.ts`, `restirPtReuseContract.test.ts`, `brdfAdjoint.test.ts`, `adjointHarness.test.ts`, `inverseSession.test.ts`; shader-gate coverage remains required for shader edits. | Still `approximate` in the promise ledger because texture-pixel/UV gradients, remaining anisotropy inverse-adjoint promotion evidence, and material-lobe reference A/B are still pending. |
| ~~`specularIntensity`, `specularColor`~~ ✅ DONE/approx | Packed in material vec4 #27 and consumed by ordinary PT BRDF/PDF paths, MNEE/SPPM receiver paths, BDPT eye/light connection and light-subpath surface scattering, ReSTIR-PT visible-domain reservoir/resolve payloads, and the path-replay adjoint direct-light fast path. The adjoint pass now reads vec4 #27, differentiates KHR_materials_specular dielectric F0 for `specularColor` / `specularIntensity`, and keeps `baseColor` / `roughness` partials aligned with non-default specular controls. Tests: `scenePack.materials.test.ts`, `wgslContract.test.ts`, `wgslLiteContract.test.ts`, `bdptGlossyLightSubpath.test.ts`, `brdfAdjoint.test.ts`, `adjointHarness.test.ts`, `inverseSession.test.ts`; shader-gate compiles production shaders. | Still `approximate` in `BACKEND_PROMISE_LEDGER` until remaining specialty reference/furnace proofs close; path-replay still degrades to finite difference for maps, transmission, layered/volume/spectral fields, alpha visibility, environment light terms, and unsupported extension lobes. |
| ~~Per-map UV transform~~ ✅ DONE for pt-webgpu consumed maps | `materialTextures.ts` now packs two UV metadata vec4s per consumed map (`texCoord`, offset, scale, rotation), and `material.wgsl.ts` passes map-specific slots for baseColor/emissive/normal/roughness/metallic/AO/light/bump/anisotropy/alpha/transmission/thickness plus the extension-lobe maps above. ReSTIR-PT suffix/visible material decode and BDPT surface light vertices now consume hit-local material payloads for the mapped paths they evaluate. Tests: `materialTextures.test.ts`, `wgslContract.test.ts`, `restirPtReuseContract.test.ts`, `bdptGlossyLightSubpath.test.ts`; `shader-gate` compiles the full-tier/ReSTIR-PT/BDPT traces. | Full-tier coverage; inverse/adjoint gradients and proof/A-B gates still keep extension-lobe rows approximate. |
| ~~`thickness` scalar / `thicknessMap`~~ ✅ DONE/approx (2026-06-15) | `materialPacking.ts` appends material vec4 #28 (`thickness`, `hasVolumeThickness`), `materialTextures.ts` appends full-tier descriptor vec4s #71..#74 for KHR volume `thicknessMap`, and `material.wgsl.ts` samples G then applies `thicknessFactor * thicknessTexture.g` as a Beer-Lambert path-length clamp. Lite consumes scalar `thickness` but still reports `thicknessMap` unsupported because it has no group-3 material texture bindings. Walkaround parity is closed separately by atlas-packing readable `thicknessMap` handles and exponentiating the pre-baked Beer tint by `thicknessTexture.g` in shade/transmitted-GI/tinted-visibility paths. Tests: `scenePack.materials.test.ts`, `materialTextures.test.ts`, `wgslContract.test.ts`, `wgslLiteContract.test.ts`, `liteTierCapabilities.test.ts`, `mutationDesyncs.test.ts`, `materialTextureAtlas.test.ts`, `consumedMaterialFields.test.ts`. | Ledger grade is `approximate`, not `native`: both PT and walkaround paths use closed-surface attenuation/tint approximations rather than exact glTF thin-shell volume integration. |
| ~~`shadingModel` branch verification~~ ✅ DONE | Packed/consumed as terminal unlit branch in pt-webgpu; ledger intentionally remains `approximate`, not `native` | Native would require different semantics (for example emissive-light participation), so do not over-promote |

**Plug-in pattern for new map:**
1. Add to `materialTextures.ts` collector.
2. Add binding index in `material.wgsl.ts` `materialTexDescriptors`.
3. Sample in `shadePrologue.wgsl.ts` / `bsdf.wgsl.ts`.
4. Update `PT_WEBGPU_MATERIALS` row.
5. Add `scenePack.materials.test.ts` + wgslContract field pin.
6. Reject from `incrementalPatch.ts` material fast-path if map handles change (already pattern for texture maps).

#### 2C — PTWG-MAT-01 integrator audit (mandatory for extension lobes)

> **SCOPE WIDENED (2026-06-12) / shortcut class CLOSED (2026-06-15):** the audit
> was not extension-lobes-only. It also closed the **grayscale single-directional
> shortcut class** across MNEE cone-search and BDPT bounce-0: both now loop packed
> N-directional RGB records instead of the retired mean-gray mirrored directional.
> ✅ **in-medium NEE CLOSED (2026-06-15):** the volumetric random-walk block now
> loops `directionalLights[]`, preserves RGB irradiance, honors the directional
> `castShadow:false` sign bit, and `volumetricSss.test.ts` pins the no-scalar
> `params.lightDir.w` regression. ✅ **SPPM directional emitter was already
> closed:** `sppmPhotonEmission.test.ts` pins packed N-directional RGB photon
> emission. ✅ **ReSTIR-PT `rptDirectAtVertex` CLOSED (2026-06-15):** suffix
> direct lighting now loops packed N-directional RGB records and honors the
> directional shadow-disable sign bit; `restirPtReuseContract.test.ts` pins the
> no-`params.lightDir.w` regression. ✅ **MNEE cone-search fallback CLOSED
> (2026-06-15):** the legacy transmissive cone-search approximation now loops
> packed N-directional RGB records and honors the directional shadow-disable
> sign bit; `mneeRefractionCaustic.test.ts` pins the no-scalar regression.
> ✅ **BDPT bounce-0 directional shortcut CLOSED (2026-06-15):** BDPT light
> subpath emitter count/power/write now loops packed N-directional RGB records,
> and pseudo-distant directional/environment vertices use packed
> `sceneCenterX/Y/Z + sceneRadius` instead of `emitPos = -lightDir * 50.0`.
> `bdptGlossyLightSubpath.test.ts`, `bdptEmitterPickCpu.test.ts`,
> `scenePack.test.ts`, and FrameParams layout tests pin the no-scalar/no-fixed
> radius path. ✅ **environment:'none' phantom skylight CLOSED (2026-06-12):**
> full + lite no-map env lookups now return black radiance and zero env pdf
> (`connect.wgsl.ts`, `connectLite.wgsl.ts`); procedural sky remains lit through
> the CPU-baked HDRI path, and `environmentPacking.test.ts` pins kind:none.

Audit **every** `evaluateBrdf` / `brdfDirectionalPdf` call site — glTF extension lobes must match across paths:

| Path | File | Status |
|------|------|--------|
| Eye path NEE | `kernel.wgsl.ts` / `kernelLite.wgsl.ts` | ✅ direct-light NEE, BSDF connection helpers, `sampleNextBounceDirection`, and BDPT eye-stack forward/reverse PDFs now use a normalized sampled-density helper for base/clearcoat/sheen. 2026-06-15 follow-up: transmissive dielectric materials now use that same source-lobe mixture too — the base lobe remains the Fresnel reflect/refract partition, while clearcoat and sheen are sampled/PDFed as same-side reflection lobes on glass. |
| BSDF connections | `connect.wgsl.ts`, `connectLite.wgsl.ts` | ✅ local helper propagation closed (area/env full-tier; env lite; area-lite paired MIS closed by PTWG-LITE-01 via packed `liteLightTex` ray intersections) |
| BDPT | `bdptConnection.wgsl.ts`, `bdptLightSubpath.wgsl.ts` | ✅ eye↔light connection uses full helpers; light-subpath scatter now reuses `sampleNextBounceDirection`, records clearcoat-normal-aware forward/reverse BSDF densities, carries row-4 hit-local material payloads plus a front-face side bit, and samples mapped base/vertexColor/AO/ORM/transmission/normal/bump/clearcoat-normal/extension/specular/anisotropy fields at surface light vertices. ✅ 2026-06-15 parity wave adds shade-prologue-equivalent layer tint/roughness, thin-film reflect tint, Cauchy IOR, and spectral reflectance scalar to that light-side material domain. 2026-06-19 proof hardening: `bdptConnectionMisFull.ts` now accepts a real light-side BSDF PDF callback for surface `L_c` connection-induced overrides, and `bdptConnectionMisFull.test.ts` pins the non-Lambertian path instead of relying on Lambertian-only CPU oracle wording. **Open:** independent radiometric/material-furnace oracle coverage. |
| SPPM / caustics | `caustic.wgsl.ts`, `sppmBindings.wgsl.ts` | ✅ receiver-side SPPM/caustic BRDF/PDF helper propagation closed; MNEE cone-vs-BSDF MIS now uses the sampled-density helper for the BRDF alternative. 2026-06-16 finite-area follow-up: rect/disc and mesh-area *reflection* MNEE now sample a finite emitter point, map its area PDF through `mneePdfJacobianDetAxes`, and MIS against the receiver BSDF PDF; finite-area refraction/slab MNEE remains a separate promotion/A-B item. |
| ReSTIR-PT | `restirPtProducer.wgsl.ts`, `restirPtCompose.wgsl.ts`, `reservoirPtHero.wgsl.ts`, `restirPtResolve.wgsl.ts` | ✅ producer direct/onward paths use full helpers; ✅ scalar-lobe reservoir payload/target/resolve now uses full visible-domain helpers, including `KHR_materials_specular` scalar colour/intensity and `clearcoatNormalV` (`ReservoirPTHero` 56-word layout); ✅ visible-vertex payload now mirrors the main shade prologue for alpha pass-through, baseColor/AO/roughness+metallic/normal/bump/clearcoat-normal/transmission/extension scalar maps, layer tint/roughness, thin-film, and spectral albedo; ✅ suffix/reconnection vertices now alpha-skip and decode the same hit-local material-map/layer/thin-film/spectral domain before Lo evaluation, using the mapped suffix normal for the reservoir geometry and mapped clearcoat normal for clearcoat BRDF/PDF paths; ✅ producer source sampler now samples a normalized base/clearcoat/sheen lobe mixture and stores the matching `pdfSrc`, with the clearcoat lobe sampled/PDFed against the mapped clearcoat normal; ✅ prefix-1 hybrid shift now consumes the previously reserved `hybridShiftPdf`/`hybridJacCache` lanes so temporal/spatial reuse applies the source/target BSDF replay-pdf ratio instead of geometry-only reuse. 2026-06-16 proof wave: `tools/radiometric-ab/ab-restir-pt-specialty.mjs` + `restirPtSpecialtyReference.test.ts` pin the one-sample producer/finalize/resolve identity for clearcoat, sheen, iridescence, and anisotropy payloads. |
| Adjoint | `adjointPass.wgsl.ts`, `pathTraceAdjoint.wgsl.ts`, `inverseSession.ts` | ◻/✅ Scoped path replay now consumes `req.samples`, replays the frozen inverse sample sequence, and covers opaque direct-light/unlit-primary fields including baseColor, roughness, metallic, `aoMapIntensity`, camera-visible `lightMapIntensity`, `envMapIntensity` for direct HDRI/procedural-sky NEE, emissive/emissiveIntensity, specular, clearcoat, `clearcoatNormalScale`, sheen, iridescence, `iridescenceThicknessRange` (map-free and readable thickness-map texel chained), anisotropy, their documented local map factors, deterministic direct emitter `color`/`intensity`, and delta/soft-sun directional plus point/spot/stochastic area-measure rect/disc/mesh-area direct lights. 2026-06-16 soft-sun follow-up mirrors the forward directional cone sample in `adjointPass.wgsl.ts` and keeps compatible soft directional material/emitter targets on path replay. 2026-06-18 environment-NEE follow-up adds the same HDRI/procedural-sky CDF replay shape used by the forward direct-light branch (bindings 20/21, texel `.w` solid-angle PDF, rotationY convention preserved), scales that replay by the hit material's descriptor `[4].w` `envMapIntensity`, and exposes a scoped path-replay gradient for that direct environment-lighting scale. 2026-06-19 normal/bump/clearcoat-normal follow-up: top-level opaque `normalMap`, `bumpMap`, and `clearcoatNormalMap` are replayed through the existing linear texture-array descriptors and UV transforms, prefer the forward shader's authored/generated tangent.xyzw handedness before UV-gradient fallback, direct-light BRDF/emitter gradients evaluate at the perturbed shading normal, clearcoat scalar gradients evaluate against the independent clearcoat lobe normal, and `normalScale` / `bumpScale` / `clearcoatNormalScale` now have scoped replay-local gradients for frozen direct lighting. Current follow-up: `adjointPass.wgsl.ts` now computes replay-local central differences of the full normal stack, so `normalScale` stays on path replay through bump/clearcoat-normal tangent frames and `bumpScale` stays on path replay through clearcoat-normal tangent frames instead of downgrading those nested-map cases to render-level finite difference. Finite-difference inverse sessions still admit transport/visibility/geometry controls (`transmission`, `thickness`, `attenuationColor`, `attenuationDistance`, `dispersionAbbeNumber`, `scatteringCoefficient`, `scatteringAnisotropy`, `scatteringCoefficientRGB`, `opacity`, `alphaCutoff`) with structured path-replay downgrades. 2026-06-17 texel-emission truthfulness follow-up: explicit `mesh-area` emitter `color`/`intensity` targets with readable `emissiveMap` stay on path replay when the authored color/intensity denominators are nonzero, because the adjoint chains through packed per-triangle radiance. 2026-06-19 source-factor follow-up: the mesh-area packer now emits a `meshAreaLightSourceFactors` side buffer and the adjoint pass differentiates `color * intensity * sourceFactor`, so zero authored color channels no longer force a finite-difference downgrade. Current follow-up: mesh-area emitter inverse fits bind an adjoint-only replay stream that preserves zero-power explicit mesh triangles, so zero-intensity mapped emitter `intensity` targets stay on path replay without changing forward render packing; capped/reordered explicit mesh-area emitter `color`/`intensity` targets now stay on path replay too because the source-factor `.w` lane carries a stable owner slot independent of packed triangle order. Non-color/intensity emitter targets still stay on `path-replay-unsupported-emitter`. 2026-06-18 classification follow-up: scene/material blockers, including receiver-material blockers for emitter-parameter fits, now propagate the same reason-specific downgrade codes (`path-replay-unsupported-visibility`, `path-replay-unsupported-transport`, `path-replay-unsupported-normal`) for alpha coverage, transmission/volume/spectral/layered, and unsupported normal/visibility domains instead of collapsing those known tails into a generic material/receiver warning. 2026-06-18 estimator-regime follow-up: the real engine now reports active BDPT, ReSTIR-PT reuse, and non-`none` caustic strategies to inverse sessions, and requested path replay downgrades with `path-replay-unsupported-render-regime` before using an adjoint that does not mirror those render contributions. Still OPEN for full-path parity: alpha-map/alpha-visibility adjoints, transmission/thickness/attenuation-color/attenuation-distance/dispersion/scattering/displacement, normal texture-pixel/UV gradients, environment BSDF-escape/indirect transport, layered/volume/spectral material domains, exact texel-PDF mesh emission promotion for emitter-parameter adjoints, forward light-selection MIS parity, indirect paths, GPU inverse-fit recaptures, and full material-lobe A/B proofs. |
| Present | `present.wgsl.ts` tonemap only — no BSDF | N/A |

**Footgun:** Fixing megakernel only used to leave BDPT/SPPM/ReSTIR-PT wrong for glTF clearcoat scenes on specialty paths. The clearcoat-normal map now reaches the megakernel, BDPT light-side surface vertices, and ReSTIR-PT reservoir/resolve/source-PDF/suffix-Lo paths; BDPT's light-side material domain now also applies layer tint/roughness, thin-film reflect tint, Cauchy IOR, and spectral reflectance scalar. ReSTIR-PT now has a deterministic CPU/static specialty-lobe identity fixture, scalar clearcoat/sheen/iridescence controls have scoped direct-light adjoint partials, iridescence thickness ranges have path-replay inverse fitting for both map-free and readable thickness-map cases, and top-level `normalMap` / `normalScale`, `bumpMap` / `bumpScale`, plus `clearcoatNormalMap` / `clearcoatNormalScale` now replay in the scoped direct-light adjoint path, including nested normal-stack scale derivatives. The remaining class still includes normal texture-pixel/UV gradients, anisotropy promotion evidence, plus GPU material-furnace/reference-render promotion, not just missed function calls.

#### 2D — pt-webgl2 scope gaps for arbitrary glTF

| Gap | Code | Footgun |
|-----|------|---------|
| ~~Analytic primitives~~ ✅ DONE/fallback-generated mesh | `PT_WEBGL2_SUPPORT` now advertises `analytic` plus the supported analytic shapes, `buildSceneTextures()` expands analytic primitives through `analyticPrimitiveToMesh()` before `partitionSceneBySupport`, and `engineContract.test.ts` pins both the capability row (`supportDetails.primitives.analytic === 'fallback-generated-mesh'`) and the runtime tessellation path. | pt-webgl2 does not traverse analytic primitives natively; it consumes generated triangle meshes. That is a truthful support grade, not a planner drop. |
| ~~Procedural sky~~ ✅ DONE/approx | pt-webgl2 now bakes `procedural-sky` through shared-samplers' Preetham equirect helper and feeds the existing HDRI/CDF path | Ledger grade is `approximate` for finite 256x128 bake resolution; glTF has no sky |
| ~~Procedural sky on PT~~ ✅ DONE/approx | Shared `bakePreethamSkyEquirect()` now feeds both pt-webgl2 and pt-webgpu | |
| Geometry/list mutations split: same-topology geometry/material/emitter/env native; topology/list fallback | ✅ PARTIAL CODE CLOSED — pt-webgl2 now fast-paths scalar material edits, analytic emitter edits, mesh-area emitter edits, and environment swaps by replacing only the affected scene textures (`mutateSceneTextures.ts`, `index.ts`, `engineContract.test.ts`). The public promise promotes `material`, `emitter`, and `environment` to `native`; mesh-area emitter changes refresh both the folded emissive material texture and mesh-light NEE texture without rebuilding BVH geometry. 2026-06-15 follow-up: material fast-path edits on a primitive referenced by a `mesh-area` emitter now preserve the folded emissive radiance in the GPU material texture, so scalar material edits no longer desync camera/path-hit emission from mesh-light NEE. 2026-06-16 follow-up: scalar material edits that create/remove ordinary emissive mesh lights now repack the implicit mesh-light texture and `uTotalEmissiveArea` without BVH rebuild. 2026-06-18 resource-churn follow-up: geometry-only primitive patches (`transform`, positions/normals/indices/uv/uv1/tangent/instance/skin/morph fields) were moved off full scene upload onto the shared skin/merge/BVH/attribute/mesh-light packers while preserving scalar material, explicit light, environment, and texture-atlas resources. 2026-06-19 vertex-color follow-up: `colors` patches use the same bounded geometry path and only refresh material data when the `vertexColorMaterialIds` set changes, so vertex-color edits no longer force full scene-texture upload solely because the material slot encodes color usage. 2026-06-19 refit/subimage follow-up: same-topology geometry edits now verify stable merged ranges/indices/material ids, refit retained BVH node bounds with `refitBvhBounds`, preserve BVH contents/index/material-index textures, and write BVH bounds, BVH positions, attributes, and vertex-color material flags with `texSubImage2D`/`texSubImage3D` into existing GL textures. Same-topology transform edits now allocate 0 textures and emit no fallback warning; vertex-color edits allocate 0 textures and emit no fallback warning (mesh-light null/add/remove remains a bounded optional texture replacement when the nullable/count shape changes). Same-day material-atlas follow-ups: material patches that retarget an already-resident atlas handle, change map UV/wrap/transform metadata, or remove a map now repack the material texture in place with `texSubImage2D` and emit no fallback warning; patches that introduce a new readable texture handle now rebuild only the material-map atlas texture plus material descriptor, leaving BVH/attributes/lights untouched; opaque/unreadable new handles emit `pt-webgl2.texture-unreadable` through `onWarning` and still avoid scene fallback. Later same-day atlas-capacity follow-up: atlas uploads retain a modest power-of-two layer capacity, and same-dimension material-map insertions that fit that capacity refresh existing atlas layers with `texSubImage3D` instead of allocating a new texture-array; larger-map dimension changes and exhausted-capacity growth now re-specify storage on the resident atlas texture with `texImage3D` instead of creating a replacement texture object, while first-atlas creation and atlas removal remain true create/drop cases. 2026-06-19 compaction follow-up: map removals/retargets now repack the next live atlas domain, shrink resident texture-array storage when the capacity tier drops, and emit `pt-webgl2.material-atlas-texture-refresh` with `reason: "capacity-compaction"` plus previous/next dimension/layer/capacity details. 2026-06-19 list/topology-resident follow-up: primitive add/remove and same-dimension topology/index patches now first CPU-pack the next geometry and, when BVH/index/position/material/attribute storage dimensions plus atlas/mesh-light residency permit it, rewrite existing BVH/material/attribute textures with `texSubImage2D`/`texSubImage3D` and emit no fallback warning; dimension-changing and storage-shape-changing topology edits still use the bounded texture-refresh fallback. Current follow-up: geometry-stable atlas-backed primitive-list edits now keep BVH/attribute resources resident across same-atlas, first-atlas, atlas-removal, and capacity-growth cases; the material descriptor texture is replaced only when its square dimension changes, the atlas texture is created/dropped/re-specified as the side resource requires, and `pt-webgl2.material-atlas-texture-refresh` reports `addPrimitive`/`removePrimitive` plus the concrete atlas reason. The runtime capability and promise ledger now grade pt-webgl2 `transform` and `positions` mutations as `native`, while topology/add/remove stay `fallback-rebuild`. Primitive add/remove no longer always allocates replacement scene textures, but arbitrary list edits still fall back after applying the same analytic-to-mesh fallback tessellation used by `setScene()` when geometry/BVH/attribute storage dimensions change. 2026-06-18 diagnostic follow-up: fallback geometry/animation/list paths include `fallbackReason` / `nativePatchMissing` details; material texture-map patch fallbacks should now be reserved for mixed/non-material patches rather than ordinary material-map retargets. | Remaining implementation tail before full mutation promotion: true list/topology splice/refit for dimension-changing edits; animation/controller geometry remains native only when the solved output preserves topology, otherwise it falls back to bounded rebuild/repack with structured diagnostics. |
| ~~No `setSize`~~ ✅ DONE (pt-webgl2) | `PTEngineWebGL2.setSize()` stores explicit canvas size, reallocates existing render targets, and resets accumulation without scene/BVH repack | `pt-webgpu` still honors `FrameInput.viewport` per frame and omits `setSize`; pt-webgl2 ledger grades resize `native` |
| ~~Denoiser~~ ✅ DONE | `oidn-final` is now an in-engine asynchronous final-pass path on pt-webgl2 (`OIDNFinalDispatcher`, aux readback, `oidnFinal.test.ts`, `engineContract.test.ts`). | Non-OIDN realtime denoisers remain unsupported on converged pt-webgl2; hosts must provide `oidn.modelUrl` + optional bridge/runtime. |
| ~~Caustics~~ ✅ DOC/TEST CLOSED | pt-webgl2 `manifold-nee` remains a deterministic refraction-walk heuristic, not Newton-solve MNEE (`options.ts`, README, core factory note). `renderer-fidelity-matrix.md` grades pt-webgl2 caustics `approximate`, and `engineContract.test.ts` pins that the backend surfaces caustic choices only through `capabilities.causticStrategy` rather than advertising native MNEE feature support. | Keep pt-webgpu `manifold-nee` as the validated reference; pt-webgl2 caustics stay approximate unless a real MNEE port lands. |

#### 2E — pt-webgpu lite tier policy

**For arbitrary glTF 100%:** lite is **not** a target. Policy/code is closed below; keep these rows as regression gates:

| Task | File | Behavior |
|------|------|----------|
| ~~`loadGltfWithEngine` enforces runtime pt-webgpu profile rows for strict modes~~ ✅ DONE (2026-06-15 follow-up; existing-engine path closed 2026-06-16; adapter-only explicit tier closed later 2026-06-16; best-effort reporting closed in current wave; strict degraded import diagnostics closed 2026-06-17) | `packages/engine/src/gltf.ts`, `gltfStrictPtWebgpuTier.test.ts`, `packages/gltf-adapter/src/engineBridge.ts`, `gltfAssetApi.test.ts` | The `@vitrum/engine/gltf` one-call wrapper probes the adapter profile before construction/attachment and evaluates the actual runtime compatibility row (`pt-webgpu` full vs `pt-webgpu-lite`) for both `reject-unsupported` and `reject-degraded`. Lite assets with unsupported rows now reject under `reject-unsupported`; degraded-but-supported lite assets pass that mode but reject under `reject-degraded`; best-effort stays non-blocking but still returns the negotiated runtime `profileId` so hosts do not mistake a lite run for full-tier support. Existing caller-supplied pt-webgpu engines now load unattached, run the same tier probe, return the runtime profile, and attach only after strict acceptance, so the factory path and existing-engine path are equivalent. The generic `@vitrum/gltf-adapter` bridge remains engine-agnostic but now accepts an explicit same-backend `runtimeProfile`, allowing adapter-only hosts to validate `backend:"pt-webgpu"` loads against `pt-webgpu-lite` after they negotiate their own device tier. `reject-degraded` also now rejects source-pathed approximate import degradations for generated tangents, missing tangent texcoords, tangent-generation failure, and dropped high-UV texture refs, while `reject-unsupported` still accepts them. |
| ~~`rankGltfBackends` lite row~~ ✅ DONE (2026-06-13) | `packages/gltf-adapter/src/featureReport.ts`, `packages/pt-webgpu/src/index.ts` | `rankGltfBackends()` now emits separate `pt-webgpu` full and `pt-webgpu-lite` profile rows (`profileId`, `traceTier`) while preserving `.backend: 'pt-webgpu'` for existing callers. Lite profile scores full-tier-only material texture/alpha/env/aniso fields as unsupported; runtime lite `supportDetails.materials` and structured `setScene()` warnings now match the shader's no-group-3 material path, including exact primitive ids and per-primitive unsupported field lists. Tests: `gltfAssetApi.test.ts`, `liteTierCapabilities.test.ts`. |
| ~~PTWG-07 verify~~ ✅ DONE (source-verified 2026-06-13) | `sceneMutationRouter.ts`, lite texture refresh tests | Emitter/env mutation refreshes `liteLightTex` / `liteEnvTex`; remaining lite work is ranking/policy, not stale sampled textures. |

**Footgun closed 2026-06-15:** lite rect/disc area lights now use paired MIS. `kernelLite.wgsl.ts` applies the light-sampled power heuristic, and `connectLite.wgsl.ts` intersects BSDF-sampled directions against the same packed `liteLightTex` rect/disc records. The historical one-sided half-MIS deficit remains pinned in `oracle.liteRectMis.test.ts`.

#### 2F — Analytic + instancing (pt-webgpu full)

Already native. **glTF instancing:** ordinary glTF multi-node reuse still flattens to separate primitives. **`EXT_mesh_gpu_instancing` import + node animation DONE (2026-06-16):** accessor-driven node instancing now imports to core `InstancedMeshPrimitive`, with `nodeWorld * instanceTRS` baked into each matrix; required use is accepted; feature reporting predicts `instanced-mesh` instead of an unsupported extension; direct and `loadGltfForEngine()` bridge-created `GltfSceneController` instances carry the local instance matrices and patch `instances[]` when the instanced node or an ancestor animates. Malformed accessors, custom per-instance attributes, and skinned/morphed instancing remain explicit fallback edges: malformed or no-transform data imports the base mesh once with an `ignored-gpu-instancing` diagnostic, custom non-transform attributes warn and are ignored, and skinned/morphed instancing preflights as `EXT_mesh_gpu_instancing.skinnedOrMorphed=unsupported` under strict compatibility because core has no instanced-skinned/morphed contract yet.

---

### Phase 3 — Walkaround-hybrid 100%

> **SCOPE DECISION LOCKED (user, 2026-06-12): the FULL texture atlas is committed
> scope.** Phases 3C (alpha), 3D (atlas: all maps + UV/tangent buffers), and 3E
> (extension lobes) execute as written — they are requirements, not options.
> "Ledger truth via `unsupported` grades" is NOT an acceptable terminal state for
> walkaround material maps; rows are expected to promote to native/approximate
> through the atlas. Permanent-unsupported exceptions remain only the 3F list
> (spectral/displacement/thin-film/layers).

#### 3A — P0 subsystem & pipeline correctness

> **2026-06-12 reconciliation: W-HYB-01/02/03 are ✅ LANDED on `main`**
> (verified on disk: NRC clear wired at `WalkaroundGPUPipeline.ts:1227`;
> atrous per-iteration 256-byte UBO strides; init failures → `onError`).
> H25/H28/H29 are closed by code + tests; H26-H27 closed in R8-B
> (keep oracle coverage).

| ID | Status | File(s) | Fix | Footgun |
|----|--------|---------|-----|---------|
| W-HYB-01 | ✅ LANDED | — | `clearSlotClaims()` wired before NRC GI-RIS | — |
| W-HYB-02 | ✅ LANDED | — | Per-iteration UBO bindings (256-byte strides) | — |
| W-HYB-03 | ✅ LANDED | — | Init failures route to `onError` | — |
| H25 | ✅ CLOSED | `PPGCoordinator.ts`, `ppgPdf.wgsl.ts`, `dTree.ts`, `dTreeInteriorFlux.test.ts`, `ppgGuidedSampling.test.ts` | Bottom-up interior flux propagation is implemented before dTree refinement; CPU/GPU pdf logic now matches leaf flux / solid angle. **2026-06-17 proof hardening:** the PPG guide CPU oracle now uses the same equal-area cylindrical `ppgDirToUv` convention as `ppgPdf.wgsl`/`ppgUpdate.wgsl` instead of the retired octahedral map. | Residual promotion risk: no broad real-GPU PPG A/B in package tests. |
| H26-H27 | ✅ CLOSED (R8-B) | `risGiNrc.wgsl.ts` | Spread + training target fixed — keep oracle tests | |
| H28 | ✅ CLOSED | `layerResourceAllocator.ts`, `reluPingPong.test.ts` | In-place ReLU layers now allocate a distinct output buffer and remap downstream tensor reads. | Broader neural weights/quality remain separate from H28. |
| H29 | ✅ CLOSED | `HybridEngineOptions.ts`, `HybridEngineConfig.ts`, `HybridEngineLifecycle.ts`, `WalkaroundGPUPipeline.ts`, `pipelineCompiler.ts`, `PPGCoordinator.ts`, `resourceManager.ts`, `giStateSnapshot.ts`, `HybridEngineGIState.ts`, `ppgUpdate.wgsl.ts` | `ppgMaxDTreeNodesPerCell` now threads from public engine options through derived config, init host, shader compile, PPG resource allocation, coordinator resize/upload/export/import, and GI snapshot v5 metadata. Focused tests cover config preservation, buffer-size math, WGSL defaults, v5 round-trip, and v4 default compatibility. | Residual promotion risk: broad real-GPU PPG A/B remains a hardware-validation queue item, not an implementation gap. |

#### 3B — Emitters, environment, shadows (ledger truth)

| Item | File(s) | Current | Required for native |
|------|---------|---------|---------------------|
| Point/spot DI | `shade.wgsl.ts` `lo_analyticNEE`, `analytic_lights` binding 13; `restir/emitterHelpers.ts`; `pipeline/BvhBufferHost.ts`; `HybridEngineDdgiSync.ts` | ✅ CODE CLOSED: point/spot emitters pack into the binding-13 analytic texture payload as `Le = color × intensity` with spot direction/cone metadata; `syncDdgiFromCoreScene()` refreshes `pipeline.updateAnalyticLights(scene)` on init and emitter updates before the scene bind group is consumed. | Regressions: `bvhBufferHost.test.ts` pins point/spot upload payload, and `HybridEngineDdgiSync.test.ts` pins the lifecycle/helper bridge that refreshes binding 13. |
| Mesh-area `color`/`intensity` | `restir/bvhCore.ts`; `restir/__tests__/directLightEmitterCore.test.ts` | ✅ CODE CLOSED: mesh-area emitters now override the referenced primitive material slot with `Le = emitter.color × emitter.intensity` before ReSTIR emitter-list packing, and the same override feeds the TLAS/merged emissive-Le glow buffer. | Keep the existing H23 regression: `color=[10,10,10]`, `intensity=10` produces `[100,100,100]` and total emissive power follows the override. |
| Emitter `castShadow` | DDGI/ReSTIR/RC direct-light paths | ✅ CODE CLOSED / ledger native: analytic point/spot payload packs binding-13 lane `[13]`; shared `EmitterTri` packs lane `[19]`; ReSTIR-DI candidate visibility + shade visibility gate on `e.castShadowDisabled`; DDGI/RC area-emitter NEE skip the emitter shadow ray; DDGI and RC fixture/sun lights carry castShadow-disabled flags; main direct sun gates its visibility ray from the scene directional emitter flag. | Regressions: `directLightEmitterCore`, `probeUpdateLights`, `rcLightsLayoutPin`, `emitterCastShadowWgsl`, `rcLightEvalWgsl`, `hybridEngineFrameOrchestrator`, and `engineContract`. |
| `primitiveCastShadow` GI-side | DDGI, ReSTIR-GI, RC | ✅ CODE CLOSED / ledger native | Shared material flag bit 1 + predicate-backed shared-BVH traversal now skip `castShadow:false` geometry in DDGI probe direct-light visibility and RC probe direct-light visibility; ReSTIR-GI + GRIS visibility use `traceSceneAnyCastMask` through `bvh_material` bit 0. Tests: `materialEntry`, `bvhCastShadowMask`, `ddgiMaterialPack`, `emitterCastShadowWgsl`, `rcLightEvalWgsl`, `engineContract`. |
| `updateLighting` sun | `HybridEngine.ts`; `HybridEngineDdgiSync.ts` | ✅ SOURCE-VERIFIED STALE: `updateLighting({ primaryLightDir })` re-syncs DDGI sun lights through `_syncDdgiLightsFromCoreScene()` or `orientDdgiSunLights(...)`; `primaryLightIntensity` also updates the DDGI sun multiplier with scene-directional single-count handling. | Keep mutation-matrix coverage |
| `procedural-sky` | `resolveHybridEnvironment.ts`; `environmentTexture.ts`; `HybridEngine.ts` | ✅ CODE CLOSED / approximate: walkaround now bakes the shared Preetham model to a 256x128 equirect + PBRT-style directional CDF and uploads it through the same env bindings as raw HDRI; scalar sky remains the fallback average | Keep `approximate` until/unless an analytic or higher-resolution validation tier replaces the finite bake; planner no longer needs to avoid WA solely for procedural-sky |
| RC sun RGB | `HybridEngineFrameOrchestrator.ts`; `hybridEngineFrameOrchestrator.test.ts` | ✅ CODE CLOSED: RC uses the scene directional emitter's RGB × intensity when present, with the legacy grey `primaryLightIntensity` fallback only when no scene directional exists. | Keep regression test with RC enabled |

#### 3C — Alpha & blending (glTF `alphaMode`)

Walkaround now has **scalar cutout alpha** (`alphaMode` / `alphaCutoff` /
`opacity`) as an approximate material tier: `packBVHRoughMetalFromCore` encodes
mask discard / fully-transparent blend endpoints in `bvh_material` bit 2;
`traceSceneFirstHitAlphaMaskTextured` skips those triangles for primary + GI
first-hit visibility; `traceSceneAnyAlphaMaskTextured` now also evaluates
readable `baseColorMap.a` / `alphaMap.r` / vertex-alpha coverage for shade,
ReSTIR-DI, ReSTIR-GI/NRC, and GRIS temporal/spatial shadow visibility; and
`HybridEngine.setScene` emits `walkaround-hybrid.alpha-blend-approximation` for
fractional `alphaMode:'blend'` because ReSTIR direct light, light-map/emissive
terms, and GI participation remain approximate.
Readable `alphaMap` cutout is
atlas-backed; camera-visible fractional blend now uses the transparent-OIT pass
after `indirect-combine` and before temporal accumulation, and its direct-sun
plus analytic point/spot and finite-emitter terms consume the same atlas-backed
GGX/specular/clearcoat/sheen/aniso/iridescence material payload as opaque
shade/ReSTIR material scoring.

| Step | Code | Footgun |
|------|------|---------|
| Pack alphaMode + cutoff | ✅ CODE CLOSED for scalar cutout + alpha-map metadata: `packingHelpers.ts` bit 2 in `bvh_material`; `materialTextureAtlas.ts` stores alpha mode/opacity/cutoff metadata; tests in `roughMetalPacking.test.ts` and `materialTextureAtlas.test.ts` | Fractional blend remains approximate, but no longer fully opaque |
| Shade discard | ✅ CODE CLOSED as traversal discard: `materialAtlas.wgsl.ts` exposes both stochastic cutout (`traceSceneFirstHitAlphaMaskTextured`) and opaque-pass (`traceSceneFirstHitAlphaMaskTexturedOpaqueOnly`) first-hit helpers. Shade, ReSTIR-DI, temporal/spatial primary casts, default ReSTIR-GI, and NRC-GI now use the opaque-only helper for alpha-blend surfaces, while alpha-mask cutout still happens before reservoir writes. | Fractional blend is intentionally OIT-owned camera composition, not a stochastic reservoir vertex; true transparent transport remains a separate promotion target. |
| Composite blend | ✅ CODE CLOSED/approximate: `transparentOit.wgsl.ts` front-to-back ray-walks fractional blend layers, skips effectively invisible blend layers (`coverage <= 0.001`) so deeper transparent layers can still composite, composites over `combinedDenoisedTexture`, shades direct sun plus analytic point/spot and four-sample fixed-stratified finite mesh emitters through the material-payload GGX/clearcoat/sheen/aniso/iridescence BRDF, uses a deterministic five-tap material-lobe estimate for camera-visible sky/environment lighting across HDRI-backed and no-HDRI scalar/procedural sky scenes, writes `transparentCompositeTexture`, and `TemporalAccumPass` consumes that output | First-hit light-map/emissive terms remain camera-visible approximations; direct sun/point/spot/finite-emitter shadows now use alpha-aware deterministic transmittance, ReSTIR/GI reservoirs use opaque-only first-hit predicates, shade/ReSTIR/GI shadow predicates evaluate texture-alpha blockers as binary/transmittance visibility, and DDGI probe direct sun/point/mesh-emitter visibility now applies atlas-backed alpha plus Beer/transmission/thickness glass visibility. Transparent layers still are not true ReSTIR/GI transport participants until validated separately |
| ~~`alphaMap`~~ | ✅ CODE CLOSED/approximate: readable alpha maps are linear atlas layers with per-map UV, transform, wrap, and alphaMode/opacity/cutoff metadata. Mask uses `opacity * baseColorMap.a * alphaMap.r < alphaCutoff`; blend coverage feeds the transparent-OIT pass for camera-visible composition and direct sun/point/spot/finite-emitter shadow transmittance, DDGI probe direct-light RGB shadow visibility, and `traceSceneAnyAlphaMaskTextured` evaluates atlas coverage for shade/ReSTIR-DI/ReSTIR-GI/NRC/GRIS shadow visibility. Transparent GI/ReSTIR transport participation remains approximate. | Keep warning until transparent GI/ReSTIR promotion lands |

#### 3D — Texture atlas (non-optional for walkaround material 100%)

**Architecture (mirror pt-webgl2):**

```
Scene MaterialSpec.*Map
  → walkaround-hybrid/src/pipeline/materialTextureAtlas.ts
  → GPU texture_2d_array atlas + metadata texture
  → per-tri materialId + uv0/uv1/tangent streams from shared BVH packing
  → shade / ReSTIR / DDGI / RC material helpers sample the supported map rows
```

**Atlas slices landed:** `baseColorMap` (2026-06-13) plus
`normalMap`/`roughnessMap`/`metallicMap`/`aoMap`/`alphaMap`/`emissiveMap`/`transmissionMap`/`lightMap` plus specular, clearcoat factor/roughness/normal, sheen color/roughness, anisotropy, iridescence factor/thickness maps (2026-06-14), and KHR volume `thicknessMap` (2026-06-15) now have real walkaround paths for
readable raw/DataTexture-shaped `TextureRef` handles on uv0 and uv1.
`pipeline/materialTextureAtlas.ts` builds a linear RGBA32F `texture_2d_array`
plus per-triangle metadata; `BvhBufferHost` uploads/binds it at scene bindings
20-21; `shade.wgsl.ts` samples map-specific wrap + `KHR_texture_transform`
metadata, multiplies visible albedo by `baseColorMap`, and overrides visible
BRDF roughness/metallic from the glTF G/B channels. AO samples the glTF R
channel and applies `aoMapIntensity` via the glTF occlusion-strength formula
before multiplying the runtime GTAO factor. Normal maps perturb the visible
smooth normal through authored/generated tangent.xyzw when present, falling back
to a derived per-triangle tangent frame, with `normalScale` applied. Alpha maps
cut out primary/RIS/GI hits. Emissive maps modulate camera-visible emitter glow
by sampling the material atlas at the hit UV; the per-triangle glow buffer now
stores scalar production Le only, so readable emissive maps are not average-folded
and sampled a second time. CPU-readable emissive maps still feed average linear RGB into the shared
direct emitter selection-power path. 2026-06-16 follow-up: merged-BVH
ReSTIR-DI material-backed emitters pack a source-triangle lane, so RIS candidate
scoring, temporal/spatial pHat reuse, and final shade evaluate the emissive map
at the stored triangle `xi`. Later 2026-06-16 follow-up: TLAS material-backed
emitters now map each world-expanded emitter triangle back to the local BLAS
material-atlas triangle and use the same sampled-texel payload. Later same-day
follow-up: mirrored TLAS instances encode reversed barycentric orientation in the
existing source-triangle lane, so they sample the same hit-local texel payload
instead of falling back to averaged radiance; analytic/extra emitters deliberately
keep the average-`Le` fallback. Transmission maps modulate shade/RIS/GI glass gating, volume thickness maps
sample glTF G and exponentiate the scalar Beer tint in shade, transmitted GI,
and tinted-visibility paths, and light maps add
first-hit baked outgoing radiance with `lightMapIntensity`. `CONSUMED_MATERIAL_FIELDS` and the
core promise ledger now grade walkaround `baseColorMap`, `roughnessMap`,
`metallicMap`, `aoMap`, `aoMapIntensity`, `normalMap`, `normalScale`, `alphaMap`, `emissiveMap`, `transmissionMap`, `thicknessMap`, `lightMap`, `lightMapIntensity`, `envMapIntensity`, `specularColorMap`, `specularIntensityMap`, `clearcoatMap`, `clearcoatRoughnessMap`, `clearcoatNormalMap`, `clearcoatNormalScale`, `sheenColorMap`, `sheenRoughnessMap`, `anisotropy`, `anisotropyRotation`, `anisotropyMap`, `iridescence`, `iridescenceIor`, `iridescenceThicknessRange`, `iridescenceMap`, and `iridescenceThicknessMap` as
`approximate`. They are deliberately not `native`: glass Beer/transmission/thickness tint,
  emissive-map light selection now uses CPU-readable UV-local per-triangle quadrature
  plus bounded barycentric micro-emitter subdivision for material-backed walkaround
  ReSTIR-DI emitters and implicit PT mesh lights rather than whole-texture average
  power, and DDGI mesh-area emitter NEE samples material-atlas emissive texels
  when TLAS source-triangle metadata is available. 2026-06-16 RC follow-up:
  RC probe-cast emitter NEE now samples UV0/UV1 material-backed emissive texels from
  the shared material atlas via `EmitterTri` source-triangle/subdivision metadata
  instead of always using averaged `Le`. 2026-06-18 follow-up: DDGI ordinary
  probe hits sample atlas-backed `baseColorMap`/`roughnessMap`/`metallicMap` for
  bounce albedo and the glossy-probe specular weight, and RC probe-cast direct
  sun/emitter/point/spot terms use atlas-backed `baseColorMap` for their
  Lambertian albedo. Later 2026-06-18 follow-up: DDGI ordinary probe hits also
  apply atlas-backed `normalMap`/`bumpMap` through a derived UV tangent frame for
  probe direct light, area-emitter NEE, SH feedback, glossy reflected bounces, and
  stored hit normals. Later 2026-06-18 RC follow-up: RC probe-cast direct
  sun/emitter/point/spot terms now use atlas-backed `roughnessMap`,
  `metallicMap`, specular color/intensity controls/maps, and derived-frame
  `normalMap`/`bumpMap` in a compact direct BRDF response instead of staying
  Lambertian-only. Later same-day RC tangent follow-up: RC receives the main
  walkaround authored/generated tangent.xyzw texture as probe binding 19 and
  prefers it for `normalMap`/`bumpMap` tangent frames before falling back to
  UV-gradient derivation. Later same-day DDGI tangent follow-up: DDGI snapshots now
  carry `bvhTangents.cpuData`, upload a DDGI-local authored/generated tangent.xyzw
  texture, and prefer it for ordinary probe-hit `normalMap`/`bumpMap` tangent
  frames before falling back to UV-gradient derivation. Later same-day lobe
  follow-up: DDGI ordinary probe hits now sample specular, clearcoat,
  clearcoat-normal, sheen, anisotropy, and iridescence atlas controls/maps and
  use them to steer/tint the reflected-SH complement with bounded lobe weights;
  RC probe-cast sun/emitter/point/spot terms now sample and consume clearcoat,
  clearcoat-normal, sheen, anisotropy, and iridescence in its compact direct BRDF
  response. 2026-06-19 RC emissive follow-up: ordinary RC probe hits now sample
  `RC_MATERIAL_MAP_EMISSIVE_TEXEL_OFFSET` through hit UV0/UV1 and modulate scalar
  `mat.emissive` before adding surface emission, matching DDGI's direct
  probe-hit emissive-map path and pinning it separately from emitter NEE in
  `rcLightEvalWgsl.test.ts`. These rows remain `approximate` pending furnace/GPU
  A/B promotion, not because the fields are dropped. 2026-06-16 ReGIR follow-up:
  the grid-build WRS target now uses the chosen packed light-tree leaf
  importance (`treeInput.powers`, AABB, and cone term) for `qHat`, so mapped
  material-backed micro-emitters no longer fall back to scalar `EmitterTri.Le`
  when storing cell `pSel`. The glTF compatibility planner now
  reports this remaining boundary as `emissiveMap.texelPdf: approximate` for
  imported emissive textures instead of letting native visible-emission support
  imply exact emitter-PDF semantics; `HybridEngine.setScene()` and material
  `updatePrimitive()` patches now emit the same truth as a structured
  `walkaround-hybrid.emissive-map-texel-pdf-approximation` runtime warning for
  non-glTF hosts. These paths still do not build full
  texel-alias emitter PDFs; analytic/extra emitter mapped payloads and
  non-direct-probe-hit GI emission remain approximate,
GI receiver/reuse targeting is now material-lobe aware but still
uses compact geometry+`Lo` reservoirs plus a temporal previous-domain fallback,
transparent blend now has camera-visible OIT composition with direct sun plus
analytic point/spot and four-sample fixed-stratified finite-emitter lighting, but
reservoir-backed ReSTIR direct light and GI participation still follow the
approximate realtime lanes,
and baked light maps' non-camera paths still use scalar packed lanes. Bump maps now feed
shade-owned, DI, and GI suffix visible-normal payloads; displacement maps remain
deliberately unsupported until a true geometry/BVH displacement path exists.

| Component | File(s) | Notes |
|-----------|---------|-------|
| Atlas build | ✅ FOURTH SLICE + 2026-06-18 truthfulness follow-up: `pipeline/materialTextureAtlas.ts` | `baseColorMap`, `emissiveMap`, specular-color, and sheen-color raw/DataTexture handles are inverse-sRGB decoded into linear RGBA32F array layers; `normalMap`, `bumpMap`, `roughnessMap`, `metallicMap`, `aoMap`, `alphaMap`, `transmissionMap`, `thicknessMap`, `lightMap`, specular-intensity, clearcoat, clearcoat-roughness, clearcoat-normal, sheen-roughness, anisotropy, iridescence, and iridescence-thickness maps are packed as linear map layers and sampled from their glTF channels. Scalar `envMapIntensity` is packed in atlas metadata and consumed by HDRI ReSTIR-DI scoring/reuse/resolve. Unreadable CPU texture handles emit `walkaround-hybrid.unreadable-material-texture-map` through `onWarning` at scene load/material rebuild with `fallback:"map ignored"` and, when the `TextureRef` came from the glTF adapter, source metadata (`sourcePath`, texture/image/sampler indices, URI, MIME type, and source extension) so hosts can locate the exact incompatible asset input. Atlas-backed maps with unsupported `TextureRef.texCoord` values now emit `walkaround-hybrid.unsupported-material-texture-texcoord` and are explicitly ignored; only UV sets 0 and 1 are supported. Displacement and the spectral/layered/scattering families stay explicit unsupported warnings, not silent atlas drops. |
| UV buffer | ✅ FIRST SLICE: `bvhCore.ts`, `shared-bvh/worldSpaceMerge.ts` | uv0 rides `bvh_position.w`; uv1 rides `bvh_normal.w` using the same packed 16:16 unorm convention. |
| Tangent buffer | ✅ FOURTH SLICE: `shared-bvh/worldSpaceMerge.ts`, `restir/bvhCore.ts`, `pipeline/bvhTangentTexture.ts`, `materialAtlas.wgsl.ts` | TLAS packs forward `packSceneFromCore().tangents`; merged-world packs transform authored/generated tangent directions and flips handedness for mirrored transforms; walkaround uploads the vec4 stream as scene binding 22 (`rgba32float` texture) and the normal/clearcoat-normal TBN path prefers it before falling back to UV-gradient derivation. Ledger rows stay approximate for reservoir/GI/PDF scope, not because tangent data is dropped. |
| Bind group | ✅ FOURTH SLICE: `bindGroupDescriptors.ts`, `bindGroupBuilders.ts`, `BvhBufferHost.ts` | Scene bindings 20-22 add a shared material-map atlas + metadata + tangent texture as textures, not storage buffers, preserving the storage-buffer budget. |
| Material index per tri | ✅ FOURTH SLICE: metadata texture keyed by triangle index | Current atlas-backed maps, including `baseColorMap`, normal/ORM/AO/alpha/emissive/transmission, `thicknessMap`, `lightMap`, and extension-lobe maps, use `triangleMaterialIds` at pack time; scalar lanes stay as fallback when no readable map exists. |
| `materialPatch` fast path | ✅ THIRD SLICE + 2026-06-18 DDGI snapshot follow-up + 2026-06-19 atlas-refresh follow-up: `HybridEnginePrimitiveUpdates.ts`, `BvhBufferHost.ts`, `WalkaroundGPUPipeline.ts`, `HybridEngine.ts` | Material-only edits stay on the slice upload path. Atlas-backed map handle/UV/wrap/transform changes and atlas metadata scalar edits (`normalScale`, `lightMapIntensity`, `envMapIntensity`, `alphaMode`, `opacity`, `alphaCutoff`) now repack and refresh only the material atlas + metadata textures instead of routing through BVH/TLAS full rebuild. Material-only edits refresh DDGI's `RestirBvhSnapshot` material payload without RC geometry propagation; atlas/radiance/visibility changes still invalidate DDGI probes and refresh emitter buffers so the old full-rebuild side effects are preserved. |
| Ledger | ✅ FOURTH SLICE + DDGI map follow-up: `WALKAROUND_MATERIALS`, `CONSUMED_MATERIAL_FIELDS` | `baseColorMap`, `normalMap`, `normalScale`, `roughnessMap`, `metallicMap`, `aoMap`, `aoMapIntensity`, `alphaMap`, `emissiveMap`, `transmissionMap`, `thicknessMap`, `lightMap`, `lightMapIntensity`, specular maps, clearcoat factor/roughness/normal maps, sheen color/roughness maps, anisotropy controls/maps, and iridescence controls/maps promoted to `approximate` with tests; readable `transmissionMap` and `thicknessMap` now modulate DDGI glass sun visibility, finite point/spot and mesh-area-emitter shadow visibility, and direct probe-hit glass environment mixing instead of staying scalar-only in probe rays. `envMapIntensity` is `native` for HDRI ReSTIR-DI scoring/reuse/resolve. Remaining maps remain unsupported until each has shader consumption or explicit routing. |

**Footguns:**
- Sampling baseColor UV for all maps (pt-webgpu v1 bug) — use per-map `TextureRef.texCoord` + `transform` from glTF.
- ~~`materialPatch` with atlas-backed maps rebuilt correctly but paid full BVH/TLAS rebuild cost~~
  ✅ CODE CLOSED 2026-06-19: map/atlas-metadata edits now refresh the material
  atlas textures directly while keeping slice uploads, DDGI material-snapshot
  refresh, probe invalidation, and emitter refresh side effects pinned by
  `mutationMatrix.test.ts` + `bvhBufferHost.test.ts`.
- ~~Merged-BVH material dedup could collapse atlas-distinct materials~~
  ✅ CODE CLOSED 2026-06-19 follow-up: `shared-bvh/src/worldSpaceMerge.ts`
  now signs atlas-consumed texture refs through Float32-precision UV
  transform/sampler metadata, unsupported/non-finite texCoord tokens, atlas
  scalar metadata, and the same bare-object texture compatibility shim consumed
  by `materialTextureAtlas.ts`. `materialSig.test.ts` pins tiny-but-GPU-visible
  transform/scalar differences, bare texture objects, and unsupported texCoord
  values.
- ~~Non-finite material texture transforms could leak NaN/Inf into atlas metadata~~
  ✅ CODE CLOSED 2026-06-19 follow-up: `materialTextureAtlas.ts` now sanitizes
  non-finite KHR texture-transform offset/scale/rotation components to identity
  fallbacks and emits `invalid-material-texture-transform` diagnostics with
  source metadata. `materialTextureAtlas.test.ts` pins finite GPU metadata plus
  source-pathed diagnostics for invalid transform components.
- ~~ReSTIR primary hit uses different UV than shade~~ ✅ SOURCE-VERIFIED STALE:
  RIS/primary/OIT paths call the shared material-atlas helpers with hit UV0 plus
  `materialAtlasUv1ForHit`.
- ~~Atlas rebuild / stale UVs when morph UVs deform~~ ✅ CODE CLOSED for core UV lanes:
  glTF morph-target `TEXCOORD_0` / `TEXCOORD_1` deltas now import into
  `morphTargetUvs` / `morphTargetUv1s`, solve through `solveSkin()`, and propagate
  through controller/backend update paths. Walkaround takes the topology attribute
  refresh path for posed UV streams; pt-webgpu/pt-webgl2 pack posed UVs during
  skinned ingestion. Remaining unsupported scope: missing-base UV morph targets
  and `TEXCOORD_2+` deltas.

#### 3E — Extension lobes on walkaround (clearcoat, sheen, iridescence, specular, anisotropy)

Scalar `specularColor` / `specularIntensity`, readable `specularColorMap` / `specularIntensityMap`, scalar `clearcoat` / `clearcoatRoughness`, readable `clearcoatMap` / `clearcoatRoughnessMap` / `clearcoatNormalMap`, scalar `sheen` / `sheenColor` / `sheenRoughness`, readable `sheenColorMap` / `sheenRoughnessMap`, scalar/map `anisotropy` / `anisotropyRotation` / `anisotropyMap`, and scalar/map `iridescence` / `iridescenceIor` / `iridescenceThicknessRange` / `iridescenceMap` / `iridescenceThicknessMap` are now code-closed as approximate: material-atlas metadata stores the scalar controls/maps, shade-owned direct/analytic/sun/glossy-indirect paths consume them, ReSTIR-DI pHat/reuse consumes them, and default/NRC GI suffix plus receiver-lobe target paths consume them through a rich-material proxy. Remaining 3E work is material-furnace/reference A/B and promotion evidence needed to move beyond approximate.

**Footgun:** Walkaround is not a path tracer — clearcoat/sheen are approximations. Grade `approximate` unless energy conservation verified; planner must surface this.

#### 3F — Fields intentionally permanent `unsupported` on walkaround

Document in ledger + planner: `displacement*`, `spectralAttenuation`, `dispersionAbbeNumber`, `thinFilmStack`, `scattering*`, `frontLayer`/`backLayer` (unless stained-glass scope). These fields are now pinned by walkaround unit + engine-warning tests on both `setScene()` and `updatePrimitive()`, including the `KHR_materials_dispersion` source path in glTF compatibility reporting. **Arbitrary glTF 100%** routes assets using these to pt-webgpu via `rankGltfBackends` — walkaround 100% ≠ all fields native.

#### 3G — Structural debt (items_to_fix §H)

| Item | File | Action |
|------|------|--------|
| ~~H32 glass TLAS shadow~~ ✅ CODE/ORACLE CLOSED | `shared-bvh/wgsl/tlasTraversal.wgsl.ts`; `sceneTraversal.wgsl.ts`; `tools/behavioral-gate/tlas-glass-shadow-oracle.mjs` | `traceTlasAny` forwards `skipGlass` into the single closest-hit path and walkaround forwards the flag. The checked-in WebGPU oracle imports the production shared-bvh BVH/TLAS WGSL and proves glass occludes when `skipGlass=false`, is ignored before the opaque hit when `skipGlass=true`, and still lets the ray hit opaque geometry behind it. |
| H33 materialSig Beer-Lambert | `shared-bvh/src/sceneBvh.ts`; `shared-bvh/src/__tests__/sceneBvhVersionTag.test.ts` | ✅ CLOSED (Wave 2): `materialSetHashFloats` now includes packed `attenuationDistance` (with the canonical no-attenuation sentinel) and `thickness`, so no-tag `SceneBvh.updateFromCore()` rebuilds when only Beer-Lambert distance/depth changes. Regression tests pin attenuationDistance-only and thickness-only edits. |
| H34 BVH degenerates | `buildArrayBvh.ts`, `tlas.ts`; `nanTriangleFilter.test.ts`, `oobIndexFilter.test.ts`, `tlas.test.ts` | ✅ CODE CLOSED: BLAS already filters non-finite and out-of-range-index triangles with warnings and returns the empty-BVH shape when every triangle is invalid; this wave added TLAS parity by filtering non-finite build instances (AABB or `worldToLocal`) with warnings, throwing clearly when no finite instance remains, and rejecting non-finite/inverted refit AABBs before mutating node bounds. |
| Phantom emitter H22 | `walkaround-hybrid/src/restir/emitterList.ts:297-369`; `walkaround-hybrid/__tests__/hRemediationItems.test.ts` | ✅ CODE CLOSED: the zero-real-emitter placeholder remains only as the required non-empty WebGPU storage binding, but it is gated inert (`Le=[0,0,0]`, `intensity=0`, `power=0`, `totalEmissivePower=0`). The CDF fallback writes finite `[1]` instead of NaN, and the regression test pins both the inert payload and finite-CDF path. |
| H24 SceneBvh equal-length edits | `shared-bvh/src/sceneBvh.ts`; `sceneBvhVersionTag.test.ts` | ✅ CODE CLOSED: untagged `SceneBvh.updateFromCore()` uses exact buffer fingerprints, and the test pins equal-length vertex edits plus large unsampled-byte edits. |
| H24 material resolver | `restir/bvhCore.ts`; `restir/__tests__/bvhCoreMaterialResolver.test.ts` | ✅ CODE CLOSED: duplicate mesh-like primitive ids now throw before TLAS packing can reuse the first material slot, and unknown resolver calls throw instead of falling back to material 0. |
| H24 GPU skinning host-resource skip | `skin/GpuSkinningSubsystem.ts`; `gpuSkinningBindRouting.test.ts` | ✅ CODE CLOSED: missing GPU skinning position/normal buffers or mesh ranges now CPU-skin every skinned mesh instead of silently skipping. Count-only-cache risk is source-verified stale because cached bind groups key on live shared buffer identity. |
| H24 NRC training diagnostics | `pipeline/WalkaroundGPUPipeline.ts`; `pipeline/__tests__/nrcTrainingDiagnostics.test.ts` | ✅ CODE CLOSED: `trainFromRecords()` rejections emit deduped non-fatal `EngineError`s through the engine error channel while live, and remain suppressed after dispose. |
| H24 denoiser state consumers | `WalkaroundGPUPipeline.getActiveDenoiserState`; `HybridEngineFrameOrchestrator.ts`; `frameStatsDenoiserState.test.ts` | ✅ SOURCE-VERIFIED STALE: active denoiser state, including OIDN failed/retryable states, is emitted through `FrameStats.denoiserState`. |
| H24 RC sun RGB | `HybridEngineFrameOrchestrator.ts`; `hybridEngineFrameOrchestrator.test.ts` | ✅ CODE CLOSED: RC sun dispatch now preserves scene directional chroma and intensity; scalar grey is only the no-scene-directional fallback. |
| H24 Welford aux allocations | `pipeline/frameResources/createCommonFrameResources.ts`; `pipeline/resourceManager.ts`; `WalkaroundGPUPipeline.ts`; `createCommonFrameResources.test.ts` | ✅ CODE CLOSED: non-`atrous-variance` denoisers now request `welfordPingPong:false`, collapsing unused Welford aux/estimate textures to 1x1 placeholders while the default Welford denoiser keeps full-size resources. `varianceBuffer` and `hdrTotalTexture` remain full-size because they are always read/written by the standard frame graph. |
| ReGIR dead alloc H24 | `pipeline/ReGIRCoordinator.ts`; `pipeline/BvhBufferHost.ts`; `pipeline/resourceManager.ts`; `pipeline/pipelineCompiler.ts`; `regirWiring.test.ts` | ✅ CODE CLOSED: ReGIR is fully opt-in. `gridRegionBytes()` returns `0` when `regirConfig.enabled` is false, `BvhBufferHost` pads the combined light-tree buffer only by that byte count, `compilePipelines()` only builds `regir-build` when `regirEnabled` is true, and the pass still gates on `coord.live`. Regression tests pin the disabled path as an unpadded light-tree upload and the enabled path as exactly `cells × survivors × REGIR_FLOATS_PER_SURVIVOR × 4` bytes of appended grid storage. |
| DDGI error swallow | `DDGI.ts:303-346` | ✅ CODE CLOSED: DDGI init/BVH/probe-frame failures now emit non-fatal `EngineError` diagnostics through `HybridEngine.onError`; failed probe frames do not advance the grid to `ready`. Focused tests pin direct DDGI reporting plus HybridEngine forwarding. |

---

### Phase 4 — Arbitrary glTF orchestration (cross-backend)

#### 4A — Single host path

```
loadGltfForEngine(url, { fetch, dracoDecode, meshoptDecode, decodeImage, decodeTextures, decodePixels })
  → optional CPU-linear textureDecodePass()
  → rankGltfBackends(report, policy)
  → createEngine({ prefer, scene, gltfAsset: result })  // NEW: optional gltfAsset
  → controller.attachEngine(engine)
  → controller.play(...); loop: controller.tick(dt); engine.renderFrame(...)
```

| Task | File | Footgun |
|------|------|---------|
| ~~`createEngine` accepts `gltfAsset?: GltfAssetResult`~~ ✅ DONE | `createEngineInternals.ts` exposes a structural `gltfAsset` hint; `createEngine.ts` passes `gltfAsset.recommendedBackend.backend` into `pickBackend`; `@vitrum/engine/gltf` supplies the loaded asset automatically. | Structural type avoids importing adapter runtime code into the core create path |
| ~~Replace triangle-only auto~~ ✅ DONE for glTF assets | `createEngineScale.ts` `pickBackend` uses the glTF recommendation when `prefer:'auto'`; explicit `prefer` still wins, and WebGL-only hosts fall back to `pt-webgl2` for WebGPU recommendations. Test: `createEngineScale.test.ts`. | Non-glTF scenes still use the 500k-triangle heuristic |
| ~~`VitrumCanvas` `gltf` prop~~ ✅ DONE (bridge parity deepened 2026-06-16) | `VitrumCanvas.tsx` now accepts `gltf` + bridge-level `gltfOptions`, loads through `loadGltfWithEngine()` instead of the asset-only loader, forwards decoder hooks / texture decode settings / compatibility modes into the same one-call glTF path, then hands the prepared engine and glTF controller to `attachVitrum`. `attachVitrum` now accepts a preconstructed engine plus structural scene controller, re-targets the controller after device-loss auto-recreate, and can opt into RAF-driven controller playback via `gltfPlayback` / `sceneControllerPlayback`; playback samples the initial RAF timestamp first, then advances with real elapsed delta on subsequent ticks. Tests: `vitrumCanvasMount.test.tsx`, `attachVitrumLoop.test.ts`. | Direct `scene` remains supported; `gltf` is the creation-time alternative; playback remains opt-in so hosts can keep owning their animation clock. |
| ~~`ProgressiveHandoffCoordinator` + glTF~~ ✅ DONE | `progressiveHandoff.ts`, `createProgressiveEngine.ts`, `progressiveHandoff.test.ts` | Structural `controller` option advances once per `frame()` (default 1/60s or host delta callback) and receives a synthetic patch target that forwards `setScene` / `updatePrimitive` to both engines through the coordinator's existing scene-authority/reset path. Empty-animation glTF controllers are skipped safely; `createProgressiveEngine` forwards the controller options. |
| ~~Shared-device handoff one-call helper~~ ✅ CODE CLOSED | `@vitrum/engine/gltf` now exports `loadGltfWithProgressiveEngine()`, which loads the asset, validates against the full `pt-webgpu` compatibility profile, builds the glTF controller, and passes that controller plus the imported scene into `createProgressiveEngine()`; test: `gltfProgressiveSubpath.test.ts` pins that this helper does **not** run a standalone lite-tier adapter probe because the progressive engine owns the realtime/converged stack construction. | Texture transcoding/upload policy still follows the adapter/backend handles; built-in Basis/GPU texture transcoding remains the separate `KHR_texture_basisu` row. Runtime full-vs-lite strict gating remains the `loadGltfWithEngine()` / adapter-explicit-runtime-profile contract, not this progressive helper. |
| ~~One-call CPU texture decode bridge~~ ✅ CODE CLOSED | `loadGltfForEngine()` / `loadGltfWithEngine()` / `loadGltfWithProgressiveEngine()` accept `decodeTextures`, `decodePixels`, `textureTarget`, `maxTextureSize`, and NPOT warning options. When decode is requested, the bridge calls `loadGltfAndDecodeTextures()` before engine construction/attachment and returns `decodedTextureCount`, `unchangedTextureCount`, `textureDecodeDiagnostics`, and `textureDecodeWarnings` beside the decoded-scene `textureDecodeReport`. 2026-06-16 browser-host follow-up: decoded loads that provide `decodePixels` but no custom `decodeImage` now preserve raw image bytes during asset conversion instead of creating browser `ImageBitmap`/opaque handles first, so the one-call decoded path is predictable in both Node and browser hosts. 2026-06-18 follow-ups: decoded browser loads without a custom `decodePixels` hook now preserve raw bytes and use `createImageBitmap` + canvas/OffscreenCanvas readback when available, producing CPU-readable RGBA payloads or a structured readback-unavailable diagnostic; Node hosts decode embedded/fetched PNG raw-image handles through `pngjs`, JPEG raw-image handles through `jpeg-js`, and WebP raw-image handles through `webp-wasm`; and selected texture-source extension hook issues (`EXT_texture_webp` / `KHR_texture_basisu` / `MSFT_texture_dds`) are reconciled after decode when the `textureDecodeReport` proves CPU-readable handles, so `reject-degraded` no longer rejects an extension image source that the decode bridge actually resolved. 2026-06-19 hardening: `decodePixels` context and decode diagnostics now carry texture/image/sampler/source-extension provenance, and selected KTX2/DDS sources without a pixel decoder report the compressed source extension explicitly instead of a generic raw-image failure. Tests: `gltfAssetApi.test.ts`, `gltfSubpathExport.test.ts`, `gltfProgressiveSubpath.test.ts`. | Default remains report-only unless decode-specific options are supplied, preserving existing host behavior; ordinary `loadGltfAsset()` browser image handles are unchanged. |
| ~~Examples~~ ✅ DONE | `examples/gltf-viewer/` | Self-contained Vite app now exercises `loadGltfWithEngine()`, backend recommendation, `textureDecodeReport`, controller attachment, and the capture protocol. |

#### 4B — Compatibility enforcement

| Mode | When to throw |
|------|----------------|
| `best-effort` | Never; converter degradations in `GltfAssetResult.warnings` plus `GltfAssetResult.diagnostics`; runtime/controller/backend warnings still surface through controller result warnings and `Engine.onWarning` |
| `reject-unsupported` | Any used field `unsupported` on selected backend |
| `reject-degraded` | Any non-`native` issue including `approximate`, `requires-hook` without hook |

✅ **UPDATED (2026-06-16):** `engineBridge.ts` strict modes now distinguish unsupported primitive modes from fallback-generated point/line topology. Unknown modes still reject under `reject-unsupported`; `POINTS`/`LINES`/`LINE_LOOP`/`LINE_STRIP` import as `fallback-generated-mesh`, so `reject-unsupported` accepts them while `reject-degraded` rejects with a source-pathed compatibility issue (`primitive:mode:1=fallback-generated-mesh at meshes[0].primitives[0].mode`).

#### 4C — Texture handle contract (all backends)

| Backend | Expects `TextureRef.handle` | Decoder output |
|---------|------------------------------|----------------|
| pt-webgl2 | `{width,height,data:Float32Array}` RGBA linear or DataTexture-shaped | `texturesArray.ts:79` |
| pt-webgpu | Opaque host handles or decoded pixel-data handles; uploaded by the backend scene pack | CPU pixel-data for adapter-decoded raw images, preserving the field's upload color space (`handleColorSpace`) and reporting `ptWebgpu:'ready'`; opaque handles are only ready when supplied through a host/backend texture path |
| walkaround (Phase 3D) | Same as pt-webgl2 for atlas build | CPU pixels → atlas |

✅ **DONE (2026-06-13; reconciled 2026-06-16; browser/Node default decode follow-ups 2026-06-18):** `@vitrum/gltf-adapter` exports `decodeSceneTextures(scene, { target: 'cpu-linear' | 'webgpu', decodePixels })`.
The `cpu-linear` path normalizes raw-image `TextureRef` handles into `{ width, height, data: Float32Array }` RGBA linear payloads with a Vitrum hint, applies the adapter's per-field sRGB-vs-linear policy, keeps alpha linear, emits source-path warnings when a raw image cannot be decoded, downsamples decoded payloads that exceed `maxTextureSize`, returns structured diagnostics for missing decoders / unsupported handles / max-size resize / NPOT-repeat hazards, and returns a fresh `textureDecodeReport`. In browser hosts, `loadGltfAndDecodeTextures()` now preserves embedded image bytes and uses `createImageBitmap` + canvas/OffscreenCanvas readback when no custom `decodePixels` hook is supplied, so ordinary browser PNG/JPEG/WebP-style image decoding no longer needs host glue for the decoded path. In Node hosts, embedded or fetched PNG/JPEG/WebP raw-image handles now fall back to built-in `pngjs` / `jpeg-js` / `webp-wasm` decoders without a host `decodePixels` hook, preserving the same sRGB-vs-linear and `webgpu` target color-space policy as custom decoders. The `webgpu` target also resolves raw-image handles through the same platform/custom pixel path, but preserves the field's backend upload color space (`baseColorMap`/`emissiveMap`/tint maps remain sRGB-valued; data maps remain linear). `textureDecodeReport.entries[]` now exposes both the material role `colorSpace` and decoded `handleColorSpace` when known. 2026-06-19 source-provenance follow-up: `decodePixels` receives texture/image/sampler/MIME/source-extension metadata, and all decode diagnostics include the same fields when present. Same-day compatibility cleanup: once the high-level decode path proves every emissive map is CPU-readable, PT profiles drop the conservative `emissiveMap.texelPdf` approximate issue while walkaround keeps its broader GI/RC/DDGI approximation warning. Compressed/transcoded texture sources such as Basis/KTX2/DDS still require host decode/transcode support where built-in PNG/JPEG/WebP/platform decode is insufficient; automatic mip generation and broader backend map consumption remain separate Road rows.

✅ **HDRI HANDLE FOLLOW-UP (2026-06-19):** PT environment maps now share the same CPU-readable handle posture as material maps. `@vitrum/shared-samplers` exports `readEnvironmentMapPixels()`, and pt-webgpu plus pt-webgl2 both accept raw `{width,height,data}` and DataTexture-shaped `{image:{width,height,data}}` HDRI handles with optional `__vitrum_hint__` channels/dataType/colorSpace, including uint8 sRGB normalization, before building environment CDFs. Opaque image/file handles still require adapter/host decode or a future backend-ready texture contract; they remain structured `*.hdri-unreadable` diagnostics rather than silent drops.

✅ **FOLLOW-UP (2026-06-15):** the engine bridge now exposes that decode path
through the one-call APIs. `loadGltfForEngine()` chooses
`loadGltfAndDecodeTextures()` when `decodeTextures` or any decode-specific option
is supplied, attaches/constructs the engine with the decoded scene, and returns
decode counts, diagnostics, and warnings. `@vitrum/engine/gltf` forwards the same
surface through both `loadGltfWithEngine()` and
`loadGltfWithProgressiveEngine()`.

✅ **FOLLOW-UP (2026-06-15):** the behavioral glTF gate now exercises the same
one-call decode bridge on a rendered asset. `pt/gltf-textured-pbr` feeds a raw
image handle into `loadGltfForEngine({ decodeTextures:true,
textureTarget:'cpu-linear', decodePixels })`, asserts the decoded/unchanged
counts, asserts a backend-ready `textureDecodeReport` row, verifies the engine
was attached to the decoded controller scene, and renders the resulting
pt-webgpu frame.

✅ **FOLLOW-UP (2026-06-15):** walkaround-hybrid textured alpha traversal now multiplies baseColorMap `.a` with optional `alphaMap.r`, so glTF assets that store MASK/BLEND coverage in `pbrMetallicRoughness.baseColorTexture.a` are honored without adapter-side fake `alphaMap` aliases. Tests: `gltfAdapter.test.ts` verifies the glTF boundary and `materialTextureAtlas.test.ts` verifies atlas/shader coverage.

✅ **FOLLOW-UP (2026-06-17):** browser-default `ImageBitmap` texture handles are now
called out explicitly in `textureDecodeReport.imageBitmapCount` /
`imageBitmapRefs`. They are treated as pt-webgpu external-image-ready but opaque
to CPU-atlas backends, so strict degraded loads either require the host to opt in
with `opaqueTextureHandlesReady` or use `decodeSceneTextures()` /
`loadGltfAndDecodeTextures()` with `decodePixels` to produce role-aware
CPU-linear atlas payloads. The old `createImageBitmap` sRGB footgun is therefore
closed for the adapter decode bridge and preserved only as an explicit host
contract boundary for opaque browser handles.

✅ **FOLLOW-UP (2026-06-18):** `loadGltfAsset()` now reconciles
`backendCompatibility` / `recommendedBackend` against the actual
`textureDecodeReport`, not just the static JSON feature scan. Opaque or ignored
map handles become structured `texture:texture-readiness:*` issues, backend
ranking is re-run after import and after decode, and `loadGltfForEngine()` treats
those issues consistently with its existing `opaqueTextureHandlesReady` strict
mode escape hatch. Disabled texture-source-extension diagnostics also point to
the exact extension path when only one source extension is present.

#### 4D — Animation + temporal GI

| Concern | Code | Footgun |
|---------|------|---------|
| ~~ReSTIR temporal reset~~ ✅ CODE CLOSED | `GltfSceneController` calls `reset()` after successful incremental animation/variant `updatePrimitive()` batches when the target exposes it; `setScene()` fallback keeps the normal full-scene invalidation path; `ProgressiveHandoffCoordinator` exposes the same reset hook on its synthetic controller target. | Custom non-engine patch targets that intentionally omit `reset()` remain responsible for their own history invalidation |
| ~~DDGI probe invalidation~~ ✅ CODE CLOSED | `updatePrimitive` material vs transform | TLAS and merged transform/positions/topology refits invalidate DDGI probes. Material-only patches skip geometry propagation, but now refresh DDGI's material snapshot; radiance/visibility changes plus roughness/metallic scalar edits invalidate probe cache. Test: `walkaround-hybrid/src/__tests__/mutationMatrix.test.ts`. |
| ~~pt-webgpu accum~~ ✅ SOURCE VERIFIED | `attachVitrum` tracks matrices and forwards `prevViewMatrix` / `prevProjMatrix` through `composeAttachVitrumFrameInput()` from the second RAF tick onward; the first tick has no prior frame by design. | Broader animated real-scene sweeps remain validation tails |
| ~~Skinning pose patches~~ ✅ CODE CLOSED | `GpuSkinningSubsystem` vs CPU `solveSkin` | `updatePrimitive({ bones | boneInverses | morphWeights })` now solves skinned pose patches through the canonical solver and routes them through the TLAS/merged positions-refit path; GPU-skinned TLAS refits invalidate DDGI probes too. Broader animated real-scene/GPU captures remain validation tails. |

#### 4E — Engine integration residue (H31)

| Item | File |
|------|------|
| ~~`backendId` on attach handle~~ ✅ CODE CLOSED | `vanilla.ts` exposes `AttachVitrumHandle.backendId` as a live getter over the current engine, so auto-recreate swaps are observable without re-grabbing `handle.engine`. Test: `attachVitrumAutoRecreate.test.ts`. |
| ~~`createProgressiveEngine` `onError` on canvas configure~~ ✅ CODE CLOSED | `createProgressiveEngine.ts` reports the final best-effort `configureWebGpuCanvas()` failure with `{ phase:'canvas-configure', backend:'walkaround-hybrid', recoverable:true }`. Test: `createProgressiveEngineCanvasError.test.ts`. |
| ~~`analyticPrimitiveToMesh` UVs~~ ✅ SOURCE-VERIFIED STALE | `packages/core/src/scene/analyticToMesh.ts` generated meshes carry `uvs`, and `analyticToMesh.test.ts` pins UV ranges plus fallback UV cloning. |
| ~~`idempotentDispose` errors~~ ✅ SOURCE-VERIFIED STALE | `idempotentDispose.ts` exposes `onDisposeError` for backend/post-dispose throws instead of swallowing them silently; existing proxy-table coverage keeps disposed behavior pinned. |

#### 4F — Extensions not yet in spec (gap fill for true arbitrary glTF)

| Extension | Status | Action |
|-----------|--------|--------|
| ~~`EXT_mesh_gpu_instancing`~~ ✅ CODE/TEST CLOSED | Node-level instancing imports to core `instanced-mesh`; required use is accepted; compatibility reports the supported extension and expected native primitive kind with the node source path. Direct and `loadGltfForEngine()` bridge-created `GltfSceneController` instances store local instance matrices and patch `instances[]` for animated instanced nodes/ancestors. Invalid accessor payloads warn and import the base mesh once. Skinned/morphed instancing now also preflights as `EXT_mesh_gpu_instancing.skinnedOrMorphed=unsupported`, so `reject-unsupported` catches the combined unrepresentable case before engine construction while best-effort import still emits `ignored-gpu-instancing` and imports the skinned/morphed representation once. | Remaining tail is true instanced skinned/morphed primitives, which needs a future core contract rather than adapter/controller plumbing. |
| ~~`KHR_texture_basisu` / texture-source hooks~~ ✅ policy closed | Host decode/transcode hook required when selected, required, or no base `texture.source` fallback, unless the one-call texture decode bridge resolves the selected source into CPU-readable handles and reconciles the `requires-hook` issue after decode; optional fallback assets load without degraded rejection. `GltfFeatureReport.extensions.textureSourceUses[]` now lists each KTX2/WebP/DDS alternate with texture index, image index, source path, MIME type, selected/fallback/required status, and `requiresHook`; `loadGltfAsset()` analyzes the same selected `textureSourceExtensions` that conversion uses. 2026-06-19 follow-up: selected KTX2/DDS sources that reach `decodeSceneTextures()` without a decoder now emit source-extension/image/MIME provenance in `raw-image-decoder-missing` diagnostics instead of an unqualified raw-image warning. | Vitrum intentionally does not bundle a KTX2/Basis/WebP/DDS transcoder; hosts opt in through `textureSourceExtensions` + `decodeImage`/`decodePixels` where the built-in platform/Node decode path is insufficient. |
| ~~`EXT_meshopt_compression` fallback buffer / codec-hook smoke~~ ✅ CODE/TEST CLOSED | Optional meshopt bufferViews with real fallback buffers import without a decode hook (`gltfCompression.test.ts`) and now analyze as hook-free compatible in the Khronos-style sweep (`featureReport.ts`, `gltfKhronosSweep.test.ts`); required meshopt or fallback-stub assets still require a host hook. 2026-06-16 follow-up: dev-only `draco3d` and `meshoptimizer` smoke fixtures now encode/decode tiny real codec payloads through the documented host hooks, so the hook signatures are no longer proven only by synthetic stubs. |
| ~~Point/line fallback loader options~~ ✅ CODE/TEST CLOSED + behavioral golden | `loadGltfAsset()` now forwards `pointLineFallbackRadius` to `gltfToScene()`, so public loader and `loadGltfForEngine()` callers get the same host-selected generated-mesh radius that low-level converter callers do for glTF `POINTS`, `LINES`, `LINE_LOOP`, and `LINE_STRIP`. 2026-06-17 follow-up: `pt/gltf-point-line-fallback` in `tools/behavioral-gate/gate.mjs` imports all four topologies through `loadGltfForEngine()`, asserts source-pathed `fallback-generated-primitive-mode` diagnostics, renders the generated mesh fallback on the adapter, and compares against `tools/reference-renders/gltf-point-line-behavioral/pt-gltf-point-line-fallback.png`; `npm run gltf-topology-proof-check` now verifies the lane's manifest, proof metadata, thresholds, and PNG presence. | Native point/line primitives remain future core/backend contract work; current support is intentionally `fallback-generated-mesh`. |
| ~~Multiple UV sets~~ ✅ CODE/TEST NARROWED | `TEXCOORD_1` imports/consumes as `uv1`; generated tangent frames use `TEXCOORD_1` when tangent-space maps select `texCoord:1`. 2026-06-16 follow-up: if a primitive material references exactly one higher glTF UV set (`TEXCOORD_N`, `N > 1`), the primitive has that accessor, and the material does not also need `texCoord:1`, `gltfToScene()` losslessly loads that higher accessor into core `uv1` and rewrites the primitive-local `TextureRef.texCoord` values to `1`; compatibility no longer rejects those remappable cases. | Native arbitrary UV-array support remains future core/backend contract work. Conflicting high UV sets, high UV plus material-visible UV1, or missing high-UV accessors emit structured `ignored-material-texcoord` / compatibility issues and drop the affected texture fields instead of silently sampling UV0. |
| ~~`KHR_materials_emissive_strength`~~ ✅ CODE/TEST CLOSED | Imports to `MaterialSpec.emissiveIntensity`; planner now asserts the scalar is supported on pt-webgl2, pt-webgpu full/lite, and walkaround. Backend evidence: pt-webgl2 packs `s2.a` and GLSL multiplies emission; pt-webgpu material packing and implicit emitter tests pre-multiply intensity; walkaround/shared-BVH `materialSpecEmissiveLe` now defaults missing intensity to ×1 and tests HDR `emissive · emissiveIntensity` classification. |
| ~~Draco hook/fallback policy~~ ✅ CODE/TEST CLOSED | Import and analysis now agree: required Draco assets require `opts.dracoDecode` even when fallback accessors exist; declaration-only optional Draco and optional Draco primitives with complete uncompressed fallback accessors are hook-free; optional Draco primitives without usable fallback accessors still report `requires-hook`. | Keep the host-supplied decoder contract; Vitrum intentionally does not bundle a Draco decoder. |

---

### Phase 5 — Closure: prove 100% (not 85%)

#### 5A — Material furnace + glTF sweep

✅ **CPU PREFLIGHT ADDED (2026-06-15):** `tools/gltf-material-sweep/` plus
`npm run gltf-material-sweep` now builds a synthetic material-heavy glTF asset,
loads it through `loadGltfAndDecodeTextures()`, asserts every base/KHR material
texture appears in `textureDecodeReport`, verifies `KHR_texture_transform`
`texCoord` survives as uv1, asserts the generated sampler/wrap/mipmap policy in
the decode report, checks the expected `*.samplerPolicy` compatibility
diagnostics per backend, and checks backend-readiness diagnostics for pt-webgl2,
pt-webgpu, and walkaround-hybrid. This caught and fixed the stale walkaround
`thicknessMap` readiness classifier: the adapter now reports it `ready`,
matching the shipped walkaround atlas/shader path.

For each fixture in `tools/reference-assets/gltf/`:
1. `loadGltfAsset` + `decodeSceneTextures`
2. `evaluateGltfBackendCompatibility` for each backend
3. Render 64spp on **recommended** backend
4. Assert `meanLum > ε`, no GPU validation errors
5. Compare hash to golden PNG (tolerance for MC noise on PT)

✅ **FOCUSED GPU BOOT/READBACK ADDED (2026-06-16):** the same synthetic
material-heavy fixture is now shared with `tools/behavioral-gate/gate.mjs` and
runs as `pt/gltf-material-sweep`. The gate drives the asset through
`loadGltfForEngine({ decodeTextures:true, textureTarget:'cpu-linear' })`,
asserts all 18 decoded texture-report rows and CPU-readable handles survived
controller attachment, boots `pt-webgpu`, renders 8 spp at 64², and requires
finite non-black readback with zero GPU validation errors.

✅ **REAL-ASSET IMPORT/RENDER SMOKE ADDED (2026-06-17):**
`tools/gltf-real-asset-sweep/assets.mjs` now owns the shared Khronos public
asset manifest plus PNG/JPEG/Draco/meshopt decode hooks. `npm run
gltf-real-asset-sweep` proves URL load, texture decode, and compression decode
for BoxTextured GLB, CesiumMilkTruck Draco, and MeshoptCubeTest meshopt assets.
`tools/behavioral-gate/gate.mjs` also renders those same assets as
`pt/gltf-real-{box-textured,draco,meshopt}` through `loadGltfForEngine()`,
normalizes the imported scene to the gate camera, boots `pt-webgpu`, and
requires finite non-black readback with zero captured GPU errors.

✅ **REAL-ASSET GOLDEN/TOLERANCE GATE ADDED (2026-06-17):**
`tools/reference-renders/gltf-real-behavioral/` now contains committed 64x64
lavapipe PNG baselines for the same three real public assets. The normal
`npm run behavioral-gate -- --filter gltf-real` path reads those PNGs and fails
the config as `GOLDEN-DELTA` if byte RMSE, mean absolute error, or max-channel
delta exceeds the per-asset tolerance. Recapture is explicit via
`tools/behavioral-gate/gate.mjs --filter gltf-real --update-goldens` with
write permission to the reference-render directory. The 2026-06-17 proof
metadata follow-up adds `tools/gltf-real-asset-sweep/proofs.mjs` plus
`tools/reference-renders/gltf-real-behavioral/manifest.json`, so the import/decode
sweep reports `renderStatus:"covered-by-behavioral-gate"` with the exact proof
label, golden path, and tolerances for each committed real asset.

✅ **SYNTHETIC MATERIAL-SWEEP GOLDEN/TOLERANCE GATE ADDED (2026-06-17):**
`tools/reference-renders/gltf-material-sweep-behavioral/` now contains the
committed 64x64 lavapipe PNG baseline for `pt/gltf-material-sweep`. The normal
`npm run behavioral-gate -- --filter gltf-material-sweep` path reads that PNG
and fails the config as `GOLDEN-DELTA` if byte RMSE, mean absolute error, or
max-channel delta exceeds the tolerance. The proof metadata follow-up adds
`tools/gltf-material-sweep/proofs.mjs` plus
`tools/reference-renders/gltf-material-sweep-behavioral/manifest.json`, so
`npm run gltf-material-sweep` now reports
`renderStatus:"covered-by-behavioral-gate"` with the exact proof label, golden
path, and tolerances instead of a stale queued status.

✅ **TOPOLOGY GOLDEN METADATA CHECK ADDED (2026-06-17):**
`tools/gltf-topology-proofs/` now owns proof metadata for the committed
`pt/gltf-point-line-fallback` and `pt/gltf-triangle-strip-fan` PNG lanes. The
behavioral gate consumes those shared labels, paths, and tolerances instead of
hard-coded topology golden entries, and `npm run gltf-topology-proof-check`
verifies both manifests, proof metadata, 64x64/8spp expectations, and PNG
headers.

✅ **FULL-TIER MATERIAL-SWEEP DZN STATUS ADDED (2026-06-17):** WSL lavapipe still
proves the default/auto glTF material-sweep PNG path in the lite profile. The
companion dzn runtime now has a committed machine-readable status artifact for
the same fixture on the pt-webgpu full tier:
`npm run behavioral-gate:dzn -- --filter gltf-material-sweep --require-full-tier`.
`tools/behavioral-gate/behavioral-gate-dzn-gltf-material-sweep-status.json` records
`verdict:"PASS"`, `tier:"full"`, zero GPU errors, `nan:false`, and golden
metrics within manifest thresholds (RMSE 0.544 ≤ 8, meanAbs 0.070 ≤ 4,
maxAbs 16 ≤ 48). `npm run gltf-material-proof-check` now verifies both the
lavapipe golden metadata and the committed dzn full-tier status artifact.
`tools/behavioral-gate/gate.mjs` prints the resolved `tier=full|lite`, and
`--require-full-tier` actively requests full tier before failing any selected
non-lite pt-webgpu config that cannot resolve there.
Broader material-furnace/reference sweeps for individual clearcoat/sheen/
iridescence/aniso/specular rows remain queued, but a green lite run can no
longer be mistaken for this full-tier synthetic material-sweep evidence.

✅ **RADIOMETRIC A/B FALSE-POSITIVE GUARDS ADDED (2026-06-17):**
`tools/radiometric-ab/{ab-sppm,ab-bdpt,ab-restir-pt}.mjs` now force
`traceTier:"full"` and pass harness-only `requireFullTier` +
`requireRadiometricSignal` flags through `helpers.mjs`. The helper strips those
flags before engine construction, fails when a required full-tier A/B resolves
to lite, and fails when a linear-HDR capture is black (`mean luminance <=
1e-5`). This does **not** close the full-adapter V28-B recapture queue; it
prevents the old failure mode where two lite/black arms could write a misleading
PASS JSON.

✅ **RADIOMETRIC RESULT-SNAPSHOT / HOST-STATUS CHECK ADDED (2026-06-17):**
`tools/radiometric-ab/proofs.mjs` plus `check-results.mjs` now make the committed
SPPM, safe-default BDPT, ReSTIR-PT result JSONs, ReSTIR-PT specialty fixture,
and walkaround host-status JSON machine-checkable without a new GPU recapture.
`npm run radiometric-ab:proof-check` verifies the expected 80x80 resolution,
frame counts, PASS verdicts, per-harness relative-error and variance thresholds,
SPPM checkpoints, the explicit BDPT multi-vertex research-mode finding recorded
in `results-bdpt.json`, specialty-lobe `requiresGpuRecapture:false`, and the
walkaround host-status/result marker. **2026-06-18 update:** the native WSL
walkaround A/B host now runs to completion and records `PASS-PARTIAL` rather
than `HOST-BLOCKED`; the proof checker requires the do-not-promote warning while
the SUN analytic lane remains partial. This is still snapshot/status evidence;
full-adapter V28-B recaptures remain the promotion queue.

✅ **READ-ONLY PROOF BUNDLE ADDED (2026-06-17):** `npm run proof-check` runs the
glTF material, real-asset, topology, radiometric snapshot/status, and ReSTIR-PT
specialty fixture checks as one non-recapture evidence gate.

**Footgun:** Testing only `analyzeGltfAsset` without render proved glTF API "done" but left textures black.

#### 5B — Oracle suite (keep green)

| Oracle | File | Regression guard |
|--------|------|------------------|
| ~~PTWG-BDPT-01~~ ✅ code/proof closed | `oracle.bdptConnectionCosine.test.ts`; `bdptGlossyLightSubpath.test.ts`; `bdptConnectionMisFull.test.ts` | Finite-area endpoint, one-bounce diffuse light tracing, and non-Lambertian light-vertex connection including CPU MIS-assembly PDF overrides |
| ~~PTWG-MAT lobe CPU proof~~ ✅ code/proof closed | `extensionLobeReference.test.ts` | Clearcoat zero-default/linearity, sheen color/scalar behavior, normalized base/clearcoat/sheen sampled PDF, iridescence F0 zero-default, and explicit sheen-PDF approximation posture |
| ~~PTWG-MAT-BUMP-01~~ ✅ code/proof closed | `materialTextures.test.ts` | Bump-map finite-difference source dimensions now have a CPU oracle: raw-UV forward differences use `textureDimensions(materialTexturesLinear) * uvFitScale` instead of a fixed texel step, and the resulting Blinn-style perturbed normal is pinned numerically while WGSL source linkage keeps the shader tied to the behavior. |
| ~~PTWG-VOLUME-DIRNEE-01~~ ✅ code/proof closed | `volumetricSss.test.ts` | In-medium directional NEE over packed N-directional records now has an arithmetic oracle: two RGB directional lights plus a zero-mean gated record are accumulated as `throughputInMedium * irradiance.rgb * hgPhase(dot(ray.direction, towardLight), g)` with literal radiance expectations. This proves the no-occluder packed-light estimator; shadowed volume visibility remains render/A-B validation. |
| ~~PTWG-ADJOINT-PACK-01~~ ✅ host-packing proof closed | `adjointPassPacking.test.ts` | `AdjointPass.computeGradient()` now has direct host-side regression coverage for the seam between inverse-session request capture and WGSL execution: `req.samples` is written to UBO slot 27, mesh-area replay count replaces the live packed count, emitter descriptor metadata marks `ADJOINT_EMITTER_TARGET_MESH`, and bindings 13/22 swap to adjoint replay buffers for explicit mesh-area emitter fits. |
| ~~HYB-GI-01/02~~ ✅ code/proof closed | `oracle.restirDiEstimator.test.ts` | RIS candidate accounting, selected-xi p-hat/finalize, selected-point shading, env + area DI characterization |
| ~~HYB-DDGI-01~~ ✅ code/proof closed | `oracle.ddgiVisibilityMoments.test.ts` | Probe miss visibility, all-sky open semantics, f32/f16 moment poisoning regression |
| ~~PTWG-LITE-01~~ ✅ code/proof closed | `oracle.liteRectMis.test.ts` | Lite rect/disc analytic records now have an independent CPU oracle: the historical one-sided MIS under-estimate is preserved as a failing-shape diagnostic, and the regression case adds light-sampled + BSDF-sampled shares and requires agreement with ground truth within 3%. |

#### 5C — Mutation matrix GPU observability

✅ NON-GPU SEAM CLOSED for the current mutation matrix: `walkaround-hybrid/src/__tests__/mutationMatrix.test.ts`
pins observable pipeline/DDGI/RC collaborator calls for `updatePrimitive()`,
`updateEmitter()`, `updateEnvironment()`, `updateLighting()`, and `setSize()`;
`pt-webgpu/src/__tests__/updatePrimitiveIncremental.test.ts` pins buffer
create/destroy/write counts for primitive material, geometry, transform,
topology, instanced, lite fallback, and analytic paths; and
`pt-webgpu/src/__tests__/mutationDesyncs.test.ts` now pins router-level
`updateEmitter()` point-light buffer writes plus same-sized HDRI
`updateEnvironment()` texel/CDF writes, scene-state commit, and accumulation
reset without falling through to `setScene()`, and directly invokes
`GpuResources.clearReservoirBuffers()` to prove ReSTIR-PT Cur/Prev/Spatial
history buffers are cleared through the production method instead of a local
test sketch.

✅ **ADAPTER-BACKED PT MATERIAL/ENVIRONMENT/EMITTER/TRANSFORM/TOPOLOGY/INSTANCED-COUNT/ADD-REMOVE MUTATION PROOFS ADDED (2026-06-17):**
`tools/behavioral-gate/gate.mjs` now includes real pt-webgpu mutation lanes that
render, patch, render again with the same camera/seeds, and require measurable
readback deltas (`meanAbs >= 2`, `maxAbs >= 8`) with zero GPU errors:
`pt/mutation-material` proves `updatePrimitive()` material patches change
GPU-visible output, `pt/mutation-environment` proves same-sized
`updateEnvironment()` HDRI patches propagate through environment texel/CDF
buffers, light-tree/lite-texture refresh, reset, and GPU-visible miss radiance,
`pt/mutation-emitter` proves `updateEmitter()` point-light patches propagate
through emitter buffers/light-tree refresh/reset to GPU-visible direct lighting,
`pt/mutation-transform` proves `updatePrimitive()` transform patches update
TLAS state into GPU-visible geometry movement on the full-tier backend,
`pt/mutation-topology` proves vertex/index-count topology resize patches rebuild
the BLAS/TLAS geometry buffers into GPU-visible output without a full scene reset,
`pt/mutation-instanced-count` proves instanced-mesh instance-count changes
reallocate TLAS buffers, reuse BLAS geometry, and produce GPU-visible output,
and `pt/mutation-add-primitive` / `pt/mutation-remove-primitive` prove
whole-primitive fallback resource rebuilds produce GPU-visible add/remove output.
This moves these seams beyond mock write-count tests on the available WSL adapter.
2026-06-17 follow-up: the same eight mutation lanes now also pass through
`npm run behavioral-gate:dzn -- --filter mutation --require-full-tier` on the
companion full-tier WSL dzn runtime, with committed machine-readable evidence in
`tools/behavioral-gate/behavioral-gate-dzn-mutation-status.json`. The dzn status
records `tier:"full"`, zero GPU errors, `nan:false`, and mutation deltas above
the gate thresholds for material (`meanAbs=41.582`, `maxAbs=174`), environment
(`meanAbs=136.250`, `maxAbs=190`), emitter (`meanAbs=36.566`, `maxAbs=187`),
transform (`meanAbs=38.924`, `maxAbs=230`), and topology
(`meanAbs=48.359`, `maxAbs=230`), plus instanced-count
(`meanAbs=15.869`, `maxAbs=230`), add-primitive
(`meanAbs=12.375`, `maxAbs=232`), and remove-primitive
(`meanAbs=12.375`, `maxAbs=232`);
`npm run behavioral-gate:dzn-status-check` verifies the artifact. That run also
exposed and fixed a real primitive-less full-tier validation bug: empty
`bvhNodes`/`tlasNodes` uploads used 16-byte generic placeholders while the WGSL
`array<BVHNode>` bindings require a 32-byte minimum stride.

2026-06-18 source-test follow-up: `pt-webgpu/src/__tests__/mutationDesyncs.test.ts`
now also pins cached bind-group invalidation for reallocating mutation fast
paths before commit/reset: vertex/index-count topology patches and
instanced-mesh count changes both call `invalidateBindGroups()`, update scene
state, reset accumulation, and avoid falling through to `setScene()`. Cached
bind-group coverage here is specifically the pt-webgpu reallocating-mutation
seam, not a blanket claim about every future cache.

2026-06-18 denoiser-history source/shader follow-up:
`WalkaroundGPUPipeline.requestAccumReset()` schedules the next frame with
`frameIndex === 0`, and walkaround temporal denoisers now share that as a
history-reset signal: à-trous-variance writes Welford `forceReset=1`, BMFR
emits `hasHistory=0`, and SVGF-real packs `forceReset` into the existing
16-byte reprojection UBO pad slot so the shader rejects prior history and writes
current-frame samples. `shared-denoisers` CPU/shader/packing tests and
walkaround denoiser-policy tests pin the seam. Remaining proof is broader
adapter-backed end-to-end promotion for walkaround GI propagation under the WSL
GPU/browser harness.

#### 5D — Documentation sync (part of 100% — prevents false claims)

| Artifact | Action |
|----------|--------|
| `BACKEND_PROMISE_LEDGER` | Sole truth; READMEs cite ledger not prose |
| `plan/renderer-fidelity-matrix.md` | Remove deleted `pt-webgl` column; add pt-webgl2 |
| `items_to_fix.md` §H | Close items as fixed or strike |
| ~~H30~~ ✅ CLOSED | Canvas backing store sizing is now applied before engine construction; `attachVitrumLoop.test.ts` pins CSS×DPR sizing |
| ~~H57~~ ✅ CLOSED | `examples/gltf-viewer/` added; `examples/README.md` lists the glTF path and debug capture fields. |

#### 5E — Behavioral gate expansion

✅ CLOSED (Wave 10 + material-sweep follow-up): behavioral gate now includes six adapter-backed glTF
fixtures on the runnable `pt-webgpu` lane: unlit (`KHR_materials_unlit`),
textured PBR (`baseColorTexture` through the decode hook), transmission glass
(`KHR_materials_transmission`), skinned animation (skin + animation channel
import), Draco (mock `KHR_draco_mesh_compression` decoder), and the synthetic
material sweep fixture covering base/KHR texture-map decode/report plumbing.
`tools/behavioral-gate/gate.mjs` now prepares those fixtures through
`loadGltfForEngine()` with an injected patch-target engine, asserts the
controller attached and called `setScene()`, advances the skinned-animation
controller and requires an `updatePrimitive()` patch, asserts the material
sweep's 18 CPU-readable decoded map rows, then boots the real renderer with the
prepared scene and requires finite non-black output.
`--filter gltf` provides the focused lane.

✅ **HDRI fixture shape corrected (2026-06-17):** the behavioral-gate Cornell
HDRI helper now supplies the core `HdriEnvironment` contract
(`{ kind: 'hdri', hdri: { width, height, data } }`) instead of a stale
`textureData` shape. This restored `pt/lite+hdri` from a black validation
fixture to a finite, non-black adapter-backed render.

Honesty boundary: on the WSL lavapipe adapter the default `--filter gltf` lane
runs through `pt-webgpu`'s lite tier because the adapter exposes 8 storage
buffers / 4 storage textures per shader stage, below the full-tier 34 / 5
requirement. The dzn companion lane can prove selected full-tier pt-webgpu cases
(`npm run behavioral-gate:dzn -- --filter gltf-material-sweep
--require-full-tier` pins the material-heavy fixture, and
`npm run behavioral-gate:dzn -- --filter mutation --require-full-tier` pins
material/environment/emitter mutation observability). Broader
material-lobe fidelity promotion remains owned by the renderer fidelity matrix
and package material oracles.

2026-06-17 full-suite dzn closure: `npm run behavioral-gate:dzn -- --filter gltf
--require-full-tier` now writes
`tools/behavioral-gate/behavioral-gate-dzn-gltf-status.json`. That artifact
confirms full-tier execution, zero GPU errors, and `nan:false` for all 11 glTF
lanes. The five synthetic/import lanes plus point/line, strip/fan, material
sweep, and all three real-asset lanes now pass. The real-asset lanes use the
explicit `dzn-full` golden variant under
`tools/reference-renders/gltf-real-behavioral-dzn-full/`, because the full-tier
dzn path is not judged against lavapipe-lite PNGs. Treat this as closed for
pt-webgpu dzn full-tier glTF proof; pt-webgl2/browser recommended-backend proof
remains queued separately.

2026-06-17 pt-webgl2 browser status: `tools/gltf-browser-proof/` now drives the
real `BoxTextured.glb`, `CesiumMilkTruck` Draco, and `MeshoptCubeTest` meshopt
assets through `examples/gltf-viewer` and the public `loadGltfWithEngine()` path
with `backend:'pt-webgl2'`. The example runs the browser texture decode bridge
(`decodeTextures` + `decodePixels`) so texture payloads reach pt-webgl2 as
CPU-readable pixels instead of opaque `ImageBitmap` handles, and it now wires
browser-side Draco WASM plus meshoptimizer decode hooks for the compressed
assets. On this WSL Playwright host all three pages reach capture readiness and
the committed `pt-webgl2-real-status.json` proves the required extension/hook
telemetry (`KHR_draco_mesh_compression` + `draco`, `KHR_meshopt_compression` +
`meshopt`), but WebGL canvas readback (`toDataURL`) stalls; the status records
each row as `HOST-BLOCKED` at `canvas-readback`. Browser PNG proof remains
queued for a host that can read back WebGL2 canvases.

Validation note: the walkaround-hybrid native-Deno behavioral lane is
fail-closed on this WSL adapter. Deno 2.8.1 / wgpu-hal can panic before the
pre-existing `wh/default` config returns a renderer verdict; the npm wrapper now
classifies that as `HOST-BLOCKED` and writes
`tools/behavioral-gate/behavioral-gate-host-status.json` instead of letting the
gap look like a renderer failure. The radiometric walkaround lane is no longer
host-blocked on this WSL setup: `npm run radiometric-ab:walkaround` now records
`PASS-PARTIAL` in `tools/radiometric-ab/walkaround-ab-host-status.json`, with a
do-not-promote warning because the low-spp SUN analytic case remains partial
and the GLASS/GLOSSY material probes are currently `PASS-WEAK` no-delta checks
(non-black/plausible, but no observed material-effect delta at 16 spp).
Walkaround glTF render-gate promotion still requires a browser/adapter harness
and higher-confidence reference captures.

---

### Master checklist: 65 material fields × walkaround path to ledger truth

| Category | Fields | Walkaround work |
|----------|--------|-----------------|
| Scalars consumed | baseColor, roughness, metallic, emissive*, transmission, ior, attenuation*, thickness, envMapIntensity, shadingModel, extensions | `shadingModel` verified `approximate`; mesh-area Le override, DDGI material-emissive direct probe hits, and HDRI envMapIntensity scaling closed; remaining scalar work belongs to atlas/lobe parity rows |
| Alpha | alphaMode, alphaCutoff, opacity, alphaMap | Scalar + alpha-map cutout code-closed in 3C/3D; fractional blend camera composition code-closed via transparent OIT with direct sun plus analytic point/spot and four-sample fixed-stratified finite-emitter lighting, plus alpha-aware direct shadow transmittance; DDGI probe direct sun/point/area-emitter visibility now samples readable baseColor/alpha-map coverage plus `COLOR_0.a` vertex alpha for blend/mask transmittance; shade/ReSTIR-DI/ReSTIR-GI/NRC/GRIS shadow visibility now evaluates readable atlas alpha coverage as binary/transmittance blocker filtering. ReSTIR-DI/GI primary and reconnection vertices now use the opaque-only first-hit predicate so fractional blend surfaces do not enter reservoirs under a different stochastic contract than shade. Transparent ReSTIR direct-light reservoir participation and GI transport remain intentionally approximate/unsupported until a true layered-transport model lands. |
| Maps (17+) | supported/readable material maps listed in 3D | 3D atlas + decode pipeline; unsupported volume/spectral/layered/displacement families remain explicit below |
| Disney scalars | specular*, clearcoat*, sheen*, anisotropy*, iridescence* | 3E; these rows are approximate after shade-owned, ReSTIR-DI, GI suffix, and receiver-lobe GI target consumption; native promotion still needs material-furnace/reference A/B where applicable |
| Volume/spectral | spectral*, scattering*, thinFilm, front/back layer | Permanent unsupported + planner routes to PT |
| Displacement | displacement* | Permanent unsupported all backends; diagnostics cover setScene, analytic authored materials, walkaround material-only mutation paths, and pt-webgl2 scalar material mutation fast paths |

**pt-webgl2 ledger residuals:** unsupported renderer fields are `displacementMap`, `displacementScale`, and `displacementBias`; `extensions` is also ledger-unsupported but host-discretionary and intentionally warning-free. Runtime `setScene()` diagnostics identify both the aggregate unsupported field list and the exact primitive/material rows that authored renderer fields; `updatePrimitive()` scalar material fast paths now emit the same structured displacement warning for `displacementScale` / `displacementBias` instead of silently accepting unsupported scalars. Approximate fields are `shadingModel`, `thickness`, `thicknessMap`, and `scatteringCoefficientRGB`; `frontLayer` and `backLayer` are native field-consumption rows after face-selected transmission/roughness plus nested normal-map/normal-scale packing and shader consumption. `emitterCastShadow` is native in the shadow matrix after folded mesh-area emitter shadow flags reached the forward emissive-hit MIS estimator.

**pt-webgpu ledger residuals:** unsupported renderer fields are `displacementMap`, `displacementScale`, and `displacementBias`; `extensions` is also ledger-unsupported but host-discretionary and intentionally warning-free. Runtime `setScene()` diagnostics identify both the aggregate unsupported field list and the exact primitive/material rows that authored renderer fields, and scalar `updatePrimitive()` material fast paths now emit the same primitive-scoped `fields` / `primitiveIds` / `primitiveFields` displacement warning shape for `displacementScale` / `displacementBias`. Approximate fields are `shadingModel`, `thickness`, `thicknessMap`, `clearcoatMap`, `clearcoatRoughnessMap`, `clearcoatNormalMap`, `sheenColorMap`, `sheenRoughnessMap`, `iridescenceMap`, `iridescenceThicknessMap`, `specularColorMap`, `specularIntensityMap`, `specularIntensity`, and `specularColor`; `frontLayer` and `backLayer` are native in the full tier after face-selected layer-normal descriptor/shader support, while the lite tier still emits structured compatibility warnings. `emitterCastShadow` is native in the shadow matrix after SPPM photon-source selection reached no-shadow parity.

---

### Phase 6 — Historical ledger residue outside the three targets (ADDED 2026-06-12; reconciled 2026-06-18)

> The three-target addendum does not retract the gap ledger's categorical close
> condition. These rows were the verified-open items not covered by Phases 0–5;
> current status is now represented row-by-row below, with closed rows retained
> for auditability and any remaining tails called out explicitly in-row.

| Item | File(s) | Fix or downgrade |
|------|---------|------------------|
| pt-webgl2 NEE 3-way selection bias | `packages/pt-webgl2/src/glsl/render/direct_light_contribution_function.glsl.js`; `packages/pt-webgl2/src/glsl/composeTraceGlsl.test.ts` | ✅ DONE (Wave 1): analytic/mesh/env NEE now use one shared strategy variate (`neeStrategyU`) with cumulative cutoffs, so slot probabilities match the PDFs. Focused source/probability tests pin the single-draw selector and the old `1/3,4/9,2/9` regression. |
| Engine `onError` shape unification | `createEngine` / `Engine.onError` / `attachVitrum.onEngineError` / `createProgressiveEngine.onError` | ✅ DONE (Wave 7, construction-event half): `createProgressiveEngine.onError` now mirrors `CreateEngineOptions.onError(error, CreateEngineErrorEvent)` instead of discarding the phase/backend/recoverability event, and its own final canvas-configure failure reports `{ phase:'canvas-configure', backend:'walkaround-hybrid', recoverable:true }`. Runtime GPU errors remain intentionally on the core `Engine.onError(EngineError)` channel; `attachVitrum.onEngineError` is the lifecycle alias for that runtime channel. Focused progressive facade tests pin both event-forwarding paths. |
| `attachVitrum` auto-recreate scene loss | `packages/engine/src/lifecycle/vanilla.ts`; `packages/engine/src/__tests__/attachVitrumAutoRecreate.test.ts` | ✅ DONE (Wave 1 + 2026-06-18 fast-path follow-up + 2026-06-19 warning pin): lifecycle tracks scenes submitted through the exposed engine handle and now snapshots `engine.getScene()` immediately before device/context-loss teardown, so add/remove/update/controller fast paths that mutate the backend-retained scene are preserved across recreate too. Regression tests simulate fatal `device-lost` for explicit `setScene`, backend-retained live-scene, and snapshot-throw cases; the throw case verifies `onError({ phase:'attach:auto-recreate', recoverable:true })` fires and the second `createEngine` call still receives the last tracked scene. |
| pt-webgpu trace-lite shader-gate mismatch | `packages/pt-webgpu/src/wgsl/pathTrace/causticLite.wgsl.ts`; `kernelLite.wgsl.ts`; `wgslContract.test.ts`; `wgslLiteContract.test.ts` | ✅ DONE (Wave 1): lite MNEE stub signature now matches the lite kernel material-extension call shape, and lite BSDF-environment reconnection receives the scalar clearcoat/sheen/iridescence fields it already evaluates. `npm run shader-gate` compiles `pt-webgpu/trace-lite`; contract tests pin stub/caller parity and the updated lite SHA/length. |
| ~~Lite tier single-BLAS~~ ✅ DONE (Wave 8 + 2026-06-17 material/light smoke closure + 2026-06-18 multi-directional closure + lite mutation fallback closure) | `packages/pt-webgpu/src/scene/uploadSceneBuffers.ts`; `packages/pt-webgpu/src/index.ts`; `SceneMutationRouter.ts`; `scenePack.test.ts`; `liteTierCapabilities.test.ts`; `updatePrimitiveIncremental.test.ts`; `tools/behavioral-gate/gate.mjs` | Lite `setScene()` now requests `buildPackedScene(..., { geometryMode:'merged' })`, which uses `mergeWorldSpaceFromCore()` to bake mesh/skinned/instanced primitives, non-identity transforms included, into one world-space BLAS rooted at node 0. The lite shader's root-0 traversal now sees multi-primitive static scenes without TLAS bindings. Static `instanced-mesh` is advertised as native; material patches update the lite material buffer in-place (`supportDetails.mutations.material='native'`). Mesh/skinned geometry/transform/topology patches and instanced transform/topology patches are accepted through a structured fallback merged-BLAS repack (`supportDetails.mutations.positions/transform/topology='fallback-rebuild'`, warning code `pt-webgpu.lite-update-primitive-fallback-rebuild`) instead of throwing or silently falling through; true BLAS/TLAS-native lite mutation remains out of scope because the lite shader intentionally binds one baked BLAS and no TLAS. The behavioral gate now uses a front-facing pt smoke scene for light rows, and adapter-backed WSL lite runs prove non-black output for rect/default, point, spot, directional, and disc emitters; multiple directional emitters are represented by packed `liteLightTex` records instead of first-directional-only UBO fallback. |
| RC exported-surface footguns | `cascadePyramid.ts`; `cascadeDispatch.ts`; `HybridEngineRC.ts`; `cascadeDimsOverride.test.ts`; `cascadeDispatchInvalidation.test.ts`; `rcMergedRefit.test.ts` | ✅ DONE (Waves 4-5): `validateCascadeDims()` rejects empty/malformed overrides, non-positive probes, non-square ray counts, broken 2× ray-grid steps, and invalid intervals before allocation/dispatch. `RCSubsystem.refitCascadeBounds()` invalidates dispatcher bindings when probe bounds change, so merge uniforms rebuild with fresh `probeOriginWorld`/`roomSize`; merged-instance refit test pins invalidation without dispatcher recreation. Raw `RCDispatcher.dispatchFrameRaw()` now self-invalidates cached bind groups when direct callers change `bvhMode`, buffer sets, env bindings, device, cascade output buffers, or cascade bounds; focused tests pin stable-frame reuse plus TLAS/bounds rebuilds. README/package-boundary docs are reconciled around the no-`/three` package surface. |
| RC direct-light glass visibility | `packages/walkaround-rc/src/wgsl/rcLightEval.wgsl.ts`; `packages/walkaround-rc/__tests__/rcLightEvalWgsl.test.ts`; `packages/walkaround-hybrid/src/HybridEngineRC.ts`; `packages/walkaround-hybrid/__tests__/rcMergedRefit.test.ts` | ✅ DONE (Wave 6): rect-area emitter NEE and point/spot fixture direct-light shadow rays now use `rcTraceAny(..., skipGlass=true)` instead of closest-hit occlusion, so transmissive geometry no longer fully blocks coarse RC direct light. Merged-mode RC now uploads the same canonical `bvhIndex.w` payload as the ReSTIR/TLAS path, so `trans4` glass filtering works outside TLAS mode too. Tests pin both the WGSL call sites and the merged-mode glass payload. |
| shared-bvh sampled fingerprint in correctness path | `bufferFingerprint.ts` + `sceneBvh.ts:131`; `bufferFingerprint.test.ts`; `sceneBvhVersionTag.test.ts` | ✅ DONE (Wave 3): sampled `fingerprintBuffer(s)` remains available for versioning/upload heuristics, but `SceneBvh.updateFromCore()` now uses exact `fingerprintBuffersExact()` for the rebuild-skip gate. Regression tests pin an unsampled interior-byte miss in the sampled helper and prove the large-scene no-tag path rebuilds instead of keeping stale buffers. |
| `solveSkin` morph-normal silent skip | `packages/core/src/skinSolver.ts:242`; `packages/core/src/__tests__/skinSolver.test.ts` | ✅ DONE (Wave 1): active morphs now throw when `morphTargetNormals.length !== morphTargets.length`, and malformed normal-delta entry lengths remain throw-on-read. Focused test pins both cases. |
| ~~Core contract additions from Wave 3~~ ✅ DONE | `packages/core/src/scene/material.ts`; `packages/core/src/scene/primitives.ts`; `packages/gltf-adapter/src/materials.ts`; `packages/gltf-adapter/src/gltfToScene.ts`; `featureReport.ts` | `thicknessMap` is a first-class `MaterialSpec` field and is now included in the shared `MaterialMapFields` slice; `KHR_materials_volume.thicknessTexture` imports to it; `doubleSided` is preserved in `MaterialSpec.extensions.doubleSided` and compatibility reports the renderer limitation as `approximate`; `morphTargetTangents` is now a first-class `SkinnedMeshPrimitive` field with glTF TANGENT-delta import and approximate compatibility diagnostics. |

### Historical commit sequence (mostly executed)

1. ✅ Land glTF API + engine bridge + controller + unlit all backends
2. ✅ Texture decode helper + pt-webgl2/pt-webgpu upload integration
3. ✅ P0 walkaround + pt-webgpu correctness (W-HYB, PTWG, H25-H29)
4. ✅ WEBGL2-01 + H49 residuals (GLTF-01 and CORE-01 are closed)
5. ✅ PTWG-MAT-01 integrator audit + material descriptor expansion
6. ✅ Walkaround texture atlas + UV/tangent buffers
7. ✅ Walkaround alpha + shadow GI parity
8. ✅ `createEngine` + `pickBackend` glTF-aware + `examples/gltf-viewer`
9. ~~Additional glTF material sweep beyond the closed behavioral gate fixtures~~
   ✅ narrowed by `pt/gltf-material-sweep`; default lavapipe golden and dzn
   full-tier assertion now cover API boot/readback for the material-heavy
   fixture. Remaining work is broader material-furnace/reference fidelity
   promotion, not API boot/readback plumbing.
10. Ledger/README/fidelity matrix reconciliation

### Active performance track

These are real SOTA/performance gaps, but they do not block the contract claim
that one predictable API can ingest and route arbitrary glTF assets. Promote
them to hard 100% blockers only if the Road definition is widened from
contract-complete to contract-complete plus SOTA throughput/convergence.

1. Low-discrepancy sampling (`LD-SAMPLING-01`): shared Sobol table generation
   plus pt-webgl2 `sampling:'sobol'` opt-in and pt-webgpu
   `sampling:'sobol'` opt-in are code-closed. pt-webgl2 is covered by the GLSL
   `sobol-on` shader gate; pt-webgpu composes a binding-free first-order
   pixel-scrambled Sobol RNG through the full/lite megakernels plus
   SPPM/ReSTIR-PT/BDPT auxiliary pipelines and surfaces an experimental
   capability/warning. Remaining work is Owen/blue-noise scrambling,
   per-dimension assignment audit, shader/adapter gate breadth for the Sobol
   WebGPU variants, and equal-time RMSE convergence proof.
2. Compressed wide BVH traversal (`WBVH-01`): the shared substrate is now
   partially landed: `shared-bvh` exports CWBVH-style 8-wide packing, quantized
   child bounds, deterministic packed metadata, conservative dequantized bounds
   checks, explicit u16→u32 WGSL upload packing, reordered triangle-payload
   overlay helpers, CPU first-hit/any-hit traversal oracles compared against
   brute force, and WGSL closest/any-hit traversal helpers
   covered by the WebGPU shader/pipeline gate. A standalone WebGPU parity oracle
   now compares the WGSL CWBVH traversal against the CPU oracle for closest-hit,
   any-hit, and glass-skip cases. Remaining work is backend opt-in capability
   flags, binary-BVH fallback policy, renderer binary-vs-CWBVH parity tests, and
   real equal-scene throughput/memory proof.

### Execution dependency

```
P0 land commit
  → P0 correctness (W-HYB, PTWG, PPG, NRC, neural binds)
    → PT material parity (2B, 2C) + WEBGL2-01
      → glTF API harden (1A-1E) + engine/gltf wrapper (1E)
        → pickBackend + compatibility enforcement (4A, 4B)
          → Walkaround alpha (3C) → Walkaround atlas (3D) → Walkaround lobes (3E)
            → Material furnace (5A) + gates (Phase 0.2, Phase 5)
              → 100% signoff
```

Walkaround atlas (3D) and PT material parity (2B) can run in parallel after P0.
LD-SAMPLING-01 can also run in parallel as a performance-quality track. WBVH-01
should wait until the current BVH/material mutation contracts are stable enough
that the wide layout can be proven against the binary-BVH oracle without moving
targets.

### Summary

- **Condensed to 5 phases:** land gates → glTF → PT → walkaround → orchestration → proof.
- **Specificity:** file-level plug-in points, decoder contracts, bind-group footguns, integrator audit matrix, texture atlas architecture.
- **Gap fill vs 85%:** the listed implementation items are now code-closed unless otherwise marked in their detailed rows: texture decode bridge, EXT_mesh_gpu_instancing policy/import, animation×temporal GI reset paths, lite-tier rejection for fidelity, PTWG-MAT specialty paths, walkaround alpha/blending, examples/gltf-viewer, `pickBackend`/`gltfAsset` passthrough, double-sided/vertex-color, tangent generation at import, and documentation sync. 2026-06-14 proof addendum: the progressive glTF engine helper has regression coverage for `textureDecodeReport` + warning passthrough, and the texture sweep covers enabled `MSFT_texture_dds` alternate-source selection through `loadGltfAsset()`. 2026-06-19 selected-scene follow-up: `analyzeGltfAsset(..., { sceneIndex })`, `gltfToScene`, compression hook detection, `loadGltfAsset()`, and `loadGltfForEngine()` now scope compatibility, camera/skin/double-sided diagnostics, Draco/meshopt hook requirements, texture decode reports, and inactive material-variant texture reporting to the selected scene; unknown required extensions remain asset-level blockers. Tests: `gltfAdapter.test.ts`, `gltfAssetApi.test.ts`, `gltfCompression.test.ts`, `gltfKhronosSweep.test.ts`, `gltfProgressiveSubpath.test.ts`, and `gltfStrictPtWebgpuTier.test.ts`. Remaining work is broader material-furnace/reference fidelity promotion and validation breadth, not missing API plumbing.
- **Performance work preserved:** low-discrepancy sampling and compressed wide
  BVH traversal are tracked as active SOTA/performance work, but not as
  arbitrary-glTF API contract blockers. Shipped denoiser weights remain
  post-100/provisioning work.

Walkaround **100%** and arbitrary glTF **100%** are not the same: arbitrary glTF routes rich assets to PT backends via the planner; walkaround 100% still means permanent `unsupported` for spectral/displacement with explicit rejection, not silent gray materials.

---

## Forward-looking — the post-100% SOTA wave (ADDED 2026-06-12)

> Phases 0–6 above deliver **contract-complete**. This section is the separate
> axis: convergence/throughput engineering where vitrum is below current SOTA
> practice even after the campaign closes. Tracked here per roadmap §0.5.
> **2026-06-12 scope update:** F1 low-discrepancy sampling and F2 compressed
> wide BVH traversal are tracked as an active performance track above, but they
> remain separate from the arbitrary-glTF API contract. F3 shipped denoiser
> weights stays post-100/provisioning work for now.
>
> Context: post-campaign vitrum is already at-or-beyond published in-browser
> SOTA on *breadth* (no public browser engine ships spectral + BDPT + ReSTIR-PT
> + MNEE + progressive SPPM + inverse rendering + a gated realtime GI track).
> The items below are where the *engineering* axis lags the field.

### F1 — Low-discrepancy sampling (biggest convergence win per effort)

PCG remains the default for both converged backends. The old pt-webgl2
Sobol-dummy complaint is no longer true: `shared-samplers` now exports a
256×256 RGBA Sobol table generator, `createPTEngine_WebGL2({ sampling:'sobol' })`
compiles `RANDOM_TYPE=1`, uploads a real RGBA32F Sobol texture, samples texel
centres, and the GLSL gate compiles a production `sobol-on` variant. Stratified
sampling is still intentionally unexposed because its textures remain
dummy-bound. pt-webgpu now also exposes `sampling:'sobol'`: the engine composes
a binding-free WGSL Sobol RNG module with the existing `pcgInit`/`rand_f32`
symbol contract so the megakernel, lite kernel, SPPM photon pass, ReSTIR-PT
reuse passes, and BDPT light-subpath pass all switch coherently. It is explicitly
tagged `pt-webgpu-sobol-sampling` in `capabilities.experimentalFeatures` and
warns that this is first-order pixel-scrambled Sobol, not the final
Owen/blue-noise/per-dimension-audited sampler.

**Remaining work:** Owen/blue-noise scrambling, per-dimension assignment audit
(bounce/lobe/light dims), real-adapter shader/behavioral gates for the Sobol
WebGPU variants, and equal-time RMSE A/B on the reference scenes
(self-validating error curves, not eyeballs).

### F2 — Compressed wide BVH traversal (biggest throughput win)

Binary SAH + stack traversal is solid but compute-shader SOTA is 8-wide
compressed BVH (CWBVH-style): ~2× traversal throughput, smaller memory
footprint. Light tree is median-split, not full adaptive Estévez-Kulla (already
documented in `lightTree.ts:33-35`).
**Landed implementation slices:** `shared-bvh` now has a CWBVH-style CPU packer
and oracle: binary SAH nodes collapse into 8-wide slots with parent-relative
u16 child bounds, explicit child kind/offset/count metadata, deterministic
outputs, empty-scene handling, conservative dequantized bounds tests,
first-hit/any-hit CPU traversal checked against brute-force triangle
intersection, reordered triangle-payload overlay for stride-4 `.w` data, and
CPU/WGSL parity for the glass-skip transmission-nibble filter when payloads are
supplied. The shared WGSL side is also started: child bounds have an explicit u16→u32 upload
packer, and `CWBVH_INTERSECT_WGSL` exposes closest-hit and any-hit traversal
helpers over the packed wide-node buffers. Shader-gate compiles a concrete
CWBVH harness pipeline through the same Naga ptr-parameter compatibility layer
used by the existing shared BVH traversal. `tools/behavioral-gate/cwbvh-parity-oracle.mjs`
also runs the CWBVH WGSL traversal on WebGPU and compares closest-hit, any-hit,
and glass-skip results against the CPU oracle; the committed
`cwbvh-parity-status.json` pins that proof. 2026-06-19 renderer-routing
follow-up: `pt-webgpu` now builds a renderer-shaped CWBVH prototype forest
beside the binary BVH during scene packing: one CWBVH tree per concatenated BLAS
subtree, packed child-bound/metadata/count buffers, TLAS BLAS-root remapping
from binary node roots to wide-node roots, GPU upload/destroy/memory accounting,
and CWBVH mirror refresh on geometry/TLAS mutation paths. Tests pin multi-BLAS
root remapping and CWBVH mirror writes for BLAS/TLAS updates. Same-day
root-routing follow-up: `CWBVH_INTERSECT_WGSL` now exposes
`cwbvhIntersectFirstHitFromRoot` / `cwbvhIntersectAnyFromRoot` while keeping the
old root-zero wrappers, so pt-webgpu's uploaded multi-BLAS CWBVH forest can be
addressed by the remapped TLAS BLAS roots instead of being limited to node 0.
Same-day renderer opt-in follow-up: `PTEngineWebGPUOptions.bvhTraversal:
'cwbvh-closest-experimental'` now composes a full-tier shader variant that binds
the five uploaded CWBVH buffers, routes mesh closest-hit and shadow/visibility
any-hit traversal through the remapped wide-node forest, advertises
`pt-webgpu-cwbvh-closest-traversal`, emits a structured experimental warning,
and keeps the default full-tier shader and 34-buffer device floor unchanged. The
CWBVH any-hit wrapper preserves `triShadowCastDisabled`, so `castShadow:false`
geometry remains camera-visible but non-occluding.

**Remaining work:** renderer binary-vs-CWBVH traversal parity tests; and
equal-scene throughput/memory A/B before any renderer default promotion. Becomes
decisive if/when a WebGPU ray-tracing extension ships (whole-field handicap
today: no RT cores in the browser for anyone).

### F3 — Shipped denoiser weights (out-of-the-box UX)

OIDN arrives via host-supplied ONNX with no weights shipped (A10 production
neural weights = declared non-goal of the campaign). SOTA UX is denoised by
default. **Work:** license-vetted OIDN weight distribution (or train the
in-repo UNet to production quality), wasm/webgpu execution path that needs no
host wiring, `denoiser:'auto'` default that engages when weights resolve.

### F4 — Wavefront path tracing (largest rearchitecture — only if profiling demands)

Megakernel with a hard 8-bounce structural cap (`kernel.wgsl.ts` bounceLimit).
Wavefront scheduling (per-bounce queue compaction) is how current GPU PTs kill
warp divergence at depth. **Work:** queue/compaction infrastructure, kernel
split (generate/extend/shade/connect), persistent state buffers. Big; gate the
decision on divergence profiling, not fashion. The 8-bounce cap lift falls out.

### F5 — Heterogeneous volumes

Homogeneous media + SSS random walk only; pt-webgl2's fog-volume GLSL is dead
code (uniforms never uploaded). SOTA is null-collision delta/ratio tracking over
grids (NanoVDB-class). **Work:** core contract for volume primitives first
(extension point exists: `AnalyticShape`/`Material.extensions`), then
delta-tracking integrator on pt-webgpu. Only if the product wants smoke/clouds —
stained-glass design center may never need it.

### F-BRIDGE — Experimental no-hardware-RT bridge (ADDED 2026-06-12)

> Strategy: compute shaders can't out-muscle RT cores at traversal, so don't
> compete — **trace fewer rays (reuse/upscale), cheaper rays (proxies/LOD), or
> none (caches)**. Stacked estimate ~25–65× effective vs the 10–50× hardware-RT
> promise (amortization-shaped: fast-moving scenes benefit less). All items pass
> the feasibility bar: public source, web-portable, not RTX-locked.
>
> | Lever | What | Gain | Feasibility |
> |-------|------|------|-------------|
> | Subgroups + f16 | Wave intrinsics for traversal compaction; f16 BVH/material bandwidth | 1.3–2× | Shipping in Chrome today; plumbing only |
> | Blue-noise error diffusion | Heitz–Belcour screen-space scrambling — same error, far cleaner look | perceptual | Paper + ref code public; tiny |
> | FSR2-class temporal upscaling | Render 50–60% res, reconstruct; motion vectors + depth already produced | 3–4× | FSR2 MIT HLSL → WGSL port; no browser PT has done it |
> | SHaRC radiance cache | World-space hash-grid cache, early path termination; non-neural cousin of NRC, can share its termination seam | ~2× path length | NVIDIA open source, plain compute |
> | SDF/proxy secondary rays | Software-Lumen trick: diffuse GI traces SDF/voxel proxy, only primary/specular touch triangles | 5–10× on secondary | Well documented (UE source readable); walkaround proves the dual-representation pattern |
> | Stochastic bounce LOD | Bounces ≥2 trace a decimated BVH (geometry version of the dead `materialLodDepth` idea) | scene-dep | Simple, low risk |
> | Shadow-map NEE assist | Rasterized shadow map answers dominant-light shadow rays; trace only the penumbra band | large (shadow rays dominate) | Walkaround raster primary shows the seam exists |
> | Server-side RT-baked GI | Converge GI state on real RT hardware off-device, ship via GI-state v4 export; browser refines | n/a (offload) | Works today; productizes app-idea #3 |
> | Persistent threads + ray sorting | Morton-binned wavefront (Aila/Laine lineage) — F4 done aggressively | 1.5–2× | Classic public literature; higher effort |
> | Subgroup matrix (experimental) | WMMA-class MLP inference for NRC/neural denoiser | NRC-specific | Chrome experimental flag; wait for stabilization |
>
> Unlock target: T2-4 (multiplayer GI editing), T2-5 (sensor twins), and
> 30–60fps path tracing of moderate scenes **without** the WebGPU RT extension.

### F6 — Unbiased realtime GI default (GRIS default-on)

Default ReSTIR-GI reuse is deliberately biased (clamped-Jacobian; A8 decision
keeps exact GRIS off-default). The unbiased-reuse literature is the published
SOTA. **Work:** GPU-validate the existing GRIS path at production settings
(perf + flicker), then flip the default; the oracle infrastructure from this
campaign (`oracle.restirDiEstimator`) extends to pin the GI estimator the same
way. **2026-06-13: the validation half of this is no longer "post-100%" — it
was pulled forward into the now-ACTIVE A8 GPU A/B task** (converged-unbiasedness
of GRIS-on + measured error of the biased default). The default-flip itself
stays gated on those numbers + the perf budget; F6 is the flip, A8 is the
evidence that justifies it.

### BUG RESOLVED (2026-06-13) — pt-webgl2 accumulation shader SSS helper compile failure

Found while driving `createPTEngine_WebGL2` from a consumer app (the honeycomb
stained-glass bench) on real WebGL2 (Chrome, NVIDIA Lovelace). The accumulation
fragment shader failed GLSL compilation at runtime:

```
ERROR: 0:3946: 'sampleExponential' : no matching overloaded function found
ERROR: 0:3950: 'sampleHG_glsl'     : no matching overloaded function found
ERROR: 0:3950: '='                 : dimension mismatch
ERROR: 0:3950: '='                 : cannot convert from 'const mediump float' to 'highp 3-component vector of float'
ERROR: 0:3953: 'hg_phase'          : no matching overloaded function found
```

- **Resolution:** closed by the D10 shader-gate wave. `bsdf_functions.glsl.js`
  now defines `sampleExponentialDistance`, `sampleHG_glsl`, and `hg_phase`
  before the SSS call site and no longer calls the undefined
  `sampleExponential(...)` symbol. `composeTraceGlsl.test.ts` pins the helper
  ordering, and `@vitrum/shader-gate` is now a workspace package whose ordinary
  root `npm test` path runs the CPU GLSL production-variant compile gate plus
  its injected-error self-test. Current proof: root `npm test` compiles all 6
  pt-webgl2 GLSL feature combinations and the self-test detects the injected
  broken shader.
