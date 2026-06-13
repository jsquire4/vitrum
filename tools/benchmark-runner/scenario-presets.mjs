/**
 * Deterministic metadata for gap-closure scenarios (host capture pages may read query params).
 * Mirrors plan/gap-closure-acceptance-matrix.md.
 */
export const GAP_CLOSURE_SCENARIOS = [
  {
    scenarioId: 'rfe03-layered-front-back',
    seed: 1337,
    resolution: '1280x720',
    bounces: 8,
    spp: 512,
    backend: 'pt-webgpu',
  },
  {
    scenarioId: 'rfe07-11-sss-mixed-panels',
    seed: 2027,
    resolution: '1280x720',
    bounces: 8,
    spp: 512,
    backend: 'pt-webgpu',
  },
  {
    scenarioId: 'rfe08-13-spectral-payload',
    seed: 4242,
    resolution: '1280x720',
    bounces: 10,
    spp: 1024,
    backend: 'pt-webgpu',
    extensions: { 'vitrum.ptWebgpu.spectralHeroWavelength': true },
  },
  {
    scenarioId: 'rfe14-thinfilm-angle-shift',
    seed: 9001,
    resolution: '1280x720',
    bounces: 10,
    spp: 1024,
    backend: 'pt-webgpu',
    extensions: { 'vitrum.ptWebgpu.spectralHeroWavelength': true },
  },
  {
    scenarioId: 'rfe09-bridge-global-cmf',
    seed: 31415,
    resolution: '1024x1024',
    bounces: 8,
    spp: 256,
    backend: 'pt-webgpu',
    extensions: { 'vitrum.ptWebgpu.spectralHeroWavelength': true },
  },
  {
    scenarioId: 'rfe05-caustic-strategy',
    seed: 27182,
    resolution: '1280x720',
    bounces: 10,
    spp: 1024,
    backend: 'pt-webgpu',
    causticVariants: ['none', 'manifold-nee', 'photon-map'],
  },
  {
    scenarioId: 'ptwgpu-parity-material-fields',
    seed: 777,
    resolution: '1280x720',
    bounces: 8,
    spp: 512,
    backend: 'pt-webgpu',
  },

  // -------------------------------------------------------------------------
  // Sweep-2026-05-11 verification scenarios (Phase A1)
  // Advisory-only until post-sweep baselines are stable (~2 capture cycles).
  // -------------------------------------------------------------------------

  // Verifies M5 Item 16 (frDielectric branch) — grazing-angle Fresnel on a
  // glass sphere forces the frDielectric code path. At 45° camera elevation
  // the highlight should narrow and brighten predictably with Fresnel theory.
  // Seed 5501 chosen to keep the highlight centred in frame at this angle.
  {
    scenarioId: 'm5-glass-fresnel-grazing',
    seed: 5501,
    resolution: '1280x720',
    bounces: 10,
    spp: 512,
    backend: 'pt-webgl2',     // also exercise pt-webgpu; run both via vitrumBackend query param
    cameraElevationDeg: 45,
  },

  // Verifies M5 Item 15 (sum-MIS over all lights) — two rect-area lights on
  // opposite Cornell walls. At convergence, irradiance on the floor should be
  // approximately 2× the single-light reference (linear in light count when
  // MIS is correct).
  // Seed 6121 produces good inter-light balance in the MC sequence.
  {
    scenarioId: 'm5-multi-light-cornell',
    seed: 6121,
    resolution: '1280x720',
    bounces: 8,
    spp: 512,
    rectAreaLightCount: 2,
  },

  // Verifies M5 Item 14 (Heitz 2018 VNDF sampling) — highlight tightness
  // should track GGX VNDF lobe shape across roughness values. Each variant
  // is captured separately; the runner iterates roughnessVariants and sets
  // VITRUM_ROUGHNESS for each capture.
  // Seed 8008 provides stable highlight framing across all four roughness
  // values; low roughness (0.1) produces a tight specular spike, 0.7 is
  // broad and diffuse-like.
  {
    scenarioId: 'm5-glossy-roughness-sweep',
    seed: 8008,
    resolution: '1280x720',
    bounces: 8,
    spp: 512,
    roughnessVariants: [0.1, 0.3, 0.5, 0.7],
  },

  // Verifies M7 corrected receiver math (Phase A, plan item m7-ddgi-grey-vs-white-cornell).
  // Grey wall (albedo=0.5) should receive ~half the indirect irradiance of the
  // white wall (albedo=1.0) because the receiver applies (albedo/π)·E.
  // Seed 1701 keeps probe layout stable across albedo variants.
  // frames=128 allows the DDGI temporal EMA to converge past transient bias.
  {
    scenarioId: 'm7-ddgi-grey-vs-white-cornell',
    seed: 1701,
    resolution: '1280x720',
    bounces: 8,
    spp: 256,
    frames: 128,
    backend: 'walkaround',
    wallAlbedoVariants: [0.5, 1.0],
  },

  // Verifies M7 Item 6 (Halton SO(3) rotation) + M7 Item 20 (Lambertian
  // cosine kernel). With both fixes, probe atlas values in a constant-irradiance
  // environment should converge to a uniform value; the rendered sphere should
  // appear uniformly lit. Pre-fix: pow(8) kernel + frozen rotation produced
  // visible directional banding.
  // Seed 2049 chosen for stable Halton frame-index spacing.
  // frames=200 gives the EMA enough temporal samples to suppress MC noise.
  {
    scenarioId: 'm7-ddgi-uniform-environment',
    seed: 2049,
    resolution: '1280x720',
    bounces: 8,
    spp: 64,
    frames: 200,
    backend: 'walkaround',
    environmentMode: 'uniform',
  },

  // Verifies M8 border-fill pass — smooth-normal surface seen at a glancing
  // angle exposes probe-atlas texel borders. Post-M8, the border-mirror fill
  // should eliminate the dark-ring seam artifact. Any per-cell grid darkening
  // visible in this scenario is a regression.
  // Seed 3333 orients a smooth-normal quad at ~10° to the camera view ray,
  // maximising probe-atlas edge traversal in the shading evaluation.
  {
    scenarioId: 'm8-ddgi-no-seam-darkening',
    seed: 3333,
    resolution: '1280x720',
    bounces: 8,
    spp: 256,
    frames: 64,
    backend: 'walkaround',
    glancingAngleDeg: 10,
  },

  // Verifies M9 Item 22 (per-bin Ω solid-angle normalisation) + M9 Item 21
  // (cascade merge integral). Mirrors m7-ddgi-uniform-environment but uses
  // the RC cascade pyramid as the GI source instead of DDGI probes.
  // Seed 2049 matches m7-ddgi-uniform-environment to enable direct comparison
  // of atlas uniformity between DDGI and RC backends.
  {
    scenarioId: 'm9-rc-uniform-environment',
    seed: 2049,
    resolution: '1280x720',
    bounces: 8,
    spp: 64,
    frames: 200,
    backend: 'walkaround',
    giMode: 'rc',
    environmentMode: 'uniform',
  },

  // Verifies M9 Item 23 (Jiménez 2016 GTAO slice integral) — a 90° interior
  // corner should show measurably darker contact shadows post-fix compared to
  // the pre-fix (h1+h2)/π HBAO approximation. Open-sky pixels outside the
  // corner should be unchanged (AO ≈ 1.0).
  // Seed 4499 keeps the corner geometry centred in the viewport.
  {
    scenarioId: 'm9-gtao-corner-shadows',
    seed: 4499,
    resolution: '1280x720',
    bounces: 8,
    spp: 256,
    frames: 64,
    backend: 'walkaround',
    sceneVariant: 'right-angle-corner',
  },

  // Verifies M9 Item 24 (atrous-variance edge-preservation) — a checkerboard
  // floor Cornell with alternating black/white tiles under indirect DDGI.
  // Material edges must remain crisp through the atrous-variance spatial filter
  // chain; colour bleeding across tile boundaries is a regression.
  // Seed 7777 chosen to avoid systematic MC bias along the tile boundary axis.
  {
    scenarioId: 'm9-albedo-edge-preservation',
    seed: 7777,
    resolution: '1280x720',
    bounces: 8,
    spp: 256,
    frames: 64,
    backend: 'walkaround',
    floorVariant: 'checkerboard',
  },

  // Verifies M4 Item 17 (transformNormal instance transform) — a sphere with
  // a non-uniform scale(2,1,1) instance transform. The specular highlight
  // should track the mathematically correct stretched-surface normal
  // (M^{-T} transform applied to geometry normal). A naive model-matrix
  // transform without the inverse-transpose produces a shearing artefact.
  // Seed 6060 keeps the specular lobe visible from the default camera.
  {
    scenarioId: 'm17-stretched-sphere-shading',
    seed: 6060,
    resolution: '1280x720',
    bounces: 8,
    spp: 512,
    instanceScale: [2, 1, 1],
  },

  // Verifies M4 Item 18 (Beer-Lambert clamp removal) — a glass slab of
  // thickness 100 world-units. Transmittance through the slab should satisfy
  // T ≈ exp(-σ·100). Pre-fix the implementation clamped the path length to
  // 32 wu, producing T ≈ exp(-σ·32) and an erroneously bright exit radiance
  // for thick dielectrics.
  // Seed 9191 keeps the camera perpendicular to the slab for a clean
  // transmittance measurement with no angular variation.
  {
    scenarioId: 'm18-thick-glass-attenuation',
    seed: 9191,
    resolution: '1280x720',
    bounces: 12,
    spp: 512,
    glassDimensions: { width: 20, height: 20, thickness: 100 },
  },
];

/**
 * PR-6 primary-release hybrid benchmarks (plan/primary-release-and-webgpu-pt-parity).
 * Host capture pages should read `vitrumScenario` + `vitrumBackend=walkaround`.
 */
/** WG-0.2 — pt-webgpu scenarios captured by the configured Playwright adapter when GPU capture is on. */
export const WG0_PT_WEBGPU_SCENARIOS = [
  'ptwgpu-parity-material-fields',
  'rfe03-layered-front-back',
  'rfe07-11-sss-mixed-panels',
  'rfe08-13-spectral-payload',
  'rfe14-thinfilm-angle-shift',
  'rfe09-bridge-global-cmf',
  'rfe05-caustic-strategy',
];

export const PR_HYBRID_BENCHMARK_SCENARIOS = [
  {
    scenarioId: 'PR-hybrid-200k-static',
    seed: 26001,
    resolution: '1280x720',
    backend: 'walkaround',
    targetTriangleCount: 200_000,
    frames: 120,
    metric: 'p95FrameMs',
  },
  {
    scenarioId: 'PR-hybrid-tlas-10-inst',
    seed: 26002,
    resolution: '1280x720',
    backend: 'walkaround',
    instanceCount: 10,
    bvhMode: 'tlas',
    frames: 120,
    metric: 'p95FrameMs',
  },
  {
    scenarioId: 'PR-hybrid-material-churn',
    seed: 26003,
    resolution: '1280x720',
    backend: 'walkaround',
    frames: 64,
    materialPatchIterations: 100,
    metric: 'zeroPipelineReinit',
  },
  {
    scenarioId: 'PR-hybrid-emitter-churn',
    seed: 26004,
    resolution: '1280x720',
    backend: 'walkaround',
    emitterCount: 3,
    frames: 64,
    emitterPatchIterations: 100,
    metric: 'emitterBufferRefreshMs',
  },
];
