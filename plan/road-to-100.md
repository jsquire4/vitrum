# Road to 100 — code completion

**Source audit date:** 2026-07-28

**Audit freeze:** base HEAD `a666411be90f432caa8ef5f0a8d1e3a40f16b3a1`
plus the current uncommitted remediation working tree; SHA-256
`55a0faed72130bdf54b59745700db31a2b0275547677fa98287d09005bd2c0e3`
for the bytewise path-sorted `packages/*/src` path-and-content manifest
(repository-relative POSIX path, NUL, raw content, NUL; 1,247 files)

**Authority:** current production source under `packages/*/src`

**Scope:** implementation quality for downstream library users

Public distribution, release governance, and cross-host evidence are outside this
plan. Historical prose, generated status snapshots, comments, and future product
ideas are not implementation authority.

## Bottom line

At the current production source, every executable semantic defect found by the
2026-07-27 deep audit, the 2026-07-28 external completeness audit, and their
independent post-fix sweeps has an implementation and focused regression in the
current working tree. No additional production code is presently known to be
required for the declared profiles. The same-tree convergence run and
source-manifest refresh are complete. This freeze describes uncommitted working
tree bytes, not a landed commit. A future failing gate reopens this plan only
when it exposes a current production-code defect under the rule below.

Production code has no provisional feature-maturity channel. Optional algorithms
use stable typed options and capability fields. A mode is implemented under its
documented bounded model or rejected before partial engine state is published.

## What “professional” means here

Vitrum is code-complete for a declared profile when all of the following hold:

1. Valid public inputs execute the documented algorithm. Fatal policy violations
   return a typed error before a successful result is published. Documented
   recoverable transformations retain the source or previous state and emit a
   structured diagnostic before publishing adapter-created output.
2. Resource replacement, async construction, and live mutation publish one
   coherent generation or preserve the previous usable generation.
3. Sampling, evaluation, PDFs, transport mode, medium state, and MIS ownership
   agree for every enabled estimator combination.
4. Scene, glTF, texture, animation, skinning, morph, compression, and model-backed
   inputs are validated before backend state is committed.
5. Capability metadata describes the implementation that actually runs.
6. Host-owned devices remain host-owned; engine-owned CPU/GPU/import resources
   have deterministic, idempotent teardown.
7. Typecheck, build, tests, lint, shader compilation, and source-derived proof
   gates converge on the same frozen tree.

An explicitly bounded algorithm is not unfinished merely because a more expensive
algorithm could be added later. Generated point/line geometry, finite realtime
cascade budgets, strict asset rejection, explicit finite-difference inverse
methods, and host-provisioned model weights are valid product boundaries when
implemented and reported end to end.

## Current code-gap queue

There are no open implementation rows.

If the final convergence run finds a source defect, add one bounded row with its
reachable failure mode and focused regression. Do not create proof-only, host-only,
distribution, governance, cross-host-evidence, or future-feature rows in this
queue.

## Closed implementation programs

### 2026-07-28 external completeness-audit remediation

- Walkaround material transport now uses a disjoint physical F0 encoding,
  retains authored zero roughness through the shared finite-alpha numerical
  floor, preserves zero-default clearcoat/sheen semantics, and applies ordered
  sheen and clearcoat attenuation in both direct and indirect evaluators.
  GTAO projects the normal in the correct slice plane, and environment
  replacement is part of transactional scene publication.
- Native WebGL2 keeps light maps receiver-local and applies transmissive
  attenuation once per represented boundary. Material maps retain native
  source rectangles in split sRGB-aware RGBA8 parameter/color and linear
  RGBA16F radiance atlases; their complete CPU mip chains and spare-capacity
  GPU storage share one exact 512 MiB preflight before allocation or
  transactional publication. Compiler flags, shader chunks, and material
  structs that cannot affect a composed permutation are removed.
- WebGPU clearcoat/sheen sample, evaluation, PDF, BDPT, SPPM, MNEE, and ReSTIR
  paths share one layered response. Thin-film TMM reflection, transmission, and
  absorption participate in the same finite microfacet families; malformed
  certified data fails dark. Presentation uses an explicit unfilterable-float
  layout and re-presents retained converged/paused output after presentation
  controls change.
- WebGPU BDPT includes the native `s=n-1,t=1` light-subpath-to-camera strategy
  with matched perspective-camera densities, arbitrary-pixel atomic
  accumulation, and one complete per-frame variance sample. Its bounded
  CPU/WGSL strategy mask agrees over the executable domain:
  `npm run shader-gate -- --bdpt-mask-parity-only` passes all 32,400 cases.
  The BDPT-off shader and resource surface remain byte-identical.
- Every WebGPU alpha walker is bounded by scene surface support rather than a
  fixed layer count and fails closed on impossible extra traversal. Production
  path replay is the exact one-bounce opaque-triangle unlit-emissive domain;
  paused frame inputs update the bounce regime before inverse preflight.
- Generalized reconnection-shift reuse is the sole live walkaround GI ABI.
  DI/GI use represented attempt counts, log-domain weights, distinct
  ping-pong generations, exact temporal correspondence, and scene-mutation
  history invalidation. Persisted DDGI, DI, GI, PPG, and NRC state publishes as
  one reversible cohort; imported GI adopts its nonzero history epoch and runs
  one full-rate initialization frame without erasing restored reservoirs.
- DDGI refit/cache invalidation, finite-difference boundary probes, NRC
  relative-L2 training loss, and BMFR demodulated-history clamping now agree
  with their live estimator domains. Checkerboard execution waits for its
  history producers, and progressive/lifecycle errors preserve the previous
  usable generation or fail before publication.
- glTF import now preserves decoder MIME identity across cold, metadata-aware,
  legacy, authored-MIME, and offline cache paths; validates reachable lights,
  animation degradation, analytic UVs, atlas budgets, and derived resources;
  and accepts tiny but Float32-invertible skinned transforms while rejecting
  singular or numerically non-reciprocal matrices.
- Built-in DDS/DX10 and KTX2/Basis texture decoding now validates complete
  headers, layouts, mip payloads, transfer/alpha metadata, resource ceilings,
  and the pinned transcoder ABI before allocation or publication. DDS coverage
  includes pitched rows, block formats, BC5 normal reconstruction, and strict
  single-image policy; KTX2 coverage includes ETC1S, UASTC, BasisLZ SGD, DFD,
  mip ordering, overlap/padding rejection, and an authoritative Khronos ETC1S
  fixture with its complete decoded RGBA output.
- Runtime animation updates now solve native skin/morph state once per shared
  deformation, preserve instance transforms, reset to bind state correctly,
  and route mixed transform-plus-pose or merged-instanced geometry changes
  through coherent rebuilds. CPU picking solves the same skinned geometry as
  rendering, and public skinned rest-stream mutation routes through the
  canonical solver.
- Public GPU detection snapshots are frozen and the dead window-global
  detection channel is removed. The public à-trous variance layout matches its
  four live bindings, while Deno and TypeScript share explicit owned-buffer and
  pipeline-constant types at the PPG compiler boundary.
- The sole generalized-reuse GI ABI no longer exposes a retired runtime config
  bit through the learned-systems proof surface: omission and legacy `true`
  spellings preserve the fixed layout, legacy `false` spellings fail closed,
  and PPG composes with the canonical reuse path. WebGPU BDPT camera splats use
  storage-array word indices for their bounded CAS loop, compiling portably
  across every shipped BDPT module variant.
- The audit's inert functions, aliases, compatibility branches, stale shader
  chunks, unused resource fields, and contradictory source headers were
  removed or reconciled. Road/source and renderer-fidelity guards enforce that
  production source, active plans, and package READMEs have no separate
  provisional maturity channel.

### 2026-07-27 renderer-integrity and persisted-state closure

- WebGL2 finite-area/environment direct lighting now preserves delta ownership
  across NEE and forward-hit strategies. WebGPU rough-glass evaluation, finite
  PDF, and sampling share the same specular/iridescence Fresnel model, and every
  finite BSDF sample reports the full mixture marginal used by MIS.
- The complete WebGPU integrator core was source-read after the initial audit.
  MNEE carries mapped, layered, spectral, and Fresnel material state; medium NEE
  has one estimator owner; refraction updates a bounded nested-medium stack and
  fails closed on invalid transitions; ReSTIR stores one canonical radiance
  domain; adjoint readback resources remain owned through completion.
- Frame inputs now derive one canonical camera origin from the affine view
  matrix and reject meaningful legacy-position disagreement. glTF camera,
  alpha, weight, quantized-instance, texture-decode, and large-world metadata
  boundaries validate the values actually consumed by renderer backends.
- DDGI relocation/classification is a scheduled CPU/WGSL-parity stage. Bounded
  normalized offsets and exact active state occupy the collision-free `(4,4)`
  irradiance-cell texel, active-prefix dispatch cannot read stale capacity, and
  inactive probes remain scheduled for recovery. v3-v5 imports synthesize the
  historical zero-offset active state; v6 persists explicit Float32 state.
- DDGI visibility uses a dedicated clamp-to-edge linear sampler through ordinary
  GI, shade, NRC, and transparent paths while SH/state reads remain exact.
  Irradiance's obsolete border scratch path is removed. Shared binary16 packing
  now implements round-to-nearest-even across normals, subnormals, signed zero,
  infinities, and NaNs.
- GI import prepares a complete ReSTIR replacement before publishing DDGI,
  commits required state as one cohort, and leaves PPG best-effort state
  transactional. The versioned decoder rejects incompatible section flags,
  non-finite reservoir lanes, invalid sample kinds, non-zero padding/reserved
  words, malformed probe state, and large-world PPG topology mismatches before
  live state mutation.

### 2026-07-27 post-audit semantic closure

- Zero-light engine boot, mapped analytic validation, pt-webgpu lite mutation
  policy, displacement subdivision invalidation, and presentation-source facade
  forwarding now preserve the same public contract at construction and mutation
  boundaries.
- Every reachable pt-webgl2 material/fog tier composes its required helpers; the
  GLSL gate covers 30 combinations. Its BDPT NEE candidate receives scene bounds,
  while pt-webgpu executes and discloses its exact bounded explicit-connection
  strategy set, including native light-subpath camera splats.
- DDGI mesh-area radiometry applies the receiver BRDF once, spot lights retain
  cone semantics, and glass face-layer selection uses the hit triangle.
  Walkaround glass includes reflection and GRIS-reused transmission, and applies
  face-layer attenuation once. GRIS reuse also preserves the valid reservoir
  direction, visibility, and weight for non-glass glossy/metal indirect
  specular; the post-glass reservoir remains transmission-owned.
- Salted, frame-varying pixel hashes, the full isotropic multiscatter roughness
  derivative, four-dimensional Sobol plus PCG continuation, quadratic
  hero-wavelength CDF inversion, and the complete NRC validation binding layout
  keep CPU/WGSL math and advertised sampling behavior aligned.
- glTF preflight accepts the declared 2.x/min-version range, quantized positions,
  underscore-prefixed application attributes, and the supported morph routes;
  exposes an explicit KTX/Basis transcoder seam; and rejects malformed reserved
  semantic aliases before fetch or decoder work.
- Sparse UV, color, joint, and weight sets retain canonical safe-integer own keys
  through validation, cloning, skin/morph solving, import/remap, displacement,
  packing, and renderer mutation, including keys outside the 32-bit array-index
  range. Canonical `JOINTS_n`/`WEIGHTS_n` pairing is checked independently of
  storage density.
- SVGF variance/fallback/history state and its four-pipeline/six-buffer rebuild
  publish transactionally. DDGI atlas/cache replacement retires only superseded
  generations, and CWBVH fixed-order traversal has a canonical correctness
  fallback.
- Radiance Cascades applies the authored environment map intensity and Y
  rotation, distinguishes a real zero-intensity map from the placeholder, and
  uses scalar sky radiance when no directional map exists.
- ReGIR config scalars, flattened cell/survivor/float indices, and the
  light-tree float offset remain in their WGSL `u32` domains. Its 64-wide build
  and X dispatch are preflighted against device workgroup limits, including
  `maxComputeWorkgroupsPerDimension`; combined-buffer byte arithmetic remains
  exact, safe, aligned, and preflighted against WebGPU allocation and
  storage-binding limits before resource publication.
- Eligible GPU skinning dispatches into the live BVH. The retained CPU solve is
  the explicit coherence path for CPU BVH/TLAS/shadow mirrors and unsupported
  skin shapes, rather than an omitted rendering step.

### Public core, engine, and lifecycle boundary

- `@vitrum/core` owns strict `validateScene` coverage for finite values, stream
  sizes, indices, transforms, materials, analytic parameters, emitters, skins,
  morphs, UV/color sets, and patch shape.
- Engine creation, progressive creation, backend construction, glTF conversion,
  scene-controller attachment, and mutation paths validate before publication.
- Skinning, morphing, optional bind data, incremental patches, device-loss
  recovery, progressive handoff, and React/vanilla attachment preserve coherent
  scene and resource generations or fail closed.
- Engine/resource disposal is idempotent, while host-owned device and cadence
  lifetimes remain outside engine ownership.

### Bounded glTF import transaction

- Every public import path uses one monotonic ledger across direct, preloaded,
  cached, data-URI, network, and GLB bytes; compression work; decoded/generated
  geometry; image acquisition; and texture normalization. Finite defaults,
  explicit byte/pixel opt-outs, safe-integer validation, identity deduplication,
  and one FIFO concurrency ceiling are enforced without resetting earlier
  charges.
- Network bodies preflight declared lengths, charge immutable chunks
  incrementally, cap read operations, cancel on rejection, exclude rejected
  bytes from cache, and settle every started peer operation before surfacing the
  first failure.
- GLB input is brand-checked through captured intrinsics. Declared length, chunk
  bounds/alignment/order/uniqueness, trailing headers, BIN-copy budget, and fatal
  UTF-8 JSON decoding are validated before publication.
- Accessor and geometry paths validate declared buffer ranges, component/type/
  count/stride arithmetic, matrix padding, sparse ordering and ranges, raw
  integer joint streams, primitive indices, finite positions, and generated
  normals, tangents, topology, transforms, camera matrices, and fallback meshes
  before allocation.
- Draco and meshopt declarations are validated as own-property schema objects
  with exact source paths, permitted enumerable fields, nested glTF-property
  structure, distinct EXT/KHR rules, declared source ranges, fallback-buffer
  ownership, typed decoder outputs, point/index consistency, and bounded working
  sets.
- Public compression obeys the import policy without restoring the standalone
  512 MiB guard when a public ceiling is disabled. Known output and retained-copy
  costs are preflighted. Built-in JS outputs charge the monotonic import ledger
  immediately before allocation, including failed attempts; successful outer
  validation skips only the duplicate hook-output charge. Snapshots, re-encoding,
  generated buffers, and retained copies remain independently charged.
- Embedded `bufferView` images are non-owning views and reuse the parent buffer
  resource identity rather than counting the same bytes twice.
- Raw-image acquisition captures one owned immutable byte snapshot. PNG/JPEG/
  WebP dimensions are preflighted, codec choice and host input use that same
  snapshot, unique decode work is bounded/deduplicated, failed cache entries are
  evicted, and hostile/shared/malformed handle shapes cannot be advertised as
  backend-ready.
- The Node JPEG bridge passes explicit non-stricter decoder limits. Exact adapter
  pixel preflight remains authoritative; zero and high finite limits cannot
  silently restore `jpeg-js`’s smaller defaults.
- Encoded-resource, acquisition-stage decoded-image, geometry, and required
  compression failures reject the transaction. Per-texture normalization/output
  failures instead preserve the original texture reference and emit a structured
  source-pathed diagnostic before publishing any invalid derived surface.
- Later texture-policy configuration is retrospectively validated and
  transactionally committed. Undefined nested values cannot relax inherited
  ceilings, defined flat aliases retain precedence, and malformed hook results
  reject.
- Decoder-created closable handles are transaction-owned and identity-deduplicated.
  Rejection closes each identity at most once across concurrent/local/outer
  failure paths; successful normalization closes superseded handles immediately;
  successful results retain reachable handles until the public idempotent
  `releaseGltfResources(result)` call.
- The release API accepts low-level, asset, adapter-engine, and engine-subpath
  results. Adapter-created/progressive engines dispose before release on failure;
  caller-owned engines are not disposed. `VitrumCanvas` retains ownership through
  pending attach, disposes before release on cancellation/failure/unmount, and
  still releases when lifecycle disposal throws.

### Asset and model interoperability

- Primitive contracts and all renderer families carry arbitrary authored UV sets
  through backend-local layouts; mapped materials are not limited to UV0/UV1.
- CPU-readable and nominal GPU texture sources have validated color-space,
  normalization, ownership, upload, and compatibility-report paths.
- glTF animation-pointer, scene-scoped animation, variants, sparse/padded
  matrices, skins, morphs, instancing fallback, texture-source selection,
  compression, and strict compatibility paths either preserve the declared
  result or report the exact bounded fallback/rejection.
- Generated point/line meshes and expanded skinned/morphed instances remain
  explicit compatibility grades rather than hidden partial implementations.

### Converged path tracers

- WebGPU advanced transport combinations are validated and composed under stable
  contracts: spectral transport, bounded general BDPT, ReSTIR-PT, SPPM, manifold
  caustics, layered/volume transport, and adjoint replay have explicit ownership,
  capability, and rejection rules.
- BDPT uses one bounded multi-vertex strategy family with delta handling, medium
  transport, native WebGPU light-subpath camera splats, and independent
  density/ownership checks.
- Exact-positive activation is consistent for medium density, transmission,
  environment/sun selection, directional/ReSTIR gates, and light-tree ownership.
- Inverse sessions report the selected analytic or finite-difference method
  without silently changing the render regime.
- WebGL2 exposes its bounded general-BDPT caustic path directly and rejects
  algorithm names it does not implement.
- Backend option validation rejects non-finite, negative, contradictory, tier-,
  or capability-incompatible requests before GPU publication.

### Realtime hybrid and Radiance Cascades

- Transparent blend participates in seeded secondary transport and alpha-aware
  visibility while primary visibility remains ordered OIT.
- Mapped emitters use represented sub-triangles and PDFs across light CDF, alias,
  tree, ReSTIR, DDGI, and RC consumers; light maps remain receiver-local
  irradiance.
- Rich material lobes, layers, spectral fields, homogeneous media, mapped
  transport, and bounded caustic controls are carried through hybrid material/GI
  payloads.
- RC sampling is runtime-sized and alias-driven; raw dispatch validates exact
  buffer ranges, offsets, layouts, and invalidation state.
- RC, DDGI, ReSTIR, PPG, NRC, skinning, scene storage, optional subsystem
  bindings, and frame publication use transactional generation/lifecycle rules.

### Learned systems and denoisers

- Neural U-Net, NRC, PPG, BMFR, SVGF, à-trous, and OIDN paths validate
  configuration, tensor/layout/model assets, device limits, and resize/
  reinitialization state before committing resources.
- Host-provisioned weights/models are explicit contract inputs; missing or
  malformed required assets fail closed or resolve through the documented stable
  fallback.
- Replacement and disposal paths release model, uniform, intermediate, and
  cached GPU resources without publishing stale async outputs.

### Stable capability surface

- Production source contains no provisional feature identifiers or maturity list.
- Optional algorithms are discoverable through stable typed options,
  active-feature identities, warnings, and capability fields.
- The renderer-fidelity matrix uses `supported`, `approximate`, and `unsupported`
  to describe implemented semantics, not a release queue.

## Final convergence gates

These rows record the final frozen-tree run. They are verification results, not
implementation rows.

| Gate | Frozen-tree result |
| --- | --- |
| `npm run proof-check` | PASS — source/Road, radiometric, ReSTIR-PT specialty, and renderer-fidelity proofs |
| `npm run typecheck` | PASS — all configured workspaces |
| `npm run build` | PASS — all seven configured example production builds |
| `npm test` | PASS — 84/84 root script tests and 8,103/8,103 workspace tests across 693 files (8,187 tests across 704 files combined; 3,057 walkaround-hybrid) |
| `npm run lint` | PASS — 0 errors and 0 warnings |
| WGSL compile gate (`--no-pipeline-gate`) | PASS — 78/78 modules and 29/29 portable shipped walkaround/RC roots; no native pipeline-graph claim |
| `npm run shader-gate -- --bdpt-mask-parity-only` | PASS — 32,400/32,400 bounded CPU/WGSL strategy-mask cases agree |
| GLSL compile gate | PASS — 30/30 shader combinations; injected-error self-test detected the broken control |
| `git diff --check` | PASS |

`npm run proof-check` remains source/code-derived: Road/source markers,
radiometric source checks, the ReSTIR-PT specialty oracle, and renderer-fidelity
proofs. Public distribution, release governance, cross-host captures, and
committed host-status snapshots are not gates in this plan.

## Deliberate future additions

The broader [roadmap](./roadmap.md) records optional expansion after this
boundary, including native line/point primitives, native instanced skinned/
morphed primitives, heterogeneous media, wavefront scheduling, temporal
upscaling, and standardized hardware ray-query acceleration. They are not
missing implementations in the currently declared profiles.

## Reopen rule

A new Road row requires all four:

1. a defect visible in current production source or reproducible behavior;
2. a violated declared contract;
3. a bounded implementation change;
4. a focused executable regression.

Documentation drift, unavailable hardware, a missing host capture, a possible
future feature, or a broader algorithm that the API does not claim is not enough.
