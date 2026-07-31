# Road to 100 — code completion

**Source audit date:** 2026-07-30

**Final source manifest:** 1,674 files; SHA-256
`17cf324ec075e7a18bb02e1d79192380780a1fcf7954a462570b649dc1908388`;
bytewise path-sorted
path-and-content records (repository-relative POSIX path, NUL, raw content,
NUL) covering executable source, tests, shaders, examples, proof tooling,
runtime WASM, dependency locks, and configuration under `packages`, `examples`,
`scripts`, and `tools`, plus root package, dependency, TypeScript, and lint
configuration. Generated builds, dependency caches, captured evidence, training
data, and output-only directories are excluded. Reproduce with
`npm run road-to-100-source-manifest`.

**Authority:** current production source under `packages/*/src`

**Scope:** implementation quality for downstream library users

Public distribution, release governance, and cross-host evidence are outside this
plan. Historical prose, generated status snapshots, comments, and future product
ideas are not implementation authority.

## Bottom line

The current source review's implementation queue is closed on the final
remediation tree. The execution baseline was commit
`a596e6413af4c6fc2809ba4670ce1b81c332ab0f`.
The source manifest above is the final freeze for this pass. The previous
closed-program records remain historical evidence only; they do not override a
reachable failure in current source.

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

Each row closes only after a pre-fix failing proof, implementation, package and
workspace convergence, applicable numerical/render evidence, independent source
review, and coordinator verification of the live call path.

| ID | Current-source gap | Primary source | Status |
| --- | --- | --- | --- |
| C0 | Public primitive mutation accepts an inexact broad partial type | `packages/core/src/scene/patchScene.ts` | Closed |
| V1 | ReSTIR A/B producer and validator use incompatible result shapes | `tools/radiometric-ab` | Closed |
| V2 | Non-finite GPU values can disappear before byte-domain validation | `tools/behavioral-gate` | Closed |
| V3 | GLSL gate compiles stages without proving shipped programs link | `tools/shader-gate/glslGate.mjs` | Closed |
| V4 | Mechanical gap-closure mode can report proof success for identical roles | `tools/benchmark-runner/run-gap-closure-verification.mjs` | Closed |
| V5 | Walkaround radiometric regions assume a fixed 128-pixel domain | `tools/radiometric-ab/walkaround-ab.mjs` | Closed |
| V6 | Browser capture selectors can silently choose an unintended canvas | `tools/benchmark-runner/capture-adapter-playwright.mjs` | Closed |
| V7 | DZN/status numeric and selector assumptions fail open | `tools/behavioral-gate` | Closed |
| S1 | Material signatures collapse distinct packed float32 values | `packages/shared-bvh/src/materialSignature.ts` | Closed |
| S2 | Scene/BVH fingerprints omit behavior-affecting packed state | `packages/shared-bvh/src/sceneBvh.ts` | Closed |
| S3 | Displacement CPU sampling disagrees with texture sampler semantics | `packages/shared-bvh/src/vertexDisplacement.ts` | Closed |
| S4 | `tlas: false` can select an invalid direct-BLAS representation | `packages/shared-bvh/src/scenePack.ts` | Closed |
| H1 | Incremental material edits can leave the walkaround atlas and dependents stale | `packages/walkaround-hybrid/src/HybridEnginePrimitiveUpdates.ts` | Closed |
| H2 | RC optional material metadata can black valid opaque lighting | `packages/walkaround-rc/src/cascadeDispatch.ts` | Closed |
| H3 | RC alias validation and shader offsets can resolve different layouts | `packages/walkaround-rc/src/cascadeDispatch.ts` | Closed |
| T1 | BDPT delta-technique recurrence terminates before all valid alternatives | `packages/shared-samplers/src/bdptMIS.ts` | Closed |
| T2 | Full WebGPU surface-owned work executes before the medium-event race | `packages/pt-webgpu/src/wgsl/pathTrace/kernel.wgsl.ts` | Closed |
| T3 | Mesh-emitter sidedness is inconsistent across estimators and backends | `packages/pt-webgpu/src/scene/emitterPacking.ts` | Closed |
| T4 | WebGL2 shadow-disabled finite lights retain an absent MIS competitor | `packages/pt-webgl2/src/glsl/render/direct_light_contribution_function.glsl.js` | Closed |
| T5 | Progressive preview radiance contaminates the canonical terminal accumulator | `packages/engine/src/progressiveHandoff.ts` | Closed |
| T6 | WebGL2 repeats the same geometry sample when the host frame seed repeats | `packages/pt-webgl2/src/index.ts` | Closed |
| T7 | WebGL2 accepted camera, distance, visibility, and BDPT endpoint domains can enter non-finite or identity-ambiguous transport | `packages/pt-webgl2/src/gl/frameUniformsPacker.ts` | Closed |
| T8 | Environment radiance scaling and live replacement can overflow receiver products or publish a split resource generation | `packages/pt-webgpu/src/environmentRadianceScale.ts` | Closed |
| T9 | Presentation can double-encode sRGB, clip wide-target raw HDR, or publish infinity to a narrower attachment | `packages/walkaround-hybrid/src/presentationTarget.ts` | Closed |
| F1 | WebGPU lite tier advertises volume fields it does not transport coherently | `packages/pt-webgpu/src/wgsl/pathTrace/kernelLite.wgsl.ts` | Closed |
| F2 | Capsule and H-channel intersection routines return invalid boundaries | `packages/pt-webgpu/src/wgsl/pathTrace/intersectionCore.wgsl.ts` | Closed |
| F3 | Native emissive analytic primitives have no complete light-sampling path | `packages/pt-webgpu/src/scene/emitterPacking.ts` | Closed |
| F4 | ReSTIR-PT consumes one more shaded vertex than the configured bounce budget | `packages/pt-webgpu/src/wgsl/pathTrace/restirPtProducer.wgsl.ts` | Closed |
| F5 | WebGL2 rejects otherwise-supported filtered or mipmapped emissive maps | `packages/pt-webgl2/src/scene/meshAreaLights.ts` | Closed |
| N1 | SVGF reprojection demodulates previous radiance with current albedo | `packages/shared-denoisers/src/svgfRealCpu.ts` | Closed |
| N2 | SVGF packed history can exceed its 16-bit domain | `packages/shared-denoisers/src/svgfRealConstants.ts` | Closed |
| N3 | Float32 validation does not prove a value is finite after float16 conversion | `packages/shared-denoisers/src/halfFloat.ts` | Closed |
| N4 | One-shot BMFR demodulates historical radiance with current-frame albedo | `packages/shared-denoisers/src/bmfrWebGPU.ts` | Closed |
| A1 | glTF zero-valued clearcoat and iridescence state is discarded | `packages/gltf-adapter/src/gltfToScene.ts` | Closed |
| A2 | Analytic bounds and engine-scale derivation are incomplete | `packages/engine/src/sceneAABB.ts` | Closed |
| A3 | CPU analytic picking is not exact for the declared shape set | `packages/shared-bvh/src/pickPrimitiveCpu.ts` | Closed |
| A4 | Analytic fallback tessellation has no complete allocation budget | `packages/core/src/scene/analyticToMesh.ts` | Closed |
| A5 | Reflected skin deformation does not update tangent handedness | `packages/core/src/skinSolver.ts` | Closed |
| A6 | Custom decoded texture pixels can publish non-finite values | `packages/gltf-adapter/src/texturePipeline.ts` | Closed |
| L1 | `attachVitrum` does not contain every frame-preparation exception | `packages/engine/src/lifecycle/vanilla.ts` | Closed |
| L2 | A transient false DDGI initialization result becomes sticky | `packages/walkaround-hybrid/src/ddgi/DDGI.ts` | Closed |
| L3 | One-pixel GTAO dimensions can collapse to zero in shaders | `packages/walkaround-hybrid/src/shaders/gtao.wgsl.ts` | Closed |
| E1 | Executable examples disagree on camera, resize, DPR, and bounds handling | `examples` | Closed |

There are no open implementation rows.

Do not create proof-only, host-only, distribution, governance,
cross-host-evidence, or future-feature rows in this queue.

## Closed implementation programs

### 2026-07-30 final adversarial source closure

- Native WebGL2 now identifies the exact sampled light through NEE, visibility,
  forward-hit MIS, and bounded BDPT; spot/directional packing and analytic
  endpoints share one finite representation. Max-float ray sentinels remain
  unbounded rather than becoming giant finite occluders. Stable normalization,
  distance, inverse-square, cone, slab, and log-domain MIS helpers prevent
  accepted finite inputs from producing NaN/Infinity or losing the represented
  technique.
- WebGL2 frame validation is sample-independent and runs before program, target,
  accumulator, or debug-state mutation. It proves the packed camera-origin and
  transport domains, secondary-origin headroom, homogeneous division, and
  active thin-lens arithmetic. Aperture scaling mirrors shader-ordered f32
  operations, anamorphic scaling never evaluates an unsafe reciprocal, and
  refocusing uses a common scale instead of subtracting large world-space
  points. Zero-aperture equirectangular input remains the exact pinhole path;
  physically incoherent active equirectangular thin-lens input is rejected.
- WebGPU PT, walkaround, DDGI, RC, ReSTIR, NRC, SPPM, and BDPT environment
  consumers use staged finite-f32 radiance products. Host packers validate the
  scalar and map/global/material envelopes before publication, allow legitimate
  per-channel underflow, and reject overflow or complete positive-map collapse.
  Environment replacement prepares independent GPU resources, makes DDGI accept
  candidate bindings before the provider pointer swap, and retires the previous
  generation only after CPU, provider, and consumers agree.
- Exposure now saturates only at finite float32 inside the shared TS/WGSL/GLSL
  operator. Final presentation is bounded at the concrete target instead:
  unorm/sRGB and RGB10A2, per-channel R11G11B10 float, RGBA16F, or RGBA32F.
  Therefore `tonemap:'none'` preserves exposed HDR above 65,504 on a float32
  target without allowing the fixed half-float path-tracer outputs to publish
  infinity. Walkaround resolves one transfer policy before frame mutation and
  threads it through initialization, dynamic pipeline replacement, full,
  throttled, sky-only, and capture paths, applying exactly one sRGB OETF.
- The final shader inventory also exposed a stale CWBVH portable-composition
  harness call after the live range-intersection signature changed. The harness
  now invokes the exact shipped signature, restoring the source-derived WGSL
  gate rather than weakening or excluding that module.

### 2026-07-29 complete code-gap audit remediation

- Shared light-tree selection now accounts conservatively for node angular
  extent on CPU and WGSL, while the three renderer families agree on
  equirectangular orientation, packed alias indices, KHR punctual range
  attenuation, stable Henyey–Greenstein sampling/evaluation, negative-input
  Reinhard behavior, and scale-normalized skin-normal rank classification:
  tiny full-rank transforms remain invertible, rank-two transforms use their
  cofactor normal, and rank-below-two transforms return zero. Walkaround preserves
  fractional-metal diffuse GI and authored emissive intensity; WebGL2 corrects
  single-scatter phase throughput, default RGB thin film, and RG texture
  expansion; WebGPU lighting mutations rebuild emitters from the complete posed
  skin snapshot.
- Preetham baking integrates the physical solar cap over every intersected
  texel and conserves its analytic solid angle at polar and high-resolution
  placements. The spectral RGB fit exposes convergence, fails closed on a bad
  solve, and uses calibrated paired color transforms so its finite white
  endpoint and reported requested-target residual agree. The standalone
  à-trous variance pass estimates and consumes variance in the same radiance or
  demodulated domain and rejects mismatched temporal Welford moments. The reported
  binary16 subnormal-carry defect was disproved against the live implementation
  and is retained as a focused regression rather than a source rewrite.
- Core incremental mutation performs field-scoped validation without rereading
  unchanged vertex/index payloads, animation clips compile to owned validated
  samplers with binary interval lookup, inverse parameter domains match the
  material contract, and ordered iridescence bounds remain ordered. Backend
  mutation routes no longer layer redundant whole-scene validation on top.
- glTF import now handles malformed variants, default WebP source selection,
  material-profile inventory, application instance attributes, numeric
  CUBICSPLINE pointer channels, morphless weight targets, optional Draco
  fallback, browser ImageBitmap CPU readback, extended-range float sRGB,
  matrix-authored TRS animation, singular unrelated node matrices, and GLB
  transport suffixes through explicit validation or structured degradation.
- Attach/recreate and React canvas lifecycles serialize asynchronous ownership
  and roll back late failures; backend/profile switches are observable.
  Initial and replacement telemetry subscriptions publish transactionally:
  synchronous fatal delivery waits for the unsubscriber, partial registration
  failures unwind and terminally dispose the replacement lifecycle, and
  recovery owns exactly one RAF chain. Debug
  instrumentation is explicit and capability-reported across built-in engines,
  timestamp queries are no longer Vite-only, tiny scene bounds retain physical
  scale, WebGL2 can allocate on pre-frame `setSize()`, and sky-only presentation
  consumes the same exposure/tonemap/output-space contract as normal frames.
- Resource and diagnostic state is bounded: emissive coverage caps before
  interval materialization, inference candidates and failed attaches release
  ownership, WebGL2 atlases reserve only live layers, and OIDN validates safe
  attempt counts, finite ordered delays, clock values, and retry deadlines
  before persistent failures back off and latch. Scene-pack warning history is
  capped by last occurrence, while hosts receive only the current-operation
  warning delta. In-place WebGPU staging respects `maxBufferSize`; size-changing
  candidates reject live/candidate aliases and preserve the prior generation
  through allocation, upload, commit/rollback, and finalize failures. Error
  histories retain bounded current evidence, CWBVH build status
  proves or reports its traversal fallback, merge diagnostics reach all hosts,
  duplicate displacement work is eliminated, seeds clear every history, and
  WebGPU memory telemetry includes material mip chains, reservoirs, SPPM,
  presentation, and denoised-linear allocations.
- Every reported Class-F item was classified from live code: missing consumers
  or observability were added, genuinely dead ABI/code removed, and already-live
  or valid standalone host APIs retained with explicit coverage. Every optional
  mode is held to the same validation, diagnostics, lifecycle, and test
  requirements as default modes; no maturity label is an implementation or
  verification exemption.
- The corrected end-to-end WebGL2 proof exposed three estimator defects beyond
  the audit rows: terminal NEE competed with an unavailable continuation,
  bounded BDPT treated eye and light depth as independent budgets, and the
  delayed `k-2` reverse-density write copied `k-1` RGB throughput into its
  target. Terminal ownership now follows the reachable technique set, every
  connection obeys one total scattering-depth bound, and the delayed patch
  preserves the target vertex's throughput while replacing only its density.
  The inward-wound Cornell scene passes the unchanged PCG/BDPT agreement band
  at one, two, three, and six bounces, including a 128-SPP confirmation.

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
| `npm test` | PASS — 104/104 root script tests across 13 files and 8,433/8,433 workspace tests across 709 Vitest files (8,537 tests across 722 files combined; 3,121 walkaround-hybrid tests across 280 files) |
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
