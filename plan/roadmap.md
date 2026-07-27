# vitrum roadmap

**Updated:** 2026-07-24

This document records possible capability expansion after the current professional
profile. It is not an implementation-completion queue. The finite code closeout is
tracked in [road-to-100.md](./road-to-100.md), and current backend semantics are
recorded in [renderer-fidelity-matrix.md](./renderer-fidelity-matrix.md).

## Current release boundary

The library currently exposes four stable renderer surfaces:

- `@vitrum/pt-webgpu`: full-tier converged path tracing with spectral transport,
  bounded general BDPT, SPPM, manifold caustics, ReSTIR-PT, layered and volume
  transport, denoising, inverse sessions, and transactional mutation; plus a
  capability-bounded lite compatibility tier.
- `@vitrum/pt-webgl2`: native WebGL2 converged path tracing with the documented
  bounded general-BDPT caustic mode, spectral transport, denoising, inverse sessions,
  and incremental/fallback mutation modes.
- `@vitrum/walkaround-hybrid`: realtime DDGI, ReSTIR-DI/GI, GTAO, SVGF/BMFR/neural
  denoising, PPG, NRC, optional Radiance Cascades, rich-material GI, transparent
  secondary transport, and realtime caustics.
- `@vitrum/engine`: host-owned lifecycle, backend selection, progressive handoff,
  strict scene validation, glTF integration, React/vanilla hosts, telemetry, and
  failure-contained resource publication.

The API intentionally reports bounded models and unsupported domains. Those are
contract boundaries, not deferred implementations of an advertised feature.

## Direction A — geometry and asset breadth

Potential additions:

1. Native point and line primitives instead of generated triangle geometry.
2. A first-class instanced-skinned/morphed primitive that avoids glTF fallback
   expansion while retaining per-instance animation control.
3. Additional standardized compression and texture codecs as browser-portable
   implementations become practical.
4. More `KHR_animation_pointer` target families where the mutable core contract has
   a precise transactional representation.
5. Adaptive, error-bounded displacement rather than bounded uniform subdivision.

These additions require a new public contract and must not be inferred from the
existing generated-mesh or strict-rejection paths.

## Direction B — transport breadth

Potential additions:

1. Heterogeneous participating media with spatially varying extinction, scattering,
   and emission.
2. Spectral fluorescence and wavelength-changing transport.
3. More general transient/time-of-flight rendering.
4. Polarisation-aware transport where a compact browser-friendly representation can
   be justified.
5. Additional inverse-rendering domains beyond the current analytic replay and
   explicit finite-difference methods.

Every transport addition needs matching sampling, evaluation, forward/reverse PDFs,
MIS ownership, and a bounded numerical oracle before it enters the public API.

## Direction C — performance architecture

Potential additions:

1. Wavefront path scheduling when profiling shows enough coherent work to offset
   queue and compaction overhead in browser WebGPU.
2. Wider/compressed BVH traversal as a default only after scene-build, update, memory,
   and traversal tradeoffs beat the current path across representative content.
3. Subgroup and `f16` acceleration behind feature detection with an equivalent
   baseline path.
4. Temporal upscaling and dynamic-resolution control with explicit history ownership.
5. Standardized browser ray-query acceleration if a portable WebGPU extension ships;
   no vendor-only or RTX-only dependency is acceptable.

Performance work must preserve the current deterministic fallback and host-owned
lifecycle contracts.

## Direction D — realtime GI evolution

Potential additions:

1. Adaptive DDGI/RC placement and update budgets driven by measured scene change.
2. More aggressive shared scene-storage compaction across DDGI, ReSTIR, RC, PPG, and
   NRC without reintroducing hidden ownership.
3. Learned guiding/cache models with independently generated teachers and certified
   checkpoint manifests.
4. More advanced glossy-radiance cache representations when they demonstrably improve
   the current rich-lobe transport.
5. Multi-resolution transparency histories for dense layered scenes.

Each addition must retain transactional publication and have a non-learned fallback.

## Direction E — tools and host integration

Potential additions:

1. Scene-inspection and capture tooling based only on public engine APIs.
2. Worker/off-main-thread host adapters with explicit transferable ownership.
3. Framework bindings beyond React where lifecycle semantics can remain identical.
4. Stable scene/package serialization for prebuilt BVH, texture, and model assets.
5. Higher-level profiling surfaces built on the existing telemetry contract.

## Scheduling rule

A roadmap direction becomes active implementation work only when all of these are
available:

1. a concrete user/content need;
2. a verified public algorithm or specification;
3. a browser-portable implementation route;
4. an explicit core/backend contract;
5. a bounded performance and memory budget;
6. an independent correctness oracle;
7. a migration and fallback story for existing hosts.

Until then, the item remains a possible product expansion and does not reduce the
completion percentage of the currently declared library.
