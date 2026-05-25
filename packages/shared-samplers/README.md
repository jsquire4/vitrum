# @vitrum/shared-samplers

Sampling utilities for path tracers and walkaround engines: QMC sequences, light tree CDF, MIS heuristics, hero-wavelength MIS, spectral upsampling.

## Public surface

**Production-canonical (consumed across packages):**
- `HAMMERSLEY_WGSL` — QMC sequence WGSL helper. Consumed by pt-webgpu, walkaround-hybrid DDGI.
- `OCTAHEDRAL_CORE_WGSL` — canonical `octEncode` / `octDecode` (Cigolle 2014 §A.1 sign-zero-safe). Use this everywhere; the previously-inlined `sign()` form had a sign-zero collapse at axis boundaries.
- `defineUbo` — declarative WGSL UBO codegen helper (W2-C13). Pack + unpack + WGSL struct generation from a single field-type table.
- `CIE_X/Y/Z_TABLE`, `CIE_X/Y/Z_INTEGRAL`, `CIE_X/Y/Z_CDF`, `sampleHeroWavelengthMIS` — spectral sampling consumed by pt-webgl.

**Hoisted WGSL primitives (also exported from package index):**
- `PCG_WGSL`, `BSDF_PRIMITIVES_WGSL`, `LUMINANCE_WGSL` — canonical declarations; consumers should import via `wgslModules.ts` `requires` rather than inlining duplicates.

**Still duplicated in some consumer packages (open follow-up):**
- Per-pixel pixel-hash — some walkaround shaders still inline the legacy `fract(sin(dot(...)))` form; prefer `HASH_WGSL` from this package when touching those files.
- Halton+Shoemake uniform-SO(3) sampler — implemented inline in `ddgi/probeUpdatePass.ts:670+` (W6-E2 work). Not yet hoisted.

**Test/spec oracle (marked `@internal`, not re-exported from package index):**
The majority of TS modules in this package — `lightTree`, `mixturePdf`, `evaluateHG`, `sampleEquiAngular`, BDPT helpers (`_full` only; `_partial` deleted), Jakob–Hanika spectral upsampling approximation, Cauchy IoR formula — exist as oracle/spec references for the WGSL implementations that actually drive renderers. They have unit tests pinning numerical contracts but aren't imported by production code.

To deep-import an oracle for tests: `import { ... } from '@vitrum/shared-samplers/src/<module>.js';`.

## Status

Pre-1.0. Adding a new shared WGSL primitive: add the canonical declaration here and update consumers' `requires` lists in `@vitrum/walkaround-hybrid`'s `wgslModules.ts`.
