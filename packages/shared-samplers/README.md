# @vitrum/shared-samplers

Sampling utilities for path tracers and walkaround engines: QMC sequences, light tree CDF, MIS heuristics, hero-wavelength MIS, spectral upsampling.

## Public surface

**Production-canonical (consumed across packages):**
- `HAMMERSLEY_WGSL` — QMC sequence WGSL helper. Consumed by pt-webgpu, walkaround-hybrid DDGI.
- `OCTAHEDRAL_CORE_WGSL` — canonical `octEncode` / `octDecode` (Cigolle 2014 §A.1 sign-zero-safe). Use this everywhere; the previously-inlined `sign()` form had a sign-zero collapse at axis boundaries.
- `defineUbo` — declarative WGSL UBO codegen helper (W2-C13). Pack + unpack + WGSL struct generation from a single field-type table.
- `CIE_X_TABLE`, `CIE_Y_TABLE`, `CIE_Z_TABLE`, `X_CMF_INTEGRAL`, `Y_CMF_INTEGRAL`, `Z_CMF_INTEGRAL`, `X_CMF_CDF`, `Y_CMF_CDF`, `Z_CMF_CDF`, `sampleHeroWavelengthMIS` — spectral sampling consumed by pt-webgl.

**Hoisted WGSL primitives (also exported from package index):**
- `PCG_WGSL`, `PCG_HASH_TO_F32_WGSL`, `BSDF_PRIMITIVES_WGSL`, `LUMINANCE_WGSL` — canonical declarations; consumers should import via `wgslModules.ts` `requires` rather than inlining duplicates.

**Still duplicated in some consumer packages (open follow-up):**
- Per-pixel pixel-hash — some walkaround shaders still inline the legacy `fract(sin(dot(...)))` form; prefer `PCG_WGSL` from this package when touching those files.
- Halton+Shoemake uniform-SO(3) sampler — implemented in `ddgi/probeUpdateFrameParams.ts` (`haltonSO3AxisAngleFromFrameIndex`). Not yet hoisted into the package index.

**Also exported from the package index:**
`buildLightTree`, `mixturePdf`, `evaluateHG`, `sampleEquiAngular`, BDPT helpers (`bdptConnectionMIS_full`, vertex/MIS types; `_partial` variants deleted), Jakob–Hanika spectral upsampling (`rgbToSpectralCoefficients`), Cauchy IOR formula (`cauchyIOR`) — all are exported from the package index and available to consumers. Unit tests pin their numerical contracts.

To deep-import an oracle for tests: `import { ... } from '@vitrum/shared-samplers/src/<module>.js';`.

## Status

Pre-1.0. Adding a new shared WGSL primitive: add the canonical declaration here and update consumers' `requires` lists in `@vitrum/walkaround-hybrid`'s `wgslModules.ts`.
