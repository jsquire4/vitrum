# @vitrum/shared-samplers

Sampling utilities for path tracers and walkaround engines: QMC sequences, light tree CDF, MIS heuristics, hero-wavelength MIS, spectral upsampling.

## Public surface

**Production-canonical (consumed across packages):**
- `HAMMERSLEY_WGSL` — QMC sequence WGSL helper. Consumed by pt-webgpu, walkaround-hybrid DDGI.
- `OCTAHEDRAL_CORE_WGSL` — canonical `octEncode` / `octDecode` (Cigolle 2014 §A.1 sign-zero-safe). Use this everywhere; the previously-inlined `sign()` form had a sign-zero collapse at axis boundaries.
- `defineUbo` — declarative WGSL UBO codegen helper (W2-C13). Pack + unpack + WGSL struct generation from a single field-type table.
- `CIE_X_TABLE`, `CIE_Y_TABLE`, `CIE_Z_TABLE`, `X_CMF_INTEGRAL`, `Y_CMF_INTEGRAL`, `Z_CMF_INTEGRAL`, `X_CMF_CDF`, `Y_CMF_CDF`, `Z_CMF_CDF`, `sampleHeroWavelengthMIS` — spectral sampling consumed by pt-webgl2 (`glResources.ts`, `composeTraceGlsl.ts`) and pt-webgpu (`kernel.wgsl.ts`, `kernelLite.wgsl.ts`).
- `haltonSO3AxisAngleFromFrameIndex` — deterministic Halton-Shoemake SO(3) axis-angle sampler used by walkaround DDGI probe rotation.

**Hoisted WGSL primitives (also exported from package index):**
- `PCG_WGSL`, `PCG_HASH_TO_F32_WGSL`, `BSDF_PRIMITIVES_WGSL`, `LUMINANCE_WGSL` — canonical declarations; consumers should import via `wgslModules.ts` `requires` rather than inlining duplicates.

**Consumer dedup status:**
- Per-pixel/cell pixel-hash — walkaround deterministic shade/transparent/noise call sites now route through shared helpers; render-sensitive direct-sun jitter keeps its legacy-compatible sequence inside the helper, while cell/world hashes use `PCG_HASH_TO_F32_WGSL`. GTAO keeps its established local jitter sequence because the T1 dzn/lavapipe smoke is calibrated to that low-res AO pattern.

**Standalone CPU/host utilities (zero backend imports are intentional):**
- `mixturePdf` — general N-strategy mixture-PDF evaluator for host algorithms
  and numerical oracles.
- `sampleEquiAngular` — homogeneous-volume distance sampler for host algorithms
  and CPU reference implementations.

**Also exported from the package index:**
`buildLightTree`, `evaluateHG`, BDPT MIS helpers (`bdptConnectionMIS_full`,
`BDPTFullVertex`; `_partial` variants deleted), Jakob–Hanika spectral upsampling
(`rgbToSpectralCoefficients`), and the Cauchy IOR formula (`cauchyIOR`) are
available to consumers. Unit tests pin their numerical contracts. BDPT vertex
storage is deliberately backend-owned: the WebGL2 and WebGPU renderers retain
different transport payloads and packed layouts, so this package does not
publish a purportedly canonical vertex packer.

To deep-import an oracle for tests: `import { ... } from '@vitrum/shared-samplers/src/<module>.js';`.

## Status

Pre-1.0. Adding a new shared WGSL primitive: add the canonical declaration here and update consumers' `requires` lists in `@vitrum/walkaround-hybrid`'s `wgslModules.ts`.
