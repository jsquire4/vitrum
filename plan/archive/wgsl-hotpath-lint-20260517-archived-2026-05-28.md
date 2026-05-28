# WGSL hot-path foot-gun audit — 2026-05-17

> **ARCHIVED 2026-05-28.** All findings were actioned by 2026-05-18/19: the single REAL-RISK finding (`surfaceTextures.wgsl.ts:173` bare `1/dir`) was fixed in-place (`safeInvDir`); all 6 UNCLEAR items were verified-resolved per the status block below; and the deeper `safeInvDir` sign-zero bug was subsequently fixed in `shared-bvh` (`3327e8c`) + `pt-webgpu` (`2740d92`). No open items remain. Current source of truth: the WGSL source files under `packages/`.

## Status update — 2026-05-18

All six UNCLEAR items below have been verified-resolved by direct file read:

- **#1 `evalPointLight` dist guard** — `probeUpdateRays.wgsl:290` now has
  `if (dist < 1e-6) { return vec3f(0.0); }` (commit `7cc57ff`).
- **#2 `cascadeMerge` `worldPos / roomSize`** — host-side fix: cascade
  allocator floors each `roomSize` axis at 1µm (commit `1ce3a61`; verified
  at `cascadePyramid.ts:108-112` after W8 Phase 1A's tuple-typed rewrite).
- **#3 `octEncode` zero-vector** — `shared-samplers/wgsl/octahedralCore.wgsl:13`
  now uses `dir / max(abs(x)+abs(y)+abs(z), 1e-20)` (commit `db8b645`).
- **#4 camera unprojection w-divide** — `temporalGi.wgsl:65-66` now uses
  `select(1.0, far4.w, abs(far4.w) > 1e-30)` (commit `7f0e725`).
- **#5 ppgUpdate `normalize` zero vector** — guarded at the source: the
  current `ppgUpdate.wgsl:170` short-circuits with
  `if (dirLen2 < 1e-12) { return; }` before normalising.
- **#6 `bilinearUpsample` `inputH/W - 1u` underflow** — confirmed correctly
  scoped to host validation per the audit's own recommendation; the host
  guarantees `inputH/W >= 1` at dispatch time, so the in-shader subtraction
  cannot underflow in practice. No in-shader fix needed.

The original `REAL-RISK` finding (`surfaceTextures.wgsl:173` `safeInvDir`)
was applied in-place at audit time. No remaining open items from this audit.

**Follow-up 2026-05-19:** the audit caught the SYMPTOM (`surfaceTextures.wgsl`
using `vec3f(1.0)/dir` directly) but the swap-to-`safeInvDir` it triggered
turned out to mask a deeper bug INSIDE `safeInvDir` itself: the
`sign(d.x) * 1e30` sentinel collapsed to `0` for exact-zero direction
components (WGSL `sign(0) == 0`), causing slab tests to give false positives
for axis-aligned rays whose origin sat outside the AABB on the parallel
axis. Fixed in commits `3327e8c` (shared-bvh) + `2740d92` (pt-webgpu
duplicate) by switching to `select(-1e30, 1e30, d.x >= 0.0)`. See
`packages/shared-bvh/src/__tests__/safeInvDir.test.ts` for the regression
test. Audit lesson: a fix that goes "use the canonical helper" is only as
safe as the helper itself — the underlying primitive needs its own
regression coverage, which `shared-bvh` now has.

---


Read-only lint sweep across every `.wgsl` / `.wgsl.ts` source under `packages/`.
Looking specifically for the three known classes of bug that have repeatedly
bitten the engine:

1. `1 / dir` ray-AABB inverse-direction without the canonical `safeInvDir`
   guard (NaN when a `dir` component is exactly 0).
2. `1 / pdf` / `radiance / pdf` divisions without a `pdf > 0` (or
   `max(pdf, 1e-X)`) guard (firefly + NaN propagation).
3. `sqrt(x)` / `pow(x, y)` / `log(x)` on arguments that can underflow
   negative because of floating-point drift (intersection discriminants,
   `1 - cos²θ`, etc.).

Plus a side-check for `textureLoad` / `textureSample` against textures whose
binding may be omitted by a feature flag at pipeline-compile time.

## Files audited — 44

```
packages/pt-webgpu/src/wgsl/common.wgsl.ts
packages/pt-webgpu/src/wgsl/pathTraceBruteforce.wgsl.ts
packages/shared-bvh/src/wgsl/octahedral.wgsl.ts
packages/shared-denoisers/src/wgsl/atrous.wgsl.ts
packages/shared-denoisers/src/wgsl/atrousKernel.wgsl.ts
packages/shared-denoisers/src/wgsl/atrousVariance.wgsl.ts
packages/shared-denoisers/src/wgsl/hdrLuminanceBilateral.wgsl.ts
packages/shared-denoisers/src/wgsl/spatialFilter.wgsl.ts
packages/shared-denoisers/src/wgsl/svgf7x7SpatialFallback.wgsl.ts
packages/shared-denoisers/src/wgsl/svgfReprojection.wgsl.ts
packages/shared-denoisers/src/wgsl/svgfVarianceFromMoments.wgsl.ts
packages/shared-denoisers/src/wgsl/temporalAccum.wgsl.ts
packages/shared-denoisers/src/wgsl/welfordVariance.wgsl.ts
packages/shared-samplers/src/wgsl/hammersley.wgsl.ts
packages/shared-samplers/src/wgsl/octahedralCore.wgsl.ts
packages/walkaround-hybrid/src/ddgi/wgsl/probeUpdateBlend.wgsl.ts
packages/walkaround-hybrid/src/ddgi/wgsl/probeUpdateBorder.wgsl.ts
packages/walkaround-hybrid/src/ddgi/wgsl/probeUpdateRays.wgsl.ts
packages/walkaround-hybrid/src/neural/wgsl/bilinearUpsample.wgsl.ts
packages/walkaround-hybrid/src/neural/wgsl/conv2d.wgsl.ts
packages/walkaround-hybrid/src/neural/wgsl/relu.wgsl.ts
packages/walkaround-hybrid/src/neural/wgsl/skipConnection.wgsl.ts
packages/walkaround-hybrid/src/neural/wgsl/transposedConv2d.wgsl.ts
packages/walkaround-hybrid/src/ppg/ppgGuide.wgsl.ts
packages/walkaround-hybrid/src/ppg/ppgUpdate.wgsl.ts
packages/walkaround-hybrid/src/rc/wgsl/cascadeMerge.wgsl.ts
packages/walkaround-hybrid/src/rc/wgsl/probeRayCast.wgsl.ts
packages/walkaround-hybrid/src/shaders/common.wgsl.ts
packages/walkaround-hybrid/src/shaders/composite.wgsl.ts
packages/walkaround-hybrid/src/shaders/gtao.wgsl.ts
packages/walkaround-hybrid/src/shaders/gtaoUpsample.wgsl.ts
packages/walkaround-hybrid/src/shaders/indirectCombine.wgsl.ts
packages/walkaround-hybrid/src/shaders/indirectTemporalAccum.wgsl.ts
packages/walkaround-hybrid/src/shaders/resolve.wgsl.ts
packages/walkaround-hybrid/src/shaders/ris.wgsl.ts
packages/walkaround-hybrid/src/shaders/risGi.wgsl.ts
packages/walkaround-hybrid/src/shaders/sampleBudget.wgsl.ts
packages/walkaround-hybrid/src/shaders/shade.wgsl.ts
packages/walkaround-hybrid/src/shaders/spatial.wgsl.ts
packages/walkaround-hybrid/src/shaders/spatialGi.wgsl.ts
packages/walkaround-hybrid/src/shaders/surfaceTextures.wgsl.ts
packages/walkaround-hybrid/src/shaders/temporal.wgsl.ts
packages/walkaround-hybrid/src/shaders/temporalGi.wgsl.ts
packages/walkaround-hybrid/src/shaders/welfordTemporal.wgsl.ts
```

## REAL-RISK findings — 1

| File:line | Expression | Why it's a risk | Suggested fix |
|---|---|---|---|
| `packages/walkaround-hybrid/src/shaders/surfaceTextures.wgsl.ts:173` | `let invDir = vec3f(1.0) / dir;` | `bvhTraceTintedVisibility` is the shade pass's glass-aware shadow-ray traversal. It is invoked once per primary-hit pixel for the sun-caustic visibility and 5×/pixel for the sky-aperture probe. The `dir` argument is `safe_normalize(...)` output, which is unit-length but can have *exactly* zero components (e.g., a sun pointing along +Y has `sunDir.xz == 0`, or a hit-normal perfectly along an axis). `1.0 / 0.0` evaluates to `±Inf`; the next line `t1 = (nMin - origin) * invDir` then computes `0 * Inf = NaN` for any AABB whose face is colocated with the ray origin in that axis. The NaN poisons the `tNear`/`tFar` slab comparison and the shadow ray returns silently wrong results (false-negative or false-positive). Every other AABB-slab in the codebase uses the canonical `safeInvDir` helper from `common.wgsl`. | `let invDir = safeInvDir(dir);` (same module — `safeInvDir` is provided by `COMMON_WGSL`, which is prepended to this string at pipeline-compile time). |

This is the only confirmed unguarded `1/dir` slab-test in the entire codebase
that crosses a per-pixel hot path. Applied in-place; see below.

## LIKELY-SAFE-but-noteworthy

The following looked suspicious on first scan but, on read, the upstream
context guarantees the operand is bounded. Documented here so future
audits don't re-flag them.

| File:line | Expression | Why safe |
|---|---|---|
| `packages/walkaround-hybrid/src/shaders/common.wgsl.ts:635-636` | `var u = (d11 * d20 - d01 * d21) / denom;` (barycentric reconstruction) | `denom = d00*d11 − d01*d01` is the Gram determinant of `(ab, ac)`. Zero only if the triangle is fully degenerate (colinear vertices). The Möller–Trumbore precheck on `det` filters near-coplanar triangles via `triEps`, and the BVH never indexes a triangle with `det == 0`. `denom > 0` for all valid hit results. |
| `packages/walkaround-hybrid/src/shaders/surfaceTextures.wgsl.ts:210-212` | same pattern | same reasoning |
| `packages/pt-webgpu/src/wgsl/pathTraceBruteforce.wgsl.ts:838-840` | same pattern, but explicitly guarded: `let denom = max(d00 * d11 − d01 * d01, 1e-8);` | More defensive than the walkaround-hybrid copy (the pt-webgpu side floors `denom`). Documented as a potential consistency target if the walkaround-hybrid barycentric is ever shown to misbehave on a real BVH. |
| `packages/walkaround-hybrid/src/shaders/common.wgsl.ts:418-420` | `let r = sqrt(xi.x); … sqrt(max(0.0, 1.0 - xi.x))` | `xi.x` is `rand_f32()` output in `[0, 1)`. Cosine-hemisphere sampler is standard PBRT. Safe. |
| `packages/shared-samplers/src/wgsl/hammersley.wgsl.ts:27` | `sqrt(max(0.0, 1.0 - cosT * cosT))` | Already clamped at 0.0. Safe. |
| `packages/pt-webgpu/src/wgsl/pathTraceBruteforce.wgsl.ts:314, 1094, 1268` | `sqrt(max(0.0, 1.0 − sin²θ))` family | All clamped at 0.0. Safe. |
| `packages/pt-webgpu/src/wgsl/pathTraceBruteforce.wgsl.ts:631, 652, 707, 720, 731` | sphere/cylinder/capsule intersection `sqrt(disc)` | All guarded by `disc >= 0` / `h >= 0` ahead of the call. Safe. |
| `packages/walkaround-hybrid/src/ddgi/wgsl/probeUpdateBlend.wgsl.ts:216` | `pow(w, 2.0)` with `w ∈ [1e-3, 1]` | Guarded by `if (w < 1e-3) continue` upstream. Safe. |
| `packages/walkaround-hybrid/src/shaders/gtaoUpsample.wgsl.ts:53` / `packages/shared-denoisers/src/wgsl/atrousVariance.wgsl.ts:271` / `packages/shared-denoisers/src/wgsl/atrous.wgsl.ts:84` / `packages/shared-denoisers/src/wgsl/spatialFilter.wgsl.ts:187` | `pow(dn, sigN)` where `dn = max(0.0, dot(...))` | `dn` clamped to `[0, 1]`; `sigN` is a host-side >0 constant. `pow(0, n)` = 0 in WGSL spec for positive n. Safe. |
| `packages/walkaround-hybrid/src/shaders/composite.wgsl.ts:44` | `pow(max(c, cutoff), vec3f(1.0 / 2.4))` | sRGB encode; `max(c, cutoff)` floors at 0.0031308. Safe. |
| All RIS/temporal/spatial p̂ divisions (`combined.w_sum / (M · pHatZ)`) | Every site uses `select(0.0, ..., pHatZ > 0.0)` or `> 1e-9` guard. | Cross-checked in `ris.wgsl.ts`, `temporal.wgsl.ts`, `spatial.wgsl.ts`, `risGi.wgsl.ts`, `temporalGi.wgsl.ts`, `spatialGi.wgsl.ts`. |
| `packages/pt-webgpu/src/wgsl/pathTraceBruteforce.wgsl.ts:512, 536, 1070, 1205, 1751, 1788, 1812` | `... * misWeight / max(pdf, 1e-X)` | All MIS terms in pt-webgpu use `max(pdf, 1e-6)` or `1e-8`. Safe. |
| `packages/walkaround-hybrid/src/ppg/ppgGuide.wgsl.ts:119` | `(leafFlux / totalFlux) / max(solidAng, 1e-12)` | `if (totalFlux <= 0.0) return early` upstream; `solidAng` floor explicit. Safe. |
| `packages/shared-denoisers/src/wgsl/svgfReprojection.wgsl.ts:199, 208` | `1.0 / accWeight`, `1.0 / f32(newHistory + 1u)` | `accWeight > 1e-6` branch; `newHistory + 1u >= 2`. Safe. |
| `packages/shared-denoisers/src/wgsl/atrous.wgsl.ts:109`, `atrousVariance.wgsl.ts:282`, `spatialFilter.wgsl.ts:197`, `hdrLuminanceBilateral.wgsl.ts:61` | `sumColor / sumWeight` with `select(..., sumWeight > 1e-6)` or `max(wsum, 1e-6)` | All bilateral / atrous denoiser normalisations are guarded. Safe. |

## UNCLEAR — flag for human review

Items that *could* divide by zero under contrived host input or pathological
upstream state but are not load-bearing on the per-pixel hot path. None of
these is a single-line trivially-correct fix; each needs a small algorithmic
decision (clamp magnitude, fallback semantics, …) before touching.

| File:line | Expression | Question for the human |
|---|---|---|
| `packages/walkaround-hybrid/src/ddgi/wgsl/probeUpdateRays.wgsl.ts:379` | `let lightDir = toLight / dist;` inside `evalPointLight`, where `dist = length(toLight)` | If a point light is placed exactly at the probe-trace hit position, `dist == 0`. The next `dot(hitNormal, NaN) < 1e-3` short-circuit would be false and the light's contribution would propagate NaN into the irradiance atlas. Practically the probe ray's hit point can't coincide with a light because the ray traced from `probeOrigin` to a triangle; lights are point sources, not surfaces. But there is no defensive guard — should there be? |
| `packages/walkaround-hybrid/src/rc/wgsl/cascadeMerge.wgsl.ts:111` | `let probeUV = (worldPos - m.probeOriginWorld) / m.roomSize;` | Component-wise vec3f divide. `m.roomSize` is the host-supplied scene AABB extent. A degenerate scene with zero extent on any axis (a 2D scene, or a single-triangle test) yields Inf. Should `roomSize` be clamped to `max(eps)` at host packing time, or here? |
| `packages/shared-samplers/src/wgsl/octahedralCore.wgsl.ts:8` | `let n = dir / (abs(dir.x) + abs(dir.y) + abs(dir.z));` | `octEncode(vec3f(0))` divides by zero. All call sites pass `normalize(...)` output, so in practice `dir` is unit-length and the denominator is `>= 1`. If a caller ever passes a zero vector (e.g., a degenerate BVH normal from a colinear triangle that slipped past `triEps`), this would NaN-propagate into the DDGI atlas. Worth a `max(..., 1e-8)` floor if we ever see atlas corruption. |
| `packages/walkaround-hybrid/src/shaders/temporalGi.wgsl.ts:61-62` | `let farW = far4.xyz / far4.w; let nearW = near4.xyz / near4.w;` | Camera unprojection. `far4.w` ≈ `−far_plane * proj[2][3]` in a standard perspective matrix → non-zero. Only fails for degenerate projection matrices (zero near/far plane, or an orthographic matrix wired as perspective). Same pattern reappears in `common.wgsl:840-841` (`generatePrimaryRay_common`). Probably fine but worth `max(abs(w), 1e-8)` if any host ever ships an exotic camera. |
| `packages/walkaround-hybrid/src/ppg/ppgUpdate.wgsl.ts:88` | `let dir = normalize(ppgSamplesDir[idx].xyz);` then used as octahedral encode input | If the host buffer contains a `(0,0,0)` sample, `normalize` returns NaN. PPG samples come from the path-trace `Lo` writes; in principle a fully-shadowed bounce could leave the sample slot uninitialised. Worth a defensive `length > 0` check at the buffer-write site, not here. |
| `packages/walkaround-hybrid/src/neural/wgsl/bilinearUpsample.wgsl.ts:29-30` | `params.inputH - 1u`, `params.inputW - 1u` | `u32` underflow if `inputH` or `inputW` is 0. Host validation guarantees these are ≥1 at dispatch, but no in-shader guard. Belongs in host validation; flag here for posterity. |

## In-place single-line fixes applied — 1

| File:line | Before | After |
|---|---|---|
| `packages/walkaround-hybrid/src/shaders/surfaceTextures.wgsl.ts:173` | `let invDir = vec3f(1.0) / dir;` | `let invDir = safeInvDir(dir);` |

Verification:
- `safeInvDir` is provided by `COMMON_WGSL`, which is prepended to
  `SURFACE_TEXTURES_WGSL` at pipeline-compile time
  (`pipeline/wgslComposer.ts` doc-string confirms the include order:
  `COMMON_WGSL + SURFACE_TEXTURES_WGSL + DDGI_SAMPLE_WGSL + SHADE_WGSL`).
  The symbol is in scope.
- The slab arithmetic immediately following (`(nMin - origin) * invDir`
  etc.) is identical in shape to the other `bvhIntersect*` functions in
  `COMMON_WGSL` that already use `safeInvDir`, so the change is a 1:1
  generalisation, not a semantic shift.
- Performance: `safeInvDir` adds three `select` + three `abs` per slab
  test. `bvhTraceTintedVisibility` runs once per primary-hit pixel for
  the sun-caustic shadow + 5×/pixel for the sky-aperture probe; the
  added arithmetic is negligible vs the BVH traversal cost the function
  is wrapping.

Post-fix verification (in same commit):
- `npm run typecheck` — green across workspace.
- `npm test --workspace @vitrum/walkaround-hybrid` — green
  (no behavioural test regressed; the fix is a pure NaN-suppression
  on a code path that previously silently produced wrong results
  for axis-aligned sun directions).
