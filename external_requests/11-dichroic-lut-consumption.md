# Request 11 — Dichroic LUT consumption in `@vitrum/three-bindings`

**Filed by**: stainedGlass (`feat/vitrum-step-11-physics-dichroic`, complexity-sweep
remediation PHY.1 dichroic addendum, 2026-05-12).

**Status**: APPLIED (RFE-10-addendum). The userDataKeys entries and three-bindings
reader edits were applied directly in this same commit, per the host-side
"fully implement; don't pause for vitrum/fork" directive — this doc is the
record-of-changes, not a request awaiting action.

(Filename note: the immediate sequence number `10-` was already taken by
`10-surface-texture-id-contract.md`; this file uses `11-` and the body refers
to the work as the "RFE-10 dichroic addendum" because it extends the
already-applied RFE-10 userData-propagation pass.)

## What changed in stainedGlass

The dichroic body baker no longer returns an empty `BakedMaps` — it now
performs the full per-angle TMM × CIE 1931 convolution and returns two
256×1 RGBA-float `DataTexture`s:

- `reflectanceLUT`: angle (θ ∈ [0°, 89°], 256 bins) → linear sRGB reflectance.
- `transmittanceLUT`: same indexing for transmittance.
- `dichroicMode: 'tmm-1d'` discriminator on the merged BakedMaps.

`createBakedGlassMaterial` stamps both textures onto the
`MeshPhysicalMaterial.userData` under two new keys:

- `userData.vitrumDichroicReflectanceLUT`
- `userData.vitrumDichroicTransmittanceLUT`

Both are guaranteed-present on dichroic profiles and `undefined` on every
other glass type. The existing `userData.vitrumThinFilmStack` stamp (the
raw stack data) is still emitted in parallel — the LUT is the
pre-convolved view and the stack is the source-of-truth definition.
PT consumers can use either.

## Vitrum-side changes APPLIED in this commit

### `packages/three-bindings/src/userDataKeys.ts`

Two new entries added to `VITRUM_USER_DATA_KEYS`:

```ts
DICHROIC_REFLECTANCE_LUT:   'vitrumDichroicReflectanceLUT',
DICHROIC_TRANSMITTANCE_LUT: 'vitrumDichroicTransmittanceLUT',
```

### `packages/three-bindings/src/material.ts`

`convertMaterial` reads both keys (if present on `m.userData`) and forwards
them through `Material.extensions.dichroicLUTs`. Backends that recognise
the field can bind the LUT directly to a fragment program; backends that
do not are unaffected (the existing extensions escape-hatch).

The `Material.extensions` map is the canonical contract for backend-specific
data per `@vitrum/core/scene.ts` ("Backends may read keyed fields from here
for backend-specific features. Core never inspects this map."). Two reasons
for using `extensions` over a first-class field:

1. The LUT is a host-side artifact of the raster baking pipeline — it's not
   a universal physical property of the material. A WebGPU PT backend that
   prefers to evaluate the TMM in-shader (per RFE-14, already shipped) can
   ignore the LUT and use `thinFilmStack` directly.
2. `Material.thinFilmStack` is and remains the source-of-truth field;
   `extensions.dichroicLUTs` is a fast-path that says "the host has already
   convolved this stack against CIE 1931 + D65, here are the bins."

## Backend consumer guidance

A raster backend (the stainedGlass app's three.js fragment-shader path) reads
the LUTs like this:

```glsl
// per-shade-point:
float angle  = acos(clamp(dot(N, V), 0.0, 1.0));   // 0..π/2
float u      = angle / (3.14159265 * 0.5);          // 0..1
vec3  refl   = texture(uDichroicReflectanceLUT, vec2(u, 0.5)).rgb;
vec3  trans  = texture(uDichroicTransmittanceLUT, vec2(u, 0.5)).rgb;
// modulate the specular lobe with refl, and the transmission lobe with trans.
```

A WebGPU PT backend that prefers to evaluate the TMM in-shader (the fork
RFE-14 path) can skip the LUT and read `thinFilmStack` from the material.
Both paths agree at the design wavelength up to floating-point noise; the
host-LUT path is cheaper per shade.

## Tests added in stainedGlass

- `packages/stained-glass-physics/src/__tests__/tmm.test.ts` —
  closed-form anchor checks for the TMM evaluator (single quarter-wave
  layer at 0° + 60°; 35-layer dichroic stack near-total reflectance at
  the design wavelength).
- `packages/stained-glass-physics/src/__tests__/dichroicLut.test.ts` —
  LUT shape contract (256×1 RGBA float DataTexture; correct format and
  type); angle-dependent variation regression (the 0° row differs from
  the 89° row by > 0.02 per channel).

## Open follow-ups (vitrum-side)

- A `pt-webgl` test that exercises the round-trip
  `THREE userData → vitrum Material.extensions.dichroicLUTs` is appropriate
  but not strictly required — the existing
  `material-vitrum-roundtrip.test.ts` pattern can be extended.
- A WebGPU PT backend that wants the LUT fast-path (currently only the
  WebGL2 fork's in-shader TMM is on the production path) can opt in later;
  the contract is already in place.
