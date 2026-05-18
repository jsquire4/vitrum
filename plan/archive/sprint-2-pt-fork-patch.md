# Sprint 2 PT fork patch — per-light `power` field

> **Status**: Deferred — requires visual A/B render verification in the fork.
> The vitrum-side prep work (userData.cellPower on each THREE light object) is
> complete. Apply this patch when the fork is being actively modified for
> Sprint 3 (light tree), since Sprint 3 will read `light.power` directly.

---

## What needs to change and why

Sprint 2's PT-side goal is to expose `cellPower = luminance(Le) × area` for
each light in the three-gpu-pathtracer lights texture, so Sprint 3's light tree
CDF builder can use it for power-weighted sampling without re-computing it in
GLSL.

The fork is at: `~/projects/three-gpu-pathtracer/` (branch `phase4-normalmap-shadow-rays`).

---

## File 1: `src/shader/structs/lights_struct.glsl.js`

**Current `Light` struct** (lines 16–37):

```glsl
struct Light {
    vec3 position;
    int type;

    vec3 color;
    float intensity;

    vec3 u;
    vec3 v;
    float area;

    // spot light fields
    float radius;
    float near;
    float decay;
    float distance;
    float coneCos;
    float penumbraCos;
    int iesProfile;
};
```

**Add `power` field** after `area`:

```glsl
struct Light {
    vec3 position;
    int type;

    vec3 color;
    float intensity;

    vec3 u;
    vec3 v;
    float area;

    // Sprint 2 (Phase 6): total radiant flux = luminance(color × intensity) × area.
    // For directional lights: 0 (no finite surface area).
    // For point/spot lights: luminance(color × intensity) (point flux convention).
    // Set in LightsInfoUniformStruct.updateFrom() from userData.cellPower.
    float power;

    // spot light fields
    float radius;
    float near;
    float decay;
    float distance;
    float coneCos;
    float penumbraCos;
    int iesProfile;
};
```

**Update `readLightInfo`** to populate `power` from the next available pixel slot.
Currently LIGHT_PIXELS = 6 (in LightsInfoUniformStruct.js). `power` can be packed
into sample 5's `.a` channel (currently unused for rect-area lights) or by bumping
LIGHT_PIXELS to 7. Packing into s5.a is least disruptive:

```glsl
// In readLightInfo(), after reading s3:
vec4 s4 = texelFetch1D( tex, i + 4u );
vec4 s5 = texelFetch1D( tex, i + 5u );

l.power = s5.a;   // Sprint 2: populated by LightsInfoUniformStruct for all light types
```

For rect-area lights (which currently only read s0–s3), add s4/s5 reads:

```glsl
} else {
    // rect-area and dir lights: read s4/s5 for the power field
    vec4 s4 = texelFetch1D( tex, i + 4u );
    vec4 s5 = texelFetch1D( tex, i + 5u );
    l.radius = 0.0;
    l.decay = 0.0;
    l.distance = 0.0;
    l.coneCos = 0.0;
    l.penumbraCos = 0.0;
    l.iesProfile = -1;
    l.power = s5.a;
}
```

---

## File 2: `src/uniforms/LightsInfoUniformStruct.js`

**Current layout** (line 4): `const LIGHT_PIXELS = 6;` — 6 RGBA pixels per light = 24 floats.

**Change**: No pixel count change needed. `s5.a` (the 4th channel of sample 5) is
currently written as 0 for rect-area and directional lights (the buffer is zeroed at
the top of the loop on lines 62–65). We just need to write `cellPower` into that slot.

**Locate the per-light encoding loop** starting at line 54.

**At the end of the loop**, after all light-type-specific writes, add:

```js
// Sprint 2 (Phase 6): write userData.cellPower into s5.a for ALL light types.
// vitrumSceneToThree() sets userData.cellPower on every THREE light created from
// a @vitrum/core SceneEmitter. Fallback = 0 for lights not from vitrum.
const s5aIndex = baseIndex + 5 * 4 + 3; // pixel 5, channel .a
floatArray[s5aIndex] = l.userData?.cellPower ?? 0;
```

`baseIndex = i * LIGHT_PIXELS * 4` so pixel 5 starts at `baseIndex + 5*4 = baseIndex + 20`,
and channel `.a` is at `baseIndex + 20 + 3 = baseIndex + 23`.

---

## File 3: `src/shader/sampling/light_sampling_functions.glsl.js`

No changes needed in Sprint 2. Sprint 3's light tree builder (a CPU-side CDF) will
be built from the JavaScript side reading `l.userData.cellPower` directly (same
data that was packed into the texture). The GLSL sampler code in this file is used
at shade time; Sprint 3's CDF is precomputed before any shader runs.

---

## Verification protocol

This patch REQUIRES a visual A/B render before committing:

1. Capture a reference render of the test scene at 64 samples PT_FINAL (before patch).
2. Apply the patch.
3. Capture the same render at 64 samples PT_FINAL (after patch).
4. Pixel-diff the two renders: they MUST be identical (cellPower is not read by any
   GLSL sampling code in Sprint 2 — it's a passive data field on the Light struct).

If the renders differ, the `readLightInfo` change incorrectly loaded `power` into a
field that IS used by a shader (e.g., `radius` or `iesProfile`). Re-check the pixel
offsets in step 5 of sample 5.

**Approved for Sprint 3 application** — no blocking issues; risk is low because
`s5.a` is currently unused for rect-area and directional lights.

---

## Summary for host integrator

To apply the fork patch:

1. Edit `~/projects/three-gpu-pathtracer/src/shader/structs/lights_struct.glsl.js`
   — add `float power;` field to `Light` struct (after `float area;`).
2. Update `readLightInfo` to read `s5.a` into `l.power` for all light types.
3. Edit `~/projects/three-gpu-pathtracer/src/uniforms/LightsInfoUniformStruct.js`
   — write `l.userData?.cellPower ?? 0` into `floatArray[ baseIndex + 23 ]` inside
   the per-light encoding loop.
4. Run the visual A/B verification above.
5. `npm run build` in the fork directory.
6. In `packages/pt-webgl/`: `npm install file:../../../three-gpu-pathtracer` to pick
   up the updated fork.

The vitrum-side prep (`userData.cellPower` on each light in `vitrumSceneToThree`) is
already committed. No further vitrum-side changes needed for this fork patch.
