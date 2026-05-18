# Request 10 — SURFACE_TEXTURE_ID wire-format contract

**Filed by**: stainedGlass (`feat/vitrum-theme-f-glass-edges`, complexity-sweep remediation
Theme F decision F.2, 2026-05-12).

**Status**: informational — no immediate vitrum action required.

## What changed in stainedGlass

The canonical `SURFACE_TEXTURE_ID` integer enum, previously inlined inside
`createBakedGlassMaterial.ts`, now lives in a dedicated module:

```
packages/stained-glass-physics/src/baking/surfaceTextureIds.ts
```

It is re-exported from `@stained-glass/physics/baking` and `@stained-glass/physics`.
Mapping (unchanged, do not renumber):

```
smooth=0  hammered=1  ripple=2     granite=3
baroque=4 waterglass=5 catspaw=6   flemish=7
```

The integer is still stamped onto `THREE.Material.userData.surfaceTextureId`
during glass-material baking — the wire format is identical to what vitrum
already consumes.

## What this means for vitrum

Vitrum currently reads `surfaceTextureId` from material userData in:

- `packages/walkaround-hybrid/src/restir/packingHelpers.ts:127` —
  packs the value into the low 4 bits of `bvhIndex[*].w`.
- `packages/walkaround-hybrid/src/shaders/surfaceTextures.wgsl.ts:117-123` —
  WGSL `switch (texId) { case 1u: ... }` dispatch by integer.
- `packages/walkaround-hybrid/src/shaders/common.wgsl.ts:681-682` —
  comment documenting the integer mapping for shader authors.

No vitrum-side TypeScript table needs to change. The contract is purely
the integer wire format on userData, and that wire format is identical to
the values produced by stainedGlass today.

## Future option (deferred)

Once the @vitrum/\* → @stained-glass/physics dependency direction is sorted
(currently stainedGlass depends on vitrum; cross-import in the other
direction is not yet set up), vitrum could optionally:

1. Import `SURFACE_TEXTURE_ID` from `@stained-glass/physics/baking` in any
   future bridge code that needs to map texture-name strings to integers
   on the vitrum side (e.g. authoring helpers, debug tooling).
2. Add a runtime assertion in `sceneFromThreeJS` that any incoming
   `userData.surfaceTextureId` is a known integer in `[0, 7]`, using the
   imported table as the source of truth.

Neither is required for current functionality. Filing this note so the
contract is discoverable from vitrum if the renumber question ever comes
up again.
