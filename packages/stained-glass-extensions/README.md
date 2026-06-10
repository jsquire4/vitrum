# @vitrum/stained-glass-extensions

Stained-glass-specific contracts and host-app helpers extracted from generic
vitrum packages.

This package is a **host-app utility**, not a render backend. It does not
produce GPU commands. It owns the wire-level contracts and CPU-side packers
that host applications use when building stained-glass scenes on top of
`@vitrum/core`.

## Exports

### `SURFACE_TEXTURE_ID`

Wire-level integer enum for the stained-glass surface texture type stamped
on scene primitives. Values are GPU-packed into BVH index words and consumed
by WGSL surface-texture switch statements in `@vitrum/walkaround-hybrid`.

**Do not renumber entries** — renumbering silently corrupts rendering. Add new
entries at the next unused integer.

```ts
import { SURFACE_TEXTURE_ID } from '@vitrum/stained-glass-extensions';
prim.userData.surfaceTextureId = SURFACE_TEXTURE_ID.waterglass; // → 5
```

### `packCameUBO` / `CameSegment` / `CameNode`

Packs H-channel came geometry (segments + nodes) into std140-aligned
`Float32Array` buffers ready for host upload to a shader UBO.

**Reserved:** no vitrum backend currently consumes these arrays directly.
The host application is responsible for uploading them to its own shader.
A future first-class `h-channel-came` backend path may consume this layout
when that feature lands (road-to-100).

```ts
import { packCameUBO } from '@vitrum/stained-glass-extensions';
const { segments, nodes, segmentCount, nodeCount } = packCameUBO(segs, nodes);
// → upload segments / nodes to your own UBO binding
```

## What was removed

`VITRUM_USER_DATA_KEYS` (the Three.js userData bridge string map) was removed
in 2026-06-09 as part of the THREE decouple (commit `e14000c`). These strings
were only needed by the `three-bindings` adapter layer, which is now gone. The
material extension fields they mapped (dispersion, scattering, thin-film, etc.)
are now first-class fields on `@vitrum/core` `MaterialSpec` — no userData
bridge is required.
