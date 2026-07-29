# @vitrum/stained-glass-extensions

Stained-glass-specific renderer contracts and opt-in host-app helpers extracted
from generic vitrum packages.

This package is not a render backend and does not produce GPU commands. Its
root export owns the wire-level surface-texture contract consumed by
`@vitrum/walkaround-hybrid`. The unrelated host-only came UBO ABI is isolated
behind an explicit subpath so it cannot be mistaken for a vitrum renderer
binding.

## Exports

### `SURFACE_TEXTURE_ID`

Wire-level integer enum for the stained-glass surface texture type stored in
`MaterialSpec.extensions.surfaceTextureId`. Values are GPU-packed into BVH
index words and consumed by WGSL surface-texture switch statements in
`@vitrum/walkaround-hybrid`.

**Do not renumber entries** — renumbering silently corrupts rendering. Add new
entries at the next unused integer. The walkaround ingestion path calls
`validateSurfaceTextureId`; a defined non-integer or an id outside `0..7`
throws instead of being silently wrapped through a low-bit mask. An absent
extension retains the wire default, `smooth`.

```ts
import {
  SURFACE_TEXTURE_ID,
  validateSurfaceTextureId,
} from '@vitrum/stained-glass-extensions';

material.extensions = {
  ...material.extensions,
  surfaceTextureId: validateSurfaceTextureId(SURFACE_TEXTURE_ID.waterglass),
};
```

### Host-only subpath: `host-came-ubo`

`packCameUBO`, `CameSegment`, and `CameNode` are available only from
`@vitrum/stained-glass-extensions/host-came-ubo`.

Packs H-channel came geometry (segments + nodes) into std140-aligned
`Float32Array` buffers ready for host upload to a shader UBO.

This is a complete host-owned ABI. No vitrum backend implicitly consumes these
arrays; the host application uploads them to its own shader binding. In
particular, the live core `h-channel-came` analytic-shape contract is not this
ABI: it carries one primitive transform plus
`[length, railWidth, blockHeight, webThickness]`, and has no segment/node UBO
binding. Converting between a segment graph and analytic primitives is host
scene-authoring policy, so this package does not invent that conversion.

```ts
import {
  packCameUBO,
} from '@vitrum/stained-glass-extensions/host-came-ubo';
const { segments, nodes, segmentCount, nodeCount } = packCameUBO(segs, nodes);
// → upload segments / nodes to your own UBO binding
```

The default capacity is 500 segments and 200 nodes. Exceeding either capacity
throws so the host cannot accidentally render incomplete came geometry. A host
that deliberately accepts truncation must opt in explicitly:

```ts
packCameUBO(segs, nodes, {
  maxSegments: 256,
  maxNodes: 128,
  overflow: 'truncate',
});
```

Caps must be positive, allocation-safe integers. Their combined segment and node
authorization cannot exceed the 256 MiB pack-operation budget, and unknown
option keys are rejected instead of being silently ignored. Coordinates must be
finite float32 values; segment endpoints must remain distinct after float32
conversion; rail width, block height, web thickness, and node radius must remain
positive after float32 conversion. Truncation warnings are emitted only after a
complete pack succeeds.

## What was removed

`VITRUM_USER_DATA_KEYS` (the Three.js userData bridge string map) was removed
in 2026-06-09 as part of the THREE decouple (commit `e14000c`). These strings
were only needed by the `three-bindings` adapter layer, which is now gone. The
material extension fields they mapped (dispersion, scattering, thin-film, etc.)
are now first-class fields on `@vitrum/core` `MaterialSpec` — no userData
bridge is required.
