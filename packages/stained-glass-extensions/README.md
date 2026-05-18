# @vitrum/stained-glass-extensions

Opt-in host-app extensions for the **stained-glass-studio** renderer. Not part
of the generic vitrum library; provided as a reference implementation of the
`EngineOptions.extensions` / `Material.extensions` / `AnalyticShape` extension
points.

The generic vitrum library is intentionally host-agnostic — domain primitives
specific to a particular host application live here so the core contract stays
clean.

## What lives here

### `h-channel-came` analytic shape (D2)

The H-channel came rail is a stained-glass-specific architectural primitive.
The `'h-channel-came'` discriminator and the parameter layout
`[length, railWidth, blockHeight, webThickness]` are exported as a tagged
constant, with a `isHChannelCame` helper for backends that opt into supporting
the shape.

```ts
import {
  STAINED_GLASS_H_CHANNEL_CAME,
  isHChannelCame,
} from '@vitrum/stained-glass-extensions';

// On the producer side:
const primitive = {
  kind: 'analytic',
  id: 'rail-0',
  shape: STAINED_GLASS_H_CHANNEL_CAME,         // = 'h-channel-came'
  params: new Float32Array([1.0, 0.012, 0.018, 0.004]),
  material,
};

// On the backend side, before specializing:
if (isHChannelCame(primitive.shape)) {
  // backend-specific came intersection / packing
}
```

Because `AnalyticShape` in `@vitrum/core` is now an open-ended string union
(`'sphere' | 'box' | 'capsule' | 'cylinder' | (string & {})`), backends can
accept this tag without core needing to know about it.

### Dichroic LUT extension converter (D3)

The stained-glass dichroic body baker emits two 256×1 RGBA-float `DataTexture`s
pre-convolving the TMM × CIE 1931 standard observer. Raster backends bind these
directly; PT backends evaluate TMM in-shader and ignore the LUT.

`@vitrum/three-bindings` exposes an `extensionConverters` registry so the
forward/reverse converters do not hardcode any extension. Wire the dichroic
converter at engine construction:

```ts
import { convertMaterial, vitrumMaterialToThree } from '@vitrum/three-bindings';
import { dichroicLUTsExtensionConverter } from '@vitrum/stained-glass-extensions';

const converters = [dichroicLUTsExtensionConverter];

const vitrumMat = convertMaterial(threeMat, { extensionConverters: converters });
const roundTripped = vitrumMaterialToThree(vitrumMat, { extensionConverters: converters });
```

Without the converter wired, three-bindings makes a clean round trip with no
dichroic-specific behavior — the library defaults to no extensions.

## Provenance

- H-channel came primitive — Phase 6 sprint 5 (vitrum), stained-glass-studio
  came geometry contract.
- Dichroic-LUT pre-convolution — stained-glass dichroic addendum
  (2026-05-12); TMM × CIE 1931 standard observer integration.
