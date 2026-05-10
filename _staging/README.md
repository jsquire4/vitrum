# `_staging/` — reference policy

The **`legacy-source/`** tree that previously mirrored an old React host has been **removed** (M3 drain). Historical copies remain in **git history** if you need line-level reference.

## Canonical implementations

| Concern | Package / location |
|--------|---------------------|
| WebGL2 path tracing `Engine` | `@vitrum/pt-webgl` |
| WebGPU hybrid GI `Engine` | `@vitrum/walkaround-hybrid` |
| THREE → core `Scene` | `@vitrum/three-bindings` |
| Minimal demos | `examples/cornell-box`, `examples/two-engines-one-scene` (G2) |

This folder may hold small **non-shipped** notes in the future; nothing here is imported by `packages/*`.
