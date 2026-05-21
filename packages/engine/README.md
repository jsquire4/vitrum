# @vitrum/engine

`@vitrum/engine` is the host-facing entrypoint that selects and wires a Vitrum backend.

Primary exports:

- `createEngine()` for direct engine construction
- `attachVitrum()` for vanilla canvas lifecycle wiring
- `VitrumCanvas` React wrapper (from `@vitrum/engine/react`)

It depends on `@vitrum/core` contracts and delegates rendering to backend implementations.
