# glTF Triangle Topology Behavioral Golden

`pt-gltf-triangle-strip-fan.png` is written and compared by
`tools/behavioral-gate/gate.mjs` for the `pt/gltf-triangle-strip-fan` lane.

The fixture imports one glTF `TRIANGLE_STRIP` primitive and one glTF
`TRIANGLE_FAN` primitive through `loadGltfForEngine()`, asserts their exact
generated triangle-list indices, rejects unexpected topology diagnostics, then
renders the resulting adapter scene on the pt-webgpu behavioral path.
