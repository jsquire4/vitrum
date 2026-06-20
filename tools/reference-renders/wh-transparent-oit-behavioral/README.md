# wh-transparent-oit behavioral golden

This fixture backs `tools/behavioral-gate/gate.mjs` label `wh/transparent-oit`.

The scene is a 64x64 walkaround-hybrid Cornell-style behavioral gate with a
fractional `alphaMode:'blend'` pane in front of the box, direct sun, an analytic
point light, and the finite rect-area emitter. The golden proves camera-visible
transparent OIT composition stays stable on the committed dzn full-tier proof
lane.

`wh-transparent-oit.dzn-full.png` is the authoritative dzn-full capture. The
generic `wh-transparent-oit.png` mirrors the same image so the shared golden
lookup never points at a missing baseline while the native WSL lavapipe
walkaround lane remains host-blocked by the known Deno/wgpu-hal GLES panic.

This is not a claim of true transparent ReSTIR/GI/RC/DDGI transport. Those rows
remain approximate until a layered-transport model and reference A/B proof land.
