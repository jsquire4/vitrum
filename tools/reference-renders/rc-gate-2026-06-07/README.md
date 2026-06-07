# RC acceptance gate — two-scene (2026-06-07)

The official Radiance-Cascades acceptance gate. Supersedes the single W8
`cornell-down-directional` pair: after the 2026-06-07 RC fix landed, RC's light
model is **sun (directSun) + emissive geometry + env + rect-area emitter NEE**
(the emitter NEE is new — `walkaround-rc/probeRayCast.wgsl` `rcEmitterNEE`,
commit `1e893fa`). A single scene cannot exercise RC's full model *and* produce
a strong gate signal: RC is a **GI cache**, so a gate-worthy `rcDeltaMean`
appears only in **indirect-dominant (enclosed)** scenes, but admitting the sun
requires an **open** scene, which is **direct-dominant** → weak RC delta. The
two requirements are physically in tension, so the gate uses **two scenes**, one
per light path.

Backend: dzn (RTX 4090) + the wsl-gpu vitem harness
(`tests/hybrid-vitem-capture.mjs`), 48 frames, 512². The browser
`run-rc-acceptance.mjs` pipeline is SwiftShader-blocked in WSL; re-capture on a
real-GPU host when that path is available.

## Scene 1 — emitter NEE (primary strength gate)

Enclosed Cornell box, single down-facing rect-area ceiling light, no sun. RC's
probe cast NEE-samples the rect emitter; the enclosure makes the scene
indirect-dominant so RC's bounce GI is a large fraction of the image.

- `emitter-nee-rc-off.png` / `emitter-nee-rc-on.png`
- Scene: `cornell` (`HYBRID_CFG` `{"scene":"cornell"}`), `rcWeight:1`.
- **Measured `rcDeltaMean` = 0.0087** (78% of pixels change). **Threshold: > 0.005** (1.7× margin).

## Scene 2 — directSun (liveness gate)

Open-top Cornell (no ceiling), no rect emitter — the only light is the engine
`primaryLight` flooding the floor through the open top. RC's probe cast picks up
the sun-lit floor (directSun term) into the cascade.

- `direct-sun-rc-off.png` / `direct-sun-rc-on.png`
- Scene: `cornell-sun-open` (`HYBRID_CFG`
  `{"scene":"cornell-sun-open","opts":{"primaryLightIntensity":3.0,"primaryLightDir":[0,-1,0]}}`),
  `rcWeight:1`.
- **Measured `rcDeltaMean` = 0.0010**. **Threshold: > 0.0005** (2× margin). This
  is a LIVENESS assertion (RC's sun path is non-zero), intentionally weaker than
  Scene 1: open sun scenes are direct-dominant, so RC's GI is a small fraction of
  the final image. RC's sun path is *separately* validated at full strength by
  the bisect harness (`wsl-gpu/scripts/tlas-zero-gi-bisect.ts --sun=2`: indirect
  tap 0.000356 → 0.0528, ~148×).

## Capture commands

```bash
cd ~/projects/wsl-gpu && source dzn-runtime/env.sh
# Scene 1 (emitter NEE)
HYBRID_CFG='{"scene":"cornell","frames":48,"width":512,"height":512,"opts":{"rcEnabled":false}}' \
  deno run --config capture-worker/deno.json --unstable-webgpu --sloppy-imports -A \
  tests/hybrid-vitem-capture.mjs --backend=dzn --tag=rc-off-rect
HYBRID_CFG='{"scene":"cornell","frames":48,"width":512,"height":512,"opts":{"rcEnabled":true,"rcWeight":1}}' \
  deno run ... --tag=rc-on-rect
# Scene 2 (directSun)
HYBRID_CFG='{"scene":"cornell-sun-open","frames":48,"width":512,"height":512,"opts":{"rcEnabled":false,"primaryLightIntensity":3.0,"primaryLightDir":[0,-1,0]}}' \
  deno run ... --tag=rc-off-sunopen
HYBRID_CFG='{"scene":"cornell-sun-open",...,"opts":{"rcEnabled":true,"rcWeight":1,"primaryLightIntensity":3.0,"primaryLightDir":[0,-1,0]}}' \
  deno run ... --tag=rc-on-sunopen
```

`rcDeltaMean` per pair = mean over pixels of `|lum(rc_on) − lum(rc_off)|`
(Rec.601 luma, normalized 0..1). Gate wiring: `rcAcceptance.gpu.test.ts`.
