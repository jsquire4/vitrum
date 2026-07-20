# radiometric-ab diagnostics (ad-hoc)

**Status: ad-hoc / scratch. NOT part of any CI gate.**

These are hand-run debugging harnesses kept for their investigative value, not
proof lanes. Nothing references them; they are not wired into `npm run` scripts,
the behavioral gate, the proof umbrella, or the road-to-100 validation queue.
They were relocated here from `tools/radiometric-ab/` (D17-6 / T3-D17) to keep
the sweep/proof harnesses uncluttered while preserving the debugging code.

Each carries its own drifted scene builders / camera helpers on purpose — do NOT
assume they match `tools/radiometric-ab/walkaround-ab.mjs` or the behavioral gate;
they were snapshots for one-off investigations.

## Scripts

- `walkaround-diag.mjs` — walkaround-hybrid render diagnostic (per-pixel dumps,
  scene/camera variations for eyeballing GI behaviour).
- `walkaround-sun-control.mjs` — directional-sun control sweep for the walkaround
  engine.
- `walkaround-material-check.mjs` — prints the packed BVH roughMetal / isMetal
  texels for a diffuse vs metal scene (imports
  `packages/walkaround-hybrid/src/restir/packingHelpers.ts` directly).

## Running

Run from the repo root; this directory ships its own `deno.json` import map
(paths are one level deeper than the parent `tools/radiometric-ab/deno.json`):

```
deno run --unstable-webgpu --sloppy-imports --allow-read --allow-env --allow-write \
  tools/radiometric-ab/diagnostics/<script>.mjs
```
