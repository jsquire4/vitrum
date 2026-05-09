# Cornell box (`examples/cornell-box`)

Validates **`@vitrum/core` → `@vitrum/three-bindings` → `@vitrum/pt-webgl`** without the legacy host app.

## Run

```bash
npm install   # from repo root (workspace)
cd examples/cornell-box
npm run dev
```

Browser: **WebGL2** only. Path tracer uses the **sibling** `three-gpu-pathtracer` repo (see `packages/pt-webgl/README.md`).

## What it does

Builds a small **THREE** Cornell box + **RectArea** ceiling light, converts with **`sceneFromThreeJS`**, **`setScene`** on **`createPTEngine_WebGL2`**, then **`renderFrame`** in a loop until the sample target is reached.
