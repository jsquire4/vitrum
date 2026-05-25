# Cornell box (`examples/cornell-box`)

Validates **`@vitrum/core` → `@vitrum/three-bindings` → `@vitrum/pt-webgl`** without the legacy host app.

## Run

```bash
npm install   # from repo root (workspace)
cd examples/cornell-box
npm run dev
```

Browser: **WebGL2** only. Path tracer uses the absorbed `packages/three-gpu-pathtracer` workspace package (see `packages/pt-webgl/README.md`).

## What it does

Builds a small **THREE** Cornell box + **RectArea** ceiling light, converts with **`sceneFromThreeJS`**, **`setScene`** on **`createPTEngine_WebGL2`**, then **`renderFrame`** in a loop until the sample target is reached.

## OIDN post-process — migration note

The current `?vitrumDisplay=oidn` path drives OIDN denoising manually from this example: it reads back the converged accumulator (`readAccumulationRgbFloat`), calls `denoiseFinal` from `@vitrum/shared-denoisers`, and writes the tone-mapped result to a side canvas.

Once `feat/w11-pt-webgl-oidn` lands in main, the **recommended path for new consumers** is to let the engine drive OIDN internally:

```ts
const engine = await createPTEngine_WebGL2({
  device: renderer,
  denoiser: 'oidn-final',
  extensions: {
    'vitrum.ptWebgl.oidnModelUrl': '<ONNX model URL>',
    // ...other extension flags
  },
});
// per frame, after renderFrame(...):
const denoised = engine.getDenoisedFrame(); // Float32Array | null
```

The manual `?vitrumDisplay=oidn` block in `src/main.ts` is annotated `MIGRATION-PENDING` and is kept as the regression baseline until W11 ships. See the comment block in `src/main.ts` for the exact migration steps and the ~26 lines of manual wiring that get deleted.
