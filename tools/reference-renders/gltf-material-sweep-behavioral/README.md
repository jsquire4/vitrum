# glTF Material Sweep Behavioral Golden

Committed 64x64 lavapipe reference PNG for the behavioral gate's synthetic
material-heavy glTF fixture:

- `pt/gltf-material-sweep`

The normal gate compares current readback against this PNG using byte RMSE,
mean absolute error, and max-channel-delta tolerances:

```sh
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
  npm run behavioral-gate -- --filter gltf-material-sweep
```

To intentionally recapture after a render-changing landing:

```sh
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
  deno run --unstable-webgpu --sloppy-imports --allow-read --allow-env \
    --allow-net --allow-write=tools/reference-renders/gltf-material-sweep-behavioral \
    tools/behavioral-gate/gate.mjs --filter gltf-material-sweep --update-goldens
```

