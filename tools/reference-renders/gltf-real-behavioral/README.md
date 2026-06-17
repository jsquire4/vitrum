# Real glTF Behavioral Goldens

Committed 64x64 lavapipe reference PNGs for the behavioral gate's public
Khronos real-asset glTF cases:

- `pt/gltf-real-box-textured`
- `pt/gltf-real-draco`
- `pt/gltf-real-meshopt`

The normal gate compares current readback against these PNGs using byte RMSE,
mean absolute error, and max-channel-delta tolerances:

```sh
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
  npm run behavioral-gate -- --filter gltf-real
```

To intentionally recapture after a render-changing landing:

```sh
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
  deno run --unstable-webgpu --sloppy-imports --allow-read --allow-env \
    --allow-net --allow-write=tools/reference-renders/gltf-real-behavioral \
    tools/behavioral-gate/gate.mjs --filter gltf-real --update-goldens
```

