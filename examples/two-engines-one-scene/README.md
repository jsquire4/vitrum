# Two engines, one `Scene` (gate G2)

Demonstrates the pattern from `plan/generalized-library-milestones.md` **G2**:

1. Build a `THREE.Scene` (here: shared Cornell box from `@vitrum-examples/shared`).
2. Convert once: `const scene = sceneFromThreeJS(threeScene)`.
3. Pass the **same** `scene` to both `createPTEngine_WebGL2` and `createWalkaroundEngine_Hybrid` via `engine.setScene(scene)`.

**Requirements**

- **Top panel:** WebGL2 + `three-gpu-pathtracer` (path trace).
- **Bottom panel:** WebGPU + `@vitrum/walkaround-hybrid`. If `navigator.gpu` is missing, the WebGPU panel stays blank and the status line explains why.

**Run**

```bash
npm install
npm run dev
```

Use a Chromium build with WebGPU enabled for the lower panel.

**Note:** `HybridEngine` still builds its BVH from the constructor `threeScene` (see `HybridEngine` docstring); hosts must keep that graph in sync with the `Scene` from `sceneFromThreeJS` until the Sprint 1 BVH migration lands.
