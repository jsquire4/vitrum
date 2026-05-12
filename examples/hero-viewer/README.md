# hero-viewer — drop-in glTF Viewer

Demonstrates the vitrum drop-in API in the simplest possible form:
drag a `.glb` or `.gltf` file onto the page and vitrum renders it with
either real-time global illumination (walkaround-hybrid, WebGPU) or
progressive path tracing (pt-webgl, WebGL2). Toggle between modes at
the top-right.

## What it demonstrates

- `attachVitrum()` for lifecycle management (RAF loop, resize, visibility pause).
- `loadGltfScene()` for drag-drop file loading.
- `three/examples/jsm/controls/OrbitControls` for camera navigation.
- Engine preference toggle (`prefer: 'realtime' | 'quality'`).

## How to run

```bash
# from repo root
npm install
cd examples/hero-viewer
npm run dev
# open http://localhost:5176
```

Then drag any `.glb` file onto the page.

## Assets

No built-in assets ship with this demo. Drag your own `.glb`, or download a
free CC0 scene from [Sketchfab](https://sketchfab.com/features/free-3d-models)
or the [Khronos glTF samples](https://github.com/KhronosGroup/glTF-Sample-Assets).

**TODO:** ship a small CC0 procedural `.glb` as a default scene so the viewer
has something to show without user action.
