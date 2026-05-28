// Canonical pure scene-snapshot patch helpers now live in `@vitrum/core` (theme
// T2 dedup). pt-webgpu was the strictest variant and is the superset that was
// lifted, so this module is a thin re-export to preserve the existing internal
// import path. Backend-specific fast-path detection (geometry-refit / TLAS
// rebuild) stays in `../index.ts` — only the snapshot-patch + invariant layer
// is shared.

export { patchEmitterInScene, patchPrimitiveInScene } from '@vitrum/core';
