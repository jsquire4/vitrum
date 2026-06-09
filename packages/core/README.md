# @vitrum/core

Public façade, types, and lifecycle contract for vitrum engines. Backend-agnostic, scene-binding-agnostic.

This is the **load-bearing** type surface every other vitrum package compiles against. Hosts that consume vitrum import from here exactly once.

## Status

**Pre-1.0.** Still in active refactor; breaking changes flagged with `!` in commit messages. Subscribers should pin a specific commit until the v1 freeze (W1–W13 refactors complete; v1 freeze pending final pt-webgpu fidelity promotion).

## Public surface

- `Engine` — the runtime engine contract every backend implements (`renderFrame`, `setScene`, `dispose`, etc.).
- `EngineOptions`, `EngineCapabilities`, `FrameInput`, `FrameOutput`, `FrameStats` — the per-call types.
- `Scene` and primitives — `MeshPrimitive`, `InstancedMeshPrimitive`, `AnalyticPrimitive`, `SkinnedMeshPrimitive`, `MaterialSpec`, `SceneEmitter`, `SceneEnvironment`.
- `Mat4`, `Vec3`, `BackendTexture<TBackend>`, `BackendTextureFormat<TBackend>` — branded primitive types.
- `FrameOutput` is a discriminated union: `FrameRendered | FrameSkipped` via `kind`.
- `EngineOptions.denoiser` — string union (`'none' | 'atrous' | 'atrous-variance' | 'svgf-real' | 'bmfr' | 'oidn-final' | 'neural'`) selecting the denoiser pipeline at engine creation. Backend-specific construction-time config (e.g. neural weights, OIDN model URL) flows through `extensions`.
- `GpuDetection`, `probeWebGPU()` — adapter capability inspection.
- `solveSkin` / `combineSkinMatrices` / `mat3InverseTranspose` — CPU linear-blend skinning solver (THREE-free; operates on `SkinnedMeshPrimitive`).

## Design principles

1. **The contract is fixed.** Backends, scene bindings, and denoisers all swap underneath; the public types defined here are the load-bearing API.
2. **The host owns lifecycle.** Engine accepts a device handle but does NOT own the device. Engine accepts frame inputs but does NOT own the cadence.
3. **Generalize over time.** `Material.extensions`, `EngineOptions.extensions`, `AnalyticShape` open-ended kind are explicit extension points for downstream packages.

## Entry points

```ts
import type { Engine, Scene, FrameInput, FrameOutput } from '@vitrum/core';
// Construct via @vitrum/engine's createEngine() — not this package directly.
```

This package exposes only types + adapter-probe helpers; concrete engine construction lives in `@vitrum/engine`.
