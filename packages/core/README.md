# @vitrum/core

Public façade, types, and lifecycle contract for vitrum engines. Backend-agnostic, scene-binding-agnostic.

This is the **load-bearing** type surface every other vitrum package compiles against. Hosts that consume vitrum import from here exactly once.

## Status

**Pre-1.0.** Still in active refactor; breaking changes flagged with `!` in commit messages. Subscribers should pin a specific commit until the v1 freeze (planned after the W1–W13 premium-grade refactor lands).

## Public surface

- `Engine` — the runtime engine contract every backend implements (`renderFrame`, `setScene`, `dispose`, etc.).
- `EngineOptions`, `EngineCapabilities`, `FrameInput`, `FrameOutput`, `FrameStats` — the per-call types.
- `Scene` and primitives — `MeshPrimitive`, `InstancedMeshPrimitive`, `AnalyticPrimitive`, `MaterialSpec`, `SceneEmitter`, `SceneEnvironment`, `Camera`.
- `Mat4`, `Vec3`, `BackendTexture<TBackend>`, `BackendTextureFormat<TBackend>` — branded primitive types.
- `DenoiserConfig` — discriminated union for denoiser selection with per-mode required config.
- `GpuDetection`, `probeWebGPU()` — adapter capability inspection.

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
