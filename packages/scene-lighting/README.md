# @vitrum/scene-lighting

Host-app utility library for deriving lighting-state values from scene parameters.
This is a **host-side helper**, not a render backend. It computes backend-agnostic
lighting snapshots that host applications convert and upload to whichever engine they
are using.

## What it provides

- `computeLightingState` — derive a `LightingState` (sun direction, intensity, sky tint,
  sky irradiance) from a `LightingStateInputs` description; both pt-webgl2 and
  walkaround-hybrid accept `LightingState` inputs.
- `skyParamsFor` / `worldSunPosition` — Preetham sky model: compute `SkyParams` from
  a time-of-day and geographic location.
- `getSunIntensity` / `COLOR_TEMP_HEX` / `SUN_INTENSITY` — directional sun-intensity
  look-up table (color temperature to approximate solar radiance).
- `pointIntensityFromLumens` / `rectAreaIntensityFromLumens` — convert physical lumen
  values to engine-intensity scalars.
- Sun-area-light geometry constants (`sunGeometry`) for the PT-mode area-emitter
  approximation of a directional sun.

## Consumers

The host application or a scene-graph adapter calls these functions to populate
engine inputs. The backends themselves do not import this package — they receive the
already-computed values via their option/update APIs.
