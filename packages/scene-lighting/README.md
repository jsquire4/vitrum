# @vitrum/scene-lighting

Host-side lighting-state primitives shared by every vitrum render backend. Owns the four modules that derive a backend-agnostic lighting snapshot — time-of-day to Preetham `SkyParams`, the directional sun-intensity lookup table, the PT-mode sun-area-light geometry constants, and the unified `LightingState` (sun direction + intensity + sky tint + sky irradiance) that PT and walkaround consume identically.
