/**
 * Cross-module DDGI compile-time constants. Lives in its own file so
 * `probeUpdatePass.ts` and the WGSL template-literal files in `./wgsl/`
 * can both import the constants without forming an import cycle (which
 * triggers ESM TDZ errors at runtime — `Cannot access X before
 * initialization`).
 */

/**
 * Per-probe ray budget per DDGI update. Drives irradiance convergence
 * quality — more rays = smoother per-probe Le. Combined with STRIDE=8
 * round-robin scheduling, each probe sees `RAYS_PER_PROBE / STRIDE = 24`
 * rays per frame on average. Injected into the WGSL ray-cast and atlas
 * blend shaders via template-literal interpolation so the CPU-side
 * buffer sizing and the GPU-side loop bounds cannot diverge.
 */
export const RAYS_PER_PROBE = 192;
