// @vitrum/walkaround-hybrid — WebGPU layered DDGI + RC + ReSTIR DI engine.
//
// Phase 4 Step 1 deliverable: DDGI subsystem extracted. Steps 2–4 land
// ReSTIR pipeline + HybridEngine + RC subsystem.

// DDGI subsystem
export { ProbeUpdatePass } from './ddgi/probeUpdatePass.js';
export type { ProbeUpdatePassOptions } from './ddgi/probeUpdatePass.js';
export { ProbeGrid } from './ddgi/probeGrid.js';
export type { ProbeGridDims, ProbeGridParams } from './ddgi/probeGrid.js';
export * from './ddgi/ddgiAtlasLayout.js';
export { DDGI_SAMPLE_WGSL } from './ddgi/ddgiSampleWgsl.js';
export type { DDGILight } from './ddgi/types.js';

// Shared lib utilities
export { upgradeToNodeMaterial } from './lib/nodeMaterialUpgrade.js';
