/**
 * Pass entries — single source of truth for non-denoiser dispatch order.
 *
 * Each file in this directory is a {@link Pass} implementation. The
 * orchestrator (`WalkaroundGPUPipeline.initialize`) instantiates each
 * pass with its compiled `GPUComputePipeline` (or `GPURenderPipeline` for
 * CompositePass) + any per-pass UBO refs, then registers them with a
 * shared {@link PassRegistry}. Frame-level dispatch becomes a simple
 * `for (const pass of registry.activePasses(opts)) pass.dispatch(ctx)`.
 *
 * Adding a pass requires exactly three edits:
 *   1. Add a `*Pass.ts` file under this directory implementing {@link Pass}.
 *      Declare `passLabels` (the timestamp-query slot labels the pass emits).
 *   2. Add a `register(new YourPass(...))` line in
 *      {@link WalkaroundGPUPipeline.initialize} after the compile step.
 *   3. Add an entry to {@link NON_DENOISER_PASS_ORDER} in `passOrder.ts` at the
 *      correct topological position. This wires the new labels into the
 *      timestamp-query layout ({@link buildPassLayout}) automatically — no edits
 *      to `timestampQueries.ts` are required for non-denoiser passes.
 *
 * No edits to `renderFrame`, `pipelineCompiler`, `bindGroupBuilders`,
 * `bindGroupLayouts`, or `timestampQueries` are required.
 */

export { CheckerboardPrefillPass } from './CheckerboardPrefillPass.js';
export { SampleBudgetPass } from './SampleBudgetPass.js';
export { ReGIRBuildPass } from './ReGIRBuildPass.js';
export { RISPass } from './RISPass.js';
export { TemporalReservoirPass } from './TemporalReservoirPass.js';
export { SpatialReservoirPass } from './SpatialReservoirPass.js';
export { RISGIPass } from './RISGIPass.js';
export { TemporalGIReservoirPass } from './TemporalGIReservoirPass.js';
export { SpatialGIReservoirPass } from './SpatialGIReservoirPass.js';
export { ShadePass } from './ShadePass.js';
export { MotionVectorsPass } from './MotionVectorsPass.js';
export { PPGUpdatePass } from './PPGUpdatePass.js';
export { GTAOPass } from './GTAOPass.js';
export { GTAOUpsamplePass } from './GTAOUpsamplePass.js';
export { DenoiserAdapterPass } from './DenoiserAdapterPass.js';
export { IndirectTemporalAccumPass } from './IndirectTemporalAccumPass.js';
export { AtrousIndirectPass } from './AtrousIndirectPass.js';
export { IndirectCombinePass } from './IndirectCombinePass.js';
export { TransparentOitPass } from './TransparentOitPass.js';
export { VarianceTrackerPass } from './VarianceTrackerPass.js';
export { TemporalAccumPass } from './TemporalAccumPass.js';
export { ResolvePass } from './ResolvePass.js';
export { CompositePass } from './CompositePass.js';
