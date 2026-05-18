/**
 * Builtin denoiser registration helper.
 *
 * `WalkaroundGPUPipeline` constructs a fresh {@link DenoiserRegistry} at
 * boot, hands it to {@link registerBuiltinDenoisers}, then looks up the
 * active denoiser by ID. Adding a new built-in denoiser is a single new
 * file under `denoisers/` + one `registry.register(...)` line here; no
 * pipeline edits required.
 *
 * Registration order is observable only via {@link DenoiserRegistry.ids}
 * — preserved here in alphabetical-ish "from cheapest to richest" order
 * (none → atrous → atrous-variance → svgf-real → neural → oidn-final) so
 * diagnostic output is stable.
 */

import { AtrousDenoiser } from './atrous.js';
import { AtrousVarianceDenoiser } from './atrousVariance.js';
import { DenoiserRegistry } from './index.js';
import { NeuralDenoiser } from './neural.js';
import { NoneDenoiser } from './none.js';
import { OIDNFinalDenoiser } from './oidnFinal.js';
import { SVGFRealDenoiser } from './svgfReal.js';

export function registerBuiltinDenoisers(registry: DenoiserRegistry): void {
  registry.register(new NoneDenoiser());
  registry.register(new AtrousDenoiser());
  registry.register(new AtrousVarianceDenoiser());
  registry.register(new SVGFRealDenoiser());
  // Disabled placeholders — registered for diagnostic enumeration via
  // DenoiserRegistry.ids(), but `lookup()` throws if a host selects one
  // before the corresponding workstream finishes (W10 / W11).
  registry.register(new NeuralDenoiser());
  registry.register(new OIDNFinalDenoiser());
}
