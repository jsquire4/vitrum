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
import type { OIDNFinalDenoiserOptions } from './oidnFinal.js';
import { SVGFRealDenoiser } from './svgfReal.js';

/**
 * Per-denoiser construction-time configuration. Backends that need
 * non-default construction args expose them via a single config bag
 * here. Today the only entry is `oidn` (the neural denoiser's own
 * weight URL is plumbed through `HybridEngineOptions.neuralWeights`,
 * not the registry, because the neural pipeline runs out-of-band — see
 * `denoisers/neural.ts` JSDoc).
 *
 * When `oidn.modelUrl` is undefined (the default), the OIDN entry
 * registers as a `disabled` placeholder: `DenoiserRegistry.ids()` still
 * lists `'oidn-final'`, but `lookup('oidn-final')` throws the canonical
 * "registered but disabled" error — so a host that selects
 * `denoiser: 'oidn-final'` without supplying a `modelUrl` fails fast at
 * pipeline boot with a clear remediation message.
 */
interface RegisterBuiltinDenoisersOptions {
  /** W11 — OIDN final-pass denoiser config. Forwarded from
   *  `HybridEngineOptions.extensions['walkaround-hybrid'].oidnModelUrl`. */
  readonly oidn?: OIDNFinalDenoiserOptions;
}

export function registerBuiltinDenoisers(
  registry: DenoiserRegistry,
  options?: RegisterBuiltinDenoisersOptions,
): void {
  registry.register(new NoneDenoiser());
  registry.register(new AtrousDenoiser());
  registry.register(new AtrousVarianceDenoiser());
  registry.register(new SVGFRealDenoiser());
  // Diagnostic stub — registered for `DenoiserRegistry.ids()` enumeration
  // and to throw the canonical "registered but disabled" error if a host
  // mistakenly routes through the registry. The actual W10 neural
  // pipeline runs out-of-band through HybridEngineLifecycle (see
  // `denoisers/neural.ts` JSDoc).
  registry.register(new NeuralDenoiser());
  // OIDN: enabled when modelUrl is provided, registered-but-disabled
  // otherwise so callers selecting 'oidn-final' without config fail fast
  // with a clear remediation message.
  registry.register(new OIDNFinalDenoiser(options?.oidn));
}
