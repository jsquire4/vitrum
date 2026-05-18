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
 * here. Today the only entry is `oidn` — but PPG / Neural will accept
 * weight URLs through this same surface when their workstreams land
 * (W9 / W10).
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
  // Disabled placeholders — registered for diagnostic enumeration via
  // DenoiserRegistry.ids(), but `lookup()` throws if a host selects one
  // before the corresponding workstream finishes (W10) OR (for OIDN, W11)
  // without supplying construction-time config.
  registry.register(new NeuralDenoiser());
  // W11 — OIDN: enabled when modelUrl is provided, registered-but-disabled otherwise.
  registry.register(new OIDNFinalDenoiser(options?.oidn));
}
