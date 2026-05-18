/**
 * materialExtensions.ts — pluggable extension-converter registry for the
 * THREE.js ↔ @vitrum/core Material round trip.
 *
 * Background
 * ----------
 * `@vitrum/core` Material has an `extensions: Record<string, unknown>` escape
 * hatch for backend- or host-specific data. The forward (THREE → vitrum) and
 * reverse (vitrum → THREE) converters live in this package, but until this
 * registry existed, the dichroic-LUT stained-glass-studio extension was
 * hardcoded directly into the converters — coupling the generic THREE
 * adapter to a single host app.
 *
 * Design
 * ------
 * A `MaterialExtensionConverter` is a pair of pure functions, one for each
 * direction. Callers pass an array of converters into `convertMaterial` /
 * `vitrumMaterialToThree`; the converters run in registration order. Each
 * converter is responsible for its own userData keys and Material.extensions
 * key — three-bindings never inspects either.
 *
 * Library default: NO converters. Hosts that want a specific extension
 * (e.g. `@vitrum/stained-glass-extensions`'s `dichroicLUTsExtensionConverter`)
 * wire it explicitly at engine construction.
 */

import type * as THREE from 'three';
import type { Material as VitrumMaterial } from '@vitrum/core';

/**
 * A pluggable handler for one Material extension.
 *
 * The `forward` function is invoked during THREE → vitrum conversion: read
 * fields off `threeMat.userData` and return a partial `Material.extensions`
 * map to be merged (or `undefined` to contribute nothing for this material).
 *
 * The `reverse` function is invoked during vitrum → THREE conversion: read
 * the extension off `vitrumMat.extensions` and mutate `threeMat.userData`
 * (or other THREE material fields) accordingly.
 *
 * Implementations MUST be pure with respect to inputs other than the
 * declared mutation target (`threeMat.userData` for reverse) and MUST NOT
 * mutate the input vitrum Material.
 */
export interface MaterialExtensionConverter {
  /**
   * Stable identifier for the extension; used for logging and diagnostics.
   * Recommended convention: dotted reverse-domain or `<host>.<feature>`.
   */
  readonly id: string;

  /**
   * THREE → vitrum direction. Read from `threeMat.userData` (or other
   * THREE material fields) and return a fragment to merge into
   * `Material.extensions`. Return `undefined` to skip.
   */
  forward(threeMat: THREE.Material): Readonly<Record<string, unknown>> | undefined;

  /**
   * vitrum → THREE direction. Read from `vitrumMat.extensions` and stamp
   * `threeMat.userData` (or other fields). No-op when the relevant
   * extension key is absent.
   */
  reverse(vitrumMat: VitrumMaterial, threeMat: THREE.Material): void;
}

/**
 * Options bag accepted by `convertMaterial` and `vitrumMaterialToThree` for
 * passing extension converters. Wrapped in an interface so callers can
 * forward an opaque options object through helper layers without caring
 * about the exact field shape.
 */
export interface MaterialConversionOptions {
  /**
   * Extension converters to run in addition to the built-in RFE handlers.
   * Run in registration order; later converters that write to the same
   * userData key win.
   *
   * Default: empty array (no extensions; library defaults to host-agnostic
   * behavior).
   */
  readonly extensionConverters?: ReadonlyArray<MaterialExtensionConverter>;
}

/**
 * Compose the per-extension forward fragments into a single
 * `Material.extensions` object. Returns `undefined` when no converter
 * contributed (so callers can preserve the "field absent" exactOptional
 * semantics).
 */
export function applyForwardExtensions(
  threeMat: THREE.Material,
  converters: ReadonlyArray<MaterialExtensionConverter> | undefined,
): Readonly<Record<string, unknown>> | undefined {
  if (converters == null || converters.length === 0) return undefined;
  let acc: Record<string, unknown> | undefined;
  for (const c of converters) {
    const fragment = c.forward(threeMat);
    if (fragment == null) continue;
    if (acc == null) acc = {};
    for (const k of Object.keys(fragment)) {
      acc[k] = fragment[k];
    }
  }
  return acc;
}

/**
 * Invoke every converter's reverse direction against a THREE material that
 * has already received its base PBR fields. Converters self-gate on whether
 * the relevant Material.extensions key is present.
 */
export function applyReverseExtensions(
  vitrumMat: VitrumMaterial,
  threeMat: THREE.Material,
  converters: ReadonlyArray<MaterialExtensionConverter> | undefined,
): void {
  if (converters == null || converters.length === 0) return;
  for (const c of converters) {
    c.reverse(vitrumMat, threeMat);
  }
}
