/**
 * giBilinearGather.wgsl.ts — shared 4-corner half-res bilinear GI-reservoir
 * gather scaffold.
 *
 * D8-4 (complexity-sweep 2026-07-20, T4-3): `lo_indirect` and `lo_transmittedGI`
 * in `shadingTerms.wgsl.ts` both read the ReSTIR-GI reservoir with the SAME
 * 4-neighbour half-res bilinear blend. The bilinear-WEIGHT preamble and the
 * inside-loop corner-selection + clamp + skip are byte-identical between them;
 * only the per-corner ACCUMULATION differs (lo_indirect tracks Meff; the
 * transmitted path folds Fresnel×Beer and does not). So the shared portion is
 * the weight setup + corner-select scaffold, emitted here as raw-string
 * fragments interpolated at each call site.
 *
 * These fragments are pure arithmetic over function-scope values (they emit
 * `let`/`var`/`if` statements that reference caller-local `gid`/`dims`/`hx0`/…),
 * so they are binding-free — but they are shared as RAW STRINGS (not a
 * WgslModule) because they are STATEMENT fragments spliced into the MIDDLE of a
 * function body, which the include-graph composer cannot express. Emitting them
 * byte-for-byte identically at each site keeps the composed `shade` WGSL
 * byte-identical.
 */

/**
 * Bilinear-weight preamble: computes `halfDims`, the fractional half-res coord,
 * and the four corner weights `bw00..bw11`. Emitted verbatim at each gather site
 * (the caller supplies `gid`/`dims` in scope and declares its own accumulators).
 */
export function giBilinearWeightsWgsl(): string {
  return /* wgsl */ `  let halfDims = dims / 2u;
  let halfPxF = vec2f(gid) * 0.5;
  let hx0 = u32(floor(halfPxF.x));
  let hy0 = u32(floor(halfPxF.y));
  let fx = halfPxF.x - f32(hx0);
  let fy = halfPxF.y - f32(hy0);
  let bw00 = (1.0 - fx) * (1.0 - fy);
  let bw10 =        fx  * (1.0 - fy);
  let bw01 = (1.0 - fx) *        fy;
  let bw11 =        fx  *        fy;`;
}

/**
 * Inside-loop corner-select scaffold: selects the (hx, hy, bw) for corner `k`,
 * clamps to `halfDims`, and `continue`s on negligible weight. Emitted verbatim
 * inside each `for (var k...)` loop; the caller opens the loop and, after this
 * scaffold, performs its own reservoir load + accumulation.
 */
export function giBilinearCornerSelectWgsl(): string {
  return /* wgsl */ `    var hx = hx0;
    var hy = hy0;
    var bw: f32 = 0.0;
    if      (k == 0u) { hx = hx0;          hy = hy0;          bw = bw00; }
    else if (k == 1u) { hx = hx0 + 1u;     hy = hy0;          bw = bw10; }
    else if (k == 2u) { hx = hx0;          hy = hy0 + 1u;     bw = bw01; }
    else              { hx = hx0 + 1u;     hy = hy0 + 1u;     bw = bw11; }
    if (hx >= halfDims.x) { hx = halfDims.x - 1u; }
    if (hy >= halfDims.y) { hy = halfDims.y - 1u; }
    if (bw < 1e-5) { continue; }`;
}
