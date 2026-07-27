/**
 * Single-source-of-truth for the 157-texel-per-triangle material-atlas decode
 * ABI *offset constants* (complexity-sweep 2026-07-20, T4-2 / D6-1·D16-2).
 *
 * The host packs each triangle's material into a fixed 157-texel meta strip in
 * `baseColorMapMeta` (RGBA32F). Three independent WGSL consumers read that
 * strip and MUST agree on the texel-slot layout byte-for-byte:
 *
 *   1. walkaround-hybrid shade path  — `shaders/materialAtlas.wgsl.ts`
 *      (`MATERIAL_MAP_*`)
 *   2. walkaround-hybrid DDGI probes — `ddgi/wgsl/probeUpdateRays.wgsl.ts`
 *      (`DDGI_MATERIAL_MAP_*`)
 *   3. walkaround-rc probe cast      — `walkaround-rc/src/wgsl/rcMaterialAtlas.wgsl.ts`
 *      (`RC_MATERIAL_MAP_*`)
 *
 * Historically the offset block was hand-cloned into all three files. The
 * VALUES were verified byte-identical across every offset that appears in more
 * than one copy (2026-07-20, zero conflicts); the copies differ ONLY in which
 * *subset* of offsets each declares and in the `*_MATERIAL_MAP_` name prefix.
 *
 * IMPORTANT — semantic-divergence note: the *decode functions* that read this
 * strip (meta-coord math, atlas sampling, tangent frames, normal/bump maps,
 * BRDF gathers, alpha-mask traversal) are NOT byte-identical across the three
 * consumers — they use different bindings (`bvh_normal` vs `rc_geom_normal`
 * vs the DDGI arrays), different meta-coord schemes (fixed
 * `BASE_COLOR_MAP_META_TEX_WIDTH` vs `textureDimensions(...).x`), different
 * atlas-filter policy (shade dispatches nearest/linear; rc+ddgi are
 * nearest-only), and different function sets (rc adds probe-material gathers,
 * ddgi adds BRDF-weight lobes, shade adds volume-scattering + alpha coverage).
 * Unifying those would be a semantic reconciliation requiring a GPU A/B, which
 * is out of scope for this byte-identity dedup. Only the offset ABI — the part
 * that MUST never drift and has no binding dependency — is single-sourced here.
 *
 * This module is binding-free pure data → it is composed as raw WGSL text
 * interpolated into each consumer body. `buildMaterialAtlasOffsetConstsWGSL`
 * emits ONLY the subset a consumer requests, in canonical order, producing the
 * exact byte string that consumer previously hand-wrote (pinned by
 * `probeRayCastByteIdentity.test.ts` for RC and the shade/DDGI composed-WGSL
 * goldens).
 */

/**
 * Canonical ordered list of the material-atlas offset constants. The suffix is
 * the token AFTER the `<prefix>MATERIAL_MAP_` stem (e.g. `SLOT_BASE_COLOR`,
 * `EMISSIVE_TEXEL_OFFSET`). Order mirrors the historical hand-written block
 * (META_TEXELS_PER_TRI, then SLOT_* ascending, then *_TEXEL_OFFSET ascending
 * by texel index) — every consumer declares its subset in this same order.
 */
export const MATERIAL_ATLAS_OFFSETS: ReadonlyArray<readonly [suffix: string, value: number]> = [
  ['META_TEXELS_PER_TRI', 157],
  ['SLOT_BASE_COLOR', 0],
  ['SLOT_ROUGHNESS', 1],
  ['SLOT_METALLIC', 2],
  ['SLOT_AO', 3],
  ['SLOT_ALPHA', 4],
  ['ALPHA_COVERAGE_TEXEL_OFFSET', 10],
  ['EMISSIVE_TEXEL_OFFSET', 11],
  ['TRANSMISSION_TEXEL_OFFSET', 13],
  ['NORMAL_TEXEL_OFFSET', 15],
  ['NORMAL_SCALE_TEXEL_OFFSET', 17],
  ['LIGHT_TEXEL_OFFSET', 18],
  ['LIGHT_INTENSITY_TEXEL_OFFSET', 20],
  ['SPECULAR_TEXEL_OFFSET', 21],
  ['CLEARCOAT_TEXEL_OFFSET', 22],
  ['SHEEN_COLOR_TEXEL_OFFSET', 23],
  ['SPECULAR_COLOR_TEXEL_OFFSET', 24],
  ['SPECULAR_INTENSITY_TEXEL_OFFSET', 26],
  ['CLEARCOAT_FACTOR_TEXEL_OFFSET', 28],
  ['CLEARCOAT_ROUGHNESS_TEXEL_OFFSET', 30],
  ['SHEEN_COLOR_MAP_TEXEL_OFFSET', 32],
  ['SHEEN_ROUGHNESS_TEXEL_OFFSET', 34],
  ['CLEARCOAT_NORMAL_TEXEL_OFFSET', 36],
  ['CLEARCOAT_NORMAL_SCALE_TEXEL_OFFSET', 38],
  ['ANISOTROPY_TEXEL_OFFSET', 39],
  ['ANISOTROPY_SCALAR_TEXEL_OFFSET', 41],
  ['IRIDESCENCE_TEXEL_OFFSET', 42],
  ['IRIDESCENCE_THICKNESS_TEXEL_OFFSET', 44],
  ['IRIDESCENCE_SCALAR_TEXEL_OFFSET', 46],
  ['THICKNESS_TEXEL_OFFSET', 47],
  ['BUMP_TEXEL_OFFSET', 49],
  ['BUMP_SCALE_TEXEL_OFFSET', 51],
  ['ENV_INTENSITY_TEXEL_OFFSET', 52],
  ['FRONT_LAYER_TEXEL_OFFSET', 53],
  ['BACK_LAYER_TEXEL_OFFSET', 54],
  ['VOLUME_SCATTERING_TEXEL_OFFSET', 55],
  ['FRONT_LAYER_NORMAL_TEXEL_OFFSET', 56],
  ['FRONT_LAYER_NORMAL_SCALE_TEXEL_OFFSET', 58],
  ['BACK_LAYER_NORMAL_TEXEL_OFFSET', 59],
  ['BACK_LAYER_NORMAL_SCALE_TEXEL_OFFSET', 61],
  ['OPTICAL_HEADER_TEXEL_OFFSET', 62],
  ['DISPERSION_IOR_RGB_TEXEL_OFFSET', 63],
  ['SPECTRAL_SAMPLES_TEXEL_OFFSET', 64],
  ['THIN_FILM_FRONT_REFLECTANCE_TEXEL_OFFSET', 96],
  ['THIN_FILM_FRONT_TRANSMITTANCE_TEXEL_OFFSET', 104],
  ['THIN_FILM_BACK_REFLECTANCE_TEXEL_OFFSET', 112],
  ['THIN_FILM_BACK_TRANSMITTANCE_TEXEL_OFFSET', 120],
  ['UV_AFFINE_BASE_TEXEL_OFFSET', 128],
  ['SIDE_FLAGS_TEXEL_OFFSET', 156],
];

const OFFSET_VALUE_BY_SUFFIX: ReadonlyMap<string, number> = new Map(MATERIAL_ATLAS_OFFSETS);
const CANONICAL_ORDER: ReadonlyMap<string, number> = new Map(
  MATERIAL_ATLAS_OFFSETS.map(([suffix], i) => [suffix, i]),
);

export interface MaterialAtlasOffsetConstsArgs {
  /**
   * Constant-name prefix. `''` for the shade path (`MATERIAL_MAP_*`), `'DDGI_'`
   * for probe update (`DDGI_MATERIAL_MAP_*`), `'RC_'` for RC
   * (`RC_MATERIAL_MAP_*`).
   */
  readonly prefix: string;
  /**
   * The exact subset of offset SUFFIXES this consumer declares. Emitted in the
   * canonical `MATERIAL_ATLAS_OFFSETS` order regardless of the array order
   * passed. Unknown suffixes throw (guards against a stale/typo drift).
   */
  readonly include: readonly string[];
}

/**
 * Emit the `const <prefix>MATERIAL_MAP_<suffix>: u32 = <value>u;` block for the
 * requested subset, one per line, joined by `\n` (no trailing newline). Byte-
 * identical to the historical hand-written block for each consumer.
 *
 * @throws if `include` names a suffix not present in `MATERIAL_ATLAS_OFFSETS`
 *         (drift guard — a renamed/removed offset fails loudly at compose time).
 */
export function buildMaterialAtlasOffsetConstsWGSL(args: MaterialAtlasOffsetConstsArgs): string {
  const { prefix, include } = args;
  for (const suffix of include) {
    if (!OFFSET_VALUE_BY_SUFFIX.has(suffix)) {
      throw new Error(
        `[materialAtlasOffsets] unknown offset suffix '${suffix}' for prefix '${prefix}MATERIAL_MAP_' ` +
        `(not in MATERIAL_ATLAS_OFFSETS — single-source-of-truth drift)`,
      );
    }
  }
  const ordered = [...include].sort(
    (a, b) => (CANONICAL_ORDER.get(a) ?? 0) - (CANONICAL_ORDER.get(b) ?? 0),
  );
  return ordered
    .map((suffix) => `const ${prefix}MATERIAL_MAP_${suffix}: u32 = ${OFFSET_VALUE_BY_SUFFIX.get(suffix)!}u;`)
    .join('\n');
}
