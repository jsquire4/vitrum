export declare const MATERIAL_PIXELS: number;
export declare const MATERIAL_WRAP_TEXEL_OFFSET: number;
export declare const MATERIAL_WRAP_TEXELS: number;
export declare const MATERIAL_SPECTRAL_REFLECTANCE_TEXEL_OFFSET: number;
export declare const MATERIAL_LAYER_NORMAL_TEXEL_OFFSET: number;
export declare const MATERIAL_LAYER_NORMAL_TEXELS: number;
export declare const MATERIAL_MAP_FIELD_ORDER: readonly string[];

/** UV-set bitmask assignments: bit k set = map k samples uv1 (ATTR_UV1).
 *  Bit assignments are single-sourced in materialStride.js; this declaration
 *  mirrors them so the TypeScript packer can import the table. */
export declare const UV_SET_BIT: Record<string, number>;

/** First texture-transform texel (fork `firstTextureTransformIdx`). */
export declare const MATERIAL_FIRST_TRANSFORM_TEXEL: number;
/** Texels per texture-transform (mat3 packed as 2 rgba texels). */
export declare const MATERIAL_TRANSFORM_TEXELS: number;
/** Per-map texture-transform texel offsets (single source for the packer). */
export declare const MATERIAL_TRANSFORM_TEXEL: Record<string, number>;
/** ao/light/bump map ids + scalars + envMapIntensity + uv-set mask (texels 85/86). */
export declare const MATERIAL_D3_AUX_TEXEL: number;
export declare const MATERIAL_AO_TRANSFORM_TEXEL: number;
export declare const MATERIAL_LIGHTMAP_TRANSFORM_TEXEL: number;
export declare const MATERIAL_BUMP_TRANSFORM_TEXEL: number;
export declare const MATERIAL_ALPHA_TRANSFORM_TEXEL: number;
export declare const MATERIAL_ANISOTROPY_TRANSFORM_TEXEL: number;
export declare const MATERIAL_VOLUME_THICKNESS_TEXEL: number;
export declare const MATERIAL_THICKNESS_TRANSFORM_TEXEL: number;
