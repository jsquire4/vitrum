export declare const MATERIAL_PIXELS: number;
export declare const MATERIAL_WRAP_TEXEL_OFFSET: number;
export declare const MATERIAL_WRAP_TEXELS: number;
export declare const MATERIAL_MAP_FIELD_ORDER: readonly string[];

/** UV-set bitmask assignments: bit k set = map k samples uv1 (ATTR_UV1).
 *  Bit assignments are single-sourced in materialStride.js; this declaration
 *  mirrors them so the TypeScript packer can import the table. */
export declare const UV_SET_BIT: Record<string, number>;
