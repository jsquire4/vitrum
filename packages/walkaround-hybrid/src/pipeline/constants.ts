/**
 * Shared numeric constants for the walkaround pipeline.
 *
 * Zero-dependency module — safe to import from anywhere in the package without
 * risk of circular imports.
 */

/**
 * Size of the WalkaroundUBO in bytes (416 = 26 × 16-byte aligned vec4 slots).
 *
 * Layout details: see `uboUpdater.ts` header comment (the full field-by-field
 * breakdown lives there alongside `writeWalkaroundUBO`).
 *
 * The value must stay in sync with:
 *   - `createBuffer({ size: … })` in `createCommonFrameResources.ts`
 *   - `WALKAROUND_UBO_SIZE_BYTES` in `uboUpdater.ts`
 *   - The WGSL `WalkaroundUBO` struct in `shaders/walkaroundUbo.wgsl.ts`
 */
export const WALKAROUND_UBO_SIZE_BYTES = 416;
