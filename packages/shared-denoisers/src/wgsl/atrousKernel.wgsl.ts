/**
 * Shared B3-spline 5×5 atrous wavelet kernel.
 *
 * Canonical values in both numeric (`ATROUS_KERNEL_VALUES`) and WGSL
 * literal (`ATROUS_KERNEL_WGSL`) forms so the kernel does not drift
 * between atrous.wgsl.ts and atrousVariance.wgsl.ts. Both consumers
 * read from the same `KERNEL_B3SPLINE_5x5` array identifier (C11
 * dedup — pre-C11 we shipped two near-identical strings differing
 * only by the const name).
 */

/**
 * 25 weights in row-major order. Each is `n / 256` so the kernel
 * matches the canonical B3-spline wavelet (Dammertz et al. 2010).
 */
export const ATROUS_KERNEL_VALUES: readonly number[] = [
  1 / 256, 4 / 256, 6 / 256, 4 / 256, 1 / 256,
  4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256,
  6 / 256, 24 / 256, 36 / 256, 24 / 256, 6 / 256,
  4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256,
  1 / 256, 4 / 256, 6 / 256, 4 / 256, 1 / 256,
] as const;

/** Stable WGSL identifier emitted by `ATROUS_KERNEL_WGSL`. */
export const ATROUS_KERNEL_IDENT = 'KERNEL_B3SPLINE_5x5' as const;

function buildKernelWgsl(name: string): string {
  const rows: string[] = [];
  for (let i = 0; i < 5; i++) {
    const row = ATROUS_KERNEL_VALUES.slice(i * 5, i * 5 + 5)
      .map((v) => {
        const n = Math.round(v * 256);
        return ` ${n}.0/256.0`;
      })
      .join(', ');
    rows.push(' ' + row.trim() + ',');
  }
  return `const ${name}: array<f32, 25> = array<f32, 25>(\n${rows.join('\n')}\n);`;
}

/**
 * Canonical 5×5 B3-spline kernel WGSL. Both consumers read the
 * identifier `KERNEL_B3SPLINE_5x5` (post-C11: collapsed from the
 * pre-C11 pair of `KERNEL` + `ATROUS_VARIANCE_KERNEL` strings).
 */
export const ATROUS_KERNEL_WGSL = buildKernelWgsl(ATROUS_KERNEL_IDENT);
