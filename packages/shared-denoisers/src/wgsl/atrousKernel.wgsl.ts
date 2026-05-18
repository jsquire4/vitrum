/**
 * Shared B3-spline 5×5 atrous wavelet kernel.
 *
 * Canonical values in both numeric (`ATROUS_KERNEL_VALUES`) and WGSL
 * literal (`ATROUS_KERNEL_WGSL` / `ATROUS_VARIANCE_KERNEL_WGSL`) forms so
 * the kernel does not drift between atrous.wgsl.ts and atrousVariance.wgsl.ts.
 */

/**
 * 25 weights in row-major order. Each is `n / 256` so the kernel
 * matches the canonical B3-spline wavelet (Dammertz et al. 2010).
 */
export const ATROUS_KERNEL_VALUES: readonly number[] = [
  1 / 256,
  4 / 256,
  6 / 256,
  4 / 256,
  1 / 256,
  4 / 256,
  16 / 256,
  24 / 256,
  16 / 256,
  4 / 256,
  6 / 256,
  24 / 256,
  36 / 256,
  24 / 256,
  6 / 256,
  4 / 256,
  16 / 256,
  24 / 256,
  16 / 256,
  4 / 256,
  1 / 256,
  4 / 256,
  6 / 256,
  4 / 256,
  1 / 256,
] as const;

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

/** Atrous-style usage (`KERNEL`). */
export const ATROUS_KERNEL_WGSL = buildKernelWgsl('KERNEL');
/** atrous-variance-pipeline usage (`ATROUS_VARIANCE_KERNEL`). */
export const ATROUS_VARIANCE_KERNEL_WGSL = buildKernelWgsl('ATROUS_VARIANCE_KERNEL');
