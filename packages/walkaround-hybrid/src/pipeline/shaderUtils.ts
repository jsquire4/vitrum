/**
 * shaderUtils.ts — Shared GPU shader compilation helpers for the
 * walkaround-hybrid pipeline.
 *
 * The `checkShaderCompile` helper centralises the pattern that was previously
 * copy-pasted across `svgfReal.ts`, `atrousVariance.ts`, and `bmfr.ts`:
 *   1. Await `GPUShaderModule.getCompilationInfo()`
 *   2. Filter messages of type `'error'`
 *   3. Log + throw on the first error
 */

/**
 * Checks a compiled `GPUShaderModule` for errors. If any compilation error
 * messages are present, logs them to `console.error` and throws an `Error`
 * naming the first error and its source line.
 *
 * @param sm     The shader module returned by `device.createShaderModule`.
 * @param label  Human-readable label used in the error message (e.g. `'bmfr'`
 *               or `'svgf-reproj'`).
 */
export interface CheckShaderCompileOptions {
  /** Bracketed prefix for the logged/thrown messages. Defaults to `[shader]`.
   *  The ReSTIR pipeline compiler passes `[ReSTIR]` to preserve its historical
   *  log-scraping surface. */
  readonly prefix?: string;
  /** Optional sink invoked with any compilation *warnings* (type === 'warning').
   *  When omitted, warnings are ignored (the pre-extension behavior for the
   *  denoiser callers). */
  readonly onWarnings?: (warnings: readonly GPUCompilationMessage[]) => void;
}

export async function checkShaderCompile(
  sm: GPUShaderModule,
  label: string,
  options: CheckShaderCompileOptions = {},
): Promise<void> {
  const prefix = options.prefix ?? '[shader]';
  const info = await sm.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === 'error');
  if (errors.length > 0) {
    console.error(
      `${prefix} Compile errors in '${label}':`,
      errors.map((e) => `line ${e.lineNum}: ${e.message}`),
    );
    throw new Error(
      `${prefix} Compile error in '${label}': ${errors[0]!.message} (line ${errors[0]!.lineNum})`,
    );
  }
  if (options.onWarnings !== undefined) {
    const warns = info.messages.filter((m) => m.type === 'warning');
    if (warns.length > 0) options.onWarnings(warns);
  }
}
