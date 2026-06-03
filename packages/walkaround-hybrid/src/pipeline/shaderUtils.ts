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
export async function checkShaderCompile(sm: GPUShaderModule, label: string): Promise<void> {
  const info = await sm.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === 'error');
  if (errors.length > 0) {
    console.error(
      `[shader] Compile errors in '${label}':`,
      errors.map((e) => `line ${e.lineNum}: ${e.message}`),
    );
    throw new Error(
      `[shader] Compile error in '${label}': ${errors[0]!.message} (line ${errors[0]!.lineNum})`,
    );
  }
}
