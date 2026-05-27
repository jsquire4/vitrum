/**
 * True when the WebGL renderer string indicates SwiftShader / llvmpipe software GL.
 * BDPT RGBA32F light-path binds break unidirectional PT on these stacks.
 */
export function isSoftwareGlRenderer(rendererLabel: string): boolean {
  return /swiftshader|llvmpipe|softpipe/i.test(rendererLabel);
}

/** Browser/capture override via `globalThis.VITRUM_BDPT_FORCE_GPU = '1'`. */
export function bdptForceGpuBind(): boolean {
  const g = globalThis as { VITRUM_BDPT_FORCE_GPU?: string };
  return g.VITRUM_BDPT_FORCE_GPU === '1';
}
