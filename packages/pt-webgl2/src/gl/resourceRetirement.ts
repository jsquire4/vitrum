/**
 * Run every resource retirement even when one host/driver cleanup throws.
 * Publication callers may catch the aggregate because a committed replacement
 * must never be rolled back by failure to retire an already-obsolete resource.
 */
export function retireIndependently(
  retirements: readonly (() => void)[],
  message: string,
): void {
  const errors: unknown[] = [];
  for (const retire of retirements) {
    try {
      retire();
    } catch (error) {
      errors.push(error);
    }
  }
  if (errors.length === 1) throw errors[0];
  if (errors.length > 1) throw new AggregateError(errors, message);
}

export function retireTexturesIndependently(
  gl: WebGL2RenderingContext,
  textures: readonly (WebGLTexture | null)[],
  message: string,
): void {
  retireIndependently(
    textures
      .filter((texture): texture is WebGLTexture => texture != null)
      .map((texture) => () => gl.deleteTexture(texture)),
    message,
  );
}
