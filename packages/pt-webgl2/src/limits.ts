/**
 * Hard limits shared by option validation and the generated GLSL. WebGL2
 * drivers must be able to prove every shader loop terminates at compile time;
 * keep these values finite and interpolate them into the trace shader.
 */
export const WEBGL2_MAX_BOUNCES = 32;

/**
 * Professional default for an unconstrained engine. Hosts can still opt into
 * the 32-bounce ceiling, but ordinary programs compile a materially smaller
 * static loop and rely on Russian roulette for deep-path termination.
 */
export const WEBGL2_DEFAULT_BOUNCES = 8;

/**
 * A transparent/volume boundary may consume one traversal without consuming a
 * scattering bounce. The frame packer grants at most one such traversal per
 * bounce, so two steps per bounce is the exact hard upper bound.
 */
export const WEBGL2_MAX_PATH_STEPS = WEBGL2_MAX_BOUNCES * 2;
