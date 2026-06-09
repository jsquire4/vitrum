// FullscreenQuad — the single-triangle VAO that covers the viewport, plus the trivial
// vertex shader that writes `vUv` (plan/three-removal/02-gl-framework.md §1; analog of
// THREE's FullScreenQuad — fork PhysicalPathTracingMaterial.js:212-225).
//
// We use the attribute-less "big triangle" trick: 3 vertices generated from gl_VertexID
// cover the whole [-1,1] clip square with no vertex buffer. The fragment kernel then runs
// once per pixel. `vUv` is the [0,1] screen UV the fork kernels read.

/**
 * The pass-through vertex shader BODY (no `#version`/preamble — GlProgram prepends those).
 * Generates a fullscreen triangle from gl_VertexID and emits `vUv` in [0,1].
 *   vertex 0 → (-1,-1) uv(0,0)   vertex 1 → (3,-1) uv(2,0)   vertex 2 → (-1,3) uv(0,2)
 */
export const FULLSCREEN_VERT = `
out vec2 vUv;
void main() {
  vec2 uv = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  vUv = uv;
  gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
`;

export class FullscreenQuad {
  readonly #vao: WebGLVertexArrayObject;

  constructor(gl: WebGL2RenderingContext) {
    const vao = gl.createVertexArray();
    if (vao == null) throw new Error('pt-webgl2: failed to create fullscreen-quad VAO');
    this.#vao = vao;
    // No vertex buffer needed — the vertex shader synthesises positions from gl_VertexID.
    // We still bind an (empty) VAO so the draw has a defined attribute state.
  }

  /** Draw the fullscreen triangle (3 vertices, no index/attribute buffers). */
  draw(gl: WebGL2RenderingContext): void {
    gl.bindVertexArray(this.#vao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
  }

  dispose(gl: WebGL2RenderingContext): void {
    gl.deleteVertexArray(this.#vao);
  }
}
