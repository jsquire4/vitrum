// blendFrag — the Regime-2 alpha-composite quad fragment shader (extracted from
// glResources.ts, T3-D / D11-6). Verbatim port of the fork's BlendMaterial.js:31-59
// (GL_FragColor → pc_fragColor). Lerps target1/target2 by `opacity = 1/(samples+1)`
// with alpha-weighted compositing, written into the ping-pong pair. Moved BYTE-
// IDENTICAL — the GlProgram compiles this string unchanged.

export const BLEND_FRAG = `
in vec2 vUv;
uniform float opacity;
uniform sampler2D target1;
uniform sampler2D target2;
void main() {
  vec4 color1 = texture(target1, vUv);
  vec4 color2 = texture(target2, vUv);
  float invOpacity = 1.0 - opacity;
  float totalAlpha = color1.a * invOpacity + color2.a * opacity;
  if (color1.a != 0.0 || color2.a != 0.0) {
    pc_fragColor.rgb = color1.rgb * (invOpacity * color1.a / totalAlpha)
                     + color2.rgb * (opacity * color2.a / totalAlpha);
    pc_fragColor.a = totalAlpha;
  } else {
    pc_fragColor = vec4(0.0);
  }
}
`;
