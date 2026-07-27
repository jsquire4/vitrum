// blendFrag — portable running-mean composite fragment shader.
//
// The raw trace target already contains the complete main + resolved-NEE sample.
// This pass performs its sole accumulation update with
// `opacity = 1/(samples+1)`, preserving transparent-background coverage while
// the auxiliary MRT attachments remain last-sample data.

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
