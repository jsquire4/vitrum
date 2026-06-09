// Ambient type for the copied `.glsl.js` kernel chunks (fork shader/** + render/**
// + the three-mesh-bvh BVH port). These are plain ES modules with NO type
// declarations — each exports one or more `export const NAME = /* glsl */\`...\``
// string constants. Without `allowJs` enabled, `moduleResolution: Bundler` would
// reject the import ("could not find a declaration file"). This wildcard typing
// makes every `.glsl.js` module importable; the composer pulls named members via a
// namespace import cast to `Record<string, string>` (TS wildcard modules cannot
// declare arbitrary named exports, so the cast is the supported pattern).
//
// The chunks are copied byte-for-byte from the fork; we do NOT hand-author per-file
// `.d.ts` siblings (that would drift from the verbatim-copy invariant). The composer
// only ever concatenates these as strings, so `string` is the only contract needed.
declare module '*.glsl.js' {
  const value: string;
  export default value;
}
