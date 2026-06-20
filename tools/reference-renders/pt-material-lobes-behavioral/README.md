# pt-material-lobes-behavioral

Committed 64x64 behavioral-gate goldens for `pt/material-lobes` and
`pt/material-lobe-maps`.

The scalar scene contains five lit pt-webgpu full-tier panels that exercise
clearcoat, sheen, iridescence, anisotropy, and dielectric specular material
fields through the normal renderer path. The mapped scene uses CPU-readable
TextureRef handles for clearcoat factor/roughness/normal, sheen color/roughness,
iridescence factor/thickness, anisotropy, and specular color/intensity maps.
The `*.dzn-full.png` variants are the WSL dzn full-tier adapter captures used by
the corresponding `npm run behavioral-gate:dzn -- --filter ... --require-full-tier`
commands.

This is render-health and golden-stability evidence for material lobe fields and
their map upload/sampling path. It does not replace the separate inverse/adjoint
parity, BDPT specialty-path, or radiometric A/B proof rows.
