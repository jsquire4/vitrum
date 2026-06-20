# pt-material-lobes-behavioral

Committed 64x64 behavioral-gate goldens for `pt/material-lobes`.

The scene contains five lit pt-webgpu full-tier panels that exercise scalar
clearcoat, sheen, iridescence, anisotropy, and dielectric specular material
fields through the normal renderer path. The `*.dzn-full.png` variant is the WSL
dzn full-tier adapter capture used by `npm run behavioral-gate:dzn -- --filter
material-lobes --require-full-tier`.

This is render-health and golden-stability evidence for scalar material lobes.
It does not replace the separate map-heavy material furnace, inverse/adjoint
parity, BDPT specialty-path, or radiometric A/B proof rows.
