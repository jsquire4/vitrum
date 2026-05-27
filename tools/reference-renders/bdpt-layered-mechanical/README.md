# BDPT vs layered-BSDF — mechanical fixtures

Acceptance PNGs for Sprint 10c / 14 (`bdptLayeredAcceptance.gpu.test.ts`).

- `cornell-layered.png` — GPU capture (512×512 quick suite) when promoted from `benchmark:bdpt-layered-refs`.
- `cornell-layered-bdpt.png` — `?vitrumBdpt=1` on layered (on hardware GL, explicit connections run; SwiftShader captures skip the float light-path bind and match unidirectional brightness).
- `cornell-parity-bdpt.png` — `?vitrumBdpt=1` on parity (diffuse reference). Re-capture via `npm run benchmark:bdpt-layered-refs` (uses hardware Chromium when available).

## Regenerate + gate

```bash
npm run benchmark:bdpt-layered-mechanical
```

## Real captures (GPU host)

```bash
npm run benchmark:bdpt-layered-refs
# or: scripts/capture-cornell-suite.sh --only layered
#     scripts/capture-cornell-suite.sh --only layered --bdpt
```

Compare `cornell-layered.png` vs the BDPT variant visually; promote into dated dirs under `tools/reference-renders/` when refreshing baselines.
