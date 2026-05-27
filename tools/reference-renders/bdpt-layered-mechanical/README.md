# BDPT vs layered-BSDF — mechanical fixtures

Acceptance PNGs for Sprint 10c / 14 (`bdptLayeredAcceptance.gpu.test.ts`).

- `cornell-layered.png` — GPU capture (512×512 quick suite) when promoted from `benchmark:bdpt-layered-refs`.
- `cornell-layered-bdpt.png` — may still be a stub until a successful `?vitrumBdpt=1` capture lands (check file size >50KB).

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
