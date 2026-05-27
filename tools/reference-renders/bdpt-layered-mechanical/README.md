# BDPT vs layered-BSDF — mechanical fixtures

64×64 PNG stubs for the Sprint 10c / 14 acceptance harness (`bdptLayeredAcceptance.gpu.test.ts`). They are **not** GPU renders.

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
