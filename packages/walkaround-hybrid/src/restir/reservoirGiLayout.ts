/**
 * Back-compat re-export shim.
 *
 * The ReSTIR-GI reservoir layout authority moved to `../gi/giLayout.ts`
 * (I3-1: `restir/` was a de-facto shared-foundation sink — the ppg / shader /
 * pipeline / GI-snapshot consumers reached through this restir/ module). The
 * symbols are re-exported here so any lingering `from '../restir/reservoirGiLayout.js'`
 * import keeps resolving. New code should import from `gi/giLayout.ts`.
 */

export {
  RESERVOIR_GI_BASE_STRIDE_U32,
  RESERVOIR_GI_GRIS_STRIDE_U32,
  RESERVOIR_GI_BASE_STRIDE_BYTES,
  RESERVOIR_GI_GRIS_STRIDE_BYTES,
  reservoirGiStrideU32ForRestirPtReuse,
  reservoirGiStrideBytesForRestirPtReuse,
} from '../gi/giLayout.js';
