// adamUbo.ts — shared packer for the 48-byte AdamParams UBO consumed by
// ADAM_WGSL (fusedMlpTrainer.ts). One layout, two producers (D7.8): the fused
// MLP trainer's weight/bias Adam passes and the hash-grid TABLE Adam in
// HashGridTableTrainer both write the identical struct:
//
//   struct AdamParams {
//     count : u32, _p0 : u32, _p1 : u32, _p2 : u32,
//     lr : f32, beta1 : f32, beta2 : f32, eps : f32,
//     bc1 : f32, bc2 : f32, _p3 : f32, _p4 : f32,
//   }
//
// Both callers use the standard hyperparameters β₁=0.9, β₂=0.999, ε=1e-8
// (Kingma & Ba 2015 defaults; also what Müller 2021/2022 use) and differ only
// in count / lr / bias-correction terms, so those are the parameters here. The
// padding words are zero (fresh ArrayBuffer), byte-identical to the previous
// inline packers at both sites.

/** Pack the AdamParams UBO bytes: `count` params, learning rate `lr`, and the
 *  step-dependent bias-correction terms `bc1 = 1-β₁^t`, `bc2 = 1-β₂^t`. */
export function packAdamUbo(count: number, lr: number, bc1: number, bc2: number): ArrayBuffer {
  const ab = new ArrayBuffer(48);
  new Uint32Array(ab, 0, 1)[0] = count;
  const f = new Float32Array(ab);
  f[4] = lr; f[5] = 0.9; f[6] = 0.999; f[7] = 1e-8; f[8] = bc1; f[9] = bc2;
  return ab;
}
