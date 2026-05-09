// Cornell box validation scene — to be authored when @vitrum/pt-webgl reaches
// the point of accepting a Scene and producing a non-throwing renderFrame.
//
// Construction sketch (target shape):
//
// import type { Scene, Mat4, Material } from '@vitrum/core';
//
// const matWhite: Material = { baseColor: [0.73, 0.73, 0.73], roughness: 1.0, metallic: 0.0 };
// const matRed:   Material = { baseColor: [0.65, 0.05, 0.05], roughness: 1.0, metallic: 0.0 };
// const matGreen: Material = { baseColor: [0.12, 0.45, 0.15], roughness: 1.0, metallic: 0.0 };
//
// const scene: Scene = {
//   primitives: [
//     // Six walls of the box, two interior cubes, all triangle meshes.
//   ],
//   emitters: [
//     {
//       kind: 'rect-area',
//       id: 'ceiling-light',
//       position: [0, 1.99, 0],
//       uAxis: [0.65, 0, 0],
//       vAxis: [0, 0, 0.52],
//       color: [1, 1, 1],
//       intensity: 17,  // matches Cornell reference radiance
//     },
//   ],
//   environment: { kind: 'none' },
// };
//
// export default scene;

export {};
