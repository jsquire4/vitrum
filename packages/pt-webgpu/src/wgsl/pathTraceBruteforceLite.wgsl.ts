import { PT_WEBGPU_COMMON_WGSL } from './common.wgsl.js';
import {
  HAMMERSLEY_WGSL,
  HERO_WAVELENGTH_WGSL,
  LUMINANCE_WGSL,
  OCTAHEDRAL_CORE_WGSL,
} from '@vitrum/shared-samplers';
import { PT_WEBGPU_PATH_TRACE_MATERIAL_LITE_WGSL } from './pathTrace/material.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_INTERSECTION_LITE_WGSL } from './pathTrace/intersectionLite.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_BSDF_WGSL } from './pathTrace/bsdf.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_CONNECT_LITE_WGSL } from './pathTrace/connectLite.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_CAUSTIC_LITE_WGSL } from './pathTrace/causticLite.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_KERNEL_LITE_WGSL } from './pathTrace/kernelLite.wgsl.js';

/**
 * Compatibility-tier path trace shader — merged-mesh BVH only, procedural sky,
 * directional direct light. Fits adapters with ≥8 storage buffers and ≥4
 * storage textures per compute stage (e.g. SwiftShader Vulkan).
 */
export const PT_WEBGPU_TRACE_LITE_WGSL = /* wgsl */ `
${PT_WEBGPU_COMMON_WGSL}
${HAMMERSLEY_WGSL}
${OCTAHEDRAL_CORE_WGSL}
${LUMINANCE_WGSL}
${HERO_WAVELENGTH_WGSL}
${PT_WEBGPU_PATH_TRACE_MATERIAL_LITE_WGSL}
${PT_WEBGPU_PATH_TRACE_INTERSECTION_LITE_WGSL}
${PT_WEBGPU_PATH_TRACE_BSDF_WGSL}
${PT_WEBGPU_PATH_TRACE_CONNECT_LITE_WGSL}
${PT_WEBGPU_PATH_TRACE_CAUSTIC_LITE_WGSL}
${PT_WEBGPU_PATH_TRACE_KERNEL_LITE_WGSL}
`;
