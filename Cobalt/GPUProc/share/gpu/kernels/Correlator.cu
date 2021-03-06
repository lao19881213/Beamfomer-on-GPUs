//# Correlator.cu
//#
//# Copyright (C) 2012-2013  ASTRON (Netherlands Institute for Radio Astronomy)
//# P.O. Box 2, 7990 AA Dwingeloo, The Netherlands
//#
//# This file is part of the LOFAR software suite.
//# The LOFAR software suite is free software: you can redistribute it and/or
//# modify it under the terms of the GNU General Public License as published
//# by the Free Software Foundation, either version 3 of the License, or
//# (at your option) any later version.
//#
//# The LOFAR software suite is distributed in the hope that it will be useful,
//# but WITHOUT ANY WARRANTY; without even the implied warranty of
//# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//# GNU General Public License for more details.
//#
//# You should have received a copy of the GNU General Public License along
//# with the LOFAR software suite. If not, see <http://www.gnu.org/licenses/>.
//#
//# $Id: Correlator.cu 26790 2013-10-01 09:00:26Z mol $

// \file
// This file contains a CUDA implementation of the GPU kernel for the
// correlator. It computes correlations between all pairs of stations
// (baselines) and X,Y polarizations, including auto-correlations.

#include "gpu_math.cuh"

#define NR_BASELINES     (NR_STATIONS * (NR_STATIONS + 1) / 2)

#if NR_STATIONS == 288
#  if defined NVIDIA_CUDA
#    define BLOCK_SIZE	 8
#  elif NR_SAMPLES_PER_CHANNEL % 6 == 0
#    define BLOCK_SIZE	 6
#  else
#    define BLOCK_SIZE	 4
#  endif
#elif NR_SAMPLES_PER_CHANNEL % 24 == 0
#  define BLOCK_SIZE	 24
#else
#  define BLOCK_SIZE	 16
#endif

typedef float2 fcomplex;
typedef float4 fcomplex2;
/* typedef LOFAR::Cobalt::gpu::complex<float> fcomplex; */
/* typedef fcomplex fcomplex2[2]; */
/* typedef fcomplex fcomplex4[4]; */

/* typedef __global fcomplex2 (*CorrectedDataType)[NR_STATIONS][NR_CHANNELS][NR_SAMPLES_PER_CHANNEL]; */
/* typedef __global fcomplex4 (*VisibilitiesType)[NR_BASELINES][NR_CHANNELS]; */

typedef fcomplex2 (*CorrectedDataType)[NR_STATIONS][NR_CHANNELS][NR_SAMPLES_PER_CHANNEL];
typedef fcomplex (*VisibilitiesType)[NR_BASELINES][NR_CHANNELS][NR_POLARIZATIONS][NR_POLARIZATIONS];

extern "C" {

/*!
 * Computes correlations between all pairs of stations (baselines) and X,Y
 * polarizations. Also computes all station (and pol) auto-correlations.
 *
 * We consider the output space shaped as a triangle of S*(S-1)/2 full
 * correlations, plus S auto-correlations at the hypothenuse (S = NR_STATIONS).
 * This correlator consists of various versions, correlate_NxN, that differ in
 * used register block size. We have 1x1 (this kernel), 2x2, 3x3, and 4x4.
 * Measure, then select the fastest for your platform.
 *
 * Beyond dozens of antenna fields (exact number depends on observation,
 * software and hardware parameters), our kernels in NewCorrelator.cl are
 * significantly faster than these correlator kernels.
 *
 * \param[out] visibilitiesPtr         2D output array of visibilities. Each visibility contains the 4 polarization pairs, XX, XY, YX, YY, each of complex float type.
 * \param[in]  correctedDataPtr        3D input array of samples. Each sample contains the 2 polarizations X, Y, each of complex float type.
 *
 * Pre-processor input symbols (some are tied to the execution configuration)
 * Symbol                  | Valid Values            | Description
 * ----------------------- | ----------------------- | -----------
 * NR_STATIONS             | >= 1                    | number of antenna fields
 * NR_SAMPLES_PER_CHANNEL  | multiple of BLOCK_SIZE  | number of input samples per channel
 * NR_CHANNELS             | >= 1                    | number of frequency channels per subband
 * Note that for > 1 channels, NR_CHANNELS-1 channels are actually processed,
 * because the second PPF has "corrupted" channel 0. (An inverse PPF can disambiguate.) \n
 * Note that if NR_CHANNELS is low (esp. 1), these kernels perform poorly.
 * Note that this kernel assumes (but does not use) NR_POLARIZATIONS == 2.
 *
 * Execution configuration:
 * - Work dim == 2  (can be 1 iff NR_CHANNELS <= 2)
 *     + Inner dim: the NxN baseline(s) the thread processes
 *     + Outer dim: the channel the thread processes
 * - Work group size: (no restrictions (but processes BLOCK_SIZE * NR_STATIONS), 1) \n
 *   Each work group loads samples from all stations to do the NxN set of correlations
 *   for one of the channels. Some threads in _NxN kernels do not write off-edge output.
 * - Global size: (>= NR_BASELINES and a multiple of work group size, number of actually processed channels)
 */

__global__ void correlate(void *visibilitiesPtr, const void *correctedDataPtr) 
{
  VisibilitiesType visibilities = (VisibilitiesType) visibilitiesPtr;
  CorrectedDataType correctedData = (CorrectedDataType) correctedDataPtr;

  __shared__ float samples[4][BLOCK_SIZE][NR_STATIONS | 1]; // avoid power-of-2

  uint baseline = blockIdx.x * blockDim.x + threadIdx.x;

#if NR_CHANNELS == 1
  uint channel = blockIdx.y;
#else
  uint channel = blockIdx.y + 1;
#endif

  /*
   * if 
   *   b = baseline
   *   x = stat1
   *   y = stat2
   *   x <= y
   * then
   *   b_xy = y * (y + 1) / 2 + x
   * let
   *   u := b_0y
   * then
   *     u            = y * (y + 1) / 2
   *     8u           = 4y^2 + 4y
   *     8u + 1       = 4y^2 + 4y + 1 = (2y + 1)^2
   *     sqrt(8u + 1) = 2y + 1
   *                y = (sqrt(8u + 1) - 1) / 2
   *
   * Let us define
   *   y'(b) = (sqrt(8b + 1) - 1) / 2
   * which increases monotonically and is a continuation of y(b).
   *
   * Because y simply increases by 1 when b increases enough, we
   * can just take the floor function to obtain the discrete y(b):
   *   y(b) = floor(y'(b))
   *        = floor(sqrt(8b + 1) - 1) / 2)
   */

  uint stat_A = __float2uint_rz(sqrtf(float(8 * baseline + 1)) - 0.99999f) / 2;

  /*
   * And, of course
   *  x = b - y * (y + 1)/2
   */
  uint stat_0 = baseline - stat_A * (stat_A + 1) / 2;

  // visR and visI will contain the real and imaginary parts, respectively, of
  // the four visibilities (i.e., the four correlation products between the two
  // antennae A and B with polarizations x and y):
  //   { Ax * Bx, Ax * By, Ay * Bx, Ay * By }
  // where * denotes complex multiplication of A with the conjugate of B.

  float4 visR = {0, 0, 0, 0};
  float4 visI = {0, 0, 0, 0};

  for (uint major = 0; major < NR_SAMPLES_PER_CHANNEL; major += BLOCK_SIZE) {
    // load data into local memory
    for (uint i = threadIdx.x; i < BLOCK_SIZE * NR_STATIONS; i += blockDim.x)
      {
        uint time = i % BLOCK_SIZE;
        uint stat = i / BLOCK_SIZE;

        fcomplex2 sample = (*correctedData)[stat][channel][major + time];

        samples[0][time][stat] = sample.x;
        samples[1][time][stat] = sample.y;
        samples[2][time][stat] = sample.z;
        samples[3][time][stat] = sample.w;
      }

    __syncthreads();

    // compute correlations
    if (baseline < NR_BASELINES) {
      for (uint time = 0; time < BLOCK_SIZE; time++) {
        fcomplex2 sample_1, sample_A;
        sample_1.x = samples[0][time][stat_0]; // sample1_X_r
        sample_1.y = samples[1][time][stat_0]; // sample1_X_i
        sample_1.z = samples[2][time][stat_0]; // sample1_Y_r
        sample_1.w = samples[3][time][stat_0]; // sample1_Y_i
        sample_A.x = samples[0][time][stat_A]; // sampleA_X_r
        sample_A.y = samples[1][time][stat_A]; // sampleA_X_i
        sample_A.z = samples[2][time][stat_A]; // sampleA_Y_r
        sample_A.w = samples[3][time][stat_A]; // sampleA_Y_i

        // Interleave calculation of the two parts of the real and imaginary
        // visibilities to improve performance.
        visR += SWIZZLE(sample_1,x,x,z,z) * SWIZZLE(sample_A,x,z,x,z); 
        visI += SWIZZLE(sample_1,y,y,w,w) * SWIZZLE(sample_A,x,z,x,z);
        visR += SWIZZLE(sample_1,y,y,w,w) * SWIZZLE(sample_A,y,w,y,w);
        visI -= SWIZZLE(sample_1,x,x,z,z) * SWIZZLE(sample_A,y,w,y,w);
      }
    }

    __syncthreads();
  }

  // write visibilities
  if (baseline < NR_BASELINES) {
    (*visibilities)[baseline][channel][0][0] = make_float2(visR.x, visI.x);
    (*visibilities)[baseline][channel][0][1] = make_float2(visR.y, visI.y);
    (*visibilities)[baseline][channel][1][0] = make_float2(visR.z, visI.z);
    (*visibilities)[baseline][channel][1][1] = make_float2(visR.w, visI.w);
  }
}


/*!
 * See the correlate() kernel.
 */
/* __kernel void correlate_2x2(__global void *visibilitiesPtr, */
/*                             __global const void *correctedDataPtr */
/*                             ) */
__global__ void correlate_2x2(void *visibilitiesPtr, const void *correctedDataPtr)
{
  VisibilitiesType visibilities = (VisibilitiesType) visibilitiesPtr;
  CorrectedDataType correctedData = (CorrectedDataType) correctedDataPtr;

  /* __local fcomplex2 samples[2][BLOCK_SIZE][(NR_STATIONS + 1) / 2 | 1]; // avoid power-of-2 */
  __shared__ fcomplex2 samples[2][BLOCK_SIZE][(NR_STATIONS + 1) / 2 | 1]; // avoid power-of-2

  /* uint block = get_global_id(0); */
  uint block =  blockIdx.x * blockDim.x + threadIdx.x;
  /* uint channel = get_global_id(1) + 1; */
#if NR_CHANNELS == 1
  uint channel = blockIdx.y;
#else
  uint channel = blockIdx.y + 1;
#endif

  /* uint x = convert_uint_rtz(sqrt(convert_float(8 * block + 1)) - 0.99999f) / 2; */
  uint x = __float2uint_rz(sqrtf(float(8 * block + 1)) - 0.99999f) / 2;
  uint y = block - x * (x + 1) / 2;

  uint stat_A = 2 * x;

  bool compute_correlations = stat_A < NR_STATIONS;

  /* float4 vis_0A_r = (float4) 0, vis_0A_i = (float4) 0; */
  /* float4 vis_0B_r = (float4) 0, vis_0B_i = (float4) 0; */
  /* float4 vis_1A_r = (float4) 0, vis_1A_i = (float4) 0; */
  /* float4 vis_1B_r = (float4) 0, vis_1B_i = (float4) 0; */
  float4 vis_0A_r = {0, 0, 0, 0}, vis_0A_i = {0, 0, 0, 0};
  float4 vis_0B_r = {0, 0, 0, 0}, vis_0B_i = {0, 0, 0, 0};
  float4 vis_1A_r = {0, 0, 0, 0}, vis_1A_i = {0, 0, 0, 0};
  float4 vis_1B_r = {0, 0, 0, 0}, vis_1B_i = {0, 0, 0, 0};

  for (uint major = 0; major < NR_SAMPLES_PER_CHANNEL; major += BLOCK_SIZE) {
    // load data into local memory
#pragma unroll 1
    /* for (uint i = get_local_id(0); i < BLOCK_SIZE * NR_STATIONS; i += get_local_size(0)) { */
    for (uint i = threadIdx.x; i < BLOCK_SIZE * NR_STATIONS; i += blockDim.x) {
      uint time = i % BLOCK_SIZE;
      uint stat = i / BLOCK_SIZE;

      samples[stat & 1][time][stat / 2] = (*correctedData)[stat][channel][major + time];
    }

    /* barrier(CLK_LOCAL_MEM_FENCE); */
    __syncthreads();

    if (compute_correlations) {
      for (uint time = 0; time < BLOCK_SIZE; time++) {
        float4 sample_0 = samples[0][time][y];
        float4 sample_A = samples[0][time][x];
        float4 sample_B = samples[1][time][x];
        float4 sample_1 = samples[1][time][y];

        /* vis_0A_r += sample_0.xxzz * sample_A.xzxz; */
        /* vis_0A_i += sample_0.yyww * sample_A.xzxz; */
        /* vis_0B_r += sample_0.xxzz * sample_B.xzxz; */
        /* vis_0B_i += sample_0.yyww * sample_B.xzxz; */
        /* vis_1A_r += sample_1.xxzz * sample_A.xzxz; */
        /* vis_1A_i += sample_1.yyww * sample_A.xzxz; */
        /* vis_1B_r += sample_1.xxzz * sample_B.xzxz; */
        /* vis_1B_i += sample_1.yyww * sample_B.xzxz; */
        vis_0A_r += SWIZZLE(sample_0,x,x,z,z) * SWIZZLE(sample_A,x,z,x,z);
        vis_0A_i += SWIZZLE(sample_0,y,y,w,w) * SWIZZLE(sample_A,x,z,x,z);
        vis_0B_r += SWIZZLE(sample_0,x,x,z,z) * SWIZZLE(sample_B,x,z,x,z);
        vis_0B_i += SWIZZLE(sample_0,y,y,w,w) * SWIZZLE(sample_B,x,z,x,z);
        vis_1A_r += SWIZZLE(sample_1,x,x,z,z) * SWIZZLE(sample_A,x,z,x,z);
        vis_1A_i += SWIZZLE(sample_1,y,y,w,w) * SWIZZLE(sample_A,x,z,x,z);
        vis_1B_r += SWIZZLE(sample_1,x,x,z,z) * SWIZZLE(sample_B,x,z,x,z);
        vis_1B_i += SWIZZLE(sample_1,y,y,w,w) * SWIZZLE(sample_B,x,z,x,z);

        /* vis_0A_r += sample_0.yyww * sample_A.ywyw; */
        /* vis_0A_i -= sample_0.xxzz * sample_A.ywyw; */
        /* vis_0B_r += sample_0.yyww * sample_B.ywyw; */
        /* vis_0B_i -= sample_0.xxzz * sample_B.ywyw; */
        /* vis_1A_r += sample_1.yyww * sample_A.ywyw; */
        /* vis_1A_i -= sample_1.xxzz * sample_A.ywyw; */
        /* vis_1B_r += sample_1.yyww * sample_B.ywyw; */
        /* vis_1B_i -= sample_1.xxzz * sample_B.ywyw; */
        vis_0A_r += SWIZZLE(sample_0,y,y,w,w) * SWIZZLE(sample_A,y,w,y,w);
        vis_0A_i -= SWIZZLE(sample_0,x,x,z,z) * SWIZZLE(sample_A,y,w,y,w);
        vis_0B_r += SWIZZLE(sample_0,y,y,w,w) * SWIZZLE(sample_B,y,w,y,w);
        vis_0B_i -= SWIZZLE(sample_0,x,x,z,z) * SWIZZLE(sample_B,y,w,y,w);
        vis_1A_r += SWIZZLE(sample_1,y,y,w,w) * SWIZZLE(sample_A,y,w,y,w);
        vis_1A_i -= SWIZZLE(sample_1,x,x,z,z) * SWIZZLE(sample_A,y,w,y,w);
        vis_1B_r += SWIZZLE(sample_1,y,y,w,w) * SWIZZLE(sample_B,y,w,y,w);
        vis_1B_i -= SWIZZLE(sample_1,x,x,z,z) * SWIZZLE(sample_B,y,w,y,w);
      }
    }

    /* barrier(CLK_LOCAL_MEM_FENCE); */
    __syncthreads();
  }

  // write visibilities
  uint stat_0 = 2 * y;
  uint stat_1 = stat_0 + 1;
  uint stat_B = stat_A + 1;
  bool do_baseline_0A = stat_A < NR_STATIONS;
  bool do_baseline_0B = stat_B < NR_STATIONS;
  bool do_baseline_1A = do_baseline_0A && stat_1 <= stat_A;
  bool do_baseline_1B = do_baseline_0B;

  if (do_baseline_0A) {
    uint baseline = (stat_A * (stat_A + 1) / 2) + stat_0;
    /* (*visibilities)[baseline][channel] = (fcomplex4) { vis_0A_r.x, vis_0A_i.x, vis_0A_r.y, vis_0A_i.y, vis_0A_r.z, vis_0A_i.z, vis_0A_r.w, vis_0A_i.w }; */
    (*visibilities)[baseline][channel][0][0] = make_float2(vis_0A_r.x, vis_0A_i.x);
    (*visibilities)[baseline][channel][0][1] = make_float2(vis_0A_r.y, vis_0A_i.y);
    (*visibilities)[baseline][channel][1][0] = make_float2(vis_0A_r.z, vis_0A_i.z);
    (*visibilities)[baseline][channel][1][1] = make_float2(vis_0A_r.w, vis_0A_i.w);
  }

  if (do_baseline_0B) {
    uint baseline = (stat_B * (stat_B + 1) / 2) + stat_0;
    /* (*visibilities)[baseline][channel] = (fcomplex4) { vis_0B_r.x, vis_0B_i.x, vis_0B_r.y, vis_0B_i.y, vis_0B_r.z, vis_0B_i.z, vis_0B_r.w, vis_0B_i.w }; */
    (*visibilities)[baseline][channel][0][0] = make_float2(vis_0B_r.x, vis_0B_i.x);
    (*visibilities)[baseline][channel][0][1] = make_float2(vis_0B_r.y, vis_0B_i.y);
    (*visibilities)[baseline][channel][1][0] = make_float2(vis_0B_r.z, vis_0B_i.z);
    (*visibilities)[baseline][channel][1][1] = make_float2(vis_0B_r.w, vis_0B_i.w);
  }

  if (do_baseline_1A) {
    uint baseline = (stat_A * (stat_A + 1) / 2) + stat_1;
    /* (*visibilities)[baseline][channel] = (fcomplex4) { vis_1A_r.x, vis_1A_i.x, vis_1A_r.y, vis_1A_i.y, vis_1A_r.z, vis_1A_i.z, vis_1A_r.w, vis_1A_i.w }; */
    (*visibilities)[baseline][channel][0][0] = make_float2(vis_1A_r.x, vis_1A_i.x);
    (*visibilities)[baseline][channel][0][1] = make_float2(vis_1A_r.y, vis_1A_i.y);
    (*visibilities)[baseline][channel][1][0] = make_float2(vis_1A_r.z, vis_1A_i.z);
    (*visibilities)[baseline][channel][1][1] = make_float2(vis_1A_r.w, vis_1A_i.w);
  }

  if (do_baseline_1B) {
    uint baseline = (stat_B * (stat_B + 1) / 2) + stat_1;
    /* (*visibilities)[baseline][channel] = (fcomplex4) { vis_1B_r.x, vis_1B_i.x, vis_1B_r.y, vis_1B_i.y, vis_1B_r.z, vis_1B_i.z, vis_1B_r.w, vis_1B_i.w }; */
    (*visibilities)[baseline][channel][0][0] = make_float2(vis_1B_r.x, vis_1B_i.x);
    (*visibilities)[baseline][channel][0][1] = make_float2(vis_1B_r.y, vis_1B_i.y);
    (*visibilities)[baseline][channel][1][0] = make_float2(vis_1B_r.z, vis_1B_i.z);
    (*visibilities)[baseline][channel][1][1] = make_float2(vis_1B_r.w, vis_1B_i.w);
  }
}


/*!
 * See the correlate() kernel.
 */
/* __kernel void correlate_3x3(__global void *visibilitiesPtr, */
/*                             __global const void *correctedDataPtr */
/*                             ) */
__global__ void correlate_3x3(void *visibilitiesPtr, const void *correctedDataPtr)
{
  VisibilitiesType visibilities = (VisibilitiesType) visibilitiesPtr;
  CorrectedDataType correctedData = (CorrectedDataType) correctedDataPtr;

  __shared__ fcomplex2 samples[3][BLOCK_SIZE][(NR_STATIONS + 2) / 3 | 1]; // avoid power-of-2

  /* uint block = get_global_id(0); */
  uint block = blockIdx.x * blockDim.x + threadIdx.x;
  /* uint channel = get_global_id(1) + 1; */
#if NR_CHANNELS == 1
  uint channel = blockIdx.y;
#else
  uint channel = blockIdx.y + 1;
#endif

  /* uint x = convert_uint_rtz(sqrt(convert_float(8 * block + 1)) - 0.99999f) / 2; */
  uint x = __float2uint_rz(sqrtf(float(8 * block + 1)) - 0.99999f) / 2;
  uint y = block - x * (x + 1) / 2;

  uint stat_A = 3 * x;

  bool compute_correlations = stat_A < NR_STATIONS;

  float4 vis_0A_r = {0, 0, 0, 0}, vis_0A_i = {0, 0, 0, 0};
  float4 vis_0B_r = {0, 0, 0, 0}, vis_0B_i = {0, 0, 0, 0};
  float4 vis_0C_r = {0, 0, 0, 0}, vis_0C_i = {0, 0, 0, 0};
  float4 vis_1A_r = {0, 0, 0, 0}, vis_1A_i = {0, 0, 0, 0};
  float4 vis_1B_r = {0, 0, 0, 0}, vis_1B_i = {0, 0, 0, 0};
  float4 vis_1C_r = {0, 0, 0, 0}, vis_1C_i = {0, 0, 0, 0};
  float4 vis_2A_r = {0, 0, 0, 0}, vis_2A_i = {0, 0, 0, 0};
  float4 vis_2B_r = {0, 0, 0, 0}, vis_2B_i = {0, 0, 0, 0};
  float4 vis_2C_r = {0, 0, 0, 0}, vis_2C_i = {0, 0, 0, 0};

  for (uint major = 0; major < NR_SAMPLES_PER_CHANNEL; major += BLOCK_SIZE) {
    // load data into local memory
#pragma unroll 1
    /* for (uint i = get_local_id(0); i < BLOCK_SIZE * NR_STATIONS; i += get_local_size(0)) { */
    for (uint i = threadIdx.x; i < BLOCK_SIZE * NR_STATIONS; i += blockDim.x) {
      uint time = i % BLOCK_SIZE;
      uint stat = i / BLOCK_SIZE;

      samples[stat % 3][time][stat / 3] = (*correctedData)[stat][channel][major + time];
    }

    /* barrier(CLK_LOCAL_MEM_FENCE); */
    __syncthreads();

    if (compute_correlations) {
      for (uint time = 0; time < BLOCK_SIZE; time++) {
        fcomplex2 sample_0 = samples[0][time][y];
        fcomplex2 sample_A = samples[0][time][x];
        fcomplex2 sample_B = samples[1][time][x];
        fcomplex2 sample_C = samples[2][time][x];
        fcomplex2 sample_1 = samples[1][time][y];
        fcomplex2 sample_2 = samples[2][time][y];

        /* vis_0A_r += sample_0.xxzz * sample_A.xzxz; */
        /* vis_0A_i += sample_0.yyww * sample_A.xzxz; */
        /* vis_0B_r += sample_0.xxzz * sample_B.xzxz; */
        /* vis_0B_i += sample_0.yyww * sample_B.xzxz; */
        /* vis_0C_r += sample_0.xxzz * sample_C.xzxz; */
        /* vis_0C_i += sample_0.yyww * sample_C.xzxz; */
        /* vis_1A_r += sample_1.xxzz * sample_A.xzxz; */
        /* vis_1A_i += sample_1.yyww * sample_A.xzxz; */
        /* vis_1B_r += sample_1.xxzz * sample_B.xzxz; */
        /* vis_1B_i += sample_1.yyww * sample_B.xzxz; */
        /* vis_1C_r += sample_1.xxzz * sample_C.xzxz; */
        /* vis_1C_i += sample_1.yyww * sample_C.xzxz; */
        /* vis_2A_r += sample_2.xxzz * sample_A.xzxz; */
        /* vis_2A_i += sample_2.yyww * sample_A.xzxz; */
        /* vis_2B_r += sample_2.xxzz * sample_B.xzxz; */
        /* vis_2B_i += sample_2.yyww * sample_B.xzxz; */
        /* vis_2C_r += sample_2.xxzz * sample_C.xzxz; */
        /* vis_2C_i += sample_2.yyww * sample_C.xzxz; */
        vis_0A_r += SWIZZLE(sample_0,x,x,z,z) * SWIZZLE(sample_A,x,z,x,z);
        vis_0A_i += SWIZZLE(sample_0,y,y,w,w) * SWIZZLE(sample_A,x,z,x,z);
        vis_0B_r += SWIZZLE(sample_0,x,x,z,z) * SWIZZLE(sample_B,x,z,x,z);
        vis_0B_i += SWIZZLE(sample_0,y,y,w,w) * SWIZZLE(sample_B,x,z,x,z);
        vis_0C_r += SWIZZLE(sample_0,x,x,z,z) * SWIZZLE(sample_C,x,z,x,z);
        vis_0C_i += SWIZZLE(sample_0,y,y,w,w) * SWIZZLE(sample_C,x,z,x,z);
        vis_1A_r += SWIZZLE(sample_1,x,x,z,z) * SWIZZLE(sample_A,x,z,x,z);
        vis_1A_i += SWIZZLE(sample_1,y,y,w,w) * SWIZZLE(sample_A,x,z,x,z);
        vis_1B_r += SWIZZLE(sample_1,x,x,z,z) * SWIZZLE(sample_B,x,z,x,z);
        vis_1B_i += SWIZZLE(sample_1,y,y,w,w) * SWIZZLE(sample_B,x,z,x,z);
        vis_1C_r += SWIZZLE(sample_1,x,x,z,z) * SWIZZLE(sample_C,x,z,x,z);
        vis_1C_i += SWIZZLE(sample_1,y,y,w,w) * SWIZZLE(sample_C,x,z,x,z);
        vis_2A_r += SWIZZLE(sample_2,x,x,z,z) * SWIZZLE(sample_A,x,z,x,z);
        vis_2A_i += SWIZZLE(sample_2,y,y,w,w) * SWIZZLE(sample_A,x,z,x,z);
        vis_2B_r += SWIZZLE(sample_2,x,x,z,z) * SWIZZLE(sample_B,x,z,x,z);
        vis_2B_i += SWIZZLE(sample_2,y,y,w,w) * SWIZZLE(sample_B,x,z,x,z);
        vis_2C_r += SWIZZLE(sample_2,x,x,z,z) * SWIZZLE(sample_C,x,z,x,z);
        vis_2C_i += SWIZZLE(sample_2,y,y,w,w) * SWIZZLE(sample_C,x,z,x,z);

        /* vis_0A_r += sample_0.yyww * sample_A.ywyw; */
        /* vis_0A_i -= sample_0.xxzz * sample_A.ywyw; */
        /* vis_0B_r += sample_0.yyww * sample_B.ywyw; */
        /* vis_0B_i -= sample_0.xxzz * sample_B.ywyw; */
        /* vis_0C_r += sample_0.yyww * sample_C.ywyw; */
        /* vis_0C_i -= sample_0.xxzz * sample_C.ywyw; */
        /* vis_1A_r += sample_1.yyww * sample_A.ywyw; */
        /* vis_1A_i -= sample_1.xxzz * sample_A.ywyw; */
        /* vis_1B_r += sample_1.yyww * sample_B.ywyw; */
        /* vis_1B_i -= sample_1.xxzz * sample_B.ywyw; */
        /* vis_1C_r += sample_1.yyww * sample_C.ywyw; */
        /* vis_1C_i -= sample_1.xxzz * sample_C.ywyw; */
        /* vis_2A_r += sample_2.yyww * sample_A.ywyw; */
        /* vis_2A_i -= sample_2.xxzz * sample_A.ywyw; */
        /* vis_2B_r += sample_2.yyww * sample_B.ywyw; */
        /* vis_2B_i -= sample_2.xxzz * sample_B.ywyw; */
        /* vis_2C_r += sample_2.yyww * sample_C.ywyw; */
        /* vis_2C_i -= sample_2.xxzz * sample_C.ywyw; */
        vis_0A_r += SWIZZLE(sample_0,y,y,w,w) * SWIZZLE(sample_A,y,w,y,w);
        vis_0A_i -= SWIZZLE(sample_0,x,x,z,z) * SWIZZLE(sample_A,y,w,y,w);
        vis_0B_r += SWIZZLE(sample_0,y,y,w,w) * SWIZZLE(sample_B,y,w,y,w);
        vis_0B_i -= SWIZZLE(sample_0,x,x,z,z) * SWIZZLE(sample_B,y,w,y,w);
        vis_0C_r += SWIZZLE(sample_0,y,y,w,w) * SWIZZLE(sample_C,y,w,y,w);
        vis_0C_i -= SWIZZLE(sample_0,x,x,z,z) * SWIZZLE(sample_C,y,w,y,w);
        vis_1A_r += SWIZZLE(sample_1,y,y,w,w) * SWIZZLE(sample_A,y,w,y,w);
        vis_1A_i -= SWIZZLE(sample_1,x,x,z,z) * SWIZZLE(sample_A,y,w,y,w);
        vis_1B_r += SWIZZLE(sample_1,y,y,w,w) * SWIZZLE(sample_B,y,w,y,w);
        vis_1B_i -= SWIZZLE(sample_1,x,x,z,z) * SWIZZLE(sample_B,y,w,y,w);
        vis_1C_r += SWIZZLE(sample_1,y,y,w,w) * SWIZZLE(sample_C,y,w,y,w);
        vis_1C_i -= SWIZZLE(sample_1,x,x,z,z) * SWIZZLE(sample_C,y,w,y,w);
        vis_2A_r += SWIZZLE(sample_2,y,y,w,w) * SWIZZLE(sample_A,y,w,y,w);
        vis_2A_i -= SWIZZLE(sample_2,x,x,z,z) * SWIZZLE(sample_A,y,w,y,w);
        vis_2B_r += SWIZZLE(sample_2,y,y,w,w) * SWIZZLE(sample_B,y,w,y,w);
        vis_2B_i -= SWIZZLE(sample_2,x,x,z,z) * SWIZZLE(sample_B,y,w,y,w);
        vis_2C_r += SWIZZLE(sample_2,y,y,w,w) * SWIZZLE(sample_C,y,w,y,w);
        vis_2C_i -= SWIZZLE(sample_2,x,x,z,z) * SWIZZLE(sample_C,y,w,y,w);
      }
    }

    /* barrier(CLK_LOCAL_MEM_FENCE); */
    __syncthreads();
  }

  // write visibilities
  uint stat_0 = 3 * y;
  uint stat_1 = stat_0 + 1;
  uint stat_2 = stat_0 + 2;
  uint stat_B = stat_A + 1;
  uint stat_C = stat_A + 2;

  bool do_baseline_0A = stat_0 < NR_STATIONS && stat_A < NR_STATIONS && stat_0 <= stat_A;
  bool do_baseline_0B = stat_0 < NR_STATIONS && stat_B < NR_STATIONS && stat_0 <= stat_B;
  bool do_baseline_0C = stat_0 < NR_STATIONS && stat_C < NR_STATIONS && stat_0 <= stat_C;
  bool do_baseline_1A = stat_1 < NR_STATIONS && stat_A < NR_STATIONS && stat_1 <= stat_A;
  bool do_baseline_1B = stat_1 < NR_STATIONS && stat_B < NR_STATIONS && stat_1 <= stat_B;
  bool do_baseline_1C = stat_1 < NR_STATIONS && stat_C < NR_STATIONS && stat_1 <= stat_C;
  bool do_baseline_2A = stat_2 < NR_STATIONS && stat_A < NR_STATIONS && stat_2 <= stat_A;
  bool do_baseline_2B = stat_2 < NR_STATIONS && stat_B < NR_STATIONS && stat_2 <= stat_B;
  bool do_baseline_2C = stat_2 < NR_STATIONS && stat_C < NR_STATIONS && stat_2 <= stat_C;

  if (do_baseline_0A) {
    uint baseline = (stat_A * (stat_A + 1) / 2) + stat_0;
    /* (*visibilities)[baseline][channel] = (fcomplex4) { vis_0A_r.x, vis_0A_i.x, vis_0A_r.y, vis_0A_i.y, vis_0A_r.z, vis_0A_i.z, vis_0A_r.w, vis_0A_i.w }; */
    (*visibilities)[baseline][channel][0][0] = make_float2(vis_0A_r.x, vis_0A_i.x);
    (*visibilities)[baseline][channel][0][1] = make_float2(vis_0A_r.y, vis_0A_i.y);
    (*visibilities)[baseline][channel][1][0] = make_float2(vis_0A_r.z, vis_0A_i.z);
    (*visibilities)[baseline][channel][1][1] = make_float2(vis_0A_r.w, vis_0A_i.w);
  }

  if (do_baseline_0B) {
    uint baseline = (stat_B * (stat_B + 1) / 2) + stat_0;
    /* (*visibilities)[baseline][channel] = (fcomplex4) { vis_0B_r.x, vis_0B_i.x, vis_0B_r.y, vis_0B_i.y, vis_0B_r.z, vis_0B_i.z, vis_0B_r.w, vis_0B_i.w }; */
    (*visibilities)[baseline][channel][0][0] = make_float2(vis_0B_r.x, vis_0B_i.x);
    (*visibilities)[baseline][channel][0][1] = make_float2(vis_0B_r.y, vis_0B_i.y);
    (*visibilities)[baseline][channel][1][0] = make_float2(vis_0B_r.z, vis_0B_i.z);
    (*visibilities)[baseline][channel][1][1] = make_float2(vis_0B_r.w, vis_0B_i.w);
  }

  if (do_baseline_0C) {
    uint baseline = (stat_C * (stat_C + 1) / 2) + stat_0;
    /* (*visibilities)[baseline][channel] = (fcomplex4) { vis_0C_r.x, vis_0C_i.x, vis_0C_r.y, vis_0C_i.y, vis_0C_r.z, vis_0C_i.z, vis_0C_r.w, vis_0C_i.w }; */
    (*visibilities)[baseline][channel][0][0] = make_float2(vis_0C_r.x, vis_0C_i.x);
    (*visibilities)[baseline][channel][0][1] = make_float2(vis_0C_r.y, vis_0C_i.y);
    (*visibilities)[baseline][channel][1][0] = make_float2(vis_0C_r.z, vis_0C_i.z);
    (*visibilities)[baseline][channel][1][1] = make_float2(vis_0C_r.w, vis_0C_i.w);
  }

  if (do_baseline_1A) {
    uint baseline = (stat_A * (stat_A + 1) / 2) + stat_1;
    /* (*visibilities)[baseline][channel] = (fcomplex4) { vis_1A_r.x, vis_1A_i.x, vis_1A_r.y, vis_1A_i.y, vis_1A_r.z, vis_1A_i.z, vis_1A_r.w, vis_1A_i.w }; */
    (*visibilities)[baseline][channel][0][0] = make_float2(vis_1A_r.x, vis_1A_i.x);
    (*visibilities)[baseline][channel][0][1] = make_float2(vis_1A_r.y, vis_1A_i.y);
    (*visibilities)[baseline][channel][1][0] = make_float2(vis_1A_r.z, vis_1A_i.z);
    (*visibilities)[baseline][channel][1][1] = make_float2(vis_1A_r.w, vis_1A_i.w);
  }

  if (do_baseline_1B) {
    uint baseline = (stat_B * (stat_B + 1) / 2) + stat_1;
    /* (*visibilities)[baseline][channel] = (fcomplex4) { vis_1B_r.x, vis_1B_i.x, vis_1B_r.y, vis_1B_i.y, vis_1B_r.z, vis_1B_i.z, vis_1B_r.w, vis_1B_i.w }; */
    (*visibilities)[baseline][channel][0][0] = make_float2(vis_1B_r.x, vis_1B_i.x);
    (*visibilities)[baseline][channel][0][1] = make_float2(vis_1B_r.y, vis_1B_i.y);
    (*visibilities)[baseline][channel][1][0] = make_float2(vis_1B_r.z, vis_1B_i.z);
    (*visibilities)[baseline][channel][1][1] = make_float2(vis_1B_r.w, vis_1B_i.w);
  }

  if (do_baseline_1C) {
    uint baseline = (stat_C * (stat_C + 1) / 2) + stat_1;
    /* (*visibilities)[baseline][channel] = (fcomplex4) { vis_1C_r.x, vis_1C_i.x, vis_1C_r.y, vis_1C_i.y, vis_1C_r.z, vis_1C_i.z, vis_1C_r.w, vis_1C_i.w }; */
    (*visibilities)[baseline][channel][0][0] = make_float2(vis_1C_r.x, vis_1C_i.x);
    (*visibilities)[baseline][channel][0][1] = make_float2(vis_1C_r.y, vis_1C_i.y);
    (*visibilities)[baseline][channel][1][0] = make_float2(vis_1C_r.z, vis_1C_i.z);
    (*visibilities)[baseline][channel][1][1] = make_float2(vis_1C_r.w, vis_1C_i.w);
  }

  if (do_baseline_2A) {
    uint baseline = (stat_A * (stat_A + 1) / 2) + stat_2;
    /* (*visibilities)[baseline][channel] = (fcomplex4) { vis_2A_r.x, vis_2A_i.x, vis_2A_r.y, vis_2A_i.y, vis_2A_r.z, vis_2A_i.z, vis_2A_r.w, vis_2A_i.w }; */
    (*visibilities)[baseline][channel][0][0] = make_float2(vis_2A_r.x, vis_2A_i.x);
    (*visibilities)[baseline][channel][0][1] = make_float2(vis_2A_r.y, vis_2A_i.y);
    (*visibilities)[baseline][channel][1][0] = make_float2(vis_2A_r.z, vis_2A_i.z);
    (*visibilities)[baseline][channel][1][1] = make_float2(vis_2A_r.w, vis_2A_i.w);
  }

  if (do_baseline_2B) {
    uint baseline = (stat_B * (stat_B + 1) / 2) + stat_2;
    /* (*visibilities)[baseline][channel] = (fcomplex4) { vis_2B_r.x, vis_2B_i.x, vis_2B_r.y, vis_2B_i.y, vis_2B_r.z, vis_2B_i.z, vis_2B_r.w, vis_2B_i.w }; */
    (*visibilities)[baseline][channel][0][0] = make_float2(vis_2B_r.x, vis_2B_i.x);
    (*visibilities)[baseline][channel][0][1] = make_float2(vis_2B_r.y, vis_2B_i.y);
    (*visibilities)[baseline][channel][1][0] = make_float2(vis_2B_r.z, vis_2B_i.z);
    (*visibilities)[baseline][channel][1][1] = make_float2(vis_2B_r.w, vis_2B_i.w);
  }

  if (do_baseline_2C) {
    uint baseline = (stat_C * (stat_C + 1) / 2) + stat_2;
    /* (*visibilities)[baseline][channel] = (fcomplex4) { vis_2C_r.x, vis_2C_i.x, vis_2C_r.y, vis_2C_i.y, vis_2C_r.z, vis_2C_i.z, vis_2C_r.w, vis_2C_i.w }; */
    (*visibilities)[baseline][channel][0][0] = make_float2(vis_2C_r.x, vis_2C_i.x);
    (*visibilities)[baseline][channel][0][1] = make_float2(vis_2C_r.y, vis_2C_i.y);
    (*visibilities)[baseline][channel][1][0] = make_float2(vis_2C_r.z, vis_2C_i.z);
    (*visibilities)[baseline][channel][1][1] = make_float2(vis_2C_r.w, vis_2C_i.w);
  }
}


/*!
 * See the correlate() kernel.
 */
/* __kernel void correlate_4x4(__global void *visibilitiesPtr, */
/*                             __global const void *correctedDataPtr */
/*                             ) */
__global__ void correlate_4x4(void *visibilitiesPtr, const void *correctedDataPtr)
{
  VisibilitiesType visibilities = (VisibilitiesType) visibilitiesPtr;
  CorrectedDataType correctedData = (CorrectedDataType) correctedDataPtr;

  __shared__ fcomplex2 samples[4][BLOCK_SIZE][(NR_STATIONS + 3) / 4 | 1]; // avoid power-of-2

  /* uint block = get_global_id(0); */
  uint block = blockIdx.x * blockDim.x + threadIdx.x;
  /* uint channel = get_global_id(1) + 1; */
#if NR_CHANNELS == 1
  uint channel = blockIdx.y;
#else
  uint channel = blockIdx.y + 1;
#endif

  /* uint x = convert_uint_rtz(sqrt(convert_float(8 * block + 1)) - 0.99999f) / 2; */
  uint x = __float2uint_rz(sqrtf(float(8 * block + 1)) - 0.99999f) / 2;
  uint y = block - x * (x + 1) / 2;

  uint stat_A = 4 * x;

  bool compute_correlations = stat_A < NR_STATIONS;

  /* float4 vis_0A_r = (float4) 0, vis_0A_i = (float4) 0; */
  /* float4 vis_0B_r = (float4) 0, vis_0B_i = (float4) 0; */
  /* float4 vis_0C_r = (float4) 0, vis_0C_i = (float4) 0; */
  /* float4 vis_0D_r = (float4) 0, vis_0D_i = (float4) 0; */
  /* float4 vis_1A_r = (float4) 0, vis_1A_i = (float4) 0; */
  /* float4 vis_1B_r = (float4) 0, vis_1B_i = (float4) 0; */
  /* float4 vis_1C_r = (float4) 0, vis_1C_i = (float4) 0; */
  /* float4 vis_1D_r = (float4) 0, vis_1D_i = (float4) 0; */
  /* float4 vis_2A_r = (float4) 0, vis_2A_i = (float4) 0; */
  /* float4 vis_2B_r = (float4) 0, vis_2B_i = (float4) 0; */
  /* float4 vis_2C_r = (float4) 0, vis_2C_i = (float4) 0; */
  /* float4 vis_2D_r = (float4) 0, vis_2D_i = (float4) 0; */
  /* float4 vis_3A_r = (float4) 0, vis_3A_i = (float4) 0; */
  /* float4 vis_3B_r = (float4) 0, vis_3B_i = (float4) 0; */
  /* float4 vis_3C_r = (float4) 0, vis_3C_i = (float4) 0; */
  /* float4 vis_3D_r = (float4) 0, vis_3D_i = (float4) 0; */
  float4 vis_0A_r = {0, 0, 0, 0}, vis_0A_i = {0, 0, 0, 0};
  float4 vis_0B_r = {0, 0, 0, 0}, vis_0B_i = {0, 0, 0, 0};
  float4 vis_0C_r = {0, 0, 0, 0}, vis_0C_i = {0, 0, 0, 0};
  float4 vis_0D_r = {0, 0, 0, 0}, vis_0D_i = {0, 0, 0, 0};
  float4 vis_1A_r = {0, 0, 0, 0}, vis_1A_i = {0, 0, 0, 0};
  float4 vis_1B_r = {0, 0, 0, 0}, vis_1B_i = {0, 0, 0, 0};
  float4 vis_1C_r = {0, 0, 0, 0}, vis_1C_i = {0, 0, 0, 0};
  float4 vis_1D_r = {0, 0, 0, 0}, vis_1D_i = {0, 0, 0, 0};
  float4 vis_2A_r = {0, 0, 0, 0}, vis_2A_i = {0, 0, 0, 0};
  float4 vis_2B_r = {0, 0, 0, 0}, vis_2B_i = {0, 0, 0, 0};
  float4 vis_2C_r = {0, 0, 0, 0}, vis_2C_i = {0, 0, 0, 0};
  float4 vis_2D_r = {0, 0, 0, 0}, vis_2D_i = {0, 0, 0, 0};
  float4 vis_3A_r = {0, 0, 0, 0}, vis_3A_i = {0, 0, 0, 0};
  float4 vis_3B_r = {0, 0, 0, 0}, vis_3B_i = {0, 0, 0, 0};
  float4 vis_3C_r = {0, 0, 0, 0}, vis_3C_i = {0, 0, 0, 0};
  float4 vis_3D_r = {0, 0, 0, 0}, vis_3D_i = {0, 0, 0, 0};

  for (uint major = 0; major < NR_SAMPLES_PER_CHANNEL; major += BLOCK_SIZE) {
    // load data into local memory
#pragma unroll 1
    /* for (uint i = get_local_id(0); i < BLOCK_SIZE * NR_STATIONS; i += get_local_size(0)) { */
    for (uint i = threadIdx.x; i < BLOCK_SIZE * NR_STATIONS; i += blockDim.x) {
      uint time = i % BLOCK_SIZE;
      uint stat = i / BLOCK_SIZE;

      samples[stat % 4][time][stat / 4] = (*correctedData)[stat][channel][major + time];
    }

    /* barrier(CLK_LOCAL_MEM_FENCE); */
    __syncthreads();

    if (compute_correlations) {
      for (uint time = 0; time < BLOCK_SIZE; time++) {
        fcomplex2 sample_0 = samples[0][time][y];
        fcomplex2 sample_A = samples[0][time][x];
        fcomplex2 sample_B = samples[1][time][x];
        fcomplex2 sample_C = samples[2][time][x];
        fcomplex2 sample_D = samples[3][time][x];
        fcomplex2 sample_1 = samples[1][time][y];
        fcomplex2 sample_2 = samples[2][time][y];
        fcomplex2 sample_3 = samples[3][time][y];

        /* vis_0A_r += sample_0.xxzz * sample_A.xzxz; */
        /* vis_0A_i += sample_0.yyww * sample_A.xzxz; */
        /* vis_0B_r += sample_0.xxzz * sample_B.xzxz; */
        /* vis_0B_i += sample_0.yyww * sample_B.xzxz; */
        /* vis_0C_r += sample_0.xxzz * sample_C.xzxz; */
        /* vis_0C_i += sample_0.yyww * sample_C.xzxz; */
        /* vis_0D_r += sample_0.xxzz * sample_D.xzxz; */
        /* vis_0D_i += sample_0.yyww * sample_D.xzxz; */
        /* vis_1A_r += sample_1.xxzz * sample_A.xzxz; */
        /* vis_1A_i += sample_1.yyww * sample_A.xzxz; */
        /* vis_1B_r += sample_1.xxzz * sample_B.xzxz; */
        /* vis_1B_i += sample_1.yyww * sample_B.xzxz; */
        /* vis_1C_r += sample_1.xxzz * sample_C.xzxz; */
        /* vis_1C_i += sample_1.yyww * sample_C.xzxz; */
        /* vis_1D_r += sample_1.xxzz * sample_D.xzxz; */
        /* vis_1D_i += sample_1.yyww * sample_D.xzxz; */
        /* vis_2A_r += sample_2.xxzz * sample_A.xzxz; */
        /* vis_2A_i += sample_2.yyww * sample_A.xzxz; */
        /* vis_2B_r += sample_2.xxzz * sample_B.xzxz; */
        /* vis_2B_i += sample_2.yyww * sample_B.xzxz; */
        /* vis_2C_r += sample_2.xxzz * sample_C.xzxz; */
        /* vis_2C_i += sample_2.yyww * sample_C.xzxz; */
        /* vis_2D_r += sample_2.xxzz * sample_D.xzxz; */
        /* vis_2D_i += sample_2.yyww * sample_D.xzxz; */
        /* vis_3A_r += sample_3.xxzz * sample_A.xzxz; */
        /* vis_3A_i += sample_3.yyww * sample_A.xzxz; */
        /* vis_3B_r += sample_3.xxzz * sample_B.xzxz; */
        /* vis_3B_i += sample_3.yyww * sample_B.xzxz; */
        /* vis_3C_r += sample_3.xxzz * sample_C.xzxz; */
        /* vis_3C_i += sample_3.yyww * sample_C.xzxz; */
        /* vis_3D_r += sample_3.xxzz * sample_D.xzxz; */
        /* vis_3D_i += sample_3.yyww * sample_D.xzxz; */
        vis_0A_r += SWIZZLE(sample_0,x,x,z,z) * SWIZZLE(sample_A,x,z,x,z);
        vis_0A_i += SWIZZLE(sample_0,y,y,w,w) * SWIZZLE(sample_A,x,z,x,z);
        vis_0B_r += SWIZZLE(sample_0,x,x,z,z) * SWIZZLE(sample_B,x,z,x,z);
        vis_0B_i += SWIZZLE(sample_0,y,y,w,w) * SWIZZLE(sample_B,x,z,x,z);
        vis_0C_r += SWIZZLE(sample_0,x,x,z,z) * SWIZZLE(sample_C,x,z,x,z);
        vis_0C_i += SWIZZLE(sample_0,y,y,w,w) * SWIZZLE(sample_C,x,z,x,z);
        vis_0D_r += SWIZZLE(sample_0,x,x,z,z) * SWIZZLE(sample_D,x,z,x,z);
        vis_0D_i += SWIZZLE(sample_0,y,y,w,w) * SWIZZLE(sample_D,x,z,x,z);
        vis_1A_r += SWIZZLE(sample_1,x,x,z,z) * SWIZZLE(sample_A,x,z,x,z);
        vis_1A_i += SWIZZLE(sample_1,y,y,w,w) * SWIZZLE(sample_A,x,z,x,z);
        vis_1B_r += SWIZZLE(sample_1,x,x,z,z) * SWIZZLE(sample_B,x,z,x,z);
        vis_1B_i += SWIZZLE(sample_1,y,y,w,w) * SWIZZLE(sample_B,x,z,x,z);
        vis_1C_r += SWIZZLE(sample_1,x,x,z,z) * SWIZZLE(sample_C,x,z,x,z);
        vis_1C_i += SWIZZLE(sample_1,y,y,w,w) * SWIZZLE(sample_C,x,z,x,z);
        vis_1D_r += SWIZZLE(sample_1,x,x,z,z) * SWIZZLE(sample_D,x,z,x,z);
        vis_1D_i += SWIZZLE(sample_1,y,y,w,w) * SWIZZLE(sample_D,x,z,x,z);
        vis_2A_r += SWIZZLE(sample_2,x,x,z,z) * SWIZZLE(sample_A,x,z,x,z);
        vis_2A_i += SWIZZLE(sample_2,y,y,w,w) * SWIZZLE(sample_A,x,z,x,z);
        vis_2B_r += SWIZZLE(sample_2,x,x,z,z) * SWIZZLE(sample_B,x,z,x,z);
        vis_2B_i += SWIZZLE(sample_2,y,y,w,w) * SWIZZLE(sample_B,x,z,x,z);
        vis_2C_r += SWIZZLE(sample_2,x,x,z,z) * SWIZZLE(sample_C,x,z,x,z);
        vis_2C_i += SWIZZLE(sample_2,y,y,w,w) * SWIZZLE(sample_C,x,z,x,z);
        vis_2D_r += SWIZZLE(sample_2,x,x,z,z) * SWIZZLE(sample_D,x,z,x,z);
        vis_2D_i += SWIZZLE(sample_2,y,y,w,w) * SWIZZLE(sample_D,x,z,x,z);
        vis_3A_r += SWIZZLE(sample_3,x,x,z,z) * SWIZZLE(sample_A,x,z,x,z);
        vis_3A_i += SWIZZLE(sample_3,y,y,w,w) * SWIZZLE(sample_A,x,z,x,z);
        vis_3B_r += SWIZZLE(sample_3,x,x,z,z) * SWIZZLE(sample_B,x,z,x,z);
        vis_3B_i += SWIZZLE(sample_3,y,y,w,w) * SWIZZLE(sample_B,x,z,x,z);
        vis_3C_r += SWIZZLE(sample_3,x,x,z,z) * SWIZZLE(sample_C,x,z,x,z);
        vis_3C_i += SWIZZLE(sample_3,y,y,w,w) * SWIZZLE(sample_C,x,z,x,z);
        vis_3D_r += SWIZZLE(sample_3,x,x,z,z) * SWIZZLE(sample_D,x,z,x,z);
        vis_3D_i += SWIZZLE(sample_3,y,y,w,w) * SWIZZLE(sample_D,x,z,x,z);

        /* vis_0A_r += sample_0.yyww * sample_A.ywyw; */
        /* vis_0A_i -= sample_0.xxzz * sample_A.ywyw; */
        /* vis_0B_r += sample_0.yyww * sample_B.ywyw; */
        /* vis_0B_i -= sample_0.xxzz * sample_B.ywyw; */
        /* vis_0C_r += sample_0.yyww * sample_C.ywyw; */
        /* vis_0C_i -= sample_0.xxzz * sample_C.ywyw; */
        /* vis_0D_r += sample_0.yyww * sample_D.ywyw; */
        /* vis_0D_i -= sample_0.xxzz * sample_D.ywyw; */
        /* vis_1A_r += sample_1.yyww * sample_A.ywyw; */
        /* vis_1A_i -= sample_1.xxzz * sample_A.ywyw; */
        /* vis_1B_r += sample_1.yyww * sample_B.ywyw; */
        /* vis_1B_i -= sample_1.xxzz * sample_B.ywyw; */
        /* vis_1C_r += sample_1.yyww * sample_C.ywyw; */
        /* vis_1C_i -= sample_1.xxzz * sample_C.ywyw; */
        /* vis_1D_r += sample_1.yyww * sample_D.ywyw; */
        /* vis_1D_i -= sample_1.xxzz * sample_D.ywyw; */
        /* vis_2A_r += sample_2.yyww * sample_A.ywyw; */
        /* vis_2A_i -= sample_2.xxzz * sample_A.ywyw; */
        /* vis_2B_r += sample_2.yyww * sample_B.ywyw; */
        /* vis_2B_i -= sample_2.xxzz * sample_B.ywyw; */
        /* vis_2C_r += sample_2.yyww * sample_C.ywyw; */
        /* vis_2C_i -= sample_2.xxzz * sample_C.ywyw; */
        /* vis_2D_r += sample_2.yyww * sample_D.ywyw; */
        /* vis_2D_i -= sample_2.xxzz * sample_D.ywyw; */
        /* vis_3A_r += sample_3.yyww * sample_A.ywyw; */
        /* vis_3A_i -= sample_3.xxzz * sample_A.ywyw; */
        /* vis_3B_r += sample_3.yyww * sample_B.ywyw; */
        /* vis_3B_i -= sample_3.xxzz * sample_B.ywyw; */
        /* vis_3C_r += sample_3.yyww * sample_C.ywyw; */
        /* vis_3C_i -= sample_3.xxzz * sample_C.ywyw; */
        /* vis_3D_r += sample_3.yyww * sample_D.ywyw; */
        /* vis_3D_i -= sample_3.xxzz * sample_D.ywyw; */
        vis_0A_r += SWIZZLE(sample_0,y,y,w,w) * SWIZZLE(sample_A,y,w,y,w);
        vis_0A_i -= SWIZZLE(sample_0,x,x,z,z) * SWIZZLE(sample_A,y,w,y,w);
        vis_0B_r += SWIZZLE(sample_0,y,y,w,w) * SWIZZLE(sample_B,y,w,y,w);
        vis_0B_i -= SWIZZLE(sample_0,x,x,z,z) * SWIZZLE(sample_B,y,w,y,w);
        vis_0C_r += SWIZZLE(sample_0,y,y,w,w) * SWIZZLE(sample_C,y,w,y,w);
        vis_0C_i -= SWIZZLE(sample_0,x,x,z,z) * SWIZZLE(sample_C,y,w,y,w);
        vis_0D_r += SWIZZLE(sample_0,y,y,w,w) * SWIZZLE(sample_D,y,w,y,w);
        vis_0D_i -= SWIZZLE(sample_0,x,x,z,z) * SWIZZLE(sample_D,y,w,y,w);
        vis_1A_r += SWIZZLE(sample_1,y,y,w,w) * SWIZZLE(sample_A,y,w,y,w);
        vis_1A_i -= SWIZZLE(sample_1,x,x,z,z) * SWIZZLE(sample_A,y,w,y,w);
        vis_1B_r += SWIZZLE(sample_1,y,y,w,w) * SWIZZLE(sample_B,y,w,y,w);
        vis_1B_i -= SWIZZLE(sample_1,x,x,z,z) * SWIZZLE(sample_B,y,w,y,w);
        vis_1C_r += SWIZZLE(sample_1,y,y,w,w) * SWIZZLE(sample_C,y,w,y,w);
        vis_1C_i -= SWIZZLE(sample_1,x,x,z,z) * SWIZZLE(sample_C,y,w,y,w);
        vis_1D_r += SWIZZLE(sample_1,y,y,w,w) * SWIZZLE(sample_D,y,w,y,w);
        vis_1D_i -= SWIZZLE(sample_1,x,x,z,z) * SWIZZLE(sample_D,y,w,y,w);
        vis_2A_r += SWIZZLE(sample_2,y,y,w,w) * SWIZZLE(sample_A,y,w,y,w);
        vis_2A_i -= SWIZZLE(sample_2,x,x,z,z) * SWIZZLE(sample_A,y,w,y,w);
        vis_2B_r += SWIZZLE(sample_2,y,y,w,w) * SWIZZLE(sample_B,y,w,y,w);
        vis_2B_i -= SWIZZLE(sample_2,x,x,z,z) * SWIZZLE(sample_B,y,w,y,w);
        vis_2C_r += SWIZZLE(sample_2,y,y,w,w) * SWIZZLE(sample_C,y,w,y,w);
        vis_2C_i -= SWIZZLE(sample_2,x,x,z,z) * SWIZZLE(sample_C,y,w,y,w);
        vis_2D_r += SWIZZLE(sample_2,y,y,w,w) * SWIZZLE(sample_D,y,w,y,w);
        vis_2D_i -= SWIZZLE(sample_2,x,x,z,z) * SWIZZLE(sample_D,y,w,y,w);
        vis_3A_r += SWIZZLE(sample_3,y,y,w,w) * SWIZZLE(sample_A,y,w,y,w);
        vis_3A_i -= SWIZZLE(sample_3,x,x,z,z) * SWIZZLE(sample_A,y,w,y,w);
        vis_3B_r += SWIZZLE(sample_3,y,y,w,w) * SWIZZLE(sample_B,y,w,y,w);
        vis_3B_i -= SWIZZLE(sample_3,x,x,z,z) * SWIZZLE(sample_B,y,w,y,w);
        vis_3C_r += SWIZZLE(sample_3,y,y,w,w) * SWIZZLE(sample_C,y,w,y,w);
        vis_3C_i -= SWIZZLE(sample_3,x,x,z,z) * SWIZZLE(sample_C,y,w,y,w);
        vis_3D_r += SWIZZLE(sample_3,y,y,w,w) * SWIZZLE(sample_D,y,w,y,w);
        vis_3D_i -= SWIZZLE(sample_3,x,x,z,z) * SWIZZLE(sample_D,y,w,y,w);
      }
    }

    /* barrier(CLK_LOCAL_MEM_FENCE); */
    __syncthreads();
  }

  // write visibilities
  uint stat_0 = 4 * y;
  uint stat_1 = stat_0 + 1;
  uint stat_2 = stat_0 + 2;
  uint stat_3 = stat_0 + 3;
  uint stat_B = stat_A + 1;
  uint stat_C = stat_A + 2;
  uint stat_D = stat_A + 3;

  bool do_baseline_0A = stat_0 < NR_STATIONS && stat_A < NR_STATIONS && stat_0 <= stat_A;
  bool do_baseline_0B = stat_0 < NR_STATIONS && stat_B < NR_STATIONS && stat_0 <= stat_B;
  bool do_baseline_0C = stat_0 < NR_STATIONS && stat_C < NR_STATIONS && stat_0 <= stat_C;
  bool do_baseline_0D = stat_0 < NR_STATIONS && stat_D < NR_STATIONS && stat_0 <= stat_D;
  bool do_baseline_1A = stat_1 < NR_STATIONS && stat_A < NR_STATIONS && stat_1 <= stat_A;
  bool do_baseline_1B = stat_1 < NR_STATIONS && stat_B < NR_STATIONS && stat_1 <= stat_B;
  bool do_baseline_1C = stat_1 < NR_STATIONS && stat_C < NR_STATIONS && stat_1 <= stat_C;
  bool do_baseline_1D = stat_1 < NR_STATIONS && stat_D < NR_STATIONS && stat_1 <= stat_D;
  bool do_baseline_2A = stat_2 < NR_STATIONS && stat_A < NR_STATIONS && stat_2 <= stat_A;
  bool do_baseline_2B = stat_2 < NR_STATIONS && stat_B < NR_STATIONS && stat_2 <= stat_B;
  bool do_baseline_2C = stat_2 < NR_STATIONS && stat_C < NR_STATIONS && stat_2 <= stat_C;
  bool do_baseline_2D = stat_2 < NR_STATIONS && stat_D < NR_STATIONS && stat_2 <= stat_D;
  bool do_baseline_3A = stat_3 < NR_STATIONS && stat_A < NR_STATIONS && stat_3 <= stat_A;
  bool do_baseline_3B = stat_3 < NR_STATIONS && stat_B < NR_STATIONS && stat_3 <= stat_B;
  bool do_baseline_3C = stat_3 < NR_STATIONS && stat_C < NR_STATIONS && stat_3 <= stat_C;
  bool do_baseline_3D = stat_3 < NR_STATIONS && stat_D < NR_STATIONS && stat_3 <= stat_D;

  if (do_baseline_0A) {
    uint baseline = (stat_A * (stat_A + 1) / 2) + stat_0;
    /* (*visibilities)[baseline][channel] = (fcomplex4) { vis_0A_r.x, vis_0A_i.x, vis_0A_r.y, vis_0A_i.y, vis_0A_r.z, vis_0A_i.z, vis_0A_r.w, vis_0A_i.w }; */
    (*visibilities)[baseline][channel][0][0] = make_float2(vis_0A_r.x, vis_0A_i.x);
    (*visibilities)[baseline][channel][0][1] = make_float2(vis_0A_r.y, vis_0A_i.y);
    (*visibilities)[baseline][channel][1][0] = make_float2(vis_0A_r.z, vis_0A_i.z);
    (*visibilities)[baseline][channel][1][1] = make_float2(vis_0A_r.w, vis_0A_i.w);
  }

  if (do_baseline_0B) {
    uint baseline = (stat_B * (stat_B + 1) / 2) + stat_0;
    /* (*visibilities)[baseline][channel] = (fcomplex4) { vis_0B_r.x, vis_0B_i.x, vis_0B_r.y, vis_0B_i.y, vis_0B_r.z, vis_0B_i.z, vis_0B_r.w, vis_0B_i.w }; */
    (*visibilities)[baseline][channel][0][0] = make_float2(vis_0B_r.x, vis_0B_i.x);
    (*visibilities)[baseline][channel][0][1] = make_float2(vis_0B_r.y, vis_0B_i.y);
    (*visibilities)[baseline][channel][1][0] = make_float2(vis_0B_r.z, vis_0B_i.z);
    (*visibilities)[baseline][channel][1][1] = make_float2(vis_0B_r.w, vis_0B_i.w);
  }

  if (do_baseline_0C) {
    uint baseline = (stat_C * (stat_C + 1) / 2) + stat_0;
    /* (*visibilities)[baseline][channel] = (fcomplex4) { vis_0C_r.x, vis_0C_i.x, vis_0C_r.y, vis_0C_i.y, vis_0C_r.z, vis_0C_i.z, vis_0C_r.w, vis_0C_i.w }; */
    (*visibilities)[baseline][channel][0][0] = make_float2(vis_0C_r.x, vis_0C_i.x);
    (*visibilities)[baseline][channel][0][1] = make_float2(vis_0C_r.y, vis_0C_i.y);
    (*visibilities)[baseline][channel][1][0] = make_float2(vis_0C_r.z, vis_0C_i.z);
    (*visibilities)[baseline][channel][1][1] = make_float2(vis_0C_r.w, vis_0C_i.w);
  }

  if (do_baseline_0D) {
    uint baseline = (stat_D * (stat_D + 1) / 2) + stat_0;
    /* (*visibilities)[baseline][channel] = (fcomplex4) { vis_0D_r.x, vis_0D_i.x, vis_0D_r.y, vis_0D_i.y, vis_0D_r.z, vis_0D_i.z, vis_0D_r.w, vis_0D_i.w }; */
    (*visibilities)[baseline][channel][0][0] = make_float2(vis_0D_r.x, vis_0D_i.x);
    (*visibilities)[baseline][channel][0][1] = make_float2(vis_0D_r.y, vis_0D_i.y);
    (*visibilities)[baseline][channel][1][0] = make_float2(vis_0D_r.z, vis_0D_i.z);
    (*visibilities)[baseline][channel][1][1] = make_float2(vis_0D_r.w, vis_0D_i.w);
  }

  if (do_baseline_1A) {
    uint baseline = (stat_A * (stat_A + 1) / 2) + stat_1;
    /* (*visibilities)[baseline][channel] = (fcomplex4) { vis_1A_r.x, vis_1A_i.x, vis_1A_r.y, vis_1A_i.y, vis_1A_r.z, vis_1A_i.z, vis_1A_r.w, vis_1A_i.w }; */
    (*visibilities)[baseline][channel][0][0] = make_float2(vis_1A_r.x, vis_1A_i.x);
    (*visibilities)[baseline][channel][0][1] = make_float2(vis_1A_r.y, vis_1A_i.y);
    (*visibilities)[baseline][channel][1][0] = make_float2(vis_1A_r.z, vis_1A_i.z);
    (*visibilities)[baseline][channel][1][1] = make_float2(vis_1A_r.w, vis_1A_i.w);
  }

  if (do_baseline_1B) {
    uint baseline = (stat_B * (stat_B + 1) / 2) + stat_1;
    /* (*visibilities)[baseline][channel] = (fcomplex4) { vis_1B_r.x, vis_1B_i.x, vis_1B_r.y, vis_1B_i.y, vis_1B_r.z, vis_1B_i.z, vis_1B_r.w, vis_1B_i.w }; */
    (*visibilities)[baseline][channel][0][0] = make_float2(vis_1B_r.x, vis_1B_i.x);
    (*visibilities)[baseline][channel][0][1] = make_float2(vis_1B_r.y, vis_1B_i.y);
    (*visibilities)[baseline][channel][1][0] = make_float2(vis_1B_r.z, vis_1B_i.z);
    (*visibilities)[baseline][channel][1][1] = make_float2(vis_1B_r.w, vis_1B_i.w);
  }

  if (do_baseline_1C) {
    uint baseline = (stat_C * (stat_C + 1) / 2) + stat_1;
    /* (*visibilities)[baseline][channel] = (fcomplex4) { vis_1C_r.x, vis_1C_i.x, vis_1C_r.y, vis_1C_i.y, vis_1C_r.z, vis_1C_i.z, vis_1C_r.w, vis_1C_i.w }; */
    (*visibilities)[baseline][channel][0][0] = make_float2(vis_1C_r.x, vis_1C_i.x);
    (*visibilities)[baseline][channel][0][1] = make_float2(vis_1C_r.y, vis_1C_i.y);
    (*visibilities)[baseline][channel][1][0] = make_float2(vis_1C_r.z, vis_1C_i.z);
    (*visibilities)[baseline][channel][1][1] = make_float2(vis_1C_r.w, vis_1C_i.w);
  }

  if (do_baseline_1D) {
    uint baseline = (stat_D * (stat_D + 1) / 2) + stat_1;
    /* (*visibilities)[baseline][channel] = (fcomplex4) { vis_1D_r.x, vis_1D_i.x, vis_1D_r.y, vis_1D_i.y, vis_1D_r.z, vis_1D_i.z, vis_1D_r.w, vis_1D_i.w }; */
    (*visibilities)[baseline][channel][0][0] = make_float2(vis_1D_r.x, vis_1D_i.x);
    (*visibilities)[baseline][channel][0][1] = make_float2(vis_1D_r.y, vis_1D_i.y);
    (*visibilities)[baseline][channel][1][0] = make_float2(vis_1D_r.z, vis_1D_i.z);
    (*visibilities)[baseline][channel][1][1] = make_float2(vis_1D_r.w, vis_1D_i.w);
  }

  if (do_baseline_2A) {
    uint baseline = (stat_A * (stat_A + 1) / 2) + stat_2;
    /* (*visibilities)[baseline][channel] = (fcomplex4) { vis_2A_r.x, vis_2A_i.x, vis_2A_r.y, vis_2A_i.y, vis_2A_r.z, vis_2A_i.z, vis_2A_r.w, vis_2A_i.w }; */
    (*visibilities)[baseline][channel][0][0] = make_float2(vis_2A_r.x, vis_2A_i.x);
    (*visibilities)[baseline][channel][0][1] = make_float2(vis_2A_r.y, vis_2A_i.y);
    (*visibilities)[baseline][channel][1][0] = make_float2(vis_2A_r.z, vis_2A_i.z);
    (*visibilities)[baseline][channel][1][1] = make_float2(vis_2A_r.w, vis_2A_i.w);
  }

  if (do_baseline_2B) {
    uint baseline = (stat_B * (stat_B + 1) / 2) + stat_2;
    /* (*visibilities)[baseline][channel] = (fcomplex4) { vis_2B_r.x, vis_2B_i.x, vis_2B_r.y, vis_2B_i.y, vis_2B_r.z, vis_2B_i.z, vis_2B_r.w, vis_2B_i.w }; */
    (*visibilities)[baseline][channel][0][0] = make_float2(vis_2B_r.x, vis_2B_i.x);
    (*visibilities)[baseline][channel][0][1] = make_float2(vis_2B_r.y, vis_2B_i.y);
    (*visibilities)[baseline][channel][1][0] = make_float2(vis_2B_r.z, vis_2B_i.z);
    (*visibilities)[baseline][channel][1][1] = make_float2(vis_2B_r.w, vis_2B_i.w);
  }

  if (do_baseline_2C) {
    uint baseline = (stat_C * (stat_C + 1) / 2) + stat_2;
    /* (*visibilities)[baseline][channel] = (fcomplex4) { vis_2C_r.x, vis_2C_i.x, vis_2C_r.y, vis_2C_i.y, vis_2C_r.z, vis_2C_i.z, vis_2C_r.w, vis_2C_i.w }; */
    (*visibilities)[baseline][channel][0][0] = make_float2(vis_2C_r.x, vis_2C_i.x);
    (*visibilities)[baseline][channel][0][1] = make_float2(vis_2C_r.y, vis_2C_i.y);
    (*visibilities)[baseline][channel][1][0] = make_float2(vis_2C_r.z, vis_2C_i.z);
    (*visibilities)[baseline][channel][1][1] = make_float2(vis_2C_r.w, vis_2C_i.w);
  }

  if (do_baseline_2D) {
    uint baseline = (stat_D * (stat_D + 1) / 2) + stat_2;
    /* (*visibilities)[baseline][channel] = (fcomplex4) { vis_2D_r.x, vis_2D_i.x, vis_2D_r.y, vis_2D_i.y, vis_2D_r.z, vis_2D_i.z, vis_2D_r.w, vis_2D_i.w }; */
    (*visibilities)[baseline][channel][0][0] = make_float2(vis_2D_r.x, vis_2D_i.x);
    (*visibilities)[baseline][channel][0][1] = make_float2(vis_2D_r.y, vis_2D_i.y);
    (*visibilities)[baseline][channel][1][0] = make_float2(vis_2D_r.z, vis_2D_i.z);
    (*visibilities)[baseline][channel][1][1] = make_float2(vis_2D_r.w, vis_2D_i.w);
  }

  if (do_baseline_3A) {
    uint baseline = (stat_A * (stat_A + 1) / 2) + stat_3;
    /* (*visibilities)[baseline][channel] = (fcomplex4) { vis_3A_r.x, vis_3A_i.x, vis_3A_r.y, vis_3A_i.y, vis_3A_r.z, vis_3A_i.z, vis_3A_r.w, vis_3A_i.w }; */
    (*visibilities)[baseline][channel][0][0] = make_float2(vis_3A_r.x, vis_3A_i.x);
    (*visibilities)[baseline][channel][0][1] = make_float2(vis_3A_r.y, vis_3A_i.y);
    (*visibilities)[baseline][channel][1][0] = make_float2(vis_3A_r.z, vis_3A_i.z);
    (*visibilities)[baseline][channel][1][1] = make_float2(vis_3A_r.w, vis_3A_i.w);
  }

  if (do_baseline_3B) {
    uint baseline = (stat_B * (stat_B + 1) / 2) + stat_3;
    /* (*visibilities)[baseline][channel] = (fcomplex4) { vis_3B_r.x, vis_3B_i.x, vis_3B_r.y, vis_3B_i.y, vis_3B_r.z, vis_3B_i.z, vis_3B_r.w, vis_3B_i.w }; */
    (*visibilities)[baseline][channel][0][0] = make_float2(vis_3B_r.x, vis_3B_i.x);
    (*visibilities)[baseline][channel][0][1] = make_float2(vis_3B_r.y, vis_3B_i.y);
    (*visibilities)[baseline][channel][1][0] = make_float2(vis_3B_r.z, vis_3B_i.z);
    (*visibilities)[baseline][channel][1][1] = make_float2(vis_3B_r.w, vis_3B_i.w);
  }

  if (do_baseline_3C) {
    uint baseline = (stat_C * (stat_C + 1) / 2) + stat_3;
    /* (*visibilities)[baseline][channel] = (fcomplex4) { vis_3C_r.x, vis_3C_i.x, vis_3C_r.y, vis_3C_i.y, vis_3C_r.z, vis_3C_i.z, vis_3C_r.w, vis_3C_i.w }; */
    (*visibilities)[baseline][channel][0][0] = make_float2(vis_3C_r.x, vis_3C_i.x);
    (*visibilities)[baseline][channel][0][1] = make_float2(vis_3C_r.y, vis_3C_i.y);
    (*visibilities)[baseline][channel][1][0] = make_float2(vis_3C_r.z, vis_3C_i.z);
    (*visibilities)[baseline][channel][1][1] = make_float2(vis_3C_r.w, vis_3C_i.w);
  }

  if (do_baseline_3D) {
    uint baseline = (stat_D * (stat_D + 1) / 2) + stat_3;
    /* (*visibilities)[baseline][channel] = (fcomplex4) { vis_3D_r.x, vis_3D_i.x, vis_3D_r.y, vis_3D_i.y, vis_3D_r.z, vis_3D_i.z, vis_3D_r.w, vis_3D_i.w }; */
    (*visibilities)[baseline][channel][0][0] = make_float2(vis_3D_r.x, vis_3D_i.x);
    (*visibilities)[baseline][channel][0][1] = make_float2(vis_3D_r.y, vis_3D_i.y);
    (*visibilities)[baseline][channel][1][0] = make_float2(vis_3D_r.z, vis_3D_i.z);
    (*visibilities)[baseline][channel][1][1] = make_float2(vis_3D_r.w, vis_3D_i.w);
  }
}

} // extern "C"
