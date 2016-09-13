//# Transpose.cu
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
//# $Id: Transpose.cu 27000 2013-10-17 09:11:13Z loose $

/*!
 * Performs data transposition from the output of the beamformer kernel to
 * a data order suitable for an inverse FFT.
 * Parallelisation is performed over the TABs and number of samples (time).
 *
 * We have 4 dimensions, but CUDA thread blocks can be up to three.
 * Mangle the TAB and sample dimension in to dim 0 (x).
 *
 * The kernel needs to determine for each thread whether to read and separately
 * whether to write back a sample, because the number of TABs may not divide by
 * the 16x16 thread arrangement (even though we have a 1D thread block).
 *
 * \param[out] TransposedDataType      4D output array of samples. For each TAB and pol, a spectrum per time step of complex floats.
 * \param[in]  ComplexVoltagesType     3D input array of samples (last dim (pol) is implicit). For each channel, the TABs per time step of two complex floats.
 *
 * Pre-processor input symbols (some are tied to the execution configuration)
 * Symbol                  | Valid Values            | Description
 * ----------------------- | ----------------------- | -----------
 * NR_SAMPLES_PER_CHANNEL  | multiple of 16 and > 0  | number of input samples per channel
 * NR_CHANNELS             | >= 1                    | number of frequency channels per subband
 * NR_TABS                 | >= 1                    | number of Tied Array Beams to create
 *
 * Note that this kernel assumes  NR_POLARIZATIONS == 2
 *
 * Execution configuration:
 * - LocalWorkSize = 1 dimensional; (256, 1, 1) is in use. Multiples of (32, 1, 1) may work too.
 * - GlobalWorkSize = 3 dimensional:
 *   + inner dim (x): always 1 block
 *   + middle dim (y): 16 TABs can be processed in a block. Number of blocks required, rounded-up. eg for 17 tabs we need 2 blocks
 *   + outer dim (z): 16 samples per channel can be processed in a block. Number of blocks required (fits exactly). 32 channels is 2 blocks
 */

typedef float2 (*TransposedDataType)[NR_TABS][NR_POLARIZATIONS][NR_CHANNELS][NR_SAMPLES_PER_CHANNEL];

// last dim within float4 is NR_POLARIZATIONS
typedef float4 (*ComplexVoltagesType)[NR_CHANNELS][NR_SAMPLES_PER_CHANNEL][NR_TABS]; // [NR_POLARIZATIONS];

extern "C"
__global__ void transpose(void *transposedDataPtr,
                          const void *complexVoltagesPtr)
{
  TransposedDataType transposedData = (TransposedDataType) transposedDataPtr;
  ComplexVoltagesType complexVoltages = (ComplexVoltagesType) complexVoltagesPtr;

  // last dim within float4 is NR_POLARIZATIONS
  __shared__ float4 tmp[16][17]; // padding to avoid bank conflicts

  unsigned tabBase = 16 * blockDim.y * blockIdx.y + threadIdx.y;
  unsigned timeBase = 16 * blockDim.z * blockIdx.z + threadIdx.z;

  unsigned tabOffsetR = threadIdx.x & 15;
  unsigned tabR = tabBase + tabOffsetR;
  unsigned timeOffsetR = threadIdx.x >> 4;
  unsigned timeR = timeBase + timeOffsetR;
  bool doR = NR_TABS % 16 == 0 || tabR < NR_TABS;

  unsigned tabOffsetW = threadIdx.x >> 4;
  unsigned tabW = tabBase + tabOffsetW;
  unsigned timeOffsetW = threadIdx.x & 15;
  unsigned timeW = timeBase + timeOffsetW;
  bool doW = NR_TABS % 16 == 0 || tabW < NR_TABS;

  for (int channel = 0; channel < NR_CHANNELS; channel++) 
  {
    if (doR)
    {
      tmp[tabOffsetR][timeOffsetR] = (*complexVoltages)[channel][timeR][tabR];
    }

    __syncthreads();

    if (doW) {
      float4 sample = tmp[tabOffsetW][timeOffsetW];
      (*transposedData)[tabW][0][channel][timeW] = make_float2(sample.x, sample.y);
      (*transposedData)[tabW][1][channel][timeW] = make_float2(sample.z, sample.w);
    }

    __syncthreads();
  }
}
