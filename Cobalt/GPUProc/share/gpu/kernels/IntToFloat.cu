//# IntToFloat.cu: Convert integer input to float; transpose time and pol dims
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
//# $Id: IntToFloat.cu 27000 2013-10-17 09:11:13Z loose $

#include "IntToFloat.cuh"

//#if NR_BITS_PER_SAMPLE   ==  4
//typedef char1  SampleType
#if NR_BITS_PER_SAMPLE ==  8
typedef char2  SampleType;
#elif NR_BITS_PER_SAMPLE == 16
typedef short2 SampleType;
#else
#error unsupported NR_BITS_PER_SAMPLE: must be 4, 8, or 16
#endif

typedef SampleType (*SampledDataType)  [NR_STATIONS][NR_SAMPLES_PER_SUBBAND][NR_POLARIZATIONS];
typedef float2     (*ConvertedDataType)[NR_STATIONS][NR_POLARIZATIONS][NR_SAMPLES_PER_SUBBAND];

/**
 * This kernel performs a conversion of the integer valued input to floats and
 * transposes the data to get per station: first all samples with polX, then polY.
 * - It supports 8 and 16 bit (char and short) input, which is selectable using
 *   the define NR_BITS_PER_SAMPLE
 * - In 8 bit mode the converted samples with value -128 are clamped to -127.0f
 *
 * @param[out] convertedDataPtr    pointer to output data of ConvertedDataType,
 *                                 a 4D array [station][polarizations][n_samples_subband][complex]
 *                                 of floats (2 complex polarizations).
 * @param[in]  sampledDataPtr      pointer to input data; this can either be a
 *                                 4D array [station][n_samples_subband][polarizations][complex]
 *                                 of shorts or chars, depending on NR_BITS_PER_SAMPLE.
 *
 * Required preprocessor symbols:
 * - NR_SAMPLES_PER_CHANNEL: > 0
 * - NR_BITS_PER_SAMPLE: 8 or 16
 *
 * Execution configuration:
 * - Use a 1D thread block. No restrictions.
 * - Use a 2D grid dim, where the x dim has 1 block and the y dim represents the
 *   number of stations (i.e. antenna fields).
 */

extern "C" {
__global__ void intToFloat(void *convertedDataPtr,
                           const void *sampledDataPtr)
{
  ConvertedDataType convertedData = (ConvertedDataType)convertedDataPtr;
  SampledDataType   sampledData   = (SampledDataType)  sampledDataPtr;

  uint station = blockIdx.y;

  for (uint time = threadIdx.x; time < NR_SAMPLES_PER_SUBBAND; time += blockDim.x)
  {
    float4 sample;
    sample = make_float4(convertIntToFloat((*sampledData)[station][time][0].x),
                         convertIntToFloat((*sampledData)[station][time][0].y),
                         convertIntToFloat((*sampledData)[station][time][1].x), 
                         convertIntToFloat((*sampledData)[station][time][1].y));

    float2 sampleX = make_float2(sample.x, sample.y);
    (*convertedData)[station][0][time] = sampleX;
    float2 sampleY = make_float2(sample.z, sample.w);
    (*convertedData)[station][1][time] = sampleY;
  }

}

}

