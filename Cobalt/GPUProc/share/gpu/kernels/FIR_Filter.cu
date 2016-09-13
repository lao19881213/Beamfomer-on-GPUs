//# FIR_Filter.cu
//# Copyright (C) 2013  ASTRON (Netherlands Institute for Radio Astronomy)
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
//# $Id: FIR_Filter.cu 27000 2013-10-17 09:11:13Z loose $

#include "IntToFloat.cuh"
#include <stdio.h>

#if !(NR_STABS >= 1)
#error Precondition violated: NR_STABS >= 1
#endif

#if !(NR_TAPS == 16)
#error Precondition violated: NR_TAPS == 16
#endif

#if !(NR_SUBBANDS > 0)
#error Precondition violated: NR_SUBBANDS > 0
#endif

#if !(NR_SAMPLES_PER_CHANNEL > 0 && NR_SAMPLES_PER_CHANNEL % NR_TAPS == 0)
#error Precondition violated: NR_SAMPLES_PER_CHANNEL > 0 && NR_SAMPLES_PER_CHANNEL % NR_TAPS == 0
#endif

#if NR_BITS_PER_SAMPLE == 16
typedef signed short SampleType;
#elif NR_BITS_PER_SAMPLE == 8
typedef signed char SampleType;
#else
#error Precondition violated: NR_BITS_PER_SAMPLE == 8 || NR_BITS_PER_SAMPLE == 16
#endif

#if NR_CHANNELS == 1
#warning TODO: NR_CHANNELS == 1 is not (yet) supported
#elif !(NR_CHANNELS > 0 && NR_CHANNELS % 16 == 0)
#error Precondition violated: NR_CHANNELS > 0 && NR_CHANNELS % 16 == 0
#endif

#if !(NR_POLARIZATIONS == 2)
#error Precondition violated: NR_POLARIZATIONS == 2
#endif

#if !(COMPLEX == 2)
#error Precondition violated: COMPLEX == 2
#endif

//# NR_STABS means #stations (correlator) or #TABs (beamformer).
typedef SampleType (*SampledDataType)[NR_STABS][NR_SAMPLES_PER_CHANNEL][NR_CHANNELS][NR_POLARIZATIONS * COMPLEX];
typedef SampleType (*HistoryDataType)[NR_SUBBANDS][NR_STABS][NR_TAPS - 1][NR_CHANNELS][NR_POLARIZATIONS * COMPLEX];
typedef float (*FilteredDataType)[NR_STABS][NR_POLARIZATIONS][NR_SAMPLES_PER_CHANNEL][NR_CHANNELS][COMPLEX];
typedef const float (*WeightsType)[NR_CHANNELS][NR_TAPS];


/*!
 * Applies the Finite Input Response filter defined by the weightsPtr array
 * to the sampledDataPtr array. Output is written into the filteredDataPtr
 * array. The filter works on complex numbers. The weights are real values only.
 *
 * Input values are first converted to (complex) float.
 * The kernel also reorders the polarization dimension and expects the weights
 * per channel in reverse order. If an FFT is applied afterwards, the weights
 * of the odd channels are often supplied negated to get the resulting channels
 * in increasing order of frequency.
 *
 * \param[out] filteredDataPtr         4D output array of floats
 * \param[in]  sampledDataPtr          4D input array of signed chars or shorts
 * \param[in]  weightsPtr              2D per-channel FIR filter coefficient array of floats (considering float16 as a dim)
 * \param[in]  historyDataPtr          5D input array of history input samples needed to initialize the FIR filter
 * \param[in]  subbandIdx              index of the subband to process
 *
 * Pre-processor input symbols (some are tied to the execution configuration)
 * Symbol                  | Valid Values                | Description
 * ----------------------- | --------------------------- | -----------
 * NR_STABS                | >= 1                        | number of antenna fields (correlator), or number of tight array beams (tabs) (beamformer)
 * NR_TAPS                 | 16                          | number of FIR filtering coefficients
 * NR_SAMPLES_PER_CHANNEL  | multiple of NR_TAPS and > 0 | number of input samples per channel
 * NR_BITS_PER_SAMPLE      | 8 or 16                     | number of bits of signed integral value type of sampledDataPtr (TODO: support 4)
 * NR_CHANNELS             | multiple of 16 and > 0      | number of frequency channels per subband
 * NR_POLARIZATIONS        | 2                           | number of polarizations
 * COMPLEX                 | 2                           | size of complex in number of floats/doubles
 *
 * Execution configuration: (TODO: enforce using __attribute__ reqd_work_group_size)
 * - Work dim == 2  (can be 1 iff NR_STABS == 1)
 *     + Inner dim: the channel, pol, real/imag the thread processes
 *     + Outer dim: the station the thread processes
 * - Work group size: must divide global size, no other kernel restrictions
 * - Global size: (NR_CHANNELS * NR_POLARIZATIONS * 2, NR_STABS)
 *
 * TODO: convert complex dim to fcomplex (=float2 in math.cl) in device code and to complex<float> in host code.
 */
extern "C" {
__global__ void FIR_filter( void *filteredDataPtr,
                            const void *sampledDataPtr,
                            const void *weightsPtr,
                            void *historyDataPtr,
                            unsigned subbandIdx)
{
  SampledDataType sampledData = (SampledDataType) sampledDataPtr;
  FilteredDataType filteredData = (FilteredDataType) filteredDataPtr;
  WeightsType weightsData = (WeightsType) weightsPtr;
  HistoryDataType historyData = (HistoryDataType) historyDataPtr;

  unsigned cpr = blockIdx.x*blockDim.x+threadIdx.x;
#if 0
  //# Straight index calc for NR_CHANNELS == 1
  uint pol_ri = cpr & 3;
  uint channel = cpr >> 2;
  uint ri = cpr & 1;
  uint pol = pol_ri >> 1;
#else
  unsigned ri = cpr & 1;                        // index (real/imag) in output data
  unsigned channel = (cpr >> 1) % NR_CHANNELS;  // index in input & output data
  unsigned pol = (cpr >> 1) / NR_CHANNELS;      // index (polarization) in output data
  unsigned pol_ri = (pol << 1) | ri;            // index (polarization & real/imag) in input data
#endif
  unsigned station = blockIdx.y;

  //# const float16 weights = (*weightsData)[channel];
  const float weights_s0 = (*weightsData)[channel][0];
  const float weights_s1 = (*weightsData)[channel][1];
  const float weights_s2 = (*weightsData)[channel][2];
  const float weights_s3 = (*weightsData)[channel][3];
  const float weights_s4 = (*weightsData)[channel][4];
  const float weights_s5 = (*weightsData)[channel][5];
  const float weights_s6 = (*weightsData)[channel][6];
  const float weights_s7 = (*weightsData)[channel][7];
  const float weights_s8 = (*weightsData)[channel][8];
  const float weights_s9 = (*weightsData)[channel][9];
  const float weights_sA = (*weightsData)[channel][10];
  const float weights_sB = (*weightsData)[channel][11];
  const float weights_sC = (*weightsData)[channel][12];
  const float weights_sD = (*weightsData)[channel][13];
  const float weights_sE = (*weightsData)[channel][14];
  const float weights_sF = (*weightsData)[channel][15];

  //# float16 delayLine;
  float delayLine_s0, delayLine_s1, delayLine_s2, delayLine_s3, 
        delayLine_s4, delayLine_s5, delayLine_s6, delayLine_s7, 
        delayLine_s8, delayLine_s9, delayLine_sA, delayLine_sB,
        delayLine_sC, delayLine_sD, delayLine_sE, delayLine_sF;
  
  delayLine_s0 = convertIntToFloat((*historyData)[subbandIdx][station][0][channel][pol_ri]);
  delayLine_s1 = convertIntToFloat((*historyData)[subbandIdx][station][1][channel][pol_ri]);
  delayLine_s2 = convertIntToFloat((*historyData)[subbandIdx][station][2][channel][pol_ri]);
  delayLine_s3 = convertIntToFloat((*historyData)[subbandIdx][station][3][channel][pol_ri]);
  delayLine_s4 = convertIntToFloat((*historyData)[subbandIdx][station][4][channel][pol_ri]);
  delayLine_s5 = convertIntToFloat((*historyData)[subbandIdx][station][5][channel][pol_ri]);
  delayLine_s6 = convertIntToFloat((*historyData)[subbandIdx][station][6][channel][pol_ri]);
  delayLine_s7 = convertIntToFloat((*historyData)[subbandIdx][station][7][channel][pol_ri]);
  delayLine_s8 = convertIntToFloat((*historyData)[subbandIdx][station][8][channel][pol_ri]);
  delayLine_s9 = convertIntToFloat((*historyData)[subbandIdx][station][9][channel][pol_ri]);
  delayLine_sA = convertIntToFloat((*historyData)[subbandIdx][station][10][channel][pol_ri]);
  delayLine_sB = convertIntToFloat((*historyData)[subbandIdx][station][11][channel][pol_ri]);
  delayLine_sC = convertIntToFloat((*historyData)[subbandIdx][station][12][channel][pol_ri]);
  delayLine_sD = convertIntToFloat((*historyData)[subbandIdx][station][13][channel][pol_ri]);
  delayLine_sE = convertIntToFloat((*historyData)[subbandIdx][station][14][channel][pol_ri]);

  float sum_s0, sum_s1, sum_s2, sum_s3,
        sum_s4, sum_s5, sum_s6, sum_s7,
        sum_s8, sum_s9, sum_sA, sum_sB,
        sum_sC, sum_sD, sum_sE, sum_sF;

  for (unsigned time = 0; time < NR_SAMPLES_PER_CHANNEL; time += NR_TAPS) 
  {
    delayLine_sF = convertIntToFloat((*sampledData)[station][time + 0][channel][pol_ri]);
    sum_s0 = weights_sF * delayLine_s0;
    delayLine_s0 = convertIntToFloat((*sampledData)[station][time + 1][channel][pol_ri]);
    sum_s0 += weights_sE * delayLine_s1;
    sum_s0 += weights_sD * delayLine_s2;
    sum_s0 += weights_sC * delayLine_s3;
    sum_s0 += weights_sB * delayLine_s4;
    sum_s0 += weights_sA * delayLine_s5;
    sum_s0 += weights_s9 * delayLine_s6;
    sum_s0 += weights_s8 * delayLine_s7;
    sum_s0 += weights_s7 * delayLine_s8;
    sum_s0 += weights_s6 * delayLine_s9;
    sum_s0 += weights_s5 * delayLine_sA;
    sum_s0 += weights_s4 * delayLine_sB;
    sum_s0 += weights_s3 * delayLine_sC;
    sum_s0 += weights_s2 * delayLine_sD;
    sum_s0 += weights_s1 * delayLine_sE;
    sum_s0 += weights_s0 * delayLine_sF;
    (*filteredData)[station][pol][time + 0][channel][ri] = sum_s0;

    sum_s1 = weights_sF * delayLine_s1;
    delayLine_s1 = convertIntToFloat((*sampledData)[station][time + 2][channel][pol_ri]);
    sum_s1 += weights_sE * delayLine_s2;
    sum_s1 += weights_sD * delayLine_s3;
    sum_s1 += weights_sC * delayLine_s4;
    sum_s1 += weights_sB * delayLine_s5;
    sum_s1 += weights_sA * delayLine_s6;
    sum_s1 += weights_s9 * delayLine_s7;
    sum_s1 += weights_s8 * delayLine_s8;
    sum_s1 += weights_s7 * delayLine_s9;
    sum_s1 += weights_s6 * delayLine_sA;
    sum_s1 += weights_s5 * delayLine_sB;
    sum_s1 += weights_s4 * delayLine_sC;
    sum_s1 += weights_s3 * delayLine_sD;
    sum_s1 += weights_s2 * delayLine_sE;
    sum_s1 += weights_s1 * delayLine_sF;
    sum_s1 += weights_s0 * delayLine_s0;
    (*filteredData)[station][pol][time + 1][channel][ri] = sum_s1;

    sum_s2 = weights_sF * delayLine_s2;
    delayLine_s2 = convertIntToFloat((*sampledData)[station][time + 3][channel][pol_ri]);
    sum_s2 += weights_sE * delayLine_s3;
    sum_s2 += weights_sD * delayLine_s4;
    sum_s2 += weights_sC * delayLine_s5;
    sum_s2 += weights_sB * delayLine_s6;
    sum_s2 += weights_sA * delayLine_s7;
    sum_s2 += weights_s9 * delayLine_s8;
    sum_s2 += weights_s8 * delayLine_s9;
    sum_s2 += weights_s7 * delayLine_sA;
    sum_s2 += weights_s6 * delayLine_sB;
    sum_s2 += weights_s5 * delayLine_sC;
    sum_s2 += weights_s4 * delayLine_sD;
    sum_s2 += weights_s3 * delayLine_sE;
    sum_s2 += weights_s2 * delayLine_sF;
    sum_s2 += weights_s1 * delayLine_s0;
    sum_s2 += weights_s0 * delayLine_s1;
    (*filteredData)[station][pol][time + 2][channel][ri] = sum_s2;

    sum_s3 = weights_sF * delayLine_s3;
    delayLine_s3 = convertIntToFloat((*sampledData)[station][time + 4][channel][pol_ri]);
    sum_s3 += weights_sE * delayLine_s4;
    sum_s3 += weights_sD * delayLine_s5;
    sum_s3 += weights_sC * delayLine_s6;
    sum_s3 += weights_sB * delayLine_s7;
    sum_s3 += weights_sA * delayLine_s8;
    sum_s3 += weights_s9 * delayLine_s9;
    sum_s3 += weights_s8 * delayLine_sA;
    sum_s3 += weights_s7 * delayLine_sB;
    sum_s3 += weights_s6 * delayLine_sC;
    sum_s3 += weights_s5 * delayLine_sD;
    sum_s3 += weights_s4 * delayLine_sE;
    sum_s3 += weights_s3 * delayLine_sF;
    sum_s3 += weights_s2 * delayLine_s0;
    sum_s3 += weights_s1 * delayLine_s1;
    sum_s3 += weights_s0 * delayLine_s2;
    (*filteredData)[station][pol][time + 3][channel][ri] = sum_s3;

    sum_s4 = weights_sF * delayLine_s4;
    delayLine_s4 = convertIntToFloat((*sampledData)[station][time + 5][channel][pol_ri]);
    sum_s4 += weights_sE * delayLine_s5;
    sum_s4 += weights_sD * delayLine_s6;
    sum_s4 += weights_sC * delayLine_s7;
    sum_s4 += weights_sB * delayLine_s8;
    sum_s4 += weights_sA * delayLine_s9;
    sum_s4 += weights_s9 * delayLine_sA;
    sum_s4 += weights_s8 * delayLine_sB;
    sum_s4 += weights_s7 * delayLine_sC;
    sum_s4 += weights_s6 * delayLine_sD;
    sum_s4 += weights_s5 * delayLine_sE;
    sum_s4 += weights_s4 * delayLine_sF;
    sum_s4 += weights_s3 * delayLine_s0;
    sum_s4 += weights_s2 * delayLine_s1;
    sum_s4 += weights_s1 * delayLine_s2;
    sum_s4 += weights_s0 * delayLine_s3;
    (*filteredData)[station][pol][time + 4][channel][ri] = sum_s4;

    sum_s5 = weights_sF * delayLine_s5;
    delayLine_s5 = convertIntToFloat((*sampledData)[station][time + 6][channel][pol_ri]);
    sum_s5 += weights_sE * delayLine_s6;
    sum_s5 += weights_sD * delayLine_s7;
    sum_s5 += weights_sC * delayLine_s8;
    sum_s5 += weights_sB * delayLine_s9;
    sum_s5 += weights_sA * delayLine_sA;
    sum_s5 += weights_s9 * delayLine_sB;
    sum_s5 += weights_s8 * delayLine_sC;
    sum_s5 += weights_s7 * delayLine_sD;
    sum_s5 += weights_s6 * delayLine_sE;
    sum_s5 += weights_s5 * delayLine_sF;
    sum_s5 += weights_s4 * delayLine_s0;
    sum_s5 += weights_s3 * delayLine_s1;
    sum_s5 += weights_s2 * delayLine_s2;
    sum_s5 += weights_s1 * delayLine_s3;
    sum_s5 += weights_s0 * delayLine_s4;
    (*filteredData)[station][pol][time + 5][channel][ri] = sum_s5;

    sum_s6 = weights_sF * delayLine_s6;
    delayLine_s6 = convertIntToFloat((*sampledData)[station][time + 7][channel][pol_ri]);
    sum_s6 += weights_sE * delayLine_s7;
    sum_s6 += weights_sD * delayLine_s8;
    sum_s6 += weights_sC * delayLine_s9;
    sum_s6 += weights_sB * delayLine_sA;
    sum_s6 += weights_sA * delayLine_sB;
    sum_s6 += weights_s9 * delayLine_sC;
    sum_s6 += weights_s8 * delayLine_sD;
    sum_s6 += weights_s7 * delayLine_sE;
    sum_s6 += weights_s6 * delayLine_sF;
    sum_s6 += weights_s5 * delayLine_s0;
    sum_s6 += weights_s4 * delayLine_s1;
    sum_s6 += weights_s3 * delayLine_s2;
    sum_s6 += weights_s2 * delayLine_s3;
    sum_s6 += weights_s1 * delayLine_s4;
    sum_s6 += weights_s0 * delayLine_s5;
    (*filteredData)[station][pol][time + 6][channel][ri] = sum_s6;

    sum_s7 = weights_sF * delayLine_s7;
    delayLine_s7 = convertIntToFloat((*sampledData)[station][time + 8][channel][pol_ri]);
    sum_s7 += weights_sE * delayLine_s8;
    sum_s7 += weights_sD * delayLine_s9;
    sum_s7 += weights_sC * delayLine_sA;
    sum_s7 += weights_sB * delayLine_sB;
    sum_s7 += weights_sA * delayLine_sC;
    sum_s7 += weights_s9 * delayLine_sD;
    sum_s7 += weights_s8 * delayLine_sE;
    sum_s7 += weights_s7 * delayLine_sF;
    sum_s7 += weights_s6 * delayLine_s0;
    sum_s7 += weights_s5 * delayLine_s1;
    sum_s7 += weights_s4 * delayLine_s2;
    sum_s7 += weights_s3 * delayLine_s3;
    sum_s7 += weights_s2 * delayLine_s4;
    sum_s7 += weights_s1 * delayLine_s5;
    sum_s7 += weights_s0 * delayLine_s6;
    (*filteredData)[station][pol][time + 7][channel][ri] = sum_s7;

    sum_s8 = weights_sF * delayLine_s8;
    delayLine_s8 = convertIntToFloat((*sampledData)[station][time + 9][channel][pol_ri]);
    sum_s8 += weights_sE * delayLine_s9;
    sum_s8 += weights_sD * delayLine_sA;
    sum_s8 += weights_sC * delayLine_sB;
    sum_s8 += weights_sB * delayLine_sC;
    sum_s8 += weights_sA * delayLine_sD;
    sum_s8 += weights_s9 * delayLine_sE;
    sum_s8 += weights_s8 * delayLine_sF;
    sum_s8 += weights_s7 * delayLine_s0;
    sum_s8 += weights_s6 * delayLine_s1;
    sum_s8 += weights_s5 * delayLine_s2;
    sum_s8 += weights_s4 * delayLine_s3;
    sum_s8 += weights_s3 * delayLine_s4;
    sum_s8 += weights_s2 * delayLine_s5;
    sum_s8 += weights_s1 * delayLine_s6;
    sum_s8 += weights_s0 * delayLine_s7;
    (*filteredData)[station][pol][time + 8][channel][ri] = sum_s8;

    sum_s9 = weights_sF * delayLine_s9;
    delayLine_s9 = convertIntToFloat((*sampledData)[station][time + 10][channel][pol_ri]);
    sum_s9 += weights_sE * delayLine_sA;
    sum_s9 += weights_sD * delayLine_sB;
    sum_s9 += weights_sC * delayLine_sC;
    sum_s9 += weights_sB * delayLine_sD;
    sum_s9 += weights_sA * delayLine_sE;
    sum_s9 += weights_s9 * delayLine_sF;
    sum_s9 += weights_s8 * delayLine_s0;
    sum_s9 += weights_s7 * delayLine_s1;
    sum_s9 += weights_s6 * delayLine_s2;
    sum_s9 += weights_s5 * delayLine_s3;
    sum_s9 += weights_s4 * delayLine_s4;
    sum_s9 += weights_s3 * delayLine_s5;
    sum_s9 += weights_s2 * delayLine_s6;
    sum_s9 += weights_s1 * delayLine_s7;
    sum_s9 += weights_s0 * delayLine_s8;
    (*filteredData)[station][pol][time + 9][channel][ri] = sum_s9;

    sum_sA = weights_sF * delayLine_sA;
    delayLine_sA = convertIntToFloat((*sampledData)[station][time + 11][channel][pol_ri]);
    sum_sA += weights_sE * delayLine_sB;
    sum_sA += weights_sD * delayLine_sC;
    sum_sA += weights_sC * delayLine_sD;
    sum_sA += weights_sB * delayLine_sE;
    sum_sA += weights_sA * delayLine_sF;
    sum_sA += weights_s9 * delayLine_s0;
    sum_sA += weights_s8 * delayLine_s1;
    sum_sA += weights_s7 * delayLine_s2;
    sum_sA += weights_s6 * delayLine_s3;
    sum_sA += weights_s5 * delayLine_s4;
    sum_sA += weights_s4 * delayLine_s5;
    sum_sA += weights_s3 * delayLine_s6;
    sum_sA += weights_s2 * delayLine_s7;
    sum_sA += weights_s1 * delayLine_s8;
    sum_sA += weights_s0 * delayLine_s9;
    (*filteredData)[station][pol][time + 10][channel][ri] = sum_sA;

    sum_sB = weights_sF * delayLine_sB;
    delayLine_sB = convertIntToFloat((*sampledData)[station][time + 12][channel][pol_ri]);
    sum_sB += weights_sE * delayLine_sC;
    sum_sB += weights_sD * delayLine_sD;
    sum_sB += weights_sC * delayLine_sE;
    sum_sB += weights_sB * delayLine_sF;
    sum_sB += weights_sA * delayLine_s0;
    sum_sB += weights_s9 * delayLine_s1;
    sum_sB += weights_s8 * delayLine_s2;
    sum_sB += weights_s7 * delayLine_s3;
    sum_sB += weights_s6 * delayLine_s4;
    sum_sB += weights_s5 * delayLine_s5;
    sum_sB += weights_s4 * delayLine_s6;
    sum_sB += weights_s3 * delayLine_s7;
    sum_sB += weights_s2 * delayLine_s8;
    sum_sB += weights_s1 * delayLine_s9;
    sum_sB += weights_s0 * delayLine_sA;
    (*filteredData)[station][pol][time + 11][channel][ri] = sum_sB;

    sum_sC = weights_sF * delayLine_sC;
    delayLine_sC = convertIntToFloat((*sampledData)[station][time + 13][channel][pol_ri]);
    sum_sC += weights_sE * delayLine_sD;
    sum_sC += weights_sD * delayLine_sE;
    sum_sC += weights_sC * delayLine_sF;
    sum_sC += weights_sB * delayLine_s0;
    sum_sC += weights_sA * delayLine_s1;
    sum_sC += weights_s9 * delayLine_s2;
    sum_sC += weights_s8 * delayLine_s3;
    sum_sC += weights_s7 * delayLine_s4;
    sum_sC += weights_s6 * delayLine_s5;
    sum_sC += weights_s5 * delayLine_s6;
    sum_sC += weights_s4 * delayLine_s7;
    sum_sC += weights_s3 * delayLine_s8;
    sum_sC += weights_s2 * delayLine_s9;
    sum_sC += weights_s1 * delayLine_sA;
    sum_sC += weights_s0 * delayLine_sB;
    (*filteredData)[station][pol][time + 12][channel][ri] = sum_sC;

    sum_sD = weights_sF * delayLine_sD;
    delayLine_sD = convertIntToFloat((*sampledData)[station][time + 14][channel][pol_ri]);
    sum_sD += weights_sE * delayLine_sE;
    sum_sD += weights_sD * delayLine_sF;
    sum_sD += weights_sC * delayLine_s0;
    sum_sD += weights_sB * delayLine_s1;
    sum_sD += weights_sA * delayLine_s2;
    sum_sD += weights_s9 * delayLine_s3;
    sum_sD += weights_s8 * delayLine_s4;
    sum_sD += weights_s7 * delayLine_s5;
    sum_sD += weights_s6 * delayLine_s6;
    sum_sD += weights_s5 * delayLine_s7;
    sum_sD += weights_s4 * delayLine_s8;
    sum_sD += weights_s3 * delayLine_s9;
    sum_sD += weights_s2 * delayLine_sA;
    sum_sD += weights_s1 * delayLine_sB;
    sum_sD += weights_s0 * delayLine_sC;
    (*filteredData)[station][pol][time + 13][channel][ri] = sum_sD;

    sum_sE = weights_sF * delayLine_sE;
    delayLine_sE = convertIntToFloat((*sampledData)[station][time + 15][channel][pol_ri]);
    sum_sE += weights_sE * delayLine_sF;
    sum_sE += weights_sD * delayLine_s0;
    sum_sE += weights_sC * delayLine_s1;
    sum_sE += weights_sB * delayLine_s2;
    sum_sE += weights_sA * delayLine_s3;
    sum_sE += weights_s9 * delayLine_s4;
    sum_sE += weights_s8 * delayLine_s5;
    sum_sE += weights_s7 * delayLine_s6;
    sum_sE += weights_s6 * delayLine_s7;
    sum_sE += weights_s5 * delayLine_s8;
    sum_sE += weights_s4 * delayLine_s9;
    sum_sE += weights_s3 * delayLine_sA;
    sum_sE += weights_s2 * delayLine_sB;
    sum_sE += weights_s1 * delayLine_sC;
    sum_sE += weights_s0 * delayLine_sD;
    (*filteredData)[station][pol][time + 14][channel][ri] = sum_sE;

    sum_sF = weights_sF * delayLine_sF;
    sum_sF += weights_sE * delayLine_s0;
    sum_sF += weights_sD * delayLine_s1;
    sum_sF += weights_sC * delayLine_s2;
    sum_sF += weights_sB * delayLine_s3;
    sum_sF += weights_sA * delayLine_s4;
    sum_sF += weights_s9 * delayLine_s5;
    sum_sF += weights_s8 * delayLine_s6;
    sum_sF += weights_s7 * delayLine_s7;
    sum_sF += weights_s6 * delayLine_s8;
    sum_sF += weights_s5 * delayLine_s9;
    sum_sF += weights_s4 * delayLine_sA;
    sum_sF += weights_s3 * delayLine_sB;
    sum_sF += weights_s2 * delayLine_sC;
    sum_sF += weights_s1 * delayLine_sD;
    sum_sF += weights_s0 * delayLine_sE;
    (*filteredData)[station][pol][time + 15][channel][ri] = sum_sF;
  }

  for (unsigned time = 0; time < NR_TAPS - 1; time++)
  {
    (*historyData)[subbandIdx][station][time][channel][pol_ri] =
      (*sampledData)[station][NR_SAMPLES_PER_CHANNEL - (NR_TAPS - 1) + time][channel][pol_ri];
  }
}
}
