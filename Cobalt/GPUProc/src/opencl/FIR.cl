//# FIR.cl
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
//# $Id: FIR.cl 24849 2013-05-08 14:51:06Z amesfoort $

#define COMPLEX 2       // do not change

#if NR_BITS_PER_SAMPLE == 16
typedef short SampleType;
#elif NR_BITS_PER_SAMPLE == 8
typedef char SampleType;
#else
#error unsupported NR_BITS_PER_SAMPLE
#endif

typedef __global SampleType (*SampledDataType)[NR_STATIONS][NR_TAPS - 1 + NR_SAMPLES_PER_CHANNEL][NR_CHANNELS][NR_POLARIZATIONS * COMPLEX];
typedef __global float (*FilteredDataType)[NR_STATIONS][NR_POLARIZATIONS][NR_SAMPLES_PER_CHANNEL][NR_CHANNELS][COMPLEX];
typedef __global const float16 (*WeightsType)[NR_CHANNELS];


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
 *
 * Pre-processor input symbols (some are tied to the execution configuration)
 * Symbol                  | Valid Values                | Description
 * ----------------------- | --------------------------- | -----------
 * NR_STATIONS             | >= 1                        | number of antenna fields
 * NR_TAPS                 | 1--16                       | number of FIR filtering coefficients
 * NR_SAMPLES_PER_CHANNEL  | multiple of NR_TAPS and > 0 | number of input samples per channel
 * NR_BITS_PER_SAMPLE      | 8 or 16                     | number of bits of signed integral value type of sampledDataPtr (TODO: support 4)
 * NR_CHANNELS             | multiple of 16 and > 0      | number of frequency channels per subband
 * NR_POLARIZATIONS        | power of 2                  | number of polarizations
 *
 * Execution configuration: (TODO: enforce using __attribute__ reqd_work_group_size)
 * - Work dim == 2  (can be 1 iff NR_STATIONS == 1)
 *     + Inner dim: the channel, pol, real/imag the thread processes
 *     + Outer dim: the station the thread processes
 * - Work group size: must divide global size, no other kernel restrictions
 * - Global size: (NR_CHANNELS * NR_POLARIZATIONS * 2, NR_STATIONS)
 *
 * TODO: convert complex dim to fcomplex (=float2 in math.cl) in device code and to complex<float> in host code.
 */
__kernel void FIR_filter(__global void *filteredDataPtr,
                         __global const void *sampledDataPtr,
                         __global const void *weightsPtr)
{
  SampledDataType sampledData = (SampledDataType) sampledDataPtr;
  FilteredDataType filteredData = (FilteredDataType) filteredDataPtr;
  WeightsType weightsData = (WeightsType) weightsPtr;

  uint cpr = get_global_id(0);
#if 0
  // Straight index calc for NR_CHANNELS == 1
  uint pol_ri = cpr & 3;
  uint channel = cpr >> 2;
  uint ri = cpr & 1;
  uint pol = pol_ri >> 1;
#else
  uint ri = cpr & 1;
  uint channel = (cpr >> 1) % NR_CHANNELS;
  uint pol = (cpr >> 1) / NR_CHANNELS;
  uint pol_ri = (pol << 1) | ri;
#endif
  uint station = get_global_id(1);

  //#pragma OPENCL EXTENSION cl_amd_printf : enable

  const float16 weights = (*weightsData)[channel];
  float16 delayLine;
  float16 sum;

  delayLine.s0 = convert_float((*sampledData)[station][0][channel][pol_ri]);
  delayLine.s1 = convert_float((*sampledData)[station][1][channel][pol_ri]);
  delayLine.s2 = convert_float((*sampledData)[station][2][channel][pol_ri]);
  delayLine.s3 = convert_float((*sampledData)[station][3][channel][pol_ri]);
  delayLine.s4 = convert_float((*sampledData)[station][4][channel][pol_ri]);
  delayLine.s5 = convert_float((*sampledData)[station][5][channel][pol_ri]);
  delayLine.s6 = convert_float((*sampledData)[station][6][channel][pol_ri]);
  delayLine.s7 = convert_float((*sampledData)[station][7][channel][pol_ri]);
  delayLine.s8 = convert_float((*sampledData)[station][8][channel][pol_ri]);
  delayLine.s9 = convert_float((*sampledData)[station][9][channel][pol_ri]);
  delayLine.sA = convert_float((*sampledData)[station][10][channel][pol_ri]);
  delayLine.sB = convert_float((*sampledData)[station][11][channel][pol_ri]);
  delayLine.sC = convert_float((*sampledData)[station][12][channel][pol_ri]);
  delayLine.sD = convert_float((*sampledData)[station][13][channel][pol_ri]);
  delayLine.sE = convert_float((*sampledData)[station][14][channel][pol_ri]);

  for (uint time = 0; time < NR_SAMPLES_PER_CHANNEL; time += NR_TAPS) {
    delayLine.sF = convert_float((*sampledData)[station][time + NR_TAPS - 1 + 0][channel][pol_ri]);
    sum.s0 = weights.sF * delayLine.s0;
    delayLine.s0 = convert_float((*sampledData)[station][time + NR_TAPS - 1 + 1][channel][pol_ri]);
    sum.s0 += weights.sE * delayLine.s1;
    sum.s0 += weights.sD * delayLine.s2;
    sum.s0 += weights.sC * delayLine.s3;
    sum.s0 += weights.sB * delayLine.s4;
    sum.s0 += weights.sA * delayLine.s5;
    sum.s0 += weights.s9 * delayLine.s6;
    sum.s0 += weights.s8 * delayLine.s7;
    sum.s0 += weights.s7 * delayLine.s8;
    sum.s0 += weights.s6 * delayLine.s9;
    sum.s0 += weights.s5 * delayLine.sA;
    sum.s0 += weights.s4 * delayLine.sB;
    sum.s0 += weights.s3 * delayLine.sC;
    sum.s0 += weights.s2 * delayLine.sD;
    sum.s0 += weights.s1 * delayLine.sE;
    sum.s0 += weights.s0 * delayLine.sF;
    (*filteredData)[station][pol][time + 0][channel][ri] = sum.s0;

    sum.s1 = weights.sF * delayLine.s1;
    delayLine.s1 = convert_float((*sampledData)[station][time + NR_TAPS - 1 + 2][channel][pol_ri]);
    sum.s1 += weights.sE * delayLine.s2;
    sum.s1 += weights.sD * delayLine.s3;
    sum.s1 += weights.sC * delayLine.s4;
    sum.s1 += weights.sB * delayLine.s5;
    sum.s1 += weights.sA * delayLine.s6;
    sum.s1 += weights.s9 * delayLine.s7;
    sum.s1 += weights.s8 * delayLine.s8;
    sum.s1 += weights.s7 * delayLine.s9;
    sum.s1 += weights.s6 * delayLine.sA;
    sum.s1 += weights.s5 * delayLine.sB;
    sum.s1 += weights.s4 * delayLine.sC;
    sum.s1 += weights.s3 * delayLine.sD;
    sum.s1 += weights.s2 * delayLine.sE;
    sum.s1 += weights.s1 * delayLine.sF;
    sum.s1 += weights.s0 * delayLine.s0;
    (*filteredData)[station][pol][time + 1][channel][ri] = sum.s1;

    sum.s2 = weights.sF * delayLine.s2;
    delayLine.s2 = convert_float((*sampledData)[station][time + NR_TAPS - 1 + 3][channel][pol_ri]);
    sum.s2 += weights.sE * delayLine.s3;
    sum.s2 += weights.sD * delayLine.s4;
    sum.s2 += weights.sC * delayLine.s5;
    sum.s2 += weights.sB * delayLine.s6;
    sum.s2 += weights.sA * delayLine.s7;
    sum.s2 += weights.s9 * delayLine.s8;
    sum.s2 += weights.s8 * delayLine.s9;
    sum.s2 += weights.s7 * delayLine.sA;
    sum.s2 += weights.s6 * delayLine.sB;
    sum.s2 += weights.s5 * delayLine.sC;
    sum.s2 += weights.s4 * delayLine.sD;
    sum.s2 += weights.s3 * delayLine.sE;
    sum.s2 += weights.s2 * delayLine.sF;
    sum.s2 += weights.s1 * delayLine.s0;
    sum.s2 += weights.s0 * delayLine.s1;
    (*filteredData)[station][pol][time + 2][channel][ri] = sum.s2;

    sum.s3 = weights.sF * delayLine.s3;
    delayLine.s3 = convert_float((*sampledData)[station][time + NR_TAPS - 1 + 4][channel][pol_ri]);
    sum.s3 += weights.sE * delayLine.s4;
    sum.s3 += weights.sD * delayLine.s5;
    sum.s3 += weights.sC * delayLine.s6;
    sum.s3 += weights.sB * delayLine.s7;
    sum.s3 += weights.sA * delayLine.s8;
    sum.s3 += weights.s9 * delayLine.s9;
    sum.s3 += weights.s8 * delayLine.sA;
    sum.s3 += weights.s7 * delayLine.sB;
    sum.s3 += weights.s6 * delayLine.sC;
    sum.s3 += weights.s5 * delayLine.sD;
    sum.s3 += weights.s4 * delayLine.sE;
    sum.s3 += weights.s3 * delayLine.sF;
    sum.s3 += weights.s2 * delayLine.s0;
    sum.s3 += weights.s1 * delayLine.s1;
    sum.s3 += weights.s0 * delayLine.s2;
    (*filteredData)[station][pol][time + 3][channel][ri] = sum.s3;

    sum.s4 = weights.sF * delayLine.s4;
    delayLine.s4 = convert_float((*sampledData)[station][time + NR_TAPS - 1 + 5][channel][pol_ri]);
    sum.s4 += weights.sE * delayLine.s5;
    sum.s4 += weights.sD * delayLine.s6;
    sum.s4 += weights.sC * delayLine.s7;
    sum.s4 += weights.sB * delayLine.s8;
    sum.s4 += weights.sA * delayLine.s9;
    sum.s4 += weights.s9 * delayLine.sA;
    sum.s4 += weights.s8 * delayLine.sB;
    sum.s4 += weights.s7 * delayLine.sC;
    sum.s4 += weights.s6 * delayLine.sD;
    sum.s4 += weights.s5 * delayLine.sE;
    sum.s4 += weights.s4 * delayLine.sF;
    sum.s4 += weights.s3 * delayLine.s0;
    sum.s4 += weights.s2 * delayLine.s1;
    sum.s4 += weights.s1 * delayLine.s2;
    sum.s4 += weights.s0 * delayLine.s3;
    (*filteredData)[station][pol][time + 4][channel][ri] = sum.s4;

    sum.s5 = weights.sF * delayLine.s5;
    delayLine.s5 = convert_float((*sampledData)[station][time + NR_TAPS - 1 + 6][channel][pol_ri]);
    sum.s5 += weights.sE * delayLine.s6;
    sum.s5 += weights.sD * delayLine.s7;
    sum.s5 += weights.sC * delayLine.s8;
    sum.s5 += weights.sB * delayLine.s9;
    sum.s5 += weights.sA * delayLine.sA;
    sum.s5 += weights.s9 * delayLine.sB;
    sum.s5 += weights.s8 * delayLine.sC;
    sum.s5 += weights.s7 * delayLine.sD;
    sum.s5 += weights.s6 * delayLine.sE;
    sum.s5 += weights.s5 * delayLine.sF;
    sum.s5 += weights.s4 * delayLine.s0;
    sum.s5 += weights.s3 * delayLine.s1;
    sum.s5 += weights.s2 * delayLine.s2;
    sum.s5 += weights.s1 * delayLine.s3;
    sum.s5 += weights.s0 * delayLine.s4;
    (*filteredData)[station][pol][time + 5][channel][ri] = sum.s5;

    sum.s6 = weights.sF * delayLine.s6;
    delayLine.s6 = convert_float((*sampledData)[station][time + NR_TAPS - 1 + 7][channel][pol_ri]);
    sum.s6 += weights.sE * delayLine.s7;
    sum.s6 += weights.sD * delayLine.s8;
    sum.s6 += weights.sC * delayLine.s9;
    sum.s6 += weights.sB * delayLine.sA;
    sum.s6 += weights.sA * delayLine.sB;
    sum.s6 += weights.s9 * delayLine.sC;
    sum.s6 += weights.s8 * delayLine.sD;
    sum.s6 += weights.s7 * delayLine.sE;
    sum.s6 += weights.s6 * delayLine.sF;
    sum.s6 += weights.s5 * delayLine.s0;
    sum.s6 += weights.s4 * delayLine.s1;
    sum.s6 += weights.s3 * delayLine.s2;
    sum.s6 += weights.s2 * delayLine.s3;
    sum.s6 += weights.s1 * delayLine.s4;
    sum.s6 += weights.s0 * delayLine.s5;
    (*filteredData)[station][pol][time + 6][channel][ri] = sum.s6;

    sum.s7 = weights.sF * delayLine.s7;
    delayLine.s7 = convert_float((*sampledData)[station][time + NR_TAPS - 1 + 8][channel][pol_ri]);
    sum.s7 += weights.sE * delayLine.s8;
    sum.s7 += weights.sD * delayLine.s9;
    sum.s7 += weights.sC * delayLine.sA;
    sum.s7 += weights.sB * delayLine.sB;
    sum.s7 += weights.sA * delayLine.sC;
    sum.s7 += weights.s9 * delayLine.sD;
    sum.s7 += weights.s8 * delayLine.sE;
    sum.s7 += weights.s7 * delayLine.sF;
    sum.s7 += weights.s6 * delayLine.s0;
    sum.s7 += weights.s5 * delayLine.s1;
    sum.s7 += weights.s4 * delayLine.s2;
    sum.s7 += weights.s3 * delayLine.s3;
    sum.s7 += weights.s2 * delayLine.s4;
    sum.s7 += weights.s1 * delayLine.s5;
    sum.s7 += weights.s0 * delayLine.s6;
    (*filteredData)[station][pol][time + 7][channel][ri] = sum.s7;

    sum.s8 = weights.sF * delayLine.s8;
    delayLine.s8 = convert_float((*sampledData)[station][time + NR_TAPS - 1 + 9][channel][pol_ri]);
    sum.s8 += weights.sE * delayLine.s9;
    sum.s8 += weights.sD * delayLine.sA;
    sum.s8 += weights.sC * delayLine.sB;
    sum.s8 += weights.sB * delayLine.sC;
    sum.s8 += weights.sA * delayLine.sD;
    sum.s8 += weights.s9 * delayLine.sE;
    sum.s8 += weights.s8 * delayLine.sF;
    sum.s8 += weights.s7 * delayLine.s0;
    sum.s8 += weights.s6 * delayLine.s1;
    sum.s8 += weights.s5 * delayLine.s2;
    sum.s8 += weights.s4 * delayLine.s3;
    sum.s8 += weights.s3 * delayLine.s4;
    sum.s8 += weights.s2 * delayLine.s5;
    sum.s8 += weights.s1 * delayLine.s6;
    sum.s8 += weights.s0 * delayLine.s7;
    (*filteredData)[station][pol][time + 8][channel][ri] = sum.s8;

    sum.s9 = weights.sF * delayLine.s9;
    delayLine.s9 = convert_float((*sampledData)[station][time + NR_TAPS - 1 + 10][channel][pol_ri]);
    sum.s9 += weights.sE * delayLine.sA;
    sum.s9 += weights.sD * delayLine.sB;
    sum.s9 += weights.sC * delayLine.sC;
    sum.s9 += weights.sB * delayLine.sD;
    sum.s9 += weights.sA * delayLine.sE;
    sum.s9 += weights.s9 * delayLine.sF;
    sum.s9 += weights.s8 * delayLine.s0;
    sum.s9 += weights.s7 * delayLine.s1;
    sum.s9 += weights.s6 * delayLine.s2;
    sum.s9 += weights.s5 * delayLine.s3;
    sum.s9 += weights.s4 * delayLine.s4;
    sum.s9 += weights.s3 * delayLine.s5;
    sum.s9 += weights.s2 * delayLine.s6;
    sum.s9 += weights.s1 * delayLine.s7;
    sum.s9 += weights.s0 * delayLine.s8;
    (*filteredData)[station][pol][time + 9][channel][ri] = sum.s9;

    sum.sA = weights.sF * delayLine.sA;
    delayLine.sA = convert_float((*sampledData)[station][time + NR_TAPS - 1 + 11][channel][pol_ri]);
    sum.sA += weights.sE * delayLine.sB;
    sum.sA += weights.sD * delayLine.sC;
    sum.sA += weights.sC * delayLine.sD;
    sum.sA += weights.sB * delayLine.sE;
    sum.sA += weights.sA * delayLine.sF;
    sum.sA += weights.s9 * delayLine.s0;
    sum.sA += weights.s8 * delayLine.s1;
    sum.sA += weights.s7 * delayLine.s2;
    sum.sA += weights.s6 * delayLine.s3;
    sum.sA += weights.s5 * delayLine.s4;
    sum.sA += weights.s4 * delayLine.s5;
    sum.sA += weights.s3 * delayLine.s6;
    sum.sA += weights.s2 * delayLine.s7;
    sum.sA += weights.s1 * delayLine.s8;
    sum.sA += weights.s0 * delayLine.s9;
    (*filteredData)[station][pol][time + 10][channel][ri] = sum.sA;

    sum.sB = weights.sF * delayLine.sB;
    delayLine.sB = convert_float((*sampledData)[station][time + NR_TAPS - 1 + 12][channel][pol_ri]);
    sum.sB += weights.sE * delayLine.sC;
    sum.sB += weights.sD * delayLine.sD;
    sum.sB += weights.sC * delayLine.sE;
    sum.sB += weights.sB * delayLine.sF;
    sum.sB += weights.sA * delayLine.s0;
    sum.sB += weights.s9 * delayLine.s1;
    sum.sB += weights.s8 * delayLine.s2;
    sum.sB += weights.s7 * delayLine.s3;
    sum.sB += weights.s6 * delayLine.s4;
    sum.sB += weights.s5 * delayLine.s5;
    sum.sB += weights.s4 * delayLine.s6;
    sum.sB += weights.s3 * delayLine.s7;
    sum.sB += weights.s2 * delayLine.s8;
    sum.sB += weights.s1 * delayLine.s9;
    sum.sB += weights.s0 * delayLine.sA;
    (*filteredData)[station][pol][time + 11][channel][ri] = sum.sB;

    sum.sC = weights.sF * delayLine.sC;
    delayLine.sC = convert_float((*sampledData)[station][time + NR_TAPS - 1 + 13][channel][pol_ri]);
    sum.sC += weights.sE * delayLine.sD;
    sum.sC += weights.sD * delayLine.sE;
    sum.sC += weights.sC * delayLine.sF;
    sum.sC += weights.sB * delayLine.s0;
    sum.sC += weights.sA * delayLine.s1;
    sum.sC += weights.s9 * delayLine.s2;
    sum.sC += weights.s8 * delayLine.s3;
    sum.sC += weights.s7 * delayLine.s4;
    sum.sC += weights.s6 * delayLine.s5;
    sum.sC += weights.s5 * delayLine.s6;
    sum.sC += weights.s4 * delayLine.s7;
    sum.sC += weights.s3 * delayLine.s8;
    sum.sC += weights.s2 * delayLine.s9;
    sum.sC += weights.s1 * delayLine.sA;
    sum.sC += weights.s0 * delayLine.sB;
    (*filteredData)[station][pol][time + 12][channel][ri] = sum.sC;

    sum.sD = weights.sF * delayLine.sD;
    delayLine.sD = convert_float((*sampledData)[station][time + NR_TAPS - 1 + 14][channel][pol_ri]);
    sum.sD += weights.sE * delayLine.sE;
    sum.sD += weights.sD * delayLine.sF;
    sum.sD += weights.sC * delayLine.s0;
    sum.sD += weights.sB * delayLine.s1;
    sum.sD += weights.sA * delayLine.s2;
    sum.sD += weights.s9 * delayLine.s3;
    sum.sD += weights.s8 * delayLine.s4;
    sum.sD += weights.s7 * delayLine.s5;
    sum.sD += weights.s6 * delayLine.s6;
    sum.sD += weights.s5 * delayLine.s7;
    sum.sD += weights.s4 * delayLine.s8;
    sum.sD += weights.s3 * delayLine.s9;
    sum.sD += weights.s2 * delayLine.sA;
    sum.sD += weights.s1 * delayLine.sB;
    sum.sD += weights.s0 * delayLine.sC;
    (*filteredData)[station][pol][time + 13][channel][ri] = sum.sD;

    sum.sE = weights.sF * delayLine.sE;
    delayLine.sE = convert_float((*sampledData)[station][time + NR_TAPS - 1 + 15][channel][pol_ri]);
    sum.sE += weights.sE * delayLine.sF;
    sum.sE += weights.sD * delayLine.s0;
    sum.sE += weights.sC * delayLine.s1;
    sum.sE += weights.sB * delayLine.s2;
    sum.sE += weights.sA * delayLine.s3;
    sum.sE += weights.s9 * delayLine.s4;
    sum.sE += weights.s8 * delayLine.s5;
    sum.sE += weights.s7 * delayLine.s6;
    sum.sE += weights.s6 * delayLine.s7;
    sum.sE += weights.s5 * delayLine.s8;
    sum.sE += weights.s4 * delayLine.s9;
    sum.sE += weights.s3 * delayLine.sA;
    sum.sE += weights.s2 * delayLine.sB;
    sum.sE += weights.s1 * delayLine.sC;
    sum.sE += weights.s0 * delayLine.sD;
    (*filteredData)[station][pol][time + 14][channel][ri] = sum.sE;

    sum.sF = weights.sF * delayLine.sF;
    sum.sF += weights.sE * delayLine.s0;
    sum.sF += weights.sD * delayLine.s1;
    sum.sF += weights.sC * delayLine.s2;
    sum.sF += weights.sB * delayLine.s3;
    sum.sF += weights.sA * delayLine.s4;
    sum.sF += weights.s9 * delayLine.s5;
    sum.sF += weights.s8 * delayLine.s6;
    sum.sF += weights.s7 * delayLine.s7;
    sum.sF += weights.s6 * delayLine.s8;
    sum.sF += weights.s5 * delayLine.s9;
    sum.sF += weights.s4 * delayLine.sA;
    sum.sF += weights.s3 * delayLine.sB;
    sum.sF += weights.s2 * delayLine.sC;
    sum.sF += weights.s1 * delayLine.sD;
    sum.sF += weights.s0 * delayLine.sE;
    (*filteredData)[station][pol][time + 15][channel][ri] = sum.sF;
  }
}

