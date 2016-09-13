//# DelayAndBandPass.cl
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
//# $Id: DelayAndBandPass.cl 24852 2013-05-08 18:26:05Z amesfoort $

#include "math.cl"

/** @file
 * This file contains an OpenCL implementation of the GPU kernel for the delay
 * and bandpass correction.
 *
 * Usually, this kernel will be run after the polyphase filter kernel FIR.cl. In
 * that case, the input data for this kernel is already in floating point format
 * (@c NR_CHANNELS > 1). However, if this kernel is the first in row, then the
 * input data is still in integer format (@c NR_CHANNELS == 1), and this kernel
 * needs to do the integer-to-float conversion.
 *
 * @attention The following pre-processor variables must be supplied when
 * compiling this program. Please take the pre-conditions for these variables
 * into account:
 * - @c NR_CHANNELS: 1 or a multiple of 16
 * - if @c NR_CHANNELS == 1 (input data is in integer format):
 *   - @c NR_BITS_PER_SAMPLE: 8 or 16
 *   - @c NR_SAMPLES_PER_SUBBAND: a multiple of 16
 * - if @c NR_CHANNELS > 1 (input data is in floating point format):
 *   - @c NR_SAMPLES_PER_CHANNEL: a multiple of 16
 * - @c NR_POLARIZATIONS: 2
 * - @c SUBBAND_WIDTH: a multiple of @c NR_CHANNELS
 */

#if NR_CHANNELS == 1
#undef BANDPASS_CORRECTION
#endif


typedef __global fcomplex2 (*restrict OutputDataType)[NR_STATIONS][NR_CHANNELS][NR_SAMPLES_PER_CHANNEL];
#if NR_CHANNELS == 1
#if NR_BITS_PER_SAMPLE == 16
typedef __global short_complex2 (*restrict InputDataType)[NR_STATIONS][NR_SAMPLES_PER_SUBBAND];
#elif NR_BITS_PER_SAMPLE == 8
typedef __global char_complex2 (*restrict InputDataType)[NR_STATIONS][NR_SAMPLES_PER_SUBBAND];
#else
#error unsupport NR_BITS_PER_SAMPLE
#endif
#else
typedef __global fcomplex (*restrict InputDataType)[NR_STATIONS][NR_POLARIZATIONS][NR_SAMPLES_PER_CHANNEL][NR_CHANNELS];
#endif
typedef __global const float2 (*restrict DelaysType)[NR_BEAMS][NR_STATIONS]; // 2 Polarizations; in seconds
typedef __global const float2 (*restrict PhaseOffsetsType)[NR_STATIONS]; // 2 Polarizations; in radians
typedef __global const float (*restrict BandPassFactorsType)[NR_CHANNELS];


/**
 * This kernel perfroms three operations on the input data:
 * - Apply a fine delay by doing a per channel phase correction.
 * - Apply a bandpass correction to compensate for the errors introduced by the
 *   polyphase filter that produced the subbands. This error is deterministic,
 *   hence it can be fully compensated for.
 * - Transpose the data so that the time slices for each channel are placed
 *   consecutively in memory.
 *
 * @param[out] correctedDataPtr    pointer to output data of ::OutputDataType,
 *                                 a 3D array [station][channel][sample]
 *                                 of ::fcomplex2 (2 complex polarizations)
 * @param[in]  filteredDataPtr     pointer to input data; this can either be a
 *                                 4D array [station][polarization][sample][channel]
 *                                 of ::fcomplex, or a 2D array [station][subband]
 *                                 of ::short_complex2 or ::char_complex2,
 *                                 depending on the value of @c NR_CHANNELS
 * @param[in]  subbandFrequency    center freqency of the subband
 * @param[in]  beam                index number of the beam
 * @param[in]  delaysAtBeginPtr    pointer to delay data of ::DelaysType,
 *                                 a 2D array [beam][station] of float2 (real:
 *                                 2 polarizations), containing delays in
 *                                 seconds at begin of integration period
 * @param[in]  delaysAfterEndPtr   pointer to delay data of ::DelaysType,
 *                                 a 2D array [beam][station] of float2 (real:
 *                                 2 polarizations), containing delays in
 *                                 seconds after end of integration period
 * @param[in]  phaseOffsetsPtr     pointer to phase offset data of
 *                                 ::PhaseOffsetsType, a 1D array [station] of
 *                                 float2 (real: 2 polarizations), containing
 *                                 phase offsets in radians
 * @param[in]  bandPassFactorsPtr  pointer to bandpass correction data of
 *                                 ::BandPassFactorsType, a 1D array [channel] of
 *                                 float, containing bandpass correction factors
 */
__kernel __attribute__((reqd_work_group_size(16 * 16, 1, 1)))
void applyDelaysAndCorrectBandPass(__global fcomplex *restrict correctedDataPtr,
                                   __global const fcomplex *restrict filteredDataPtr,
                                   float subbandFrequency,
                                   unsigned beam,
                                   __global const float2 *restrict delaysAtBeginPtr,
                                   __global const float2 *restrict delaysAfterEndPtr,
                                   __global const float2 *restrict phaseOffsetsPtr,
                                   __global const float *restrict bandPassFactorsPtr)
{
  OutputDataType outputData = (OutputDataType) correctedDataPtr;
  InputDataType inputData = (InputDataType) filteredDataPtr;
  DelaysType delaysAtBegin = (DelaysType) delaysAtBeginPtr;
  DelaysType delaysAfterEnd = (DelaysType) delaysAfterEndPtr;
  PhaseOffsetsType phaseOffsets = (PhaseOffsetsType) phaseOffsetsPtr;

#if NR_CHANNELS > 1
  BandPassFactorsType bandPassFactors = (BandPassFactorsType) bandPassFactorsPtr;

  __local fcomplex2 tmp[16][17]; // one too wide to allow coalesced reads

  uint major = get_global_id(0) / 16;
  uint minor = get_global_id(0) % 16;
  uint channel = get_global_id(1) * 16;
#endif
  uint station = get_global_id(2);

#if defined DELAY_COMPENSATION
#if NR_CHANNELS == 1
  float frequency = subbandFrequency;
#else
  float frequency = subbandFrequency - .5f * SUBBAND_BANDWIDTH + (channel + minor) * (SUBBAND_BANDWIDTH / NR_CHANNELS);
#endif
  float2 delayAtBegin = (*delaysAtBegin)[beam][station];
  float2 delayAfterEnd = (*delaysAfterEnd)[beam][station];
  float2 phiBegin = -2 * 3.1415926535f * delayAtBegin;
  float2 phiEnd = -2 * 3.1415926535f * delayAfterEnd;
  float2 deltaPhi = (phiEnd - phiBegin) / NR_SAMPLES_PER_CHANNEL;
#if NR_CHANNELS == 1
  float2 myPhiBegin = (phiBegin + get_local_id(0) * deltaPhi) * frequency + (*phaseOffsets)[station];
  float2 myPhiDelta = get_local_size(0) * deltaPhi * frequency;
#else
  float2 myPhiBegin = (phiBegin + major * deltaPhi) * frequency + (*phaseOffsets)[station];
  float2 myPhiDelta = 16 * deltaPhi * frequency;
#endif
  fcomplex vX = cexp(myPhiBegin.x);
  fcomplex vY = cexp(myPhiBegin.y);
  fcomplex dvX = cexp(myPhiDelta.x);
  fcomplex dvY = cexp(myPhiDelta.y);
#endif

#if defined BANDPASS_CORRECTION
  float weight = (*bandPassFactors)[channel + minor];
#endif

#if defined DELAY_COMPENSATION && defined BANDPASS_CORRECTION
  vX *= weight;
  vY *= weight;
#endif

#if NR_CHANNELS == 1
  for (uint time = get_local_id(0); time < NR_SAMPLES_PER_SUBBAND; time += get_local_size(0)) {
    fcomplex2 samples = convert_float4((*inputData)[station][time]);
    fcomplex sampleX = samples.s01;
    fcomplex sampleY = samples.s23;
#else
  for (uint time = 0; time < NR_SAMPLES_PER_CHANNEL; time += 16) {
    fcomplex sampleX = (*inputData)[station][0][time + major][channel + minor];
    fcomplex sampleY = (*inputData)[station][1][time + major][channel + minor];
#endif

#if defined DELAY_COMPENSATION
    sampleX = cmul(sampleX, vX);
    sampleY = cmul(sampleY, vY);
    vX = cmul(vX, dvX);
    vY = cmul(vY, dvY);
#elif defined BANDPASS_CORRECTION
    sampleX *= weight;
    sampleY *= weight;
#endif

#if NR_CHANNELS == 1
    (*outputData)[station][0][time] = (float4) (sampleX, sampleY);
#else
    tmp[major][minor] = (float4) (sampleX, sampleY);
    barrier(CLK_LOCAL_MEM_FENCE);

    (*outputData)[station][channel + major][time + minor] = tmp[minor][major];
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
  }
}

