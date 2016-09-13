//# BeamFormer.cu
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
//# $Id: BeamFormer.cu 27000 2013-10-17 09:11:13Z loose $

#include "gpu_math.cuh"

//# Some defines used to determine the correct way the process the data
#define MAX(A,B) ((A)>(B) ? (A) : (B))

#define NR_PASSES MAX((NR_STATIONS + 6) / 16, 1) // gives best results on GTX 680

#ifndef NR_STATIONS_PER_PASS  // Allow overriding for testing optimalizations 
  #define NR_STATIONS_PER_PASS ((NR_STATIONS + NR_PASSES - 1) / NR_PASSES)
#endif
#if NR_STATIONS_PER_PASS > 32
#error "need more passes to beam for this number of stations"
#endif

//# Typedefs used to map input data on arrays
typedef  double (*DelaysType)[NR_SAPS][NR_STATIONS][NR_TABS];
typedef  float4 (*BandPassCorrectedType)[NR_STATIONS][NR_CHANNELS][NR_SAMPLES_PER_CHANNEL];
typedef  float2 (*ComplexVoltagesType)[NR_CHANNELS][NR_SAMPLES_PER_CHANNEL][NR_TABS][NR_POLARIZATIONS];

/*!
 * The beamformer kernel performs a complex weighted multiply-add of each sample
 * of the provided input data.
 *
 * \param[out] complexVoltagesPtr      4D output array of beams. For each channel a number of Tied Array Beams time serires is created for two polarizations
 * \param[in]  samplesPtr              3D input array of samples. A time series for each station and channel pair. Each sample contains the 2 polarizations X, Y, each of complex float type.
 * \param[in]  delaysPtr               3D input array of complex valued delays to be applied to the correctData samples. There is a delay for each Sub-Array Pointing, station, and Tied Array Beam triplet.
 * \param[in]  subbandFrequency        central frequency of the subband
 * \param[in]  sap                     number (index) of the Sub-Array Pointing (aka (station) beam)
 *
 * Pre-processor input symbols (some are tied to the execution configuration)
 * Symbol                  | Valid Values            | Description
 * ----------------------- | ----------------------- | -----------
 * NR_STATIONS             | >= 1                    | number of antenna fields
 * NR_SAMPLES_PER_CHANNEL  | >= 1                    | number of input samples per channel
 * NR_CHANNELS             | >= 1                    | number of frequency channels per subband
 * NR_SAPS                 | >= 1 && > sap           | number of Sub-Array Pointings
 * NR_TABS                 | >= 1                    | number of Tied Array Beams (old name: pencil beams) to create
 * WEIGHT_CORRECTION       | float                   | weighting applied to all weights derived from the delays, primarily used for correcting FFT and iFFT chain multiplication correction
 * SUBBAND_BANDWIDTH       | double, multiple of NR_CHANNELS | Bandwidth of a subband in Hz
 * NR_STATIONS_PER_PASS    | 1 >= && <= 32           | Set to overide default: Parallelization parameter, controls the number stations to beamform in a single pass over the input data. 
 *
 * Note that this kernel assumes  NR_POLARIZATIONS == 2
 *
 * Execution configuration:
 * - LocalWorkSize = (NR_POLARIZATIONS, NR_TABS, NR_CHANNELS) Note that for full utilization NR_TABS * NR_CHANNELS % 16 = 0
 */
extern "C" __global__ void beamFormer( void *complexVoltagesPtr,
                                       const void *samplesPtr,
                                       const void *delaysPtr,
                                       double subbandFrequency,
                                       unsigned sap)
{
  ComplexVoltagesType complexVoltages = (ComplexVoltagesType) complexVoltagesPtr;
  BandPassCorrectedType samples = (BandPassCorrectedType) samplesPtr;
  DelaysType delays = (DelaysType) delaysPtr;

  unsigned pol = threadIdx.x;
  unsigned tab = threadIdx.y;
  unsigned channel = blockDim.z * blockIdx.z + threadIdx.z; // The parallelization in the channel is controllable with extra blocks

  // This union is in shared memory because it is used by all threads in the block
  __shared__ union { // Union: Maps two variables to the same adress space
    float2 samples[NR_STATIONS_PER_PASS][16][NR_POLARIZATIONS];
    float4 samples4[NR_STATIONS_PER_PASS][16];
  } _local;

#if NR_CHANNELS == 1
  double frequency = subbandFrequency;
#else
  double frequency = subbandFrequency - 0.5 * SUBBAND_BANDWIDTH + channel * (SUBBAND_BANDWIDTH / NR_CHANNELS);
#endif

#pragma unroll
  for (unsigned first_station = 0;  // Step over data with NR_STATIONS_PER_PASS stride
       first_station < NR_STATIONS;
       first_station += NR_STATIONS_PER_PASS) 
  { // this for loop spans the whole file
#if NR_STATIONS_PER_PASS >= 1
    fcomplex weight_00;                     // assign the weights to register variables
    if (first_station + 0 < NR_STATIONS) {  // Number of station might be larger then 32:
                                            // We then do multiple passes to span all stations
      double delay = (*delays)[sap][first_station + 0][tab];
      dcomplex weight = dphaseShift(frequency, delay);
      weight_00 = make_float2(weight.x, weight.y) * WEIGHT_CORRECTION;
    }
#endif
    // Loop unrolling allows usage of registers for weights
#if NR_STATIONS_PER_PASS >= 2
    fcomplex weight_01;
    if (first_station + 1 < NR_STATIONS) {
      double delay = (*delays)[sap][first_station + 1][tab];
      dcomplex weight = dphaseShift(frequency, delay);
      weight_01 = make_float2(weight.x, weight.y) * WEIGHT_CORRECTION;
    }
#endif

#if NR_STATIONS_PER_PASS >= 3
    fcomplex weight_02;
    if (first_station + 2 < NR_STATIONS) {
      double delay = (*delays)[sap][first_station + 2][tab];
      dcomplex weight = dphaseShift(frequency, delay);
      weight_02 = make_float2(weight.x, weight.y) * WEIGHT_CORRECTION;
    }
#endif

#if NR_STATIONS_PER_PASS >= 4
    fcomplex weight_03;
    if (first_station + 3 < NR_STATIONS) {
      double delay = (*delays)[sap][first_station + 3][tab];
      dcomplex weight = dphaseShift(frequency, delay);
      weight_03 = make_float2(weight.x, weight.y) * WEIGHT_CORRECTION;
    }
#endif

#if NR_STATIONS_PER_PASS >= 5
    fcomplex weight_04;
    if (first_station + 4 < NR_STATIONS) {
      double delay = (*delays)[sap][first_station + 4][tab];
      dcomplex weight = dphaseShift(frequency, delay);
      weight_04 = make_float2(weight.x, weight.y) * WEIGHT_CORRECTION;
    }
#endif

#if NR_STATIONS_PER_PASS >= 6
    fcomplex weight_05;
    if (first_station + 5 < NR_STATIONS) {
      double delay = (*delays)[sap][first_station + 5][tab];
      dcomplex weight = dphaseShift(frequency, delay);
      weight_05 = make_float2(weight.x, weight.y) * WEIGHT_CORRECTION;
    }
#endif

#if NR_STATIONS_PER_PASS >= 7
    fcomplex weight_06;
    if (first_station + 6 < NR_STATIONS) {
      double delay = (*delays)[sap][first_station + 6][tab];
      dcomplex weight = dphaseShift(frequency, delay);
      weight_06 = make_float2(weight.x, weight.y) * WEIGHT_CORRECTION;
    }
#endif

#if NR_STATIONS_PER_PASS >= 8
    fcomplex weight_07;
    if (first_station + 7 < NR_STATIONS) {
      double delay = (*delays)[sap][first_station + 7][tab];
      dcomplex weight = dphaseShift(frequency, delay);
      weight_07 = make_float2(weight.x, weight.y) * WEIGHT_CORRECTION;
    }
#endif

#if NR_STATIONS_PER_PASS >= 9
    fcomplex weight_08;
    if (first_station + 8 < NR_STATIONS) {
      double delay = (*delays)[sap][first_station + 8][tab];
      dcomplex weight = dphaseShift(frequency, delay);
      weight_08 = make_float2(weight.x, weight.y) * WEIGHT_CORRECTION;
    }
#endif

#if NR_STATIONS_PER_PASS >= 10
    fcomplex weight_09;
    if (first_station + 9 < NR_STATIONS) {
      double delay = (*delays)[sap][first_station + 9][tab];
      dcomplex weight = dphaseShift(frequency, delay);
      weight_09 = make_float2(weight.x, weight.y) * WEIGHT_CORRECTION;
    }
#endif

#if NR_STATIONS_PER_PASS >= 11
    fcomplex weight_10;
    if (first_station + 10 < NR_STATIONS) {
      double delay = (*delays)[sap][first_station + 10][tab];
      dcomplex weight = dphaseShift(frequency, delay);
      weight_10 = make_float2(weight.x, weight.y) * WEIGHT_CORRECTION;
    }
#endif

#if NR_STATIONS_PER_PASS >= 12
    fcomplex weight_11;
    if (first_station + 11 < NR_STATIONS) {
      double delay = (*delays)[sap][first_station + 11][tab];
      dcomplex weight = dphaseShift(frequency, delay);
      weight_11 = make_float2(weight.x, weight.y) * WEIGHT_CORRECTION;
    }
#endif

#if NR_STATIONS_PER_PASS >= 13
    fcomplex weight_12;
    if (first_station + 12 < NR_STATIONS) {
      double delay = (*delays)[sap][first_station + 12][tab];
      dcomplex weight = dphaseShift(frequency, delay);
      weight_12 = make_float2(weight.x, weight.y) * WEIGHT_CORRECTION;
    }
#endif

#if NR_STATIONS_PER_PASS >= 14
    fcomplex weight_13;
    if (first_station + 13 < NR_STATIONS) {
      double delay = (*delays)[sap][first_station + 13][tab];
      dcomplex weight = dphaseShift(frequency, delay);
      weight_13 = make_float2(weight.x, weight.y) * WEIGHT_CORRECTION;
    }
#endif

#if NR_STATIONS_PER_PASS >= 15
    fcomplex weight_14;
    if (first_station + 14 < NR_STATIONS) {
      double delay = (*delays)[sap][first_station + 14][tab];
      dcomplex weight = dphaseShift(frequency, delay);
      weight_14 = make_float2(weight.x, weight.y) * WEIGHT_CORRECTION;
    }
#endif

#if NR_STATIONS_PER_PASS >= 16
    fcomplex weight_15;
    if (first_station + 15 < NR_STATIONS) {
      double delay = (*delays)[sap][first_station + 15][tab];
      dcomplex weight = dphaseShift(frequency, delay);
      weight_15 = make_float2(weight.x, weight.y) * WEIGHT_CORRECTION;
    }
#endif

#if NR_STATIONS_PER_PASS >= 17
    fcomplex weight_16;
    if (first_station + 16 < NR_STATIONS) {
      double delay = (*delays)[sap][first_station + 16][tab];
      dcomplex weight = dphaseShift(frequency, delay);
      weight_16 = make_float2(weight.x, weight.y) * WEIGHT_CORRECTION;
    }
#endif

#if NR_STATIONS_PER_PASS >= 18
    fcomplex weight_17;
    if (first_station + 17 < NR_STATIONS) {
      double delay = (*delays)[sap][first_station + 17][tab];
      dcomplex weight = dphaseShift(frequency, delay);
      weight_17 = make_float2(weight.x, weight.y) * WEIGHT_CORRECTION;
    }
#endif

#if NR_STATIONS_PER_PASS >= 19
    fcomplex weight_18;
    if (first_station + 18 < NR_STATIONS)
      double delay = (*delays)[sap][first_station + 18][tab];
      dcomplex weight = dphaseShift(frequency, delay);
      weight_18 = make_float2(weight.x, weight.y) * WEIGHT_CORRECTION;
    }
#endif

#if NR_STATIONS_PER_PASS >= 20
    fcomplex weight_19;
    if (first_station + 19 < NR_STATIONS) {
      double delay = (*delays)[sap][first_station + 19][tab];
      dcomplex weight = dphaseShift(frequency, delay);
      weight_19 = make_float2(weight.x, weight.y) * WEIGHT_CORRECTION;
    }
#endif

#if NR_STATIONS_PER_PASS >= 21
    fcomplex weight_20;
    if (first_station + 20 < NR_STATIONS) {
      double delay = (*delays)[sap][first_station + 20][tab];
      dcomplex weight = dphaseShift(frequency, delay);
      weight_20 = make_float2(weight.x, weight.y) * WEIGHT_CORRECTION;
    }
#endif

#if NR_STATIONS_PER_PASS >= 22
    fcomplex weight_21;
    if (first_station + 21 < NR_STATIONS) {
      double delay = (*delays)[sap][first_station + 21][tab];
      dcomplex weight = dphaseShift(frequency, delay);
      weight_21 = make_float2(weight.x, weight.y) * WEIGHT_CORRECTION;
    }
#endif

#if NR_STATIONS_PER_PASS >= 23
    fcomplex weight_22;
    if (first_station + 22 < NR_STATIONS) {
      double delay = (*delays)[sap][first_station + 22][tab];
      dcomplex weight = dphaseShift(frequency, delay);
      weight_22 = make_float2(weight.x, weight.y) * WEIGHT_CORRECTION;
    }
#endif

#if NR_STATIONS_PER_PASS >= 24
    fcomplex weight_23;
    if (first_station + 23 < NR_STATIONS) {
      double delay = (*delays)[sap][first_station + 23][tab];
      dcomplex weight = dphaseShift(frequency, delay);
      weight_23 = make_float2(weight.x, weight.y) * WEIGHT_CORRECTION;
    }
#endif

#if NR_STATIONS_PER_PASS >= 25
    fcomplex weight_24;
    if (first_station + 24 < NR_STATIONS) {
      double delay = (*delays)[sap][first_station + 24][tab];
      dcomplex weight = dphaseShift(frequency, delay);
      weight_24 = make_float2(weight.x, weight.y) * WEIGHT_CORRECTION;
    }
#endif

#if NR_STATIONS_PER_PASS >= 26
    fcomplex weight_25;
    if (first_station + 25 < NR_STATIONS) {
      double delay = (*delays)[sap][first_station + 25][tab];
      dcomplex weight = dphaseShift(frequency, delay);
      weight_25 = make_float2(weight.x, weight.y) * WEIGHT_CORRECTION;
    }
#endif

#if NR_STATIONS_PER_PASS >= 27
    fcomplex weight_26;
    if (first_station + 26 < NR_STATIONS) {
      double delay = (*delays)[sap][first_station + 26][tab];
      dcomplex weight = dphaseShift(frequency, delay);
      weight_26 = make_float2(weight.x, weight.y) * WEIGHT_CORRECTION;
    }
#endif

#if NR_STATIONS_PER_PASS >= 28
    fcomplex weight_27;
    if (first_station + 27 < NR_STATIONS) {
      double delay = (*delays)[sap][first_station + 27][tab];
      dcomplex weight = dphaseShift(frequency, delay);
      weight_27 = make_float2(weight.x, weight.y) * WEIGHT_CORRECTION;
    }
#endif

#if NR_STATIONS_PER_PASS >= 29
    fcomplex weight_28;
    if (first_station + 28 < NR_STATIONS) {
      double delay = (*delays)[sap][first_station + 28][tab];
      dcomplex weight = dphaseShift(frequency, delay);
      weight_28 = make_float2(weight.x, weight.y) * WEIGHT_CORRECTION;
    }
#endif

#if NR_STATIONS_PER_PASS >= 30
    fcomplex weight_29;
    if (first_station + 29 < NR_STATIONS) {
      double delay = (*delays)[sap][first_station + 29][tab];
      dcomplex weight = dphaseShift(frequency, delay);
      weight_29 = make_float2(weight.x, weight.y) * WEIGHT_CORRECTION;
    }
#endif

#if NR_STATIONS_PER_PASS >= 31
    fcomplex weight_30;
    if (first_station + 30 < NR_STATIONS) {
      double delay = (*delays)[sap][first_station + 30][tab];
      dcomplex weight = dphaseShift(frequency, delay);
      weight_30 = make_float2(weight.x, weight.y) * WEIGHT_CORRECTION;
    }
#endif

#if NR_STATIONS_PER_PASS >= 32
    fcomplex weight_31;
    if (first_station + 31 < NR_STATIONS) {
      double delay = (*delays)[sap][first_station + 31][tab];
      dcomplex weight = dphaseShift(frequency, delay);
      weight_31 = make_float2(weight.x, weight.y) * WEIGHT_CORRECTION;
    }
#endif

    // Loop over all the samples in time. Perform the addition for 16 time steps.
    for (unsigned time = 0; time < NR_SAMPLES_PER_CHANNEL; time += 16)
    {
      // Optimized memory transfer: Threads load from memory in parallel
      for (unsigned i = threadIdx.x + NR_POLARIZATIONS * threadIdx.y;
                    i < NR_STATIONS_PER_PASS * 16;
                    i += NR_TABS * NR_POLARIZATIONS) 
      {
        unsigned t = i % 16;
        unsigned s = i / 16;

        if (NR_SAMPLES_PER_CHANNEL % 16 == 0 || time + t < NR_SAMPLES_PER_CHANNEL)
          if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + s < NR_STATIONS)
            _local.samples4[0][i] = (*samples)[first_station + s][channel][time + t];
      }

       __syncthreads();


      for (unsigned t = 0; 
                    t < (NR_SAMPLES_PER_CHANNEL % 16 == 0 ? 16 : min(16U, NR_SAMPLES_PER_CHANNEL - time));
                    t++) 
      {
        fcomplex sum = first_station == 0 ? // The first run the sum should be zero, otherwise we need to take the sum of the previous run
                    make_float2(0,0) :
                    (*complexVoltages)[channel][time + t][tab][pol];

        // Calculate the weighted complex sum of the samples
#if NR_STATIONS_PER_PASS >= 1
        if (first_station + 1 <= NR_STATIONS) {  // Remember that the number of stations might not be a multiple of 32. Skip if station does not exist
          fcomplex sample = _local.samples[ 0][t][pol];
          sum = sum + weight_00 * sample;
        }
#endif

#if NR_STATIONS_PER_PASS >= 2
        if (first_station + 2 <= NR_STATIONS) {
          fcomplex sample = _local.samples[ 1][t][pol];
          sum = sum + weight_01 * sample;
        }
#endif


#if NR_STATIONS_PER_PASS >= 3
        if (first_station + 3 <= NR_STATIONS) {
          fcomplex sample = _local.samples[ 2][t][pol];
          sum = sum + weight_02 * sample;
        }
#endif

#if NR_STATIONS_PER_PASS >= 4
        if (first_station + 4 <= NR_STATIONS) {
          fcomplex sample = _local.samples[ 3][t][pol];
          sum = sum + weight_03 * sample;
        }
#endif

#if NR_STATIONS_PER_PASS >= 5
        if (first_station + 5 <= NR_STATIONS) {
          fcomplex sample = _local.samples[ 4][t][pol];
          sum = sum + weight_04 * sample;
        }
#endif

#if NR_STATIONS_PER_PASS >= 6
        if (first_station + 6 <= NR_STATIONS) {
          fcomplex sample = _local.samples[ 5][t][pol];
          sum = sum + weight_05 * sample;
        }
#endif

#if NR_STATIONS_PER_PASS >= 7
        if (first_station + 7 <= NR_STATIONS) {
          fcomplex sample = _local.samples[ 6][t][pol];
          sum = sum + weight_06 * sample;
        }
#endif

#if NR_STATIONS_PER_PASS >= 8
        if (first_station + 8 <= NR_STATIONS) {
          fcomplex sample = _local.samples[ 7][t][pol];
          sum = sum + weight_07 * sample;
        }
#endif

#if NR_STATIONS_PER_PASS >= 9
        if (first_station + 9 <= NR_STATIONS) {
          fcomplex sample = _local.samples[ 8][t][pol];
          sum = sum + weight_08 * sample;
        }
#endif

#if NR_STATIONS_PER_PASS >= 10
        if (first_station + 10 <= NR_STATIONS) {
          fcomplex sample = _local.samples[ 9][t][pol];
          sum = sum + weight_09 * sample;
        }
#endif

#if NR_STATIONS_PER_PASS >= 11
        if (first_station + 11 <= NR_STATIONS) {
          fcomplex sample = _local.samples[10][t][pol];
          sum = sum + weight_10 * sample;
        }
#endif

#if NR_STATIONS_PER_PASS >= 12
        if (first_station + 12 <= NR_STATIONS) {
          fcomplex sample = _local.samples[11][t][pol];
          sum = sum + weight_11 * sample;
        }
#endif

#if NR_STATIONS_PER_PASS >= 13
        if (first_station + 13 <= NR_STATIONS) {
          fcomplex sample = _local.samples[12][t][pol];
          sum = sum + weight_12 * sample;
        }
#endif

#if NR_STATIONS_PER_PASS >= 14
        if (first_station + 14 <= NR_STATIONS) {
          fcomplex sample = _local.samples[13][t][pol];
          sum = sum + weight_13 * sample;
        }
#endif

#if NR_STATIONS_PER_PASS >= 15
        if (first_station + 15 <= NR_STATIONS) {
          fcomplex sample = _local.samples[14][t][pol];
          sum = sum + weight_14 * sample;
        }
#endif

#if NR_STATIONS_PER_PASS >= 16
        if (first_station + 16 <= NR_STATIONS) {
          fcomplex sample = _local.samples[15][t][pol];
          sum = sum + weight_15 * sample;
        }
#endif

#if NR_STATIONS_PER_PASS >= 17
        if (first_station + 17 <= NR_STATIONS) {
          fcomplex sample = _local.samples[16][t][pol];
          sum = sum + weight_16 * sample;
        }
#endif

#if NR_STATIONS_PER_PASS >= 18
        if (first_station + 18 <= NR_STATIONS) {
          fcomplex sample = _local.samples[17][t][pol];
          sum = sum + weight_17 * sample;
        }
#endif

#if NR_STATIONS_PER_PASS >= 19
        if (first_station + 19 <= NR_STATIONS) {
          fcomplex sample = _local.samples[18][t][pol];
          sum = sum + weight_18 * sample;
        }
#endif

#if NR_STATIONS_PER_PASS >= 20
        if (first_station + 20 <= NR_STATIONS) {
          fcomplex sample = _local.samples[19][t][pol];
          sum = sum + weight_19 * sample;
        }
#endif

#if NR_STATIONS_PER_PASS >= 21
        if (first_station + 21 <= NR_STATIONS) {
          fcomplex sample = _local.samples[20][t][pol];
          sum = sum + weight_20 * sample;
        }
#endif

#if NR_STATIONS_PER_PASS >= 22
        if (first_station + 22 <= NR_STATIONS) {
          fcomplex sample = _local.samples[21][t][pol];
          sum = sum + weight_21 * sample;
        }
#endif

#if NR_STATIONS_PER_PASS >= 23
        if (first_station + 23 <= NR_STATIONS) {
          fcomplex sample = _local.samples[22][t][pol];
          sum = sum + weight_22 * sample;
        }
#endif

#if NR_STATIONS_PER_PASS >= 24
        if (first_station + 24 <= NR_STATIONS) {
          fcomplex sample = _local.samples[23][t][pol];
          sum = sum + weight_23 * sample;
        }
#endif

#if NR_STATIONS_PER_PASS >= 25
        if (first_station + 25 <= NR_STATIONS) {
          fcomplex sample = _local.samples[24][t][pol];
          sum = sum + weight_24 * sample;
        }
#endif

#if NR_STATIONS_PER_PASS >= 26
        if (first_station + 26 <= NR_STATIONS) {
          fcomplex sample = _local.samples[25][t][pol];
          sum = sum + weight_25 * sample;
        }
#endif

#if NR_STATIONS_PER_PASS >= 27
        if (first_station + 27 <= NR_STATIONS) {
          fcomplex sample = _local.samples[26][t][pol];
          sum = sum + weight_26 * sample;
        }
#endif

#if NR_STATIONS_PER_PASS >= 28
        if (first_station + 28 <= NR_STATIONS) {
          fcomplex sample = _local.samples[27][t][pol];
          sum = sum + weight_27 * sample;
        }
#endif

#if NR_STATIONS_PER_PASS >= 29
        if (first_station + 29 <= NR_STATIONS) {
          fcomplex sample = _local.samples[28][t][pol];
          sum = sum + weight_28 * sample;
        }
#endif

#if NR_STATIONS_PER_PASS >= 30
        if (first_station + 30 <= NR_STATIONS) {
          fcomplex sample = _local.samples[29][t][pol];
          sum = sum + weight_29 * sample;
        }
#endif

#if NR_STATIONS_PER_PASS >= 31
        if (first_station + 31 <= NR_STATIONS) {
          fcomplex sample = _local.samples[30][t][pol];
          sum = sum + weight_30 * sample;
        }
#endif

#if NR_STATIONS_PER_PASS >= 32
        if (first_station + 32 <= NR_STATIONS) {
          fcomplex sample = _local.samples[31][t][pol];
          sum = sum + weight_31 * sample;
        }
#endif
        // Write data to global mem
        (*complexVoltages)[channel][time + t][tab][pol] = sum;
      }

      __syncthreads();
    }
  }
}

