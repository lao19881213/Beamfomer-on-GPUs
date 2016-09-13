//# BeamFormer.cl
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
//# $Id: BeamFormer.cl 24388 2013-03-26 11:14:29Z amesfoort $

#define MAX(A,B) ((A)>(B) ? (A) : (B))
#define NR_PASSES MAX((NR_STATIONS + 6) / 16, 1) // gives best results on GTX 680
#define NR_STATIONS_PER_PASS ((NR_STATIONS + NR_PASSES - 1) / NR_PASSES)

#if NR_STATIONS_PER_PASS > 48
#error "need more passes to beam form this number of stations"
#endif

#if NR_BITS_PER_SAMPLE == 8
typedef char4 SampleType;
#else
typedef short4 SampleType;
#endif


typedef __global float2 (*ComplexVoltagesType)[NR_SUBBANDS][NR_SAMPLES_PER_SUBBAND + NR_STATION_FILTER_TAPS - 1][NR_TABS][NR_POLARIZATIONS];
typedef __global SampleType (*SamplesType)[NR_STATIONS][NR_SUBBANDS][NR_SAMPLES_PER_SUBBAND + NR_STATION_FILTER_TAPS - 1];
typedef __global float2 (*WeightsType)[NR_STATIONS][NR_SUBBANDS][NR_TABS];


__kernel void complexVoltages(__global void *complexVoltagesPtr,
                              __global const void *samplesPtr,
                              __global const void *weightsPtr)
{
  ComplexVoltagesType complexVoltages = (ComplexVoltagesType) complexVoltagesPtr;
  SamplesType samples = (SamplesType) samplesPtr;
  WeightsType weights = (WeightsType) weightsPtr;

  uint pol = get_local_id(0);
  uint tab = get_local_id(1);
  uint subband = get_global_id(2);

  float2 sample;
  __local union {
    float2 samples[NR_STATIONS_PER_PASS][16][NR_POLARIZATIONS];
    float4 samples4[NR_STATIONS_PER_PASS][16];
  } _local;

#pragma unroll
  for (uint first_station = 0; first_station < NR_STATIONS; first_station += NR_STATIONS_PER_PASS) {
#if NR_STATIONS_PER_PASS >= 1
    float2 weight_00;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 0 < NR_STATIONS)
      weight_00 = (*weights)[first_station + 0][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 2
    float2 weight_01;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 1 < NR_STATIONS)
      weight_01 = (*weights)[first_station + 1][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 3
    float2 weight_02;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 2 < NR_STATIONS)
      weight_02 = (*weights)[first_station + 2][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 4
    float2 weight_03;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 3 < NR_STATIONS)
      weight_03 = (*weights)[first_station + 3][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 5
    float2 weight_04;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 4 < NR_STATIONS)
      weight_04 = (*weights)[first_station + 4][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 6
    float2 weight_05;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 5 < NR_STATIONS)
      weight_05 = (*weights)[first_station + 5][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 7
    float2 weight_06;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 6 < NR_STATIONS)
      weight_06 = (*weights)[first_station + 6][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 8
    float2 weight_07;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 7 < NR_STATIONS)
      weight_07 = (*weights)[first_station + 7][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 9
    float2 weight_08;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 8 < NR_STATIONS)
      weight_08 = (*weights)[first_station + 8][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 10
    float2 weight_09;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 9 < NR_STATIONS)
      weight_09 = (*weights)[first_station + 9][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 11
    float2 weight_10;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 10 < NR_STATIONS)
      weight_10 = (*weights)[first_station + 10][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 12
    float2 weight_11;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 11 < NR_STATIONS)
      weight_11 = (*weights)[first_station + 11][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 13
    float2 weight_12;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 12 < NR_STATIONS)
      weight_12 = (*weights)[first_station + 12][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 14
    float2 weight_13;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 13 < NR_STATIONS)
      weight_13 = (*weights)[first_station + 13][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 15
    float2 weight_14;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 14 < NR_STATIONS)
      weight_14 = (*weights)[first_station + 14][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 16
    float2 weight_15;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 15 < NR_STATIONS)
      weight_15 = (*weights)[first_station + 15][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 17
    float2 weight_16;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 16 < NR_STATIONS)
      weight_16 = (*weights)[first_station + 16][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 18
    float2 weight_17;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 17 < NR_STATIONS)
      weight_17 = (*weights)[first_station + 17][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 19
    float2 weight_18;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 18 < NR_STATIONS)
      weight_18 = (*weights)[first_station + 18][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 20
    float2 weight_19;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 19 < NR_STATIONS)
      weight_19 = (*weights)[first_station + 19][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 21
    float2 weight_20;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 20 < NR_STATIONS)
      weight_20 = (*weights)[first_station + 20][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 22
    float2 weight_21;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 21 < NR_STATIONS)
      weight_21 = (*weights)[first_station + 21][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 23
    float2 weight_22;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 22 < NR_STATIONS)
      weight_22 = (*weights)[first_station + 22][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 24
    float2 weight_23;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 23 < NR_STATIONS)
      weight_23 = (*weights)[first_station + 23][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 25
    float2 weight_24;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 24 < NR_STATIONS)
      weight_24 = (*weights)[first_station + 24][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 26
    float2 weight_25;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 25 < NR_STATIONS)
      weight_25 = (*weights)[first_station + 25][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 27
    float2 weight_26;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 26 < NR_STATIONS)
      weight_26 = (*weights)[first_station + 26][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 28
    float2 weight_27;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 27 < NR_STATIONS)
      weight_27 = (*weights)[first_station + 27][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 29
    float2 weight_28;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 28 < NR_STATIONS)
      weight_28 = (*weights)[first_station + 28][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 30
    float2 weight_29;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 29 < NR_STATIONS)
      weight_29 = (*weights)[first_station + 29][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 31
    float2 weight_30;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 30 < NR_STATIONS)
      weight_30 = (*weights)[first_station + 30][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 32
    float2 weight_31;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 31 < NR_STATIONS)
      weight_31 = (*weights)[first_station + 31][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 33
    float2 weight_32;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 32 < NR_STATIONS)
      weight_32 = (*weights)[first_station + 32][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 34
    float2 weight_33;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 33 < NR_STATIONS)
      weight_33 = (*weights)[first_station + 33][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 35
    float2 weight_34;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 34 < NR_STATIONS)
      weight_34 = (*weights)[first_station + 34][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 36
    float2 weight_35;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 35 < NR_STATIONS)
      weight_35 = (*weights)[first_station + 35][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 37
    float2 weight_36;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 36 < NR_STATIONS)
      weight_36 = (*weights)[first_station + 36][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 38
    float2 weight_37;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 37 < NR_STATIONS)
      weight_37 = (*weights)[first_station + 37][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 39
    float2 weight_38;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 38 < NR_STATIONS)
      weight_38 = (*weights)[first_station + 38][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 40
    float2 weight_39;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 39 < NR_STATIONS)
      weight_39 = (*weights)[first_station + 39][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 41
    float2 weight_40;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 40 < NR_STATIONS)
      weight_40 = (*weights)[first_station + 40][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 42
    float2 weight_41;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 41 < NR_STATIONS)
      weight_41 = (*weights)[first_station + 41][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 43
    float2 weight_42;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 42 < NR_STATIONS)
      weight_42 = (*weights)[first_station + 42][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 44
    float2 weight_43;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 43 < NR_STATIONS)
      weight_43 = (*weights)[first_station + 43][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 45
    float2 weight_44;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 44 < NR_STATIONS)
      weight_44 = (*weights)[first_station + 44][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 46
    float2 weight_45;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 45 < NR_STATIONS)
      weight_45 = (*weights)[first_station + 45][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 47
    float2 weight_46;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 46 < NR_STATIONS)
      weight_46 = (*weights)[first_station + 46][subband][tab];
#endif

#if NR_STATIONS_PER_PASS >= 48
    float2 weight_47;

    if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 47 < NR_STATIONS)
      weight_47 = (*weights)[first_station + 47][subband][tab];
#endif

    for (uint time = 0; time < NR_SAMPLES_PER_SUBBAND + NR_STATION_FILTER_TAPS - 1; time += 16) {
      barrier(CLK_LOCAL_MEM_FENCE);

      for (uint i = get_local_id(0) + NR_POLARIZATIONS * get_local_id(1); i < NR_STATIONS_PER_PASS * 16; i += NR_TABS * NR_POLARIZATIONS) {
        uint t = i % 16;
        uint s = i / 16;

        if (time + t < NR_SAMPLES_PER_SUBBAND + NR_STATION_FILTER_TAPS - 1)
          if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + s < NR_STATIONS)
            _local.samples4[0][i] = convert_float4((*samples)[first_station + s][subband][time + t]);
      }

      barrier(CLK_LOCAL_MEM_FENCE);

      for (uint t = 0; t < min(16U, (NR_SAMPLES_PER_SUBBAND + NR_STATION_FILTER_TAPS - 1 - time)); t++) {
        float2 sum = first_station == 0 ? 0 : (*complexVoltages)[subband][time + t][tab][pol];

#if NR_STATIONS_PER_PASS >= 1
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 1 < NR_STATIONS) {
          sample = _local.samples[ 0][t][pol];
          sum += weight_00.xx * sample;
          sum += weight_00.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 2
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 2 < NR_STATIONS) {
          sample = _local.samples[ 1][t][pol];
          sum += weight_01.xx * sample;
          sum += weight_01.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 3
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 3 < NR_STATIONS) {
          sample = _local.samples[ 2][t][pol];
          sum += weight_02.xx * sample;
          sum += weight_02.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 4
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 4 < NR_STATIONS) {
          sample = _local.samples[ 3][t][pol];
          sum += weight_03.xx * sample;
          sum += weight_03.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 5
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 5 < NR_STATIONS) {
          sample = _local.samples[ 4][t][pol];
          sum += weight_04.xx * sample;
          sum += weight_04.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 6
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 6 < NR_STATIONS) {
          sample = _local.samples[ 5][t][pol];
          sum += weight_05.xx * sample;
          sum += weight_05.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 7
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 7 < NR_STATIONS) {
          sample = _local.samples[ 6][t][pol];
          sum += weight_06.xx * sample;
          sum += weight_06.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 8
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 8 < NR_STATIONS) {
          sample = _local.samples[ 7][t][pol];
          sum += weight_07.xx * sample;
          sum += weight_07.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 9
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 9 < NR_STATIONS) {
          sample = _local.samples[ 8][t][pol];
          sum += weight_08.xx * sample;
          sum += weight_08.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 10
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 10 < NR_STATIONS) {
          sample = _local.samples[ 9][t][pol];
          sum += weight_09.xx * sample;
          sum += weight_09.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 11
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 11 < NR_STATIONS) {
          sample = _local.samples[10][t][pol];
          sum += weight_10.xx * sample;
          sum += weight_10.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 12
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 12 < NR_STATIONS) {
          sample = _local.samples[11][t][pol];
          sum += weight_11.xx * sample;
          sum += weight_11.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 13
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 13 < NR_STATIONS) {
          sample = _local.samples[12][t][pol];
          sum += weight_12.xx * sample;
          sum += weight_12.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 14
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 14 < NR_STATIONS) {
          sample = _local.samples[13][t][pol];
          sum += weight_13.xx * sample;
          sum += weight_13.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 15
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 15 < NR_STATIONS) {
          sample = _local.samples[14][t][pol];
          sum += weight_14.xx * sample;
          sum += weight_14.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 16
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 15 < NR_STATIONS) {
          sample = _local.samples[15][t][pol];
          sum += weight_15.xx * sample;
          sum += weight_15.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 17
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 16 < NR_STATIONS) {
          sample = _local.samples[16][t][pol];
          sum += weight_16.xx * sample;
          sum += weight_16.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 18
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 17 < NR_STATIONS) {
          sample = _local.samples[17][t][pol];
          sum += weight_17.xx * sample;
          sum += weight_17.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 19
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 18 < NR_STATIONS) {
          sample = _local.samples[18][t][pol];
          sum += weight_18.xx * sample;
          sum += weight_18.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 20
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 19 < NR_STATIONS) {
          sample = _local.samples[19][t][pol];
          sum += weight_19.xx * sample;
          sum += weight_19.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 21
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 20 < NR_STATIONS) {
          sample = _local.samples[20][t][pol];
          sum += weight_20.xx * sample;
          sum += weight_20.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 22
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 21 < NR_STATIONS) {
          sample = _local.samples[21][t][pol];
          sum += weight_21.xx * sample;
          sum += weight_21.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 23
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 22 < NR_STATIONS) {
          sample = _local.samples[22][t][pol];
          sum += weight_22.xx * sample;
          sum += weight_22.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 24
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 23 < NR_STATIONS) {
          sample = _local.samples[23][t][pol];
          sum += weight_23.xx * sample;
          sum += weight_23.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 25
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 25 < NR_STATIONS) {
          sample = _local.samples[24][t][pol];
          sum += weight_24.xx * sample;
          sum += weight_24.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 26
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 25 < NR_STATIONS) {
          sample = _local.samples[25][t][pol];
          sum += weight_25.xx * sample;
          sum += weight_25.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 27
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 26 < NR_STATIONS) {
          sample = _local.samples[26][t][pol];
          sum += weight_26.xx * sample;
          sum += weight_26.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 28
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 27 < NR_STATIONS) {
          sample = _local.samples[27][t][pol];
          sum += weight_27.xx * sample;
          sum += weight_27.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 29
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 28 < NR_STATIONS) {
          sample = _local.samples[28][t][pol];
          sum += weight_28.xx * sample;
          sum += weight_28.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 30
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 29 < NR_STATIONS) {
          sample = _local.samples[29][t][pol];
          sum += weight_29.xx * sample;
          sum += weight_29.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 31
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 30 < NR_STATIONS) {
          sample = _local.samples[30][t][pol];
          sum += weight_30.xx * sample;
          sum += weight_30.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 32
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 31 < NR_STATIONS) {
          sample = _local.samples[31][t][pol];
          sum += weight_31.xx * sample;
          sum += weight_31.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 33
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 32 < NR_STATIONS) {
          sample = _local.samples[32][t][pol];
          sum += weight_32.xx * sample;
          sum += weight_32.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 34
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 33 < NR_STATIONS) {
          sample = _local.samples[33][t][pol];
          sum += weight_33.xx * sample;
          sum += weight_33.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 35
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 34 < NR_STATIONS) {
          sample = _local.samples[34][t][pol];
          sum += weight_34.xx * sample;
          sum += weight_34.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 36
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 35 < NR_STATIONS) {
          sample = _local.samples[35][t][pol];
          sum += weight_35.xx * sample;
          sum += weight_35.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 37
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 36 < NR_STATIONS) {
          sample = _local.samples[36][t][pol];
          sum += weight_36.xx * sample;
          sum += weight_36.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 38
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 37 < NR_STATIONS) {
          sample = _local.samples[37][t][pol];
          sum += weight_37.xx * sample;
          sum += weight_37.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 39
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 38 < NR_STATIONS) {
          sample = _local.samples[38][t][pol];
          sum += weight_38.xx * sample;
          sum += weight_38.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 40
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 39 < NR_STATIONS) {
          sample = _local.samples[39][t][pol];
          sum += weight_39.xx * sample;
          sum += weight_39.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 41
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 40 < NR_STATIONS) {
          sample = _local.samples[40][t][pol];
          sum += weight_40.xx * sample;
          sum += weight_40.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 42
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 41 < NR_STATIONS) {
          sample = _local.samples[41][t][pol];
          sum += weight_41.xx * sample;
          sum += weight_41.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 43
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 42 < NR_STATIONS) {
          sample = _local.samples[42][t][pol];
          sum += weight_42.xx * sample;
          sum += weight_42.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 44
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 43 < NR_STATIONS) {
          sample = _local.samples[43][t][pol];
          sum += weight_43.xx * sample;
          sum += weight_43.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 45
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 44 < NR_STATIONS) {
          sample = _local.samples[44][t][pol];
          sum += weight_44.xx * sample;
          sum += weight_44.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 46
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 45 < NR_STATIONS) {
          sample = _local.samples[45][t][pol];
          sum += weight_45.xx * sample;
          sum += weight_45.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 47
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 46 < NR_STATIONS) {
          sample = _local.samples[46][t][pol];
          sum += weight_46.xx * sample;
          sum += weight_46.yy * (float2) (-sample.y, sample.x);
        }
#endif

#if NR_STATIONS_PER_PASS >= 48
        if (NR_STATIONS % NR_STATIONS_PER_PASS == 0 || first_station + 47 < NR_STATIONS) {
          sample = _local.samples[47][t][pol];
          sum += weight_47.xx * sample;
          sum += weight_47.yy * (float2) (-sample.y, sample.x);
        }
#endif

        (*complexVoltages)[subband][time + t][tab][pol] = sum;
      }
    }
  }
}

