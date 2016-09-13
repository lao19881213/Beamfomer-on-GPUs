//# Transpose.cl
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
//# $Id: Transpose.cl 24388 2013-03-26 11:14:29Z amesfoort $

typedef __global float2 (*TransposedDataType)[NR_TABS][NR_POLARIZATIONS][NR_SAMPLES_PER_SUBBAND + NR_STATION_FILTER_TAPS - 1][512];
typedef __global float4 (*ComplexVoltagesType)[NR_SUBBANDS][NR_SAMPLES_PER_SUBBAND + NR_STATION_FILTER_TAPS - 1][NR_TABS];


__kernel void UHEP_Transpose(__global void *restrict transposedDataPtr,
                             __global const void *restrict complexVoltagesPtr,
                             __global int reverseSubbandMapping[512])
{
  TransposedDataType transposedData = (TransposedDataType) transposedDataPtr;
  ComplexVoltagesType complexVoltages = (ComplexVoltagesType) complexVoltagesPtr;

  __local float4 tmp[16][17];

  uint tabBase = 16 * get_global_id(1);
  uint sbBase = 16 * get_global_id(2);

  uint tabOffsetR = get_local_id(0) & 15;
  uint tabR = tabBase + tabOffsetR;
  uint sbOffsetR = get_local_id(0) >> 4;
  int sbSourceR = reverseSubbandMapping[sbBase + sbOffsetR];
  bool doR = (NR_TABS % 16 == 0 || tabR < NR_TABS) && sbSourceR >= 0;

  uint tabOffsetW = get_local_id(0) >> 4;
  uint tabW = tabBase + tabOffsetW;
  uint sbOffsetW = get_local_id(0) & 15;
  int sbSourceW = reverseSubbandMapping[sbBase + sbOffsetW];
  bool doW = NR_TABS % 16 == 0 || tabW < NR_TABS;

  for (int time = 0; time < NR_SAMPLES_PER_SUBBAND + NR_STATION_FILTER_TAPS - 1; time++) {
    if (doR)
      tmp[tabOffsetR][sbOffsetR] = (*complexVoltages)[sbSourceR][time][tabR];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (doW) {
      float4 sample = sbSourceW >= 0 ? tmp[tabOffsetW][sbOffsetW] : 0;
      (*transposedData)[tabW][0][time][sbBase + sbOffsetW] = sample.xy;
      (*transposedData)[tabW][1][time][sbBase + sbOffsetW] = sample.zw;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

