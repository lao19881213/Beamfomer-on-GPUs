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

#if 0
typedef __global float2 (*TransposedDataType)[NR_TABS][NR_POLARIZATIONS][NR_SAMPLES_PER_CHANNEL][NR_CHANNELS];
typedef __global float4 (*ComplexVoltagesType)[NR_CHANNELS][NR_SAMPLES_PER_CHANNEL][NR_TABS];

__kernel void transposeComplexVoltages(__global void *restrict transposedDataPtr,
                                       __global const void *restrict complexVoltagesPtr)
{
  TransposedDataType transposedData = (TransposedDataType) transposedDataPtr;
  ComplexVoltagesType complexVoltages = (ComplexVoltagesType) complexVoltagesPtr;

  __local float4 tmp[16][17];

  uint tabBase = 16 * get_global_id(1);
  uint chBase = 16 * get_global_id(2);

  uint tabOffsetR = get_local_id(0) & 15;
  uint tabR = tabBase + tabOffsetR;
  uint chOffsetR = get_local_id(0) >> 4;
  uint channelR = chBase + chOffsetR;
  bool doR = NR_TABS % 16 == 0 || tabR < NR_TABS;

  uint tabOffsetW = get_local_id(0) >> 4;
  uint tabW = tabBase + tabOffsetW;
  uint chOffsetW = get_local_id(0) & 15;
  uint channelW = chBase + chOffsetW;
  bool doW = NR_TABS % 16 == 0 || tabW < NR_TABS;

  for (int time = 0; time < NR_SAMPLES_PER_CHANNEL; time++) {
    if (doR)
      tmp[tabOffsetR][chOffsetR] = (*complexVoltages)[channelR][time][tabR];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (doW) {
      float4 sample = tmp[tabOffsetW][chOffsetW];
      (*transposedData)[tabW][0][time][channelW] = sample.xy;
      (*transposedData)[tabW][1][time][channelW] = sample.zw;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

#else

typedef __global float2 (*TransposedDataType)[NR_TABS][NR_POLARIZATIONS][NR_CHANNELS][NR_SAMPLES_PER_CHANNEL];
typedef __global float4 (*ComplexVoltagesType)[NR_CHANNELS][NR_SAMPLES_PER_CHANNEL][NR_TABS];

__kernel void transposeComplexVoltages(__global void *restrict transposedDataPtr,
                                       __global const void *restrict complexVoltagesPtr)
{
  TransposedDataType transposedData = (TransposedDataType) transposedDataPtr;
  ComplexVoltagesType complexVoltages = (ComplexVoltagesType) complexVoltagesPtr;

  __local float4 tmp[16][17];

  uint tabBase = 16 * get_global_id(1);
  uint timeBase = 16 * get_global_id(2);

  uint tabOffsetR = get_local_id(0) & 15;
  uint tabR = tabBase + tabOffsetR;
  uint timeOffsetR = get_local_id(0) >> 4;
  uint timeR = timeBase + timeOffsetR;
  bool doR = NR_TABS % 16 == 0 || tabR < NR_TABS;

  uint tabOffsetW = get_local_id(0) >> 4;
  uint tabW = tabBase + tabOffsetW;
  uint timeOffsetW = get_local_id(0) & 15;
  uint timeW = timeBase + timeOffsetW;
  bool doW = NR_TABS % 16 == 0 || tabW < NR_TABS;

  for (int channel = 0; channel < NR_CHANNELS; channel++) {
    if (doR)
      tmp[tabOffsetR][timeOffsetR] = (*complexVoltages)[timeR][channel][tabR];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (doW) {
      float4 sample = tmp[tabOffsetW][timeOffsetW];
      (*transposedData)[tabW][0][channel][timeW] = sample.xy;
      (*transposedData)[tabW][1][channel][timeW] = sample.zw;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

#endif

