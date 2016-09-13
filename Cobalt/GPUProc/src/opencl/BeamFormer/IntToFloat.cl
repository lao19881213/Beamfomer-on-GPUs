//# IntToFloat.cl
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
//# $Id: IntToFloat.cl 24388 2013-03-26 11:14:29Z amesfoort $

#if NR_BITS_PER_SAMPLE == 16
typedef short2 SampleType;
#elif NR_BITS_PER_SAMPLE == 8
typedef char2 SampleType;
#else
#error unsupport NR_BITS_PER_SAMPLE
#endif

typedef __global SampleType (*SampledDataType)[NR_STATIONS][NR_SAMPLES_PER_SUBBAND][NR_POLARIZATIONS];
typedef __global float2 (*ConvertedDataType)[NR_STATIONS][NR_POLARIZATIONS][NR_SAMPLES_PER_SUBBAND];


__kernel void intToFloat(__global void *restrict convertedDataPtr,
                         __global const void *restrict sampledDataPtr)
{
  ConvertedDataType convertedData = (ConvertedDataType) convertedDataPtr;
  SampledDataType sampledData = (SampledDataType) sampledDataPtr;

  uint station = get_global_id(1);

  for (uint time = get_local_id(0); time < NR_SAMPLES_PER_SUBBAND; time += get_local_size(0)) {
    (*convertedData)[station][0][time] = convert_float2((*sampledData)[station][time][0]);
    (*convertedData)[station][1][time] = convert_float2((*sampledData)[station][time][1]);
  }
}

