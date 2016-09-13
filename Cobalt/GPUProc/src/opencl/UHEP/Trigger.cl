//# Trigger.cl
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
//# $Id: Trigger.cl 24388 2013-03-26 11:14:29Z amesfoort $

typedef __global struct {
  float mean, variance, bestValue;
  uint bestApproxIndex;
} (*TriggerInfoType)[NR_TABS];

typedef __global float (*InvFIRfilteredDataType)[NR_TABS][NR_POLARIZATIONS][16][16][NR_SAMPLES_PER_SUBBAND / 4][16];


#if 0
float2 computeThreshold(__global const float *invFIRfilteredDataPtr)
{
  float M = 0, S = 0;
  uint count = 0;

  for (uint i = get_local_id(0); i < sizeof(InvFIRfilteredDataType) / sizeof(float); i += get_local_size(0)) {
    ++count;
    float sample = invFIRfilteredDataPtr[i];
    float t = sample - M;
    M += t / count;
    S += t * (sample - M);
  }

  barrier(CLK_GLOBAL_MEM_FENCE);

  __local float2 local_MS[256];

  local_MS[get_local_id(0)] = (float2) (M, S);

  for (uint i = get_local_size(0); (i >>= 1) != 0; ) {
    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_local_id(0) < i)
      local_MS[get_local_id(0)] += local_MS[get_local_id(0) + i];
  }

  if (get_local_id(0) == 0)
    local_MS[0].y = native_sqrt(local_MS[0].y);

  barrier(CLK_LOCAL_MEM_FENCE);
  return local_MS[0];
}
#endif


__kernel void trigger(__global const void *triggerInfoPtr,
                      __global const float *invFIRfilteredDataPtr)
{
  TriggerInfoType triggerInfo = (TriggerInfoType) triggerInfoPtr;
  InvFIRfilteredDataType invFIRfilteredData = (InvFIRfilteredDataType) invFIRfilteredDataPtr;

  uint minor = get_local_id(0);
  uint major = get_local_id(1);
  uint me = 16 * major + minor;
  uint tab = get_global_id(2);

  float mean = 0, sumsqdiff = 0;
  float count = 0;

  __local union {
    float f[16][16][16];
    float16 f16[16][16];
    struct {
      float means[256], sumsqdiffs[256], values[256];
      uint approxIndices[256];
    } best;
  } tmp;

  float16 h0, h1;
  h1 /*.s789ABCDEF*/ = 0;
  float16 sum_0;
  float bestValue = 0;
  uint bestApproxIndex = 0;

  for (uint time = 0; time < 1024 * NR_SAMPLES_PER_SUBBAND / 4096; time++) {
    for (uint i = 0; i < 16; i++) {
      float sampleX = (*invFIRfilteredData)[tab][0][i][major][time][minor];
      float sampleY = (*invFIRfilteredData)[tab][1][i][major][time][minor];
      float power = sampleX * sampleX + sampleY * sampleY;
      tmp.f[i][major][minor] = power;

      count += 1.0f;
      float delta = power - mean;
      mean += delta / count;
      sumsqdiff += delta * (power - mean);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    h0 = tmp.f16[major][minor];

    sum_0.s0 = sum_0.sF + h0.s0 - h1.s5;
    sum_0.s1 = sum_0.s0 + h0.s1 - h1.s6;
    sum_0.s2 = sum_0.s1 + h0.s2 - h1.s7;
    sum_0.s3 = sum_0.s2 + h0.s3 - h1.s8;
    sum_0.s4 = sum_0.s3 + h0.s4 - h1.s9;
    sum_0.s5 = sum_0.s4 + h0.s5 - h1.sA;
    sum_0.s6 = sum_0.s5 + h0.s6 - h1.sB;
    sum_0.s7 = sum_0.s6 + h0.s7 - h1.sC;
    sum_0.s8 = sum_0.s7 + h0.s8 - h1.sD;
    sum_0.s9 = sum_0.s8 + h0.s9 - h1.sE;
    sum_0.sA = sum_0.s9 + h0.sA - h1.sF;
    sum_0.sB = sum_0.sA + h0.sB - h0.s0;
    sum_0.sC = sum_0.sB + h0.sC - h0.s1;
    sum_0.sD = sum_0.sC + h0.sD - h0.s2;
    sum_0.sE = sum_0.sD + h0.sE - h0.s3;
    sum_0.sF = sum_0.sE + h0.sF - h0.s4;

    float m0 = max(max(sum_0.s0, sum_0.s1), max(sum_0.s2, sum_0.s3));
    float m1 = max(max(sum_0.s4, sum_0.s5), max(sum_0.s6, sum_0.s7));
    float m2 = max(max(sum_0.s8, sum_0.s9), max(sum_0.sA, sum_0.sB));
    float m3 = max(max(sum_0.sC, sum_0.sD), max(sum_0.sE, sum_0.sF));
    float m = max(max(m0, m1), max(m2, m3));

    if (m >= bestValue) {
      bestValue = m;
      bestApproxIndex = me * 1024 * NR_SAMPLES_PER_SUBBAND / 256 + time * 16;
    }

    h1 /*.s56789ABCDEF*/ = h0 /*.s56789ABCDEF*/;

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  tmp.best.means[me] = mean;
  tmp.best.sumsqdiffs[me] = sumsqdiff;
  tmp.best.values[me] = bestValue;
  tmp.best.approxIndices[me] = bestApproxIndex;

  for (uint i = 256; (i >>= 1) != 0; ) {
    if (me < i) {
      float meanA = tmp.best.means[me], meanB = tmp.best.means[me + i];
      float sumsqdiffA = tmp.best.sumsqdiffs[me], sumsqdiffB = tmp.best.sumsqdiffs[me + i];
      float delta = meanB - meanA;
      tmp.best.means[me] = (meanA + meanB) / 2;
      tmp.best.sumsqdiffs[me] = sumsqdiffA + sumsqdiffB + delta * delta * count / 2;
      count *= 2;

      if (tmp.best.values[me] < tmp.best.values[me + i]) {
        tmp.best.values[me] = tmp.best.values[me + i];
        tmp.best.approxIndices[me] = tmp.best.approxIndices[me + i];
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (me == 0) {
    (*triggerInfo)[tab].mean = tmp.best.means[0];
    (*triggerInfo)[tab].variance = tmp.best.sumsqdiffs[0] / (count - 1);
    (*triggerInfo)[tab].bestValue = tmp.best.values[0];
    (*triggerInfo)[tab].bestApproxIndex = tmp.best.approxIndices[0];
  }
}

