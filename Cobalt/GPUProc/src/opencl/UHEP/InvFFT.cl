//# InvFFT.cl
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
//# $Id: InvFFT.cl 24388 2013-03-26 11:14:29Z amesfoort $

#include "math.cl"


typedef __global float (*InvFFTedDataType)[NR_TABS][NR_POLARIZATIONS][NR_SAMPLES_PER_SUBBAND + NR_STATION_FILTER_TAPS - 1][1024];
typedef __global float2 (*TransposedDataType)[NR_TABS][NR_POLARIZATIONS][NR_SAMPLES_PER_SUBBAND + NR_STATION_FILTER_TAPS - 1][512];


float2 inv(float2 a)
{
  return (float2) (-a.y, a.x);
}


void inv4(float2 *R0, float2 *R1, float2 *R2, float2 *R3)
{
  float2 T0, T1, T2, T3;

  T1 = (*R1);
  (*R1) = (*R2);
  (*R2) = T1;

  T0 = (*R0), T1 = (*R1), (*R0) = T0 + T1, (*R1) = T0 - T1;
  T2 = (*R2), T3 = (*R3), (*R2) = T2 + T3, (*R3) = T2 - T3;

  T0 = (*R0), T2 = (*R2), (*R0) = T0 + T2, (*R2) = T0 - T2;
  T1 = (*R1), T3 = inv(*R3), (*R1) = T1 + T3, (*R3) = T1 - T3;
}


void inv8(float2 *R0, float2 *R1, float2 *R2, float2 *R3, float2 *R4, float2 *R5, float2 *R6, float2 *R7)
{
  const float HSQR2 = .70710678118654752440f;
  float2 T0, T1, T2, T3, T4, T5, T6, T7;

  T1 = (*R1);
  (*R1) = (*R4);
  (*R4) = T1;
  T3 = (*R3);
  (*R3) = (*R6);
  (*R6) = T3;

  T0 = (*R0), T1 = (*R1), (*R0) = T0 + T1, (*R1) = T0 - T1;
  T2 = (*R2), T3 = (*R3), (*R2) = T2 + T3, (*R3) = T2 - T3;
  T4 = (*R4), T5 = (*R5), (*R4) = T4 + T5, (*R5) = T4 - T5;
  T6 = (*R6), T7 = (*R7), (*R6) = T6 + T7, (*R7) = T6 - T7;

  T0 = (*R0), T2 = (*R2), (*R0) = T0 + T2, (*R2) = T0 - T2;
  T1 = (*R1), T3 = inv(*R3), (*R1) = T1 + T3, (*R3) = T1 - T3;
  T4 = (*R4), T6 = (*R6), (*R4) = T4 + T6, (*R6) = T4 - T6;
  T5 = (*R5), T7 = inv(*R7), (*R5) = T5 + T7, (*R7) = T5 - T7;

  T0 = (*R0), T4 = (*R4), (*R0) = T0 + T4, (*R4) = T0 - T4;
  T1 = (*R1), T5 = HSQR2 * (inv(*R5) + (*R5)), (*R1) = T1 + T5, (*R5) = T1 - T5;
  T2 = (*R2), T6 = inv(*R6), (*R2) = T2 + T6, (*R6) = T2 - T6;
  T3 = (*R3), T7 = HSQR2 * (inv(*R7) - (*R7)), (*R3) = T3 + T7, (*R7) = T3 - T7;
}


__kernel __attribute__((reqd_work_group_size(128, 1, 1)))
void inv_fft(__global float2 *outputPtr, __global const float *inputPtr)
{
  InvFFTedDataType invFFTedData = (InvFFTedDataType) outputPtr;
  TransposedDataType transposedData = (TransposedDataType) inputPtr;

  const float PI = 3.14159265358979323844f;

  __local union {
    float f1[1024];
    float2 f2[512];
    float4 f4[256];
    float8 f8[128];
  } lds;

  uint windex;
  float ang;
  float2 R0, R1, R2, R3, R4, R5, R6, R7;
  float2 W0, W1, W2, W3;

#if 0
  __global float2 *bufIn = inputPtr + get_global_id(1) * 512;
  __global float *bufOut = outputPtr + get_global_id(1) * 1024;

  R0 = bufIn[get_local_id(0) + 0];
  R1 = bufIn[get_local_id(0) + 128];
  R2 = bufIn[get_local_id(0) + 256];
  R3 = bufIn[get_local_id(0) + 384];
#else
  R0 = (*transposedData)[0][0][get_global_id(1)][get_local_id(0) + 0];
  R1 = (*transposedData)[0][0][get_global_id(1)][get_local_id(0) + 128];
  R2 = (*transposedData)[0][0][get_global_id(1)][get_local_id(0) + 256];
  R3 = (*transposedData)[0][0][get_global_id(1)][get_local_id(0) + 384];
#endif

  lds.f2[get_local_id(0) + 0] = R0;
  lds.f2[get_local_id(0) + 128] = R1;
  lds.f2[get_local_id(0) + 256] = R2;
  lds.f2[get_local_id(0) + 384] = R3;

  barrier(CLK_LOCAL_MEM_FENCE);

  if (get_local_id(0) == 0) {
    R4 = (float2) (R0.y, 0);
    R0 = (float2) (R0.x, 0);
    //R4 = (float2) (bufIn[512].x, 0);
  } else {
    R4 = lds.f2[512 - get_local_id(0)];
  }

  R5 = lds.f2[384 - get_local_id(0)];
  R6 = lds.f2[256 - get_local_id(0)];
  R7 = lds.f2[128 - get_local_id(0)];
  R4.y = -R4.y;
  R5.y = -R5.y;
  R6.y = -R6.y;
  R7.y = -R7.y;

  inv8(&R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);

  barrier(CLK_LOCAL_MEM_FENCE);

  lds.f8[get_local_id(0)] = (float8) (R0.x, R1.x, R2.x, R3.x, R4.x, R5.x, R6.x, R7.x);

  barrier(CLK_LOCAL_MEM_FENCE);

  R0.x = lds.f1[get_local_id(0) + 0];
  R1.x = lds.f1[get_local_id(0) + 128];
  R2.x = lds.f1[get_local_id(0) + 256];
  R3.x = lds.f1[get_local_id(0) + 384];
  R4.x = lds.f1[get_local_id(0) + 512];
  R5.x = lds.f1[get_local_id(0) + 640];
  R6.x = lds.f1[get_local_id(0) + 768];
  R7.x = lds.f1[get_local_id(0) + 896];

  barrier(CLK_LOCAL_MEM_FENCE);

  lds.f8[get_local_id(0)] = (float8) (R0.y, R1.y, R2.y, R3.y, R4.y, R5.y, R6.y, R7.y);

  barrier(CLK_LOCAL_MEM_FENCE);

  R0.y = lds.f1[get_local_id(0) + 0];
  R1.y = lds.f1[get_local_id(0) + 128];
  R2.y = lds.f1[get_local_id(0) + 256];
  R3.y = lds.f1[get_local_id(0) + 384];
  R4.y = lds.f1[get_local_id(0) + 512];
  R5.y = lds.f1[get_local_id(0) + 640];
  R6.y = lds.f1[get_local_id(0) + 768];
  R7.y = lds.f1[get_local_id(0) + 896];

  barrier(CLK_LOCAL_MEM_FENCE);

  ang = 2.0f * PI / 64.0f * (float) (get_local_id(0) % 8);
  R1 = cmul(cexp(       ang), R1);
  R2 = cmul(cexp(2.0f * ang), R2);
  R3 = cmul(cexp(3.0f * ang), R3);
  R4 = cmul(cexp(4.0f * ang), R4);
  R5 = cmul(cexp(5.0f * ang), R5);
  R6 = cmul(cexp(6.0f * ang), R6);
  R7 = cmul(cexp(7.0f * ang), R7);

  inv8(&R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);

  windex = 64 * (get_local_id(0) / 8) + get_local_id(0) % 8;
  lds.f1[windex + 0] = R0.x;
  lds.f1[windex + 8] = R1.x;
  lds.f1[windex + 16] = R2.x;
  lds.f1[windex + 24] = R3.x;
  lds.f1[windex + 32] = R4.x;
  lds.f1[windex + 40] = R5.x;
  lds.f1[windex + 48] = R6.x;
  lds.f1[windex + 56] = R7.x;

  barrier(CLK_LOCAL_MEM_FENCE);

  W0 = lds.f2[get_local_id(0) + 0];
  W1 = lds.f2[get_local_id(0) + 128];
  W2 = lds.f2[get_local_id(0) + 256];
  W3 = lds.f2[get_local_id(0) + 384];
  R0.x = W0.x;
  R1.x = W1.x;
  R2.x = W2.x;
  R3.x = W3.x;
  R4.x = W0.y;
  R5.x = W1.y;
  R6.x = W2.y;
  R7.x = W3.y;

  barrier(CLK_LOCAL_MEM_FENCE);

  lds.f1[windex + 0] = R0.y;
  lds.f1[windex + 8] = R1.y;
  lds.f1[windex + 16] = R2.y;
  lds.f1[windex + 24] = R3.y;
  lds.f1[windex + 32] = R4.y;
  lds.f1[windex + 40] = R5.y;
  lds.f1[windex + 48] = R6.y;
  lds.f1[windex + 56] = R7.y;

  barrier(CLK_LOCAL_MEM_FENCE);

  W0 = lds.f2[get_local_id(0) + 0];
  W1 = lds.f2[get_local_id(0) + 128];
  W2 = lds.f2[get_local_id(0) + 256];
  W3 = lds.f2[get_local_id(0) + 384];
  R0.y = W0.x;
  R1.y = W1.x;
  R2.y = W2.x;
  R3.y = W3.x;
  R4.y = W0.y;
  R5.y = W1.y;
  R6.y = W2.y;
  R7.y = W3.y;

  barrier(CLK_LOCAL_MEM_FENCE);

  ang = 2.0f * PI / 256.0f * (2 * (get_local_id(0) % 32));
  R1 = cmul(cexp(       ang), R1);
  R2 = cmul(cexp(2.0f * ang), R2);
  R3 = cmul(cexp(3.0f * ang), R3);
  ang += 2.0f * PI / 256.0f;
  R5 = cmul(cexp(       ang), R5);
  R6 = cmul(cexp(2.0f * ang), R6);
  R7 = cmul(cexp(3.0f * ang), R7);

  inv4(&R0, &R1, &R2, &R3);
  inv4(&R4, &R5, &R6, &R7);

  windex = 128 * (get_local_id(0) / 32) + get_local_id(0) % 32;
  lds.f2[windex + 0] = (float2) (R0.x, R4.x);
  lds.f2[windex + 32] = (float2) (R1.x, R5.x);
  lds.f2[windex + 64] = (float2) (R2.x, R6.x);
  lds.f2[windex + 96] = (float2) (R3.x, R7.x);

  barrier(CLK_LOCAL_MEM_FENCE);

  W0 = lds.f2[get_local_id(0) + 0];
  W1 = lds.f2[get_local_id(0) + 128];
  W2 = lds.f2[get_local_id(0) + 256];
  W3 = lds.f2[get_local_id(0) + 384];
  R0.x = W0.x;
  R1.x = W1.x;
  R2.x = W2.x;
  R3.x = W3.x;
  R4.x = W0.y;
  R5.x = W1.y;
  R6.x = W2.y;
  R7.x = W3.y;

  lds.f2[get_local_id(0) + 0] = (float2) (R0.y, R4.y);
  lds.f2[get_local_id(0) + 128] = (float2) (R1.y, R5.y);
  lds.f2[get_local_id(0) + 256] = (float2) (R2.y, R6.y);
  lds.f2[get_local_id(0) + 384] = (float2) (R3.y, R7.y);

  barrier(CLK_LOCAL_MEM_FENCE);

  W0 = lds.f2[windex + 0];
  W1 = lds.f2[windex + 32];
  W2 = lds.f2[windex + 64];
  W3 = lds.f2[windex + 96];
  R0.y = W0.x;
  R1.y = W1.x;
  R2.y = W2.x;
  R3.y = W3.x;
  R4.y = W0.y;
  R5.y = W1.y;
  R6.y = W2.y;
  R7.y = W3.y;

  ang = 2.0f * PI / 1024.0f * (2 * get_local_id(0));
  R1 = cmul(cexp(       ang), R1);
  R2 = cmul(cexp(2.0f * ang), R2);
  R3 = cmul(cexp(3.0f * ang), R3);
  ang += 2.0f * PI / 1024.0f;
  R5 = cmul(cexp(       ang), R5);
  R6 = cmul(cexp(2.0f * ang), R6);
  R7 = cmul(cexp(3.0f * ang), R7);

  inv4(&R0, &R1, &R2, &R3);
  inv4(&R4, &R5, &R6, &R7);

#if 0
  __global float2 *out = (__global float2 *) bufOut;
  out[get_local_id(0) + 0] = 9.765625e-04f * (float2) (R0.x, R4.x);
  out[get_local_id(0) + 128] = 9.765625e-04f * (float2) (R1.x, R5.x);
  out[get_local_id(0) + 256] = 9.765625e-04f * (float2) (R2.x, R6.x);
  out[get_local_id(0) + 384] = 9.765625e-04f * (float2) (R3.x, R7.x);
#else
  __global float2 *out = (__global float2 *) &(*invFFTedData)[0][0][get_global_id(1)][0] + get_local_id(0);
  //out[  0] = 9.765625e-04f * (float2) (R0.x, R4.x);
  //out[128] = 9.765625e-04f * (float2) (R1.x, R5.x);
  //out[256] = 9.765625e-04f * (float2) (R2.x, R6.x);
  //out[384] = 9.765625e-04f * (float2) (R3.x, R7.x);
#endif
}

