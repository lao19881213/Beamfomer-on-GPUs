//# gpu_math.cuh: Functions and operators for CUDA-specific types.
//#
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
//# $Id: gpu_math.cuh 27000 2013-10-17 09:11:13Z loose $

#ifndef LOFAR_GPUPROC_CUDA_GPU_MATH_CUH
#define LOFAR_GPUPROC_CUDA_GPU_MATH_CUH

// \file
// Functions and operators for CUDA-specific types.
// This file contains functions and operators for CUDA-specific types, like
// float4. Only a minimal set of operators is provided, the ones that are
// currently needed. It can be extended when needed. We do \e not plan to
// provide a complete set of C++-operators for all the different CUDA types.


// Provide the equivalent of the OpenCL swizzle feature. Obviously, the OpenCL
// \c swizzle operation is more powerful, because it's a language built-in.
#define SWIZZLE(ARG, X, Y, Z, W) make_float4((ARG).X, (ARG).Y, (ARG).Z, (ARG).W)

inline __device__ float4 operator + (float4 a, float4 b)
{
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

inline __device__ float4 operator - (float4 a, float4 b)
{
  return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

inline __device__ float4 operator * (float4 a, float4 b)
{
  return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

inline __device__ float4 operator / (float4 a, float4 b)
{
  return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

inline __device__ float4& operator += (float4 &a, float4 b)
{
  a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
  return a;
}

inline __device__ float4& operator -= (float4 &a, float4 b)
{
  a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
  return a;
}

inline __device__ float4& operator *= (float4 &a, float4 b)
{
  a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w;
  return a;
}

inline __device__ float4& operator /= (float4 &a, float4 b)
{
  a.x /= b.x; a.y /= b.y; a.z /= b.z; a.w /= b.w;
  return a;
}


// to distinguish complex float/double from other uses of float2/double2
typedef float2  fcomplex;
typedef double2 dcomplex;

typedef char2  char_complex;
typedef short2 short_complex;


// Operator overloads for complex values
//
// Do not implement a type with these, as it must be non-POD.
// This introduces redundant member inits in the constructor,
// causing races when declaring variables in shared memory.
inline __device__ fcomplex operator+(fcomplex a, fcomplex b)
{
  return make_float2(a.x + b.x, a.y + b.y);
}

inline __device__ fcomplex operator*(fcomplex a, fcomplex b)
{
  return make_float2(a.x * b.x - a.y * b.y,
                     a.x * b.y + a.y * b.x);
}

inline __device__ fcomplex operator*(fcomplex a, float b)
{
  return make_float2(a.x * b, a.y * b);
}

inline __device__ fcomplex operator*(float a, fcomplex b)
{
  return make_float2(a * b.x, a * b.y);
}


inline __device__ dcomplex dphaseShift(double frequency, double delay)
{
  // Convert the fraction of sample duration (delayAtBegin/delayAfterEnd) to fractions of a circle.
  // Because we `undo' the delay, we need to rotate BACK.
  //
  // This needs to be done in double precision, because phi may become
  // large after we multiply a small delay and a very large freq,
  // Then we need to compute a good sin(phi) and cos(phi).
  double phi = -2.0 * delay * frequency; // -2.0 * M_PI: M_PI below in sincospi()

  dcomplex rv;
  sincospi(phi, &rv.y, &rv.x); // store (cos(), sin())
  return rv;
}

#endif

