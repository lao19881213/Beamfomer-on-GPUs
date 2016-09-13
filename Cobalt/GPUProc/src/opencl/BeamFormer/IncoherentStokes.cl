//# IncoherentStokes.cl
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
//# $Id: IncoherentStokes.cl 24388 2013-03-26 11:14:29Z amesfoort $

typedef __global float (*IncoherentStokesType)[NR_INCOHERENT_STOKES][NR_SAMPLES_PER_CHANNEL / INCOHERENT_STOKES_TIME_INTEGRATION_FACTOR][NR_CHANNELS];
typedef __global float4 (*InputType)[NR_STATIONS][NR_CHANNELS][NR_SAMPLES_PER_CHANNEL / INCOHERENT_STOKES_TIME_INTEGRATION_FACTOR][INCOHERENT_STOKES_TIME_INTEGRATION_FACTOR];


__kernel void incoherentStokes(__global void *restrict stokesPtr,
                               __global const void *restrict inputPtr)
{
  IncoherentStokesType stokes = (IncoherentStokesType) stokesPtr;
  InputType input = (InputType) inputPtr;

  uint time = get_global_id(0);
  uint channel = get_global_id(1);

  if (time >= NR_SAMPLES_PER_CHANNEL / INCOHERENT_STOKES_TIME_INTEGRATION_FACTOR)
    return;

  float stokesI = 0;
#if NR_INCOHERENT_STOKES == 4
  float stokesQ = 0, halfStokesU = 0, halfStokesV = 0;
#endif

  for (uint station = 0; station < NR_STATIONS; station++) {
    for (uint t = 0; t < INCOHERENT_STOKES_TIME_INTEGRATION_FACTOR; t++) {
      float4 sample = (*input)[station][channel][time][t];
      float2 X = sample.xy;
      float2 Y = sample.zw;
      float powerX = X.x * X.x + X.y * X.y;
      float powerY = Y.x * Y.x + Y.y * Y.y;

      stokesI += powerX + powerY;
#if NR_INCOHERENT_STOKES == 4
      stokesQ += powerX - powerY;
      halfStokesU += X.x * Y.x + X.y * Y.y;
      halfStokesV += X.y * Y.x - X.x * Y.y;
#endif
    }
  }

  (*stokes)[0][time][channel] = stokesI;
#if NR_INCOHERENT_STOKES == 4
  (*stokes)[1][time][channel] = stokesQ;
  (*stokes)[2][time][channel] = 2 * halfStokesU;
  (*stokes)[3][time][channel] = 2 * halfStokesV;
#endif
}

