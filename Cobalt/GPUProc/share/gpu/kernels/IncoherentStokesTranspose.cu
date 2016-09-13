//# IncoherentStokesTranspose.cu
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
//# $Id: IncoherentStokesTranspose.cu 27248 2013-11-05 17:37:41Z amesfoort $

#if !(NR_CHANNELS > 0)
#error Precondition violated: NR_CHANNELS > 0
#endif

#if !(NR_POLARIZATIONS == 2)
#error Precondition violated: NR_POLARIZATIONS == 2
#endif

#if !(NR_SAMPLES_PER_CHANNEL > 0)
#error Precondition violated: NR_SAMPLES_PER_CHANNEL > 0
#endif

#if !(NR_STATIONS > 0)
#error Precondition violated: NR_STATIONS > 0
#endif

#if !(TILE_SIZE > 0 && TILE_SIZE <= 32)
#error Precondition violated: TILE_SIZE > 0 && TILE_SIZE <= 32
#endif

// 3-D input data array of band-pass corrected data. Note that, actually, the
// data is 4-D (<tt>[station][channel][time][pol]</tt>), but the 4th dimension
// has been squashed into a single float4 (i.e., two complex polarizations).
typedef float4 (*InputDataType)[NR_STATIONS][NR_CHANNELS][NR_SAMPLES_PER_CHANNEL];

// 4-D output data array of band-pass corrected data. The output data
// (<tt>[station][pol][time][channel]</tt>) can be fed into an inverse FFT.
typedef float2 (*OutputDataType)[NR_STATIONS][NR_POLARIZATIONS][NR_SAMPLES_PER_CHANNEL][NR_CHANNELS];


// Transpose the output of the band-pass correction kernel into a format
// suitable for doing an inverse 4k FFT. 
//
// The transpose is done tile-wise, using shared memory. This moves the
// expensive non-coalesced global memory accesses to the more efficient
// non-coalesced shared memory accesses.
// For each thread, the kernel needs to determine if data must be read and/or
// written, because the number of frequency channels and the number of times
// samples per channel need not be a multiple of the shared memory tile size (\c
// TILE_SIZE) used by the kernel.
//
// Note that, to avoid bank conflicts, the shared memory tile has one extra
// column. I.e., its has dimensions [\c TILE_SIZE, \c TILE_SIZE +
// 1]. Furthermore, since \c NR_CHANNELS and \c NR_SAMPLES_PER_CHANNEL need not
// be multiples of TILE_SIZE, we need to make sure that we're not reading and
// writing outside of the bounds of the global memory arrays.
//
// \param[in] input 4-D input array of band-pass corrected data.
// \param[out] output 4-D output array of band-pass corrected data that can be
// fed into an inverse FFT. 
//
// Pre-processor input symbols (some are tied to the execution configuration)
// Symbol                 | Valid Values  | Description
// ---------------------- | ------------- | -----------
// NR_CHANNELS            | > 0           | number of channels per subband
// NR_POLARIZATIONS       | 2             | number of polarizations
// NR_SAMPLES_PER_CHANNEL | > 0           | number of input samples per channel
// NR_STATIONS            | > 0           | number of stations
// TILE_SIZE              | > 0 and <= 32 | size of shared memory tile
//
// Execution configuration:
//
// - LocalWorkSize: 2 dimensional (\c TILE_SIZE, \c TILE_SIZE, 1)
// - GlobalWorkSize: 2 dimensional
//   + X-dimension: \c NR_SAMPLES_PER_CHANNEL, rounded up to nearest multiple of
//                  \c TILE_SIZE
//   + Y-dimension: \c NR_CHANNELS, rounded up to nearest multiple of
//                  \c TILE_SIZE
extern "C"
__global__ void transpose(OutputDataType output,
                          const InputDataType input)
{
  unsigned time, channel;

  __shared__ float4 tmp[TILE_SIZE][TILE_SIZE + 1];

  for (int station = 0; station < NR_STATIONS; station++) {
    time = blockIdx.x * blockDim.x + threadIdx.x;
    channel = blockIdx.y * blockDim.y + threadIdx.y;
    if (channel < NR_CHANNELS && time < NR_SAMPLES_PER_CHANNEL) {
      tmp[threadIdx.y][threadIdx.x] =  (*input)[station][channel][time];
    }
    __syncthreads();

    time = blockIdx.x * blockDim.x + threadIdx.y;
    channel = blockIdx.y * blockDim.y + threadIdx.x;
    if (channel < NR_CHANNELS && time < NR_SAMPLES_PER_CHANNEL) {
      float4 sample = tmp[threadIdx.x][threadIdx.y];
      (*output)[station][0][time][channel] = make_float2(sample.x, sample.y);
      (*output)[station][1][time][channel] = make_float2(sample.z, sample.w);
    }
    __syncthreads();

  }

}
