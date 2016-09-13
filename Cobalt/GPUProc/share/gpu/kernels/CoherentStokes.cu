//# CoherentStokes.cu: Calculate the Stokes parameters
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
//# $Id: CoherentStokes.cu 27228 2013-11-04 12:40:38Z loose $

#if !(INTEGRATION_SIZE >= 1)
#error Precondition violated: INTEGRATION_SIZE >= 1
#endif

#if !(NR_CHANNELS >= 1)
#error Precondition violated: NR_CHANNELS >= 1
#endif

#if !(NR_COHERENT_STOKES == 1 || NR_COHERENT_STOKES == 4)
#error Precondition violated: NR_COHERENT_STOKES == 1 || NR_COHERENT_STOKES == 4
#endif

#if !(NR_POLARIZATIONS == 2)
#error Precondition violated: NR_POLARIZATIONS == 2
#endif

#if !(NR_SAMPLES_PER_CHANNEL > 0 && NR_SAMPLES_PER_CHANNEL % INTEGRATION_SIZE == 0)
#error Precondition violated: NR_SAMPLES_PER_CHANNEL > 0 && NR_SAMPLES_PER_CHANNEL % INTEGRATION_SIZE == 0
#endif

#if !(NR_TABS >= 1)
#error Precondition violated: NR_TABS >= 1
#endif

#if !(TIME_PARALLEL_FACTOR >= 1)
#error Precondition violated: TIME_PARALLEL_FACTOR >= 1
#endif

//4D input array of complex samples. For each tab and polarization there are
//time lines with data for each channel
typedef float2 (*InputDataType)[NR_TABS][NR_POLARIZATIONS][NR_SAMPLES_PER_CHANNEL][NR_CHANNELS]; 

//4D output array of stokes values. Each sample contains 1 or 4 stokes
//paramters. For each tab, there are NR_COHERENT_STOKES timeseries of channels
typedef float (*OutputDataType)[NR_TABS][NR_COHERENT_STOKES][NR_SAMPLES_PER_CHANNEL/INTEGRATION_SIZE][NR_CHANNELS];

/*!
 * Computes the first or all 4 stokes parameters.
 * http://www.astron.nl/~romein/papers/EuroPar-11/EuroPar-11.pdf 
 * \code
 * I = X * conj(X) + Y * conj(Y)
 * Q = X * conj(X) - Y * conj(Y)
 * U = 2 * real(X * con(Y))
 * V = 2 * imag(X * con(Y))
 * \endcode
 * This reduces to (validated on paper by Wouter and John):
 * \code
 * Px = real(X) * real(X) + imag(X) * imag(X)
 * Py = real(Y) * real(Y) + imag(Y) * imag(Y)
 * I = Px + Py
 * Q = Px - Py
 * U = 2 * (real(X) * real(Y) + imag(X) * imag(Y))
 * V = 2 * (imag(X) * real(Y) - real(X) * imag(Y))
 * \endcode
 * 
 * The kernel's first parallel dimension is on the channels; the second
 * dimension is in time; the third on the tabs.  The thread block size based on
 * these factors could be larger then the hardmare max.  Therefore<tt>
 * NR_CHANNELS * NR_TABS * TIME_PARALLEL_FACTOR </tt>should not exceed the
 * hardware maximum of threads (1024 on a K10).
 *
 * \param[out] output
 *             4D output array of stokes values. Each sample contains 1 or 4
 *             stokes paramters. For each tab, there are \c NR_COHERENT_STOKES
 *             time series of channels. The dimensions are: \c NR_TABS, \c
 *             NR_COHERENT_STOKES,
 *             <tt>(NR_SAMPLES_PER_CHANNEL/INTEGRATION_SIZE)</tt>, \c
 *             NR_CHANNELS.
 * \param[in]  input
 *             4D input array of complex samples. For each tab and polarization
 *             there are time lines with data for each channel. The dimensions
 *             are: \c NR_TABS, \c NR_POLARIZATIONS, \c NR_SAMPLES_PER_CHANNEL,
 *             \c NR_CHANNELS
 *
 * Pre-processor input symbols (some are tied to the execution configuration)
 * Symbol                 | Valid Values  | Description
 * ---------------------- | ------------- | -----------
 * INTEGRATION_SIZE       | >= 1          | amount of samples to integrate to a single output sample
 * NR_CHANNELS            | >= 1          | number of frequency channels per subband
 * NR_COHERENT_STOKES     | 1 or 4        | number of stokes paramters to create
 * NR_POLARIZATIONS       | 2             | number of polarizations
 * NR_SAMPLES_PER_CHANNEL | multiple of INTEGRATION_SIZE | number of input samples per channel
 * NR_TABS                | >= 1          | number of tabs to create
 * TIME_PARALLEL_FACTOR   | >= 1          | amount of parallel threads to work on a full timerange
 * 
 * The \c TIME_PARALLEL_FACTOR splits the time range in a number of portions
 * which get worked on by seperate threads (in parallel).
 *  
 * Execution configuration:
 * - LocalWorkSize = 3 dimensional; (\c NR_CHANNELS, \c TIME_PARALLEL_FACTOR, \c
 *                   NR_TABS).  The product of the three should not be larger
 *                   then max thread size.  The max thread size depends on the
 *                   hardware used: 512 for old hardware, 1024 for K10 and
 *                   higher.
 * - GlobalWorkSize = 3 dimensional; depends on the size of \c NR_TABS, \c
 *                   NR_CHANNELS and the max thread size. Ideally the work fits
 *                   in a single block. If not the remainder could be computed
 *                   with a second (differently sized) block.
 */
extern "C" __global__ void coherentStokes(OutputDataType output,
                                          const InputDataType input) 
{
  //# Define the indexes in the data depending on the block and thread idx
  unsigned channel_idx = threadIdx.x;  //# If we have channels do the read and write with 16 in parallel
  unsigned time_idx = threadIdx.y;     
  unsigned tab_idx = threadIdx.z;    

  //# Step over (part of) the timerange of samples with INTEGRATION_SIZE steps
  //# The time_idx determines which part of (or the whole of) the time range this
  //# thread is working on.  Work from the start of the time frame (pending your
  //# threadIdx.y) until the next timeframe.  Step within this time range with
  //# integration size steps. These substeps are done in the inner loop.
  for (unsigned idx_stride = time_idx * (NR_SAMPLES_PER_CHANNEL / TIME_PARALLEL_FACTOR) ; 
                   idx_stride < (time_idx + 1) * (NR_SAMPLES_PER_CHANNEL / TIME_PARALLEL_FACTOR)
                && idx_stride < NR_SAMPLES_PER_CHANNEL;
                idx_stride += INTEGRATION_SIZE)
  {
    //# We are integrating all values in the current stride
    //# local variable
    float stokesI = 0;
#   if NR_COHERENT_STOKES == 4
    float stokesQ = 0;
    float halfStokesU = 0;
    float halfStokesV = 0;
#   endif

    //# Do the integration
    for (unsigned idx_step = 0; idx_step < INTEGRATION_SIZE; idx_step++) 
    {
      float2 X = (*input)[tab_idx][0][idx_stride + idx_step][channel_idx];    
      float2 Y = (*input)[tab_idx][1][idx_stride + idx_step][channel_idx];

      //# Calculate the partial solutions
      float powerX = X.x * X.x + X.y * X.y;
      float powerY = Y.x * Y.x + Y.y * Y.y;
      stokesI += powerX + powerY;
#     if NR_COHERENT_STOKES == 4
      stokesQ += powerX - powerY;
      halfStokesU += X.x * Y.x + X.y * Y.y;
      halfStokesV += X.y * Y.x - X.x * Y.y;
#     endif
    }
    //# We step in the data with INTEGRATION_SIZE
    unsigned write_idx = idx_stride / INTEGRATION_SIZE;

    (*output)[tab_idx][0][write_idx][channel_idx] = stokesI;
#   if NR_COHERENT_STOKES == 4
    (*output)[tab_idx][1][write_idx][channel_idx] = stokesQ;
    (*output)[tab_idx][2][write_idx][channel_idx] = 2 * halfStokesU;
    (*output)[tab_idx][3][write_idx][channel_idx] = 2 * halfStokesV;
#   endif  
    //# No baries needed. All computations are fully parallel
  }
}
