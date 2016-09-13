//# FFT_Kernel.cc
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
//# $Id: FFT_Kernel.cc 25358 2013-06-18 09:05:40Z loose $

#include <lofar_config.h>

#include <vector>

#include <Common/LofarLogger.h>
#include <GPUProc/global_defines.h>

#include "FFT_Kernel.h"

namespace LOFAR
{
  namespace Cobalt
  {

    FFT_Kernel::FFT_Kernel(cl::Context &context, unsigned fftSize, unsigned nrFFTs, bool forward, cl::Buffer &buffer)
      :
      nrFFTs(nrFFTs),
      fftSize(fftSize)
#if defined USE_CUSTOM_FFT
    {
      ASSERT(fftSize == 256);
      ASSERT(forward);
      std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
      cl::Program program = createProgram(context, devices, "FFT.cl", "");
      kernel = cl::Kernel(program, "fft0");
      kernel.setArg(0, buffer);
    }
#else
      , direction(forward ? clFFT_Forward : clFFT_Inverse),
      plan(context, fftSize),
      buffer(buffer)
    {
    }
#endif

    void FFT_Kernel::enqueue(cl::CommandQueue &queue, PerformanceCounter &counter)
    {
#if defined USE_CUSTOM_FFT
      queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(nrFFTs * 64 / 4, 4), cl::NDRange(64, 4), 0, &event);
#else
      cl_int error = clFFT_ExecuteInterleaved(queue(), plan.plan, nrFFTs, direction, buffer(), buffer(), 0, 0, &event());

      if (error != CL_SUCCESS)
        throw cl::Error(error, "clFFT_ExecuteInterleaved");
#endif

      counter.doOperation(event,
                          (size_t) nrFFTs * 5 * fftSize * log2(fftSize),
                          (size_t) nrFFTs * fftSize * sizeof(std::complex<float>),
                          (size_t) nrFFTs * fftSize * sizeof(std::complex<float>));
    }

    size_t FFT_Kernel::bufferSize(const Parset& ps, BufferType bufferType)
    {
      switch (bufferType) {
      case INPUT_DATA: 
      case OUTPUT_DATA:
        return
          ps.nrStations() * NR_POLARIZATIONS * 
          ps.nrSamplesPerSubband() * sizeof(std::complex<float>);
      default:
        THROW(GPUProcException, "Invalid bufferType (" << bufferType << ")");
      }
    }

  }
}

