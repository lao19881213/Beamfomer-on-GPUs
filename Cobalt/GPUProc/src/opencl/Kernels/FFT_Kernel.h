//# FFT_Kernel.h
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
//# $Id: FFT_Kernel.h 25358 2013-06-18 09:05:40Z loose $

#ifndef LOFAR_GPUPROC_OPENCL_FFT_KERNEL_H
#define LOFAR_GPUPROC_OPENCL_FFT_KERNEL_H

#include <CoInterface/Parset.h>

#include <GPUProc/gpu_incl.h>
#include <GPUProc/PerformanceCounter.h>
#include "FFT_Plan.h"

namespace LOFAR
{
  namespace Cobalt
  {
    class FFT_Kernel
    {
    public:
      FFT_Kernel(cl::Context &context, unsigned fftSize,
                 unsigned nrFFTs, bool forward, cl::Buffer &buffer);

      void enqueue(cl::CommandQueue &queue, PerformanceCounter &counter);

      enum BufferType
      {
        INPUT_DATA,
        OUTPUT_DATA
      };

      // Return required buffer size for \a bufferType
      static size_t bufferSize(const Parset& ps, BufferType bufferType);

    private:
      unsigned nrFFTs, fftSize;
#if defined USE_CUSTOM_FFT
      cl::Kernel kernel;
#else
      clFFT_Direction direction;
      FFT_Plan plan;
      cl::Buffer   &buffer;
#endif
      cl::Event event;
    };
  }
}

#endif

