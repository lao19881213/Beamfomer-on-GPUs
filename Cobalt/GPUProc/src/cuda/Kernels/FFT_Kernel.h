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
//# $Id: FFT_Kernel.h 26758 2013-09-30 11:47:21Z loose $

#ifndef LOFAR_GPUPROC_CUDA_FFT_KERNEL_H
#define LOFAR_GPUPROC_CUDA_FFT_KERNEL_H

#include <CoInterface/Parset.h>

#include <GPUProc/gpu_wrapper.h>
#include "FFT_Plan.h"
#include <GPUProc/PerformanceCounter.h>

namespace LOFAR
{
  namespace Cobalt
  {
    //# Forward declarations
    struct BlockID;

    class FFT_Kernel
    {
    public:
      FFT_Kernel(const gpu::Stream &stream, unsigned fftSize, unsigned nrFFTs,
                 bool forward, const gpu::DeviceMemory &buffer);

      void enqueue(const BlockID &blockId) const;

      void enqueue(const BlockID &blockId, PerformanceCounter &counter) const;

      enum BufferType
      {
        INPUT_DATA,
        OUTPUT_DATA
      };

      // Return required buffer size for \a bufferType
      static size_t bufferSize(const Parset& ps, BufferType bufferType);

    private:

      gpu::Context context;

      const unsigned nrFFTs, fftSize;
      const int direction;
      FFT_Plan plan;
      gpu::DeviceMemory buffer;
      gpu::Stream itsStream;
    };
  }
}
#endif
