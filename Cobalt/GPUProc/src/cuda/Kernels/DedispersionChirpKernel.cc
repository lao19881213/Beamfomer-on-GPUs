//# DedispersionChirpKernel.cc
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
//# $Id: DedispersionChirpKernel.cc 27077 2013-10-24 01:00:58Z amesfoort $

#include <lofar_config.h>

#include "DedispersionChirpKernel.h"

#include <Common/lofar_complex.h>
#include <Common/LofarLogger.h>

#include <GPUProc/global_defines.h>

namespace LOFAR
{
  namespace Cobalt
  {
    DedispersionChirpKernel::
    DedispersionChirpKernel(const Parset &ps,
                            gpu::Context &context,
                            gpu::DeviceMemory &buffer,
                            gpu::DeviceMemory &DMs)
      :
      Kernel(ps, context, "BeamFormer/Dedispersion.cu", "applyChirp")
    {
      setArg(0, buffer);
      setArg(1, DMs);

      unsigned maxNrThreads;
      //getWorkGroupInfo(queue.getInfo<CL_QUEUE_DEVICE>(), CL_KERNEL_WORK_GROUP_SIZE, &maxNrThreads);
      maxNrThreads = getAttribute(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
      unsigned fftSize = ps.dedispersionFFTsize();

      gpu::Grid globalWorkSize(fftSize, ps.nrSamplesPerChannel() / fftSize, ps.nrChannelsPerSubband());
      //std::cout << "globalWorkSize = NDRange(" << fftSize << ", " << ps.nrSamplesPerChannel() / fftSize << ", " << ps.nrChannelsPerSubband() << ')' << std::endl;
      gpu::Block localWorkSize;

      if (fftSize <= maxNrThreads) {
        localWorkSize = gpu::Block(fftSize, 1, maxNrThreads / fftSize);
        //std::cout << "localWorkSize = NDRange(" << fftSize << ", 1, " << maxNrThreads / fftSize << ')' << std::endl;
      } else {
        unsigned divisor;

        for (divisor = 1; fftSize / divisor > maxNrThreads || fftSize % divisor != 0; divisor++)
          ;

        localWorkSize = gpu::Block(fftSize / divisor, 1, 1);
        //std::cout << "localWorkSize = NDRange(" << fftSize / divisor << ", 1, 1))" << std::endl;
      }

      setEnqueueWorkSizes(globalWorkSize, localWorkSize);

      nrOperations = (size_t) NR_POLARIZATIONS * ps.nrChannelsPerSubband() * ps.nrSamplesPerChannel() * (9 * ps.nrTABs(0) + 17),
      nrBytesRead = nrBytesWritten = sizeof(std::complex<float>) * ps.nrTABs(0) * NR_POLARIZATIONS * ps.nrChannelsPerSubband() * ps.nrSamplesPerChannel();
    }

    void DedispersionChirpKernel::enqueue(gpu::Stream &queue/*, PerformanceCounter &counter*/, double subbandFrequency)
    {
      setArg(2, (float) subbandFrequency);
      Kernel::enqueue(queue/*, counter*/);
    }
  }
}

