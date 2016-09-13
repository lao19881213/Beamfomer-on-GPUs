//# UHEP_InvFIR_Kernel.cc
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
//# $Id: UHEP_InvFIR_Kernel.cc 24977 2013-05-21 13:46:16Z amesfoort $

#include <lofar_config.h>

#include "UHEP_InvFIR_Kernel.h"

#include <GPUProc/global_defines.h>

namespace LOFAR
{
  namespace Cobalt
  {
    UHEP_InvFIR_Kernel::UHEP_InvFIR_Kernel(const Parset &ps, cl::CommandQueue &queue,
                                           cl::Program &program, cl::Buffer &devInvFIRfilteredData, cl::Buffer &devFFTedData,
                                           cl::Buffer &devInvFIRfilterWeights)
      :
      Kernel(ps, program, "invFIRfilter")
    {
      setArg(0, devInvFIRfilteredData);
      setArg(1, devFFTedData);
      setArg(2, devInvFIRfilterWeights);

      size_t maxNrThreads, nrThreads;
      getWorkGroupInfo(queue.getInfo<CL_QUEUE_DEVICE>(), CL_KERNEL_WORK_GROUP_SIZE, &maxNrThreads);
      // round down to nearest power of two
      for (nrThreads = 1024; nrThreads > maxNrThreads; nrThreads /= 2)
        ;

      globalWorkSize = cl::NDRange(1024, NR_POLARIZATIONS, ps.nrTABs(0));
      localWorkSize = cl::NDRange(nrThreads, 1, 1);

      size_t count = ps.nrTABs(0) * NR_POLARIZATIONS * 1024;
      nrOperations = count * ps.nrSamplesPerChannel() * NR_STATION_FILTER_TAPS * 2;
      nrBytesRead = count * (ps.nrSamplesPerChannel() + NR_STATION_FILTER_TAPS - 1) * sizeof(float);
      nrBytesWritten = count * ps.nrSamplesPerChannel() * sizeof(float);
    }
  }
}

