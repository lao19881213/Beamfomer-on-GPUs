//# IncoherentStokesKernel.cc
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
//# $Id: IncoherentStokesKernel.cc 24977 2013-05-21 13:46:16Z amesfoort $

#include <lofar_config.h>

#include "IncoherentStokesKernel.h"

#include <Common/lofar_complex.h>

#include <GPUProc/global_defines.h>

namespace LOFAR
{
  namespace Cobalt
  {
    IncoherentStokesKernel::IncoherentStokesKernel(const Parset &ps, cl::CommandQueue &queue,
                                                   cl::Program &program, cl::Buffer &devIncoherentStokes, cl::Buffer &devInputSamples)
      :
      Kernel(ps, program, "incoherentStokes")
    {
      setArg(0, devIncoherentStokes);
      setArg(1, devInputSamples);

      unsigned nrTimes = ps.nrSamplesPerChannel() / ps.incoherentStokesTimeIntegrationFactor();
      size_t maxNrThreads;
      getWorkGroupInfo(queue.getInfo<CL_QUEUE_DEVICE>(), CL_KERNEL_WORK_GROUP_SIZE, &maxNrThreads);
      unsigned nrPasses = (nrTimes + maxNrThreads - 1) / maxNrThreads;
      unsigned nrTimesPerPass = (nrTimes + nrPasses - 1) / nrPasses;
      globalWorkSize = cl::NDRange(nrTimesPerPass * nrPasses, ps.nrChannelsPerSubband());
      localWorkSize = cl::NDRange(nrTimesPerPass, 1);

      nrOperations = ps.nrChannelsPerSubband() * ps.nrSamplesPerChannel() * ps.nrStations() * (ps.nrIncoherentStokes() == 1 ? 8 : 20 + 2.0 / ps.incoherentStokesTimeIntegrationFactor());
      nrBytesRead = (size_t) ps.nrStations() * ps.nrChannelsPerSubband() * ps.nrSamplesPerChannel() * NR_POLARIZATIONS * sizeof(std::complex<float>);
      nrBytesWritten = (size_t) ps.nrIncoherentStokes() * nrTimes * ps.nrChannelsPerSubband() * sizeof(float);
    }

  }
}

