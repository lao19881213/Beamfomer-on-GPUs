//# CoherentStokesKernel.cc
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
//# $Id: CoherentStokesKernel.cc 24977 2013-05-21 13:46:16Z amesfoort $

#include <lofar_config.h>

#include "CoherentStokesKernel.h"

#include <Common/lofar_complex.h>
#include <Common/LofarLogger.h>

#include <GPUProc/global_defines.h>

namespace LOFAR
{
  namespace Cobalt
  {

    CoherentStokesKernel::CoherentStokesKernel(const Parset &ps, cl::Program &program, cl::Buffer &devStokesData, cl::Buffer &devComplexVoltages)
      :
      Kernel(ps, program, "coherentStokes")
    {
      ASSERT(ps.nrChannelsPerSubband() >= 16 && ps.nrChannelsPerSubband() % 16 == 0);
      ASSERT(ps.nrCoherentStokes() == 1 || ps.nrCoherentStokes() == 4);
      setArg(0, devStokesData);
      setArg(1, devComplexVoltages);

      globalWorkSize = cl::NDRange(256, (ps.nrTABs(0) + 15) / 16, (ps.nrChannelsPerSubband() + 15) / 16);
      localWorkSize = cl::NDRange(256, 1, 1);

      nrOperations = (size_t) ps.nrChannelsPerSubband() * ps.nrSamplesPerChannel() * ps.nrTABs(0) * (ps.nrCoherentStokes() == 1 ? 8 : 20 + 2.0 / ps.coherentStokesTimeIntegrationFactor());
      nrBytesRead = (size_t) ps.nrChannelsPerSubband() * ps.nrSamplesPerChannel() * ps.nrTABs(0) * NR_POLARIZATIONS * sizeof(std::complex<float>);
      nrBytesWritten = (size_t) ps.nrTABs(0) * ps.nrCoherentStokes() * ps.nrSamplesPerChannel() / ps.coherentStokesTimeIntegrationFactor() * ps.nrChannelsPerSubband() * sizeof(float);
    }

  }
}

