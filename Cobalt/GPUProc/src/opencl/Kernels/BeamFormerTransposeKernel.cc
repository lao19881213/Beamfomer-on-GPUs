//# BeamFormerTransposeKernel.cc
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
//# $Id: BeamFormerTransposeKernel.cc 24977 2013-05-21 13:46:16Z amesfoort $

#include <lofar_config.h>

#include "BeamFormerTransposeKernel.h"

#include <Common/lofar_complex.h>
#include <Common/LofarLogger.h>

#include <GPUProc/global_defines.h>

namespace LOFAR
{
  namespace Cobalt
  {

    BeamFormerTransposeKernel::BeamFormerTransposeKernel(const Parset &ps, cl::Program &program, cl::Buffer &devTransposedData, cl::Buffer &devComplexVoltages)
      :
      Kernel(ps, program, "transposeComplexVoltages")
    {
      ASSERT(ps.nrSamplesPerChannel() % 16 == 0);
      setArg(0, devTransposedData);
      setArg(1, devComplexVoltages);

      //globalWorkSize = cl::NDRange(256, (ps.nrTABs(0) + 15) / 16, (ps.nrChannelsPerSubband() + 15) / 16);
      globalWorkSize = cl::NDRange(256, (ps.nrTABs(0) + 15) / 16, ps.nrSamplesPerChannel() / 16);
      localWorkSize = cl::NDRange(256, 1, 1);

      nrOperations = 0;
      nrBytesRead = (size_t) ps.nrChannelsPerSubband() * ps.nrSamplesPerChannel() * ps.nrTABs(0) * NR_POLARIZATIONS * sizeof(std::complex<float>),
      //nrBytesWritten = (size_t) ps.nrTABs(0) * NR_POLARIZATIONS * ps.nrSamplesPerChannel() * ps.nrChannelsPerSubband() * sizeof(std::complex<float>);
      nrBytesWritten = (size_t) ps.nrTABs(0) * NR_POLARIZATIONS * ps.nrChannelsPerSubband() * ps.nrSamplesPerChannel() * sizeof(std::complex<float>);
    }

  }
}

