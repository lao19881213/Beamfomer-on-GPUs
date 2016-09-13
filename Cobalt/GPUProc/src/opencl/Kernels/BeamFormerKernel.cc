//# BeamFormerKernel.cc
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
//# $Id: BeamFormerKernel.cc 24977 2013-05-21 13:46:16Z amesfoort $

#include <lofar_config.h>

#include "BeamFormerKernel.h"

#include <algorithm>

#include <Common/lofar_complex.h>

#include <GPUProc/global_defines.h>

namespace LOFAR
{
  namespace Cobalt
  {
    BeamFormerKernel::BeamFormerKernel(const Parset &ps, cl::Program &program, cl::Buffer &devComplexVoltages, cl::Buffer &devCorrectedData, cl::Buffer &devBeamFormerWeights)
      :
      Kernel(ps, program, "complexVoltages")
    {
      setArg(0, devComplexVoltages);
      setArg(1, devCorrectedData);
      setArg(2, devBeamFormerWeights);

      globalWorkSize = cl::NDRange(NR_POLARIZATIONS, ps.nrTABs(0), ps.nrChannelsPerSubband());
      localWorkSize = cl::NDRange(NR_POLARIZATIONS, ps.nrTABs(0), 1);

      // FIXME: nrTABs
      //queue.enqueueNDRangeKernel(*this, cl::NullRange, cl::NDRange(16, ps.nrTABs(0), ps.nrChannelsPerSubband()), cl::NDRange(16, ps.nrTABs(0), 1), 0, &event);

      size_t count = ps.nrChannelsPerSubband() * ps.nrSamplesPerChannel() * NR_POLARIZATIONS;
      size_t nrWeightsBytes = ps.nrStations() * ps.nrTABs(0) * ps.nrChannelsPerSubband() * NR_POLARIZATIONS * sizeof(std::complex<float>);
      size_t nrSampleBytesPerPass = count * ps.nrStations() * sizeof(std::complex<float>);
      size_t nrComplexVoltagesBytesPerPass = count * ps.nrTABs(0) * sizeof(std::complex<float>);
      unsigned nrPasses = std::max((ps.nrStations() + 6) / 16, 1U);
      nrOperations = count * ps.nrStations() * ps.nrTABs(0) * 8;
      nrBytesRead = nrWeightsBytes + nrSampleBytesPerPass + (nrPasses - 1) * nrComplexVoltagesBytesPerPass;
      nrBytesWritten = nrPasses * nrComplexVoltagesBytesPerPass;
    }

  }
}

