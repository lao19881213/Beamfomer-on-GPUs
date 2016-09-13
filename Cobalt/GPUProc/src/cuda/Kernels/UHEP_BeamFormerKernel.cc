//# UHEP_BeamFormerKernel.cc
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
//# $Id: UHEP_BeamFormerKernel.cc 27077 2013-10-24 01:00:58Z amesfoort $

#include <lofar_config.h>

#include "UHEP_BeamFormerKernel.h"

#include <algorithm>

#include <Common/lofar_complex.h>

#include <GPUProc/global_defines.h>

namespace LOFAR
{
  namespace Cobalt
  {
    UHEP_BeamFormerKernel::UHEP_BeamFormerKernel(const Parset &ps, gpu::Module &program, gpu::DeviceMemory &devComplexVoltages, gpu::DeviceMemory &devInputSamples, gpu::DeviceMemory &devBeamFormerWeights)
      :
      Kernel(ps, program, "complexVoltages")
    {
      setArg(0, devComplexVoltages);
      setArg(1, devInputSamples);
      setArg(2, devBeamFormerWeights);

#if 1
      setEnqueueWorkSizes( gpu::Grid(NR_POLARIZATIONS, ps.nrTABs(0), ps.nrSubbands()),
                           gpu::Block(NR_POLARIZATIONS, ps.nrTABs(0), 1) );

      size_t count = ps.nrSubbands() * (ps.nrSamplesPerChannel() + NR_STATION_FILTER_TAPS - 1) * NR_POLARIZATIONS;
      size_t nrWeightsBytes = ps.nrStations() * ps.nrTABs(0) * ps.nrSubbands() * NR_POLARIZATIONS * sizeof(std::complex<float>);
      size_t nrSampleBytes = count * ps.nrStations() * ps.nrBytesPerComplexSample();
      size_t nrComplexVoltagesBytesPerPass = count * ps.nrTABs(0) * sizeof(std::complex<float>);
      unsigned nrPasses = std::max((ps.nrStations() + 6) / 16, 1U);
      nrOperations = count * ps.nrStations() * ps.nrTABs(0) * 8;
      nrBytesRead = nrWeightsBytes + nrSampleBytes + (nrPasses - 1) * nrComplexVoltagesBytesPerPass;
      nrBytesWritten = nrPasses * nrComplexVoltagesBytesPerPass;
#else
      ASSERT(ps.nrTABs(0) % 3 == 0);
      ASSERT(ps.nrStations() % 6 == 0);
      unsigned nrThreads = NR_POLARIZATIONS * (ps.nrTABs(0) / 3) * (ps.nrStations() / 6);
      globalWorkSize = gpu::Grid(nrThreads, ps.nrSubbands());
      localWorkSize = gpu::Block(nrThreads, 1);
      //globalWorkSize = gpu::Grid(ps.nrStations() / 6, ps.nrTABs(0) / 3, ps.nrSubbands());
      //localWorkSize  = gpu::dim3(ps.nrStations() / 6, ps.nrTABs(0) / 3, 1);

      size_t count = ps.nrSubbands() * (ps.nrSamplesPerChannel() + NR_STATION_FILTER_TAPS - 1) * NR_POLARIZATIONS;
      size_t nrWeightsBytes = ps.nrStations() * ps.nrTABs(0) * ps.nrSubbands() * NR_POLARIZATIONS * sizeof(std::complex<float>);
      size_t nrSampleBytes = count * ps.nrStations() * ps.nrBytesPerComplexSample();
      size_t nrComplexVoltagesBytes = count * ps.nrTABs(0) * sizeof(std::complex<float>);
      nrOperations = count * ps.nrStations() * ps.nrTABs(0) * 8;
      nrBytesRead = nrWeightsBytes + nrSampleBytes;
      nrBytesWritten = nrComplexVoltagesBytes;
#endif
    }
  }
}

