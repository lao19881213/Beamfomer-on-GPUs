//# DelayAndBandPassKernel.cc
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
//# $Id: DelayAndBandPassKernel.cc 25358 2013-06-18 09:05:40Z loose $

#include <lofar_config.h>

#include "DelayAndBandPassKernel.h"

#include <Common/lofar_complex.h>
#include <Common/LofarLogger.h>

#include <GPUProc/global_defines.h>

namespace LOFAR
{
  namespace Cobalt
  {
    DelayAndBandPassKernel::DelayAndBandPassKernel(const Parset &ps, cl::Program &program,
                                                   cl::Buffer &devCorrectedData, cl::Buffer &devFilteredData,
                                                   cl::Buffer &devDelaysAtBegin, cl::Buffer &devDelaysAfterEnd,
                                                   cl::Buffer &devPhaseOffsets, cl::Buffer &devBandPassCorrectionWeights)
      :
      Kernel(ps, program, "applyDelaysAndCorrectBandPass")
    {
      ASSERT(ps.nrChannelsPerSubband() % 16 == 0 || ps.nrChannelsPerSubband() == 1);
      ASSERT(ps.nrSamplesPerChannel() % 16 == 0);

      setArg(0, devCorrectedData);
      setArg(1, devFilteredData);
      setArg(4, devDelaysAtBegin);
      setArg(5, devDelaysAfterEnd);
      setArg(6, devPhaseOffsets);
      setArg(7, devBandPassCorrectionWeights);

      globalWorkSize = cl::NDRange(256, ps.nrChannelsPerSubband() == 1 ? 1 : ps.nrChannelsPerSubband() / 16, ps.nrStations());
      localWorkSize = cl::NDRange(256, 1, 1);

      size_t nrSamples = ps.nrStations() * ps.nrChannelsPerSubband() * ps.nrSamplesPerChannel() * NR_POLARIZATIONS;
      nrOperations = nrSamples * 12;
      nrBytesRead = nrBytesWritten = nrSamples * sizeof(std::complex<float>);
    }

    void DelayAndBandPassKernel::enqueue(cl::CommandQueue &queue, PerformanceCounter &counter, unsigned subband)
    {
      setArg(2, (float) ps.subbandToFrequencyMapping()[subband]);
      setArg(3, 0);       // beam
      Kernel::enqueue(queue, counter);
    }

    size_t
    DelayAndBandPassKernel::bufferSize(const Parset& ps, BufferType bufferType)
    {
      switch (bufferType) {
      case INPUT_DATA: 
        if (ps.nrChannelsPerSubband() == 1)
          return 
            ps.nrStations() * NR_POLARIZATIONS * 
            ps.nrSamplesPerSubband() * ps.nrBytesPerComplexSample();
        else
          return 
            ps.nrStations() * NR_POLARIZATIONS * 
            ps.nrSamplesPerSubband() * sizeof(std::complex<float>);
      case OUTPUT_DATA:
        return
          ps.nrStations() * NR_POLARIZATIONS * 
          ps.nrSamplesPerSubband() * sizeof(std::complex<float>);
      case DELAYS:
        return 
          ps.nrBeams() * ps.nrStations() * NR_POLARIZATIONS * sizeof(float);
      case PHASE_OFFSETS:
        return
          ps.nrStations() * NR_POLARIZATIONS * sizeof(float);
      case BAND_PASS_CORRECTION_WEIGHTS:
        return
          ps.nrChannelsPerSubband() * sizeof(float);
      default:
        THROW(GPUProcException, "Invalid bufferType (" << bufferType << ")");
      }
    }

  }
}

