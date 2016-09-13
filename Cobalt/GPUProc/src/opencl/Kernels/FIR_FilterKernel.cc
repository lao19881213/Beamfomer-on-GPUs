//# FIR_FilterKernel.cc
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
//# $Id: FIR_FilterKernel.cc 25358 2013-06-18 09:05:40Z loose $

#include <lofar_config.h>

#include "FIR_FilterKernel.h"

#include <Common/lofar_complex.h>

#include <GPUProc/global_defines.h>

namespace LOFAR
{
  namespace Cobalt
  {
    FIR_FilterKernel::FIR_FilterKernel(const Parset &ps, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &devFilteredData, cl::Buffer &devInputSamples, cl::Buffer &devFIRweights)
      :
      Kernel(ps, program, "FIR_filter")
    {
      setArg(0, devFilteredData);
      setArg(1, devInputSamples);
      setArg(2, devFIRweights);

      size_t maxNrThreads;
      getWorkGroupInfo(queue.getInfo<CL_QUEUE_DEVICE>(), CL_KERNEL_WORK_GROUP_SIZE, &maxNrThreads);
      unsigned totalNrThreads = ps.nrChannelsPerSubband() * NR_POLARIZATIONS * 2;
      unsigned nrPasses = (totalNrThreads + maxNrThreads - 1) / maxNrThreads;
      globalWorkSize = cl::NDRange(totalNrThreads, ps.nrStations());
      localWorkSize = cl::NDRange(totalNrThreads / nrPasses, 1);

      size_t nrSamples = (size_t) ps.nrStations() * ps.nrChannelsPerSubband() * NR_POLARIZATIONS;
      nrOperations = nrSamples * ps.nrSamplesPerChannel() * NR_TAPS * 2 * 2;
      nrBytesRead = nrSamples * (NR_TAPS - 1 + ps.nrSamplesPerChannel()) * ps.nrBytesPerComplexSample();
      nrBytesWritten = nrSamples * ps.nrSamplesPerChannel() * sizeof(std::complex<float>);
    }

    size_t FIR_FilterKernel::bufferSize(const Parset& ps, BufferType bufferType)
    {
      switch (bufferType) {
      case INPUT_DATA: 
        return
          (ps.nrHistorySamples() + ps.nrSamplesPerSubband()) * 
          ps.nrStations() * NR_POLARIZATIONS * ps.nrBytesPerComplexSample();
      case OUTPUT_DATA:
        return
          ps.nrSamplesPerSubband() * ps.nrStations() * 
          NR_POLARIZATIONS * sizeof(std::complex<float>);
      case FILTER_WEIGHTS:
        return 
          ps.nrChannelsPerSubband() * NR_TAPS * sizeof(float);
      default:
        THROW(GPUProcException, "Invalid bufferType (" << bufferType << ")");
      }
    }

  }
}

