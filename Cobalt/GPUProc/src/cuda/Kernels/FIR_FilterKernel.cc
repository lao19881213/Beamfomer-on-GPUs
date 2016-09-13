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
//# $Id: FIR_FilterKernel.cc 27077 2013-10-24 01:00:58Z amesfoort $

#include <lofar_config.h>

#include "FIR_FilterKernel.h"
#include <GPUProc/global_defines.h>
#include <GPUProc/gpu_utils.h>
#include <CoInterface/BlockID.h>

#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>

#include <complex>
#include <fstream>

using namespace std;
using boost::lexical_cast;
using boost::format;

namespace LOFAR
{
  namespace Cobalt
  {
    string FIR_FilterKernel::theirSourceFile = "FIR_Filter.cu";
    string FIR_FilterKernel::theirFunction = "FIR_filter";

    FIR_FilterKernel::Parameters::Parameters(const Parset& ps) :
      Kernel::Parameters(ps),
      nrBitsPerSample(ps.nrBitsPerSample()),
      nrBytesPerComplexSample(ps.nrBytesPerComplexSample()),
      nrSTABs(nrStations), // default to filter station data
      nrSubbands(1)
    {
      dumpBuffers = 
        ps.getBool("Cobalt.Kernels.FIR_FilterKernel.dumpOutput", false);
      dumpFilePattern = 
        str(format("L%d_SB%%03d_BL%%03d_FIR_FilterKernel.dat") % 
            ps.settings.observationID);

    }

    const unsigned FIR_FilterKernel::Parameters::nrTaps;

    unsigned FIR_FilterKernel::Parameters::nrHistorySamples() const
    {
      return (nrTaps - 1) * nrChannelsPerSubband;
    }

    FIR_FilterKernel::FIR_FilterKernel(const gpu::Stream& stream,
                                       const gpu::Module& module,
                                       const Buffers& buffers,
                                       const Parameters& params) :
      Kernel(stream, gpu::Function(module, theirFunction), buffers, params),
      params(params),
      historyFlags(boost::extents[params.nrSubbands][params.nrSTABs])
    {
      setArg(0, buffers.output);
      setArg(1, buffers.input);
      setArg(2, buffers.filterWeights);
      setArg(3, buffers.historySamples);

      unsigned maxNrThreads = 
        getAttribute(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK);

      unsigned totalNrThreads = 
        params.nrChannelsPerSubband * params.nrPolarizations * 2;
      unsigned nrPasses = (totalNrThreads + maxNrThreads - 1) / maxNrThreads;

      setEnqueueWorkSizes( gpu::Grid(totalNrThreads, params.nrSTABs),
                           gpu::Block(totalNrThreads / nrPasses, 1) );

      unsigned nrSamples = 
        params.nrSTABs * params.nrChannelsPerSubband * 
        params.nrPolarizations;

      nrOperations = 
        (size_t) nrSamples * params.nrSamplesPerChannel * params.nrTaps * 2 * 2;

      nrBytesRead = 
        (size_t) nrSamples * (params.nrTaps - 1 + params.nrSamplesPerChannel) * 
          params.nrBytesPerComplexSample;

      nrBytesWritten = 
        (size_t) nrSamples * params.nrSamplesPerChannel * sizeof(std::complex<float>);

      // Note that these constant weights are now (unnecessarily) stored on the
      // device for every workqueue. A single copy per device could be used, but
      // first verify that the device platform still allows workqueue overlap.
      FilterBank filterBank(true, params.nrTaps, 
                            params.nrChannelsPerSubband, KAISER);
      filterBank.negateWeights();

      gpu::HostMemory firWeights(stream.getContext(), buffers.filterWeights.size());
      std::memcpy(firWeights.get<void>(), filterBank.getWeights().origin(),
                  firWeights.size());
      stream.writeBuffer(buffers.filterWeights, firWeights, true);

      // start with all history samples flagged
      for (size_t n = 0; n < historyFlags.num_elements(); ++n)
        historyFlags.origin()[n].include(0, params.nrHistorySamples());
    }

    void FIR_FilterKernel::enqueue(const BlockID &blockId,
                                   PerformanceCounter &counter,
                                   unsigned subbandIdx)
    {
      setArg(4, subbandIdx);
      Kernel::enqueue(blockId, counter);
    }

    void FIR_FilterKernel::prefixHistoryFlags(MultiDimArray<SparseSet<unsigned>, 1> &inputFlags, unsigned subbandIdx) {
      for (unsigned stationIdx = 0; stationIdx < params.nrSTABs; ++stationIdx) {
        // shift sample flags to the right to make room for the history flags
        inputFlags[stationIdx] += params.nrHistorySamples();

        // add the history flags.
        inputFlags[stationIdx] |= historyFlags[subbandIdx][stationIdx];

        // Save the new history flags for the next block.
        // Note that the nrSamples is the number of samples
        // WITHOUT history samples, but we've also just shifted everything
        // by nrHistorySamples.
        historyFlags[subbandIdx][stationIdx] =
          inputFlags[stationIdx].subset(params.nrSamplesPerSubband, params.nrSamplesPerSubband + params.nrHistorySamples());

        // Shift the flags to index 0
        historyFlags[subbandIdx][stationIdx] -= params.nrSamplesPerSubband;
      }
    }

    //--------  Template specializations for KernelFactory  --------//

    template<> size_t 
    KernelFactory<FIR_FilterKernel>::bufferSize(BufferType bufferType) const
    {
      switch (bufferType) {
      case FIR_FilterKernel::INPUT_DATA: 
        return
          (size_t) itsParameters.nrSamplesPerSubband *
            itsParameters.nrSTABs * itsParameters.nrPolarizations * 
            itsParameters.nrBytesPerComplexSample;
      case FIR_FilterKernel::OUTPUT_DATA:
        return
          (size_t) itsParameters.nrSamplesPerSubband * itsParameters.nrSTABs * 
            itsParameters.nrPolarizations * sizeof(std::complex<float>);
      case FIR_FilterKernel::FILTER_WEIGHTS:
        return 
          (size_t) itsParameters.nrChannelsPerSubband * itsParameters.nrTaps *
            sizeof(float);
      case FIR_FilterKernel::HISTORY_DATA:
        return
          (size_t) itsParameters.nrSubbands *
            itsParameters.nrHistorySamples() * itsParameters.nrSTABs * 
            itsParameters.nrPolarizations * itsParameters.nrBytesPerComplexSample;
      default:
        THROW(GPUProcException, "Invalid bufferType (" << bufferType << ")");
      }
    }

    template<> CompileDefinitions
    KernelFactory<FIR_FilterKernel>::compileDefinitions() const
    {
      CompileDefinitions defs =
        KernelFactoryBase::compileDefinitions(itsParameters);

      defs["NR_BITS_PER_SAMPLE"] =
        lexical_cast<string>(itsParameters.nrBitsPerSample);
      defs["NR_TAPS"] = 
        lexical_cast<string>(itsParameters.nrTaps);
      defs["NR_STABS"] = 
        lexical_cast<string>(itsParameters.nrSTABs);
      defs["NR_SUBBANDS"] = 
        lexical_cast<string>(itsParameters.nrSubbands);

      return defs;
    }
  }
}

