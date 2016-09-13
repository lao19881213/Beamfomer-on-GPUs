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
//# $Id: DelayAndBandPassKernel.cc 27477 2013-11-21 13:08:20Z loose $

#include <lofar_config.h>

#include "DelayAndBandPassKernel.h"

#include <GPUProc/global_defines.h>
#include <GPUProc/gpu_utils.h>
#include <GPUProc/BandPass.h>
#include <CoInterface/BlockID.h>
#include <Common/lofar_complex.h>
#include <Common/LofarLogger.h>

#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>

#include <fstream>

using boost::lexical_cast;
using boost::format;

namespace LOFAR
{
  namespace Cobalt
  {
    string DelayAndBandPassKernel::theirSourceFile = "DelayAndBandPass.cu";
    string DelayAndBandPassKernel::theirFunction = "applyDelaysAndCorrectBandPass";

    DelayAndBandPassKernel::Parameters::Parameters(const Parset& ps) :
      Kernel::Parameters(ps),
      nrBitsPerSample(ps.settings.nrBitsPerSample),
      nrBytesPerComplexSample(ps.nrBytesPerComplexSample()),
      nrSAPs(ps.settings.SAPs.size()),
      delayCompensation(ps.settings.delayCompensation.enabled),
      correctBandPass(ps.settings.corrections.bandPass),
      transpose(correctBandPass), // sane for correlator; bf redefines
      subbandBandwidth(ps.settings.subbandWidth())
    {
      dumpBuffers = 
        ps.getBool("Cobalt.Kernels.DelayAndBandPassKernel.dumpOutput", false);
      dumpFilePattern = 
        str(format("L%d_SB%%03d_BL%%03d_DelayAndBandPassKernel_%c%c%c.dat") % 
            ps.settings.observationID %
            (correctBandPass ? "B" : "b") %
            (delayCompensation ? "D" : "d") %
            (transpose ? "T" : "t"));
    }

    DelayAndBandPassKernel::DelayAndBandPassKernel(const gpu::Stream& stream,
                                       const gpu::Module& module,
                                       const Buffers& buffers,
                                       const Parameters& params) :
      Kernel(stream, gpu::Function(module, theirFunction), buffers, params)
    {
      LOG_DEBUG_STR("DelayAndBandPassKernel: " <<
                    "correctBandPass=" << 
                    (params.correctBandPass ? "true" : "false") <<
                    ", delayCompensation=" <<
                    (params.delayCompensation ? "true" : "false") <<
                    ", transpose=" << (params.transpose ? "true" : "false"));

      ASSERT(params.nrChannelsPerSubband % 16 == 0 || params.nrChannelsPerSubband == 1);
      ASSERT(params.nrSamplesPerChannel % 16 == 0);

      setArg(0, buffers.output);
      setArg(1, buffers.input);
      setArg(4, buffers.delaysAtBegin);
      setArg(5, buffers.delaysAfterEnd);
      setArg(6, buffers.phaseOffsets);
      setArg(7, buffers.bandPassCorrectionWeights);

      setEnqueueWorkSizes( gpu::Grid(256, params.nrChannelsPerSubband == 1 ? 1 : params.nrChannelsPerSubband / 16, params.nrStations),
                           gpu::Block(256, 1, 1) );

      size_t nrSamples = (size_t)params.nrStations * params.nrChannelsPerSubband * params.nrSamplesPerChannel * NR_POLARIZATIONS;
      nrOperations = nrSamples * 12;
      nrBytesRead = nrBytesWritten = nrSamples * sizeof(std::complex<float>);

      // Initialise bandpass correction weights
      if (params.correctBandPass)
      {
        gpu::HostMemory bpWeights(stream.getContext(), buffers.bandPassCorrectionWeights.size());
        BandPass::computeCorrectionFactors(bpWeights.get<float>(), params.nrChannelsPerSubband);
        stream.writeBuffer(buffers.bandPassCorrectionWeights, bpWeights, true);
      }
    }


    void DelayAndBandPassKernel::enqueue(const BlockID &blockId,
                                         PerformanceCounter &counter,
                                         double subbandFrequency, unsigned SAP)
    {
      setArg(2, subbandFrequency);
      setArg(3, SAP);
      Kernel::enqueue(blockId, counter);
    }

    //--------  Template specializations for KernelFactory  --------//

    template<> size_t 
    KernelFactory<DelayAndBandPassKernel>::bufferSize(BufferType bufferType) const
    {
      switch (bufferType) {
      case DelayAndBandPassKernel::INPUT_DATA: 
        if (itsParameters.nrChannelsPerSubband == 1)
          return 
            (size_t) itsParameters.nrStations * NR_POLARIZATIONS * 
              itsParameters.nrSamplesPerSubband *
              itsParameters.nrBytesPerComplexSample;
        else
          return 
            (size_t) itsParameters.nrStations * NR_POLARIZATIONS * 
              itsParameters.nrSamplesPerSubband * sizeof(std::complex<float>);
      case DelayAndBandPassKernel::OUTPUT_DATA:
        return
          (size_t) itsParameters.nrStations * NR_POLARIZATIONS * 
            itsParameters.nrSamplesPerSubband * sizeof(std::complex<float>);
      case DelayAndBandPassKernel::DELAYS:
        return 
          (size_t) itsParameters.nrSAPs * itsParameters.nrStations * 
            NR_POLARIZATIONS * sizeof(double);
      case DelayAndBandPassKernel::PHASE_OFFSETS:
        return
          (size_t) itsParameters.nrStations * NR_POLARIZATIONS * sizeof(double);
      case DelayAndBandPassKernel::BAND_PASS_CORRECTION_WEIGHTS:
        return
          (size_t) itsParameters.nrChannelsPerSubband * sizeof(float);
      default:
        THROW(GPUProcException, "Invalid bufferType (" << bufferType << ")");
      }
    }

    template<> CompileDefinitions
    KernelFactory<DelayAndBandPassKernel>::compileDefinitions() const
    {
      CompileDefinitions defs =
        KernelFactoryBase::compileDefinitions(itsParameters);
      defs["NR_BITS_PER_SAMPLE"] =
        lexical_cast<string>(itsParameters.nrBitsPerSample);
      defs["NR_SAPS"] =
        lexical_cast<string>(itsParameters.nrSAPs);
      defs["SUBBAND_BANDWIDTH"] =
        str(format("%.7f") % itsParameters.subbandBandwidth);

      if (itsParameters.delayCompensation) {
        defs["DELAY_COMPENSATION"] = "1";
      }

      if (itsParameters.correctBandPass) {
        defs["BANDPASS_CORRECTION"] = "1";
      }

      if (itsParameters.transpose) {
        defs["DO_TRANSPOSE"] = "1";
      }

      return defs;
    }
  }
}
