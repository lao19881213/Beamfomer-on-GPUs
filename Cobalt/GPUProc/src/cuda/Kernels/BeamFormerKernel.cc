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
//# $Id: BeamFormerKernel.cc 27077 2013-10-24 01:00:58Z amesfoort $

#include <lofar_config.h>

#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>

#include "BeamFormerKernel.h"

#include <Common/lofar_complex.h>
#include <Common/LofarLogger.h>
#include <GPUProc/global_defines.h>
#include <GPUProc/gpu_utils.h>
#include <CoInterface/BlockID.h>

#include <fstream>

using boost::lexical_cast;
using boost::format;

namespace LOFAR
{
  namespace Cobalt
  {
    string BeamFormerKernel::theirSourceFile = "BeamFormer.cu";
    string BeamFormerKernel::theirFunction = "beamFormer";

    BeamFormerKernel::Parameters::Parameters(const Parset& ps) :
      Kernel::Parameters(ps),
      nrSAPs(ps.settings.beamFormer.SAPs.size()),
      nrTABs(ps.settings.beamFormer.maxNrTABsPerSAP()),
      weightCorrection(1.0f), // TODO: pass FFT size
      subbandBandwidth(ps.settings.subbandWidth())
    {
      // override the correlator settings with beamformer specifics
      nrChannelsPerSubband = 
        ps.settings.beamFormer.coherentSettings.nrChannels;
      nrSamplesPerChannel =
        ps.settings.beamFormer.coherentSettings.nrSamples(ps.nrSamplesPerSubband());
      dumpBuffers = 
        ps.getBool("Cobalt.Kernels.BeamFormerKernel.dumpOutput", false);
      dumpFilePattern = 
        str(format("L%d_SB%%03d_BL%%03d_BeamFormerKernel.dat") % 
            ps.settings.observationID);
    }

    BeamFormerKernel::BeamFormerKernel(const gpu::Stream& stream,
                                       const gpu::Module& module,
                                       const Buffers& buffers,
                                       const Parameters& params) :
      Kernel(stream, gpu::Function(module, theirFunction), buffers, params)
    {
      setArg(0, buffers.output);
      setArg(1, buffers.input);
      setArg(2, buffers.beamFormerDelays);

      // Try #chnl/blk where 256 <= block size <= maxThreadsPerBlock && blockDim.z <= maxBlockDimZ.
      // Violations for small input are fine. This should be auto-tuned for large inputs.
      unsigned nrThreadsXY = params.nrPolarizations * params.nrTABs;
      unsigned prefBlockSize = 256;
      unsigned prefNrThreadsZ = (prefBlockSize + nrThreadsXY-1) / nrThreadsXY;
      prefBlockSize = prefNrThreadsZ * nrThreadsXY;
      const unsigned maxBlockDimZ = stream.getContext().getDevice().getMaxBlockDims().z; // low, so check
      unsigned maxNrThreadsPerBlock = std::min(std::min(maxThreadsPerBlock,
                                                        maxBlockDimZ * nrThreadsXY),
                                               params.nrChannelsPerSubband * nrThreadsXY);
      if (prefBlockSize > maxNrThreadsPerBlock) {
        prefBlockSize = maxNrThreadsPerBlock;
        prefNrThreadsZ = (prefBlockSize + nrThreadsXY-1) / nrThreadsXY;
      }
      unsigned nrChannelsPerBlock = prefNrThreadsZ;

      setEnqueueWorkSizes( gpu::Grid (params.nrPolarizations, params.nrTABs, params.nrChannelsPerSubband),
                           gpu::Block(params.nrPolarizations, params.nrTABs, nrChannelsPerBlock) );

#if 0
      size_t nrDelaysBytes = bufferSize(ps, BEAM_FORMER_DELAYS);
      size_t nrSampleBytesPerPass = bufferSize(ps, INPUT_DATA);
      size_t nrComplexVoltagesBytesPerPass = bufferSize(ps, OUTPUT_DATA);

      size_t count = 
        params.nrChannelsPerSubband * params.nrSamplesPerChannel * params.nrPolarizations;
      unsigned nrPasses = std::max((params.nrStations + 6) / 16, 1U);

      nrOperations = count * params.nrStations * params.nrTABs * 8;
      nrBytesRead = 
        nrDelaysBytes + nrSampleBytesPerPass + (nrPasses - 1) * 
        nrComplexVoltagesBytesPerPass;
      nrBytesWritten = nrPasses * nrComplexVoltagesBytesPerPass;
#endif
    }

    void BeamFormerKernel::enqueue(const BlockID &blockId,
                                   PerformanceCounter &counter,
                                   double subbandFrequency, unsigned SAP)
    {
      setArg(3, subbandFrequency);
      setArg(4, SAP);
      Kernel::enqueue(blockId, counter);
    }

    //--------  Template specializations for KernelFactory  --------//

    template<> size_t 
    KernelFactory<BeamFormerKernel>::bufferSize(BufferType bufferType) const
    {
      switch (bufferType) {
      case BeamFormerKernel::INPUT_DATA: 
        return
          (size_t) itsParameters.nrChannelsPerSubband * itsParameters.nrSamplesPerChannel *
            itsParameters.nrPolarizations * itsParameters.nrStations * sizeof(std::complex<float>);
      case BeamFormerKernel::OUTPUT_DATA:
        return
          (size_t) itsParameters.nrChannelsPerSubband * itsParameters.nrSamplesPerChannel *
            itsParameters.nrPolarizations * itsParameters.nrTABs * sizeof(std::complex<float>);
      case BeamFormerKernel::BEAM_FORMER_DELAYS:
        return 
          (size_t) itsParameters.nrSAPs * itsParameters.nrStations * itsParameters.nrTABs *
            sizeof(double);
      default:
        THROW(GPUProcException, "Invalid bufferType (" << bufferType << ")");
      }
    }

    
    
    template<> CompileDefinitions
    KernelFactory<BeamFormerKernel>::compileDefinitions() const
    {
      CompileDefinitions defs =
        KernelFactoryBase::compileDefinitions(itsParameters);
      defs["NR_SAPS"] =
        lexical_cast<string>(itsParameters.nrSAPs);
      defs["NR_TABS"] =
        lexical_cast<string>(itsParameters.nrTABs);
      defs["WEIGHT_CORRECTION"] =
        str(format("%.7ff") % itsParameters.weightCorrection);
      defs["SUBBAND_BANDWIDTH"] =
        str(format("%.7f") % itsParameters.subbandBandwidth);

      return defs;
    }
  }
}

