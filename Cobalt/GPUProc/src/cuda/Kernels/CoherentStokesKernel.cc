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
//# $Id: CoherentStokesKernel.cc 27077 2013-10-24 01:00:58Z amesfoort $

#include <lofar_config.h>

#include "CoherentStokesKernel.h"

#include <GPUProc/global_defines.h>
#include <GPUProc/gpu_utils.h>
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
    string CoherentStokesKernel::theirSourceFile = "CoherentStokes.cu";
    string CoherentStokesKernel::theirFunction = "coherentStokes";

    CoherentStokesKernel::Parameters::Parameters(const Parset& ps) :
      Kernel::Parameters(ps),
      nrTABs(ps.settings.beamFormer.maxNrTABsPerSAP()),
      nrStokes(ps.settings.beamFormer.coherentSettings.nrStokes),
      timeIntegrationFactor(
        ps.settings.beamFormer.coherentSettings.timeIntegrationFactor)
     {
      nrChannelsPerSubband = ps.settings.beamFormer.coherentSettings.nrChannels;
      nrSamplesPerChannel  = ps.settings.beamFormer.coherentSettings.nrSamples(ps.nrSamplesPerSubband());

      timeParallelFactor = gpu::Platform().getMaxThreadsPerBlock() / (nrTABs * nrChannelsPerSubband);
      dumpBuffers = 
        ps.getBool("Cobalt.Kernels.CoherentStokesKernel.dumpOutput", false);
      dumpFilePattern = 
        str(format("L%d_SB%%03d_BL%%03d_CoherentStokesKernel.dat") % 
            ps.settings.observationID);

    }


    CoherentStokesKernel::CoherentStokesKernel(const gpu::Stream& stream,
                                       const gpu::Module& module,
                                       const Buffers& buffers,
                                       const Parameters& params) :
      Kernel(stream, gpu::Function(module, theirFunction), buffers, params)
    {
      ASSERT(params.nrChannelsPerSubband == 1 || (params.nrChannelsPerSubband >= 16 && params.nrChannelsPerSubband % 16 == 0));
      ASSERT(params.timeIntegrationFactor > 0 && params.nrSamplesPerChannel % params.timeIntegrationFactor == 0);
      ASSERT(params.nrStokes == 1 || params.nrStokes == 4);
      setArg(0, buffers.output);
      setArg(1, buffers.input);

      // TODO: params.nrTABs only works for one SAP
      // TODO: this enqueues 1 block, which is very inefficient
      setEnqueueWorkSizes( gpu::Grid (params.nrTABs, params.timeParallelFactor, params.nrChannelsPerSubband),
                           gpu::Block(params.nrTABs, params.timeParallelFactor, params.nrChannelsPerSubband) );

      nrOperations = (size_t) params.nrChannelsPerSubband * params.nrSamplesPerChannel * params.nrTABs * (params.nrStokes == 1 ? 8 : 20 + 2.0 / params.timeIntegrationFactor);
      nrBytesRead = (size_t) params.nrChannelsPerSubband * params.nrSamplesPerChannel * params.nrTABs * NR_POLARIZATIONS * sizeof(std::complex<float>);
      nrBytesWritten = (size_t) params.nrTABs * params.nrStokes * params.nrSamplesPerChannel / params.timeIntegrationFactor * params.nrChannelsPerSubband * sizeof(float);
    }

    //--------  Template specializations for KernelFactory  --------//

    template<> size_t
    KernelFactory<CoherentStokesKernel>::bufferSize(BufferType bufferType) const
    {
      switch (bufferType) {
      case CoherentStokesKernel::INPUT_DATA:
        return
          (size_t) itsParameters.nrChannelsPerSubband * itsParameters.nrSamplesPerChannel *
            NR_POLARIZATIONS * itsParameters.nrTABs * sizeof(std::complex<float>);
      case CoherentStokesKernel::OUTPUT_DATA:
        return 
          (size_t) itsParameters.nrTABs * itsParameters.nrStokes * itsParameters.nrSamplesPerChannel /
            itsParameters.timeIntegrationFactor * itsParameters.nrChannelsPerSubband * sizeof(float);
      default:
        THROW(GPUProcException, "Invalid bufferType (" << bufferType << ")");
      }
    }

    template<> CompileDefinitions
    KernelFactory<CoherentStokesKernel>::compileDefinitions() const
    {
      CompileDefinitions defs =
        KernelFactoryBase::compileDefinitions(itsParameters);
      defs["NR_TABS"] =
        lexical_cast<string>(itsParameters.nrTABs);
      defs["NR_COHERENT_STOKES"] =
        lexical_cast<string>(itsParameters.nrStokes); // TODO: nrStokes and timeIntegrationFactor cannot differentiate between coh and incoh, while there are separate defines for coh and incoh. Correct?
      defs["INTEGRATION_SIZE"] =
        lexical_cast<string>(itsParameters.timeIntegrationFactor);
      defs["TIME_PARALLEL_FACTOR"] =
        lexical_cast<string>(itsParameters.timeParallelFactor);

      return defs;
    }

  }
}

