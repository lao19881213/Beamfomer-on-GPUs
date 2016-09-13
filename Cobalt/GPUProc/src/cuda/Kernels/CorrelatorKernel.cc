//# CorrelatorKernel.cc
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
//# $Id: CorrelatorKernel.cc 27473 2013-11-21 12:36:36Z mol $

#include <lofar_config.h>

#include "CorrelatorKernel.h"

#include <GPUProc/global_defines.h>
#include <GPUProc/gpu_utils.h>
#include <CoInterface/BlockID.h>
#include <Common/lofar_complex.h>
#include <Common/LofarLogger.h>

#include <boost/format.hpp>

#include <fstream>

using boost::format;

// For Cobalt (= up to 80 antenna fields), the 2x2 kernel gives the best
// performance.

#define USE_2X2

namespace LOFAR
{
  namespace Cobalt
  {
    string CorrelatorKernel::theirSourceFile = "Correlator.cu";
# if defined USE_4X4
    string CorrelatorKernel::theirFunction = "correlate_4x4";
# elif defined USE_3X3
    string CorrelatorKernel::theirFunction = "correlate_3x3";
# elif defined USE_2X2
    string CorrelatorKernel::theirFunction = "correlate_2x2";
# else
    string CorrelatorKernel::theirFunction = "correlate";
# endif

    CorrelatorKernel::Parameters::Parameters(const Parset& ps) :
      Kernel::Parameters(ps)
    {
      dumpBuffers = 
        ps.getBool("Cobalt.Kernels.CorrelatorKernel.dumpOutput", false);
      dumpFilePattern = 
        str(format("L%d_SB%%03d_BL%%03d_CorrelatorKernel.dat") % 
            ps.settings.observationID);
    }

    CorrelatorKernel::CorrelatorKernel(const gpu::Stream& stream,
                                       const gpu::Module& module,
                                       const Buffers& buffers,
                                       const Parameters& params) :
      Kernel(stream, gpu::Function(module, theirFunction), buffers, params)
    {
      setArg(0, buffers.output);
      setArg(1, buffers.input);

      unsigned preferredMultiple;

      gpu::Platform pf;
      if (pf.getName() == "AMD Accelerated Parallel Processing") {
        preferredMultiple = 256;
      } else {
        preferredMultiple = 64; // FOR NVIDIA CUDA, there is no call to get this. Could check what the NV OCL pf says, but set to 64 for now. Generalize later (TODO).
      }

      unsigned nrBaselines = params.nrStations * (params.nrStations + 1) / 2;

# if defined USE_4X4
      unsigned quartStations = (params.nrStations + 3) / 4;
      unsigned nrBlocks = quartStations * (quartStations + 1) / 2;
# elif defined USE_3X3
      unsigned thirdStations = (params.nrStations + 2) / 3;
      unsigned nrBlocks = thirdStations * (thirdStations + 1) / 2;
# elif defined USE_2X2
      unsigned halfStations = (params.nrStations + 1) / 2;
      unsigned nrBlocks = halfStations * (halfStations + 1) / 2;
# else
      unsigned nrBlocks = nrBaselines;
# endif
      unsigned nrPasses = (nrBlocks + maxThreadsPerBlock - 1) / maxThreadsPerBlock;
      unsigned nrThreads = (nrBlocks + nrPasses - 1) / nrPasses;
      nrThreads = (nrThreads + preferredMultiple - 1) / preferredMultiple * preferredMultiple;

      //LOG_DEBUG_STR("nrBlocks = " << nrBlocks << ", nrPasses = " << nrPasses << ", preferredMultiple = " << preferredMultiple << ", nrThreads = " << nrThreads);

      unsigned nrUsableChannels = std::max(params.nrChannelsPerSubband - 1, 1U);
      setEnqueueWorkSizes( gpu::Grid(nrPasses * nrThreads, nrUsableChannels),
                           gpu::Block(nrThreads, 1) );

      nrOperations = (size_t) nrUsableChannels * nrBaselines * params.nrSamplesPerChannel * 32;
      nrBytesRead = (size_t) nrPasses * params.nrStations * nrUsableChannels * params.nrSamplesPerChannel * NR_POLARIZATIONS * sizeof(std::complex<float>);
      nrBytesWritten = (size_t) nrBaselines * nrUsableChannels * NR_POLARIZATIONS * NR_POLARIZATIONS * sizeof(std::complex<float>);
    }

    //--------  Template specializations for KernelFactory  --------//

    template<> size_t 
    KernelFactory<CorrelatorKernel>::bufferSize(BufferType bufferType) const
    {
      unsigned nrBaselines = itsParameters.nrStations * (itsParameters.nrStations + 1) / 2;

      switch (bufferType) {
      case CorrelatorKernel::INPUT_DATA:
        return
          (size_t) itsParameters.nrSamplesPerSubband * itsParameters.nrStations * 
            NR_POLARIZATIONS * sizeof(std::complex<float>);
      case CorrelatorKernel::OUTPUT_DATA:
        return 
          (size_t) nrBaselines * itsParameters.nrChannelsPerSubband * 
            NR_POLARIZATIONS * NR_POLARIZATIONS * sizeof(std::complex<float>);
      default: 
        THROW(GPUProcException, "Invalid bufferType (" << bufferType << ")");
      }
    }
  }
}

