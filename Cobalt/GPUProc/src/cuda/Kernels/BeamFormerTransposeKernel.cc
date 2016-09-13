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
//# $Id: BeamFormerTransposeKernel.cc 27077 2013-10-24 01:00:58Z amesfoort $

#include <lofar_config.h>

#include "BeamFormerTransposeKernel.h"

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
    string BeamFormerTransposeKernel::theirSourceFile = "Transpose.cu";
    string BeamFormerTransposeKernel::theirFunction = "transpose";

    BeamFormerTransposeKernel::Parameters::Parameters(const Parset& ps) :
      Kernel::Parameters(ps),
      nrTABs(ps.settings.beamFormer.maxNrTABsPerSAP())
    {
      nrChannelsPerSubband =
        ps.settings.beamFormer.coherentSettings.nrChannels;
      nrSamplesPerChannel =
        ps.settings.beamFormer.coherentSettings.nrSamples(ps.nrSamplesPerSubband());
      dumpBuffers = 
        ps.getBool("Cobalt.Kernels.BeamFormerTransposeKernel.dumpOutput", false);
      dumpFilePattern = 
        str(format("L%d_SB%%03d_BL%%03d_BeamFormerTransposeKernel.dat") % 
            ps.settings.observationID);

    }

    BeamFormerTransposeKernel::
    BeamFormerTransposeKernel(const gpu::Stream& stream,
                                       const gpu::Module& module,
                                       const Buffers& buffers,
                                       const Parameters& params) :
      Kernel(stream, gpu::Function(module, theirFunction), buffers, params)
    {
      ASSERT(params.nrSamplesPerChannel % 16 == 0);
      setArg(0, buffers.output);
      setArg(1, buffers.input);

      setEnqueueWorkSizes( gpu::Grid(256, (params.nrTABs + 15) / 16, params.nrSamplesPerChannel / 16),
                           gpu::Block(256, 1, 1) );

      nrOperations = 0;
      nrBytesRead = nrBytesWritten =
        (size_t) params.nrTABs * NR_POLARIZATIONS * params.nrChannelsPerSubband * 
        params.nrSamplesPerChannel * sizeof(std::complex<float>);
    }

    //--------  Template specializations for KernelFactory  --------//

    template<> size_t 
    KernelFactory<BeamFormerTransposeKernel>::bufferSize(BufferType bufferType) const
    {
      switch (bufferType) {
      case BeamFormerTransposeKernel::INPUT_DATA: 
      case BeamFormerTransposeKernel::OUTPUT_DATA:
        return
          (size_t) itsParameters.nrChannelsPerSubband * itsParameters.nrSamplesPerChannel * 
            NR_POLARIZATIONS * itsParameters.nrTABs * sizeof(std::complex<float>);
      default:
        THROW(GPUProcException, "Invalid bufferType (" << bufferType << ")");
      }
    }

    template<> CompileDefinitions
    KernelFactory<BeamFormerTransposeKernel>::compileDefinitions() const
    {
      CompileDefinitions defs =
        KernelFactoryBase::compileDefinitions(itsParameters);
      defs["NR_TABS"] =
        lexical_cast<string>(itsParameters.nrTABs);

      return defs;
    }
  }
}

