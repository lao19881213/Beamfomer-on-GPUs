//# IntToFloatKernel.cc
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
//# $Id: IntToFloatKernel.cc 27077 2013-10-24 01:00:58Z amesfoort $

#include <lofar_config.h>

#include "IntToFloatKernel.h"

#include <GPUProc/global_defines.h>
#include <GPUProc/gpu_utils.h>
#include <CoInterface/BlockID.h>
#include <Common/lofar_complex.h>

#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>

#include <fstream>

using boost::lexical_cast;
using boost::format;

namespace LOFAR
{
  namespace Cobalt
  {
    string IntToFloatKernel::theirSourceFile = "IntToFloat.cu";
    string IntToFloatKernel::theirFunction = "intToFloat";

    IntToFloatKernel::Parameters::Parameters(const Parset& ps) :
      Kernel::Parameters(ps),
      nrBitsPerSample(ps.settings.nrBitsPerSample),
      nrBytesPerComplexSample(ps.nrBytesPerComplexSample())
    {
      dumpBuffers = 
        ps.getBool("Cobalt.Kernels.IntToFloatKernel.dumpOutput", false);
      dumpFilePattern = 
        str(format("L%d_SB%%03d_BL%%03d_IntToFloatKernel.dat") % 
            ps.settings.observationID);
    }

    IntToFloatKernel::IntToFloatKernel(const gpu::Stream& stream,
                                       const gpu::Module& module,
                                       const Buffers& buffers,
                                       const Parameters& params) :
      Kernel(stream, gpu::Function(module, theirFunction), buffers, params)
    {
      setArg(0, buffers.output);
      setArg(1, buffers.input);

      unsigned maxNrThreads;
      maxNrThreads = getAttribute(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
      setEnqueueWorkSizes( gpu::Grid(maxNrThreads, params.nrStations),
                           gpu::Block(maxNrThreads, 1) );

      unsigned nrSamples = params.nrStations * params.nrChannelsPerSubband * NR_POLARIZATIONS;
      nrOperations = (size_t) nrSamples * 2;
      nrBytesRead = (size_t) nrSamples * 2 * params.nrBitsPerSample / 8;
      nrBytesWritten = (size_t) nrSamples * sizeof(std::complex<float>);
    }

    //--------  Template specializations for KernelFactory  --------//

    template<> size_t 
    KernelFactory<IntToFloatKernel>::bufferSize(BufferType bufferType) const
    {
      switch (bufferType) {
      case IntToFloatKernel::INPUT_DATA:
        return
          (size_t) itsParameters.nrStations * NR_POLARIZATIONS * 
            itsParameters.nrSamplesPerSubband * itsParameters.nrBytesPerComplexSample;
      case IntToFloatKernel::OUTPUT_DATA:
        return
          (size_t) itsParameters.nrStations * NR_POLARIZATIONS * 
            itsParameters.nrSamplesPerSubband * sizeof(std::complex<float>);
      default:
        THROW(GPUProcException, "Invalid bufferType (" << bufferType << ")");
      }
    }

    template<> CompileDefinitions
    KernelFactory<IntToFloatKernel>::compileDefinitions() const
    {
      CompileDefinitions defs =
        KernelFactoryBase::compileDefinitions(itsParameters);
      defs["NR_BITS_PER_SAMPLE"] =
        lexical_cast<string>(itsParameters.nrBitsPerSample);
      return defs;
    }

  }
}

