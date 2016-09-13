//# Kernel.cc
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
//# $Id: Kernel.cc 27477 2013-11-21 13:08:20Z loose $

#include <lofar_config.h>

#include <ostream>
#include <sstream>
#include <boost/format.hpp>
#include <cuda_runtime.h>

#include <GPUProc/global_defines.h>
#include <GPUProc/Kernels/Kernel.h>
#include <GPUProc/PerformanceCounter.h>
#include <CoInterface/Parset.h>
#include <CoInterface/BlockID.h>
#include <Common/LofarLogger.h>

using namespace std;

namespace LOFAR
{
  namespace Cobalt
  {
    Kernel::Parameters::Parameters(const Parset& ps) :
      nrStations(ps.nrStations()),
      nrChannelsPerSubband(ps.nrChannelsPerSubband()),
      nrSamplesPerChannel(ps.nrSamplesPerChannel()),
      nrSamplesPerSubband(ps.nrSamplesPerSubband()),
      nrPolarizations(NR_POLARIZATIONS),
      dumpBuffers(false)
    {
    }

    Kernel::~Kernel()
    {
    }

    Kernel::Kernel(const gpu::Stream& stream, 
                   const gpu::Function& function,
                   const Buffers &buffers,
                   const Parameters &params)
      : 
      gpu::Function(function),
      maxThreadsPerBlock(
        stream.getContext().getDevice().getMaxThreadsPerBlock()),
      itsStream(stream),
      itsBuffers(buffers),
      itsParameters(params)
    {
      LOG_INFO_STR(
        "Function " << function.name() << ":" << 
        "\n  max. threads per block: " << 
        function.getAttribute(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK) <<
        "\n  nr. of registers used : " <<
        function.getAttribute(CU_FUNC_ATTRIBUTE_NUM_REGS));
    }

    void Kernel::setEnqueueWorkSizes(gpu::Grid globalWorkSize, 
                                     gpu::Block localWorkSize)
    {
      gpu::Grid grid;
      ostringstream errMsgs;

      // Enforce by the hardware supported work sizes to see errors clearly and
      // early.

      gpu::Block maxLocalWorkSize = 
        itsStream.getContext().getDevice().getMaxBlockDims();
      if (localWorkSize.x > maxLocalWorkSize.x ||
          localWorkSize.y > maxLocalWorkSize.y ||
          localWorkSize.z > maxLocalWorkSize.z)
        errMsgs << "  - localWorkSize must be at most " << maxLocalWorkSize 
                << endl;

      if (localWorkSize.x * localWorkSize.y * localWorkSize.z > 
          maxThreadsPerBlock)
        errMsgs << "  - localWorkSize total must be at most " 
                << maxThreadsPerBlock << " threads/block" << endl;

      // globalWorkSize may (in theory) be all zero (no work). Reject such
      // localWorkSize.
      if (localWorkSize.x == 0 || 
          localWorkSize.y == 0 ||
          localWorkSize.z == 0) {
        errMsgs << "  - localWorkSize components must be non-zero" << endl;
      } else {
        // TODO: to globalWorkSize in terms of localWorkSize (CUDA)
        // ('gridWorkSize').
        if (globalWorkSize.x % localWorkSize.x != 0 ||
            globalWorkSize.y % localWorkSize.y != 0 ||
            globalWorkSize.z % localWorkSize.z != 0)
          errMsgs << "  - globalWorkSize must divide localWorkSize" << endl;
        grid = gpu::Grid(globalWorkSize.x / localWorkSize.x,
                         globalWorkSize.y / localWorkSize.y,
                         globalWorkSize.z / localWorkSize.z);

        gpu::Grid maxGridWorkSize =
          itsStream.getContext().getDevice().getMaxGridDims();
        if (grid.x > maxGridWorkSize.x ||
            grid.y > maxGridWorkSize.y ||
            grid.z > maxGridWorkSize.z)
          errMsgs << "  - globalWorkSize / localWorkSize must be at most "
                  << maxGridWorkSize << endl;
      }

      string errStr(errMsgs.str());
      if (!errStr.empty())
        THROW(gpu::GPUException,
              "setEnqueueWorkSizes(): unsupported globalWorkSize " <<
              globalWorkSize << " and/or localWorkSize " << localWorkSize <<
              " selected:" << endl << errStr);

      LOG_DEBUG_STR("CUDA Grid size: " << grid);
      LOG_DEBUG_STR("CUDA Block size: " << localWorkSize);

      itsGridDims = grid;
      itsBlockDims = localWorkSize;
    }

    void Kernel::enqueue(const BlockID &blockId) const
    {
      itsStream.launchKernel(*this, itsGridDims, itsBlockDims);

      if (itsParameters.dumpBuffers && blockId.block >= 0) {
        itsStream.synchronize();
        dumpBuffers(blockId);
      }
    }

    void Kernel::enqueue(const BlockID &blockId, 
                         PerformanceCounter &counter) const
    {
      itsStream.recordEvent(counter.start);
      enqueue(blockId);
      itsStream.recordEvent(counter.stop);
    }

    void Kernel::dumpBuffers(const BlockID &blockId) const
    {
      dumpBuffer(itsBuffers.output,
                 str(boost::format(itsParameters.dumpFilePattern) %
                     blockId.globalSubbandIdx %
                     blockId.block));
    }

  }
}

