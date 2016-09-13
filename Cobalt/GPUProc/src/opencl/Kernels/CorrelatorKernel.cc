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
//# $Id: CorrelatorKernel.cc 25358 2013-06-18 09:05:40Z loose $

#include <lofar_config.h>

#include "CorrelatorKernel.h"

#include <vector>
#include <algorithm>

#include <Common/lofar_complex.h>
#include <Common/LofarLogger.h>
#include <CoInterface/Align.h>

#include <GPUProc/global_defines.h>

namespace LOFAR
{
  namespace Cobalt
  {

    CorrelatorKernel::CorrelatorKernel(const Parset &ps,
                                       cl::CommandQueue &queue,
                                       cl::Program &program,
                                       cl::Buffer &devVisibilities,
                                       cl::Buffer &devCorrectedData)
      :
# if defined USE_4X4
      Kernel(ps, program, "correlate_4x4")
# elif defined USE_3X3
      Kernel(ps, program, "correlate_3x3")
# elif defined USE_2X2
      Kernel(ps, program, "correlate_2x2")
# else
      Kernel(ps, program, "correlate")
# endif
    {
      setArg(0, devVisibilities);
      setArg(1, devCorrectedData);

      size_t maxNrThreads, preferredMultiple;
      getWorkGroupInfo(queue.getInfo<CL_QUEUE_DEVICE>(), CL_KERNEL_WORK_GROUP_SIZE, &maxNrThreads);

      std::vector<cl_context_properties> properties;
      queue.getInfo<CL_QUEUE_CONTEXT>().getInfo(CL_CONTEXT_PROPERTIES, &properties);

      if (cl::Platform((cl_platform_id) properties[1]).getInfo<CL_PLATFORM_NAME>() == "AMD Accelerated Parallel Processing")
        preferredMultiple = 256;
      else
        getWorkGroupInfo(queue.getInfo<CL_QUEUE_DEVICE>(), CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, &preferredMultiple);

# if defined USE_4X4
      unsigned quartStations = (ps.nrStations() + 2) / 4;
      unsigned nrBlocks = quartStations * (quartStations + 1) / 2;
# elif defined USE_3X3
      unsigned thirdStations = (ps.nrStations() + 2) / 3;
      unsigned nrBlocks = thirdStations * (thirdStations + 1) / 2;
# elif defined USE_2X2
      unsigned halfStations = (ps.nrStations() + 1) / 2;
      unsigned nrBlocks = halfStations * (halfStations + 1) / 2;
# else
      unsigned nrBlocks = ps.nrBaselines();
# endif
      unsigned nrPasses = (nrBlocks + maxNrThreads - 1) / maxNrThreads;
      unsigned nrThreads = (nrBlocks + nrPasses - 1) / nrPasses;
      nrThreads = (nrThreads + preferredMultiple - 1) / preferredMultiple * preferredMultiple;
      //LOG_DEBUG_STR("nrBlocks = " << nrBlocks << ", nrPasses = " << nrPasses << ", preferredMultiple = " << preferredMultiple << ", nrThreads = " << nrThreads);

      unsigned nrUsableChannels = std::max(ps.nrChannelsPerSubband() - 1, 1U);
      globalWorkSize = cl::NDRange(nrPasses * nrThreads, nrUsableChannels);
      localWorkSize = cl::NDRange(nrThreads, 1);

      nrOperations = (size_t) nrUsableChannels * ps.nrBaselines() * ps.nrSamplesPerChannel() * 32;
      nrBytesRead = (size_t) nrPasses * ps.nrStations() * nrUsableChannels * ps.nrSamplesPerChannel() * NR_POLARIZATIONS * sizeof(std::complex<float>);
      nrBytesWritten = (size_t) ps.nrBaselines() * nrUsableChannels * NR_POLARIZATIONS * NR_POLARIZATIONS * sizeof(std::complex<float>);
    }

    size_t CorrelatorKernel::bufferSize(const Parset& ps, BufferType bufferType)
    {
      switch (bufferType) {
      case INPUT_DATA:
        return
          ps.nrSamplesPerSubband() * ps.nrStations() * 
          NR_POLARIZATIONS * sizeof(std::complex<float>);
      case OUTPUT_DATA:
        return 
          ps.nrBaselines() * ps.nrChannelsPerSubband() * 
          NR_POLARIZATIONS * NR_POLARIZATIONS * sizeof(std::complex<float>);
      default: 
        THROW(GPUProcException, "Invalid bufferType (" << bufferType << ")");
      }
    }

#if defined USE_NEW_CORRELATOR

//     CorrelatorKernel::CorrelatorKernel(const Parset &ps, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &devVisibilities, cl::Buffer &devCorrectedData)
//       :
// # if defined USE_2X2
//       Kernel(ps, program, "correlate")
// # else
// #  error not implemented
// # endif
//     {
//       setArg(0, devVisibilities);
//       setArg(1, devCorrectedData);

//       unsigned nrRectanglesPerSide = (ps.nrStations() - 1) / (2 * 16);
//       unsigned nrRectangles = nrRectanglesPerSide * (nrRectanglesPerSide + 1) / 2;
//       //LOG_DEBUG_STR("nrRectangles = " << nrRectangles);

//       unsigned nrBlocksPerSide = (ps.nrStations() + 2 * 16 - 1) / (2 * 16);
//       unsigned nrBlocks = nrBlocksPerSide * (nrBlocksPerSide + 1) / 2;
//       //LOG_DEBUG_STR("nrBlocks = " << nrBlocks);

//       unsigned nrUsableChannels = std::max(ps.nrChannelsPerSubband() - 1, 1U);
//       globalWorkSize = cl::NDRange(16 * 16, nrBlocks, nrUsableChannels);
//       localWorkSize = cl::NDRange(16 * 16, 1, 1);

//       // FIXME
//       //nrOperations   = (size_t) (32 * 32) * nrRectangles * nrUsableChannels * ps.nrSamplesPerChannel() * 32;
//       nrOperations = (size_t) ps.nrBaselines() * ps.nrSamplesPerSubband() * 32;
//       nrBytesRead = (size_t) (32 + 32) * nrRectangles * nrUsableChannels * ps.nrSamplesPerChannel() * NR_POLARIZATIONS * sizeof(std::complex<float>);
//       nrBytesWritten = (size_t) (32 * 32) * nrRectangles * nrUsableChannels * NR_POLARIZATIONS * NR_POLARIZATIONS * sizeof(std::complex<float>);
//     }

    CorrelateRectangleKernel::CorrelateRectangleKernel(const Parset &ps, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &devVisibilities, cl::Buffer &devCorrectedData)
      :
# if defined USE_2X2
      Kernel(ps, program, "correlateRectangleKernel")
# else
#  error not implemented
# endif
    {
      setArg(0, devVisibilities);
      setArg(1, devCorrectedData);

      unsigned nrRectanglesPerSide = (ps.nrStations() - 1) / (2 * 16);
      unsigned nrRectangles = nrRectanglesPerSide * (nrRectanglesPerSide + 1) / 2;
      LOG_DEBUG_STR("nrRectangles = " << nrRectangles);

      unsigned nrUsableChannels = std::max(ps.nrChannelsPerSubband() - 1, 1U);
      globalWorkSize = cl::NDRange(16 * 16, nrRectangles, nrUsableChannels);
      localWorkSize = cl::NDRange(16 * 16, 1, 1);

      nrOperations = (size_t) (32 * 32) * nrRectangles * nrUsableChannels * ps.nrSamplesPerChannel() * 32;
      nrBytesRead = (size_t) (32 + 32) * nrRectangles * nrUsableChannels * ps.nrSamplesPerChannel() * NR_POLARIZATIONS * sizeof(std::complex<float>);
      nrBytesWritten = (size_t) (32 * 32) * nrRectangles * nrUsableChannels * NR_POLARIZATIONS * NR_POLARIZATIONS * sizeof(std::complex<float>);
    }


    size_t CorrelatorRectangleKernel::bufferSize(const Parset& ps, BufferType bufferType)
    {
      switch (bufferType) {
      case INPUT_DATA:
        return
          ps.nrSamplesPerSubband() * ps.nrStations() * 
          NR_POLARIZATIONS * sizeof(std::complex<float>);
      case OUTPUT_DATA:
        return 
          ps.nrBaselines() * ps.nrChannelsPerSubband() * 
          NR_POLARIZATIONS * NR_POLARIZATIONS * sizeof(std::complex<float>);
      default: 
        THROW(GPUProcException, "Invalid bufferType (" << bufferType << ")");
      }
    }


    CorrelateTriangleKernel::CorrelateTriangleKernel(const Parset &ps, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &devVisibilities, cl::Buffer &devCorrectedData)
      :
# if defined USE_2X2
      Kernel(ps, program, "correlateTriangleKernel")
# else
#  error not implemented
# endif
    {
      setArg(0, devVisibilities);
      setArg(1, devCorrectedData);

      unsigned nrTriangles = (ps.nrStations() + 2 * 16 - 1) / (2 * 16);
      unsigned nrMiniBlocksPerSide = 16;
      unsigned nrMiniBlocks = nrMiniBlocksPerSide * (nrMiniBlocksPerSide + 1) / 2;
      size_t preferredMultiple;
      getWorkGroupInfo(queue.getInfo<CL_QUEUE_DEVICE>(), CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, &preferredMultiple);
      unsigned nrThreads = align(nrMiniBlocks, preferredMultiple);

      LOG_DEBUG_STR("nrTriangles = " << nrTriangles << ", nrMiniBlocks = " << nrMiniBlocks << ", nrThreads = " << nrThreads);

      unsigned nrUsableChannels = std::max(ps.nrChannelsPerSubband() - 1, 1U);
      globalWorkSize = cl::NDRange(nrThreads, nrTriangles, nrUsableChannels);
      localWorkSize = cl::NDRange(nrThreads, 1, 1);

      nrOperations = (size_t) (32 * 32 / 2) * nrTriangles * nrUsableChannels * ps.nrSamplesPerChannel() * 32;
      nrBytesRead = (size_t) 32 * nrTriangles * nrUsableChannels * ps.nrSamplesPerChannel() * NR_POLARIZATIONS * sizeof(std::complex<float>);
      nrBytesWritten = (size_t) (32 * 32 / 2) * nrTriangles * nrUsableChannels * NR_POLARIZATIONS * NR_POLARIZATIONS * sizeof(std::complex<float>);
    }

    size_t CorrelatorTriangleKernel::bufferSize(const Parset& ps, BufferType bufferType)
    {
      switch (bufferType) {
      case INPUT_DATA:
        return
          ps.nrSamplesPerSubband() * ps.nrStations() * 
          NR_POLARIZATIONS * sizeof(std::complex<float>);
      case OUTPUT_DATA:
        return 
          ps.nrBaselines() * ps.nrChannelsPerSubband() * 
          NR_POLARIZATIONS * NR_POLARIZATIONS * sizeof(std::complex<float>);
      default: 
        THROW(GPUProcException, "Invalid bufferType (" << bufferType << ")");
      }
    }

#endif

  }
}

