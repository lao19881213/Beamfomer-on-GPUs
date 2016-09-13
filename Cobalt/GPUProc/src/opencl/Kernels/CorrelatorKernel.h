//# CorrelatorKernel.h
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
//# $Id: CorrelatorKernel.h 25358 2013-06-18 09:05:40Z loose $

#ifndef LOFAR_GPUPROC_OPENCL_CORRELATOR_KERNEL_H
#define LOFAR_GPUPROC_OPENCL_CORRELATOR_KERNEL_H

#include <CoInterface/Parset.h>

#include "Kernel.h"
#include <GPUProc/global_defines.h>
#include <GPUProc/gpu_incl.h>

namespace LOFAR
{
  namespace Cobalt
  {

    class CorrelatorKernel : public Kernel
    {
    public:
      CorrelatorKernel(const Parset &ps, 
                       cl::CommandQueue &queue,
                       cl::Program &program,
                       cl::Buffer &devVisibilities,
                       cl::Buffer &devCorrectedData);

      enum BufferType
      {
        INPUT_DATA,
        OUTPUT_DATA
      };

      // Return required buffer size for \a bufferType
      static size_t bufferSize(const Parset& ps, BufferType bufferType);

    };

#if defined USE_NEW_CORRELATOR

    class CorrelateRectangleKernel : public Kernel
    {
    public:
      CorrelateRectangleKernel(const Parset &ps, 
                               cl::CommandQueue &queue,
                               cl::Program &program,
                               cl::Buffer &devVisibilities,
                               cl::Buffer &devCorrectedData);

      enum BufferType
      {
        INPUT_DATA,
        OUTPUT_DATA
      };

      // Return required buffer size for \a bufferType
      static size_t bufferSize(const Parset& ps, BufferType bufferType);

    };

    class CorrelateTriangleKernel : public Kernel
    {
    public:
      CorrelateTriangleKernel(const Parset &ps,
                              cl::CommandQueue &queue,
                              cl::Program &program,
                              cl::Buffer &devVisibilities,
                              cl::Buffer &devCorrectedData);

      enum BufferType
      {
        INPUT_DATA,
        OUTPUT_DATA
      };

      // Return required buffer size for \a bufferType
      static size_t bufferSize(const Parset& ps, BufferType bufferType);

    };

#endif

  }
}

#endif

