//# Filter_FFT_Kernel.h
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
//# $Id: Filter_FFT_Kernel.h 26635 2013-09-23 14:18:53Z loose $

#ifndef LOFAR_GPUPROC_CUDA_FILTER_FFT_KERNEL_H
#define LOFAR_GPUPROC_CUDA_FILTER_FFT_KERNEL_H

#include <CoInterface/Parset.h>

#include "FFT_Kernel.h"
#include <GPUProc/gpu_wrapper.h>

namespace LOFAR
{
  namespace Cobalt
  {
    class Filter_FFT_Kernel : public FFT_Kernel
    {
    public:
      Filter_FFT_Kernel(const Parset &ps, const gpu::Stream &stream,
                        gpu::DeviceMemory &devFilteredData);

    };

  }
}

#endif

