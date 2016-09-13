//# UHEP_TriggerKernel.h
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
//# $Id: UHEP_TriggerKernel.h 24885 2013-05-13 23:37:27Z amesfoort $

#ifndef LOFAR_GPUPROC_CUDA_UHEP_TRIGGER_KERNEL_H
#define LOFAR_GPUPROC_CUDA_UHEP_TRIGGER_KERNEL_H

#include <CoInterface/Parset.h>

#include "Kernel.h"
#include <GPUProc/gpu_wrapper.h>

namespace LOFAR
{
  namespace Cobalt
  {


    struct TriggerInfo {
      float mean, variance, bestValue;
      unsigned bestApproxIndex;
    };

    class UHEP_TriggerKernel : public Kernel
    {
    public:
      UHEP_TriggerKernel(const Parset &ps, gpu::Module &program,
                         gpu::DeviceMemory &devTriggerInfo, gpu::DeviceMemory &devInvFIRfilteredData);

    };

  }
}

#endif

