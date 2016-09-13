//# UHEP_TriggerKernel.cc
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
//# $Id: UHEP_TriggerKernel.cc 27077 2013-10-24 01:00:58Z amesfoort $

#include <lofar_config.h>

#include "UHEP_TriggerKernel.h"

#include <GPUProc/global_defines.h>

namespace LOFAR
{
  namespace Cobalt
  {
    UHEP_TriggerKernel::UHEP_TriggerKernel(const Parset &ps, gpu::Module &program, gpu::DeviceMemory &devTriggerInfo, gpu::DeviceMemory &devInvFIRfilteredData)
      :
      Kernel(ps, program, "trigger")
    {
      setArg(0, devTriggerInfo);
      setArg(1, devInvFIRfilteredData);

      setEnqueueWorkSizes( gpu::Grid(16, 16, ps.nrTABs(0)),
                           gpu::Block(16, 16, 1) );

      nrOperations = (size_t) ps.nrTABs(0) * ps.nrSamplesPerChannel() * 1024 * (3 /* power */ + 2 /* window */ + 1 /* max */ + 7 /* mean/variance */);
      nrBytesRead = (size_t) ps.nrTABs(0) * NR_POLARIZATIONS * ps.nrSamplesPerChannel() * 1024 * sizeof(float);
      nrBytesWritten = (size_t) ps.nrTABs(0) * sizeof(TriggerInfo);
    }

  }
}

