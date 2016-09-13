//# TestUtil.h
//# Copyright (C) 2013  ASTRON (Netherlands Institute for Radio Astronomy)
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
//# $Id: TestUtil.h 25978 2013-08-08 03:26:50Z amesfoort $

#ifndef LOFAR_GPUPROC_TESTUTIL_H
#define LOFAR_GPUPROC_TESTUTIL_H

#include <GPUProc/gpu_wrapper.h>

namespace LOFAR {
  namespace Cobalt {

    // Allocate and initialise host memory
    template<typename T> gpu::HostMemory getInitializedArray(gpu::Context &ctx, unsigned size, const T &defaultValue)
    {
      gpu::HostMemory memory(ctx, size);
      T* createdArray = memory.get<T>();

      for (unsigned idx = 0; idx < size / sizeof(T); ++idx)
        createdArray[idx] = defaultValue;

      return memory;
    }

  }
}

#endif

