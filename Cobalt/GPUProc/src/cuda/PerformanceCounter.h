//# PerformanceCounter.h
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
//# $Id: PerformanceCounter.h 27000 2013-10-17 09:11:13Z loose $

#ifndef LOFAR_GPUPROC_CUDA_PERFORMANCECOUNTER_H
#define LOFAR_GPUPROC_CUDA_PERFORMANCECOUNTER_H


#include <GPUProc/gpu_wrapper.h>
#include <GPUProc/RunningStatistics.h>

namespace LOFAR
{
  namespace Cobalt
  {
    class PerformanceCounter
    {
    public:
      PerformanceCounter(const gpu::Context &context);

      void logTime();

      // Public event: it needs to be inserted into a stream.
      // @{
      gpu::Event start;
      gpu::Event stop;
      // @}

      RunningStatistics stats;
    private:
      
    };
  }
}

#endif

