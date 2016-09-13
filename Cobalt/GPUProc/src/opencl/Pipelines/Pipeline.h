//# Pipeline.h
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
//# $Id: Pipeline.h 25727 2013-07-22 12:56:14Z mol $

#ifndef LOFAR_GPUPROC_OPENCL_PIPELINE_H
#define LOFAR_GPUPROC_OPENCL_PIPELINE_H

#include <string>
#include <vector>

#include <Common/LofarTypes.h>
#include <Common/Thread/Queue.h>
#include <CoInterface/Parset.h>
#include <CoInterface/SmartPtr.h>

#include <GPUProc/global_defines.h>
#include <GPUProc/OpenMP_Lock.h>
#include <GPUProc/gpu_incl.h>
#include <GPUProc/SubbandProcs/SubbandProc.h>
#include <GPUProc/PerformanceCounter.h>

namespace LOFAR
{
  namespace Cobalt
  {
    class Pipeline
    {
    public:
      Pipeline(const Parset &ps);

      cl::Program             createProgram(const char *sources);

      const Parset            &ps;
      cl::Context context;
      std::vector<cl::Device> devices;

#if defined USE_B7015
      OMP_Lock hostToDeviceLock[4], deviceToHostLock[4];
#endif

    protected:
      // combines all functionality needed for getting the total from a set of counters
      struct Performance {
        std::map<std::string, PerformanceCounter::figures> total_counters;
        std::map<std::string, SmartPtr<NSTimer> > total_timers;
        // lock on the shared data
        Mutex totalsMutex;
        // add the counter in this queue
        void addQueue(SubbandProc &queue);
        // Print a logline with results
        void log(size_t nrSubbandProcs);
      } performance;
    };
  }
}

#endif

