//# SubbandProc.h
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
//# $Id: WorkQueue.h 25727 2013-07-22 12:56:14Z mol $

#ifndef LOFAR_GPUPROC_OPENCL_WORKQUEUE_H
#define LOFAR_GPUPROC_OPENCL_WORKQUEUE_H

#include <string>
#include <map>

#include <Common/Timer.h>
#include <CoInterface/Parset.h>
#include <CoInterface/SmartPtr.h>
#include <GPUProc/PerformanceCounter.h>
#include <GPUProc/gpu_incl.h>

namespace LOFAR
{
  namespace Cobalt
  {
    class SubbandProc
    {
    public:
      SubbandProc(cl::Context &context, cl::Device &device, unsigned gpuNumber, const Parset &ps);

      const unsigned gpu;
      cl::Device &device;
      cl::CommandQueue queue;

      std::map<std::string, SmartPtr<PerformanceCounter> > counters;
      std::map<std::string, SmartPtr<NSTimer> > timers;

    protected:
      const Parset &ps;

      void addCounter(const std::string &name);
      void addTimer(const std::string &name);
    };
  }
}

#endif

