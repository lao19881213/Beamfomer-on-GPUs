//# PerformanceCounter.cc
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
//# $Id: PerformanceCounter.cc 24984 2013-05-21 16:18:43Z amesfoort $

#include <lofar_config.h>

#include "PerformanceCounter.h"

#include <iostream>
#include <iomanip>
#include <sstream>

#include <Common/LofarLogger.h>
#include <Common/PrettyUnits.h>
#include <GPUProc/OpenMP_Lock.h>

using namespace std;

namespace LOFAR
{
  namespace Cobalt
  {
    PerformanceCounter::PerformanceCounter(const std::string &name, bool profiling, bool logAtDestruction)
      :
      name(name),
      profiling(profiling),
      logAtDestruction(logAtDestruction),
      nrActiveEvents(0)
    {
    }


    PerformanceCounter::~PerformanceCounter()
    {
      waitForAllOperations();

      if (logAtDestruction) {
        LOG_INFO_STR(total.log(name));
      }
    }


    void PerformanceCounter::waitForAllOperations()
    {
      ScopedLock sl(mutex);

      while (nrActiveEvents > 0)
        activeEventsLowered.wait(mutex);
    }


    struct PerformanceCounter::figures PerformanceCounter::getTotal()
    {
      ScopedLock sl(mutex);

      return total;
    }


    std::string PerformanceCounter::figures::log(const std::string &name) const
    {
      std::stringstream str;

      // Mimic output of NSTimer::print (in LCS/Common/Timer.cc)
      str << left << setw(25) << name << ": " << right
          << "avg = " << PrettyTime(avrRuntime()) << ", "
          << "total = " << PrettyTime(runtime) << ", "
          << "count = " << setw(9) << nrEvents << ", "

          << setprecision(3)
          << "GFLOP/s = " << FLOPs() / 1e9 << ", "
          << "read = " << readSpeed() / 1e9 << " GB/s, "
          << "written = " << writeSpeed() / 1e9 << " GB/s, "
          << "total I/O = " << (readSpeed() + writeSpeed()) / 1e9 << " GB/s";

      return str.str();
    }


    void PerformanceCounter::eventCompleteCallBack(cl_event ev, cl_int /*status*/, void *userdata)
    {
      struct callBackArgs *args = static_cast<struct callBackArgs *>(userdata);

      try {
        // extract performance information
        cl::Event event(ev);

        size_t queued = event.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
        size_t submitted = event.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>();
        size_t start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        size_t stop = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        double seconds = (stop - start) / 1e9;

        // sanity checks
        ASSERT(seconds >= 0);
        ASSERTSTR(seconds < 15, "Kernel took " << seconds << " seconds to execute: thread " << omp_get_thread_num() << ": " << queued << ' ' << submitted - queued << ' ' << start - queued << ' ' << stop - queued);

        args->figures.runtime = seconds;

        // add figures to total
        {
          ScopedLock sl(args->this_->mutex);
          args->this_->total += args->figures;
        }

        // cl::~Event() decreases ref count
      } catch (cl::Error &error) {
        // ignore errors in callBack function (OpenCL library not exception safe)
      }

      // we're done -- release event and possibly signal destructor
      {
        ScopedLock sl(args->this_->mutex);
        args->this_->nrActiveEvents--;
        args->this_->activeEventsLowered.signal();
      }

      delete args;
    }


    void PerformanceCounter::doOperation(cl::Event &event, size_t nrOperations, size_t nrBytesRead, size_t nrBytesWritten)
    {
      if (!profiling)
        return;

      // reference count between C and C++ conversions is serously broken in C++ wrapper
      cl_event ev = event();
      cl_int error = clRetainEvent(ev);

      if (error != CL_SUCCESS)
        throw cl::Error(error, "clRetainEvent");

      // obtain run time information
      struct callBackArgs *args = new callBackArgs;
      args->this_ = this;
      args->figures.nrOperations = nrOperations;
      args->figures.nrBytesRead = nrBytesRead;
      args->figures.nrBytesWritten = nrBytesWritten;
      args->figures.runtime = 0.0;
      args->figures.nrEvents = 1;

      {
        // allocate event as active
        ScopedLock sl(mutex);
        nrActiveEvents++;
      }

      event.setCallback(CL_COMPLETE, &PerformanceCounter::eventCompleteCallBack, args);
    }

  }
}

