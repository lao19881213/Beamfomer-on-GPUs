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
//# $Id: PerformanceCounter.h 24849 2013-05-08 14:51:06Z amesfoort $

#ifndef LOFAR_GPUPROC_OPENCL_PERFORMANCECOUNTER_H
#define LOFAR_GPUPROC_OPENCL_PERFORMANCECOUNTER_H

#include <string>

#include <Common/Thread/Mutex.h>
#include <Common/Thread/Condition.h>

#include "gpu_incl.h"

namespace LOFAR
{
  namespace Cobalt
  {
    class PerformanceCounter
    {
    public:
      // name of counter, for logging purposes
      const std::string name;

      // whether we collect profiling information in the first place
      const bool profiling;

      // Initialise the counter, giving it a name.
      //
      // If profiling == false, no actual performance statistics are
      // gathered.
      PerformanceCounter(const std::string &name, bool profiling, bool logAtDestruction = false);
      ~PerformanceCounter();

      // register an operation covered by `event'. runtime will be determined by OpenCL, the
      // rest of the figures have to be provided.
      void doOperation(cl::Event &, size_t nrOperations, size_t nrBytesRead, size_t nrBytesWritten);

      // performance figures
      struct figures {
        size_t nrOperations;
        size_t nrBytesRead;
        size_t nrBytesWritten;
        double runtime;

        size_t nrEvents;

        figures() : nrOperations(0), nrBytesRead(0), nrBytesWritten(0), runtime(0.0), nrEvents(0)
        {
        }

        struct figures &operator+=(const struct figures &other)
        {
          nrOperations += other.nrOperations;
          nrBytesRead += other.nrBytesRead;
          nrBytesWritten += other.nrBytesWritten;
          runtime += other.runtime;
          nrEvents += other.nrEvents;

          return *this;
        }

        double avrRuntime() const
        {
          return runtime / nrEvents;
        }
        double FLOPs() const
        {
          return nrOperations / runtime;
        }
        double readSpeed() const
        {
          return nrBytesRead / runtime;
        }
        double writeSpeed() const
        {
          return nrBytesWritten / runtime;
        }

        std::string log(const std::string &name = "timer") const;
      };

      // Return once all scheduled operations have completed
      void waitForAllOperations();

      // Return current running total figures
      struct figures getTotal();

      // Log the total figures
      void logTotal();

    private:
      // whether to log the performance when ~PerformanceCounter is
      // called
      const bool logAtDestruction;

      // performance totals
      struct figures total;

      // number of events that still have a callback waiting
      size_t nrActiveEvents;
      Condition activeEventsLowered;

      // lock for total and nrActiveEvents
      Mutex mutex;

      // call-back to get runtime information
      struct callBackArgs {
        PerformanceCounter *this_;
        struct figures figures;
      };

      static void eventCompleteCallBack(cl_event, cl_int /*status*/, void *userdata);
    };
  }
}

#endif

