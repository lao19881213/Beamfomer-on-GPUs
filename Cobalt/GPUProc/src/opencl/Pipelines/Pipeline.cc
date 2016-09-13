//# Pipeline.cc
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
//# $Id: Pipeline.cc 25727 2013-07-22 12:56:14Z mol $

#include <lofar_config.h>

#include "Pipeline.h"

#include <Common/LofarLogger.h>
#include <Common/lofar_iomanip.h>

#include <GPUProc/opencl/gpu_utils.h>

namespace LOFAR
{
  namespace Cobalt
  {


    Pipeline::Pipeline(const Parset &ps)
      :
      ps(ps)
    {
      createContext(context, devices);
    }


    cl::Program Pipeline::createProgram(const char *sources)
    {
      return LOFAR::Cobalt::createProgram(ps, context, devices, sources);
    }


    void Pipeline::Performance::addQueue(SubbandProc &queue)
    {
      ScopedLock sl(totalsMutex);

      // add performance counters
      for (map<string, SmartPtr<PerformanceCounter> >::iterator i = queue.counters.begin(); i != queue.counters.end(); ++i) {

        const string &name = i->first;
        PerformanceCounter *counter = i->second.get();

        counter->waitForAllOperations();

        total_counters[name] += counter->getTotal();
      }

      // add timers
      for (map<string, SmartPtr<NSTimer> >::iterator i = queue.timers.begin(); i != queue.timers.end(); ++i) {

        const string &name = i->first;
        NSTimer *timer = i->second.get();

        if (!total_timers[name])
          total_timers[name] = new NSTimer(name, false, false);

        *total_timers[name] += *timer;
      }
    }
    
    void Pipeline::Performance::log(size_t nrSubbandProcs)
    {
      // Group figures based on their prefix before " - ", so "compute - FIR"
      // belongs to group "compute".
      map<string, PerformanceCounter::figures> counter_groups;

      for (map<string, PerformanceCounter::figures>::const_iterator i = total_counters.begin(); i != total_counters.end(); ++i) {
        size_t n = i->first.find(" - ");

        // discard counters without group
        if (n == string::npos)
          continue;

        // determine group name
        string group = i->first.substr(0, n);

        // add to group
        counter_groups[group] += i->second;
      }

      // Log all performance totals at DEBUG level
      for (map<string, PerformanceCounter::figures>::const_iterator i = total_counters.begin(); i != total_counters.end(); ++i) {
        LOG_DEBUG_STR(i->second.log(i->first));
      }

      for (map<string, SmartPtr<NSTimer> >::const_iterator i = total_timers.begin(); i != total_timers.end(); ++i) {
        LOG_DEBUG_STR(*(i->second));
      }

      // Log all group totals at INFO level
      for (map<string, PerformanceCounter::figures>::const_iterator i = counter_groups.begin(); i != counter_groups.end(); ++i) {
        LOG_INFO_STR(i->second.log(i->first));
      }

      // Log specific performance figures for regression tests at INFO level
      double wall_seconds = total_timers["CPU - total"]->getAverage();
      double gpu_seconds = counter_groups["compute"].runtime / nrGPUs;
      double spin_seconds = total_timers["GPU - wait"]->getAverage();
      double input_seconds = total_timers["CPU - read input"]->getElapsed() / nrSubbandProcs;
      double cpu_seconds = total_timers["CPU - process"]->getElapsed() / nrSubbandProcs;
      double postprocess_seconds = total_timers["CPU - postprocess"]->getElapsed() / nrSubbandProcs;

      LOG_INFO_STR("Wall seconds spent processing        : " << fixed << setw(8) << setprecision(3) << wall_seconds);
      LOG_INFO_STR("GPU  seconds spent computing, per GPU: " << fixed << setw(8) << setprecision(3) << gpu_seconds);
      LOG_INFO_STR("Spin seconds spent polling, per block: " << fixed << setw(8) << setprecision(3) << spin_seconds);
      LOG_INFO_STR("CPU  seconds spent on input,   per WQ: " << fixed << setw(8) << setprecision(3) << input_seconds);
      LOG_INFO_STR("CPU  seconds spent processing, per WQ: " << fixed << setw(8) << setprecision(3) << cpu_seconds);
      LOG_INFO_STR("CPU  seconds spent postprocessing, per WQ: " << fixed << setw(8) << setprecision(3) << postprocess_seconds);
    }

  }
}

