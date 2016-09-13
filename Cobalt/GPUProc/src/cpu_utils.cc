//# cpu_utils.cc
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
//# $Id$

#include <lofar_config.h>

#include <sched.h>
#include <fstream>
#include <boost/format.hpp>

#include <Common/SystemCallException.h>
#include <CoInterface/Parset.h>
#include <CoInterface/Exceptions.h>
#include <CoInterface/PrintVector.h>

namespace LOFAR
{
  namespace Cobalt
  {
    // Set the correct processer affinity
    void setProcessorAffinity(unsigned cpuId)
    {
      // Get the number of cores (32)
      unsigned numCores = sysconf(_SC_NPROCESSORS_ONLN);


      // Determine the cores local to the specified cpuId
      vector<unsigned> localCores;

      for (unsigned core = 0; core < numCores; ++core) {
        // The file below contains an integer indicating the physical CPU
        // hosting this core.
        std::ifstream fs(str(boost::format("/sys/devices/system/cpu/cpu%u/topology/physical_package_id") % core).c_str());

        unsigned physical_cpu;
        fs >> physical_cpu;

        if (!fs.good())
          continue;

        // Add this core to the mask if it matches the requested CPU
        if (physical_cpu == cpuId)
          localCores.push_back(core);
      }

      if (localCores.empty())
        THROW(GPUProcException, "Request to bind to non-existing CPU: " << cpuId);

      LOG_DEBUG_STR("Binding to CPU " << cpuId << ": cores " << localCores);

      // put localCores in a cpu_set
      cpu_set_t mask;  

      CPU_ZERO(&mask); 

      for (vector<unsigned>::const_iterator i = localCores.begin(); i != localCores.end(); ++i)
        CPU_SET(*i, &mask);

      // now assign the mask and set the affinity
      if (sched_setaffinity(0, sizeof(cpu_set_t), &mask) != 0)
        THROW_SYSCALL("sched_setaffinity");
    }
  }
}

