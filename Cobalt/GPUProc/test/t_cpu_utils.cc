//# t_cpu_utils.cc: test cpu utilities
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
//# $Id: t_cpu_utils.cc 25199 2013-06-05 23:46:56Z amesfoort $

#include <lofar_config.h>

#include <Common/LofarLogger.h>
#include <CoInterface/Parset.h>
#include <GPUProc/cpu_utils.h>

#include <mpi.h>
#include <iostream>
#include <string>
#include <sched.h>

using namespace std;
using namespace LOFAR::Cobalt;

int main(int argc, char **argv)
{
  INIT_LOGGER("t_cpu_utils.cc");

  Parset ps("t_cpu_utils.in_parset");
  
    // Initialise and query MPI
  int provided_mpi_thread_support;
  if (MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided_mpi_thread_support) != MPI_SUCCESS) {
    cerr << "MPI_Init_thread failed" << endl;
    exit(1);
  }

  // Get the name of the processor
  // skip test if we are not on cobalt!!!
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  string name(processor_name);
  
  if (!(name.find("cbm") == 0))
  {
    cout << "Test is not running on cobalt hardware and therefore skipped" << endl;
    MPI_Finalize();
    return 0;
  }
    


  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  //exercise the set processorAffinity functionality
  int cpuId = ps.settings.nodes[rank].cpu;
  setProcessorAffinity(cpuId);
  unsigned numCPU = sysconf( _SC_NPROCESSORS_ONLN );


  // Validate the correct setting of the affinity
  // Get the cpu of the current thread
  cpu_set_t mask;  
  sched_getaffinity(0, sizeof(cpu_set_t), &mask);

  // Test if affinity is set correctly!
  // The cpu with rank 0 should be all even cpus the rank should have odd cpu's
  for (unsigned idx_cpu = rank; idx_cpu < numCPU; idx_cpu += 2) 
  {
    if (1 != CPU_ISSET(idx_cpu, &mask))
    {
      LOG_FATAL_STR("Found a cpu that is NOT set while is should be set!");
      LOG_FATAL_STR(rank);
      exit(1);
    }
    if (0 != CPU_ISSET(idx_cpu + 1, &mask))
    {
      LOG_FATAL_STR("Found a cpu that is set while is should be NOT set!");
      LOG_FATAL_STR(rank);
      exit(1);
    }
  }

  MPI_Finalize();
      
  return 0;
}

