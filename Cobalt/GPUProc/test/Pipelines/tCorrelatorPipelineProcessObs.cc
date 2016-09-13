//# tCorrelatorPipelineProcessObs.cc: test processObservation() using corr.
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
//# $Id: tCorrelatorPipelineProcessObs.cc 26555 2013-09-15 08:42:46Z mol $

#include <lofar_config.h>

#include <vector>

#ifdef HAVE_MPI
#  include <mpi.h>
#endif

#include <Common/LofarLogger.h>
#include <CoInterface/Parset.h>
#include <InputProc/OMPThread.h>
#include <GPUProc/Pipelines/CorrelatorPipeline.h>
#include <GPUProc/Station/StationInput.h>

using namespace std;
using namespace LOFAR;
using namespace LOFAR::Cobalt;

int main(int argc, char *argv[]) {
  INIT_LOGGER("tCorrelatorPipelineProcessObs");

  omp_set_nested(true);
  OMPThread::init();

  if (setenv("DISPLAY", ":0", 1) < 0)
  {
    perror("error setting DISPLAY");
    exit(1);
  }

  int rank = 0;
  int nrHosts = 1;
#ifdef HAVE_MPI
  // Initialize and query MPI
  int mpi_thread_support;
  if (MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_thread_support) != MPI_SUCCESS) {
    cerr << "MPI_Init failed" << endl;
    exit(1);
  }

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nrHosts);
#endif

  try {
    gpu::Platform pf;
    cout << "Detected " << pf.size() << " CUDA devices" << endl;
  } catch (gpu::CUDAException& e) {
    cerr << e.what() << endl;
    return 3;
  }

  gpu::Device device(0);
  vector<gpu::Device> devices(1, device);
  gpu::Context ctx(device);

  // Note: we just need to test the Pipeline part (processObservation()),
  // not the SubbandProc and other logic below it, as that is already covered
  // in other tests. (This is not a feature or integration test.)
  // Testing Pipeline in isolation is next to impossible, but fake it
  // with a parset where the number of input data blocks is 0. Then we don't
  // have to start input procs etc. The disadvantage is that some code of
  // processObservation() remains untested...
  Parset ps("tCorrelatorPipelineProcessObs.parset");

#ifndef HAVE_MPI
  DirectInput::instance(&ps);
#endif

  // "distribute" subbands over 1 node
  vector<size_t> subbands;
  for (size_t sb = 0; sb < ps.nrSubbands(); sb++)
  {
    subbands.push_back(sb);
  }

  // no data, so no need to run a sender:
  // receiver(s) from processObservation() will fwd a end of data NULL pool item immediately.
  // idem for storage proc: we'll get a failed to connect to storage log msg, but don't care.
  CorrelatorPipeline(ps, subbands).processObservation(CORRELATED_DATA);

#ifdef HAVE_MPI
  MPI_Finalize();
#endif

  return 0;
}

