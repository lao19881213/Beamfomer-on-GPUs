//# tAH_TestStorage.cc
//# Copyright (C) 2008-2013  ASTRON (Netherlands Institute for Radio Astronomy)
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
//# $Id: tAH_TestStorage.cc 24385 2013-03-26 10:43:55Z amesfoort $

//# Always #include <lofar_config.h> first!
#include <lofar_config.h>

#include <string>
#include <stdexcept>
#if defined HAVE_MPI
#include <mpi.h>
#endif

#include <Common/lofar_iostream.h>
#include <Common/LofarLogger.h>
#include <Common/LofarLocators.h>
#include <CoInterface/Parset.h>
#include <CoInterface/Exceptions.h>
#include <OutputProc/SubbandWriter.h>
#include <OutputProc/Package__Version.h>

using namespace LOFAR;
using namespace LOFAR::Cobalt;


int main(int argc, char *argv[])
{
  std::string type = "brief";
  Version::show<StorageVersion> (std::cout, "Storage", type);

  ConfigLocator aCL;
  string progName = basename(argv[0]);
  string logPropFile(progName + ".log_prop");
  INIT_LOGGER (aCL.locate(logPropFile).c_str());
  LOG_DEBUG_STR("Initialized logsystem with: " << aCL.locate(logPropFile));

#if defined HAVE_MPI
  int rank;
  int size;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
#else
  int rank = 0;
  int size = 1;
#endif

  try {
    if (argc == 3)
      std::cerr << "WARNING: specifying nrRuns is deprecated --- ignored" << std::endl;
    else if (argc != 2)
      THROW(StorageException, std::string("usage: ") << argv[0] << " parset");

    std::clog << "trying to use parset \"" << argv[1] << '"' << std::endl;
    Parset parset(argv[1]);
    parset.adoptFile("OLAP.parset");

    SubbandWriter subbandWriter(&parset, rank, size);

  } catch (Exception &ex) {
    std::cerr << "caught Exception: " << ex.what() << std::endl;
    exit(1);
  } catch (std::exception &ex) {
    std::cerr << "caught std::exception: " << ex.what() << std::endl;
    exit(1);
  } catch (...) {
    std::cerr << "caught unknown exception" << std::endl;
    exit(1);
  }

#if defined HAVE_MPI
  MPI_Finalize();
#endif

  return 0;
}

