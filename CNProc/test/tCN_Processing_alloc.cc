//#  tCN_Processing_alloc.cc: test preprocess/postprocess functionality of CNProc
//#
//#  Copyright (C) 2006
//#  ASTRON (Netherlands Foundation for Research in Astronomy)
//#  P.O.Box 2, 7990 AA Dwingeloo, The Netherlands, seg@astron.nl
//#
//#  This program is free software; you can redistribute it and/or modify
//#  it under the terms of the GNU General Public License as published by
//#  the Free Software Foundation; either version 2 of the License, or
//#  (at your option) any later version.
//#
//#  This program is distributed in the hope that it will be useful,
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//#  GNU General Public License for more details.
//#
//#  You should have received a copy of the GNU General Public License
//#  along with this program; if not, write to the Free Software
//#  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//#
//#  $Id: tCN_Processing.cc 19102 2011-10-26 11:52:29Z mol $

//# Always #include <lofar_config.h> first!
#include <lofar_config.h>
#include <CNProc/LocationInfo.h>
#include <CNProc/CN_Processing.h>
#include <Interface/Parset.h>
#include <Stream/NullStream.h>
#include <cstdlib>

#if defined HAVE_MPI
#define MPICH_IGNORE_CXX_SEEK
#include <mpi.h>
#endif

using namespace LOFAR;
using namespace LOFAR::RTCP;

Stream *createIONstream(unsigned channel, const LocationInfo &locationInfo)
{
  (void)channel;
  (void)locationInfo;

  return new NullStream();
}

int main(int argc, char **argv) {
#if defined HAVE_MPI
  MPI_Init(&argc, &argv);
#else
  argc = argc; argv = argv;    // Keep compiler happy ;-)
#endif

  setenv("NR_PSETS", "64", 1);
  setenv("PSET_SIZE", "64", 1);

  LocationInfo locationInfo;
  CN_Processing_Base *proc;
  Parset parset;
  std::vector<SmartPtr<Stream> > inputStreams(1);
  inputStreams[0] = new NullStream;

  parset.adoptFile("tCN_Processing_alloc.parset");

  // preprocess
  switch (parset.nrBitsPerSample()) {
    case 4:  proc = new CN_Processing<i4complex>(parset, inputStreams, &createIONstream, locationInfo);
             break;

    case 8:  proc = new CN_Processing<i8complex>(parset, inputStreams, &createIONstream, locationInfo);
             break;

    case 16: proc = new CN_Processing<i16complex>(parset, inputStreams, &createIONstream, locationInfo);
             break;

    default: return 1;         
  }

  // postprocess
  delete proc;

#if defined HAVE_MPI
  MPI_Finalize();
#endif
  
  return 0;
}
