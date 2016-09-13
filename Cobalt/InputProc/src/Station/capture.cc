//# capture.cc: Capture data from a single station.
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
//# $Id: capture.cc 25540 2013-07-02 13:20:36Z mol $

#include <lofar_config.h>

#include <string>
#include <vector>
#include <omp.h>

#include <Common/LofarLogger.h>
#include <CoInterface/Stream.h>

#include <InputProc/Station/PacketsToBuffer.h>

using namespace LOFAR;
using namespace Cobalt;
using namespace std;

int main( int argc, char **argv )
{
  INIT_LOGGER( "capture" );

  omp_set_nested(true);
  omp_set_num_threads(16);

  if (argc < 3) {
    cerr << "Syntax: capture stationname stream [stream] ..." << endl;
    exit(1);
  }

  const string stationName = argv[1];

  vector< SmartPtr<Stream> > inputStreams;
  for (int i = 2; i < argc; ++i) {
    const string desc = argv[i];

    LOG_INFO_STR("Connecting to " << desc);
    inputStreams.push_back(createStream(desc, true));
    LOG_INFO_STR("Connected.");
  }

  // Fill a BufferSettings object
  struct StationID stationID(stationName, "LBA");
  struct BufferSettings settings(stationID, false);

  MultiPacketsToBuffer mpb(settings, inputStreams);

  mpb.process();

  return 0;
}


