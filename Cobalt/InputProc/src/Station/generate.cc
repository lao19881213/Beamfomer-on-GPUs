//# generator.cc: Generates fake data resembling a single station.
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
//# $Id: generate.cc 25598 2013-07-08 12:31:36Z mol $

#include <lofar_config.h>

#include <string>
#include <vector>
#include <omp.h>

#include <Common/LofarLogger.h>
#include <CoInterface/Stream.h>

#include <InputProc/Station/PacketFactory.h>
#include <InputProc/Station/Generator.h>

using namespace LOFAR;
using namespace Cobalt;
using namespace std;

int main( int argc, char **argv )
{
  INIT_LOGGER( "generator" );

  omp_set_nested(true);
  omp_set_num_threads(16);

  if (argc < 3) {
    cerr << "Syntax: generator stationname stream [stream] ..." << endl;
    exit(1);
  }

  const string stationName = argv[1];
  vector< SmartPtr<Stream> > outputStreams;
  for (int i = 2; i < argc; ++i) {
    const string desc = argv[i];

    LOG_INFO_STR("Connecting to " << desc);
    outputStreams.push_back(createStream(desc, false));
    LOG_INFO_STR("Connected.");
  }

  struct StationID stationID(stationName, "LBA");
  struct BufferSettings settings(stationID, false);
  struct BoardMode mode(16, 200);

  const TimeStamp from(time(0), 0, mode.clockHz());
  const TimeStamp to(0);

  PacketFactory factory(mode);
  Generator g(settings, outputStreams, factory, from, to);

  // Generate packets
  g.process();
}

