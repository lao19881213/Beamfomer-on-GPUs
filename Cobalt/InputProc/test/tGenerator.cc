//# tGenerator.cc
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
//# $Id: tGenerator.cc 25598 2013-07-08 12:31:36Z mol $

#include <lofar_config.h>

#include <unistd.h>
#include <string>
#include <vector>
#include <omp.h>

#include <Common/LofarLogger.h>
#include <CoInterface/Stream.h>

#include <InputProc/OMPThread.h>
#include <InputProc/Station/PacketFactory.h>
#include <InputProc/Station/Generator.h>
#include <InputProc/Station/PacketReader.h>

using namespace LOFAR;
using namespace Cobalt;
using namespace std;

// The number of packets to transmit (note: there are 16 time samples/packet)
#define NUMPACKETS (200000000 / 1024 / 16)

int main( int, char **argv )
{
  INIT_LOGGER( argv[0] );

  // Don't run forever if communication fails for some reason
  alarm(10);

  omp_set_nested(true);
  omp_set_num_threads(16);

  OMPThread::init();

  const string desc = "tcp:localhost:54321";

  vector< SmartPtr<Stream> > inputStreams(1);
  vector< SmartPtr<Stream> > outputStreams(1);

  #pragma omp parallel sections
  {
    #pragma omp section
    inputStreams[0] = createStream(desc, true);

    #pragma omp section
    outputStreams[0] = createStream(desc, false);
  }

  struct StationID stationID("RS106", "LBA");
  struct BufferSettings settings(stationID, false);
  struct BoardMode mode(16, 200);

  const TimeStamp from(time(0), 0, mode.clockHz());
  const TimeStamp to = from + NUMPACKETS * 16; /* 16 timeslots/packet */
  PacketFactory factory(mode);
  Generator g(settings, outputStreams, factory, from, to);

  bool error = false;

  #pragma omp parallel sections
  {
    #pragma omp section
    {
      // Generate packets

      try {
        g.process();
      } catch(Exception &ex) {
        LOG_ERROR_STR("Caught exception: " << ex);
        error = true;
      }
    }

    #pragma omp section
    {
      // Read and verify the generated packets

      try {
        PacketReader reader("", *inputStreams[0]);

        for(size_t nr = 0; nr < NUMPACKETS; ++nr) {
          struct RSP packet;

          if (!reader.readPacket(packet)) {
            reader.logStatistics();

            ASSERT(false);
          }
        }

        // We received NUMPACKETS packets, kill the generator
        g.stop();
      } catch(Exception &ex) {
        LOG_ERROR_STR("Caught exception: " << ex);
        error = true;
      }
    }
  }

  return error ? 1 : 0;
}

