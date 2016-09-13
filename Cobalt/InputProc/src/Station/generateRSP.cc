//# generateRSP.cc
//#
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
//# $Id: generateRSP.cc 26881 2013-10-07 07:45:48Z loose $

#include <lofar_config.h>

#include <climits>
#include <cstdlib>
#include <ctime>

#include <fstream>
#include <iostream>
#include <vector>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/shared_ptr.hpp>

#include <Common/LofarLogger.h>
#include <ApplCommon/PosixTime.h>
#include <CoInterface/Stream.h>
#include <InputProc/Buffer/BoardMode.h>
#include <InputProc/RSPTimeStamp.h>
#include <InputProc/Station/RSP.h>
#include <InputProc/Station/RSPPacketFactory.h>

using namespace std;
using namespace boost;
using namespace LOFAR;
using namespace LOFAR::Cobalt;

struct options_t {
  time_t from;
  time_t to;
  unsigned bitmode;
  unsigned clockmode;
  unsigned subbands;
  string streamdesc;
};

options_t default_options = { 0, INT_MAX, 16, 200, 61, "file:/dev/stdout" };

void usage()
{
  char from_string[256], to_string[256];
  const char* format = "%Y-%m-%d %H:%M:%S";
  strftime(from_string, sizeof from_string, format, gmtime(&default_options.from));
  strftime(to_string, sizeof to_string, format, gmtime(&default_options.to));
  cerr << "\nUsage: generateRSP [options] < input.asc > output.rsp\n\n"
       << "-b bitmode    Bitmode `bitmode' (16, 8, or 4)"
       << " (default: " << default_options.bitmode << ")\n"
       << "-c clockmode  Clock frequency (MHz) `clock' (200 or 160)"
       << " (default: " << default_options.clockmode << ")\n"
       << "-f from       Start time `from' (format: '2012-01-01 11:12:00')"
       << " (default: '" << from_string << "')\n"
       << "-h help       Print this help message\n"
       << "-o streamdesc Stream descriptor for output"
       << " (default: '" << default_options.streamdesc << "')\n"
       << "-s subbands   Number of `subbands` (or beamlets)"
       << " (default: " << default_options.subbands << ")\n"
       << "-t to         End time `to' (format: '2012-01-01 11:12:00')"
       << " (default: '" << to_string << "')\n"
       << endl;
}

time_t parseTime(const char *str)
{
  try {
    return to_time_t(posix_time::time_from_string(str));
  } catch (std::exception &err) {
    THROW (Exception, "Invalid date/time: " << err.what());
  }
}

int main(int argc, char **argv)
{
  INIT_LOGGER("generateRSP");

  int opt;

  time_t from = default_options.from;
  time_t to = default_options.to;
  unsigned bitmode = default_options.bitmode;
  unsigned clockmode = default_options.clockmode;
  unsigned subbands = default_options.subbands;
  string streamdesc = default_options.streamdesc;

  try {
    // parse all command-line options
    while((opt = getopt(argc, argv, "b:c:f:ho:s:t:")) != -1) {
      switch(opt) {
      default:
        usage();
        return 1;
      case 'b':
        bitmode = atoi(optarg);
        break;
      case 'c':
        clockmode = atoi(optarg);
        break;
      case 'f':
        from = parseTime(optarg);
        break;
      case 'h':
        usage();
        return 0;
      case 'o':
        streamdesc = optarg;
        break;
      case 's':
        subbands = atoi(optarg);
        break;
      case 't':
        to = parseTime(optarg);
        break;
      }
    }

    // validate command-line options
    ASSERTSTR(from < to, from << " < " << to);
    ASSERTSTR(bitmode == 16 || bitmode == 8 || bitmode == 4,
              "bitmode = " << bitmode);
    ASSERTSTR(clockmode == 160 || clockmode == 200, "clockmode = " << clockmode);
    ASSERTSTR(subbands > 0, "subbands = " << subbands);

    // we expect no further arguments
    if (optind != argc) {
      usage();
      return 1;
    }

    ifstream inStream("/dev/stdin");
    SmartPtr<Stream> outStream = createStream(streamdesc, false);

    BoardMode boardMode(bitmode, clockmode);
    unsigned boardNr(0);

    RSPPacketFactory packetFactory(inStream, boardMode, subbands);
    RSP packet;

    TimeStamp current(from);
    TimeStamp end(to);

    while(current < end && packetFactory.makePacket(packet, current, boardNr)) {
      // Write packet
      outStream->write(&packet, packet.packetSize());
      // Increment time stamp
      current += packet.header.nrBlocks;
    }

  } catch (Exception& ex) {
    cerr << ex << endl;
    return 1;
  }

}

