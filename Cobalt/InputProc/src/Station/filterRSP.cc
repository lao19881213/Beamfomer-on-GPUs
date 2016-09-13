//# filterRSP.h
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
//# $Id: filterRSP.cc 26855 2013-10-03 12:18:51Z mol $

#include <lofar_config.h>

#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <unistd.h>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <Common/LofarLogger.h>
#include <ApplCommon/PosixTime.h>
#include <CoInterface/Stream.h>
#include <CoInterface/SmartPtr.h>
#include "RSP.h"
#include "PacketReader.h"

using namespace LOFAR;
using namespace Cobalt;

time_t parseTime(const char *str)
{
  return to_time_t(boost::posix_time::time_from_string(str));
}

void usage()
{
  puts("Usage: filterRSP [options] < input.udp > output.udp");
  puts("");
  puts("-f from       Discard packets before `from' (format: '2012-01-01 11:12:00')");
  puts("-t to         Discard packets at or after `to' (format: '2012-01-01 11:12:00')");
  puts("-s nrbeamlets Reduce or expand the number of beamlets per packet");
  puts("-b bitmode    Discard packets with bitmode other than `bitmode' (16, 8, or 4)");
  puts("-c clock      Discard packets with a clock other than `clock' (200 or 160)");
  puts("-i streamdesc Stream descriptor for input (default = file:/dev/stdin)");
  puts("-o streamdesc Stream descriptor for output (default = file:/dev/stdout)");
  puts("");
  puts("Note: invalid packets are always discarded.");
}

int main(int argc, char **argv)
{
  INIT_LOGGER("filterRSP");

  int opt;

  time_t from = 0;
  time_t to = 0;
  unsigned nrbeamlets = 0;
  unsigned bitmode = 0;
  unsigned clock = 0;

  string inputStreamDesc  = "file:/dev/stdin";
  string outputStreamDesc = "file:/dev/stdout";

  // parse all command-line options
  while ((opt = getopt(argc, argv, "f:t:s:b:c:i:o:")) != -1) {
    switch (opt) {
    case 'f':
      from = parseTime(optarg);
      break;

    case 't':
      to = parseTime(optarg);
      break;

    case 's':
      nrbeamlets = atoi(optarg);
      break;

    case 'b':
      bitmode = atoi(optarg);
      break;

    case 'c':
      clock = atoi(optarg);
      break;

    case 'i':
      inputStreamDesc = optarg;
      break;

    case 'o':
      outputStreamDesc = optarg;
      break;

    default: /* '?' */
      usage();
      exit(1);
    }
  }

  // we expect no further arguments
  if (optind != argc) {
    usage();
    exit(1);
  }

  SmartPtr<Stream> inputStream = createStream(inputStreamDesc, true);
  SmartPtr<Stream> outputStream = createStream(outputStreamDesc, false);
  PacketReader reader("", *inputStream);
  struct RSP packet;

  try {
    for(;; ) {
      if( reader.readPacket(packet) ) {
        // **** Apply FROM filter ****
        if (from > 0 && packet.header.timestamp < from)
          continue;

        // **** Apply TO filter ****
        if (to > 0 && packet.header.timestamp >= to)
          continue;

        // **** Apply BITMODE filter ****
        if (bitmode > 0 && packet.bitMode() != bitmode)
          continue;

        // **** Apply CLOCK filter ****
        if (clock > 0 && packet.clockMHz() != clock)
          continue;

        // **** Apply NRBEAMLETS filter ****
        if (nrbeamlets > 0) {
          // the new number of beamlets has to be valid
          ASSERT(nrbeamlets <= 62 * (16 / packet.bitMode()));

          // convert
          packet.header.nrBeamlets = nrbeamlets;
        }

        // Write packet
        outputStream->write(&packet, packet.packetSize());
      }
    }
  } catch(Stream::EndOfStreamException&) {
  }
}

