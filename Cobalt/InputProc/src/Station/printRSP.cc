//# printRSP.cc: print one RSP header retrieved from stdin
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
//# $Id: printRSP.cc 25497 2013-06-27 05:36:14Z mol $

//# Always #include <lofar_config.h> first!
#include <lofar_config.h>

#include "RSP.h"

#include <ctime>
#include <cstring>
#include <cstdlib>
#include <string>
#include <iostream>

#include <Common/LofarLogger.h>
#include <Common/DataConvert.h>
#include <Stream/FileStream.h>
#include <InputProc/RSPTimeStamp.h>

using namespace LOFAR;
using namespace LOFAR::Cobalt;
using namespace std;

void report( const string &filename )
{
  FileStream f(filename);

  struct RSP packet;

  // read header
  f.read( &packet.header, sizeof (RSP::Header) );

#ifdef WORDS_BIGENDIAN
  dataConvert(LittleEndian, packet.header.configuration);
  dataConvert(LittleEndian, packet.header.timestamp);
  dataConvert(LittleEndian, packet.header.blockSequenceNumber);
#endif

  TimeStamp timeStamp = packet.timeStamp();
  time_t seconds = timeStamp.getSeqId();

  char buf[26];
  ctime_r(&seconds, buf);
  buf[strlen(buf) -1] = 0; // remove trailing \n

  cout << "Time stamp:   " << buf << " sample " << timeStamp.getBlockId() << endl;
  cout << "RSP version:  " << (int)packet.header.version << endl;
  cout << "RSP board nr: " << packet.rspBoard() << endl;
  cout << "Payload OK:   " << (packet.payloadError() ? "NO" : "YES") << endl;
  cout << "Clock:        " << packet.clockMHz() << " MHz" << endl;
  cout << "Bit mode:     " << packet.bitMode() << " bit" << endl;
  cout << "Blocks:       " << (int)packet.header.nrBlocks << endl;
  cout << "Beamlets:     " << (int)packet.header.nrBeamlets << endl;

  // read payload
  f.read( &packet.payload, packet.packetSize() - sizeof (RSP::Header) );

#ifdef WORDS_BIGENDIAN
  if (packet.bitMode() == 16)
    dataConvert(LittleEndian, (int16*)&packet.payload, packet.header.nrBlocks * packet.header.nrBeamlets * 2 * 2);
#endif
}

int main()
{
  // Force printing times in UTC
  setenv("TZ", "UTC", 1);

  INIT_LOGGER("printRSP");

  report("/dev/stdin");

  return 0;
}

