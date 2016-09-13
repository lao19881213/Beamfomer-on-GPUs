//# tPacketReader.cc
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
//# $Id: tPacketReader.cc 24442 2013-03-28 09:09:18Z loose $

#include <lofar_config.h>

#include <string>

#include <Common/LofarLogger.h>
#include <Stream/FileStream.h>

#include <InputProc/Station/PacketReader.h>
#include <InputProc/Station/RSP.h>

using namespace LOFAR;
using namespace Cobalt;

void test(const std::string &filename, unsigned bitmode, unsigned nrPackets)
{
  FileStream fs(filename);

  PacketReader reader("", fs);

  struct RSP packet;

  // We should be able to read these packets
  for( size_t i = 0; i < nrPackets; ++i) {
    ASSERT( reader.readPacket(packet) );
    ASSERT( packet.bitMode() == bitmode );
    ASSERT( packet.clockMHz() == 200 );
  }

  // The file contains no more packets; test if readPacket survives
  // a few calls on the rest of the stream.
  for( size_t i = 0; i < 3; ++i) {
    try {
      ASSERT( !reader.readPacket(packet) );
    } catch (Stream::EndOfStreamException &ex) {
      // expected
    }
  }
}

int main()
{
  INIT_LOGGER("tPacketReader");

  test("tPacketReader.in_16bit", 16, 2);
  test("tPacketReader.in_8bit",   8, 2);
}

