
/* tPacketFactory.cc
 * Copyright (C) 2012-2013  ASTRON (Netherlands Institute for Radio Astronomy)
 * P.O. Box 2, 7990 AA Dwingeloo, The Netherlands
 *
 * This file is part of the LOFAR software suite.
 * The LOFAR software suite is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * The LOFAR software suite is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the LOFAR software suite. If not, see <http://www.gnu.org/licenses/>.
 *
 * $Id: tPacketFactory.cc 25598 2013-07-08 12:31:36Z mol $
 */

#include <lofar_config.h>

#include <InputProc/Station/PacketFactory.h>
#include <time.h>

using namespace LOFAR;
using namespace Cobalt;

void test()
{
  struct BoardMode mode(16, 200);
  PacketFactory factory(mode);

  // Just generate packets.
  time_t now = time(0);
  TimeStamp start(now,     0, mode.clockHz());
  TimeStamp end  (now + 1, 0, mode.clockHz());

  // The number of time slots per packet, which will
  // be read from the generated packets.
  size_t timesPerPacket = 16;

  for (TimeStamp i = start; i < end; i += timesPerPacket) {
    struct RSP packet;

    factory.makePacket(packet, i, 0);
    timesPerPacket = packet.header.nrBlocks;

    // Basic sanity checks
    ASSERT(packet.packetSize() <= sizeof packet);
    ASSERT(packet.timeStamp() == i);

    // Prevent infinite loops
    ASSERT(timesPerPacket > 0);
  }
}

int main( int, char ** )
{
  INIT_LOGGER( "tPacketFactory" );

  // Don't run forever if communication fails for some reason
  alarm(10);

  test();
}

