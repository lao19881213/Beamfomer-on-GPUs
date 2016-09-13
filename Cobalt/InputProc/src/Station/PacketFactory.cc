/* PacketFactory.cc
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
 * $Id: PacketFactory.cc 26881 2013-10-07 07:45:48Z loose $
 */

#include <lofar_config.h>

#include "PacketFactory.h"

#include <string.h>

namespace LOFAR
{
  namespace Cobalt
  {

    PacketFactory::PacketFactory( const struct BoardMode &mode )
      :
      boardMode(mode)
    {
    }

    PacketFactory::~PacketFactory()
    {
    }

    bool PacketFactory::makePacket( struct RSP &packet, const TimeStamp &timestamp, size_t boardNr )
    {
      return 
        makeHeader(packet, timestamp, boardNr) &&
        makePayload(packet);
    }
    
    bool PacketFactory::makeHeader( struct RSP &packet, const TimeStamp &timestamp, size_t boardNr )
    {
      // configure the packet header
      packet.header.version = 3; // we emulate BDI 6.0

      packet.header.sourceInfo1 =
        (boardNr & 0x1F) | (boardMode.clockMHz == 200 ? 1 << 7 : 0);

      switch (boardMode.bitMode) {
      case 16:
        packet.header.sourceInfo2 = 0;
        break;

      case 8:
        packet.header.sourceInfo2 = 1;
        break;

      case 4:
        packet.header.sourceInfo2 = 2;
        break;
      }

      packet.header.nrBeamlets = boardMode.nrBeamletsPerBoard();
      packet.header.nrBlocks = 16;

      packet.header.timestamp = timestamp.getSeqId();
      packet.header.blockSequenceNumber = timestamp.getBlockId();

      // verify whether the packet really reflects what we intended
      ASSERT(packet.rspBoard()     == boardNr);
      ASSERT(packet.payloadError() == false);
      ASSERT(packet.bitMode()      == boardMode.bitMode);
      ASSERT(packet.clockMHz()     == boardMode.clockMHz);

      // verify that the packet has a valid size
      ASSERT(packet.packetSize()   <= sizeof packet);

      return true;
    }

    bool PacketFactory::makePayload( struct RSP &packet )
    {
      // insert data that is different for each packet
      int64 data = packet.timeStamp();

      memset(packet.payload.data, data & 0xFF, sizeof packet.payload.data);

      return true;
    }

  }
}

