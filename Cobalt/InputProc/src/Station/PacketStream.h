/* PacketStream.h
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
 * $Id: PacketStream.h 25497 2013-06-27 05:36:14Z mol $
 */

#ifndef LOFAR_INPUT_PROC_PACKETSTREAM_H
#define LOFAR_INPUT_PROC_PACKETSTREAM_H

#include <Stream/Stream.h>
#include <Common/Thread/Cancellation.h>
#include <InputProc/RSPTimeStamp.h>
#include "PacketFactory.h"
#include "RSP.h"

namespace LOFAR
{
  namespace Cobalt
  {
    /* Generate a Stream of RSP packets. */

    class PacketStream: public Stream
    {
    public:
      // 'factory' will be copied.
      PacketStream( const PacketFactory &factory, const TimeStamp &from, const TimeStamp &to, size_t boardNr = 0 )
      :
        factory(factory),
        from(from),
        to(to),
        current(from),
        boardNr(boardNr),
        offset(0)
      {
      }

      virtual size_t tryRead(void *ptr, size_t size)
      {
        Cancellation::point();

        if (current >= to)
          THROW(EndOfStreamException, "No data beyond " << to);

        if (offset == 0) {
          // generate new packet
          factory.makePacket(packet, current, boardNr);

          current += packet.header.nrBlocks;
        }

        size_t numBytes = std::min(packet.packetSize() - offset, size);

        memcpy(ptr, reinterpret_cast<char*>(&packet) + offset, numBytes);

        offset += numBytes;

        if (offset == packet.packetSize()) {
          // written full packet, so we'll need a new one on next read
          offset = 0;
        }

        return numBytes;
      }

      virtual size_t tryWrite(const void *ptr, size_t size)
      {
        // not supported
        (void)ptr;
        (void)size;

        THROW(EndOfStreamException, "Writing to PacketStream is not supported");
      }

    private:
      PacketFactory factory;

      const TimeStamp from;
      const TimeStamp to;
      TimeStamp current;
      const size_t boardNr;

      struct RSP packet;

      // Write offset within packet. If 0, a new
      // packet is required.
      size_t offset;
    };
  }
}

#endif

