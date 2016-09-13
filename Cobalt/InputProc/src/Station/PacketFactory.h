/* PacketFactory.h
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
 * $Id: PacketFactory.h 26881 2013-10-07 07:45:48Z loose $
 */

#ifndef LOFAR_INPUT_PROC_PACKETFACTORY_H
#define LOFAR_INPUT_PROC_PACKETFACTORY_H

#include <InputProc/Buffer/SampleBuffer.h>
#include <InputProc/RSPTimeStamp.h>

#include "RSP.h"

namespace LOFAR
{
  namespace Cobalt
  {
    // Generic factory for creating standard RSP packets.
    class PacketFactory
    {
    public:
      PacketFactory( const BoardMode &mode );

      virtual ~PacketFactory();

      // Fill an RSP packet for a certain RSP board and time stamp.
      // \return \c true if successful, \c false otherwise.
      bool makePacket( RSP &packet, const TimeStamp &timestamp, size_t boardNr);

    protected:
      const BoardMode boardMode;

      // Fill RSP packet header.
      // \return \c true if successful, \c false otherwise.
      virtual bool makeHeader( RSP &packet, const TimeStamp &timestamp, size_t boardNr);

      // Fill RSP packet payload.
      // \return \c true if successful, \c false otherwise.
      // \attention This method creates dummy data. Please override it in a
      // derived class if you want to have useful payload data.
      virtual bool makePayload( RSP &packet );
    };

  }
}

#endif

