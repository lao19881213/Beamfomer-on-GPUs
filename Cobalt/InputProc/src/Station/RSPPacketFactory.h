/* RSPPacketFactory.h
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
 * $Id: RSPPacketFactory.h 26881 2013-10-07 07:45:48Z loose $
 */

#ifndef LOFAR_INPUT_PROC_RSPPACKETFACTORY_H
#define LOFAR_INPUT_PROC_RSPPACKETFACTORY_H

#include "PacketFactory.h"
#include <istream>

namespace LOFAR
{
  namespace Cobalt
  {
    // Specialization of the generic PacketFactory class. This class make it
    // possible, e.g., to manually override the number of beamlets that will be
    // used.
    class RSPPacketFactory : public PacketFactory
    {
    public:
      // Construct a factory that will read its input from the input stream \a
      // inStream.
      RSPPacketFactory(std::istream &inStream, 
                       const BoardMode &mode,
                       unsigned nrSubbands);

      virtual ~RSPPacketFactory();

    private:
      // Fill RSP packet header.
      virtual bool makeHeader(RSP &packet, const TimeStamp &timestamp,
                              size_t boardNr);

      // Fill RSP packet payload.
      virtual bool makePayload(RSP &packet);

      // Create a RSP payload of 16-bit samples
      bool make16bitPayload(RSP &packet);

      // Create a RSP payload of 8-bit samples
      bool make8bitPayload(RSP &packet);

      // Create a RSP payload of 4-bit samples
      bool make4bitPayload(RSP &packet);

      // Number of subbands (or beamlets) to produce.
      unsigned itsNrSubbands;

      // Input stream
      std::istream &itsInputStream;
    };

  }
}

#endif

