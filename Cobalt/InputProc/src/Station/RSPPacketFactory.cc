/* RSPPacketFactory.cc
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
 * $Id: RSPPacketFactory.cc 26881 2013-10-07 07:45:48Z loose $
 */

#include <lofar_config.h>

#include "RSPPacketFactory.h"
#include <cstring>

using namespace std;

namespace LOFAR
{
  namespace Cobalt
  {

    RSPPacketFactory::RSPPacketFactory(istream &inStream,
                                       const BoardMode &mode,
                                       unsigned nrSubbands) :
      PacketFactory(mode),
      itsNrSubbands(nrSubbands),
      itsInputStream(inStream)
    {
      if (nrSubbands == 0 || nrSubbands > mode.nrBeamletsPerBoard()) {
        THROW(Exception, "Invalid number of subbands: " << nrSubbands);
      }
    }

    RSPPacketFactory::~RSPPacketFactory()
    {
    }

    bool RSPPacketFactory::makeHeader( RSP &packet, const TimeStamp &timestamp, size_t boardNr )
    {
      if (PacketFactory::makeHeader(packet, timestamp, boardNr)) {
        packet.header.nrBeamlets = itsNrSubbands;
        return true;
      } else {
        return false;
      }
    }

    bool RSPPacketFactory::makePayload( RSP &packet )
    {
      // Clear payload first, to make sure a packet is always zero padded.
      memset(packet.payload.data, 0, sizeof(packet.payload.data));
      switch(boardMode.bitMode) {
      case 16: return make16bitPayload(packet);
      case 8 : return make8bitPayload(packet);
      case 4 : return make4bitPayload(packet);
      default: THROW(Exception, "Invalid bit mode: " << boardMode.bitMode);
      }
    }

    bool RSPPacketFactory::make16bitPayload( RSP &packet )
    {
      int Xr, Xi, Yr, Yi;
      for(size_t i = 0; i < itsNrSubbands * packet.header.nrBlocks; i++) {
        if (!(itsInputStream >> Xr >> Xi >> Yr >> Yi)) return false;
        RSP::Payload::samples16bit_t& s = packet.payload.samples16bit[i];
        s.Xr = Xr; s.Xi = Xi; s.Yr = Yr; s.Yi = Yi;
      }
      return true;
    }

    bool RSPPacketFactory::make8bitPayload( RSP &packet )
    {
      int Xr, Xi, Yr, Yi;
      for(size_t i = 0; i < itsNrSubbands * packet.header.nrBlocks; i++) {
        if (!(itsInputStream >> Xr >> Xi >> Yr >> Yi)) return false;
        RSP::Payload::samples8bit_t& s = packet.payload.samples8bit[i];
        s.Xr = Xr; s.Xi = Xi; s.Yr = Yr; s.Yi = Yi;
      }
      return true;
    }

    bool RSPPacketFactory::make4bitPayload( RSP &packet )
    {
      int Xr, Xi, Yr, Yi;
      for(size_t i = 0; i < itsNrSubbands * packet.header.nrBlocks; i++) {
        if (!(itsInputStream >> Xr >> Xi >> Yr >> Yi)) return false;
        RSP::Payload::samples4bit_t& s = packet.payload.samples4bit[i];
        s.X = (Xr & 0xF) | ((Xi & 0xF) << 4);
        s.Y = (Yr & 0xF) | ((Yi & 0xF) << 4);
      }
      return true;
    }


  }
}

