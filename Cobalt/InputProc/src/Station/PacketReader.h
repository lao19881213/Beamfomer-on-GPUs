//# PacketReader.h
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
//# $Id: PacketReader.h 26852 2013-10-03 10:48:20Z mol $

#ifndef LOFAR_INPUT_PROC_PACKETREADER_H
#define LOFAR_INPUT_PROC_PACKETREADER_H

#include <string>

#include <Common/Exception.h>
#include <Stream/Stream.h>

#include "RSP.h"

namespace LOFAR
{
  namespace Cobalt
  {

    /*
     * Reads RSP packets from a Stream, and collects statistics.
     *
     * Thread-safefy: none.
     */
    class PacketReader
    {
    public:
      PacketReader( const std::string &logPrefix, Stream &inputStream );

      // Reads a packet from the input stream. Returns true if a packet was
      // succesfully read.
      bool readPacket( struct RSP &packet );

      // Logs (and resets) statistics about the packets read.
      void logStatistics();

    private:
      const std::string logPrefix;

      // The stream from which packets are read.
      Stream &inputStream;

      // Whether inputStream can do a small read() without data loss.
      bool supportPartialReads;

      // Statistics covering the packets read so far
      size_t nrReceived; // nr. of packets received
      size_t nrBadSize; // nr. of packets with wrong size (only if supportPartialReads == false)
      size_t nrBadTime; // nr. of packets with an illegal time stamp
      size_t nrBadData; // nr. of packets with payload errors
      size_t nrBadOther; // nr. of packets that are bad in another fashion (illegal header, etc)

      bool hadSizeError; // already reported about wrongly sized packets since last logStatistics()

      double lastLogTime; // time since last log print, to monitor data rates
    };


  }
}

#endif

