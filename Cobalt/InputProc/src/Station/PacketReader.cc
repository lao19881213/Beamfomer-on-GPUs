//# PacketReader.cc
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
//# $Id: PacketReader.cc 26852 2013-10-03 10:48:20Z mol $

#include <lofar_config.h>

#include "PacketReader.h"

#include <typeinfo>
#include <sys/time.h>
#include <boost/format.hpp>

#include <Common/LofarLogger.h>
#include <Stream/SocketStream.h>
#include <CoInterface/Stream.h>


namespace LOFAR
{
  namespace Cobalt
  {


    PacketReader::PacketReader( const std::string &logPrefix, Stream &inputStream )
      :
      logPrefix(str(boost::format("%s [PacketReader] ") % logPrefix)),
      inputStream(inputStream),

      nrReceived(0),
      nrBadSize(0),
      nrBadTime(0),
      nrBadData(0),
      nrBadOther(0),
      hadSizeError(false),
      lastLogTime(0)
    {
      // Partial reads are not supported on UDP streams, because each read()
      // will consume a full packet.
      try {
        SocketStream &asSocket = dynamic_cast<SocketStream &>(inputStream);
        const bool isUDP = asSocket.protocol == SocketStream::UDP;

        supportPartialReads = !isUDP;
      } catch (std::bad_cast&) {
        // inputStream is not a SocketStream
        supportPartialReads = true;
      }
    }


    bool PacketReader::readPacket( struct RSP &packet )
    {
      if (supportPartialReads) {
        // read header first
        inputStream.read(&packet.header, sizeof packet.header);

        // read rest of packet
        inputStream.read(&packet.payload.data, packet.packetSize() - sizeof packet.header);

        ++nrReceived;
      } else {
        // read full packet at once -- numbytes will tell us how much we've actually read
        size_t numbytes = inputStream.tryRead(&packet, sizeof packet);

        ++nrReceived;

        if( numbytes < sizeof(struct RSP::Header)
            || numbytes != packet.packetSize() ) {

          if (!hadSizeError) {
            LOG_ERROR_STR( logPrefix << "Packet is " << numbytes << " bytes, but should be " << packet.packetSize() << " bytes" );
            hadSizeError = true;
          }

          ++nrBadSize;
          return false;
        }
      }

      // illegal version means illegal packet
      if (packet.header.version < 2) {
        // This mainly catches packets that are all zero (f.e. /dev/zero or
        // null: streams).
        ++nrBadOther;
        return false;
      }

      // illegal timestamp means illegal packet
      if (packet.header.timestamp == ~0U) {
        ++nrBadTime;
        return false;
      }

      // discard packets with errors
      if (packet.payloadError()) {
        ++nrBadData;
        return false;
      }

      // everything is ok
      return true;
    }


    void PacketReader::logStatistics()
    {
      // Determine time since last log
      struct timeval tv;
      gettimeofday(&tv, NULL);
      const double now = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;

      const double interval = now - lastLogTime;

      // Emit log line
      LOG_INFO_STR( logPrefix << (nrReceived/interval) << " pps: received " << nrReceived << " packets: " << nrBadTime << " bad timestamps, " << nrBadSize << " bad sizes, " << nrBadData << " payload errors, " << nrBadOther << " otherwise bad packets" );

      // Reset counters
      nrReceived = 0;
      nrBadTime = 0;
      nrBadSize = 0;
      nrBadData = 0;
      nrBadOther = 0;

      hadSizeError = false;
    }


  }
}

