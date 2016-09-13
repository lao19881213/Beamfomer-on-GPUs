//# Generator.cc
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
//# $Id: Generator.cc 25015 2013-05-22 22:35:17Z amesfoort $

#include <lofar_config.h>

#include "Generator.h"

#include <boost/format.hpp>

#include <Common/LofarLogger.h>
#include <Stream/Stream.h>
#include <CoInterface/SmartPtr.h>
#include <CoInterface/Stream.h>


namespace LOFAR
{
  namespace Cobalt
  {

    Generator::Generator( const BufferSettings &settings, const std::vector< SmartPtr<Stream> > &outputStreams_, PacketFactory &packetFactory, const TimeStamp &from, const TimeStamp &to )
      :
      RSPBoards(str(boost::format("[station %s %s] [Generator] ") % settings.station.stationName % settings.station.antennaField), outputStreams_.size()),
      settings(settings),
      outputStreams(outputStreams_.size()),
      packetFactory(packetFactory),
      nrSent(nrBoards, 0),
      from(from),
      to(to)
    {
      for (size_t i = 0; i < outputStreams.size(); ++i) {
        outputStreams[i] = outputStreams_[i];
      }

      LOG_INFO_STR( logPrefix << "Initialised" );
    }


    void Generator::processBoard( size_t nr )
    {
      const std::string logPrefix(str(boost::format("[station %s %s board %u] [Generator] ") % settings.station.stationName % settings.station.antennaField % nr));

      try {
        Stream &s = *outputStreams[nr];

        LOG_INFO_STR( logPrefix << "Start" );

        for(TimeStamp current = from; !to || current < to; /* increment in loop */ ) {
          struct RSP packet;

          // generate packet
          packetFactory.makePacket( packet, current, nr );

          // wait until it is due
          if (!waiter.waitUntil(current))
            break;

          // send packet
          try {
            s.write(&packet, packet.packetSize());
          } catch (SystemCallException &ex) {
            // UDP can return ECONNREFUSED or EINVAL if server does not have its port open
            if (ex.error != ECONNREFUSED && ex.error != EINVAL)
              throw;
          }

          nrSent[nr]++;

          current += packet.header.nrBlocks;
        }
      } catch (Stream::EndOfStreamException &ex) {
        LOG_INFO_STR( logPrefix << "End of stream");
      } catch (SystemCallException &ex) {
        if (ex.error == EINTR)
          LOG_INFO_STR( logPrefix << "Aborted: " << ex.what());
        else
          LOG_ERROR_STR( logPrefix << "Caught Exception: " << ex);
      } catch (Exception &ex) {
        LOG_ERROR_STR( logPrefix << "Caught Exception: " << ex);
      }

      LOG_INFO_STR( logPrefix << "End");
    }

    void Generator::logStatistics()
    {
      for( size_t nr = 0; nr < nrBoards; nr++ ) {
        const std::string logPrefix(str(boost::format("[station %s %s board %u] [Generator] ") % settings.station.stationName % settings.station.antennaField % nr));

        LOG_INFO_STR( logPrefix << nrSent[nr] << " packets sent.");

        nrSent[nr] = 0;
      }
    }

  }
}

