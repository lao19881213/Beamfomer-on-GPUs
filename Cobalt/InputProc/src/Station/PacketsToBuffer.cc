//# PacketsToBuffer.cc
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
//# $Id: PacketsToBuffer.cc 27154 2013-10-30 18:43:16Z mol $

#include <lofar_config.h>

#include "PacketsToBuffer.h"

#include <boost/format.hpp>

#include <Common/LofarLogger.h>

#include <InputProc/SampleType.h>
#include <InputProc/Buffer/SampleBuffer.h>

#include "PacketWriter.h"

namespace LOFAR
{
  namespace Cobalt
  {


    PacketsToBuffer::PacketsToBuffer( Stream &inputStream, const BufferSettings &settings, unsigned boardNr )
      :
      logPrefix(str(boost::format("[station %s board %u] [PacketsToBuffer] ") % settings.station.name() % boardNr)),
      inputStream(inputStream),
      lastlog_timestamp(0),
      settings(settings),
      boardNr(boardNr)
    {
      LOG_DEBUG_STR( logPrefix << "Initialised" );
    }


    void PacketsToBuffer::process()
    {
      // Holder for packet
      struct RSP packet;

      // Create the buffer, regardless of mode
      GenericSampleBuffer buffer(settings, SharedMemoryArena::CREATE);

      // Keep track of the desired mode
      struct BoardMode mode;

      // Whether packet has been read already
      bool packetValid = false;

      // Keep reading if mode changes
      for(;; ) {
        try {
          // Process packets based on (expected) bit mode
          switch(mode.bitMode) {
          case 16:
            process< SampleType<i16complex> >(packet, mode, packetValid);
            break;

          case 8:
            process< SampleType<i8complex> >(packet, mode, packetValid);
            break;

          case 4:
            process< SampleType<i4complex> >(packet, mode, packetValid);
            break;
          }

          // process<>() exited gracefully, so we're done
          break;
        } catch (BadModeException &ex) {
          // Mode switch detected
          LOG_INFO_STR(logPrefix << "Mode switch detected to " << packet.clockMHz() << " MHz, " << packet.bitMode() << " bit");

          // change mode
          mode.bitMode = packet.bitMode();
          mode.clockMHz = packet.clockMHz();

          // Process packet again
          packetValid = true;
        }
      }
    }


    void PacketsToBuffer::logStatistics( PacketReader &reader, const struct RSP &packet )
    {
      if (packet.header.timestamp > lastlog_timestamp + LOG_INTERVAL) {
        lastlog_timestamp = packet.header.timestamp;

        reader.logStatistics();
      }
    }


    template<typename T>
    void PacketsToBuffer::process( struct RSP &packet, const struct BoardMode &mode, bool writeGivenPacket )
    {
      // Create input structures
      PacketReader reader(logPrefix, inputStream);

      // Create output structures
      SampleBuffer<T> buffer(settings, SharedMemoryArena::READWRITE);
      PacketWriter<T> writer(logPrefix, buffer, mode, boardNr);

      LOG_DEBUG_STR( logPrefix << "Processing packets" );

      try {
        // Process lingering packet from previous run, if any
        if (writeGivenPacket) {
          writer.writePacket(packet);
          logStatistics(reader, packet);
        }

        // Transport packets from reader to writer
        for(;; ) {
          if (reader.readPacket(packet)) {
            writer.writePacket(packet);
            logStatistics(reader, packet);
          }
        }

      } catch (BadModeException &ex) {
        // Packet has different clock or bitmode
        throw;

      } catch (Stream::EndOfStreamException &ex) {
        // Ran out of data
        LOG_INFO_STR( logPrefix << "End of stream");

      } catch (SystemCallException &ex) {
        if (ex.error == EINTR)
          LOG_INFO_STR( logPrefix << "Aborted: " << ex.what());
        else
          LOG_ERROR_STR( logPrefix << "Caught Exception: " << ex);

      } catch (Exception &ex) {
        LOG_ERROR_STR( logPrefix << "Caught Exception: " << ex);
      }

      writer.noMoreWriting();

      LOG_DEBUG_STR( logPrefix << "End");
    }


    // Explcitly create the instances we use
    template void PacketsToBuffer::process< SampleType<i16complex> >( struct RSP &packet, const struct BoardMode &mode, bool writeGivenPacket );
    template void PacketsToBuffer::process< SampleType<i8complex> >( struct RSP &packet, const struct BoardMode &mode, bool writeGivenPacket );
    template void PacketsToBuffer::process< SampleType<i4complex> >( struct RSP &packet, const struct BoardMode &mode, bool writeGivenPacket );


    MultiPacketsToBuffer::MultiPacketsToBuffer( const BufferSettings &settings, const std::vector< SmartPtr<Stream> > &inputStreams_ )
    :
      RSPBoards("", inputStreams_.size()),

      settings(settings),
      buffer(settings, SharedMemoryArena::CREATE),
      inputStreams(inputStreams_.size()),

      lastlog_time(now()),
      sum_flags(buffer.nrBoards, 0.0),
      num_flags(0.0)
    {
      // Don't take over ownership!
      for (size_t i = 0; i < inputStreams.size(); ++i) {
        inputStreams[i] = inputStreams_[i];
      }
    }


    MultiPacketsToBuffer::~MultiPacketsToBuffer()
    {
      // collect the log line
      std::stringstream logstr;

      // compute average loss per board
      for (size_t b = 0; b < buffer.nrBoards; ++b) {
        const double avgloss = num_flags == 0.0 ? 0.0 : sum_flags[b] / num_flags;

        if (b > 0)
          logstr << ", ";

        logstr << avgloss << "%";
      }

      // report average loss
      LOG_INFO_STR(str(boost::format("[station %s] ") % settings.station.name()) << "Average data loss per board: " << logstr.str());
    }


    void MultiPacketsToBuffer::processBoard( size_t boardNr )
    {
      PacketsToBuffer board(*inputStreams[boardNr], settings, boardNr);

      board.process();
    }



    void MultiPacketsToBuffer::logStatistics()
    {
      // No use to collect drop rates in non-real-time mode
      if (settings.sync)
        return;

      const double from_ts  = lastlog_time;
      const double to_ts    = now();
      const double maxdelay = 0.5; // wait this many seconds for data to arrive

      std::vector<double> flags(buffer.nrBoards);

      // only log if at least one board flagged
      bool do_log = false;

      for (size_t b = 0; b < buffer.nrBoards; ++b) {
        // collect loss for [from_ts - maxdelay, to_ts - maxdelay)
        const struct BoardMode mode = *(buffer.boards[b].mode);
        const size_t Hz = mode.clockHz();

        // timestamp = (seconds since 1970) * clock / 1024
        flags[b] = buffer.boards[b].flagPercentage(
          TimeStamp((from_ts - maxdelay) * Hz / 1024, Hz),
          TimeStamp((to_ts   - maxdelay) * Hz / 1024, Hz));

        do_log = do_log || flags[b] > 0.0;

        // update statistics
        sum_flags[b] += flags[b] * (to_ts - from_ts);
      }

      // update statistics
      num_flags += to_ts - from_ts;

      if (do_log) {
        // collect the log line
        std::stringstream logstr;

        for (size_t b = 0; b < buffer.nrBoards; ++b) {
          if (b > 0)
            logstr << ", ";

          logstr << flags[b] << "%";
        }

        LOG_WARN_STR(str(boost::format("[station %s] ") % settings.station.name()) << "Data loss per board: " << logstr.str());
      }

      // start from here next time
      lastlog_time = to_ts;
    }


    double MultiPacketsToBuffer::now() const
    {
      struct timeval tv;
      gettimeofday(&tv, NULL);

      return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
    }
  }
}
