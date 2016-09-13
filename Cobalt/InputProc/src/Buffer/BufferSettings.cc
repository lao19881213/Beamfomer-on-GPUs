//# BufferSettings.cc
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
//# $Id: BufferSettings.cc 26336 2013-09-03 10:01:41Z mol $

#include <lofar_config.h>

#include "BufferSettings.h"
#include "BoardMode.h"

#include <Common/LofarLogger.h>
#include "SharedMemory.h"

namespace LOFAR
{
  namespace Cobalt
  {


    BufferSettings::BufferSettings()
      :
      version(currentVersion)
    {
    }

    BufferSettings::BufferSettings(const struct StationID &station, bool attach, time_t timeout)
      :
      version(currentVersion),
      station(station)
    {
      if (attach) {
        do {
          SharedStruct<struct BufferSettings> shm(station.hash(), false, timeout);

          *this = shm.get();
        } while (!valid());

        ASSERTSTR( valid(), "SHM buffer has invalid header (version mismatch?) for " << station );
      } else {
        deriveDefaultSettings();
      }
    }


    void BufferSettings::deriveDefaultSettings()
    {
      sync = false;
      syncLock = 0;

      nrBoards = MAX_NR_RSP_BOARDS;
      nrAvailableRanges = 64;

      // 1 second buffer
      setBufferSize(1.0);

      dataKey = station.hash();
    }


    void BufferSettings::setBufferSize(double seconds)
    {
      // Make sure nrSamples is a multiple of 16, which
      // is the expected number of samples in a block.
      //
      // We use a multiple of 256 for maximum memory-
      // alignment benefits.
      //
      // Doing so allows the writer to prevent split
      // writes of packets. (TODO: That's not implemented,
      // because the timestamps of the packets are not
      // necessarily a multiple of 16).
      //
      // We use 200 MHz clock as a reference.
      const BoardMode mode(16, 200);
      nrSamples_16bit = mode.secondsToSamples(seconds) & ~0xFFLL;
    }


    std::ostream& operator<<( std::ostream &str, const struct BufferSettings &s )
    {
      const BoardMode mode(16, 200);
      str << s.station << " boards: " << s.nrBoards << " buffer: " << (1.0 * s.nrSamples_16bit * 1024/mode.clockHz()) << " sec";

      if (s.sync) {
        str << " [r/w sync]";
      } else {
        str << " [realtime]";
      }

      return str;
    }


  }
}

