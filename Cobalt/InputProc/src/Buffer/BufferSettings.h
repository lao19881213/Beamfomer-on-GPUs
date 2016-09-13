//# BufferSettings.h
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
//# $Id: BufferSettings.h 26336 2013-09-03 10:01:41Z mol $

#ifndef LOFAR_INPUT_PROC_BUFFER_SETTINGS_H
#define LOFAR_INPUT_PROC_BUFFER_SETTINGS_H

#include <ostream>

#include <Common/LofarLogger.h>
#include <Common/LofarTypes.h>
#include <CoInterface/SparseSet.h>
#include <CoInterface/SlidingPointer.h>
#include "StationID.h"
#include "BoardMode.h"

namespace LOFAR
{
  namespace Cobalt
  {
    class SyncLock;

    struct BufferSettings {
    private:
      static const unsigned currentVersion = 1;

      unsigned version;

      bool valid() const
      {
        return version == currentVersion;
      }

    public:
      static const unsigned MAX_NR_RSP_BOARDS = 4;

      typedef uint64 range_type;
      typedef SparseSet<range_type> flags_type;

      struct StationID station;

      // true: sync reader and writer, useful in real-time mode
      bool sync;
      SyncLock *syncLock;

      size_t nrSamples_16bit;

      unsigned nrBoards;
      size_t nrAvailableRanges;

      key_t dataKey;

      BufferSettings();

      // if attach=true, read settings from shared memory, using the given stationID
      // if attach=false, set sane default values
      BufferSettings(const struct StationID &station, bool attach, time_t timeout = 60);

      // Shortcut to set nrSamples to represent `seconds' of buffer.
      void setBufferSize(double seconds);

      bool operator==(const struct BufferSettings &other) const
      {
        return station == other.station
               && sync == other.sync
               && nrSamples_16bit == other.nrSamples_16bit
               && nrBoards == other.nrBoards
               && nrAvailableRanges == other.nrAvailableRanges
               && dataKey == other.dataKey;
      }

      size_t nrSamples(unsigned bitMode) const {
        (void)bitMode;

        // The number of samples is invariant to the bitmode, because
        // smaller bitmodes introduce more beamlets.
        return nrSamples_16bit;
      }

    private:

      // Derive sane values from the station field.
      void deriveDefaultSettings();
    };

    std::ostream& operator<<( std::ostream &str, const struct BufferSettings &s );

    class SyncLock {
    public:
      typedef SlidingPointer<BufferSettings::range_type> LockType;

      /*
       * We have a write lock per RSP board, which unlocks readers according to
       * the writer's mode. So the reader must be in the same mode!
       *
       * We have a read lock per beamlet. We assume all beamlets are used up
       * to readLock.size(). Failure to read one of the beamlets will block
       * the writer for the given board.
       */

      std::vector<LockType> writeLock; // [board]
      std::vector<LockType> readLock;  // [beamlet]

      SyncLock(const BufferSettings &settings, size_t nrBeamlets)
      :
        writeLock(settings.nrBoards),
        readLock(nrBeamlets)
      {
      }
    };
  }
}

#endif

