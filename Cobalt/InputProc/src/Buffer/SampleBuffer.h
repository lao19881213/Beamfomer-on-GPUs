//# SampleBuffer.h
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
//# $Id: SampleBuffer.h 27149 2013-10-30 15:44:16Z mol $

#ifndef LOFAR_INPUT_PROC_SAMPLEBUFFER_H
#define LOFAR_INPUT_PROC_SAMPLEBUFFER_H

#include <string>
#include <vector>
#include <map>

#include <CoInterface/MultiDimArray.h>
#include <CoInterface/Allocator.h>
#include <InputProc/RSPTimeStamp.h>
#include <InputProc/SampleType.h>
#include "BufferSettings.h"
#include "BoardMode.h"
#include "SharedMemory.h"
#include "Ranges.h"

namespace LOFAR
{
  namespace Cobalt
  {
    /*
     * Maintains a sample buffer in shared memory, which can be created
     * or attached to.
     *
     * The sample buffer contains the following information:
     *
     *   1. A copy of `settings', against which attaches are verified.
     *   2. A beamlets matrix. [subband][sample]
     *   3. A boards vector. [board]
     *
     * The IPC key used for the shared memory is settings.dataKey.
     *
     * The buffer can run in synchronised mode (settings.sync = true), in which case
     * we support exactly one writer and one reader to run in sync through
     * the Boards's functions. The writer synchronisation makes sure that
     * no data is being overwritten that is still in the reader's future. The
     * reader synchronisation makes sure that the reader will wait for the
     * writer to complete writing the requested range.
     */
    template<typename T>
    class SampleBuffer
    {
    public:

      // Read/write pointers to keep readers and writers in sync
      // if buffer.sync == true. The pointers assume that data will both be read
      // and written in-order.
      SampleBuffer( const struct BufferSettings &settings, SharedMemoryArena::Mode shmMode );

    private:
      const std::string logPrefix;
      SharedMemoryArena data;

      SparseSetAllocator allocator;

      struct BufferSettings *initSettings( const struct BufferSettings &localSettings, bool create );

      static size_t dataSize( const struct BufferSettings &settings );

    public:
      const bool create;
      struct BufferSettings *settings;

      // Keep readers/writers in sync
      const bool sync;
      SyncLock * const syncLock;

      const size_t nrSamples;
      const size_t nrBoards;
      const size_t nrAvailableRanges; // width of each available range

      MultiDimArray<T,2>  beamlets; // [subband][sample]

      size_t offset( const TimeStamp &timestamp ) const { return (int64)timestamp % nrSamples; }

      // Signal that there will be no reads before the given epoch
      void noReadBefore( size_t beamlet, const TimeStamp &epoch );

      // Signal start of read intent for data in [begin, end). Waits for data to arrive
      // until or after `end'.
      void startRead( size_t beamlet, const TimeStamp &begin, const TimeStamp &end );

      // Signal release of data before end, thus allowing it to be overwritten by newer data.
      void stopRead( size_t beamlet, const TimeStamp &end );

      // Signal that we're done reading.
      void noMoreReading( size_t beamlet );

      class Board {
      public:
        Board( SampleBuffer<T> &buffer, size_t boardNr = 0 );

        // The mode this board is operating in.
        volatile struct BoardMode *mode;

        Ranges available;
        size_t boardNr; // Caller can modify this

        // Change the mode of this board
        void changeMode( const struct BoardMode &mode );

        // Report the percentage of missing data in the requested range
        double flagPercentage( const TimeStamp &from, const TimeStamp &to ) const;

        // Signal start of write intent for data in [begin, end). The flags will be updated
        // for any data that will be overwritten, but not set for any data that is
        // written.
        void startWrite( const TimeStamp &begin, const TimeStamp &end );

        // Signal stop of write intent at end.
        void stopWrite( const TimeStamp &end );

        // Signal end-of-data (we're done writing).
        void noMoreWriting();

      private:
        SampleBuffer<T> &buffer;
      };

      std::vector<Board> boards;

    private:
      static const size_t ALIGNMENT = 256;
    };

    // Removes the sample buffers that correspond to settings.dataKey,
    // as well as any sample buffer that refers to the same station.
    void removeSampleBuffers( const BufferSettings &settings );

    // Generic type for SampleType-agnostic functionality.
    typedef SampleBuffer< SampleType<i16complex> > GenericSampleBuffer;
  }
}

#include "SampleBuffer.tcc"

#endif

