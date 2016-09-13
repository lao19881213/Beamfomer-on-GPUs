/* BlockReader.tcc
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
 * $Id: BlockReader.tcc 26419 2013-09-09 11:19:56Z mol $
 */

#include <Common/LofarLogger.h>

namespace LOFAR {

  namespace Cobalt {

    template<typename T>
    BlockReader<T>::BlockReader( const BufferSettings &settings, const struct BoardMode &mode, const std::vector<size_t> beamlets, double maxDelay )
    :
      settings(settings),
      mode(mode),
      buffer(settings, SharedMemoryArena::READ),

      beamlets(beamlets),
      maxDelay(mode.secondsToSamples(maxDelay), mode.clockHz())
    {
      // Check whether the selected beamlets exist, to prevent out-of-bounds access
      const size_t nrBeamlets = buffer.beamlets.shape()[0];

      for (size_t i = 0; i < beamlets.size(); ++i) {
        ASSERTSTR( beamlets[i] < nrBeamlets, beamlets[i] << " < " << nrBeamlets );
      }
    }


    template<typename T>
    BlockReader<T>::~BlockReader()
    {
      // signal end of reading
      for (std::vector<size_t>::const_iterator b = beamlets.begin(); b != beamlets.end(); ++b) {
        buffer.noMoreReading(*b);
      }
    }


    template<typename T>
    SubbandMetaData::flags_type BlockReader<T>::LockedBlock::flags( size_t beamletIdx ) const
    {
      const struct Block<T>::Beamlet &ib = this->beamlets[beamletIdx];

      // Determine corresponding RSP board
      size_t boardIdx = reader.mode.boardIndex(ib.stationBeamlet);

      ssize_t beam_offset = ib.offset;

      // Translate available samples to missing samples.
      const BufferSettings::range_type from = this->from + beam_offset;
      const BufferSettings::range_type to   = this->to   + beam_offset;
      BufferSettings::flags_type bufferFlags = reader.buffer.boards[boardIdx].available.sparseSet(from, to).invert(from, to);

      if (reader.mode != *(reader.buffer.boards[boardIdx].mode)) {
        LOG_DEBUG_STR("Board in wrong mode -- flagging all data between " << from << " and " << to);

        // Board is in wrong mode! We can't use this data, so flag everything.

        // NOTE: Since a board clears all data every time it switches modes, we know that
        // the data is valid if the mode is, regardless of the number of switches in the past.
        //
        // Race conditions are avoided, because the writer respects this order:
        //
        // 1. clear flags
        // 2. update mode
        // 3. write packets under new mode
        bufferFlags.include(from, to);
      }

      // Convert from global to local indices and types
      SubbandMetaData::flags_type blockFlags;
      for (BufferSettings::flags_type::const_iterator it = bufferFlags.getRanges().begin(); it != bufferFlags.getRanges().end(); it++) {
        blockFlags.include(it->begin - from, it->end - from);
      }

      return blockFlags;
    }


    template<typename T>
    BlockReader<T>::LockedBlock::LockedBlock( BlockReader<T> &reader, const TimeStamp &from, const TimeStamp &to, const std::vector<ssize_t> &beamletOffsets )
    :
      reader(reader)
    {
      this->from = from;
      this->to   = to;

      ASSERT(beamletOffsets.size() == reader.beamlets.size());

      // fill static beamlet info
      this->beamlets.resize(reader.beamlets.size());
      for (size_t i = 0; i < this->beamlets.size(); ++i) {
        this->beamlets[i] = getBeamlet(i, beamletOffsets[i]);
      }

      // clear path for writer
      for (std::vector<size_t>::const_iterator b = reader.beamlets.begin(); b != reader.beamlets.end(); ++b) {
        reader.buffer.noReadBefore(*b, this->from);
      }

      // signal read intent on all buffers
      for (std::vector<size_t>::const_iterator b = reader.beamlets.begin(); b != reader.beamlets.end(); ++b) {
        reader.buffer.startRead(*b, this->from, this->to);
      }

      //LOG_DEBUG_STR("Locked block " << this->from << " to " << this->to);

      // record initial flags
      for (size_t i = 0; i < this->beamlets.size(); ++i) {
        this->beamlets[i].flagsAtBegin = flags(i);
      }
    }


    template<typename T>
    struct Block<T>::Beamlet BlockReader<T>::LockedBlock::getBeamlet( size_t beamletIdx, ssize_t offset )
    {
      // Create instructions for copying this beamlet
      typename Block<T>::Beamlet b;

      // Cache the actual beam number at the station
      b.stationBeamlet = reader.beamlets[beamletIdx];
      
      // Store the sample offset with which this beamlet is read
      b.offset = offset;

      // Determine the relevant offsets in the buffer, processing:
      //   offset: the shift applied to compensate geometric delays (etc)
      size_t from_offset = reader.buffer.offset(this->from + offset);
      size_t to_offset   = reader.buffer.offset(this->to   + offset);

      const size_t bufferSize = reader.buffer.nrSamples;

      if (to_offset == 0)
        // we need the other end, actually
        to_offset = bufferSize;

      // Determine whether we need to wrap around the end of the buffer
      size_t wrap_offset = from_offset < to_offset ? 0 : bufferSize - from_offset;

      const T* origin = &reader.buffer.beamlets[b.stationBeamlet][0];

      if (wrap_offset > 0) {
        // Copy as two parts
        b.nrRanges = 2;

        b.ranges[0].from = origin + from_offset;
        b.ranges[0].to   = origin + bufferSize;

        b.ranges[1].from = origin;
        b.ranges[1].to   = origin + to_offset;
      } else {
        // Copy as one part
        b.nrRanges = 1;

        b.ranges[0].from = origin + from_offset;
        b.ranges[0].to   = origin + to_offset;
      }

      return b;
    }


    template<typename T>
    BlockReader<T>::LockedBlock::~LockedBlock()
    {
      //LOG_DEBUG_STR("Unlocking block " << this->from << " to " << this->to);

      // Signal end of read intent on all buffers
      for (std::vector<size_t>::const_iterator b = reader.beamlets.begin(); b != reader.beamlets.end(); ++b) {
        // Unlock data
        reader.buffer.stopRead(*b, this->to);
      }
    }


    template<typename T>
    SmartPtr<typename BlockReader<T>::LockedBlock> BlockReader<T>::block( const TimeStamp &from, const TimeStamp &to, const std::vector<ssize_t> &beamletOffsets )
    {
      ASSERT( to > from );
      ASSERTSTR( (to - from) < buffer.nrSamples, 
                 "Requested to read block " << from << " to " << to << 
                 ", which results in " << (to - from) <<
                 " samples, but buffer is only " << buffer.nrSamples << 
                 " wide" );

      // wait for block start (but only in real-time mode)
      if (!buffer.sync) {
        LOG_DEBUG_STR("Waiting until " << (to + maxDelay) << " for " << from << " to " << to);
        waiter.waitUntil(to + maxDelay);
      }

      return new LockedBlock(*this, from, to, beamletOffsets);
    }

  }
}

