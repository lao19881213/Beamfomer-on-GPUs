#include "SampleBuffer.h"

#include <string>
#include <vector>
#include <boost/format.hpp>

#include <Common/LofarLogger.h>
#include "BufferSettings.h"
#include "SharedMemory.h"
#include "Ranges.h"

#define DEBUG_SYNCLOCK(s)
//#define DEBUG_SYNCLOCK(s) LOG_DEBUG_STR(s)

namespace LOFAR
{
  namespace Cobalt
  {
    template <typename T>
    size_t SampleBuffer<T>::dataSize( const struct BufferSettings &settings )
    {
      return // header
             sizeof settings
             // flags (aligned to ALIGNMENT)
             + settings.nrBoards * (Ranges::size(settings.nrAvailableRanges) + ALIGNMENT)
             // beamlets (aligned to ALIGNMENT)
             + settings.nrBoards * (BoardMode::nrBeamletsPerBoard(T::bitMode()) * settings.nrSamples(T::bitMode()) * sizeof(T) + ALIGNMENT)
             // mode (aligned to ALIGNMENT)
             + settings.nrBoards * (sizeof(struct BoardMode) + ALIGNMENT);
    }


    template<typename T>
    SampleBuffer<T>::SampleBuffer( const struct BufferSettings &_settings, SharedMemoryArena::Mode shmMode )
      :
      logPrefix(str(boost::format("[station %s %s board] [SampleBuffer] ") % _settings.station.stationName % _settings.station.antennaField)),
      data(_settings.dataKey, dataSize(_settings), shmMode),
      allocator(data),
      create(shmMode == SharedMemoryArena::CREATE || shmMode == SharedMemoryArena::CREATE_EXCL),
      settings(initSettings(_settings, create)),
      sync(settings->sync),
      syncLock(settings->syncLock),

      nrSamples(settings->nrSamples(T::bitMode())),
      nrBoards(settings->nrBoards),
      nrAvailableRanges(settings->nrAvailableRanges),

      beamlets(boost::extents[nrBoards * BoardMode::nrBeamletsPerBoard(T::bitMode())][nrSamples], ALIGNMENT, allocator, true, false),
      boards(nrBoards,Board(*this))
    {
      // Check if non-realtime mode is set up correctly
      if (sync) {
        const BoardMode mode(T::bitMode());

        ASSERTSTR(syncLock, "Synced buffer requires syncLock object");
        ASSERTSTR(syncLock->writeLock.size() == nrBoards, "SampleBuffer has " << nrBoards << " RSP boards, but syncLock expects " << syncLock->writeLock.size() << " boards");
      }

      for (size_t b = 0; b < boards.size(); b++) {
        size_t numBytes = Ranges::size(nrAvailableRanges);

        boards[b].available = Ranges(static_cast<int64*>(allocator.allocate(numBytes, ALIGNMENT)), numBytes, nrSamples, create);
        boards[b].boardNr = b;
        boards[b].mode = allocator.allocateTyped(ALIGNMENT);
      }

      LOG_DEBUG_STR( logPrefix << "Initialised" );
    }
    

    template<typename T>
    struct BufferSettings *SampleBuffer<T>::initSettings( const struct BufferSettings &localSettings, bool create )
    {
      // settings are ALWAYS at the start of the buffer, regardless of alignment!
      struct BufferSettings *sharedSettings = allocator.allocateTyped(1);

      if (create) {
        // register settings
        LOG_INFO_STR( logPrefix << "Registering " << localSettings.station );
        *sharedSettings = localSettings;
      } else {
        // verify settings
        ASSERT( *sharedSettings == localSettings );
        LOG_INFO_STR( logPrefix << "Connected to " << localSettings.station );
      }

      return sharedSettings;
    }


    template<typename T>
    SampleBuffer<T>::Board::Board( SampleBuffer<T> &buffer, size_t boardNr )
      :
      mode(0),
      boardNr(boardNr),
      buffer(buffer)
    {
    }


    template<typename T>
    void SampleBuffer<T>::Board::changeMode( const struct BoardMode &mode )
    {
      // only act if something changes
      if (mode == *this->mode)
        return;

      // invalidate all data
      available.clear();

      // set the new mode
      *this->mode = mode;
    }


    template<typename T>
    double SampleBuffer<T>::Board::flagPercentage( const TimeStamp &from, const TimeStamp &to ) const
    {
      // count available samples in the given range
      const size_t nrSamples = available.sparseSet(from, to).count();

      // return percentage of samples NOT available in the given range
      return 100.0 - (100.0 * nrSamples / (to - from));
    }


    template<typename T>
    void SampleBuffer<T>::noReadBefore( size_t beamlet, const TimeStamp &epoch )
    {
      if (sync) {
        // Free up read intent up until `epoch'.
        DEBUG_SYNCLOCK("noReadBefore(" << beamlet << ", " << epoch << ")");

        ASSERT(beamlet < syncLock->readLock.size());
        syncLock->readLock[beamlet].advanceTo(epoch);
      }
    }


    template<typename T>
    void SampleBuffer<T>::startRead( size_t beamlet, const TimeStamp &begin, const TimeStamp &end )
    {
      /*
       * Note: callers might want to free up the read intent up until `begin':
       *  noReadBefore(beamlet, begin);
       */
      (void)begin;

      if (sync) {
        ASSERT(beamlet < syncLock->readLock.size());

        // Wait for writer to finish writing until `end'.
        const BoardMode mode(T::bitMode());
        size_t boardNr = mode.boardIndex(beamlet);

        DEBUG_SYNCLOCK("startRead(" << beamlet << ", " << begin << ", " << end << "): waits on board " << boardNr);

        syncLock->writeLock[boardNr].waitFor(end);

        DEBUG_SYNCLOCK("startRead(" << beamlet << ", " << begin << ", " << end << "): reading from board " << boardNr);
      }
    }


    template<typename T>
    void SampleBuffer<T>::stopRead( size_t beamlet, const TimeStamp &end )
    {
      if (sync) {
        DEBUG_SYNCLOCK("stopRead(" << beamlet << ", " << end << ")");
      }

      // Signal we're done reading
      noReadBefore(beamlet, end);
    }


    template<typename T>
    void SampleBuffer<T>::noMoreReading( size_t beamlet )
    {
      DEBUG_SYNCLOCK("noMoreReading(" << beamlet << ")");

      // Signal we're done reading

      // Put the readPtr into the far future.
      // We only use this TimeStamp for comparison so clockSpeed does not matter.
      noReadBefore(beamlet, TimeStamp(0xFFFFFFFFFFFFFFFFULL));
    }


    template<typename T>
    void SampleBuffer<T>::Board::startWrite( const TimeStamp &begin, const TimeStamp &end )
    {
      ASSERT((uint64)end > buffer.nrSamples);

      if (buffer.sync) {
        DEBUG_SYNCLOCK("[board " << boardNr << "] startWrite(" << begin << ", " << end << ")");

        // Signal write intent, to let reader know we don't have data older than
        // this.
        buffer.syncLock->writeLock[boardNr].advanceTo(begin);

        ASSERT(this->mode);

        const size_t nrBeamletsPerBoard = this->mode->nrBeamletsPerBoard();
        const size_t nrBeamlets = buffer.syncLock->readLock.size();

        // Wait for readers to finish what we're about to overwrite
        for (size_t i = 0; i < nrBeamletsPerBoard; ++i) {
          size_t beamlet = boardNr * nrBeamletsPerBoard + i;

          if (beamlet >= nrBeamlets)
            break;

          DEBUG_SYNCLOCK("[board " << boardNr << "] startWrite: waiting for readLock on beamlet " << beamlet << " to reach " << (end - buffer.nrSamples));
          buffer.syncLock->readLock[beamlet].waitFor(end - buffer.nrSamples);
          DEBUG_SYNCLOCK("[board " << boardNr << "] startWrite: passed readLock on beamlet " << beamlet);
        }
      }

      // Mark overwritten range (and everything before it to prevent a mix) as invalid
      available.excludeBefore(end - buffer.nrSamples);
    }


    template<typename T>
    void SampleBuffer<T>::Board::stopWrite( const TimeStamp &end )
    {
      if (buffer.sync) {
        DEBUG_SYNCLOCK("[board " << boardNr << "] stopWrite(" << end << ")");

        // Signal we're done writing
        buffer.syncLock->writeLock[boardNr].advanceTo(end);
      }
    }


    template<typename T>
    void SampleBuffer<T>::Board::noMoreWriting()
    {
      LOG_DEBUG_STR("[board " << boardNr << "] noMoreWriting()");

      if (buffer.sync) {
        // Signal we're done writing

        // Put the writePtr into the far future.
        // We only use this TimeStamp for comparison so clockSpeed does not matter.
        buffer.syncLock->writeLock[boardNr].advanceTo(TimeStamp(0xFFFFFFFFFFFFFFFFULL));
      }
    }
  }
}

