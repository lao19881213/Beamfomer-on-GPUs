/* MPISendStation.cc
 * Copyright (C) 2013  ASTRON (Netherlands Institute for Radio Astronomy)
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
 * $Id: MPISendStation.cc 26336 2013-09-03 10:01:41Z mol $
 */

#include <lofar_config.h>
#include "MPISendStation.h"
#include "MapUtil.h"
#include "MPIUtil.h"

#include <InputProc/SampleType.h>

#include <Common/LofarLogger.h>
#include <CoInterface/PrintVector.h>

#include <boost/format.hpp>

using namespace std;
using namespace LOFAR::Cobalt::MPIProtocol;

//#define DEBUG_MPI

#ifdef DEBUG_MPI
//#define DEBUG(str)    LOG_DEBUG_STR(__PRETTY_FUNCTION__ << ": " << str)
#define DEBUG(str)    LOG_DEBUG_STR(str)
#else
#define DEBUG(str)
#endif

namespace LOFAR {

  namespace Cobalt {

    MPISendStation::MPISendStation( const struct BufferSettings &settings, size_t stationIdx, int targetRank, const std::vector<size_t> &beamlets )
    :
      logPrefix(str(boost::format("[station %s] [MPISendStation] ") % settings.station.stationName)),
      settings(settings),
      stationIdx(stationIdx),
      targetRank(targetRank),
      beamlets(beamlets)
    {
      LOG_DEBUG_STR(logPrefix << "Initialised");

      // Check whether we send each subband to at most one node
      ASSERT(!beamlets.empty());

      int sourceRank = MPI_Rank();

      // Allocate MPI memory
      header = mpiAllocator.allocateTyped();
      metaDatas = static_cast<MPIProtocol::MetaData *>(mpiAllocator.allocate(beamlets.size() * sizeof *metaDatas));

      // Set the static header info
      header->station      = settings.station;
      header->sourceRank   = sourceRank;

      // Set beamlet info
      header->nrBeamlets = beamlets.size();
      ASSERT(header->nrBeamlets < sizeof header->beamlets / sizeof header->beamlets[0]);

      std::copy(beamlets.begin(), beamlets.end(), &header->beamlets[0]);
    }


    MPISendStation::~MPISendStation()
    {
    }


    template<typename T>
    MPI_Request MPISendStation::sendHeader( const struct Block<T> &block )
    {
      DEBUG(logPrefix << "Sending header to rank " << targetRank);

      // Copy dynamic header info
      header->from             = block.from;
      header->to               = block.to;

      // Copy the beam-specific data
      ASSERT(header->nrBeamlets <= sizeof header->wrapOffsets / sizeof header->wrapOffsets[0]);

      for(size_t i = 0; i < header->nrBeamlets; ++i) {
        const struct Block<T>::Beamlet &ib = block.beamlets[i];

        header->wrapOffsets[i] = ib.nrRanges == 1 ? 0 : ib.ranges[0].to - ib.ranges[0].from;
      }

      // Send the actual header
      union tag_t tag;
      tag.bits.type     = CONTROL;
      tag.bits.station  = stationIdx;

      return Guarded_MPI_Isend(header, sizeof *header, targetRank, tag.value);
    }


    template<typename T>
    unsigned MPISendStation::sendData( unsigned beamlet, const struct Block<T>::Beamlet &ib, MPI_Request requests[2] )
    {
      DEBUG(logPrefix << "Sending beamlet " << beamlet << " to rank " << targetRank << " using " << ib.nrRanges << " transfers");

      // Send beamlet using 1 or 2 transfers
      for(unsigned transfer = 0; transfer < ib.nrRanges; ++transfer) {
        union tag_t tag;

        tag.bits.type     = BEAMLET;
        tag.bits.station  = stationIdx;
        tag.bits.beamlet  = beamlet;
        tag.bits.transfer = transfer;

        const T *from = ib.ranges[transfer].from;
        const T *to   = ib.ranges[transfer].to;

        ASSERT( from < to ); // There must be data to send, or MPI will error

        requests[transfer] = Guarded_MPI_Issend((void*)from, (to - from) * sizeof(T), targetRank, tag.value);
      }

      return ib.nrRanges;
    }


    MPI_Request MPISendStation::sendMetaData( unsigned beamlet, const struct MPIProtocol::MetaData &metaData )
    {
      DEBUG("Sending flags to rank " << targetRank);

      union tag_t tag;
      tag.bits.type     = METADATA;
      tag.bits.station  = stationIdx;
      tag.bits.beamlet  = beamlet;

      // Flags are sent if the data have been transferred,
      // and the flags Irecv is posted before the data Irecvs,
      // so we are sure that the flag Irecv is posted.
      return Guarded_MPI_Irsend(&metaData, sizeof metaData, targetRank, tag.value);
    }


    template<typename T>
    void MPISendStation::sendBlock( const struct Block<T> &block, std::vector<SubbandMetaData> &metaData )
    {
      DEBUG("entry");

      ASSERT(metaData.size() == block.beamlets.size());

      /*
       * SEND HEADER
       */

      {
        ScopedLock sl(MPIMutex);

        MPI_Request headerRequest;
     
        headerRequest = sendHeader<T>(block);

        // If other requests are received, the header will be as well
        freeRequest(headerRequest);
      }

      /*
       * SEND BEAMLETS
       */

      std::vector<MPI_Request> beamletRequests(block.beamlets.size() * 2, MPI_REQUEST_NULL); // [beamlet][transfer]
      std::vector<size_t> beamletNrs(block.beamlets.size() * 2, 0);
      std::vector<size_t> transferNrs(block.beamlets.size() * 2, 0);
      size_t nrBeamletRequests = 0;

      {
        ScopedLock sl(MPIMutex);

        for(size_t b = 0; b < beamlets.size(); ++b) {
          const size_t globalBeamletIdx = beamlets[b];

          // Send beamlet
          const struct Block<T>::Beamlet &ib = block.beamlets[b];

          MPI_Request requests[2];

          size_t nrTransfers = sendData<T>(globalBeamletIdx, ib, requests);

          for (size_t i = 0; i < nrTransfers; ++i) {
            beamletNrs[nrBeamletRequests] = b;
            transferNrs[nrBeamletRequests] = i;
            beamletRequests[nrBeamletRequests] = requests[i];

            nrBeamletRequests++;
          }
        }
      }

      // Cut off beamletRequests
      ASSERT(nrBeamletRequests <= beamletRequests.size());
      beamletRequests.resize(nrBeamletRequests);

      /*
       * SEND METADATA
       */

      std::vector<MPI_Request> metaDataRequests(block.beamlets.size(), MPI_REQUEST_NULL);

      for(size_t b = 0; b < nrBeamletRequests; ) { // inner loop increments b, once per request
#if 0
        // waitSome is bugged for an unknown reason, that is, using
        // this causes rtcp to freeze.
        vector<int> finishedRequests = waitSome(beamletRequests);
#else
        vector<int> finishedRequests(1, waitAny(beamletRequests));
#endif

        ASSERT(!finishedRequests.empty());

        for(size_t f = 0; f < finishedRequests.size(); ++f, ++b) {
          const int sendIdx              = finishedRequests[f];

          ASSERT(sendIdx >= 0);
          ASSERT((size_t)sendIdx < beamletRequests.size());
          ASSERT(beamletRequests[sendIdx] == MPI_REQUEST_NULL);

          //const size_t globalBeamletIdx  = sendIdx / 2;
          //const size_t transfer          = sendIdx % 2;
          const size_t beamletIdx  = beamletNrs[sendIdx];
          const size_t transfer    = transferNrs[sendIdx];

          const struct Block<T>::Beamlet &ib = block.beamlets[beamletIdx];

          // waitSome sets finished requests to MPI_REQUEST_NULL in our array.
          if (ib.nrRanges == 1 || beamletRequests[sendIdx + (transfer == 0 ? 1 : -1)] == MPI_REQUEST_NULL) {
            /*
             * SEND FLAGS FOR BEAMLET
             */

            const size_t globalBeamletIdx = beamlets[beamletIdx];

            /*
             * OBTAIN FLAGS AFTER DATA IS SENT
             */

            // The only valid samples are those that existed both
            // before and after the transfer.

            SubbandMetaData &md = metaData[beamletIdx];
            md.flags = block.beamlets[beamletIdx].flagsAtBegin | block.flags(beamletIdx);

            struct MetaData &buffer = metaDatas[beamletIdx];
            buffer = md;

            /*
             * SEND FLAGS
             */
            {
              ScopedLock sl(MPIMutex);
              metaDataRequests[beamletIdx] = sendMetaData(globalBeamletIdx, buffer);
            }
          }
        }
      }

      /*
       * WRAP UP ASYNC SENDS
       */

      // Wait on all pending requests
      waitAll(metaDataRequests);

      DEBUG("exit");
    }

    // Create all necessary instantiations
#define INSTANTIATE(T) \
      template MPI_Request MPISendStation::sendHeader<T>( const struct Block<T> &block ); \
      template unsigned MPISendStation::sendData<T>( unsigned beamlet, const struct Block<T>::Beamlet &ib, MPI_Request requests[2] ); \
      template void MPISendStation::sendBlock<T>( const struct Block<T> &block, std::vector<SubbandMetaData> &metaData );

    INSTANTIATE(SampleType<i4complex>);
    INSTANTIATE(SampleType<i8complex>);
    INSTANTIATE(SampleType<i16complex>);

  }
}

