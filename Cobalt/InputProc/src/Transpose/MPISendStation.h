/* MPISendStation.h
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
 * $Id: MPISendStation.h 26336 2013-09-03 10:01:41Z mol $
 */

#ifndef LOFAR_INPUT_PROC_MPI_SEND_STATIONS_H
#define LOFAR_INPUT_PROC_MPI_SEND_STATIONS_H

#include <mpi.h>

#include <Common/LofarTypes.h>
#include <CoInterface/MultiDimArray.h>
#include <CoInterface/SubbandMetaData.h>
#include <CoInterface/SmartPtr.h>

#include <InputProc/RSPTimeStamp.h>
#include <InputProc/Buffer/Block.h>
#include <InputProc/Buffer/BufferSettings.h>
#include "MPIProtocol.h"
#include "MPIUtil.h"

#include <map>
#include <set>
#include <vector>

namespace LOFAR
{
  namespace Cobalt
  {

    /*
     * Sends a Block of beamlets to all receiving MPI nodes.
     * Blocks are sent in a sequential fashion: a block must be received
     * completely before the next one is sent. Performance is barely affected,
     * because the output to all nodes has to go through a shared pipe (IB
     * port). We thus cannot really start sending to the first node earlier if
     * we're not yet done sending to other nodes.
     */
    class MPISendStation
    {
    public:
      // Create a sender of blocks over MPI.
      //
      // settings
      //   The station to send info from.
      //
      // stationIdx
      //   The station index within this observation.
      // beamletDistribution
      //   The distribution of beamlets:
      //     key   = receiver MPI rank
      //     value = beamlets to send in [0, ps.nrSubbands())
      MPISendStation( const struct BufferSettings &settings, size_t stationIdx, int targetRank, const std::vector<size_t> &beamlets );

      ~MPISendStation();

      // Send one block. The caller is responsible for matching the number of
      // posted receiveBlocks.
      template<typename T>
      void sendBlock( const struct Block<T> &block, std::vector<SubbandMetaData> &metaData );

      // Cache for the header to send
      SmartPtr<MPIProtocol::Header, SmartPtrMPI<MPIProtocol::Header> > header;

      // Cache for meta data to send
      SmartPtr<MPIProtocol::MetaData, SmartPtrMPI<MPIProtocol::MetaData> > metaDatas;

    private:
      const std::string logPrefix;
      const BufferSettings &settings;

      // Station number in observation [0..nrStations)
      const size_t stationIdx;

      // Rank to send data to
      const int targetRank;

      // Which beamlets to send
      const std::vector<size_t> beamlets;

    public:
      // Construct and send a header to the given rank (async).
      template<typename T>
      MPI_Request sendHeader( const struct Block<T> &block );

      // Send beamlet data (in 1 or 2 transfers) to the given rank (async).
      // Returns the number of MPI_Requests made.
      template<typename T>
      unsigned sendData( unsigned beamlet, const struct Block<T>::Beamlet &ib, MPI_Request requests[2] );

      // Send flags data to the given rank (async).
      MPI_Request sendMetaData( unsigned beamlet, const struct MPIProtocol::MetaData &metaData );
    };

  }
}

#endif

