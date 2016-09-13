/* tMPITransfer.cc
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
 * $Id: tMPITransfer.cc 26419 2013-09-09 11:19:56Z mol $
 */

#include <lofar_config.h>

#include <string>
#include <vector>
#include <map>
#include <omp.h>
#include <mpi.h>

#include <boost/format.hpp>

#include <Common/lofar_complex.h>
#include <Common/LofarLogger.h>
#include <Common/Exception.h>
#include <Common/Thread/Mutex.h>
#include <CoInterface/MultiDimArray.h>
#include <CoInterface/Stream.h>
#include <Stream/Stream.h>
#include <Stream/SocketStream.h>

#include <InputProc/RSPTimeStamp.h>
#include <InputProc/OMPThread.h>
#include <InputProc/SampleType.h>
#include <InputProc/Buffer/StationID.h>
#include <InputProc/Buffer/BufferSettings.h>
#include <InputProc/Buffer/BlockReader.h>
#include <InputProc/Station/PacketFactory.h>
#include <InputProc/Station/Generator.h>
#include <InputProc/Station/PacketsToBuffer.h>
#include <InputProc/Transpose/MPIReceiveStations.h>
#include <InputProc/Transpose/MPISendStation.h>
#include <InputProc/Transpose/MapUtil.h>

#include <map>
#include <vector>

#define DURATION 3
#define BLOCKSIZE 0.2
#define NRSTATIONS 3
#define NRBEAMLETS 4

using namespace LOFAR;
using namespace Cobalt;
using namespace std;
using boost::format;

Exception::TerminateHandler t(Exception::terminate);

const size_t clockMHz = 200;
const size_t clockHz = clockMHz * 1000 * 1000;
typedef SampleType<i16complex> SampleT;
TimeStamp from;
TimeStamp to;
const size_t blockSize = BLOCKSIZE * clockHz / 1024;
map<int, std::vector<size_t> > beamletDistribution;

SubbandMetaData metaData;

// Rank in MPI set of hosts
int rank;

// Number of MPI hosts
int nrHosts;

// Number of MPI hosts acting as stations.
int nrStations;

// Number of MPI hosts acting as receivers.
int nrReceivers;

void sender()
{
  struct StationID stationID(str(format("CS%03d") % rank), "LBA");
  struct BufferSettings settings(stationID, false);
  struct BoardMode mode(16, clockMHz);

  settings.nrBoards = 1;

  // sync readers and writers to prevent data loss
  // if the reader is delayed w.r.t. the generator

  SmartPtr<SyncLock> syncLock;

  syncLock = new SyncLock(settings, NRBEAMLETS);
  settings.syncLock = syncLock;
  settings.sync = true;

  omp_set_nested(true);
  //omp_set_num_threads(32);
  OMPThread::init();

  // Transfer of packets from generator to buffer
  std::vector< SmartPtr<Stream> > inputStreams(settings.nrBoards);
  std::vector< SmartPtr<Stream> > outputStreams(settings.nrBoards);
  for (size_t i = 0; i < inputStreams.size(); ++i) {
    const string desc = str(format("tcp:127.0.0.%d:%u") % (rank + 1) % (4346+i));


    #pragma omp parallel sections
    {
      #pragma omp section
      inputStreams[i] = createStream(desc, true);

      #pragma omp section
      outputStreams[i] = createStream(desc, false);
    }
  }

  removeSampleBuffers(settings);

  MultiPacketsToBuffer station( settings, inputStreams );
  PacketFactory factory( mode );
  Generator generator( settings, outputStreams, factory, from, to );

  #pragma omp parallel sections
  {
    // Generate the data to send
    #pragma omp section
    generator.process();

    // Start a circular buffer
    #pragma omp section
    station.process();

    // Send data to receivers
    #pragma omp section
    {
      struct BufferSettings s(stationID, true);

      LOG_INFO_STR("[" << stationID << "] Detected: " << s);
      LOG_INFO_STR("[" << stationID << "] Connecting to receivers to send " << from << " to " << to);

      vector<int> targetRanks = keys(beamletDistribution);

      #pragma omp parallel for num_threads(targetRanks.size())
      for (size_t i = 0; i < targetRanks.size(); ++i) {
        int targetRank = targetRanks[i];

        LOG_INFO_STR("[" << stationID << "] Sending to receiver " << i << " at rank " << targetRank);

        vector<size_t> &subbands = beamletDistribution.at(targetRank);
        MPISendStation sender(s, rank, targetRank, subbands);
        BlockReader<SampleT> reader(s, mode, subbands, 0.1);

        size_t blockNr = 0;
        for (TimeStamp current = from; current + blockSize < to; current += blockSize, ++blockNr) {
          LOG_INFO_STR("[" << stationID << " to rank " << targetRank << "] Reading block " << blockNr);

          SmartPtr<struct BlockReader<SampleT>::LockedBlock> block(reader.block(current, current + blockSize, std::vector<ssize_t>(subbands.size(), 0)));

          LOG_INFO_STR("[" << stationID << " to rank " << targetRank << "] Sending block " << blockNr);
          std::vector<SubbandMetaData> metaDatas(block->beamlets.size(), metaData);
          sender.sendBlock<SampleT>(*block, metaDatas);
        }
      }
      LOG_INFO_STR("[" << stationID << "] Done");

      generator.stop();
      station.stop();
    }
  }
}

void receiver()
{
  const std::vector<size_t> &beamlets = beamletDistribution[rank];
  const size_t nrBeamlets = beamlets.size();

  LOG_INFO_STR("Receiver node " << rank << " starts, handling " << beamlets.size() << " subbands from " << nrStations << " stations." );
  LOG_INFO_STR("Connecting to senders to receive " << from << " to " << to);

  MPIReceiveStations receiver(nrStations, beamlets, blockSize);

  // create space for the samples
  MultiDimArray<SampleT, 3> samples(boost::extents[nrStations][nrBeamlets][blockSize]);

  // create blocks -- they all have to be the right size already
  std::vector< struct MPIReceiveStations::Block<SampleT> > blocks(nrStations);

  for (int s = 0; s < nrStations; ++s) {
    blocks[s].beamlets.resize(nrBeamlets);

    for (size_t b = 0; b < nrBeamlets; ++b) {
      blocks[s].beamlets[b].samples = &samples[s][b][0];
    }
  }

  // start transfer
  size_t blockIdx = 0;

  for(TimeStamp current = from; current + blockSize < to; current += blockSize) {
    receiver.receiveBlock<SampleT>(blocks);

    // validate meta data
    for (int s = 0; s < nrStations; ++s) {
      for (size_t b = 0; b < nrBeamlets; ++b) {
        ASSERTSTR(blocks[s].beamlets[b].metaData.flags == metaData.flags, "Got flags " << blocks[s].beamlets[b].metaData.flags << " but expected flags " << metaData.flags);
      }
    }

    // calculate flagging average
    const size_t nrSamples = nrStations * nrBeamlets * blockSize;
    size_t nrFlaggedSamples = 0;

    for (int s = 0; s < nrStations; ++s) {
      for (size_t b = 0; b < nrBeamlets; ++b) {
        nrFlaggedSamples = blocks[s].beamlets[b].metaData.flags.count();
      }
    }

    float flagPerc = 100.0f * nrFlaggedSamples / nrSamples;

    LOG_INFO_STR("Receiver " << rank << " received block " << blockIdx << " flags: " << flagPerc << "%" );
    ++blockIdx;
  }

  LOG_INFO_STR("Receiver " << rank << " done");
}

int main( int argc, char **argv )
{
  INIT_LOGGER( "tMPITransfer" );

  // Prevent stalling.
  alarm(30);

  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    LOG_ERROR_STR("MPI_Init failed");
    return 1;
  }

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nrHosts);

  // Need at least one sender and one receiver
  ASSERT( nrHosts >= 2 );

  // agree on the start time
  time_t now = time(0L);
  MPI_Bcast(&now, sizeof now, MPI_CHAR, 0, MPI_COMM_WORLD);
  from = TimeStamp(now + 5, 0, clockHz);
  to   = TimeStamp(now + 5 + DURATION, 0, clockHz);

  // Use half of the nodes as stations
  nrStations = nrHosts/2;
  nrReceivers = nrHosts - nrStations;

  // Divide the subbands over the receivers
  // Make sure not to use the identity list to detect
  // mixups of stationBeamlet and beamletIdx.
  for (unsigned i = 0; i < NRBEAMLETS; ++i) {
    unsigned stationBeamlet = NRBEAMLETS - i - 1;
    int receiverNr = i % nrReceivers;

    beamletDistribution[nrStations + receiverNr].push_back(stationBeamlet);
  }

  if (rank < nrStations) {
    sender();
  } else {
    receiver();
  }

  MPI_Finalize();
}

