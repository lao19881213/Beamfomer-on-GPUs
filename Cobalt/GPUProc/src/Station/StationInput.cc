//# StationInput.cc: Routines to manage I/O from the stations.
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
//# $Id: StationInput.cc 26792 2013-10-01 09:09:43Z klijn $

#include <lofar_config.h>

#include "StationInput.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <boost/format.hpp>

#ifdef HAVE_MPI
#include <mpi.h>
#include <InputProc/Transpose/MPISendStation.h>
#include <InputProc/Transpose/MapUtil.h>
#endif

#include <Common/LofarLogger.h>
#include <Common/Thread/Semaphore.h>
#include <CoInterface/Parset.h>

#include <InputProc/SampleType.h>
#include <InputProc/Buffer/StationID.h>
#include <InputProc/Buffer/BufferSettings.h>
#include <InputProc/Buffer/BlockReader.h>
#include <InputProc/Buffer/SampleBuffer.h>
#include <InputProc/Station/PacketsToBuffer.h>
#include <InputProc/Delays/Delays.h>

#include "StationNodeAllocation.h"

using namespace LOFAR;
using namespace LOFAR::Cobalt;
using namespace std;
using boost::format;

namespace LOFAR {
  namespace Cobalt {

void receiveStation(const Parset &ps, const struct StationID &stationID, Semaphore &bufferReady, Semaphore &stopSignal)
{
  // settings for the circular buffer
  struct BufferSettings settings(stationID, false);
  settings.nrSamples_16bit = 5 * ps.nrSamplesPerSubband(); // Align with our block increment
  //settings.setBufferSize(5.0);

  // Remove lingering buffers
  removeSampleBuffers(settings);

  // fetch input streams
  StationNodeAllocation allocation(stationID, ps);
  vector< SmartPtr<Stream> > inputStreams(allocation.inputStreams());

  settings.nrBoards = inputStreams.size();

  // Force buffer reader/writer syncing if observation is non-real time
  SyncLock syncLock(settings, ps.settings.subbands.size());
  if (!ps.realTime()) {
    settings.sync = true;
    settings.syncLock = &syncLock;
  }

  // Set up the circular buffer
  GenericSampleBuffer buffer(settings, SharedMemoryArena::CREATE);

  // Set up the data transports
  MultiPacketsToBuffer station(settings, inputStreams);

  // Signal the creation of the SHM buffer
  bufferReady.up();

  #pragma omp parallel sections num_threads(2)
  {
    // Start a circular buffer
    #pragma omp section
    {
      LOG_INFO_STR("Starting circular buffer");
      station.process();
    }

    // Wait for parent to stop us (for cases that do not
    // trigger EOF in the inputStreams, such as reading from
    // UDP).
    #pragma omp section
    {
      stopSignal.down();
      station.stop();
    }
  }
}

template<typename SampleT> void sendInputToPipeline(const Parset &ps, size_t stationIdx, const SubbandDistribution &subbandDistribution)
{
  /*
   * Construct our stationID.
   */

  // fetch station name (e.g. CS001HBA0)
  const string fullFieldName = ps.settings.stations[stationIdx].name;

  // split into station name and antenna field name
  const string stationName = fullFieldName.substr(0,5); // CS001
  const string fieldName   = fullFieldName.substr(5);   // HBA0

  const struct StationID stationID(stationName, fieldName);

  StationNodeAllocation allocation(stationID, ps);

  if (!allocation.receivedHere()) {
    // Station is not sending from this node
    return;
  }

  LOG_INFO_STR("Processing data from station " << stationID);

  const TimeStamp from(ps.startTime() * ps.subbandBandwidth(), ps.clockSpeed());
  const TimeStamp to(ps.stopTime() * ps.subbandBandwidth(), ps.clockSpeed());

  const size_t nrBlocks = (to - from) / ps.nrSamplesPerSubband();

  // Don't run if there is no data to process
  if (nrBlocks == 0)
    return;

  Semaphore bufferReady;
  Semaphore stopSignal;

  /*
   * Stream the data.
   */
  #pragma omp parallel sections num_threads(2)
  {
    // Start a circular buffer
    #pragma omp section
    { 
      receiveStation(ps, stationID, bufferReady, stopSignal);
    }

    // Send data to receivers
    #pragma omp section
    {
      // Wait for SHM buffer to be created and initialised
      bufferReady.down();

      { // Make sure the SHM buffer is unused when we raise stopSignal

        // Fetch buffer settings from SHM.
        const struct BufferSettings settings(stationID, true);
        const struct BoardMode mode(ps.settings.nrBitsPerSample, ps.settings.clockMHz);

        LOG_INFO_STR("Detected " << settings);

        const TimeStamp from(ps.startTime() * ps.subbandBandwidth(), ps.clockSpeed());
        const TimeStamp to(ps.stopTime() * ps.subbandBandwidth(), ps.clockSpeed());

        // Determine the subband -> beamlet mapping for THIS station
        vector<size_t> beamlets(ps.nrSubbands());
        for( size_t i = 0; i < ps.nrSubbands(); ++i) {
          unsigned board = ps.settings.stations[stationIdx].rspBoardMap[i];
          unsigned slot  = ps.settings.stations[stationIdx].rspSlotMap[i];

          size_t beamlet = board * mode.nrBeamletsPerBoard() + slot;

          beamlets[i] = beamlet;
        }

#ifdef HAVE_MPI
        // All receiver ranks -- we have a dedicated thread for each one
        const vector<int> targetRanks(keys(subbandDistribution));

        // Send to all receivers in PARALLEL for higher performance
#       pragma omp parallel for num_threads(targetRanks.size())
        for(size_t i = 0; i < targetRanks.size(); ++i) {
          int rank = targetRanks.at(i);
          const vector<size_t> &targetSubbands(subbandDistribution.at(rank));

          // Do not set up a send engine to nodes that do not process data
          if (targetSubbands.empty())
            continue;

          /*
           * Set up the MPI send engine for this rank.
           */

          MPISendStation sender(settings, stationIdx, rank, targetSubbands);
#else
          int rank = -1;

          (void)subbandDistribution;
          (void)rank;

          vector<size_t> targetSubbands(ps.nrSubbands());
          for( size_t i = 0; i < targetSubbands.size(); ++i) {
            targetSubbands[i] = i;
          }
#endif

          /*
           * Set up circular buffer data reader.
           */

          // subband -> beamlet conversion for the reader
          vector<size_t> targetBeamlets(targetSubbands.size());
          for( size_t i = 0; i < targetSubbands.size(); ++i) {
            targetBeamlets[i] = beamlets[targetSubbands[i]];
          }

          BlockReader<SampleT> reader(settings, mode, targetBeamlets, 1.0);

          /*
           * Set up delay compensation.
           */
          Delays delays(ps, stationIdx, from, ps.nrSamplesPerSubband());
          delays.start();

          // We keep track of the delays at the beginning and end of each block.
          // After each block, we'll swap the afterEnd delays into atBegin.
          Delays::AllDelays delaySet1(ps), delaySet2(ps);
          Delays::AllDelays *delaysAtBegin  = &delaySet1;
          Delays::AllDelays *delaysAfterEnd = &delaySet2;

          // Get delays at begin of first block
          delays.getNextDelays(*delaysAtBegin);

          /*
           * Transfer all blocks.
           */

          vector<SubbandMetaData> metaDatas(targetSubbands.size());
          vector<ssize_t> read_offsets(targetSubbands.size());

          // Send data to Pipeline::receiveInput.
          //
          // Start with block -1, to allow the initialisation of historySamples,
          // etc for block 0.
          ssize_t block = -1;

          size_t blockSize = ps.nrSamplesPerSubband();

          for (TimeStamp current = from + block * blockSize; current + blockSize < to; current += blockSize, ++block) {
            LOG_DEBUG_STR(str(format("[rank %i block %d] Sending data from %s") % rank % block % stationID));

            // Fetch end delays (start delays are set by the previous block, or
            // before the loop).
            delays.getNextDelays(*delaysAfterEnd);

            // Compute the next set of metaData and read_offsets from the new
            // delays pair.
            delays.generateMetaData(*delaysAtBegin, *delaysAfterEnd, targetSubbands, metaDatas, read_offsets);

            //LOG_DEBUG_STR("Delays obtained");

            // Read the next block from the circular buffer.
            SmartPtr<struct BlockReader<SampleT>::LockedBlock> block(reader.block(current, current + blockSize, read_offsets));

            //LOG_INFO_STR("Block read");

#ifdef HAVE_MPI
            // Send the block to the receivers
            sender.sendBlock<SampleT>(*block, metaDatas);
#else
            DirectInput::instance().sendBlock<SampleT>(stationIdx, *block, metaDatas);
#endif

            //LOG_INFO_STR("Block sent");

            // Swap delay sets to accomplish delaysAtBegin = delaysAfterEnd
            swap(delaysAtBegin, delaysAfterEnd);
          }
#ifdef HAVE_MPI
        }
#endif
      }

      /*
       * The end.
       */
      stopSignal.up();
    }
  }
}

void sendInputToPipeline(const Parset &ps, size_t stationIdx, const SubbandDistribution &subbandDistribution)
{
  switch (ps.nrBitsPerSample()) {
    default:
    case 16: 
      sendInputToPipeline< SampleType<i16complex> >(ps, stationIdx, subbandDistribution);
      break;

    case 8: 
      sendInputToPipeline< SampleType<i8complex> >(ps, stationIdx, subbandDistribution);
      break;

    case 4: 
      sendInputToPipeline< SampleType<i4complex> >(ps, stationIdx, subbandDistribution);
      break;
  }
}

#ifndef HAVE_MPI
DirectInput &DirectInput::instance(const Parset *ps) {
  static DirectInput *theInstance = NULL;

  if (!theInstance) {
    ASSERT(ps != NULL);
    theInstance = new DirectInput(*ps);
  }

  return *theInstance;
}

DirectInput::DirectInput(const Parset &ps)
:
  ps(ps),
  stationDataQueues(boost::extents[ps.nrStations()][ps.nrSubbands()])
{
  // Create queues to forward station data

  for (size_t stat = 0; stat < ps.nrStations(); ++stat) {
    for (size_t sb = 0; sb < ps.nrSubbands(); ++sb) {
      stationDataQueues[stat][sb] = new BestEffortQueue< SmartPtr<struct InputBlock> >(1, ps.realTime());
    }
  }
}

template<typename T>
void DirectInput::sendBlock(unsigned stationIdx, const struct BlockReader<T>::LockedBlock &block, const vector<SubbandMetaData> &metaDatas)
{
  // Send the block to the stationDataQueues global object
  for (size_t subband = 0; subband < block.beamlets.size(); ++subband) {
    const struct Cobalt::Block<T>::Beamlet &beamlet = block.beamlets[subband];

    /* create new block */
    SmartPtr<struct InputBlock> pblock = new InputBlock;

    pblock->samples.resize(ps.nrSamplesPerSubband() * sizeof(T));

    /* copy metadata */
    pblock->metaData = metaDatas[subband];

    /* copy data */
    beamlet.copy(reinterpret_cast<T*>(&pblock->samples[0]));

    if (subband == 0) {
      LOG_DEBUG_STR("Flags at begin: " << beamlet.flagsAtBegin);
    }

    /* obtain flags (after reading the data!) */
    pblock->metaData.flags = beamlet.flagsAtBegin | block.flags(subband);

    /* send to pipeline */
    stationDataQueues[stationIdx][subband]->append(pblock);
  }
}

// Instantiate the required templates
template void DirectInput::sendBlock< SampleType<i16complex> >(unsigned stationIdx, const struct BlockReader< SampleType<i16complex> >::LockedBlock &block, const vector<SubbandMetaData> &metaDatas);
template void DirectInput::sendBlock< SampleType<i8complex> >(unsigned stationIdx, const struct BlockReader< SampleType<i8complex> >::LockedBlock &block, const vector<SubbandMetaData> &metaDatas);
template void DirectInput::sendBlock< SampleType<i4complex> >(unsigned stationIdx, const struct BlockReader< SampleType<i4complex> >::LockedBlock &block, const vector<SubbandMetaData> &metaDatas);

template<typename T>
void DirectInput::receiveBlock(std::vector<struct ReceiveStations::Block<T> > &blocks)
{
  for (size_t subbandIdx = 0; subbandIdx < ps.nrSubbands(); ++subbandIdx) {
    for (size_t stationIdx = 0; stationIdx < ps.nrStations(); ++stationIdx) {
      // Read all data directly
      SmartPtr<struct DirectInput::InputBlock> pblock = stationDataQueues[stationIdx][subbandIdx]->remove();

      // Copy data
      memcpy(blocks[stationIdx].beamlets[subbandIdx].samples, &pblock->samples[0], pblock->samples.size() * sizeof(pblock->samples[0]));

      // Copy meta data
      blocks[stationIdx].beamlets[subbandIdx].metaData = pblock->metaData;
    }
  }
}

// Instantiate the required templates
template void DirectInput::receiveBlock< SampleType<i16complex> >(std::vector<struct ReceiveStations::Block< SampleType<i16complex> > > &blocks);
template void DirectInput::receiveBlock< SampleType<i8complex> >(std::vector<struct ReceiveStations::Block< SampleType<i8complex> > > &blocks);
template void DirectInput::receiveBlock< SampleType<i4complex> >(std::vector<struct ReceiveStations::Block< SampleType<i4complex> > > &blocks);
#endif

  }
}

