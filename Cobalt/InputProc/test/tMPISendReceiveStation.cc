/* tMPISendReceiveStation.cc
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
 * $Id: tMPISendReceiveStation.cc 26336 2013-09-03 10:01:41Z mol $
 */

#include <lofar_config.h>

#include <string>
#include <vector>
#include <map>
#include <mpi.h>

#include <boost/format.hpp>

#include <Common/lofar_complex.h>
#include <Common/LofarLogger.h>
#include <CoInterface/SmartPtr.h>

#include <InputProc/SampleType.h>
#include <InputProc/Transpose/MPIReceiveStations.h>
#include <InputProc/Transpose/MPISendStation.h>
#include <InputProc/Transpose/MPIUtil.h>

#include <UnitTest++.h>

#include <map>
#include <vector>

using namespace LOFAR;
using namespace Cobalt;
using namespace std;

using boost::format;

#define NRBEAMLETS 3


typedef SampleType<i16complex> SampleT;
std::vector<char> metaDataBlob(100, 42);

// Rank in MPI set of hosts
int rank;

// Number of MPI hosts
int nrHosts;

// Common transfer objects
std::vector<size_t> beamletDistribution;
SmartPtr<MPISendStation> sender;
SmartPtr<MPIReceiveStations> receiver;

// Create structures for data tests
const size_t blockSize = 1024;

vector<SampleT> data_in(blockSize);


TEST(Header) {
  LOG_INFO_STR("Header");

  struct Block<SampleT> block;

  block.from = TimeStamp(1, 2);
  block.to   = TimeStamp(3, 4);

  for (size_t b = 0; b < NRBEAMLETS; ++b) {
    struct Block<SampleT>::Beamlet ib;

    ib.stationBeamlet = b;
    ib.ranges[0].from = &data_in[b];
    ib.ranges[0].to   = &data_in[blockSize/2];
    ib.ranges[1].from = &data_in[blockSize/2];
    ib.ranges[1].to   = &data_in[blockSize];
    ib.nrRanges       = 2;
    ib.offset         = b;

    block.beamlets.push_back(ib);
  }

  MPIProtocol::Header header_in;
  MPIProtocol::Header header_out;

  // NOTE: Headers are already partially filled by MPISendStation constructor,
  // so we need to do some filling as well
  header_in.nrBeamlets = block.beamlets.size();
  for (size_t b = 0; b < block.beamlets.size(); ++b) {
    header_in.beamlets[b] = b;
  }

  // Post requests
  vector<MPI_Request> requests(1);
 
  if (rank == 0) {
    *(sender->header) = header_in;

    requests[0] = sender->sendHeader<SampleT>(block);
  } else {
    requests[0] = receiver->receiveHeader(0, header_out);
  }

  // Wait for results
  waitAll(requests);

  if (rank == 1) {
    // Validate results
    CHECK_EQUAL((int64)block.from,       header_out.from);
    CHECK_EQUAL((int64)block.to,         header_out.to);
    CHECK_EQUAL(block.beamlets.size(),   header_out.nrBeamlets);

    for (size_t b = 0; b < block.beamlets.size(); ++b) {
      CHECK_EQUAL(blockSize/2 - b, header_out.wrapOffsets[b]);
    }
  }
}


TEST(Data_OneTransfer) {
  LOG_INFO_STR("Data_OneTransfer");

  struct Block<SampleT>::Beamlet ib;
  vector<SampleT> data_out(blockSize);

  ib.stationBeamlet = 0;
  ib.ranges[0].from = &data_in[0];
  ib.ranges[0].to   = &data_in[blockSize];
  ib.nrRanges       = 1;
  ib.offset         = 0;

  // Post requests
  vector<MPI_Request> requests(1);

  if (rank == 0) {
    unsigned nrTransfers = sender->sendData<SampleT>(42, ib, &requests[0]);
    CHECK_EQUAL(1U, nrTransfers);
  } else {
    requests[0] = receiver->receiveData<SampleT>(0, 42, 0, &data_out[0], data_out.size());
  }

  // Wait for results
  waitAll(requests);

  if (rank == 1) {
    // Validate results
    CHECK(data_in == data_out);
  }
}


TEST(Data_TwoTransfers) {
  LOG_INFO_STR("Data_TwoTransfers");

  struct Block<SampleT>::Beamlet ib;
  vector<SampleT> data_out(blockSize);

  ib.stationBeamlet = 0;
  ib.ranges[0].from = &data_in[0];
  ib.ranges[0].to   = &data_in[blockSize/2];
  ib.ranges[1].from = &data_in[blockSize/2];
  ib.ranges[1].to   = &data_in[blockSize];
  ib.nrRanges       = 2;
  ib.offset         = 0;

  // Post requests
  vector<MPI_Request> requests(2);
 
  if (rank == 0) {
    unsigned nrTransfers = sender->sendData<SampleT>(42, ib, &requests[0]);
    CHECK_EQUAL(2U, nrTransfers);
  } else {
    requests[0] = receiver->receiveData<SampleT>(0, 42, 0, &data_out[0], blockSize/2);
    requests[1] = receiver->receiveData<SampleT>(0, 42, 1, &data_out[blockSize/2], blockSize/2);
  }

  // Wait for results
  waitAll(requests);

  if (rank == 1) {
    // Validate results
    CHECK(data_in == data_out);
  }
}


TEST(MetaData) {
  LOG_INFO_STR("MetaData");

  // Create structures for input and output
  SubbandMetaData metaData_in;
  SubbandMetaData metaData_out;

  MPIProtocol::MetaData mdBuf_in;
  MPIProtocol::MetaData mdBuf_out;

  // Fill input
  metaData_in.flags.include(10, 20);
  metaData_in.flags.include(30, 40);
  mdBuf_in = metaData_in;

  // Post requests
  vector<MPI_Request> requests(1);
 
  if (rank == 0) {
    requests[0] = sender->sendMetaData(42, mdBuf_in);
  } else {
    requests[0] = receiver->receiveMetaData(0, 42, mdBuf_out);
  }

  // Wait for results
  waitAll(requests);

  if (rank == 1) {
    // Process results
    metaData_out = mdBuf_out;

    // Validate results
    CHECK_EQUAL(metaData_in.flags, metaData_out.flags);
  }
}


TEST(Block_OneStation) {
  LOG_INFO_STR("Block_OneStation");

  struct Block<SampleT> block_in;
  SubbandMetaData metaData_in;

  vector< struct MPIReceiveStations::Block<SampleT> > blocks_out;
  SubbandMetaData metaData_out;

  size_t nrStations = 1;

  // Fill input
  block_in.from = TimeStamp(1, 2);
  block_in.to   = TimeStamp(3, 4);

  for (size_t b = 0; b < NRBEAMLETS; ++b) {
    struct Block<SampleT>::Beamlet ib;

    ib.stationBeamlet = b;

    ib.ranges[0].from = &data_in[0];
    ib.ranges[0].to   = &data_in[blockSize/2];
    ib.ranges[1].from = &data_in[blockSize/2];
    ib.ranges[1].to   = &data_in[blockSize];
    ib.nrRanges       = 2;
    ib.offset         = 0;

    ib.flagsAtBegin.include(10, 20);
    ib.flagsAtBegin.include(30, 40);

    block_in.beamlets.push_back(ib);
  }

  // create a place to store the samples
  MultiDimArray<SampleT, 3> samples_out(boost::extents[nrStations][NRBEAMLETS][blockSize]);

  // create blocks_out -- they all have to be the right size already
  blocks_out.resize(nrStations);

  for (size_t s = 0; s < nrStations; ++s) {
    blocks_out[s].beamlets.resize(NRBEAMLETS);

    for (size_t b = 0; b < NRBEAMLETS; ++b) {
      blocks_out[s].beamlets[b].samples = &samples_out[s][b][0];
    }
  }

  if (rank == 0) {
    // sender
    std::vector<SubbandMetaData> metaDatas(NRBEAMLETS, metaData_in);
    sender->sendBlock<SampleT>(block_in, metaDatas);
  } else {
    // receiver
    receiver->receiveBlock<SampleT>(blocks_out);
  }

  LOG_INFO_STR("Block transferred");

  if (rank == 1) {
    // Validate results
    for (size_t b = 0; b < NRBEAMLETS; ++b) {
      for (size_t t = 0; t < blockSize; ++t) {
        CHECK_EQUAL(data_in[t], samples_out[0][b][t]);
      }

      CHECK_EQUAL(block_in.beamlets[b].flagsAtBegin, blocks_out[0].beamlets[b].metaData.flags);
    }
  }
}


int main( int argc, char **argv )
{
  INIT_LOGGER( "tMPISendReceiveStation" );

  // Prevent stalling.
  alarm(30);

  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    LOG_ERROR_STR("MPI_Init failed");
    return 1;
  }

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nrHosts);

  // Set up
  struct StationID stationID(str(format("CS%03d") % rank), "LBA");
  struct BufferSettings settings(stationID, false);

  size_t blockSize = 1024;

  // Rank 0 is the sender, rank 1 is the receiver
  ASSERT(nrHosts == 2);

  for(size_t i = 0; i < NRBEAMLETS; ++i) {
    beamletDistribution.push_back(i);
  }

  sender = new MPISendStation(settings, 0, 1, beamletDistribution);
  receiver = new MPIReceiveStations(1, beamletDistribution, blockSize);

  // Fill input
  for (size_t i = 0; i < data_in.size(); ++i) {
    data_in[i].x = i16complex(0,    i);
    data_in[i].y = i16complex(1000, 1000 + i);
  }

  // Run tests
  int result = UnitTest::RunAllTests() > 0;

  // Tear down
  receiver = 0;
  sender = 0;

  MPI_Finalize();

  return result;
}

