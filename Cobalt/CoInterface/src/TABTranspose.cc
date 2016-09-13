//# TABTranspose.cc
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
//# $Id: BlockID.h 26419 2013-09-09 11:19:56Z mol $

#include <lofar_config.h>
#include "TABTranspose.h"

#include <Common/LofarLogger.h>
#include <boost/format.hpp>
#include <algorithm>

using namespace std;
using boost::format;

namespace LOFAR {
namespace Cobalt {
namespace TABTranspose {

Subband::Subband( size_t nrSamples, size_t nrChannels )
: 
  data(boost::extents[nrSamples][nrChannels])
{
  id.fileIdx = 0;
  id.subband = 0;
  id.block   = 0;
}


void Subband::write(Stream &stream) const {
  stream.write(&id, sizeof id);

  size_t dim1 = data.shape()[0];
  size_t dim2 = data.shape()[1];
  stream.write(&dim1, sizeof dim1);
  stream.write(&dim2, sizeof dim2);
  stream.write(data.origin(), data.num_elements() * sizeof *data.origin());
  LOG_DEBUG_STR("Written block");
}


void Subband::read(Stream &stream) {
  LOG_DEBUG_STR("Reading block id");
  stream.read(&id, sizeof id);
  LOG_DEBUG_STR("Read block id");

  size_t dim1, dim2;
  stream.read(&dim1, sizeof dim1);
  stream.read(&dim2, sizeof dim2);
  data.resize(boost::extents[dim1][dim2]);
  stream.read(data.origin(), data.num_elements() * sizeof *data.origin());
  LOG_DEBUG_STR("Read block");
}


Block::Block( size_t nrSubbands, size_t nrSamples, size_t nrChannels )
:
  data(boost::extents[nrSubbands][nrSamples][nrChannels]),
  subbandWritten(nrSubbands, false),
  fileIdx(0),
  block(0),
  nrSubbandsLeft(nrSubbands)
{
}


void Block::addSubband( const Subband &subband ) {
  ASSERT(nrSubbandsLeft > 0);

  // Only add subbands that match our ID
  ASSERT(subband.id.fileIdx == fileIdx);
  ASSERT(subband.id.block   == block);

  memcpy(data[subband.id.subband].origin(), subband.data.origin(), subband.data.num_elements() * sizeof *subband.data.origin());
  subbandWritten[subband.id.subband] = true;

  nrSubbandsLeft--;
}


void Block::zeroRemainingSubbands() {
  for (size_t subbandIdx = 0; subbandIdx < subbandWritten.size(); ++subbandIdx) {
    if (!subbandWritten[subbandIdx]) {
      memset(data[subbandIdx].origin(), 0, data[subbandIdx].size() * sizeof *data[subbandIdx].origin());
    }
  }
}


bool Block::complete() const {
  return nrSubbandsLeft == 0;
}


BlockCollector::BlockCollector( Pool<Block> &outputPool, size_t fileIdx, size_t maxBlocksInFlight )
:
  outputPool(outputPool),
  fileIdx(fileIdx),
  maxBlocksInFlight(maxBlocksInFlight),
  canDrop(maxBlocksInFlight > 0),
  lastEmitted(-1)
{
}


void BlockCollector::addSubband( const Subband &subband ) {
  ScopedLock sl(mutex);

  const size_t &blockIdx = subband.id.block;

  if (!have(blockIdx)) {
    if (canDrop) {
      if ((ssize_t)blockIdx <= lastEmitted) {
	// too late -- discard packet
	return;
      }
    } else {
      // if we can't drop, we shouldn't have written
      // this block yet.
      ASSERTSTR((ssize_t)blockIdx > lastEmitted, "Received block " << blockIdx << ", but already emitted up to " << lastEmitted << " for TAB " << subband.id.fileIdx << " subband " << subband.id.subband);
    }

    fetch(blockIdx);
  }

  // augment existing block
  SmartPtr<Block> &block = blocks.at(blockIdx);

  block->addSubband(subband);

  if (block->complete()) {
    // Block is complete -- send it downstream,
    // and everything before it. We know we won't receive
    // data from earlier blocks, because all subbands
    // are sent in-order.
    emitUpTo(blockIdx);
  }
}


void BlockCollector::finish() {
  ScopedLock sl(mutex);

  if (!blocks.empty()) {
    emitUpTo(maxBlock());
  }

  // Signal end-of-stream
  outputPool.filled.append(NULL);
}


size_t BlockCollector::minBlock() const {
  ASSERT(!blocks.empty());

  return blocks.begin()->first;
}

size_t BlockCollector::maxBlock() const {
  ASSERT(!blocks.empty());

  return blocks.rbegin()->first;
}


void BlockCollector::emit(size_t blockIdx) {
  // should emit in-order
  if (!canDrop) {
    ASSERT((ssize_t)blockIdx == lastEmitted + 1);
  } else {
    ASSERT((ssize_t)blockIdx > lastEmitted);
  }
  lastEmitted = blockIdx;

  // clear data we didn't receive
  SmartPtr<Block> &block = blocks.at(blockIdx);

  block->zeroRemainingSubbands();
  
  // emit to outputPool.filled()
  outputPool.filled.append(block);

  // remove from our administration
  blocks.erase(blockIdx);
}


void BlockCollector::emitUpTo(size_t block) {
  while (!blocks.empty() && minBlock() <= block) {
    emit(minBlock());
  }
}


bool BlockCollector::have(size_t block) const {
  return blocks.find(block) != blocks.end();
}


void BlockCollector::fetch(size_t block) {
  ASSERT(!have(block));

  if (canDrop && blocks.size() >= maxBlocksInFlight) {
    // No more room -- force out oldest block
    emit(minBlock());
  }

  blocks[block] = outputPool.free.remove();

  // Annotate
  blocks[block]->fileIdx = fileIdx;
  blocks[block]->block = block;
}


Receiver::Receiver( Stream &stream, CollectorMap &collectors )
:
  stream(stream),
  collectors(collectors),
  thread(this, &Receiver::receiveLoop)
{
}


Receiver::~Receiver()
{
  kill();
}


void Receiver::kill()
{
  thread.cancel();
}


bool Receiver::finish()
{
  thread.wait();

  return !thread.caughtException();
}


void Receiver::receiveLoop()
{
  try {
    Subband subband;

    for(;;) {
      subband.read(stream);

      const size_t fileIdx = subband.id.fileIdx;

      ASSERTSTR(collectors.find(fileIdx) != collectors.end(), "Received a piece of TAB " << fileIdx << ", which is unknown to me");

      LOG_DEBUG_STR("TAB " << fileIdx << ": Adding subband " << subband.id.subband);

      collectors.at(fileIdx)->addSubband(subband);
    }
  } catch (Stream::EndOfStreamException &) {
  }
}


MultiReceiver::MultiReceiver( const std::string &servicePrefix, Receiver::CollectorMap &collectors )
:
  servicePrefix(servicePrefix),
  collectors(collectors),
  thread(this, &MultiReceiver::listenLoop)
{
}


MultiReceiver::~MultiReceiver()
{
  kill(0);
}


void MultiReceiver::kill(size_t minNrClients)
{
  {
    ScopedLock sl(mutex);

    while(clients.size() < minNrClients) {
      LOG_DEBUG_STR("MultiReceiver::kill: " << clients.size() << " clients connected, waiting for " << minNrClients);
      newClient.wait(mutex);
    }

  }

  // Kill listenLoop, but keep clients alive
  thread.cancel();

  // Wait for thread to die so we can access `clients'
  // safely.
  thread.wait();

  // Kill all client connections
  for (size_t i = 0; i < clients.size(); ++i) {
    if (minNrClients == 0) {
      clients[i].receiver->kill();
    } else {
      clients[i].receiver->finish();
    }
  }
}


void MultiReceiver::listenLoop()
{
  for(;;) {
    // Accept a new client
    SmartPtr<PortBroker::ServerStream> stream;
   
    try {
      stream = new PortBroker::ServerStream(servicePrefix, true);
    } catch(SocketStream::TimeOutException &) {
      // fail silently if no client connected
      LOG_DEBUG_STR("TABTranspose::MultiReceiver: Timed out");
      break;
    }

    LOG_DEBUG_STR("TABTranspose::MultiReceiver: Accepted resource " << stream->getResource());

    // Dispatch a Receiver for the client
    dispatch(stream.release());
  }
}


void MultiReceiver::dispatch( PortBroker::ServerStream *stream )
{
  ScopedLock sl(mutex);

  struct Client client;

  client.stream = stream;
  client.receiver = new Receiver(*stream, collectors);

  clients.push_back(client);

  // Inform of change in collectors
  newClient.signal();

  LOG_DEBUG_STR("TABTranspose::MultiReceiver: Dispatched client for resource " << stream->getResource());
}


MultiSender::MultiSender( const HostMap &hostMap, size_t queueSize, bool canDrop )
:
  hostMap(hostMap)
{
  for (HostMap::const_iterator i = hostMap.begin(); i != hostMap.end(); ++i) {
    if(find(hosts.begin(), hosts.end(), i->second) == hosts.end())
      hosts.push_back(i->second);
  }

  for (vector<struct Host>::const_iterator i = hosts.begin(); i != hosts.end(); ++i) {
    queues[*i] = new BestEffortQueue< SmartPtr<struct Subband> >(queueSize, canDrop);
  }
}


void MultiSender::process()
{
#pragma omp parallel for num_threads(hosts.size())
  for (int i = 0; i < (ssize_t)hosts.size(); ++i) {
    try {
      const struct Host &host = hosts[i];

      LOG_DEBUG_STR("MultiSender: Connecting to " << host.hostName << ":" << host.brokerPort << ":" << host.service);

      PortBroker::ClientStream stream(host.hostName, host.brokerPort, host.service);

      LOG_DEBUG_STR("MultiSender->" << host.hostName << ": connected");

      SmartPtr< BestEffortQueue< SmartPtr<struct Subband> > > &queue = queues.at(host);

      LOG_DEBUG_STR("MultiSender->" << host.hostName << ": processing queue");

      SmartPtr<struct Subband> subband;

      while ((subband = queue->remove()) != NULL) {
        subband->write(stream);
      }

      LOG_DEBUG_STR("MultiSender->" << host.hostName << ": done");
    } catch (Exception &ex) {
      LOG_ERROR_STR("Caught exception: " << ex);
    }
  }
}


void MultiSender::append( SmartPtr<struct Subband> &subband )
{
  // Find the host to send these data to
  const struct Host &host = hostMap.at(subband->id.fileIdx);

  // Append the data to the respective queue
  queues.at(host)->append(subband);
}


void MultiSender::finish()
{
  for (vector<struct Host>::const_iterator i = hosts.begin(); i != hosts.end(); ++i) {
    queues.at(*i)->noMore();
  }
}

} // namespace TABTranspose
} // namespace Cobalt
} // namespace LOFAR

