//# StreamMultiplexer.cc: 
//#
//# Copyright (C) 2010
//# ASTRON (Netherlands Institute for Radio Astronomy)
//# P.O.Box 2, 7990 AA Dwingeloo, The Netherlands
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
//# $Id: FileDescriptorBasedStream.cc 14333 2009-10-28 08:43:36Z romein $

#include <lofar_config.h>

#include <StreamMultiplexer.h>
#include <Scheduling.h>
#include <Common/Thread/Cancellation.h>
#include <Common/LofarLogger.h>

#include <cstring>


namespace LOFAR {
namespace RTCP {


template <typename K, typename V> void StreamMultiplexer::Map<K, V>::insert(K key, V value)
{
  ScopedLock sl(itsMutex);

  itsMap[key] = value;
  itsReevaluate.broadcast();
}


template <typename K, typename V> V StreamMultiplexer::Map<K, V>::remove(K key)
{
  ScopedLock sl(itsMutex);

  std::map<unsigned, Request *>::iterator it;
  
  while ((it = itsMap.find(key)) == itsMap.end())
    itsReevaluate.wait(itsMutex);
  
  V v = it->second;
  itsMap.erase(it);

  return v;
}


StreamMultiplexer::StreamMultiplexer(Stream &stream)
:
  itsStream(stream)
{
}


void StreamMultiplexer::start()
{
  itsReceiveThread = new Thread(this, &StreamMultiplexer::receiveThread, "[StreamMultiplexer] ", 65536);
}


StreamMultiplexer::~StreamMultiplexer()
{
  RequestMsg msg;

#if defined USE_VALGRIND
  memset(&msg, 0, sizeof msg);
#endif

  msg.type = RequestMsg::STOP_REQ;

  {
    ScopedLock sl(itsSendMutex);
    ScopedDelayCancellation dc;

    itsStream.write(&msg, sizeof msg);
  }
}


void StreamMultiplexer::registerChannel(MultiplexedStream *stream, unsigned channel)
{
  RequestMsg msg;

#if defined USE_VALGRIND
  memset(&msg, 0, sizeof msg);
#endif

  msg.type	= RequestMsg::REGISTER;
  msg.reqPtr	= &stream->itsRequest;
  msg.size	= channel; // FIXME: abuse size field

  {
    ScopedLock sl(itsSendMutex);
    itsStream.write(&msg, sizeof msg);
  }

  stream->itsPeerRequestAddr = itsOutstandingRegistrations.remove(channel);
}


void StreamMultiplexer::receiveThread()
{
#if defined HAVE_BGP_ION
  doNotRunOnCore0();
#endif

  while (1) {
    RequestMsg msg;

    try {
      itsStream.read(&msg, sizeof msg);
    } catch(Stream::EndOfStreamException &) {
      LOG_FATAL("[StreamMultiplexer] Connection reset by peer");
      return;
    }

    switch (msg.type) {
      case RequestMsg::RECV_REQ : msg.reqPtr->msg = msg;
				  msg.reqPtr->received.up();
				  break;

      case RequestMsg::RECV_ACK : itsStream.read(msg.recvPtr, msg.size);
				  *msg.sizePtr = msg.size;
				  msg.recvFinished->up();
				  break;

      case RequestMsg::REGISTER : itsOutstandingRegistrations.insert(msg.size, msg.reqPtr);
				  break;

      case RequestMsg::STOP_REQ : return;
    }
  }
}


size_t StreamMultiplexer::tryRead(MultiplexedStream *stream, void *ptr, size_t size)
{
  Semaphore  recvFinished;
  RequestMsg msg;

#if defined USE_VALGRIND
  memset(&msg, 0, sizeof msg);
#endif

  msg.type	   = RequestMsg::RECV_REQ;
  msg.size	   = size;
  msg.reqPtr       = stream->itsPeerRequestAddr;
  msg.sizePtr	   = &size;
  msg.recvPtr	   = ptr;
  msg.recvFinished = &recvFinished;

  {
    ScopedLock sl(itsSendMutex);
    itsStream.write(&msg, sizeof msg);
  }

  recvFinished.down();

  return size;
}


size_t StreamMultiplexer::tryWrite(MultiplexedStream *stream, const void *ptr, size_t size)
{
  stream->itsRequest.received.down();

  RequestMsg ack = stream->itsRequest.msg;

  ack.type = RequestMsg::RECV_ACK;
  ack.size = std::min(size, ack.size);

  {
    ScopedLock sl(itsSendMutex);
    itsStream.write(&ack, sizeof ack);
    itsStream.write(ptr, ack.size);
  }  

  return ack.size;
}


MultiplexedStream::MultiplexedStream(StreamMultiplexer &multiplexer, unsigned channel)
:
  itsMultiplexer(multiplexer)
{
  itsMultiplexer.registerChannel(this, channel);
}


MultiplexedStream::~MultiplexedStream()
{
}


size_t MultiplexedStream::tryRead(void *ptr, size_t size)
{
  return itsMultiplexer.tryRead(this, ptr, size);
}


size_t MultiplexedStream::tryWrite(const void *ptr, size_t size)
{
  return itsMultiplexer.tryWrite(this, ptr, size);
}


} // namespace RTCP
} // namespace LOFAR
