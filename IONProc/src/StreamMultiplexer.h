//# StreamMultiplexer.h: 
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
//# $Id: Stream.h 14333 2009-10-28 08:43:36Z romein $

#ifndef LOFAR_LCS_STREAM_STREAM_MULTIPLEXER_H
#define LOFAR_LCS_STREAM_STREAM_MULTIPLEXER_H

#include <Stream/Stream.h>
#include <Common/Thread/Condition.h>
#include <Common/Thread/Mutex.h>
#include <Common/Thread/Semaphore.h>
#include <Common/Thread/Thread.h>
#include <Interface/SmartPtr.h>

#include <map>


namespace LOFAR {
namespace RTCP {

class MultiplexedStream;

class StreamMultiplexer
{
  public:
	   StreamMultiplexer(Stream &);
	   ~StreamMultiplexer();

    void   start();       

    void   registerChannel(MultiplexedStream *, unsigned channel);

    size_t tryRead(MultiplexedStream *, void *ptr, size_t size);
    size_t tryWrite(MultiplexedStream *, const void *ptr, size_t size);

  private:
    friend class MultiplexedStream;

    void   receiveThread();

    struct Request;

    struct RequestMsg {
      enum { RECV_REQ, RECV_ACK, REGISTER, STOP_REQ } type;
      size_t			  size;
      Request			  *reqPtr;	 // in addr space of callee
      size_t			  *sizePtr;	 // in addr space of caller
      void			  *recvPtr;	 // in addr space of caller
      Semaphore			  *recvFinished; // in addr space of caller
    };

    struct Request {
      RequestMsg msg;
      Semaphore	 received;
    };

    Stream &itsStream;
    Mutex  itsSendMutex;

    template <typename K, typename V> class Map {
      public:
        void insert(K, V);
	V    remove(K);

      private:
	std::map<K, V>	itsMap;
        Mutex		itsMutex;
	Condition	itsReevaluate;
    };

    Map<unsigned, Request *> itsOutstandingRegistrations;

    SmartPtr<Thread>	     itsReceiveThread;
};


class MultiplexedStream : public Stream
{
  public:
		   MultiplexedStream(StreamMultiplexer &, unsigned channel);
    virtual	   ~MultiplexedStream();

    virtual size_t tryRead(void *ptr, size_t size);
    virtual size_t tryWrite(const void *ptr, size_t size);

  private:
    friend class   StreamMultiplexer;

    StreamMultiplexer &itsMultiplexer;
    unsigned	      itsChannel;

    StreamMultiplexer::Request itsRequest, *itsPeerRequestAddr;
};

} // namespace RTCP
} // namespace LOFAR

#endif
