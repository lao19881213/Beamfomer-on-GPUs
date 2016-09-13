//#  InputThread.cc: the thread that reads from a Stream and places data into
//#  the buffer of the input section
//#
//#  Copyright (C) 2006
//#  ASTRON (Netherlands Foundation for Research in Astronomy)
//#  P.O.Box 2, 7990 AA Dwingeloo, The Netherlands, seg@astron.nl
//#
//#  This program is free software; you can redistribute it and/or modify
//#  it under the terms of the GNU General Public License as published by
//#  the Free Software Foundation; either version 2 of the License, or
//#  (at your option) any later version.
//#
//#  This program is distributed in the hope that it will be useful,
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//#  GNU General Public License for more details.
//#
//#  You should have received a copy of the GNU General Public License
//#  along with this program; if not, write to the Free Software
//#  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//#
//#  $Id: InputThread.cc 22045 2012-09-17 14:57:53Z mol $

//# Always #include <lofar_config.h> first!
#include <lofar_config.h>

//# Includes
#include <Common/DataConvert.h>
#include <Common/LofarLogger.h>
#include <Common/SystemCallException.h>
#include <Common/Timer.h>
#include <Interface/AlignedStdAllocator.h>
#include <Interface/Exceptions.h>
#include <Interface/SmartPtr.h>
#include <Stream/NullStream.h>
#include <Stream/SocketStream.h>
#include <BeamletBuffer.h>
#include <InputThread.h>
#include <RSP.h>
#include <Scheduling.h>
#include <Common/Thread/Cancellation.h>

#include <cstddef>

#include <boost/multi_array.hpp>


namespace LOFAR {
namespace RTCP {


template <typename SAMPLE_TYPE> InputThread<SAMPLE_TYPE>::InputThread(ThreadArgs args /* call by value! */)
:
  itsArgs(args)
{
  LOG_DEBUG_STR(itsArgs.logPrefix << "InputThread::InputThread(...)");
}


template <typename SAMPLE_TYPE> void InputThread<SAMPLE_TYPE>::start()
{
  itsThread = new Thread(this, &InputThread<SAMPLE_TYPE>::mainLoop, itsArgs.logPrefix + "[InputThread] ", 65536);
}


template <typename SAMPLE_TYPE> InputThread<SAMPLE_TYPE>::~InputThread()
{
  LOG_DEBUG_STR(itsArgs.logPrefix << "InputThread::~InputThread()");

  if (itsThread)
    itsThread->cancel();
}


template <typename SAMPLE_TYPE> void InputThread<SAMPLE_TYPE>::mainLoop()
{
#if 0 && defined HAVE_BGP_ION
  if (0 && itsArgs.threadID == 0)
    runOnCore0();
  else
    doNotRunOnCore0();
#endif

#if 1 && defined HAVE_BGP_ION
  doNotRunOnCore0();
#endif

  const unsigned maxNrPackets = 128;
  TimeStamp	 actualstamp  = itsArgs.startTime - itsArgs.nrTimesPerPacket;
  unsigned	 packetSize   = sizeof(struct RSP::Header) + itsArgs.nrSlotsPerPacket * itsArgs.nrTimesPerPacket * NR_POLARIZATIONS * sizeof(SAMPLE_TYPE);

  std::vector<TimeStamp> timeStamps(maxNrPackets);
  boost::multi_array<char, 2, AlignedStdAllocator<char, 32> > packets(boost::extents[maxNrPackets][packetSize]);
  //boost::multi_array<char, 2, FlatMemoryAllocator<char> > packets(boost::extents[maxNrPackets][packetSize]);

  char		*currentPacketPtr	    = packets.origin();
  unsigned	currentPacket		    = 0;

  unsigned	previousSeqid		    = 0;
  bool		previousSeqidIsAccepted	    = false;
  
  bool		dataShouldContainValidStamp = dynamic_cast<NullStream *>(itsArgs.stream) == 0;
  bool		isUDPstream		    = dynamic_cast<SocketStream *>(itsArgs.stream) != 0 && dynamic_cast<SocketStream *>(itsArgs.stream)->protocol == SocketStream::UDP;
  WallClockTime wallClockTime;

  LOG_DEBUG_STR(itsArgs.logPrefix << " input thread " << itsArgs.threadID << " entering loop");

  while (true) {
    try {
      // cancelable read, to allow stopping this thread even if the station
      // does not send data

      if (isUDPstream) {
	if (itsArgs.stream->tryRead(currentPacketPtr, packetSize) != packetSize) {
	  ++ itsArgs.packetCounters->received;
	  ++ itsArgs.packetCounters->badSize;
	  continue;
	}
      } else {
	Cancellation::point(); // allow cancellation from null:
	itsArgs.stream->read(currentPacketPtr, packetSize);
      }
    } catch (Stream::EndOfStreamException &) {
      break;
    } catch (SystemCallException &ex) {
      if (ex.error == EINTR)
	break;
      else
	throw;
    }

    ++ itsArgs.packetCounters->received;

    if (dataShouldContainValidStamp) {
#if defined __PPC__
      unsigned seqid, blockid;

      asm volatile ("lwbrx %0,%1,%2" : "=r" (seqid)   : "b" (currentPacketPtr), "r" (offsetof(RSP, header.timestamp)));
      asm volatile ("lwbrx %0,%1,%2" : "=r" (blockid) : "b" (currentPacketPtr), "r" (offsetof(RSP, header.blockSequenceNumber)));
#else
      unsigned seqid   = reinterpret_cast<RSP *>(currentPacketPtr)->header.timestamp;
      unsigned blockid = reinterpret_cast<RSP *>(currentPacketPtr)->header.blockSequenceNumber;

#if defined WORDS_BIGENDIAN
      seqid   = byteSwap(seqid);
      blockid = byteSwap(blockid);
#endif
#endif

      //if the seconds counter is 0xFFFFFFFF, the data cannot be trusted.
      if (seqid == ~0U) {
	++ itsArgs.packetCounters->badTimeStamp;
	continue;
      }

      // Sanity check on seqid. Note, that seqid is in seconds,
      // so a value which is greater than the previous one with more 
      // than (say) 10 seconds probably means that the sequence number 
      // in the packet is wrong. This can happen, since communication is not
      // reliable.
      if (seqid >= previousSeqid + 10 && previousSeqidIsAccepted) {
	previousSeqidIsAccepted = false;
	++ itsArgs.packetCounters->badTimeStamp;
	continue;
      }

      // accept seqid
      previousSeqidIsAccepted = true;
      previousSeqid	      = seqid;
	
      actualstamp.setStamp(seqid, blockid);
    } else {
      actualstamp += itsArgs.nrTimesPerPacket; 

      if (itsArgs.isRealTime)
	wallClockTime.waitUntil(actualstamp);
    }

    // expected packet received so write data into corresponding buffer
    //itsArgs.BBuffer->writePacketData(reinterpret_cast<SAMPLE_TYPE *>(&packet.data), actualstamp);

    timeStamps[currentPacket] = actualstamp;
    currentPacketPtr += packetSize;

    if (++ currentPacket == maxNrPackets) {
      itsArgs.BBuffer->writeMultiplePackets(packets.origin(), timeStamps);
      // pthread_yield();
      currentPacket    = 0;
      currentPacketPtr = packets.origin();
    }
  }

  timeStamps.resize(currentPacket);
  itsArgs.BBuffer->writeMultiplePackets(packets.origin(), timeStamps);
  itsArgs.BBuffer->noMoreWriting();

  LOG_DEBUG_STR(itsArgs.logPrefix << "InputThread::mainLoop() exiting");
}


template class InputThread<i4complex>;
template class InputThread<i8complex>;
template class InputThread<i16complex>;

} // namespace RTCP
} // namespace LOFAR
