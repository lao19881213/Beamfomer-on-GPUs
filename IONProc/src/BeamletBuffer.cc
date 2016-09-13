//#  BeamletBuffer.cc: one line description
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
//#  $Id: BeamletBuffer.cc 25528 2013-07-02 09:23:01Z loose $

//# Always #include <lofar_config.h> first!
#include <lofar_config.h>

#include <Interface/Align.h>
#include <Interface/Exceptions.h>
#include <BeamletBuffer.h>
#include <ION_Allocator.h>
#include <InputThreadAsm.h>
#include <RSP.h>

#include <boost/lexical_cast.hpp>
#include <cstring>
#include <stdexcept>

#include <boost/format.hpp>
using boost::format;


namespace LOFAR {
namespace RTCP {

template<typename SAMPLE_TYPE> const unsigned BeamletBuffer<SAMPLE_TYPE>::itsNrTimesPerPacket;


// The buffer size is a multiple of the input packet size.  By setting
// itsOffset to a proper value, we can assure that input packets never
// wrap around the circular buffer

template<typename SAMPLE_TYPE> BeamletBuffer<SAMPLE_TYPE>::BeamletBuffer(const Parset *ps, string &stationName, unsigned rspBoard)
:
  itsRSPboard(rspBoard),
  itsNrSubbands(ps->nrSlotsInFrame()),
  itsPacketSize(sizeof(struct RSP::Header) + itsNrTimesPerPacket * itsNrSubbands * NR_POLARIZATIONS * sizeof(SAMPLE_TYPE)),
  itsSize(align(ps->inputBufferSize(), itsNrTimesPerPacket)),
  itsHistorySize(ps->nrHistorySamples()),
  itsIsRealTime(ps->realTime()),
  itsSynchronizedReaderWriter(itsIsRealTime ? 0 : new SynchronizedReaderAndWriter(itsSize)), // FIXME: does not work for multiple observations
  itsLockedRanges(itsSize),
  itsSBBuffers(boost::extents[itsNrSubbands][itsSize][NR_POLARIZATIONS], 128, hugeMemoryAllocator),
  itsOffset(0),
  itsPreviousTimeStamp(0),
  itsPreviousI(0),
  itsCurrentTimeStamp(0),
  itsCurrentI(0),
#if defined HAVE_BGP && !defined USE_VALGRIND
  itsStride(itsSBBuffers[0].num_elements() * sizeof(SAMPLE_TYPE)),
#else
  itsStride(itsSBBuffers[0].num_elements()),
#endif
  itsReadTimer("buffer read", true, true),
  itsWriteTimer("buffer write", true, true)
{
  itsLogPrefix = str(format("[station %s board %u] ") % stationName % rspBoard);

  if (ps->getUint32("OLAP.nrTimesInFrame", 16) != itsNrTimesPerPacket)
    THROW(IONProcException, "OLAP.nrTimesInFrame should be " << boost::lexical_cast<std::string>(itsNrTimesPerPacket));

#if 0
  if (ps->realTime())
    itsSynchronizedReaderWriter = new TimeSynchronizedReader(ps->maxNetworkDelay());  
  else
    itsSynchronizedReaderWriter = new SynchronizedReaderAndWriter(itsSize);
#endif

#if defined USE_VALGRIND
  memset(itsSBBuffers.origin(), 0, itsSBBuffers.num_elements() * sizeof(SAMPLE_TYPE));
#endif

  LOG_DEBUG_STR(itsLogPrefix << "Circular buffer at " << itsSBBuffers.origin() << "; contains " << itsSize << " samples");
}


#if defined HAVE_BGP && !defined USE_VALGRIND

template<> inline void BeamletBuffer<i4complex>::writePacket(i4complex *dst, const i4complex *src)
{
  _copy_pkt_to_bbuffer_32_bytes(dst, itsStride, src, itsNrSubbands);
}

template<> inline void BeamletBuffer<i8complex>::writePacket(i8complex *dst, const i8complex *src)
{
  _copy_pkt_to_bbuffer_64_bytes(dst, itsStride, src, itsNrSubbands);
}

template<> inline void BeamletBuffer<i16complex>::writePacket(i16complex *dst, const i16complex *src)
{
  _copy_pkt_to_bbuffer_128_bytes(dst, itsStride, src, itsNrSubbands);
}

#endif


template<typename SAMPLE_TYPE> inline void BeamletBuffer<SAMPLE_TYPE>::writePacket(SAMPLE_TYPE *dst, const SAMPLE_TYPE *src)
{
  for (unsigned sb = 0; sb < itsNrSubbands; sb ++) {
    for (unsigned i = 0; i < itsNrTimesPerPacket * NR_POLARIZATIONS; i ++)
      dst[i] = *src ++;

    dst += itsStride;
  }
}


template<typename SAMPLE_TYPE> inline void BeamletBuffer<SAMPLE_TYPE>::updateValidData(const TimeStamp &begin, const TimeStamp &end)
{
  ScopedLock sl(itsValidDataMutex);

  itsValidData.exclude(0, end - itsSize);  // forget old ValidData

  // add new ValidData (except if range list will grow too long, to avoid long
  // computations)

  const SparseSet<TimeStamp>::Ranges &ranges = itsValidData.getRanges();

  if (ranges.size() < 64 || ranges.back().end == begin) 
    itsValidData.include(begin, end);
}


template<typename SAMPLE_TYPE> void BeamletBuffer<SAMPLE_TYPE>::writeConsecutivePackets(unsigned count)
{
  unsigned  nrTimes = count * itsNrTimesPerPacket;
  TimeStamp begin   = itsCurrentTimeStamp, end  = begin + nrTimes;
  unsigned  startI  = itsCurrentI,	   endI = startI + nrTimes;

  if (endI >= itsSize)
    endI -= itsSize;

  SAMPLE_TYPE *dst = itsSBBuffers[0][startI].origin();
  
  // in synchronous mode, do not overrun tail of reader
  if (!itsIsRealTime)
    itsSynchronizedReaderWriter->startWrite(begin, end);

  // do not write in circular buffer section that is being read
  itsLockedRanges.lock(startI, endI);

  while (itsCurrentI != endI) {
    writePacket(dst, reinterpret_cast<const SAMPLE_TYPE *>(itsCurrentPacketPtr));
    itsCurrentPacketPtr += itsPacketSize;
    dst			+= itsNrTimesPerPacket * NR_POLARIZATIONS;

    if ((itsCurrentI += itsNrTimesPerPacket) == itsSize) {
      itsCurrentI = 0;
      dst	  = itsSBBuffers.origin();
    }
  }

  itsCurrentTimeStamp = end;
  updateValidData(begin, end);

  itsLockedRanges.unlock(startI, endI);

  if (!itsIsRealTime)
    itsSynchronizedReaderWriter->finishedWrite(end);
}


template<typename SAMPLE_TYPE> void BeamletBuffer<SAMPLE_TYPE>::resetCurrentTimeStamp(const TimeStamp &newTimeStamp)
{
  // A packet with unexpected timestamp was received.  Handle accordingly.
  bool firstPacket = !itsCurrentTimeStamp; // the first timestamp is always unexpected

  itsCurrentTimeStamp = newTimeStamp;
  itsCurrentI	      = mapTime2Index(newTimeStamp);

  if (!aligned(itsCurrentI, itsNrTimesPerPacket)) {
    // RSP board reset?  Recompute itsOffset and clear the entire buffer.

    ScopedLock sl(itsReadMutex); // avoid reset while other thread reads

    int oldOffset = itsOffset;
    itsOffset   = - (newTimeStamp % itsNrTimesPerPacket);
    itsCurrentI = mapTime2Index(newTimeStamp);
    assert(aligned(itsCurrentI, itsNrTimesPerPacket));

    {
      ScopedLock sl(itsValidDataMutex);
      itsValidData.reset();
    }

    if (!firstPacket) {
      LOG_WARN_STR(itsLogPrefix << "Reset BeamletBuffer at " << newTimeStamp << "; itsOffset was " << oldOffset << " and becomes " << itsOffset);
    }  
  }
}


template<typename SAMPLE_TYPE> void BeamletBuffer<SAMPLE_TYPE>::writeMultiplePackets(const void *rspData, const std::vector<TimeStamp> &timeStamps)
{
  itsWriteTimer.start();
  itsCurrentPacketPtr = reinterpret_cast<const char *>(rspData) + sizeof(struct RSP::Header);

  for (unsigned first = 0, last; first < timeStamps.size();) {
    if (timeStamps[first] != itsCurrentTimeStamp)
      resetCurrentTimeStamp(timeStamps[first]);

    // find a series of consecutively timed packets
    for (last = first + 1; last < timeStamps.size() && timeStamps[last] == timeStamps[last - 1] + itsNrTimesPerPacket; last ++)
      ;

    writeConsecutivePackets(last - first);
    first = last;
  }

  itsWriteTimer.stop();
}


template<typename SAMPLE_TYPE> void BeamletBuffer<SAMPLE_TYPE>::writePacketData(const SAMPLE_TYPE *data, const TimeStamp &begin)
{
  itsWriteTimer.start();

  TimeStamp end = begin + itsNrTimesPerPacket;

  // cache previous index, to avoid expensive mapTime2Index()
  unsigned startI;

  if (begin == itsPreviousTimeStamp) {
    startI = itsPreviousI;
  } else {
    startI = mapTime2Index(begin);

    if (!aligned(startI, itsNrTimesPerPacket)) {
      // RSP board reset?  Recompute itsOffset and clear the entire buffer.
      itsOffset = - (startI % itsNrTimesPerPacket);
      startI    = mapTime2Index(begin);

      {
        ScopedLock sl(itsValidDataMutex);
        itsValidData.reset();
      }
    }

    //LOG_DEBUG_STR(""timestamp = " << (uint64_t) begin << ", itsOffset = " << itsOffset");
  }

  unsigned endI = startI + itsNrTimesPerPacket;

  if (endI >= itsSize)
    endI -= itsSize;

  itsPreviousTimeStamp = end;
  itsPreviousI	       = endI;

  // in synchronous mode, do not overrun tail of reader
  if (!itsIsRealTime)
    itsSynchronizedReaderWriter->startWrite(begin, end);

  // do not write in circular buffer section that is being read
  itsLockedRanges.lock(startI, endI);

  writePacket(itsSBBuffers[0][startI].origin(), data);
  
  // forget old ValidData
  {
    ScopedLock sl(itsValidDataMutex);
    itsValidData.exclude(0, end - itsSize);

    unsigned rangesSize = itsValidData.getRanges().size();

    // add new ValidData (except if range list will grow too long, to avoid long
    // computations)
    if (rangesSize < 64 || itsValidData.getRanges()[rangesSize - 1].end == begin) 
      itsValidData.include(begin, end);
  }  

  itsLockedRanges.unlock(startI, endI);

  if (!itsIsRealTime)
    itsSynchronizedReaderWriter->finishedWrite(end);

  itsWriteTimer.stop();
}


template<typename SAMPLE_TYPE> void BeamletBuffer<SAMPLE_TYPE>::startReadTransaction(const std::vector<TimeStamp> &begin, unsigned nrElements)
{
  // in synchronous mode, do not overrun writer
  if (!itsIsRealTime) {
    TimeStamp minBegin = *std::min_element(begin.begin(), begin.end());
    TimeStamp maxEnd   = *std::max_element(begin.begin(), begin.end()) + nrElements;
    itsSynchronizedReaderWriter->startRead(minBegin, maxEnd);
  }

  itsReadMutex.lock(); // only one reader per BeamletBuffer allowed
  itsReadTimer.start();

  unsigned nrBeams = begin.size();

  itsEnd.resize(nrBeams);
  itsStartI.resize(nrBeams);
  itsEndI.resize(nrBeams);

  itsBegin = begin;

  for (unsigned beam = 0; beam < begin.size(); beam ++) {
    itsEnd[beam]    = begin[beam] + nrElements;
    itsStartI[beam] = mapTime2Index(begin[beam]);
    itsEndI[beam]   = mapTime2Index(itsEnd[beam]);
  }
 
  itsMinEnd	     = *std::min_element(itsEnd.begin(),    itsEnd.end());
  itsMinStartI	     = *std::min_element(itsStartI.begin(), itsStartI.end());
  itsMaxEndI	     = *std::max_element(itsEndI.begin(),   itsEndI.end());

  // do not read from circular buffer section that is being written
  itsLockedRanges.lock(itsMinStartI, itsMaxEndI);
}


template<typename SAMPLE_TYPE> void BeamletBuffer<SAMPLE_TYPE>::sendSubband(Stream *str, unsigned subband, unsigned beam) const
{
  // Align to 32 bytes and make multiple of 32 bytes by prepending/appending
  // extra data.  Always send 32 bytes extra, even if data was already aligned.
  unsigned startI = align(itsStartI[beam] - itsAlignment + 1, itsAlignment); // round down
  unsigned endI   = align(itsEndI[beam] + 1, itsAlignment); // round up, possibly adding 32 bytes
  
  if (endI < startI) {
    // the data wraps around the allocated memory, so copy in two parts
    unsigned firstChunk = itsSize - startI;

    str->write(itsSBBuffers[subband][startI].origin(), sizeof(SAMPLE_TYPE[firstChunk][NR_POLARIZATIONS]));
    str->write(itsSBBuffers[subband][0].origin(),      sizeof(SAMPLE_TYPE[endI][NR_POLARIZATIONS]));
  } else {
    str->write(itsSBBuffers[subband][startI].origin(), sizeof(SAMPLE_TYPE[endI - startI][NR_POLARIZATIONS]));
  }
}


template<typename SAMPLE_TYPE> void BeamletBuffer<SAMPLE_TYPE>::sendUnalignedSubband(Stream *str, unsigned subband, unsigned beam) const
{
  if (itsEndI[beam] < itsStartI[beam]) {
    // the data wraps around the allocated memory, so copy in two parts
    unsigned firstChunk = itsSize - itsStartI[beam];

    str->write(itsSBBuffers[subband][itsStartI[beam]].origin(), sizeof(SAMPLE_TYPE[firstChunk][NR_POLARIZATIONS]));
    str->write(itsSBBuffers[subband][0].origin(),		sizeof(SAMPLE_TYPE[itsEndI[beam]][NR_POLARIZATIONS]));
  } else {
    str->write(itsSBBuffers[subband][itsStartI[beam]].origin(), sizeof(SAMPLE_TYPE[itsEndI[beam] - itsStartI[beam]][NR_POLARIZATIONS]));
  }
}


template<typename SAMPLE_TYPE> SparseSet<unsigned> BeamletBuffer<SAMPLE_TYPE>::readFlags(unsigned beam)
{
  itsValidDataMutex.lock();
  SparseSet<TimeStamp> validTimes = itsValidData.subset(itsBegin[beam], itsEnd[beam]);
  itsValidDataMutex.unlock();

  SparseSet<unsigned> flags;
  flags.include(0, static_cast<unsigned>(itsEnd[beam] - itsBegin[beam]));
  
  for (SparseSet<TimeStamp>::const_iterator it = validTimes.getRanges().begin(); it != validTimes.getRanges().end(); it ++)
    flags.exclude(static_cast<unsigned>(it->begin - itsBegin[beam]),
		  static_cast<unsigned>(it->end - itsBegin[beam]));

  return flags;
}


template<typename SAMPLE_TYPE> void BeamletBuffer<SAMPLE_TYPE>::stopReadTransaction()
{
  itsLockedRanges.unlock(itsMinStartI, itsMaxEndI);

  if (!itsIsRealTime)
    itsSynchronizedReaderWriter->finishedRead(itsMinEnd - (itsHistorySize + 16));
    // subtract 16 extra; due to alignment restrictions and the changing delays,
    // it is hard to predict where the next read will begin.
  
  itsReadTimer.stop();
  itsReadMutex.unlock();
}


template<typename SAMPLE_TYPE> void BeamletBuffer<SAMPLE_TYPE>::noMoreReading()
{
  if (!itsIsRealTime)
    itsSynchronizedReaderWriter->noMoreReading();
}


template<typename SAMPLE_TYPE> void BeamletBuffer<SAMPLE_TYPE>::noMoreWriting()
{
  if (!itsIsRealTime)
    itsSynchronizedReaderWriter->noMoreWriting();
}


template class BeamletBuffer<i4complex>;
template class BeamletBuffer<i8complex>;
template class BeamletBuffer<i16complex>;

} // namespace RTCP
} // namespace LOFAR
