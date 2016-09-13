//# 
//#
//# Copyright (C) 2000, 2001
//# ASTRON (Netherlands Foundation for Research in Astronomy)
//# P.O.Box 2, 7990 AA Dwingeloo, The Netherlands, seg@astron.nl
//#
//# This program is free software; you can redistribute it and/or modify
//# it under the terms of the GNU General Public License as published by
//# the Free Software Foundation; either version 2 of the License, or
//# (at your option) any later version.
//#
//# This program is distributed in the hope that it will be useful,
//# but WITHOUT ANY WARRANTY; without even the implied warranty of
//# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//# GNU General Public License for more details.
//#
//# You should have received a copy of the GNU General Public License
//# along with this program; if not, write to the Free Software
//# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//#
//# $Id: ReaderWriterSynchronization.cc 17893 2011-04-29 09:04:10Z romein $

//# Always #include <lofar_config.h> first!
#include <lofar_config.h>

#include <ReaderWriterSynchronization.h>


namespace LOFAR {
namespace RTCP {


ReaderAndWriterSynchronization::~ReaderAndWriterSynchronization()
{
}




SynchronizedReaderAndWriter::SynchronizedReaderAndWriter(unsigned bufferSize)
:
  itsBufferSize(bufferSize)
{
}


SynchronizedReaderAndWriter::~SynchronizedReaderAndWriter()
{
}


void SynchronizedReaderAndWriter::startRead(const TimeStamp &begin, const TimeStamp &end)
{
  itsReadPointer.advanceTo(begin);
  itsWritePointer.waitFor(end);
}


void SynchronizedReaderAndWriter::finishedRead(const TimeStamp &advanceTo)
{
  itsReadPointer.advanceTo(advanceTo);
}


void SynchronizedReaderAndWriter::startWrite(const TimeStamp &begin, const TimeStamp &end)
{
  itsWritePointer.advanceTo(begin);
  itsReadPointer.waitFor(end - itsBufferSize);
}


void SynchronizedReaderAndWriter::finishedWrite(const TimeStamp &advanceTo)
{
  itsWritePointer.advanceTo(advanceTo);
}


void SynchronizedReaderAndWriter::noMoreReading()
{
  // advance read pointer to infinity, to unblock thread that waits in startWrite
  itsReadPointer.advanceTo(TimeStamp(0x7FFFFFFFFFFFFFFFLL)); // we only use this TimeStamp for comparison so clockSpeed does not matter
}


void SynchronizedReaderAndWriter::noMoreWriting()
{
  itsWritePointer.advanceTo(TimeStamp(0x7FFFFFFFFFFFFFFFLL));
}


TimeSynchronizedReader::TimeSynchronizedReader(unsigned maximumNetworkLatency)
:
  itsMaximumNetworkLatency(maximumNetworkLatency)
{
}


TimeSynchronizedReader::~TimeSynchronizedReader()
{
}


void TimeSynchronizedReader::startRead(const TimeStamp & /*begin*/, const TimeStamp &end)
{
  itsWallClock.waitUntil(end + itsMaximumNetworkLatency);
}


void TimeSynchronizedReader::finishedRead(const TimeStamp & /*advanceTo*/)
{
}


void TimeSynchronizedReader::startWrite(const TimeStamp & /*begin*/, const TimeStamp & /*end*/)
{
}


void TimeSynchronizedReader::finishedWrite(const TimeStamp & /*advanceTo*/)
{
}

} // namespace RTCP
} // namespace LOFAR
