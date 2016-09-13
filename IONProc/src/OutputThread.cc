//#  OldOutputThread.cc:
//#
//#  Copyright (C) 2008
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
//#  $Id: OutputThread.cc 24629 2013-04-17 12:12:51Z mol $

//# Always #include <lofar_config.h> first!
#include <lofar_config.h>

#include <Common/SystemCallException.h>
#include <ION_Allocator.h>
#include <OutputThread.h>
#include <Scheduling.h>
#include <Interface/DataFactory.h>
#include <Interface/SmartPtr.h>
#include <Interface/Stream.h>
#include <Stream/SocketStream.h>

#include <boost/format.hpp>


namespace LOFAR {
namespace RTCP {


OutputThread::OutputThread(const Parset &parset, OutputType outputType, unsigned streamNr, unsigned adderNr)
:
  itsLogPrefix(str(boost::format("[obs %u type %u stream %3u adder %3u] ") % parset.observationID() % outputType % streamNr % adderNr)),
  itsOutputDescriptor(getStreamDescriptorBetweenIONandStorage(parset, outputType, streamNr)),
  itsDeadline(parset.realTime() ? parset.stopTime() : 0)
{
  for (unsigned i = 0; i < maxSendQueueSize; i ++)
    itsFreeQueue.append(newStreamableData(parset, outputType, streamNr, hugeMemoryAllocator));
}


void OutputThread::start()
{
  itsThread = new Thread(this, &OutputThread::mainLoop, itsLogPrefix + "[OutputThread] ", 65536);
}


void OutputThread::mainLoop()
{
#if defined HAVE_BGP_ION
  doNotRunOnCore0();
  //nice(19);
  //runOnCore0();
#endif

  try {
    LOG_DEBUG_STR(itsLogPrefix << "Creating connection to " << itsOutputDescriptor << "...");
    SmartPtr<Stream> streamToStorage(createStream(itsOutputDescriptor, false, static_cast<time_t>(itsDeadline)));
    LOG_DEBUG_STR(itsLogPrefix << "Creating connection to " << itsOutputDescriptor << ": done");

    for (SmartPtr<StreamableData> data; (data = itsSendQueue.remove()) != 0; itsFreeQueue.append(data.release()))
      data->write(streamToStorage, true); // write data, including serial nr
  } catch (SystemCallException &ex) {
    LOG_WARN_STR(itsLogPrefix << "Connection to " << itsOutputDescriptor << " failed: " << ex.text());
  } catch (SocketStream::TimeOutException &ex) {
    LOG_WARN_STR(itsLogPrefix << "Connection to " << itsOutputDescriptor << " timed out");
  }
}


} // namespace RTCP
} // namespace LOFAR
