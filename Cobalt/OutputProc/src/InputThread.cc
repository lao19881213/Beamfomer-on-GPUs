//# InputThread.cc
//# Copyright (C) 2008-2013  ASTRON (Netherlands Institute for Radio Astronomy)
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
//# $Id: InputThread.cc 25312 2013-06-12 15:48:13Z mol $

//# Always #include <lofar_config.h> first!
#include <lofar_config.h>

#include "InputThread.h"

#include <Common/Timer.h>
#include <Stream/NullStream.h>
#include <Stream/SocketStream.h>
#include <CoInterface/Stream.h>


namespace LOFAR
{
  namespace Cobalt
  {


    InputThread::InputThread(const Parset &parset, OutputType outputType, unsigned streamNr, Queue<SmartPtr<StreamableData> > &freeQueue, Queue<SmartPtr<StreamableData> > &receiveQueue, const std::string &logPrefix)
      :
      itsLogPrefix(logPrefix + "[InputThread] "),
      itsInputDescriptor(getStreamDescriptorBetweenIONandStorage(parset, outputType, streamNr)),
      itsFreeQueue(freeQueue),
      itsReceiveQueue(receiveQueue),
      itsDeadline(parset.realTime() ? parset.stopTime() : 0)
    {
    }


    void InputThread::start()
    {
      itsThread = new Thread(this, &InputThread::mainLoop, itsLogPrefix);
    }


    void InputThread::cancel()
    {
      if (itsThread)
        itsThread->cancel();
    }


    void InputThread::mainLoop()
    {
      try {
        LOG_INFO_STR(itsLogPrefix << "Creating connection from " << itsInputDescriptor << "..." );
        SmartPtr<Stream> streamFromION(createStream(itsInputDescriptor, true, itsDeadline));
        LOG_INFO_STR(itsLogPrefix << "Creating connection from " << itsInputDescriptor << ": done" );

        // limit reads from NullStream to 10 blocks; otherwise unlimited
        bool nullInput = dynamic_cast<NullStream *>(streamFromION.get()) != 0;

        for (unsigned count = 0; !nullInput || count < 10; count++) {
          SmartPtr<StreamableData> data(itsFreeQueue.remove());

          data->read(streamFromION, true, 1); // Cobalt writes with an alignment of 1

          if (nullInput)
            data->setSequenceNumber(count);

          LOG_DEBUG_STR(itsLogPrefix << "Read block with seqno = " << data->sequenceNumber());

          itsReceiveQueue.append(data.release());
        }
      } catch (SocketStream::TimeOutException &) {
        LOG_WARN_STR(itsLogPrefix << "Connection from " << itsInputDescriptor << " timed out");
      } catch (Stream::EndOfStreamException &) {
        LOG_INFO_STR(itsLogPrefix << "Connection from " << itsInputDescriptor << " closed");
      } catch (SystemCallException &ex) {
        LOG_WARN_STR(itsLogPrefix << "Connection from " << itsInputDescriptor << " failed: " << ex.text());
      } catch (...) {
        itsReceiveQueue.append(0); // no more data
        throw;
      }

      itsReceiveQueue.append(0); // no more data
    }


  } // namespace Cobalt
} // namespace LOFAR

