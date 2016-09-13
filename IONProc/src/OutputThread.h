//#  OldOutputThread.h
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
//#  $Id: OutputThread.h 22419 2012-10-19 09:04:16Z mol $

#ifndef LOFAR_IONPROC_OUTPUT_THREAD_H
#define LOFAR_IONPROC_OUTPUT_THREAD_H

//# Never #include <config.h> or #include <lofar_config.h> in a header file!

#include <Interface/Parset.h>
#include <Interface/OutputTypes.h>
#include <Interface/SmartPtr.h>
#include <Interface/StreamableData.h>
#include <Common/Thread/Queue.h>
#include <Common/Thread/Semaphore.h>
#include <Common/Thread/Thread.h>

#include <string>


namespace LOFAR {
namespace RTCP {

class OutputThread
{
  public:
			    OutputThread(const Parset &, OutputType outputType, unsigned streamNr, unsigned adderNr);

    void                    start();

    static const unsigned   maxSendQueueSize = 4; // use 2 if you run out of memory, but test carefully to avoid data loss

    Queue<SmartPtr<StreamableData> > itsFreeQueue, itsSendQueue;

  private:
    void		    mainLoop();

    const std::string       itsLogPrefix;
    const std::string       itsOutputDescriptor;
    const double            itsDeadline;

  public:
    SmartPtr<Thread>	    itsThread;
};

} // namespace RTCP
} // namespace LOFAR

#endif
