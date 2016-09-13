//#  OutputThread.h
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
//#  $Id: OutputThread.h 14194 2009-10-06 09:54:51Z romein $

#ifndef LOFAR_RTCP_STORAGE_OUTPUT_THREAD_H
#define LOFAR_RTCP_STORAGE_OUTPUT_THREAD_H

//# Never #include <config.h> or #include <lofar_config.h> in a header file!

#include <Interface/OutputTypes.h>
#include <Interface/SmartPtr.h>
#include <Interface/StreamableData.h>
#include <Interface/FinalMetaData.h>
#include <Storage/MSWriter.h>
#include <Stream/FileStream.h>
#include <Common/Thread/Queue.h>
#include <Common/Thread/Thread.h>

#include <string>
#include <vector>


namespace LOFAR {
namespace RTCP {


class OutputThread
{
  public:
				     OutputThread(const Parset &, OutputType, unsigned streamNr, Queue<SmartPtr<StreamableData> > &freeQueue, Queue<SmartPtr<StreamableData> > &receiveQueue, const std::string &logPrefix, bool isBigEndian, const std::string &targetDirectory = "");

    void			     start();

    // needed in createHeaders.cc
    void           createMS();
    void			     cleanUp();

    void           augment(const FinalMetaData &finalMetaData);

  private:
    void			     checkForDroppedData(StreamableData *);
    void			     doWork();
    void			     mainLoop();

    const Parset		     &itsParset;
    const OutputType		     itsOutputType;
    const unsigned		     itsStreamNr;
    const bool			     itsIsBigEndian;
    const std::string		     itsLogPrefix;
    const bool                       itsCheckFakeData;
    const std::string                itsTargetDirectory;

    Queue<SmartPtr<StreamableData> > &itsFreeQueue, &itsReceiveQueue;

    unsigned		 	     itsBlocksWritten, itsBlocksDropped;
    unsigned           itsNrExpectedBlocks;
    unsigned			     itsNextSequenceNumber;
    SmartPtr<MSWriter>		     itsWriter;
    SmartPtr<Thread>		     itsThread;
};


} // namespace RTCP
} // namespace LOFAR

#endif
