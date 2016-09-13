//# OutputThread.h
//# Copyright (C) 2009-2013  ASTRON (Netherlands Institute for Radio Astronomy)
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
//# $Id: OutputThread.h 25646 2013-07-12 09:58:39Z mol $

#ifndef LOFAR_RTCP_STORAGE_OUTPUT_THREAD_H
#define LOFAR_RTCP_STORAGE_OUTPUT_THREAD_H

//# Never #include <config.h> or #include <lofar_config.h> in a header file!

#include <string>
#include <vector>

#include <Common/Thread/Queue.h>
#include <Common/Thread/Thread.h>
#include <Stream/FileStream.h>
#include <CoInterface/OutputTypes.h>
#include <CoInterface/SmartPtr.h>
#include <CoInterface/StreamableData.h>
#include <CoInterface/FinalMetaData.h>
#include "MSWriter.h"

namespace LOFAR
{
  namespace Cobalt
  {


    class OutputThread
    {
    public:
      OutputThread(const Parset &, OutputType, unsigned streamNr, Queue<SmartPtr<StreamableData> > &freeQueue, Queue<SmartPtr<StreamableData> > &receiveQueue, const std::string &logPrefix, bool isBigEndian, const std::string &targetDirectory = "");

      void                             start();

      // needed in createHeaders.cc
      void           createMS();
      void           cleanUp() const;

      void           augment(const FinalMetaData &finalMetaData);

      ParameterSet feedbackLTA() const;

    private:
      void                             checkForDroppedData(StreamableData *);
      void                             doWork();
      void                             mainLoop();

      const Parset                     &itsParset;
      const OutputType itsOutputType;
      const unsigned itsStreamNr;
      const bool itsIsBigEndian;
      const std::string itsLogPrefix;
      const std::string itsTargetDirectory;

      Queue<SmartPtr<StreamableData> > &itsFreeQueue, &itsReceiveQueue;

      unsigned itsBlocksWritten, itsBlocksDropped;
      unsigned itsNrExpectedBlocks;
      unsigned itsNextSequenceNumber;
      SmartPtr<MSWriter>               itsWriter;
      SmartPtr<Thread>                 itsThread;
    };


  } // namespace Cobalt
} // namespace LOFAR

#endif

