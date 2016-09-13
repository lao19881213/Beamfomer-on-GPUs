//# StorageProcesses.h
//# Copyright (C) 2012-2013  ASTRON (Netherlands Institute for Radio Astronomy)
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
//# $Id: StorageProcesses.h 26956 2013-10-14 09:49:52Z mol $

#ifndef LOFAR_GPUPROC_STORAGE_PROCESSES
#define LOFAR_GPUPROC_STORAGE_PROCESSES

#include <sys/time.h>
#include <string>
#include <vector>

#include <Common/Thread/Thread.h>
#include <Common/Thread/Trigger.h>
#include <CoInterface/Parset.h>
#include <CoInterface/SmartPtr.h>
#include <CoInterface/FinalMetaData.h>

#include "StorageProcess.h"

namespace LOFAR
{
  namespace Cobalt
  {

    /*
     * Manage a set of StorageProcess objects. The control sequence is as follows:
     *
     * 1. StorageProcess() creates and starts the StorageProcess objects from the
     *    parset.
     * 2. ... process observation ...
     * 3. forwardFinalMetaData(deadline) starts the FinalMetaDataGatherer, reads the
     *    final meta data and forwards it to the StorageProcess objects.
     * 4. stop(deadline) stops the StorageProcesses with a termination period.
     *
     * FinalMetaDataGatherer is started as:
     *     FinalMetaDataGatherer observationID
     *
     * A Storage process is expected to follow the following protocol:
     *
       // establish control connection
       std::string resource = getStorageControlDescription(observationID, -1);
       PortBroker::ServerStream stream(resource);

       // read parset
       Parset parset(&stream);

       // write meta data
       FinalMetaData finalMetaData;
       finalMetaData.write(stream);
     */

    class StorageProcesses
    {
    public:
      // calls start()
      StorageProcesses( const Parset &parset, const std::string &logPrefix );

      // calls stop(0)
      ~StorageProcesses();

      // start the FinalMetaDataGatherer process and forward the obtained
      // meta data to the Storage processes. The deadline is an absolute time out.
      void forwardFinalMetaData( time_t deadline );

      // stop the processes and control threads, given an absolute time out.
      void stop( time_t deadline );

      ParameterSet feedbackLTA() const;

    private:
      const Parset                         &itsParset;
      const std::string itsLogPrefix;

      std::vector<SmartPtr<StorageProcess> > itsStorageProcesses;
      FinalMetaData itsFinalMetaData;
      Trigger itsFinalMetaDataAvailable;

      // All feedback for the LTA obtained by the storage processes
      ParameterSet itsFeedbackLTA;

      // start the processes and control threads
      void start();

      void finalMetaDataThread();
    };

  }
}

#endif

