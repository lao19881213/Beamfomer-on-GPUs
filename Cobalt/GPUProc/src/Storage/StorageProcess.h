//# StorageProcess.h
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
//# $Id: StorageProcess.h 25646 2013-07-12 09:58:39Z mol $

#ifndef LOFAR_GPUPROC_STORAGE_PROCESS
#define LOFAR_GPUPROC_STORAGE_PROCESS

#include <sys/time.h>
#include <string>

#include <Common/Thread/Thread.h>
#include <Common/Thread/Trigger.h>
#include <CoInterface/Parset.h>
#include <CoInterface/SmartPtr.h>
#include <CoInterface/FinalMetaData.h>

namespace LOFAR
{
  namespace Cobalt
  {

    /* A single Storage process.
     *
     * Storage is started as:
     *     Storage_main observationID rank isBigEndian
     *
     * A Storage process is expected to follow the following protocol:
     *
       // establish control connection
       std::string resource = getStorageControlDescription(observationID, rank);
       PortBroker::ServerStream stream(resource);

       // read parset
       Parset parset(&stream);

       ... process observation ...

       // read meta data
       FinalMetaData finalMetaData;
       finalMetaData.read(stream);
     */

    class StorageProcess
    {
    public:
      // user must call start()
      StorageProcess( const Parset &parset, const std::string &logPrefix, int rank, const std::string &hostname, FinalMetaData &finalMetaData, Trigger &finalMetaDataAvailable );

      // calls stop(0)
      ~StorageProcess();

      void start();
      void stop( struct timespec deadline );
      bool isDone() const;

      // Returns feedback for the LTA -- only access this once the
      // StorageProcess has finished!
      ParameterSet feedbackLTA() const;

    private:
      void                               controlThread();

      const Parset &itsParset;
      const std::string itsLogPrefix;

      const int itsRank;
      const std::string itsHostname;

      FinalMetaData                      &itsFinalMetaData;
      Trigger                            &itsFinalMetaDataAvailable;

      ParameterSet                       itsFeedbackLTA;

      SmartPtr<Thread> itsThread;
    };

  }
}

#endif

