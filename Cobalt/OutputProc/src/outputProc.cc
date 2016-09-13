//# outputProc.cc
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
//# $Id: outputProc.cc 27371 2013-11-12 14:55:11Z loose $

//# Always #include <lofar_config.h> first!
#include <lofar_config.h>

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <sys/select.h>
#include <unistd.h>
#include <libgen.h>

#include <string>
#include <vector>
#include <stdexcept>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>

#include <Common/LofarLogger.h>
#include <Common/CasaLogSink.h>
#include <Common/StringUtil.h>
#include <Common/Exceptions.h>
#include <Common/NewHandler.h>
#include <Common/Thread/Thread.h>
#include <ApplCommon/Observation.h>
#include <ApplCommon/PVSSDatapointDefs.h>
#include <ApplCommon/StationInfo.h>
#include <Stream/PortBroker.h>
#include <CoInterface/Exceptions.h>
#include <CoInterface/Parset.h>
#include <CoInterface/Stream.h>
#include <CoInterface/FinalMetaData.h>
#include <OutputProc/Package__Version.h>
#include "SubbandWriter.h"
#include "IOPriority.h"

// install a new handler to produce backtraces for bad_alloc
LOFAR::NewHandler h(LOFAR::BadAllocException::newHandler);

using namespace LOFAR;
using namespace LOFAR::Cobalt;
using namespace std;
using boost::format;

// Use a terminate handler that can produce a backtrace.
Exception::TerminateHandler t(Exception::terminate);

char stdoutbuf[1024], stderrbuf[1024];

int main(int argc, char *argv[])
{
#if defined HAVE_LOG4CPLUS
  INIT_LOGGER("outputProc");
#else
  INIT_LOGGER_WITH_SYSINFO(str(boost::format("OutputProc@%02d") % (argc > 2 ? atoi(argv[2]) : -1)));
#endif

  CasaLogSink::attach();

  string obsLogPrefix = "[obs unknown] ";

  try {
    if (argc != 4)
      throw StorageException(str(boost::format("usage: %s obsid rank is_bigendian") % argv[0]), THROW_ARGS);

    setvbuf(stdout, stdoutbuf, _IOLBF, sizeof stdoutbuf);
    setvbuf(stderr, stderrbuf, _IOLBF, sizeof stderrbuf);

    LOG_DEBUG_STR("Started: " << argv[0] << ' ' << argv[1] << ' ' << argv[2] << ' ' << argv[3]);

    int observationID = boost::lexical_cast<int>(argv[1]);
    unsigned myRank = boost::lexical_cast<unsigned>(argv[2]);
    bool isBigEndian = boost::lexical_cast<bool>(argv[3]);

    setIOpriority();
    setRTpriority();
    lockInMemory();

    PortBroker::createInstance(storageBrokerPort(observationID));

    // retrieve the parset
    string resource = getStorageControlDescription(observationID, myRank);
    PortBroker::ServerStream controlStream(resource);

    Parset parset(&controlStream);

    // Send identification string to the MAC Log Processor
    LOG_INFO_STR("MACProcessScope: " << 
                 str(format(createPropertySetName(
                              PSN_COBALT_OUTPUT_PROC, "", 
                              parset.getString("_DPname")))
                     % myRank));

    Observation obs(&parset, false, 64); // FIXME: assume 64 psets, because Observation still deals with BG/P

    const vector<string> &hostnames = parset.settings.outputProcHosts;
    ASSERT(myRank < hostnames.size());
    string myHostName = hostnames[myRank];

    obsLogPrefix = str(boost::format("[obs %u] ") % parset.observationID());

    {
      // make sure "parset" stays in scope for the lifetime of the SubbandWriters

      vector<SmartPtr<SubbandWriter> > subbandWriters;

      for (OutputType outputType = FIRST_OUTPUT_TYPE; outputType < LAST_OUTPUT_TYPE; outputType++) {
        for (unsigned streamNr = 0; streamNr < parset.nrStreams(outputType); streamNr++) {
          if (parset.getHostName(outputType, streamNr) == myHostName) {
            string sbLogPrefix = str(boost::format("[obs %u type %u stream %3u] ") % parset.observationID() % outputType % streamNr);

            try {
              subbandWriters.push_back(new SubbandWriter(parset, outputType, streamNr, isBigEndian, sbLogPrefix));
            } catch (Exception &ex) {
              LOG_WARN_STR(sbLogPrefix << "Could not create writer: " << ex);
            } catch (exception &ex) {
              LOG_WARN_STR(sbLogPrefix << "Could not create writer: " << ex.what());
            }
          }
        }
      }

      /*
       * FINAL META DATA
       */
      // Add final meta data (broken tile information, etc)
      // that is obtained after the end of an observation.
      LOG_INFO_STR(obsLogPrefix << "Waiting for final meta data");
      FinalMetaData finalMetaData;
      finalMetaData.read(controlStream);

      LOG_INFO_STR(obsLogPrefix << "Processing final meta data");
      for (size_t i = 0; i < subbandWriters.size(); ++i)
        try {
          subbandWriters[i]->augment(finalMetaData);
        } catch (Exception &ex) {
          LOG_WARN_STR(obsLogPrefix << "Could not add final meta data: " << ex);
        }

      /*
       * LTA FEEDBACK
       */
      LOG_INFO_STR(obsLogPrefix << "Retrieving LTA feedback");
      Parset feedbackLTA;
      for (size_t i = 0; i < subbandWriters.size(); ++i)
        try {
          feedbackLTA.adoptCollection(subbandWriters[i]->feedbackLTA());
        } catch (Exception &ex) {
          LOG_WARN_STR(obsLogPrefix << "Could not obtain feedback for LTA: " << ex);
        }

      LOG_INFO_STR(obsLogPrefix << "Forwarding LTA feedback");
      feedbackLTA.write(&controlStream);
    }
  } catch (Exception &ex) {
    LOG_FATAL_STR(obsLogPrefix << "Caught Exception: " << ex);
    return 1;
  }

  LOG_INFO_STR(obsLogPrefix << "Program end");
  return 0;
}

