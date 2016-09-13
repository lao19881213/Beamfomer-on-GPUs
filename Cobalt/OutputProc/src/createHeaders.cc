//# createHeaders.cc: Generates all .h5/.MS files given a (OLAP) parset
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
//# $Id: createHeaders.cc 25646 2013-07-12 09:58:39Z mol $

//# Always #include <lofar_config.h> first!
#include <lofar_config.h>

#include <cstdlib>
#include <string>
#include <iostream>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>

#include <Common/LofarLogger.h>
#include <Common/CasaLogSink.h>
#include <Common/Exceptions.h>
#include <Common/Thread/Queue.h>
#include <CoInterface/Exceptions.h>
#include <CoInterface/OutputTypes.h>
#include <CoInterface/Parset.h>
#include <CoInterface/StreamableData.h>
#include <CoInterface/SmartPtr.h>
#include <OutputProc/Package__Version.h>
#include "OutputThread.h"

using namespace LOFAR;
using namespace LOFAR::Cobalt;
using namespace std;

// Use a terminate handler that can produce a backtrace.
Exception::TerminateHandler t(Exception::terminate);

int main(int argc, char *argv[])
{
  bool isBigEndian = true;

  if (argc < 2 || argc > 3) {
    cout << str(boost::format("usage: %s parset [is_bigendian]") % argv[0]) << endl;
    cout << endl;
    cout << "parset: the filename of the parset to convert (parset must have been produced by RTCP/Run/src/LOFAR/Parset.py, aka an 'OLAP parset')." << endl;
    cout << "is_bigendian: 1 if data is written big endian (f.e. data comes from the BlueGene/P), 0 otherwise. Default: " << (int)isBigEndian << endl;
    return 1;
  }

#if defined HAVE_LOG4CPLUS
  INIT_LOGGER(string(getenv("LOFARROOT") ? : ".") + "/etc/createHeaders.log_prop");
#elif defined HAVE_LOG4CXX
  #error LOG4CXX support is broken (nonsensical?) -- please fix this code if you want to use it
  Context::initialize();
  setLevel("Global",8);
#else
  INIT_LOGGER_WITH_SYSINFO("createHeaders");
#endif

  CasaLogSink::attach();

  ParameterSet feedbackLTA;

  try {
    Parset parset(argv[1]);
    if (argc > 2) isBigEndian = boost::lexical_cast<bool>(argv[2]);

    for (OutputType outputType = FIRST_OUTPUT_TYPE; outputType < LAST_OUTPUT_TYPE; outputType++) {
      const unsigned nrStreams = parset.nrStreams(outputType);

      for (unsigned streamNr = 0; streamNr < nrStreams; streamNr++) {
        const string logPrefix = str(boost::format("[obs %u type %u stream %3u] ") % parset.observationID() % outputType % streamNr);

        try {
          // a dummy queue
          Queue<SmartPtr<StreamableData> > queue;

          OutputThread ot(parset, outputType, streamNr, queue, queue, logPrefix, isBigEndian, ".");

          // create measurement set
          ot.createMS();

          // wrap up
          ot.cleanUp();

          // obtain LTA feedback
          feedbackLTA.adoptCollection(ot.feedbackLTA());
        } catch (Exception &ex) {
          LOG_WARN_STR(logPrefix << "Could not create header: " << ex);
        } catch (exception &ex) {
          LOG_WARN_STR(logPrefix << "Could not create header: " << ex.what());
        }
      }
    }

    // taken from IONProc/src/Job.cc
    // Augment the LTA feedback logging
    {
      feedbackLTA.add("Observation.DataProducts.nrOfOutput_Beamformed_", str(boost::format("%u") % parset.nrStreams(BEAM_FORMED_DATA)));
      feedbackLTA.add("Observation.DataProducts.nrOfOutput_Correlated_", str(boost::format("%u") % parset.nrStreams(CORRELATED_DATA)));

      for (ParameterSet::const_iterator i = feedbackLTA.begin(); i != feedbackLTA.end(); ++i)
        LOG_INFO_STR("[obs " << parset.observationID() << "] LTA FEEDBACK: " << i->first << " = " << i->second);
    }  
  } catch (Exception &ex) {
    LOG_FATAL_STR("[obs unknown] Caught Exception: " << ex);
    return 1;
  }

  LOG_INFO_STR("[obs unknown] Program end");
  return 0;
}

