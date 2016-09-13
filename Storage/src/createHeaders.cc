//#  createHeaders.cc: Generates all .h5/.MS files given a (OLAP) parset
//#
//#  Copyright (C) 2002-2004
//#  ASTRON (Netherlands Foundation for Research in Astronomy)
//#  P.O.Box 2, 7990 AA Dwingeloo, The Netherlands, seg@astron.nl
//#
//#  $Id: createHeaders.cc 21558 2012-07-12 09:35:39Z mol $

//# Always #include <lofar_config.h> first!
#include <lofar_config.h>

#include <Common/LofarLogger.h>
#include <Common/CasaLogSink.h>
#include <Common/Exceptions.h>
#include <Interface/Exceptions.h>
#include <Interface/Parset.h>
#include <Storage/Package__Version.h>
#include <Storage/OutputThread.h>

#include <string>

#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>

using namespace LOFAR;
using namespace LOFAR::RTCP;
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

  try {
    Parset parset(argv[1]);
    if (argc > 2) isBigEndian = boost::lexical_cast<bool>(argv[2]);

    for (OutputType outputType = FIRST_OUTPUT_TYPE; outputType < LAST_OUTPUT_TYPE; outputType ++) {
      const unsigned nrStreams = parset.nrStreams(outputType);

      for (unsigned streamNr = 0; streamNr < nrStreams; streamNr ++) {
        const string logPrefix = str(boost::format("[obs %u type %u stream %3u] ") % parset.observationID() % outputType % streamNr);

        try {
          // a dummy queue
          Queue<SmartPtr<StreamableData> > queue;

          OutputThread ot(parset, outputType, streamNr, queue, queue, logPrefix, isBigEndian, ".");

          // create measurement set
          ot.createMS();

          // output LTA feedback
          ot.cleanUp();
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
      ParameterSet feedbackLTA;
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
