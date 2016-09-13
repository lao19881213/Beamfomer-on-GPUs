//#  tMeasurementSetFormat.cc: Test program for class MeasurementSetFormat
//#
//#  Copyright (C) 2011
//#  ASTRON (Netherlands Foundation for Research in Astronomy)
//#  P.O.Box 2, 7990 AA Dwingeloo, The Netherlands, seg@astron.nl
//#
//#  $Id: tMeasurementSetFormat.cc 21558 2012-07-12 09:35:39Z loose $

#include <lofar_config.h>
#include <Storage/MeasurementSetFormat.h>
#include <Common/LofarLogger.h>
#include <Common/Exception.h>

#include <casa/IO/RegularFileIO.h>

using namespace LOFAR;
using namespace LOFAR::RTCP;
using namespace casa;
using namespace std;

// Define handler that tries to print a backtrace.
Exception::TerminateHandler t(Exception::terminate);

int main()
{
  const string suffixes[] = { "-j2000", "-sun" };

  for( unsigned i = 0; i < sizeof suffixes / sizeof suffixes[0]; i++ ) {
    try {
      const string parsetName = string("tMeasurementSetFormat.parset") + suffixes[i];
      const string msName     = string("tMeasurementSetFormat") + suffixes[i] + "_tmp.ms";

      LOG_DEBUG_STR("Testing " << parsetName);

      Parset parset(parsetName.c_str());
      MeasurementSetFormat msf(parset);
      msf.addSubband(msName, 0, false);
      // Also create the data file, otherwise it is not a true table.
      ///FILE* file= fopen ("tMeasurementSetFormat_tmp.ms/f0data", "w");
      ///fclose (file);
      RegularFileIO file(String(msName+"/table.f0data"),
                         ByteIO::New);
    } catch (LOFAR::Exception &err) {
      std::cerr << "LOFAR Exception detected: " << err << std::endl;
      return 1;
    }
  }

  return 0;
}
