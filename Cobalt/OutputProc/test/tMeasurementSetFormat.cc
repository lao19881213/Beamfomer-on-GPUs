//# tMeasurementSetFormat.cc: Test program for class MeasurementSetFormat
//# Copyright (C) 2011-2013  ASTRON (Netherlands Institute for Radio Astronomy)
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
//# $Id: tMeasurementSetFormat.cc 25931 2013-08-05 13:07:35Z klijn $

#include <lofar_config.h>

#include <string>

#include <Common/LofarLogger.h>
#include <Common/Exception.h>
#include <OutputProc/MeasurementSetFormat.h>

#include <casa/IO/RegularFileIO.h>

using namespace LOFAR;
using namespace LOFAR::Cobalt;
using namespace casa;
using namespace std;

// Define handler that tries to print a backtrace.
Exception::TerminateHandler t(Exception::terminate);

int main()
{
  INIT_LOGGER("tMeasurementSetFormat");
  const string suffixes[] = { "-j2000", "-sun" };

  for( unsigned i = 0; i < sizeof suffixes / sizeof suffixes[0]; i++ ) 
  {
      const string parsetName = string("tMeasurementSetFormat.parset") + suffixes[i];
      const string msName = string("tMeasurementSetFormat_tmp") + suffixes[i] + ".MS";

      LOG_DEBUG_STR("Testing " << parsetName);

      Parset parset(parsetName);
      MeasurementSetFormat msf(parset);
      msf.addSubband(msName, 0, false);
      // Also create the data file, otherwise it is not a true table.
      ///FILE* file= fopen ("tMeasurementSetFormat_tmp.ms/f0data", "w");
      ///fclose (file);
      RegularFileIO file(String(msName + "/table.f0data"),
                         ByteIO::New);
  }

  return 0;
}

