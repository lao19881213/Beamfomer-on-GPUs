//# plotMS.cc
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
//# $Id: plotMS.cc 25662 2013-07-13 19:10:59Z mol $

//# Always #include <lofar_config.h> first!
#include <lofar_config.h>

#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include <string>
#include <vector>
#include <boost/format.hpp>

#include <Common/LofarTypes.h>
#include <Common/LofarLogger.h>
#include <Common/Exception.h>
#include <Common/StringUtil.h>
#include <Common/DataConvert.h>
#include <Stream/FileStream.h>
#include <CoInterface/Parset.h>
#include <CoInterface/DataFactory.h>
#include <CoInterface/CorrelatedData.h>

#include <casa/IO/AipsIO.h>
#include <casa/Containers/Block.h>
#include <casa/Containers/BlockIO.h>

using namespace LOFAR;
using namespace LOFAR::Cobalt;
using namespace std;

using boost::format;

// Use a terminate handler that can produce a backtrace.
Exception::TerminateHandler t(Exception::terminate);

bool shouldSwap = false;

float power( fcomplex s )
{
  float r = real(s);
  float i = imag(s);

  if (shouldSwap) {
    byteSwap32(&r);
    byteSwap32(&i);
  }

  return r * r + i * i;
}

static void usage(char *progname, int exitcode)
{
  printf("Usage: %s -p parset [-b baseline | -B station1-station2] [-c channel]\n", progname);
  printf("\n");
  printf("Run within the MS directory of the subband to plot.\n");
  exit(exitcode);
}

int main(int argc, char *argv[])
{
#if defined HAVE_LOG4CPLUS
  INIT_LOGGER(string(getenv("LOFARROOT") ? : ".") + "/etc/Storage.log_prop");
#elif defined HAVE_LOG4CXX
  #error LOG4CXX support is broken (nonsensical?) -- please fix this code if you want to use it
  Context::initialize();
  setLevel("Global",8);
#else
  INIT_LOGGER_WITH_SYSINFO(str(boost::format("Storage@%02d") % (argc > 1 ? atoi(argv[1]) : -1)));
#endif

  try {
    int opt;
    const char *parset_filename = 0;
    const char *table_filename = "table.f0data";
    const char *meta_filename = "table.f0meta";
    const char *baselinestr = 0;
    unsigned baseline = 0;
    int channel = -1;

    while ((opt = getopt(argc, argv, "p:b:B:c:")) != -1) {
      switch (opt) {
      case 'p':
        parset_filename = strdup(optarg);
        break;

      case 'b':
        baseline = atoi(optarg);
        break;

      case 'B':
        baselinestr = strdup(optarg);
        break;

      case 'c':
        channel = atoi(optarg);
        break;

      default:   /* '?' */
        usage(argv[0], 1);
      }
    }

    if (!parset_filename)
      usage(argv[0], 1);

    Parset parset(parset_filename);
    FileStream datafile(table_filename);
    CorrelatedData *data = dynamic_cast<CorrelatedData*>(newStreamableData(parset, CORRELATED_DATA, 0));

    if (channel == -1)
      channel = parset.nrChannelsPerSubband() == 1 ? 0 : 1;  // default to first useful channel

    ASSERT( data );
    ASSERT( channel >= 0 && (unsigned)channel < parset.nrChannelsPerSubband() );

    // determine base line from string
    casa::Block<int32> itsAnt1;
    casa::Block<int32> itsAnt2;

    casa::AipsIO aio(meta_filename);
    uint32 itsVersion = aio.getstart("LofarStMan");
    if (itsVersion == 2) {
      aio >> itsAnt2 >> itsAnt1;
    } else {
      aio >> itsAnt1 >> itsAnt2;
    }
    aio.close();

    printf("# MS version %d\n", itsVersion);

    std::vector<std::string> stationNames = parset.allStationNames();

    if (baselinestr) {
      std::vector<std::string> specified_stations = StringUtil::split(string(baselinestr), '-');
      ASSERTSTR( specified_stations.size() == 2, "-B: Specify as STATION1-STATION2, not " << baselinestr );

      unsigned station1index = std::find(stationNames.begin(),stationNames.end(),specified_stations[0]) - stationNames.begin();
      unsigned station2index = std::find(stationNames.begin(),stationNames.end(),specified_stations[1]) - stationNames.begin();

      ASSERTSTR( station1index < stationNames.size(), "Could not find station " << specified_stations[0] );
      ASSERTSTR( station2index < stationNames.size(), "Could not find station " << specified_stations[1] );

      for (baseline = 0; baseline < itsAnt1.size(); baseline++) {
        if ((unsigned)itsAnt1[baseline] == station1index
            && (unsigned)itsAnt2[baseline] == station2index)
          break;

        if ((unsigned)itsAnt2[baseline] == station1index
            && (unsigned)itsAnt1[baseline] == station2index)
          break;
      }
    }

    ASSERTSTR( baseline < parset.nrBaselines(), "The specified baseline is not present in this measurement set." );

    std::string firstStation = stationNames[itsAnt1[baseline]];
    std::string secondStation = stationNames[itsAnt2[baseline]];

    printf( "# baseline %s - %s channel %d\n", firstStation.c_str(), secondStation.c_str(), channel);
    printf( "# observation %u\n", parset.observationID());

    for(;; ) {
      try {
        data->read(&datafile, true, 512);
      } catch (Stream::EndOfStreamException &) {
        break;
      }
      //data->peerMagicNumber = 0xda7a0000; // fake wrong endianness to circumvent bug
      shouldSwap = data->shouldByteSwap();

      printf( "# valid samples: %u\n", data->getNrValidSamples(baseline,channel));

      printf( "%6d %10g %10g %10g %10g\n",
              data->sequenceNumber(),
              power( data->visibilities[baseline][channel][0][0] ),
              power( data->visibilities[baseline][channel][0][1] ),
              power( data->visibilities[baseline][channel][1][0] ),
              power( data->visibilities[baseline][channel][1][1] ) );

    }

  } catch (LOFAR::Exception &ex) {
    LOG_FATAL_STR("[obs unknown] Caught LOFAR Exception: " << ex);
    return 1;
  } catch (casa::AipsError& ex) {
    LOG_FATAL_STR("[obs unknown] Caught Aips Error: " << ex.what());
    return 1;
  }

  return 0;
}

