//#  Copyright (C) 2012
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

//# Always #include <lofar_config.h> first!
#include <lofar_config.h>

#include <Delays.h>
#include <Interface/Parset.h>
#include <Common/Exception.h>
#include <Common/LofarLogger.h>
#include <Interface/RSPTimeStamp.h>
#include <Common/lofar_math.h>
#include <Common/Exception.h>

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>

#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <cstdio>
#include <cstring>

using namespace LOFAR;
using namespace LOFAR::RTCP;
using namespace std;

// Use a terminate handler that can produce a backtrace.
Exception::TerminateHandler t(Exception::terminate);

bool print_tabs = false;
bool ascii_ts = false;

void generateDelays( const string &parsetFilename, const string &station )
{
  Parset parset(parsetFilename);

  unsigned nrBeams    = parset.nrBeams();
  unsigned maxNrTABs = parset.maxNrTABs();
  vector<unsigned> nrTABs = parset.nrTABs();

  double   startTime  = parset.startTime();
  double   stopTime   = parset.stopTime();
  double   sampleFreq = parset.subbandBandwidth();
  unsigned samplesPerBlock = parset.nrSamplesPerSubband();
  double   blockSize  = parset.CNintegrationTime();
  unsigned nrBlocks   = static_cast<unsigned>(floor((stopTime - startTime) / blockSize));

  unsigned seconds    = static_cast<unsigned>(floor(startTime));
  unsigned samples    = static_cast<unsigned>((startTime - floor(startTime)) * sampleFreq);

  TimeStamp ts = TimeStamp(seconds, samples, parset.clockSpeed());

  Delays w(parset, station, ts);
  w.start();

  Matrix<double> delays(nrBeams, maxNrTABs + 1);
  Matrix<casa::MVDirection> prev_directions(nrBeams, maxNrTABs + 1), directions(nrBeams, maxNrTABs + 1);

  for( unsigned block = 0; block < nrBlocks; block++ ) {
    w.getNextDelays(directions, delays);
    struct timespec spec = ts;

    if (ascii_ts) {
      time_t seconds =  spec.tv_sec;

      char buf[26];
      ctime_r(&seconds, buf);
      buf[strlen(buf) - 1] = 0; // remove trailing \n

      cout << buf << " sample " << ts.getBlockId() << " delay ";
    } else {
      double seconds = 1.0 * spec.tv_sec + spec.tv_nsec / 1.0e9;
      cout << fixed << setprecision(9) << seconds << " delay ";
    }   

    for( unsigned beam = 0; beam < nrBeams; beam++ ) {
      unsigned nr_delays = print_tabs ? nrTABs[beam] + 1 : 1;

      for( unsigned pencil = 0; pencil < nr_delays; pencil++ )
        cout << fixed << setprecision(12) << delays[beam][pencil] << " ";
    }    

    cout << endl;

    ts += samplesPerBlock;
  }
}

void print_usage()
{
  cout << "Usage: generateDelays -p parset -s station [-t]" << endl;
  cout << endl;
  cout << "-p parset      The filename of the parset to use" << endl;
  cout << "-s station     The name of the station (f.e. UK608HBA)" << endl;
  cout << "-t             Print delays for the tied-array beams" << endl;
  cout << "-a             Print ASCII timestamps" << endl;
}


int main(int argc, char *argv[])
{
  INIT_LOGGER("generateDelays");

  try {
    int opt;

    char *parset = 0;
    char *station = 0;

    while ((opt = getopt(argc, argv, "p:s:ta")) != -1) {
      switch (opt) {
        case 'p':
          parset = strdup(optarg);
          break;

        case 's':
          station = strdup(optarg);
          break;

        case 't':
          print_tabs = true;
          break;

        case 'a':
          ascii_ts = true;
          break;

        default:
          print_usage();
          exit(1);
      }
    }

    if (!parset || !station) {
      print_usage();
      exit(1);
    }

    if (!Casacore_Init()) {
      cerr << "Casacore subsystem init failed." << endl;
      exit(1);
    }

    generateDelays(parset, station);

    free(station);
    free(parset);
  } catch (Exception &ex) {
    cerr << "Caught Exception: " << ex << endl;
    return 1;
  }

  return 0;
}
