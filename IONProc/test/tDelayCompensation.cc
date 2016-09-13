//#  tDelayCompensation.cc: stand-alone test program for Delays
//#
//#  Copyright (C) 2009
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
#include <Interface/RSPTimeStamp.h>
#include <Common/lofar_math.h>

#include <cassert>

using namespace LOFAR;
using namespace LOFAR::RTCP;

double lat( casa::MVDirection &d )
{
  return d.get()[1];
}

double lng( casa::MVDirection &d )
{
  return d.get()[0];
}

void doTest()
{
  unsigned psetNumber = 0;

  Parset parset("tDelayCompensation.parset");

  std::vector<Parset::StationRSPpair> inputs = parset.getStationNamesAndRSPboardNumbers(psetNumber);

  unsigned nrBeams    = parset.nrBeams();
  double   startTime  = parset.startTime();
  double   sampleFreq = parset.subbandBandwidth();
  unsigned seconds    = static_cast<unsigned>(floor(startTime));
  unsigned samples    = static_cast<unsigned>((startTime - floor(startTime)) * sampleFreq);

  TimeStamp ts = TimeStamp(seconds, samples, parset.clockSpeed());

  Delays w(parset, inputs[0].station, ts);
  w.start();

  unsigned nrTABs = 0;
  Matrix<double> delays(nrBeams, nrTABs + 1);
  Matrix<casa::MVDirection> prev_directions(nrBeams, nrTABs + 1), directions(nrBeams, nrTABs + 1);
 
  for (unsigned i = 0; i < 256; i ++) {
    prev_directions = directions;

    w.getNextDelays(directions, delays);
    cout << "Directions & Delay: (" << lng(directions[0][0]) << ", " << lat(directions[0][0]) << "), " << delays[0][0] << endl;

    assert(!isnan(delays[0][0]));

    // source (NCP) should traverse with decreasing longitude and latitude
    if (i > 0) {
      assert(lng(directions[0][0]) < lng(prev_directions[0][0]));
      assert(lat(directions[0][0]) < lat(prev_directions[0][0]));
    }
  }
}


int main()
{
  try {
    doTest();
  } catch (Exception &ex) {
    std::cerr << "Caught Exception: " << ex.what() << std::endl;
    return 1;
  } catch (std::exception &ex) {
    std::cerr << "Caught std::exception: " << ex.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Caught unknown exception" << std::endl;
    return 1;
  }

  return 0;
}
