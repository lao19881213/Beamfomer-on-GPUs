/* tDelays.cc
 * Copyright (C) 2013  ASTRON (Netherlands Institute for Radio Astronomy)
 * P.O. Box 2, 7990 AA Dwingeloo, The Netherlands
 *
 * This file is part of the LOFAR software suite.
 * The LOFAR software suite is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * The LOFAR software suite is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the LOFAR software suite. If not, see <http://www.gnu.org/licenses/>.
 *
 * $Id: tDelays.cc 25660 2013-07-13 13:11:46Z mol $
 */

#include <lofar_config.h>

#include <ctime>
#include <UnitTest++.h>

#include <Common/LofarLogger.h>
#include <Stream/FixedBufferStream.h>

#include <InputProc/Delays/Delays.h>

using namespace LOFAR;
using namespace Cobalt;
using namespace std;

const size_t dayOfSamples = 24UL * 3600 * 192315;

TEST(Tracking) {
  Parset ps;

  ps.add( "Observation.referencePhaseCenter", "[0, 0, 0]" ); // center of earth
  ps.add( "PIC.Core.CS001LBA.phaseCenter", "[0, 0, 299792458]" ); // 1 lightsecond away from earth center
  ps.add( "Observation.VirtualInstrument.stationList", "[CS001]" );
  ps.add( "Observation.antennaSet", "LBA_INNER" );

  ps.add( "Observation.nrBeams", "1" );
  ps.add( "Observation.Beam[0].directionType", "J2000" );
  ps.add( "Observation.Beam[0].angle1", "0" );
  ps.add( "Observation.Beam[0].angle2", "0" );
  ps.add( "Observation.Beam[0].nrTiedArrayBeams", "0" );
  ps.updateSettings();

  // blockSize is ~1s
  Delays delays(ps, 0, TimeStamp(time(0), 0, 200000000), dayOfSamples);
  delays.start();

  Delays::AllDelays delaySet(ps), prevDelaySet(ps);

  for (size_t block = 0; block < 1024; ++block) {
    delays.getNextDelays(delaySet);

    // There must be exactly one SAP
    CHECK_EQUAL(1U, delaySet.SAPs.size());
    CHECK_EQUAL(0U, delaySet.SAPs[0].TABs.size());

#ifdef HAVE_CASACORE
    // Delays must change over time
    if (block > 0) {
      CHECK(delaySet.SAPs[0].SAP.delay != prevDelaySet.SAPs[0].SAP.delay);
    }
#endif

    prevDelaySet = delaySet;
  }
}

TEST(TiedArrayBeam) {
  Parset ps;

  ps.add( "Observation.DataProducts.Output_Beamformed.enabled", "true" );
  ps.add( "Observation.DataProducts.Output_Beamformed.filenames", "[beam0.raw]" );
  ps.add( "Observation.DataProducts.Output_Beamformed.locations", "[localhost:.]" );

  ps.add( "Observation.referencePhaseCenter", "[0, 0, 0]" ); // center of earth
  ps.add( "PIC.Core.CS001LBA.phaseCenter", "[0, 0, 299792458]" ); // 1 lightsecond away from earth center
  ps.add( "Observation.VirtualInstrument.stationList", "[CS001]" );
  ps.add( "Observation.antennaSet", "LBA_INNER" );

  // Delays for SAP 0 and TAB 0 of SAP 1 should be equal
  ps.add( "Observation.nrBeams", "2" );
  ps.add( "Observation.Beam[0].directionType", "J2000" );
  ps.add( "Observation.Beam[0].angle1", "1" );
  ps.add( "Observation.Beam[0].angle2", "1" );
  ps.add( "Observation.Beam[0].nrTiedArrayBeams", "0" );
  ps.add( "Observation.Beam[1].directionType", "J2000" );
  ps.add( "Observation.Beam[1].angle1", "1" );
  ps.add( "Observation.Beam[1].angle2", "0" );
  ps.add( "Observation.Beam[1].nrTiedArrayBeams", "1" );
  ps.add( "Observation.Beam[1].TiedArrayBeam[0].directionType", "J2000" );
  ps.add( "Observation.Beam[1].TiedArrayBeam[0].angle1", "0" );
  ps.add( "Observation.Beam[1].TiedArrayBeam[0].angle2", "1" );
  ps.updateSettings();

  // blockSize is ~1s
  Delays delays(ps, 0, TimeStamp(time(0), 0, 200000000), dayOfSamples);
  delays.start();

  Delays::AllDelays delaySet(ps);

  for (size_t block = 0; block < 10; ++block) {
    delays.getNextDelays(delaySet);

    // check dimensions of result
    CHECK_EQUAL(2U, delaySet.SAPs.size());
    CHECK_EQUAL(0U, delaySet.SAPs[0].TABs.size());
    CHECK_EQUAL(1U, delaySet.SAPs[1].TABs.size());

    // check values
    CHECK_CLOSE(delaySet.SAPs[0].SAP.delay, delaySet.SAPs[1].TABs[0].delay, 0.00001);
  }
}


TEST(AllDelayIO) {
  Parset ps;

  ps.add( "Observation.DataProducts.Output_Beamformed.enabled", "true" );
  ps.add( "Observation.DataProducts.Output_Beamformed.filenames", "[beam0.raw]" );
  ps.add( "Observation.DataProducts.Output_Beamformed.locations", "[localhost:.]" );
  ps.add( "Observation.nrBeams", "2" );
  ps.add( "Observation.Beam[0].directionType", "J2000" );
  ps.add( "Observation.Beam[0].angle1", "1" );
  ps.add( "Observation.Beam[0].angle2", "1" );
  ps.add( "Observation.Beam[0].nrTiedArrayBeams", "0" );
  ps.add( "Observation.Beam[1].directionType", "J2000" );
  ps.add( "Observation.Beam[1].angle1", "1" );
  ps.add( "Observation.Beam[1].angle2", "0" );
  ps.add( "Observation.Beam[1].nrTiedArrayBeams", "1" );
  ps.add( "Observation.Beam[1].TiedArrayBeam[0].directionType", "J2000" );
  ps.add( "Observation.Beam[1].TiedArrayBeam[0].angle1", "0" );
  ps.add( "Observation.Beam[1].TiedArrayBeam[0].angle2", "1" );
  ps.updateSettings();

  Delays::AllDelays delaySet_in(ps), delaySet_out(ps);

  delaySet_in.SAPs[0].SAP.delay = 1.0;
  delaySet_in.SAPs[1].SAP.delay = 0.5;
  delaySet_in.SAPs[1].TABs[0].delay = 1.0;

  vector<char> buffer(1024);

  FixedBufferStream str_in(&buffer[0], buffer.size());
  FixedBufferStream str_out(&buffer[0], buffer.size());
 
  delaySet_in.write(&str_in);
  delaySet_out.read(&str_out);

  CHECK( delaySet_in == delaySet_out );
}


int main()
{
  INIT_LOGGER( "tDelays" );

  // Don't run forever if communication fails for some reason
  alarm(10);

  return UnitTest::RunAllTests() > 0;
}

