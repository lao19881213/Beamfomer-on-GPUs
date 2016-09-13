//# tParset.cc
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
//# $Id: tParset.cc 26993 2013-10-16 15:43:48Z mol $

#include <lofar_config.h>

#include <Common/LofarLogger.h>
#include <CoInterface/Parset.h>

#include <UnitTest++.h>
#include <vector>
#include <string>
#include <sstream>
#include <boost/format.hpp>

using namespace LOFAR;
using namespace LOFAR::Cobalt;
using namespace std;
using boost::format;

// macro to create a Parset out of one key/value pair
#define MAKEPS(key, value) \
  Parset ps; \
  ps.add(key, value); \
  ps.updateSettings();

// macros for testing true/false keys
#define TESTKEYS(new, old) for ( string k = "x", keystr = new; k != "xxx"; k += "x", keystr = old)
#define TESTBOOL for( unsigned val = 0; val < 2; ++val )
#define valstr ((val) ? "true" : "false")

// generate a vector of zeroes
vector<unsigned> zeroes(size_t n) {
  return vector<unsigned>(n, 0);
}

// generate a vector 0,1,2..n
vector<unsigned> sequence(size_t n) {
  vector<unsigned> result(n);

  for (size_t i = 0; i < n; ++i) {
    result[i] = i;
  }

  return result;
}

// convert a vector to a string
template<typename T> string toStr( const vector<T> &v )
{
  stringstream sstr;

  sstr << v;

  return sstr.str();
}

/*
 * ===============================================
 * Test individual Parset fields through UnitTests
 * ===============================================
 */

/*
 * Test generic information.
 */

TEST(realTime) {
  TESTKEYS("Cobalt.realTime", "OLAP.realTime") {
    TESTBOOL {
      MAKEPS(keystr, valstr);

      CHECK_EQUAL(val, ps.settings.realTime);
      CHECK_EQUAL(val, ps.realTime());
    }
  }
}

TEST(observationID) {
  MAKEPS("Observation.ObsID", "12345");

  CHECK_EQUAL(12345U, ps.settings.observationID);
  CHECK_EQUAL(12345U, ps.observationID());
}

TEST(startTime) {
  MAKEPS("Observation.startTime", "2013-03-17 10:55:08");

  CHECK_CLOSE(1363517708.0, ps.settings.startTime, 0.1);
  CHECK_CLOSE(1363517708.0, ps.startTime(), 0.1);
}

TEST(stopTime) {
  MAKEPS("Observation.stopTime", "2013-03-17 10:55:08");

  CHECK_CLOSE(1363517708.0, ps.settings.stopTime, 0.1);
  CHECK_CLOSE(1363517708.0, ps.stopTime(), 0.1);
}

SUITE(clockMHz) {
  TEST(200) {
    MAKEPS("Observation.sampleClock", "200");

    CHECK_EQUAL(200U, ps.settings.clockMHz);
    CHECK_EQUAL(200000000U, ps.clockSpeed());

    CHECK_CLOSE(195312.5, ps.settings.subbandWidth(), 0.001);
    CHECK_CLOSE(195312.5, ps.subbandBandwidth(), 0.001);
    CHECK_CLOSE(1.0/195312.5, ps.sampleDuration(), 0.001);
  }

  TEST(160) {
    MAKEPS("Observation.sampleClock", "160");

    CHECK_EQUAL(160U, ps.settings.clockMHz);
    CHECK_EQUAL(160000000U, ps.clockSpeed());

    CHECK_CLOSE(156250.0, ps.settings.subbandWidth(), 0.001);
    CHECK_CLOSE(156250.0, ps.subbandBandwidth(), 0.001);
    CHECK_CLOSE(1.0/156250.0, ps.sampleDuration(), 0.001);
  }
}

SUITE(nrBitsPerSample) {
  TEST(16) {
    MAKEPS("Observation.nrBitsPerSample", "16");

    CHECK_EQUAL(16U, ps.settings.nrBitsPerSample);
    CHECK_EQUAL(16U, ps.nrBitsPerSample());
    CHECK_EQUAL(16U * 2 / 8, ps.nrBytesPerComplexSample());
  }

  TEST(8) {
    MAKEPS("Observation.nrBitsPerSample", "8");

    CHECK_EQUAL(8U, ps.settings.nrBitsPerSample);
    CHECK_EQUAL(8U, ps.nrBitsPerSample());
    CHECK_EQUAL(8U * 2 / 8, ps.nrBytesPerComplexSample());
  }

  TEST(4) {
    MAKEPS("Observation.nrBitsPerSample", "4");

    CHECK_EQUAL(4U, ps.settings.nrBitsPerSample);
    CHECK_EQUAL(4U, ps.nrBitsPerSample());
    CHECK_EQUAL(4U * 2 / 8, ps.nrBytesPerComplexSample());
  }
}

TEST(nrPolarisations) {
  size_t nPol = 2;

  MAKEPS("foo", "bar");

  CHECK_EQUAL(nPol,        ps.settings.nrPolarisations);
  CHECK_EQUAL(nPol * nPol, ps.settings.nrCrossPolarisations());
  CHECK_EQUAL(nPol * nPol, ps.nrCrossPolarisations());
}

SUITE(corrections) {
  TEST(bandPass) {
    TESTKEYS("Cobalt.correctBandPass", "OLAP.correctBandPass") {
      TESTBOOL {
        MAKEPS(keystr, valstr);

        CHECK_EQUAL(val, ps.settings.corrections.bandPass);
        CHECK_EQUAL(val, ps.correctBandPass());
      }
    }
  }

  TEST(clock) {
    TESTKEYS("Cobalt.correctClocks", "OLAP.correctClocks") {
      TESTBOOL {
        MAKEPS(keystr, valstr);

        CHECK_EQUAL(val, ps.settings.corrections.clock);
        CHECK_EQUAL(val, ps.correctClocks());
      }
    }
  }

  TEST(dedisperse) {
    TESTKEYS("Cobalt.BeamFormer.coherentDedisperseChannels", "OLAP.coherentDedisperseChannels") {
      TESTBOOL {
        MAKEPS(keystr, valstr);

        CHECK_EQUAL(val, ps.settings.corrections.dedisperse);
      }
    }
  }
}

SUITE(delayCompensation) {
  TEST(enabled) {
    TESTKEYS("Cobalt.delayCompensation", "OLAP.delayCompensation") {
      TESTBOOL {
        MAKEPS(keystr, valstr);

        CHECK_EQUAL(val, ps.settings.delayCompensation.enabled);
        CHECK_EQUAL(val, ps.delayCompensation());
      }
    }
  }

  TEST(referencePhaseCenter) {
      MAKEPS("Observation.referencePhaseCenter", "[1,2,3]");

      vector<double> refPhaseCenter(3);
      refPhaseCenter[0] = 1.0;
      refPhaseCenter[1] = 2.0;
      refPhaseCenter[2] = 3.0;

      CHECK_ARRAY_CLOSE(refPhaseCenter, ps.settings.delayCompensation.referencePhaseCenter, 3, 0.001);
  }
}

/*
 * Test station information.
 */

TEST(antennaSet) {
  vector<string> antennaSets;
  antennaSets.push_back("LBA_INNER");
  antennaSets.push_back("LBA_OUTER");
  antennaSets.push_back("HBA_ZERO");
  antennaSets.push_back("HBA_ONE");
  antennaSets.push_back("HBA_DUAL");
  antennaSets.push_back("HBA_JOINED");
  antennaSets.push_back("HBA_ZERO_INNER");
  antennaSets.push_back("HBA_ONE_INNER");
  antennaSets.push_back("HBA_DUAL_INNER");
  antennaSets.push_back("HBA_JOINED_INNER");

  for (vector<string>::iterator i = antennaSets.begin(); i != antennaSets.end(); ++i) {
    MAKEPS("Observation.antennaSet", *i);

    CHECK_EQUAL(*i, ps.settings.antennaSet);
    CHECK_EQUAL(*i, ps.antennaSet());
  }
}

TEST(bandFilter) {
  // bandFilter[filter] = nyquistZone
  map<string, unsigned> bandFilters;
  bandFilters["LBA_10_90"]   = 1;
  bandFilters["LBA_30_90"]   = 1;
  bandFilters["HBA_110_190"] = 2;
  bandFilters["HBA_170_230"] = 3;
  bandFilters["HBA_210_250"] = 3;

  for (map<string, unsigned>::iterator i = bandFilters.begin(); i != bandFilters.end(); ++i) {
    MAKEPS("Observation.bandFilter", i->first);

    CHECK_EQUAL(i->first, ps.settings.bandFilter);
    CHECK_EQUAL(i->first, ps.bandFilter());
    
    CHECK_EQUAL(i->second, ps.settings.nyquistZone());
  }
}

SUITE(antennaFields) {
  TEST(LBA) {
    vector<string> stations, expectedFields;
    stations.push_back("CS001");
    expectedFields.push_back("CS001LBA");
    stations.push_back("CS002");
    expectedFields.push_back("CS002LBA");
    stations.push_back("RS210");
    expectedFields.push_back("RS210LBA");
    stations.push_back("DE603");
    expectedFields.push_back("DE603LBA");

    vector<ObservationSettings::AntennaFieldName> antennaFields = ObservationSettings::antennaFields(stations, "LBA_INNER");

    CHECK_EQUAL(expectedFields.size(), antennaFields.size());

    for (size_t i = 0; i < std::min(expectedFields.size(), antennaFields.size()); ++i) {
      CHECK_EQUAL(expectedFields[i], antennaFields[i].fullName());
    }
  }

  TEST(HBA0) {
    vector<string> stations, expectedFields;
    stations.push_back("CS001");
    expectedFields.push_back("CS001HBA0");
    stations.push_back("CS002");
    expectedFields.push_back("CS002HBA0");
    stations.push_back("RS210");
    expectedFields.push_back("RS210HBA");
    stations.push_back("DE603");
    expectedFields.push_back("DE603HBA");

    vector<ObservationSettings::AntennaFieldName> antennaFields = ObservationSettings::antennaFields(stations, "HBA_ZERO");

    CHECK_EQUAL(expectedFields.size(), antennaFields.size());

    for (size_t i = 0; i < std::min(expectedFields.size(), antennaFields.size()); ++i) {
      CHECK_EQUAL(expectedFields[i], antennaFields[i].fullName());
    }
  }

  TEST(HBA1) {
    vector<string> stations, expectedFields;
    stations.push_back("CS001");
    expectedFields.push_back("CS001HBA1");
    stations.push_back("CS002");
    expectedFields.push_back("CS002HBA1");
    stations.push_back("RS210");
    expectedFields.push_back("RS210HBA");
    stations.push_back("DE603");
    expectedFields.push_back("DE603HBA");

    vector<ObservationSettings::AntennaFieldName> antennaFields = ObservationSettings::antennaFields(stations, "HBA_ONE");

    CHECK_EQUAL(expectedFields.size(), antennaFields.size());

    for (size_t i = 0; i < std::min(expectedFields.size(), antennaFields.size()); ++i) {
      CHECK_EQUAL(expectedFields[i], antennaFields[i].fullName());
    }
  }

  TEST(HBA_DUAL) {
    vector<string> stations, expectedFields;
    stations.push_back("CS001");
    expectedFields.push_back("CS001HBA0");
    expectedFields.push_back("CS001HBA1");
    stations.push_back("CS002");
    expectedFields.push_back("CS002HBA0");
    expectedFields.push_back("CS002HBA1");
    stations.push_back("RS210");
    expectedFields.push_back("RS210HBA");
    stations.push_back("DE603");
    expectedFields.push_back("DE603HBA");

    vector<ObservationSettings::AntennaFieldName> antennaFields = ObservationSettings::antennaFields(stations, "HBA_DUAL");

    CHECK_EQUAL(expectedFields.size(), antennaFields.size());

    for (size_t i = 0; i < std::min(expectedFields.size(), antennaFields.size()); ++i) {
      CHECK_EQUAL(expectedFields[i], antennaFields[i].fullName());
    }
  }

  TEST(HBA_JOINED) {
    vector<string> stations, expectedFields;
    stations.push_back("CS001");
    expectedFields.push_back("CS001HBA");
    stations.push_back("CS002");
    expectedFields.push_back("CS002HBA");
    stations.push_back("RS210");
    expectedFields.push_back("RS210HBA");
    stations.push_back("DE603");
    expectedFields.push_back("DE603HBA");

    vector<ObservationSettings::AntennaFieldName> antennaFields = ObservationSettings::antennaFields(stations, "HBA_JOINED");

    CHECK_EQUAL(expectedFields.size(), antennaFields.size());

    for (size_t i = 0; i < std::min(expectedFields.size(), antennaFields.size()); ++i) {
      CHECK_EQUAL(expectedFields[i], antennaFields[i].fullName());
    }
  }
}

SUITE(stations) {
  TEST(phaseCenter) {
    Parset ps;

    // set
    ps.add("Observation.VirtualInstrument.stationList", "[CS001]");
    ps.add("Observation.antennaSet", "LBA_INNER");
    ps.add("PIC.Core.CS001LBA.phaseCenter", "[1.0, 2.0, 3.0]");
    ps.updateSettings();

    // verify settings
    CHECK_EQUAL(3U, ps.settings.stations[0].phaseCenter.size());
    CHECK_CLOSE(1.0, ps.settings.stations[0].phaseCenter[0], 0.01);
    CHECK_CLOSE(2.0, ps.settings.stations[0].phaseCenter[1], 0.01);
    CHECK_CLOSE(3.0, ps.settings.stations[0].phaseCenter[2], 0.01);
  }

  TEST(default_map) {
    Parset ps;

    // add a station and default board/slot lists
    ps.add("Observation.VirtualInstrument.stationList", "[CS001]");
    ps.add("Observation.antennaSet", "LBA_INNER");
    ps.add("Observation.rspBoardList", "[1]");
    ps.add("Observation.rspSlotList",  "[2]");
    ps.updateSettings();

    // verify settings
    CHECK_EQUAL(1U, ps.settings.stations.size());
    CHECK_EQUAL(1U, ps.settings.stations[0].rspBoardMap.size());
    CHECK_EQUAL(1U, ps.settings.stations[0].rspBoardMap[0]);
    CHECK_EQUAL(1U, ps.settings.stations[0].rspSlotMap.size());
    CHECK_EQUAL(2U, ps.settings.stations[0].rspSlotMap[0]);
  }

  TEST(station_map) {
    Parset ps;

    // add a station and station-specific board/slot lists
    ps.add("Observation.VirtualInstrument.stationList", "[CS001]");
    ps.add("Observation.antennaSet", "LBA_INNER");
    ps.add("Observation.Dataslots.CS001LBA.RSPBoardList", "[1]");
    ps.add("Observation.Dataslots.CS001LBA.DataslotList", "[2]");
    ps.updateSettings();

    // verify settings
    CHECK_EQUAL(1U, ps.settings.stations.size());
    CHECK_EQUAL(1U, ps.settings.stations[0].rspBoardMap.size());
    CHECK_EQUAL(1U, ps.settings.stations[0].rspBoardMap[0]);
    CHECK_EQUAL(1U, ps.settings.stations[0].rspSlotMap.size());
    CHECK_EQUAL(2U, ps.settings.stations[0].rspSlotMap[0]);
  }
}

SUITE(SAPs) {
  TEST(nr) {
    Parset ps;

    for (size_t nrSAPs = 1; nrSAPs < 244; ++nrSAPs) {
      MAKEPS("Observation.nrBeams", str(format("%u") % nrSAPs));

      CHECK_EQUAL(nrSAPs, ps.settings.SAPs.size());
    }
  }

  TEST(target) {
    Parset ps;

    // set
    ps.add("Observation.nrBeams", "2");
    ps.add("Observation.Beam[0].target", "target 1");
    ps.add("Observation.Beam[1].target", "target 2");

    ps.updateSettings();

    // verify settings
    CHECK_EQUAL("target 1", ps.settings.SAPs[0].target);
    CHECK_EQUAL("target 2", ps.settings.SAPs[1].target);
  }

  TEST(direction) {
    Parset ps;

    // set
    ps.add("Observation.nrBeams", "1");
    ps.add("Observation.Beam[0].angle1", "1.0");
    ps.add("Observation.Beam[0].angle2", "2.0");
    ps.add("Observation.Beam[0].directionType", "AZEL");

    ps.updateSettings();

    // verify settings
    CHECK_CLOSE(1.0,    ps.settings.SAPs[0].direction.angle1, 0.1);
    CHECK_CLOSE(2.0,    ps.settings.SAPs[0].direction.angle2, 0.1);
    CHECK_EQUAL("AZEL", ps.settings.SAPs[0].direction.type);
  }
}

SUITE(anaBeam) {
  TEST(enabled) {
    TESTBOOL {
      MAKEPS("Observation.antennaSet", val ? "HBA_ZERO" : "LBA_INNER");

      CHECK_EQUAL(val, ps.settings.anaBeam.enabled);
    }
  }

  TEST(direction) {
    Parset ps;

    // set
    ps.add("Observation.antennaSet", "HBA_INNER");
    ps.add("Observation.AnaBeam[0].angle1", "1.0");
    ps.add("Observation.AnaBeam[0].angle2", "2.0");
    ps.add("Observation.AnaBeam[0].directionType", "AZEL");

    ps.updateSettings();

    // verify settings
    CHECK_CLOSE(1.0,    ps.settings.anaBeam.direction.angle1, 0.1);
    CHECK_CLOSE(2.0,    ps.settings.anaBeam.direction.angle2, 0.1);
    CHECK_EQUAL("AZEL", ps.settings.anaBeam.direction.type);
  }
}

SUITE(subbands) {
  TEST(nr) {
    for (size_t nrSubbands = 0; nrSubbands < 244; ++nrSubbands) {
      Parset ps;

      // add subbands
      ps.add("Observation.nrBeams", "1");
      ps.add("Observation.Beam[0].subbandList", str(format("[%u*42]") % nrSubbands));
      ps.updateSettings();

      // verify settings
      CHECK_EQUAL(nrSubbands, ps.settings.subbands.size());
    }
  }

  TEST(idx_stationIdx) {
    Parset ps;

    // set
    ps.add("Observation.nrBeams", "1");
    ps.add("Observation.Beam[0].subbandList", "[42]");
    ps.updateSettings();

    // verify settings
    CHECK_EQUAL(0U,  ps.settings.subbands[0].idx);
    CHECK_EQUAL(42U, ps.settings.subbands[0].stationIdx);
  }

  TEST(SAP) {
    Parset ps;

    // set -- note: for now, omitting actual SAP specifications is allowed
    ps.add("Observation.nrBeams", "2");
    ps.add("Observation.Beam[1].subbandList", "[1]");
    ps.updateSettings();

    // verify settings
    CHECK_EQUAL(1U, ps.settings.subbands[0].SAP);
  }

  TEST(centralFrequency) {
    // test for both 200 and 160 MHz clocks,
    // and for all three Nyquist zones.

    map<unsigned, string> bandFilters;
    bandFilters[1] = "LBA_10_90";
    bandFilters[2] = "HBA_110_190";
    bandFilters[3] = "HBA_170_230";

    for (unsigned clocks = 0; clocks < 2; ++clocks) {
      unsigned clock = clocks == 0 ? 200 : 160;

      for (unsigned zones = 0; zones < 3; ++zones) {
        unsigned nyquistZone = zones + 1;

        Parset ps;

        // set
        ps.add("Observation.sampleClock", str(format("%u") % clock));
        ps.add("Observation.bandFilter",  bandFilters[nyquistZone]);
        ps.add("Observation.nrBeams",     "1");
        ps.add("Observation.Beam[0].subbandList", "[0..511]");
        ps.updateSettings();

        // verify settings
        for (unsigned sb = 0; sb < 512; ++sb) {
          CHECK_CLOSE(ps.settings.subbandWidth() * (512 * (nyquistZone - 1) + sb), ps.settings.subbands[sb].centralFrequency, 0.001);
        }

        // override
        ps.add("Observation.Beam[0].frequencyList", "[1..512]");
        ps.updateSettings();

        // verify settings
        for (unsigned sb = 0; sb < 512; ++sb) {
          CHECK_CLOSE(sb + 1.0, ps.settings.subbands[sb].centralFrequency, 0.001);
        }
      }
    }
  }
}

/*
 * Test correlator pipeline settings.
 */

SUITE(correlator) {
  TEST(enabled) {
    TESTBOOL {
      MAKEPS("Observation.DataProducts.Output_Correlated.enabled", valstr);

      CHECK_EQUAL(val, ps.settings.correlator.enabled);
    }
  }

  TEST(nrChannels) {
    // for now, nrChannels is also defined if the correlator is disabled
    TESTKEYS("Cobalt.Correlator.nrChannelsPerSubband", "Observation.channelsPerSubband") {
      Parset ps;

      ps.add("Observation.DataProducts.Output_Correlated.enabled", "true");
      ps.add(keystr, "256");
      ps.updateSettings();

      CHECK_EQUAL(256U, ps.settings.correlator.nrChannels);
      CHECK_EQUAL(256U, ps.nrChannelsPerSubband());
    }
  }

  TEST(channelWidth) {
    // validate all powers of 2 in [1, 4096]
    for (size_t nrChannels = 1; nrChannels <= 4096; nrChannels <<= 1) {
      Parset ps;

      ps.add("Observation.DataProducts.Output_Correlated.enabled", "true");
      ps.add("Observation.channelsPerSubband", str(format("%u") % nrChannels));
      ps.updateSettings();

      CHECK_CLOSE(ps.settings.subbandWidth() / nrChannels, ps.settings.correlator.channelWidth, 0.00001);
      CHECK_CLOSE(ps.settings.subbandWidth() / nrChannels, ps.channelWidth(), 0.00001);
    }
  }

  TEST(nrSamplesPerChannel) {
    TESTKEYS("Cobalt.Correlator.nrChannelsPerSubband", "Observation.nrChannelsPerSubband") {
      Parset ps;
      
      // set
      ps.add("Observation.DataProducts.Output_Correlated.enabled", "true");
      ps.add("Cobalt.blockSize", "256");
      ps.add(keystr, "64");
      ps.updateSettings();

      // verify settings
      CHECK_EQUAL(4U, ps.settings.correlator.nrSamplesPerChannel);
      CHECK_EQUAL(4U, ps.CNintegrationSteps());
      CHECK_EQUAL(4U, ps.nrSamplesPerChannel());
    }
  }

  TEST(nrBlocksPerIntegration) {
    TESTKEYS("Cobalt.Correlator.nrBlocksPerIntegration", "OLAP.IONProc.integrationSteps") {
      Parset ps;
      
      // set
      ps.add("Observation.DataProducts.Output_Correlated.enabled", "true");
      ps.add(keystr, "42");
      ps.updateSettings();

      // verify settings
      CHECK_EQUAL(42U, ps.settings.correlator.nrBlocksPerIntegration);
      CHECK_EQUAL(42U, ps.IONintegrationSteps());
    }
  }

  /* TODO: test super-station beam former */

  SUITE(files) {
    TEST(filenames_mandatory) {
      Parset ps;
      
      // set
      ps.add("Observation.DataProducts.Output_Correlated.enabled", "true");
      ps.add("Observation.nrBeams",             "1");
      ps.add("Observation.Beam[0].subbandList", "[0]");
      ps.add("Observation.DataProducts.Output_Correlated.locations", "[localhost:.]");

      // forget filenames == throw
      CHECK_THROW(ps.updateSettings(), CoInterfaceException);

      // add filenames
      ps.add("Observation.DataProducts.Output_Correlated.filenames", "[SB000.MS]");

      // should be OK now
      ps.updateSettings();
    }

    TEST(locations_mandatory) {
      Parset ps;
      
      // set
      ps.add("Observation.DataProducts.Output_Correlated.enabled", "true");
      ps.add("Observation.nrBeams",             "1");
      ps.add("Observation.Beam[0].subbandList", "[0]");
      ps.add("Observation.DataProducts.Output_Correlated.filenames", "[SB000.MS]");

      // forget locations == throw
      CHECK_THROW(ps.updateSettings(), CoInterfaceException);

      // add locations
      ps.add("Observation.DataProducts.Output_Correlated.locations", "[localhost:.]");

      // should be OK now
      ps.updateSettings();
    }

    TEST(nr) {
      // this test is expensive, so select a few values to test
      vector<size_t> testNrSubbands;
      testNrSubbands.push_back(0);
      testNrSubbands.push_back(1);
      testNrSubbands.push_back(2);
      testNrSubbands.push_back(61);
      testNrSubbands.push_back(122);
      testNrSubbands.push_back(244);

      for (size_t i = 0; i < testNrSubbands.size(); ++i) {
        size_t nrSubbands = testNrSubbands[i];
        Parset ps;

        // set
        ps.add("Observation.DataProducts.Output_Correlated.enabled", "true");
        ps.add("Observation.nrBeams", "1");
        ps.add("Observation.Beam[0].subbandList", str(format("[%u*42]") % nrSubbands));
        ps.add("Observation.DataProducts.Output_Correlated.filenames", str(format("[%u*SBxxx.MS]") % nrSubbands));
        ps.add("Observation.DataProducts.Output_Correlated.locations", str(format("[%u*localhost:.]") % nrSubbands));
        ps.updateSettings();

        // verify settings
        CHECK_EQUAL(nrSubbands, ps.settings.correlator.files.size());
      }
    }

    TEST(location) {
      Parset ps;

      // set
      ps.add("Observation.DataProducts.Output_Correlated.enabled", "true");
      ps.add("Observation.nrBeams", "1");
      ps.add("Observation.Beam[0].subbandList", "[0]");
      ps.add("Observation.DataProducts.Output_Correlated.filenames", "[SB000.MS]");
      ps.add("Observation.DataProducts.Output_Correlated.locations", "[host:/dir]");
      ps.updateSettings();

      // verify settings
      CHECK_EQUAL("SB000.MS", ps.settings.correlator.files[0].location.filename);
      CHECK_EQUAL("host",     ps.settings.correlator.files[0].location.host);
      CHECK_EQUAL("/dir",     ps.settings.correlator.files[0].location.directory);
    }
  }
}

/*
 * TODO: Test beam former pipeline settings.
 */

/*
 * ===============================================
 * Test interaction with full Parsets for coherency
 * between fields.
 * ===============================================
 */

SUITE(integration) {
  TEST(99275) {
    // ===== read parset of observation L99275
    Parset ps("tParset.parset_obs99275");

    // some constants we expect
    const size_t nrSubbands = 26;
    const size_t nrStations = 28;
    const size_t nrSAPs = 2;

    // ===== test the basics
    CHECK_EQUAL(99275U,      ps.settings.observationID);
    CHECK_EQUAL(true,        ps.settings.realTime);
    CHECK_EQUAL(16U,         ps.settings.nrBitsPerSample);
    CHECK_EQUAL(200U,        ps.settings.clockMHz);
    CHECK_EQUAL("LBA_OUTER", ps.settings.antennaSet);
    CHECK_EQUAL("LBA_10_90", ps.settings.bandFilter);
    CHECK_EQUAL(false,       ps.settings.anaBeam.enabled);

    // test station list
    CHECK_EQUAL(nrStations,  ps.settings.stations.size());
    for (unsigned st = 0; st < nrStations; ++st) {
      CHECK_EQUAL(nrSubbands, ps.settings.stations[st].rspBoardMap.size());
      CHECK_ARRAY_EQUAL(zeroes(nrSubbands),   ps.settings.stations[st].rspBoardMap, nrSubbands);

      CHECK_EQUAL(nrSubbands, ps.settings.stations[st].rspSlotMap.size());
      CHECK_ARRAY_EQUAL(sequence(nrSubbands), ps.settings.stations[st].rspSlotMap, nrSubbands);
    }

    // check core stations
    for (unsigned st = 0; st < 21; ++st) {
      CHECK_EQUAL("CS", ps.settings.stations[st].name.substr(0,2));
      CHECK_CLOSE(3827000.0, ps.settings.stations[st].phaseCenter[0], 2000);
      CHECK_CLOSE( 460900.0, ps.settings.stations[st].phaseCenter[1], 2000);
      CHECK_CLOSE(5065000.0, ps.settings.stations[st].phaseCenter[2], 2000);
    }

    // check remote stations
    for (unsigned st = 21; st < nrStations; ++st) {
      CHECK_EQUAL("RS", ps.settings.stations[st].name.substr(0,2));
      CHECK_CLOSE(3827000.0, ps.settings.stations[st].phaseCenter[0], 30000);
      CHECK_CLOSE( 460900.0, ps.settings.stations[st].phaseCenter[1], 20000);
      CHECK_CLOSE(5065000.0, ps.settings.stations[st].phaseCenter[2], 20000);

      CHECK_EQUAL(0.0, ps.settings.stations[st].clockCorrection);
    }

    // test subband/sap configuration
    CHECK_EQUAL(nrSubbands,  ps.settings.subbands.size());
    CHECK_EQUAL(nrSAPs,      ps.settings.SAPs.size());

    // check SAP 0
    CHECK_EQUAL("Sun",   ps.settings.SAPs[0].target);
    CHECK_EQUAL("J2000", ps.settings.SAPs[0].direction.type);
    for (unsigned sb = 0; sb < 13; ++sb) {
      CHECK_EQUAL(sb, ps.settings.subbands[sb].idx);
      CHECK_EQUAL(0U, ps.settings.subbands[sb].SAP);

      // subband list is increasing and positive
      CHECK(ps.settings.subbands[sb].stationIdx > (sb == 0 ? 0 : ps.settings.subbands[sb-1].stationIdx));
    }

    // check SAP 1
    CHECK_EQUAL("3C444", ps.settings.SAPs[1].target);
    CHECK_EQUAL("J2000", ps.settings.SAPs[1].direction.type);
    for (unsigned sb = 13; sb < nrSubbands; ++sb) {
      CHECK_EQUAL(sb, ps.settings.subbands[sb].idx);
      CHECK_EQUAL(1U, ps.settings.subbands[sb].SAP);

      // subband list of SAP 1 is equal to SAP 0
      CHECK_EQUAL(ps.settings.subbands[sb - 13].stationIdx, ps.settings.subbands[sb].stationIdx);
    }

    // ===== test correlator settings
    CHECK_EQUAL(true,       ps.settings.correlator.enabled);
    CHECK_EQUAL(64U,        ps.settings.correlator.nrChannels);
    CHECK_CLOSE(3051.76,    ps.settings.correlator.channelWidth, 0.01);
    CHECK_EQUAL(720U,       ps.settings.correlator.nrSamplesPerChannel);
    CHECK_EQUAL(4U,         ps.settings.correlator.nrBlocksPerIntegration);
    CHECK_EQUAL(nrStations, ps.settings.correlator.stations.size());
    for (unsigned st = 0; st < nrStations; ++st) {
      CHECK_EQUAL(ps.settings.stations[st].name, ps.settings.correlator.stations[st].name);

      CHECK_EQUAL(1U, ps.settings.correlator.stations[st].inputStations.size());
      CHECK_EQUAL(st, ps.settings.correlator.stations[st].inputStations[0]);
    }
    CHECK_EQUAL(nrSubbands, ps.settings.correlator.files.size());

    // ===== test beam-former settings
    CHECK_EQUAL(true,       ps.settings.beamFormer.enabled);
    CHECK_EQUAL(nrSAPs,     ps.settings.beamFormer.SAPs.size());

    // check SAP 0
    CHECK_EQUAL(2U,         ps.settings.beamFormer.SAPs[0].TABs.size());
    CHECK_EQUAL(true,       ps.settings.beamFormer.SAPs[0].TABs[0].coherent);
    CHECK_EQUAL(false,      ps.settings.beamFormer.SAPs[0].TABs[1].coherent);

    // check SAP 1
    CHECK_EQUAL(2U,         ps.settings.beamFormer.SAPs[1].TABs.size());
    CHECK_EQUAL(true,       ps.settings.beamFormer.SAPs[1].TABs[0].coherent);
    CHECK_EQUAL(false,      ps.settings.beamFormer.SAPs[1].TABs[1].coherent);

    // check coherent settings
    CHECK_EQUAL(STOKES_I,   ps.settings.beamFormer.coherentSettings.type);
    CHECK_EQUAL(1U,         ps.settings.beamFormer.coherentSettings.nrStokes);
    CHECK_EQUAL(64U,        ps.settings.beamFormer.coherentSettings.nrChannels);
    CHECK_EQUAL(30U,        ps.settings.beamFormer.coherentSettings.timeIntegrationFactor);

    // check incoherent settings
    CHECK_EQUAL(STOKES_I,   ps.settings.beamFormer.incoherentSettings.type);
    CHECK_EQUAL(1U,         ps.settings.beamFormer.incoherentSettings.nrStokes);
    CHECK_EQUAL(64U,        ps.settings.beamFormer.incoherentSettings.nrChannels);
    CHECK_EQUAL(1U,         ps.settings.beamFormer.incoherentSettings.timeIntegrationFactor);
  }
}

int main(void)
{
  INIT_LOGGER("tParset");

  return UnitTest::RunAllTests() > 0;
}

