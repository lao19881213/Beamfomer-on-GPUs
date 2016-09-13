//#  tWH_CN_Processing.cc: stand-alone test program for WH_CN_Processing
//#
//#  Copyright (C) 2006
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
//#
//#  $Id: tCN_Processing.cc 23195 2012-12-06 16:01:41Z mol $

//# Always #include <lofar_config.h> first!
#include <lofar_config.h>

#include <Common/DataConvert.h>
#include <Common/Exception.h>
#include <Common/Timer.h>
#include <Interface/Parset.h>
#include <PPF.h>
#include <BeamFormer.h>
#include <Correlator.h>

#if defined HAVE_MPI
#define MPICH_IGNORE_CXX_SEEK
#include <mpi.h>
#endif

#include <cmath>
#include <cstring>
#include <exception>

#include <boost/format.hpp>


using namespace LOFAR;
using namespace LOFAR::RTCP;
using boost::format;


template <typename T> void toComplex(double phi, T &z);

template <> inline void toComplex<i4complex>(double phi, i4complex &z)
{
    double s, c;

    sincos(phi, &s, &c);
    z = makei4complex(8 * c, 8 * s);
}

template <> inline void toComplex<i8complex>(double phi, i8complex &z)
{
    double s, c;

    sincos(phi, &s, &c);
    z = makei8complex((int) rint(127 * c), (int) rint(127 * s));
}

template <> inline void toComplex<i16complex>(double phi, i16complex &z)
{
    double s, c;

    sincos(phi, &s, &c);
    z = makei16complex((int) rint(32767 * c), (int) rint(32767 * s));
}


template <typename SAMPLE_TYPE> void setSubbandTestPattern(SubbandMetaData &metaData, TransposedData<SAMPLE_TYPE> &transposedData, unsigned nrStations, double signalFrequency, double subbandBandwidth)
{
  // Simulate a monochrome complex signal into the PPF, with station 1 at a
  // distance of .25 labda to introduce a delay.  Also, a few samples can be
  // flagged.

  std::clog << "setSubbandTestPattern() ... ";

  NSTimer timer("setTestPattern", true);
  timer.start();

  const double distance   = .25; // labda
  const double phaseShift = 2 * M_PI * distance;

  for (unsigned stat = 0; stat < nrStations; stat ++) {
    metaData.beams(stat)[0].delayAtBegin   = 0;
    metaData.beams(stat)[0].delayAfterEnd  = 0;
    metaData.alignmentShift(stat) = 0;
    metaData.setFlags(stat, SparseSet<unsigned>());
  }

  for (unsigned time = 0; time < transposedData.samples[0].size(); time ++) {
    double phi = 2 * M_PI * signalFrequency * time / subbandBandwidth;
    SAMPLE_TYPE sample;
    toComplex(phi, sample);

    for (unsigned stat = 0; stat < nrStations; stat ++) {
      transposedData.samples[stat][time][0] = sample;
      transposedData.samples[stat][time][1] = sample;
    }

    if (NR_POLARIZATIONS >= 2 && nrStations >= 2) {
      toComplex(phi + phaseShift, transposedData.samples[1][time][1]);
      metaData.beams(1)[0].delayAtBegin  = distance / signalFrequency;
      metaData.beams(1)[0].delayAfterEnd = distance / signalFrequency;
    }
  }
  
#if 1
  if (transposedData.samples[0].size() > 17000 && nrStations >= 6) {
    metaData.setFlags(4, SparseSet<unsigned>().include(14000, 15000));
    metaData.setFlags(5, SparseSet<unsigned>().include(17000));
  }
#endif

  std::clog << "done." << std::endl;;

#if 1 && defined WORDS_BIGENDIAN
  std::clog << "swap bytes" << std::endl;
  dataConvert(LittleEndian, transposedData.samples.data(), transposedData.samples.num_elements());
#endif

  timer.stop();
}


void checkCorrelatorTestPattern(const CorrelatedData &correlatedData, unsigned nrStations, unsigned nrChannels)
{
  const boost::multi_array_ref<fcomplex, 4> &visibilities = correlatedData.visibilities;

  static const unsigned channels[] = { 0, 201, 255 };

  for (unsigned stat1 = 0; stat1 < std::min(nrStations, 8U); stat1 ++) {
    for (unsigned stat2 = stat1; stat2 < std::min(nrStations, 8U); stat2 ++) {
      int bl = Correlator::baseline(stat1, stat2);

      std::cout << "S(" << stat1 << ") * ~S(" << stat2 << ") :\n";

      for (unsigned pol1 = 0; pol1 < NR_POLARIZATIONS; pol1 ++) {
	for (unsigned pol2 = 0; pol2 < NR_POLARIZATIONS; pol2 ++) {
	  std::cout << " " << (char) ('x' + pol1) << (char) ('x' + pol2) << ':';

	  for (size_t chidx = 0; chidx < sizeof(channels) / sizeof(int); chidx ++) {
	    unsigned ch = channels[chidx];

	    if (ch < nrChannels) {
	      std::cout << ' ' << visibilities[bl][ch][pol1][pol2] << '/' << correlatedData.nrValidSamples(bl, ch);
	    }
	  }

	  std::cout << '\n';
	}
      }
    }
  }

  std::cout << "newgraph newcurve linetype solid marktype none pts\n";
  float max = 0.0;

  for (unsigned ch = 1; ch < nrChannels; ch ++)
    if (abs(visibilities[0][ch][1][1]) > max)
      max = abs(visibilities[0][ch][1][1]);

  //std::clog << "max = " << max << std::endl;

  for (unsigned ch = 1; ch < nrChannels; ch ++)
    std::cout << ch << ' ' << (10 * std::log10(abs(visibilities[0][ch][1][1]) / max)) << '\n';
}


template <typename SAMPLE_TYPE> void doWork()
{
  unsigned   nrStations			= 288;
  unsigned   nrChannels			= 256;
  unsigned   nrSamplesPerIntegration	= 196608 / nrChannels;
  double     subbandBandwidth		= 195312.5;
  double     centerFrequency		= 384 * subbandBandwidth;
  double     baseFrequency		= centerFrequency - .5 * subbandBandwidth;
  double     testSignalChannel		= nrChannels / 5.0;
  double     signalFrequency		= baseFrequency + testSignalChannel * subbandBandwidth / nrChannels;
  unsigned   nrHistorySamples		= nrChannels > 1 ? nrChannels * (NR_TAPS - 1) : 0;
  unsigned   nrSamplesToCNProc		= nrChannels * nrSamplesPerIntegration + nrHistorySamples + 32 / sizeof(SAMPLE_TYPE[NR_POLARIZATIONS]);

  std::vector<unsigned> station2SuperStation;

# if 0
  station2SuperStation.resize(nrStations);

  for(unsigned i=0; i<nrStations; i++) {
    station2SuperStation[i] = (i / 7);
//      cerr << station2SuperStation[i] << endl;
  }
#endif

#if 0
  // just to get the factors!
  LOFAR::RTCP::BandPass bandpass(true, nrChannels);
  const float *f = bandpass.correctionFactors();
  
  std::clog << "bandpass correction:" << std::endl;

  for (unsigned i = 0; i < nrChannels; i ++)
    std::clog << i << ' ' << f[i] << std::endl;
#endif

  if (testSignalChannel >= nrChannels) {
    std::cerr << " signal lies outside the range." << std::endl;
    exit(1);
  }

  string stationNames = "[";
  for(unsigned i = 0; i < nrStations; i++) {
    if(i>0) stationNames += ", ";

    stationNames += str(format("CS%03u") % i);
  }

  stationNames += "]";

  Parset parset;
  parset.add("Observation.channelsPerSubband",       str(format("%u") % nrChannels));
  parset.add("OLAP.CNProc.integrationSteps",         str(format("%u") % nrSamplesPerIntegration));
  parset.add("Observation.sampleClock",              "200");
  parset.add("OLAP.storageStationNames",             stationNames);
  parset.add("Observation.beamList",                 "[0]");
  parset.add("Observation.Beam[0].nrTiedArrayBeams", "0");
  parset.add("OLAP.CNProc.tabList",                  "[]");

  BeamFormer beamFormer(parset);

  const char *env;
  unsigned nrBeamFormedStations = nrStations;

  if ((env = getenv("SIGNAL_FREQUENCY")) != 0)
    signalFrequency = atof(env);

  std::clog << "base   frequency = " << baseFrequency   << std::endl;
  std::clog << "center frequency = " << centerFrequency << std::endl;
  std::clog << "signal frequency = " << signalFrequency << std::endl;

  TransposedData<SAMPLE_TYPE> transposedData(nrStations, nrSamplesToCNProc);
  FilteredData   filteredData(nrStations, nrChannels, nrSamplesPerIntegration);
  CorrelatedData correlatedData(nrBeamFormedStations, nrChannels, nrChannels * nrSamplesPerIntegration);
  SubbandMetaData metaData(nrStations, 1);

  PPF<SAMPLE_TYPE> ppf(nrStations, nrChannels, nrSamplesPerIntegration, subbandBandwidth / nrChannels, true /* use delay compensation */, true /* use bandpass correction */, true /* verbose in filter bank */);
  Correlator	   correlator(beamFormer.getStationMapping(), nrChannels, nrSamplesPerIntegration);

  setSubbandTestPattern(metaData, transposedData, nrStations, signalFrequency, subbandBandwidth);

  for (unsigned stat = 0; stat < nrStations; stat ++) {
    static NSTimer ppfTimer("PPF", true);
    ppfTimer.start();
    ppf.doWork(stat, centerFrequency, &metaData, &transposedData, &filteredData);
    ppfTimer.stop();

    for(unsigned ch = 0; ch < nrChannels; ch++) {
      if (filteredData.flags[stat][ch].count() != 0)
	std::cout << "flags of station " << stat << " channel " << ch << ": " << filteredData.flags[stat][ch] << std::endl;
    }
  }

  beamFormer.mergeStations(&filteredData);

  static NSTimer correlateTimer("correlate", true);
  correlateTimer.start();
  correlator.computeFlags(&filteredData, &correlatedData);
  correlator.correlate(&filteredData, &correlatedData);
  correlateTimer.stop();

  checkCorrelatorTestPattern(correlatedData, nrBeamFormedStations, nrChannels);
}


int main(int argc, char **argv)
{
  int retval = 0;

#if defined HAVE_BGP
  INIT_LOGGER(argv[0]);
#endif

#if defined HAVE_MPI
  MPI_Init(&argc, &argv);
#else
  argc = argc; argv = argv;    // Keep compiler happy ;-)
#endif

  try {
    doWork<i16complex>();
  } catch (Exception &ex) {
    std::cerr << "Caught exception: " << ex.what() << std::endl;
    retval = 1;
  } catch (std::exception &ex) {
    std::cerr << "Caught exception: " << ex.what() << std::endl;
    retval = 1;
  } catch (...) {
    std::cerr << "Caught ... exception" << std::endl;
    retval = 1;
  }

#if defined HAVE_MPI
  MPI_Finalize();
#endif

  return retval;
}
