#include <lofar_config.h>

#include <Common/LofarLogger.h>
#include <Common/Timer.h>
#include <CN_Math.h>
#include <Dedispersion.h>

#include <cassert>
#include <cstring>

#include <boost/lexical_cast.hpp>


#define BLOCK_SIZE	4096
#define FFT_SIZE	4096
#define DM		10
#define NR_STATIONS	64
#define NR_BEAMS	64
#define NR_CHANNELS	16


using namespace LOFAR;
using namespace LOFAR::RTCP;
using namespace LOFAR::TYPES;


void init(Parset &parset)
{
  assert(BLOCK_SIZE % FFT_SIZE == 0);

  std::string stationList("[ST0");

  for (int stat = 1; stat < NR_STATIONS; stat ++)
    stationList.append(",").append(boost::lexical_cast<std::string>(stat));

  stationList.append("]");

  parset.add("OLAP.storageStationNames", stationList);
  parset.add("Observation.Beam[0].nrTiedArrayBeams", "64");
  parset.add("Observation.channelsPerSubband", boost::lexical_cast<std::string>(NR_CHANNELS));
  parset.add("OLAP.CNProc.integrationSteps", boost::lexical_cast<std::string>(BLOCK_SIZE));
  parset.add("OLAP.CNProc.dedispersionFFTsize", boost::lexical_cast<std::string>(FFT_SIZE));
  parset.add("OLAP.CNProc.tabList", "[]");
  parset.add("Observation.bandFilter", "LBA_30_90");
  parset.add("Observation.sampleClock", "200");
  parset.add("OLAP.dispersionMeasure", boost::lexical_cast<std::string>(DM));
  parset.add("Observation.subbandList", "[50]");
}


void setTestPattern(FilteredData &filteredData)
{
  memset(&filteredData.samples[0][0][0][0], 0, filteredData.samples.num_elements() * sizeof(fcomplex));

  for (unsigned i = 0; i < BLOCK_SIZE; i ++)
    filteredData.samples[0][0][i][0] = cosisin(2 * M_PI * i * 5 / BLOCK_SIZE) /* + cosisin(2 * M_PI * i * 22 / BLOCK_SIZE) */;
}


void setTestPattern(BeamFormedData &beamFormedData)
{
  memset(&beamFormedData.samples[0][0][0][0], 0, beamFormedData.samples.num_elements() * sizeof(fcomplex));

  for (unsigned i = 0; i < BLOCK_SIZE; i ++)
    beamFormedData.samples[0][0][i][0] = cosisin(2 * M_PI * i * 5 / BLOCK_SIZE) /* + cosisin(2 * M_PI * i * 22 / BLOCK_SIZE) */;
}


void plot(const FilteredData &filteredData, float r, float g, float b)
{
  std::cout << "newcurve linetype solid linethickness 3 marktype none color " << r << ' ' << g << ' ' << b << " pts" << std::endl;

  for (unsigned i = 0; i < FFT_SIZE; i ++)
    std::cout << i << ' ' << real(filteredData.samples[0][0][i][0]) << std::endl;
}


void plot(const BeamFormedData &beamFormedData, float r, float g, float b)
{
  std::cout << "newcurve linetype solid linethickness 3 marktype none color " << r << ' ' << g << ' ' << b << " pts" << std::endl;

  for (unsigned i = 0; i < FFT_SIZE; i ++)
    std::cout << i << ' ' << real(beamFormedData.samples[0][0][i][0]) << std::endl;
}


int main()
{
#if defined HAVE_BGP
  INIT_LOGGER_WITH_SYSINFO("tDedispersion");
#endif  

  Parset parset;
  init(parset);

#if 1
  BeamFormedData beamFormedData(NR_BEAMS, NR_CHANNELS, BLOCK_SIZE);
  std::vector<unsigned> subbandIndices(1, 0);
  std::vector<double> DMs(1, DM);
  DedispersionAfterBeamForming dedispersion(parset, &beamFormedData, subbandIndices, DMs);

  setTestPattern(beamFormedData);
  std::cout << "newgraph xaxis size 7 yaxis size 7" << std::endl;
  plot(beamFormedData, 1, 0, 0);

  NSTimer timer("dedisperse total", true, true);
  timer.start();

  for (unsigned beam = 0; beam < NR_BEAMS; beam ++)
    dedispersion.dedisperse(&beamFormedData, 0, beam, DMs[0]);

  timer.stop();

  plot(beamFormedData, 0, 0, 1);
#else
  FilteredData filteredData(NR_STATIONS, NR_CHANNELS, BLOCK_SIZE);
  std::vector<unsigned> subbandIndices(1, 0);
  std::vector<double> DMs(1, DM);
  DedispersionBeforeBeamForming dedispersion(parset, &filteredData, subbandIndices, DMs);

  setTestPattern(filteredData);
  std::cout << "newgraph xaxis size 7 yaxis size 7" << std::endl;
  plot(filteredData, 1, 0, 0);

  NSTimer timer("dedisperse total", true, true);
  timer.start();
  dedispersion.dedisperse(&filteredData, DMs[0]);
  timer.stop();

  plot(filteredData, 0, 0, 1);
#endif

  return 0;
}
