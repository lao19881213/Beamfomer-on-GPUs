#ifndef LOFAR_CNPROC_CORRELATOR_H
#define LOFAR_CNPROC_CORRELATOR_H

#if 0 || !defined HAVE_BGP
#define CORRELATOR_C_IMPLEMENTATION
#endif


#include <Interface/CorrelatedData.h>
#include <Interface/StreamableData.h>

#include <cassert>
#include <cmath>

#include <boost/multi_array.hpp>

namespace LOFAR {
namespace RTCP {



class Correlator
{
  public:
    Correlator(const std::vector<unsigned> &stationMapping, unsigned nrChannels, unsigned nrSamplesPerIntegration);

    // We can correlate arrays of size
    // samples[nrChannels][nrStations][nrSamplesPerIntegration][nrPolarizations]
    void	    correlate(const SampleData<> *, CorrelatedData *);
    void	    computeFlags(const SampleData<> *, CorrelatedData *);

    static unsigned baseline(unsigned station1, unsigned station2);
    static void baselineToStations(const unsigned baseline, unsigned& station1, unsigned& station2);
    static bool baselineIsAutoCorrelation(const unsigned baseline);

  private:
    template <typename T> void  setNrValidSamples(const SampleData<> *sampleData, Matrix<T> &);

    const unsigned  itsNrStations, itsNrBaselines, itsNrChannels, itsNrSamplesPerIntegration;
    std::vector<float> itsCorrelationWeights; //[itsNrSamplesPerIntegration + 1]

    // A list indexed by station number, result is the station position in the input data.
    // This is needed in case of tied array beam forming.
    const std::vector<unsigned> &itsStationMapping; //[itsNrStations]
};


inline unsigned Correlator::baseline(unsigned station1, unsigned station2)
{
  assert(station1 <= station2);
  return station2 * (station2 + 1) / 2 + station1;
}

inline void Correlator::baselineToStations(const unsigned baseline, unsigned& station1, unsigned& station2)
{
  station2 = (unsigned) (sqrtf(2 * baseline + .25f) - .5f);
  station1 = baseline - station2 * (station2 + 1) / 2;
}

inline bool Correlator::baselineIsAutoCorrelation(const unsigned baseline)
{
  unsigned station1, station2;
  baselineToStations(baseline, station1, station2);
  return station1 == station2;
}

} // namespace RTCP
} // namespace LOFAR

#endif
