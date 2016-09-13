#ifndef LOFAR_CNPROC_STOKES_H
#define LOFAR_CNPROC_STOKES_H

#include <Interface/FilteredData.h>
#include <Interface/StreamableData.h>
#include <Interface/BeamFormedData.h>
#include <Interface/MultiDimArray.h>
#include <Interface/Parset.h>
#include <Dedispersion.h>

#if 0 || !defined HAVE_BGP
#define STOKES_C_IMPLEMENTATION
#endif

namespace LOFAR {
namespace RTCP {


class Stokes
{
  public:
    static const float MAX_FLAGGED_PERCENTAGE = 0.9f;

    Stokes(unsigned nrChannels, unsigned nrSamples);

  protected:
    const unsigned          itsNrChannels;
    const unsigned          itsNrSamples;
};

class CoherentStokes: public Stokes
{
  public:
    CoherentStokes(unsigned nrChannels, unsigned nrSamples);

    template <bool ALLSTOKES> void calculate(const SampleData<> *sampleData, PreTransposeBeamFormedData *stokesData, unsigned inbeam, const StreamInfo &info);
};

class IncoherentStokes: public Stokes
{
  public:
    IncoherentStokes(unsigned nrChannels, unsigned nrSamples, unsigned nrStations, unsigned channelIntegrations, DedispersionBeforeBeamForming *dedispersion, Allocator &allocator);

    template <bool ALLSTOKES> void calculate(const FilteredData *sampleData, PreTransposeBeamFormedData *stokesData, const std::vector<unsigned> &stationMapping, const StreamInfo &info, unsigned subband, double dm);

  private:  
    Allocator                     &itsAllocator;
    SmartPtr<FilteredData>        itsDedispersedData;
    DedispersionBeforeBeamForming *itsDedispersion;
    const unsigned                itsMaxChannelIntegrations;
};

} // namespace RTCP
} // namespace LOFAR

#endif
