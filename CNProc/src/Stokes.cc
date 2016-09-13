//# Always #include <lofar_config.h> first!
#include <lofar_config.h>

#include <Stokes.h>
#include <Interface/MultiDimArray.h>
#include <Common/LofarLogger.h>

template <typename T> static inline T sqr(const T x) { return x * x; }

#if defined STOKES_C_IMPLEMENTATION
static void inline _StokesIQUV(
  float *I, float *Q, float *U, float *V,
  const LOFAR::fcomplex (*XY)[2],
  unsigned length)
{
  for (unsigned i = 0; i < length; i ++, I ++, Q ++, U ++, V ++, XY ++) {
    LOFAR::fcomplex polX = (*XY)[0];
    LOFAR::fcomplex polY = (*XY)[1];
    float powerX = sqr(real(polX)) + sqr(imag(polX));
    float powerY = sqr(real(polY)) + sqr(imag(polY));

    *I = powerX + powerY;
    *Q = powerX - powerY;
    *U = 2 * real(polX * conj(polY));
    *V = 2 * imag(polX * conj(polY));
  }
}

static void inline _StokesI(
  float *I,
  const LOFAR::fcomplex (*XY)[2],
  unsigned length)
{
  for (unsigned i = 0; i < length; i ++, I ++, XY ++) {
    LOFAR::fcomplex polX = (*XY)[0];
    LOFAR::fcomplex polY = (*XY)[1];
    float powerX = sqr(real(polX)) + sqr(imag(polX));
    float powerY = sqr(real(polY)) + sqr(imag(polY));

    *I = powerX + powerY;
  }
}
#else
#include <StokesAsm.h>
#endif

namespace LOFAR {
namespace RTCP {

Stokes::Stokes(unsigned nrChannels, unsigned nrSamples)
:
  itsNrChannels(nrChannels),
  itsNrSamples(nrSamples)
{
}
CoherentStokes::CoherentStokes(unsigned nrChannels, unsigned nrSamples)
:
  Stokes(nrChannels, nrSamples)
{
}

IncoherentStokes::IncoherentStokes(unsigned nrChannels, unsigned nrSamples, unsigned nrStations, unsigned maxChannelIntegrations, DedispersionBeforeBeamForming *dedispersion, Allocator &allocator)
:
  Stokes(nrChannels, nrSamples),
  itsAllocator(allocator),
  itsDedispersedData(dedispersion ? new FilteredData(nrStations, maxChannelIntegrations, itsNrSamples, allocator) : 0),
  itsDedispersion(dedispersion),
  itsMaxChannelIntegrations(maxChannelIntegrations)
{
}

// Calculate coherent stokes values from pencil beams.
template <bool ALLSTOKES> void CoherentStokes::calculate(const SampleData<> *sampleData, PreTransposeBeamFormedData *stokesData, unsigned inbeam, const StreamInfo &info)
{
  // TODO: divide by #valid stations
  ASSERT(sampleData->samples.shape()[0] > inbeam);
  ASSERT(sampleData->samples.shape()[1] == itsNrChannels);
  ASSERT(sampleData->samples.shape()[2] >= itsNrSamples);
  ASSERT(sampleData->samples.shape()[3] == NR_POLARIZATIONS);

  const unsigned &timeIntegrations = info.timeIntFactor;
  const unsigned channelIntegrations = itsNrChannels / info.nrChannels;

#ifndef STOKES_C_IMPLEMENTATION
  // restrictions demanded by assembly routines
  ASSERT(itsNrSamples % 4 == 0);
  ASSERT(itsNrSamples >= 8);
#endif  

  // process flags
  for(unsigned ch = 0; ch < info.nrChannels; ch++) {
    stokesData->flags[ch].reset();
  }
  for(unsigned ch=0; ch < itsNrChannels; ch++) {
    stokesData->flags[ch/channelIntegrations] |= sampleData->flags[inbeam][ch];
    stokesData->flags[ch/channelIntegrations] /= timeIntegrations;
  }

  // process data
  const boost::detail::multi_array::const_sub_array<fcomplex,3> &in = sampleData->samples[inbeam];
  MultiDimArray<float,3> &out = stokesData->samples;

  if (timeIntegrations <= 1 && channelIntegrations <= 1) {
    for (unsigned ch = 0; ch < itsNrChannels; ch ++) {
      if (ALLSTOKES) {
        _StokesIQUV(&out[0][ch][0],
                    &out[1][ch][0],
                    &out[2][ch][0],
                    &out[3][ch][0],
                    reinterpret_cast<const fcomplex (*)[2]>(&in[ch][0][0]),
                    itsNrSamples);
      } else {
        _StokesI(   &out[0][ch][0],
                    reinterpret_cast<const fcomplex (*)[2]>(&in[ch][0][0]),
                    itsNrSamples);
      }
    }  
  } else {
    // process per channel, as there are |2 samples between them, and _StokesI* routines only
    // takes multiples of 4.
    Cube<float> stokes(channelIntegrations, ALLSTOKES ? 4 : 1, itsNrSamples);

    for (unsigned ch = 0; ch < itsNrChannels; ch += channelIntegrations) {
      if (ALLSTOKES) {
        for (unsigned c = 0; c < channelIntegrations; c++)
          _StokesIQUV(&stokes[c][0][0],
                       &stokes[c][1][0],
                       &stokes[c][2][0],
                       &stokes[c][3][0],
                       reinterpret_cast<const fcomplex (*)[2]>(&in[ch][0][0]),
                       itsNrSamples);

        // integrate
        unsigned outchnum = ch / channelIntegrations;

        float *outch[4] = {
	  &out[0][outchnum][0],
	  &out[1][outchnum][0],
	  &out[2][outchnum][0],
	  &out[3][outchnum][0]
	};

        for (unsigned i = 0; i < itsNrSamples;) {
          float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

          for (unsigned j = 0; j < timeIntegrations; j ++) {
            for (unsigned c = 0; c < channelIntegrations; c ++) {
              for (unsigned s = 0; s < 4; s ++)
                acc[s] += stokes[c][s][i];
            }

            i++;
          }

          for (unsigned s = 0; s < 4; s ++)
            *(outch[s]++) = acc[s];
        }
      } else {
        for (unsigned c = 0; c < channelIntegrations; c ++)
          _StokesI(&stokes[c][0][0],
                   reinterpret_cast<const fcomplex (*)[2]>(&in[ch][0][0]),
                   itsNrSamples);

        // integrate             
        float *outch = &out[0][ch / channelIntegrations][0];

        for (unsigned i = 0; i < itsNrSamples;) {
          float acc = 0.0f;

          for (unsigned j = 0; j < timeIntegrations; j ++) {
            for (unsigned c = 0; c < channelIntegrations; c ++)
              acc += stokes[c][0][i];

            i ++;
          }

          *(outch ++) = acc;
        }
      }
    }  
  }  
}

template void CoherentStokes::calculate<true>(const SampleData<> *, PreTransposeBeamFormedData *, unsigned, const StreamInfo&);
template void CoherentStokes::calculate<false>(const SampleData<> *, PreTransposeBeamFormedData *, unsigned, const StreamInfo&);

template <bool ALLSTOKES> struct stokes {
  // the sums of stokes values over a number of stations or beams
};

template<> struct stokes<true> {
  double i, q, u, v;

  stokes(): i(0.0), q(0.0), u(0.0), v(0.0) {}

  double &I() { return i; }
  double &Q() { return q; }
  double &U() { return u; }
  double &V() { return v; }
};

template<> struct stokes<false> {
  double i;

  stokes(): i(0.0) {}

  double &I() { return i; }
  double &Q() { return i; }
  double &U() { return i; }
  double &V() { return i; }
};

// compute Stokes values, and add them to an existing stokes array
template <bool ALLSTOKES> static inline void addStokes(struct stokes<ALLSTOKES> &stokes, const LOFAR::fcomplex (*XY)[2], unsigned nrIntegrations = 1)
{
  // assert: two polarizations
  for (unsigned i = 0 ; i < nrIntegrations; i++, XY++) {
    const LOFAR::fcomplex polX = (*XY)[0];
    const LOFAR::fcomplex polY = (*XY)[1];

    const double powerX = sqr(real(polX)) + sqr(imag(polX));
    const double powerY = sqr(real(polY)) + sqr(imag(polY));

    stokes.I() += powerX + powerY;

    if (ALLSTOKES) {
      stokes.Q() += powerX - powerY;
      stokes.U() += 2 * real(polX * conj(polY));
      stokes.V() += 2 * imag(polX * conj(polY));
    }
  }  
}

// Calculate incoherent stokes values from (filtered) station data.
template <bool ALLSTOKES> void IncoherentStokes::calculate(const FilteredData *in, PreTransposeBeamFormedData *out, const std::vector<unsigned> &stationMapping, const StreamInfo &info, unsigned subband, double dm)
{
  const unsigned nrStations = stationMapping.size();

  ASSERT(in->samples.shape()[0] == itsNrChannels);
  // in->samples.shape()[1] has to be bigger than all elements in stationMapping
  ASSERT(in->samples.shape()[2] >= itsNrSamples);
  ASSERT(in->samples.shape()[3] == NR_POLARIZATIONS);

  const unsigned &timeIntegrations = info.timeIntFactor;
  const unsigned channelIntegrations = itsNrChannels / info.nrChannels;
  std::vector<unsigned> stationList;

  for(unsigned ch = 0; ch < info.nrChannels; ch++) {
    out->flags[ch].reset();
  }

  ASSERT(channelIntegrations <= itsMaxChannelIntegrations);

  for (unsigned stat = 0; stat < nrStations; stat ++) {
    const unsigned upperBound = static_cast<unsigned>(itsNrSamples * Stokes::MAX_FLAGGED_PERCENTAGE);
    const unsigned srcStat = stationMapping[stat];

    unsigned count = 0;
    for(unsigned ch = 0; ch < itsNrChannels; ch++) {
      count += in->flags[ch][srcStat].count();
    }

    if(count > upperBound) {
      // drop station due to too much flagging
    } else {
      stationList.push_back(srcStat);  

      // conservative flagging: flag anything that is flagged in one of the stations
      for(unsigned ch = 0; ch < itsNrChannels; ch++) {
        out->flags[ch/channelIntegrations] |= in->flags[ch][srcStat];
      }
    }
  }

  const unsigned nrValidStations = stationList.size();

  if (nrValidStations == 0) {
    /* if no valid samples, insert zeroes */

    for (unsigned stokes = 0; stokes < info.nrStokes; stokes++)
      for (unsigned ch = 0; ch < info.nrChannels; ch++)
        memset(&out->samples[stokes][ch][0], 0, info.nrSamples * sizeof out->samples[0][0][0]);

    // flag everything
    for(unsigned ch=0; ch<info.nrChannels; ch++) {
      out->flags[ch].include(0, info.nrSamples);
    }

    return;
  }

  // shorten the flags over the integration length
  for(unsigned ch = 0; ch < info.nrChannels; ch++) {
    out->flags[ch] /= timeIntegrations;
  }

  const bool dedisperse = dm != 0.0 && itsDedispersion;
  
  for (unsigned inch = 0, outch = 0; inch < itsNrChannels; inch += channelIntegrations, outch++) {

    if (dedisperse) {
      // dedisperse channelIntegration channels for all stations
      for (unsigned outstat = 0; outstat < stationList.size(); outstat ++) {
        unsigned instat = stationList[outstat];

        itsDedispersion->dedisperse( in, itsDedispersedData.get(), instat, outstat, inch, channelIntegrations, subband, dm );
      }
    }

    for (unsigned inTime = 0, outTime = 0; inTime < itsNrSamples; inTime += timeIntegrations, outTime ++) {
      struct stokes<ALLSTOKES> stokes;

      if (dedisperse) {
         for (unsigned i = 0; i < nrValidStations; i ++)
          for (unsigned c = 0; c < channelIntegrations; c++)
            addStokes<ALLSTOKES>(stokes, reinterpret_cast<const fcomplex (*)[2]>(&itsDedispersedData->samples[c][i][inTime][0]), timeIntegrations);
      } else {
         for (unsigned i = 0; i < nrValidStations; i ++) {
          unsigned stat = stationList[i];

          for (unsigned c = 0; c < channelIntegrations; c++)
            addStokes<ALLSTOKES>(stokes, reinterpret_cast<const fcomplex (*)[2]>(&in->samples[inch + c][stat][inTime][0]), timeIntegrations);
        }  
      }  

      #define dest(stokes) out->samples[stokes][outch][outTime]
      dest(0) = stokes.I() / nrValidStations;

      if (ALLSTOKES) {
        dest(1) = stokes.Q() / nrValidStations;
        dest(2) = stokes.U() / nrValidStations;
        dest(3) = stokes.V() / nrValidStations;
      }
      #undef dest
    }
  }
}

template void IncoherentStokes::calculate<true>(const FilteredData *, PreTransposeBeamFormedData *, const std::vector<unsigned> &, const StreamInfo&, unsigned, double);
template void IncoherentStokes::calculate<false>(const FilteredData *, PreTransposeBeamFormedData *, const std::vector<unsigned> &, const StreamInfo&, unsigned, double);

} // namespace RTCP
} // namespace LOFAR
