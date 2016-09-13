//# Always #include <lofar_config.h> first!
#include <lofar_config.h>

#include <CN_Math.h>
#include <Dedispersion.h>
#include <DedispersionAsm.h>
#include <Common/Timer.h>
#include <Common/LofarLogger.h>

#include <algorithm>


#if defined HAVE_FFTW3
#include <fftw3.h>
#include <vector>
#elif defined HAVE_FFTW2
#include <fftw.h>
#else
#error Should have FFTW3 or FFTW2 installed
#endif


namespace LOFAR {
namespace RTCP {


Dedispersion::Dedispersion(const Parset &parset, const std::vector<unsigned> &subbandIndices, std::vector<double> &DMs, Allocator &allocator)
:
  itsNrChannels(parset.nrChannelsPerSubband()),
  itsNrSamplesPerIntegration(parset.CNintegrationSteps()),
  itsFFTsize(parset.dedispersionFFTsize()),
  itsChannelBandwidth(parset.subbandBandwidth() / itsNrChannels),
  itsFFTedBuffer(NR_POLARIZATIONS, itsFFTsize),
  itsAllocator(allocator)
{
#if defined HAVE_FFTW3
  itsFFTWforwardPlan = 0;
  itsFFTWbackwardPlan = 0;
#elif defined HAVE_FFTW2
  itsFFTWforwardPlan = 0;
  itsFFTWbackwardPlan = 0;
#endif

  // initialise the list of dispersion measures
  unsigned nrDifferentDMs = 0;
  std::vector<double> uniqueDMs;

  for (unsigned i = 0; i < DMs.size(); i++) {
     double dm = DMs[i];

    if (dm == 0.0)
      continue;

    if (itsDMindices.find(dm) == itsDMindices.end()) {
      uniqueDMs.push_back(dm);
      itsDMindices[dm] = nrDifferentDMs;
      nrDifferentDMs++;
    }
  }

  initChirp(parset, subbandIndices, uniqueDMs);
}


DedispersionBeforeBeamForming::DedispersionBeforeBeamForming(const Parset &parset, FilteredData *filteredData, const std::vector<unsigned> &subbandIndices, std::vector<double> &DMs, Allocator &allocator)
:
  Dedispersion(parset, subbandIndices, DMs, allocator),
  itsNrStations(parset.nrMergedStations())
{
  initFFT(&filteredData->samples[0][0][0][0]);
}


DedispersionAfterBeamForming::DedispersionAfterBeamForming(const Parset &parset, BeamFormedData *beamFormedData, const std::vector<unsigned> &subbandIndices, std::vector<double> &DMs, Allocator &allocator)
:
  Dedispersion(parset, subbandIndices, DMs, allocator)
{
  initFFT(&beamFormedData->samples[0][0][0][0]);
}


Dedispersion::~Dedispersion()
{
#if defined HAVE_FFTW3
  if(itsFFTWforwardPlan != 0) {
    fftwf_destroy_plan(itsFFTWforwardPlan);
  }
  if(itsFFTWbackwardPlan != 0) {
    fftwf_destroy_plan(itsFFTWbackwardPlan);
  }
#else
  if(itsFFTWforwardPlan != 0) {
    fftw_destroy_plan(itsFFTWforwardPlan);
  }
  if(itsFFTWbackwardPlan != 0) {
    fftw_destroy_plan(itsFFTWbackwardPlan);
  }
#endif
}


void Dedispersion::initFFT(fcomplex *data)
{
#if defined HAVE_FFTW3
  itsFFTWforwardPlan  = fftwf_plan_many_dft(1, (int *) &itsFFTsize, 2, (fftwf_complex *) data, 0, 2, 1, (fftwf_complex *) &itsFFTedBuffer[0][0], 0, 1, itsFFTsize, FFTW_FORWARD, FFTW_MEASURE);
  itsFFTWbackwardPlan = fftwf_plan_many_dft(1, (int *) &itsFFTsize, 2, (fftwf_complex *) &itsFFTedBuffer[0][0], 0, 1, itsFFTsize, (fftwf_complex *) data, 0, 2, 1, FFTW_BACKWARD, FFTW_MEASURE);
#elif defined HAVE_FFTW2
#if defined HAVE_BGP
  fftw_import_wisdom_from_string("(FFTW-2.1.5 (196608 529 1 0 1 2 1 77 0) (98304 529 1 0 1 2 1 99 0) (49152 529 1 0 1 2 1 715 0) (24576 529 1 0 1 2 1 715 0) (12288 529 1 0 1 2 1 715 0) (6144 529 1 0 1 2 1 77 0) (3072 529 1 0 1 2 1 715 0) (1536 529 1 0 1 2 1 187 0) (768 529 1 0 1 2 1 143 0) (384 529 1 0 1 2 1 143 0) (192 529 1 0 1 2 1 143 0) (96 529 1 0 1 2 1 143 0) (48 529 1 0 1 2 1 143 0) (24 529 1 0 1 2 1 143 0) (12 529 1 0 1 2 0 276 0) (6 529 1 0 1 2 0 144 0) (3 529 1 0 1 2 0 78 0) (196608 529 -1 0 2 1 1 704 0) (98304 529 -1 0 2 1 1 704 0) (49152 529 -1 0 2 1 1 704 0) (24576 529 -1 0 2 1 1 704 0) (12288 529 -1 0 2 1 1 704 0) (6144 529 -1 0 2 1 1 704 0) (3072 529 -1 0 2 1 1 132 0) (1536 529 -1 0 2 1 1 132 0) (768 529 -1 0 2 1 1 132 0) (384 529 -1 0 2 1 1 132 0) (192 529 -1 0 2 1 1 352 0) (96 529 -1 0 2 1 1 132 0) (48 529 -1 0 2 1 1 132 0) (24 529 -1 0 2 1 1 132 0) (12 529 -1 0 2 1 0 265 0) (6 529 -1 0 2 1 0 133 0) (3 529 -1 0 2 1 0 67 0) (2 529 -1 0 2 1 0 45 0) (4 529 -1 0 2 1 0 89 0) (8 529 -1 0 2 1 0 177 0) (16 529 -1 0 2 1 0 353 0) (32 529 -1 0 2 1 0 705 0) (64 529 -1 0 2 1 0 1409 0) (128 529 -1 0 2 1 0 2817 0) (256 529 -1 0 2 1 1 352 0) (512 529 -1 0 2 1 1 352 0) (1024 529 -1 0 2 1 1 704 0) (2048 529 -1 0 2 1 1 704 0) (4096 529 -1 0 2 1 1 704 0) (8192 529 -1 0 2 1 1 352 0) (16384 529 -1 0 2 1 1 704 0) (32768 529 -1 0 2 1 1 704 0) (65536 529 -1 0 2 1 1 704 0) (2 529 1 0 1 2 0 56 0) (4 529 1 0 1 2 0 100 0) (8 529 1 0 1 2 0 188 0) (16 529 1 0 1 2 0 364 0) (32 529 1 0 1 2 0 716 0) (64 529 1 0 1 2 0 1420 0) (128 529 1 0 1 2 0 2828 0) (256 529 1 0 1 2 1 715 0) (512 529 1 0 1 2 1 187 0) (1024 529 1 0 1 2 1 715 0) (2048 529 1 0 1 2 1 715 0) (4096 529 1 0 1 2 1 715 0) (8192 529 1 0 1 2 1 1419 0) (16384 529 1 0 1 2 1 99 0) (32768 529 1 0 1 2 1 715 0) (65536 529 1 0 1 2 1 715 0))");
#endif

  itsFFTWforwardPlan  = fftw_create_plan_specific(itsFFTsize, FFTW_FORWARD,  FFTW_ESTIMATE | FFTW_USE_WISDOM, (fftw_complex *) data, 2, (fftw_complex *) &itsFFTedBuffer[0][0], 1);
  itsFFTWbackwardPlan = fftw_create_plan_specific(itsFFTsize, FFTW_BACKWARD, FFTW_ESTIMATE | FFTW_USE_WISDOM, (fftw_complex *) &itsFFTedBuffer[0][0], 1, (fftw_complex *) data, 2);
#endif
}


void Dedispersion::forwardFFT(const fcomplex *data)
{
#if defined HAVE_FFTW3
  fftwf_execute_dft(itsFFTWforwardPlan, (fftwf_complex *) data, (fftwf_complex *) &itsFFTedBuffer[0][0]);
#elif defined HAVE_FFTW2
  fftw(itsFFTWforwardPlan, 2, (fftw_complex *) data, 2, 1, (fftw_complex *) &itsFFTedBuffer[0][0], 1, itsFFTsize);
#endif
}


void Dedispersion::backwardFFT(fcomplex *data)
{
#if defined HAVE_FFTW3
  fftwf_execute_dft(itsFFTWbackwardPlan, (fftwf_complex *) &itsFFTedBuffer[0][0], (fftwf_complex *) data);
#elif defined HAVE_FFTW2
  fftw(itsFFTWbackwardPlan, 2, (fftw_complex *) &itsFFTedBuffer[0][0], 1, itsFFTsize, (fftw_complex *) data, 2, 1);
#endif
}


void Dedispersion::initChirp(const Parset &parset, const std::vector<unsigned> &subbandIndices, std::vector<double> &uniqueDMs)
{
  itsChirp.resize(*std::max_element(subbandIndices.begin(), subbandIndices.end()) + 1, uniqueDMs.size());
//std::cout << "newcurve linetype solid linethickness 3 marktype none color 0 .7 0 pts" << std::endl;

  for (unsigned i = 0; i < subbandIndices.size(); i ++) {
    for (unsigned dmIndex = 0; dmIndex < uniqueDMs.size(); dmIndex++) {
      double   dm                = uniqueDMs[dmIndex];
      unsigned subbandIndex      = subbandIndices[i];
      double   channel0frequency = parset.channel0Frequency(subbandIndex);
      double   binWidth	       = itsChannelBandwidth / itsFFTsize;
      double   dmConst	       = dm * 2 * M_PI / 2.41e-16;

      itsChirp[subbandIndex][dmIndex] = new Matrix<fcomplex>(itsNrChannels, itsFFTsize, 32, itsAllocator);

      for (unsigned channel = 0; channel < itsNrChannels; channel ++) {
        double channelFrequency = channel0frequency + channel * itsChannelBandwidth;

        for (unsigned n = 0; n < itsFFTsize; n ++) {
          double binFrequency = n * binWidth;

          if (n > itsFFTsize / 2)
            binFrequency -= itsChannelBandwidth;

          double	 frequencyDiv = binFrequency / channelFrequency;
          double	 frequencyFac = frequencyDiv * frequencyDiv / (channelFrequency + binFrequency);
          dcomplex dfactor      = cosisin(dmConst * frequencyFac);
          fcomplex factor	      = makefcomplex(real(dfactor), -imag(dfactor));
          float	 taper	      = sqrt(1 + pow(binFrequency / (.47 * itsChannelBandwidth), 80));
  //if (channel == 0) std::cout << n << ' ' << 1/taper << std::endl;

          (*itsChirp[subbandIndex][dmIndex])[channel][n] = factor / (taper * itsFFTsize);
        }
      }  
    }
  }
}


void Dedispersion::applyChirp(unsigned subbandIndex, unsigned dmIndex, unsigned channel)
{
  static NSTimer chirpTimer("chirp timer", true, true);
  const fcomplex *chirp = &(*itsChirp[subbandIndex][dmIndex])[channel][0];

  chirpTimer.start();

#if defined HAVE_BGP
  _apply_chirp(&itsFFTedBuffer[0][0], &itsFFTedBuffer[1][0], chirp, itsFFTsize);
#else
  for (unsigned time = 0; time < itsFFTsize; time ++) {
    itsFFTedBuffer[0][time] *= chirp[time];
    itsFFTedBuffer[1][time] *= chirp[time];
  }
#endif

  chirpTimer.stop();
}


void DedispersionBeforeBeamForming::dedisperse(const FilteredData *inData, FilteredData *outData, unsigned instat, unsigned outstat, unsigned firstch, unsigned numch, unsigned subbandIndex, double dm)
{
  if (dm == 0.0)
    return;

  unsigned dmIndex = itsDMindices[dm];

  for (unsigned channel = 0; channel < numch; channel ++) {
    for (unsigned block = 0; block < itsNrSamplesPerIntegration; block += itsFFTsize) {
      forwardFFT(&inData->samples[firstch + channel][instat][block][0]);
      applyChirp(subbandIndex, dmIndex, channel);
      backwardFFT(&outData->samples[channel][outstat][block][0]);
    }
  }
}


void DedispersionAfterBeamForming::dedisperse(BeamFormedData *beamFormedData, unsigned subbandIndex, unsigned beam, double dm)
{
  if (dm == 0.0)
    return;

  unsigned dmIndex = itsDMindices[dm];

  for (unsigned channel = 0; channel < itsNrChannels; channel ++) {
    for (unsigned block = 0; block < itsNrSamplesPerIntegration; block += itsFFTsize) {
      forwardFFT(&beamFormedData->samples[beam][channel][block][0]);
      applyChirp(subbandIndex, dmIndex, channel);
      backwardFFT(&beamFormedData->samples[beam][channel][block][0]);
    }
  }
}


} // namespace RTCP
} // namespace LOFAR
