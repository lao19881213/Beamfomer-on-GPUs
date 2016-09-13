//# Always #include <lofar_config.h> first!
#include <lofar_config.h>

//# Includes
#include <PPF.h>
#include <FFT_Asm.h>
#include <FIR_Asm.h>
#include <CN_Math.h>

#include <Interface/Align.h>
#include <Interface/AlignedStdAllocator.h>
#include <Interface/Exceptions.h>

#include <Common/DataConvert.h>
#include <Common/Timer.h>

#include <complex>
#include <cmath>
#include <stdexcept>


namespace LOFAR {
namespace RTCP {


static NSTimer computeFlagsTimer("PPF::computeFlags()", true, true);
static NSTimer FIRtimer("PPF::FIRtimer", true, true);
static NSTimer FFTtimer("PPF::FFT", true, true);
static NSTimer PPFtimer("PPF::filter()", true, true);


template <typename SAMPLE_TYPE> PPF<SAMPLE_TYPE>::PPF(unsigned nrStations, unsigned nrChannels, unsigned nrSamplesPerIntegration, double channelBandwidth, bool delayCompensation, bool correctBandPass, bool verbose)
:
  itsNrStations(nrStations),
  itsNrSamplesPerIntegration(nrSamplesPerIntegration),
  itsNrChannels(nrChannels),
  itsChannelBandwidth(channelBandwidth),
  itsDelayCompensation(delayCompensation),
  itsCorrectBandPass(correctBandPass),
  itsBandPass(correctBandPass, nrChannels),
  itsFilterBank(verbose, NR_TAPS, nrChannels, KAISER),
  itsFIRs(boost::extents[nrStations][NR_POLARIZATIONS][nrChannels]),

#if defined PPF_C_IMPLEMENTATION
  itsFFTinData(boost::extents[NR_TAPS - 1 + nrSamplesPerIntegration][NR_POLARIZATIONS][nrChannels])
#else
  itsDelayLines(boost::extents[4][nrSamplesPerIntegration]),
  itsFFTinData(boost::extents[nrSamplesPerIntegration][NR_POLARIZATIONS][nrChannels + 4]),
  itsFFToutData(boost::extents[2][NR_POLARIZATIONS][nrChannels])
#endif
{
  if (!powerOfTwo(nrChannels))
    THROW(CNProcException, "nrChannels must be a power of 2");

#if USE_ORIGINAL_FILTER
  if (nrChannels != 256)
    THROW(CNProcException, "nrChannels != 256 not implemented, turn USE_ORIGINAL_FILTER off in FIR.h");
#endif

  for (itsLogNrChannels = 0; 1U << itsLogNrChannels != itsNrChannels; itsLogNrChannels ++)
    ;

  init_fft();

#if !defined PPF_C_IMPLEMENTATION
  initConstantTable();
#endif

  // Init the FIR filters themselves with the weights of the filterbank.
  for (unsigned stat = 0; stat < nrStations; stat ++)
    for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol ++)
      for (unsigned chan = 0; chan < nrChannels; chan ++)
        itsFIRs[stat][pol][chan].initFilter(&itsFilterBank, chan);

  // In CEP, the first subband is from -98 KHz to 98 KHz, rather than from 0 to 195 KHz.
  // To avoid that the FFT outputs the channels in the wrong order (from 128 to
  // 255 followed by channels 0 to 127), we multiply each second FFT input by -1.
  // This is efficiently achieved by negating the FIR filter constants of all
  // uneven FIR filters.
  itsFilterBank.negateWeights();
}


#if !defined PPF_C_IMPLEMENTATION

template <> void PPF<i4complex>::initConstantTable()
{
  extern fcomplex _FIR_fp_table[16][16];

  static const float map[] = {
/*
 * TODO: Test whether this is better:
 *
     0.5,  1.5,  2.5,  3.5,  4.5,  5.5,  6.5,  7.5, 
    -7.5, -6.5, -5.5, -4.5, -3.5, -2.5, -1.5, -0.5
 */
     0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0, 
    -7.0, -7.0, -6.0,  5.0, -4.0, -3.0, -2.0, -1.0
  };

  for (unsigned i = 0; i < 16; i ++)
    for (unsigned j = 0; j < 16; j ++)
      _FIR_fp_table[i][j] = makefcomplex(map[j], map[i]);
}


template <> void PPF<i8complex>::initConstantTable()
{
  // This takes up pretty much space (.5 MB)
  extern fcomplex _FIR_fp_table[256][256];

  for (unsigned i = 0; i < 256; i ++)
    for (unsigned j = 0; j < 256; j ++)
      _FIR_fp_table[i][j] = makefcomplex((float) (signed char) i, (float) (signed char) j);
}


template <> void PPF<i16complex>::initConstantTable()
{
#if 0
  extern float _FIR_fp_table[65536];

  for (unsigned i = 0; i < 65536; i ++)
    _FIR_fp_table[i] = (float) byteSwap((signed short) i);
#endif
}

#endif // PPF_C_IMPLEMENTATION


template <typename SAMPLE_TYPE> PPF<SAMPLE_TYPE>::~PPF()
{
  destroy_fft();
}


template <typename SAMPLE_TYPE> void PPF<SAMPLE_TYPE>::init_fft()
{
#if defined HAVE_FFTW3
  fftwf_complex *buf = static_cast<fftwf_complex *>(fftwf_malloc(2 * itsNrChannels * sizeof(fftwf_complex)));

  if (buf == 0)
    THROW(AssertError, "Out of memory");
     

  itsFFTWPlan = fftwf_plan_dft_1d(itsNrChannels, buf, buf + itsNrChannels, FFTW_FORWARD, FFTW_ESTIMATE);
  fftwf_free(buf);
#elif defined HAVE_FFTW2
  itsFFTWPlan = fftw_create_plan(itsNrChannels, FFTW_FORWARD, FFTW_ESTIMATE);
#endif

  //FFTtest();
}


template <typename SAMPLE_TYPE> void PPF<SAMPLE_TYPE>::destroy_fft()
{
#if defined HAVE_FFTW3
  fftwf_destroy_plan(itsFFTWPlan);
#elif defined HAVE_FFTW2
  fftw_destroy_plan(itsFFTWPlan);
#endif
}


template <typename SAMPLE_TYPE> void PPF<SAMPLE_TYPE>::computeFlags(unsigned stat, const SubbandMetaData *metaData, FilteredData *filteredData)
{
  computeFlagsTimer.start();

  for (unsigned ch = 0; ch < itsNrChannels; ch++) {
    filteredData->flags[ch][stat].reset();
  }

  SparseSet<unsigned> flags = metaData->getFlags(stat);
  const SparseSet<unsigned>::Ranges &ranges = flags.getRanges();

  for (SparseSet<unsigned>::const_iterator it = ranges.begin(); it != ranges.end(); it ++) {
    unsigned begin = itsNrChannels == 1 ? it->begin : std::max(0, (signed) (it->begin >> itsLogNrChannels) - NR_TAPS + 1);
    unsigned end   = std::min(itsNrSamplesPerIntegration, ((it->end - 1) >> itsLogNrChannels) + 1);

    for (unsigned ch = 0; ch < itsNrChannels; ch++) {
      filteredData->flags[ch][stat].include(begin, end);
    }
  }

  computeFlagsTimer.stop();
}


#if defined PPF_C_IMPLEMENTATION

template <typename SAMPLE_TYPE> fcomplex PPF<SAMPLE_TYPE>::phaseShift(unsigned time, unsigned chan, double baseFrequency, double delayAtBegin, double delayAfterEnd) const
{
  float timeInterpolatedDelay = delayAtBegin + ((float) time / itsNrSamplesPerIntegration) * (delayAfterEnd - delayAtBegin);
  float frequency	      = baseFrequency + chan * itsChannelBandwidth;
  float phaseShift	      = timeInterpolatedDelay * frequency;
  float phi		      = -2 * M_PI * phaseShift;

  return makefcomplex(std::cos(phi), std::sin(phi));
}

#else

template <typename SAMPLE_TYPE> void PPF<SAMPLE_TYPE>::computePhaseShifts(struct phase_shift phaseShifts[/*itsNrSamplesPerIntegration*/], double delayAtBegin, double delayAfterEnd, double baseFrequency) const
{
  if (itsDelayCompensation) {
    double   phiBegin = -2 * M_PI * delayAtBegin;
    double   phiEnd   = -2 * M_PI * delayAfterEnd;
    double   deltaPhi = (phiEnd - phiBegin) / itsNrSamplesPerIntegration;
    dcomplex v	      = cosisin(phiBegin * baseFrequency);
    dcomplex dv       = cosisin(phiBegin * itsChannelBandwidth);
    dcomplex vf       = cosisin(deltaPhi * baseFrequency);
    dcomplex dvf      = cosisin(deltaPhi * itsChannelBandwidth);

    for (unsigned time = 0; time < itsNrSamplesPerIntegration; time ++) {
      phaseShifts[time].v0 =  v;  v *=  vf;
      phaseShifts[time].dv = dv; dv *= dvf;
    }
  } else {
    for (unsigned time = 0; time < itsNrSamplesPerIntegration; time ++) {
      phaseShifts[time].v0 = makefcomplex(1.0f, 0.0f);
      phaseShifts[time].dv = makefcomplex(1.0f, 0.0f);
    }
  }
}

#endif // PPF_C_IMPLEMENTATION


template <typename SAMPLE_TYPE> void PPF<SAMPLE_TYPE>::filter(unsigned stat, double centerFrequency, const SubbandMetaData *metaData, const TransposedData<SAMPLE_TYPE> *transposedData, FilteredData *filteredData)
{
  PPFtimer.start();

  double   baseFrequency  = centerFrequency - (itsNrChannels * 0.5) * itsChannelBandwidth;
  unsigned alignmentShift = metaData->alignmentShift(stat);

#if 0
  LOG_DEBUG_STR(setprecision(15) << "stat " << stat << ", basefreq " << baseFrequency << ": delay from " << delays[stat].delayAtBegin << " to " << delays[stat].delayAfterEnd << " sec");
#endif

#if defined PPF_C_IMPLEMENTATION
  FIRtimer.start();

#pragma omp parallel for
  for (int chan = 0; chan < (int) itsNrChannels; chan ++) {
    for (unsigned time = 0; time < NR_TAPS - 1 + itsNrSamplesPerIntegration; time ++) {
      for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol ++) {
	SAMPLE_TYPE currSample = transposedData->samples[stat][itsNrChannels * time + chan + alignmentShift][pol];

#if defined WORDS_BIGENDIAN
	dataConvert(LittleEndian, &currSample, 1);
#endif
	fcomplex sample = makefcomplex(real(currSample), imag(currSample));
	itsFFTinData[time][pol][chan] = itsFIRs[stat][pol][chan].processNextSample(sample);
      }
    }
  }

  FIRtimer.stop();

  FFTtimer.start();

#pragma omp parallel
  {
    std::vector<fcomplex, AlignedStdAllocator<fcomplex, 32> > fftOutData(itsNrChannels);

    // The flags of all channels are still the same here, so we just use channel 1.
    // Flags are kept per channel, since we will do online flagging on FilteredData later.

#pragma omp for
    for (int time = 0; time < (int) itsNrSamplesPerIntegration; time ++) {
      for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol ++) {
	if (filteredData->flags[1][stat].test(time)) {
	  for (unsigned chan = 0; chan < itsNrChannels; chan ++)
	    filteredData->samples[chan][stat][time][pol] = makefcomplex(0, 0);
	} else {
#if defined HAVE_FFTW3
	  fftwf_execute_dft(itsFFTWPlan,
			    (fftwf_complex *) itsFFTinData[NR_TAPS - 1 + time][pol].origin(),
			    (fftwf_complex *) (void *) &fftOutData[0]);
#else
	  fftw_one(itsFFTWPlan,
		   (fftw_complex *) itsFFTinData[NR_TAPS - 1 + time][pol].origin(),
		   (fftw_complex *) (void *) &fftOutData[0]);
#endif

	  for (unsigned chan = 0; chan < itsNrChannels; chan ++) {
	    if (itsDelayCompensation)
	      fftOutData[chan] *= phaseShift(time, chan, baseFrequency, metaData->beams(stat)[0].delayAtBegin, metaData->beams(stat)[0].delayAfterEnd);

	    if (itsCorrectBandPass)
	      fftOutData[chan] *= itsBandPass.correctionFactors()[chan];

	    filteredData->samples[chan][stat][time][pol] = fftOutData[chan];
	  }
	}
      }
    }
  }

  FFTtimer.stop();

#else // assembly implementation
  int transpose_stride = sizeof(fcomplex) * (NR_POLARIZATIONS * (itsNrSamplesPerIntegration | 2) * itsNrStations - 3);

  for (unsigned chan = 0; chan < itsNrChannels; chan += 4) {
    for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol ++) {
#if defined __GNUC__	// work around bug ???
      for (register unsigned ch asm ("r28") = 0; ch < 4; ch ++)
#else
      for (unsigned ch = 0; ch < 4; ch ++)
#endif
      {
	FIRtimer.start();
	_filter(itsNrChannels,
		itsFIRs[stat][pol][chan + ch].getWeights(),
//		FIR::weights[chan + ch].origin(),
		&transposedData->samples[stat][chan + ch + alignmentShift][pol],
		itsDelayLines[ch].origin(),
		itsNrSamplesPerIntegration / NR_TAPS);
	FIRtimer.stop();
      }

      _transpose_4x8(&itsFFTinData[0][pol][chan],
		     itsDelayLines.origin(),
		     itsNrSamplesPerIntegration,
		     sizeof(fcomplex) * itsNrSamplesPerIntegration,
		     sizeof(fcomplex) * NR_POLARIZATIONS * (itsNrChannels + 4));
    }
  }

  struct phase_shift phaseShifts[itsNrSamplesPerIntegration];

  computePhaseShifts(phaseShifts, metaData->beams(stat)[0].delayAtBegin, metaData->beams(stat)[0].delayAfterEnd, baseFrequency);

  // The flags of all channels are still the same here, so we just use channel 1.
  // Flags are kept per channel, since we will do online flagging on FilteredData later.

  const SparseSet<unsigned>::Ranges &ranges = filteredData->flags[1][stat].getRanges();
  SparseSet<unsigned>::const_iterator it = ranges.begin();

  for (unsigned time = 0; time < itsNrSamplesPerIntegration; time ++) {
    bool good = it == ranges.end() || time < it->begin || (time == it->end && (++ it, true));

    if (good) {
      FFTtimer.start();
#if 0
      _prefetch(itsFFTinData[time].origin(),
		sizeof(fcomplex[NR_POLARIZATIONS][itsNrChannels]) / CACHE_LINE_SIZE,
		CACHE_LINE_SIZE);
#endif

      for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol ++)
	if(itsNrChannels == 256)
	  _fft256(itsFFTinData[time][pol].origin(),
		  itsFFToutData[time & 1][pol].origin());
	else
	  fftw_one(itsFFTWPlan,
		   (fftw_complex *) itsFFTinData[time][pol].origin(),
		   (fftw_complex *) itsFFToutData[time & 1][pol].origin());

      FFTtimer.stop();
    } else {
	_memzero(itsFFToutData[time & 1].origin(),
		 itsFFToutData[time & 1].num_elements() * sizeof(fcomplex));
    }

    if (time & 1)
      _phase_shift_and_transpose(&filteredData->samples[0][stat][time - 1][0],
				 itsFFToutData.origin(),
				 &phaseShifts[time - 1],
				 transpose_stride,
				 itsNrChannels,
				 itsBandPass.correctionFactors());
  }
#endif // PPF_C_IMPLEMENTATION

  PPFtimer.stop();
}


template <typename SAMPLE_TYPE> void PPF<SAMPLE_TYPE>::bypass(unsigned stat, double frequency, const SubbandMetaData *metaData, const TransposedData<SAMPLE_TYPE> *transposedData, FilteredData *filteredData)
{
  PPFtimer.start();

  unsigned alignmentShift = metaData->alignmentShift(stat);

  // The flags of all channels are still the same here, so we just use channel 1.
  // Flags are kept per channel, since we will do online flagging on FilteredData later.

#if defined PPF_C_IMPLEMENTATION
  for (unsigned time = 0; time < itsNrSamplesPerIntegration; time ++) {
    for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol ++) {
      if ((itsNrChannels > 1 && filteredData->flags[1][stat].test(time)) || (itsNrChannels == 1 && filteredData->flags[0][stat].test(time))) {
	filteredData->samples[0][stat][time][pol] = makefcomplex(0, 0);
      } else {
	SAMPLE_TYPE currSample = transposedData->samples[stat][time + alignmentShift][pol];

#if defined WORDS_BIGENDIAN
	dataConvert(LittleEndian, &currSample, 1);
#endif

	fcomplex sample = makefcomplex(real(currSample), imag(currSample));

	if (itsDelayCompensation)
	  sample *= phaseShift(time, 0, frequency, metaData->beams(stat)[0].delayAtBegin, metaData->beams(stat)[0].delayAfterEnd);

	filteredData->samples[0][stat][time][pol] = sample;
      }
    }
  }
#else // assembly implementation
  // convert little-endian integers to floating point
  _convert(filteredData->samples[0][stat].origin(), transposedData->samples[stat][alignmentShift].origin(), itsNrSamplesPerIntegration * NR_POLARIZATIONS);

  if (itsDelayCompensation) {
    double   phiBegin = -2 * M_PI * metaData->beams(stat)[0].delayAtBegin;
    double   phiEnd   = -2 * M_PI * metaData->beams(stat)[0].delayAfterEnd;
    double   deltaPhi = (phiEnd - phiBegin) / itsNrSamplesPerIntegration;
    dcomplex v	 __attribute__((aligned(16))) = cosisin(phiBegin * frequency);
    dcomplex vf  __attribute__((aligned(16))) = cosisin(deltaPhi * frequency) ;

    _apply_single_channel_delays(filteredData->samples[0][stat].origin(), itsNrSamplesPerIntegration, &v, &vf);
  }

  // clear flagged data
  const SparseSet<unsigned>::Ranges &ranges = filteredData->flags[0][stat].getRanges();

  for (SparseSet<unsigned>::const_iterator it = ranges.begin(); it != ranges.end(); it ++)
    memset(filteredData->samples[0][stat][it->begin].origin(), 0, (it->end - it->begin) * NR_POLARIZATIONS * sizeof(fcomplex));
#endif

  PPFtimer.stop();
}


template <typename SAMPLE_TYPE> void PPF<SAMPLE_TYPE>::doWork(unsigned stat, double frequency, const SubbandMetaData *metaData, const TransposedData<SAMPLE_TYPE> *transposedData, FilteredData *filteredData)
{
  computeFlags(stat, metaData, filteredData);

  if (itsNrChannels > 1)
    filter(stat, frequency, metaData, transposedData, filteredData);
  else
    bypass(stat, frequency, metaData, transposedData, filteredData);
}


template class PPF<i4complex>;
template class PPF<i8complex>;
template class PPF<i16complex>;

} // namespace RTCP
} // namespace LOFAR
