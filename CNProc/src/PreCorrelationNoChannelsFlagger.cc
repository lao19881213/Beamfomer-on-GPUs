//# Always #include <lofar_config.h> first!
#include <lofar_config.h>

#include <Common/Timer.h>

#include <PreCorrelationNoChannelsFlagger.h>

#if SAVE_REAL_TIME_FLAGGER_INTERMEDIATE_DEBUG
#include <Interface/CN_Mapping.h>
#endif

/*
  Some notes: 
  - We cannot integrate by adding samples, and then taking the power. We have to calculate the power for each sample, 
    and then add the powers.
  - The history is kept per subband, as we can get different subbands over time on this compute node.
  - We always flag poth polarizations seperately, and then take the union.
  - An FFT followed by an inverse FFT multiplies all samples by N. Thus, we have to divide by N after we are done.

  We first flag in the frequency direction, as most RFI is narrowband.
  Next, we flag in the time direction, while integrating to improve signal-to-noise.
  This was empirically verified to work much better than flagging on the raw data.
  We can then replace flagged samples with 0s or mean/median values.

  For the flagging in the frequency direction, we do the following:

  - Move over raw data in full time resolution, in chunks of the FFT size.
  - Do the FFT.
  - Perform bandpass correction. Here, we correct for the bandpass of the station polyphase filter bank.
    We found that the ripple introduced by the station PPF can cause many false positives in specific frequencies.
    Bandpass correction only is one multiplication, and it completely removes this effect.
  - Computer powers of the FFT-ed data.
    NOTE, both itsPowers and itsFlagsFrequency are stored in FFTshifted order!
  - Add powers of the chunks together to integrate over time, increasing signal-to-noise.
  - Flag on the integrated data, using SumThreshold, recording which frequencies are poluted.
    We perform two SumThreshold passes: a first pass, and then recalculate statistics while omitting the 
    flagged data, then second pass with the corrected statistics.
    This way, very strong RFI does not polute the stddev, etc. Making more than two passes was emperically found to 
    improve the result only very slightly, at high computational costs.
  - Next, we make another pass on the raw data, to replace flagged samples with corrected values in the full resolution. 
  - This means: FFT
    We have several options here:
     * Replace with 0. This removes the RFI, but decreases the total power, and causes jumps in the output signal. 
       It is therefore undesirable.
     * Replace with a sample where the imaginary part is 0, and the real part is the mean of the reals of the unflagged samples
     * Replace with a sample where the imaginary part is 0, and the real part is the sqrt of the mean power of the 
       unflagged samples.
       This way, we keep the total signal power the same. This currently is the best method.
     * Suggestion, TODO: replace with a random imaginary part, and keep the total power the same.
@@@ replace with median
  - Inverse fft.
*/

// TODO iteratively change the integration time when flagging in time? Is this really needed? 
//      SumThreshold already does this by changing the window sizes...
// TODO: for both time and freq, if more than p% of a block is flagged, don't replace with median, but with sample from historical data?
// TODO: keep a history of frequency as well: one history per channel. We can use this both for flagging and replacing samples...

namespace LOFAR {
namespace RTCP {

#if SAVE_REAL_TIME_FLAGGER_INTERMEDIATE_DEBUG
static  FILE* intermediateOutputFile;
#endif
#if SAVE_REAL_TIME_FLAGGER_REPLACED_DEBUG
static  FILE* replacedOutputFile;
#endif


  PreCorrelationNoChannelsFlagger::PreCorrelationNoChannelsFlagger(const Parset& parset, unsigned myPset, unsigned myCoreInPset, 
								   bool correctBandPass, const unsigned nrStations, 
								   const unsigned nrSubbands, const unsigned nrChannels, 
								   const unsigned nrSamplesPerIntegration, 
								   const float cutoffThreshold)
:
  Flagger(parset, nrStations, nrSubbands, nrChannels, cutoffThreshold, 
	  /*baseSentitivity*/ 0.6f, // 0.6 was emperically found to be a good setting for LOFAR
	  getFlaggerStatisticsType(parset.onlinePreCorrelationFlaggingStatisticsType(getFlaggerStatisticsTypeString(FLAGGER_STATISTICS_WINSORIZED)))),
  itsNrSamplesPerIntegration(nrSamplesPerIntegration), itsCorrectBandPass(correctBandPass), itsBandPass(correctBandPass, itsFFTSize)
{
  assert(itsNrSamplesPerIntegration % itsFFTSize == 0);
  assert(nrChannels == 1);

  itsIntegrationFactor = itsNrSamplesPerIntegration / itsFFTSize;

  LOG_DEBUG_STR("PreCorrelationNoChannelsFlagger: nrSamplesPerIntegration = " << itsNrSamplesPerIntegration << ", fft size = " << itsFFTSize << ", integrationFactor = " << itsIntegrationFactor);

  itsSamples.resize(itsFFTSize);
  itsFFTBuffer.resize(itsFFTSize);

  itsPowers.resize(NR_POLARIZATIONS);
  itsFlagsFrequency.resize(NR_POLARIZATIONS);
  itsFlagsTime.resize(NR_POLARIZATIONS);
  for(unsigned pol=0; pol<NR_POLARIZATIONS; pol++) {
    itsPowers[pol].resize(itsFFTSize);
    itsFlagsFrequency[pol].resize(itsFFTSize);
    itsFlagsTime[pol].resize(itsFFTSize);
  }

#if USE_HISTORY_FLAGGER
  itsHistory.resize(boost::extents[itsNrStations][nrSubbands]);
#endif

  initFFT();

  if(itsCorrectBandPass) {
    LOG_DEBUG_STR("PreCorrelationNoChannelsFlagger: bandpass correction enabled");
  } else {
    LOG_DEBUG_STR("PreCorrelationNoChannelsFlagger: bandpass correction disabled");
  }

#if SAVE_REAL_TIME_FLAGGER_INTERMEDIATE_DEBUG 
  stringstream intermediateFileName;
  intermediateFileName << "/var/scratch/rob/" << myPset << "." << myCoreInPset << ".myIntermediateData";
  intermediateOutputFile = fopen(intermediateFileName.str().c_str(), "w");
  fwrite(&itsNrStations, sizeof(unsigned), 1, intermediateOutputFile);
  fwrite(&itsNrSubbands, sizeof(unsigned), 1, intermediateOutputFile);
  unsigned tmp = itsFFTSize;
  fwrite(&tmp, sizeof(unsigned), 1, intermediateOutputFile);
  tmp = NR_POLARIZATIONS;
  fwrite(&tmp, sizeof(unsigned), 1, intermediateOutputFile);
  fflush(intermediateOutputFile);
#else
  // avoid warnings
  (void)myPset;
  (void)myCoreInPset;
#endif // SAVE_REAL_TIME_FLAGGER_INTERMEDIATE_DEBUG

#if SAVE_REAL_TIME_FLAGGER_REPLACED_DEBUG 
  stringstream replacedFileName;
  replacedFileName << "/var/scratch/rob/" << myPset << "." << myCoreInPset << ".myReplacedData";
  replacedOutputFile = fopen(replacedFileName.str().c_str(), "w");
  fwrite(&itsNrStations, sizeof(unsigned), 1, replacedOutputFile);
  fwrite(&itsNrSubbands, sizeof(unsigned), 1, replacedOutputFile);
  unsigned tmp2 = itsFFTSize;
  fwrite(&tmp2, sizeof(unsigned), 1, replacedOutputFile);
  tmp2 = NR_POLARIZATIONS;
  fwrite(&tmp2, sizeof(unsigned), 1, replacedOutputFile);
  fflush(replacedOutputFile);
#else
  // avoid warnings
  (void)myPset;
  (void)myCoreInPset;
#endif // SAVE_REAL_TIME_FLAGGER_REPLACED_DEBUG
}


void PreCorrelationNoChannelsFlagger::flag(FilteredData* filteredData, unsigned globalTime, unsigned subband)
{
  NSTimer flaggerTimer("RFI noChannels flagger total", true, true);

  flaggerTimer.start();
  for(unsigned station = 0; station < itsNrStations; station++) {
    flagStation(filteredData, globalTime, station, subband);
  }
  flaggerTimer.stop();
}


void PreCorrelationNoChannelsFlagger::flagStation(FilteredData* filteredData, unsigned globalTime, unsigned station, unsigned subband)
{
  NSTimer flaggerTimeTimer("RFI noChannels time flagger", true, true);
  NSTimer flaggerFrequencyTimer("RFI noChannels frequency flagger", true, true);


#if FLAG_IN_FREQUENCY_DIRECTION
  flaggerFrequencyTimer.start();

  for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol++) {
    // init frequency flags
    for (unsigned i = 0; i < itsFFTSize; i++) {
      itsFlagsFrequency[pol][i] = false;
    }

    calcIntegratedPowersFrequency(filteredData, station, subband, pol);

    // Flag twice, the second time with corrected statistics
    sumThresholdFlagger1D(itsPowers[pol], itsFlagsFrequency[pol], itsBaseSensitivity);
    sumThresholdFlagger1D(itsPowers[pol], itsFlagsFrequency[pol], itsBaseSensitivity);
  }

  // Compute the union of flags of the polarizations
  takeUnionOfFlags(itsFlagsFrequency);

  // Scale-Invariant-Rank operator, to expand the flagged windows a bit, and to fill in the holes.
  unsigned flaggedCountFrequency = SIROperator(itsFlagsFrequency[0], 0.4f); 

  LOG_DEBUG_STR("samples flagged in frequency: " << flaggedCountFrequency);

  if(flaggedCountFrequency == itsFFTSize) {
    LOG_DEBUG_STR("all samples flagged in frequency!");
  }

#if SAVE_REAL_TIME_FLAGGER_INTERMEDIATE_DEBUG
  for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol++) {
    saveIntermediate(globalTime, station, subband, pol, false);
  }
#endif

#if 0
#if SAVE_REAL_TIME_FLAGGER_REPLACED_DEBUG
  // do forward FFT; fix samples; backward FFT on the original samples in full resolution
  applyFlagsFrequency(filteredData, globalTime, station, subband, flaggedCountFrequency); 
#else
  if(flaggedCountFrequency > 0) {
    // do forward FFT; fix samples; backward FFT on the original samples in full resolution
    applyFlagsFrequency(filteredData, station, subband, flaggedCountFrequency); 
  }
#endif
#endif
  flaggerFrequencyTimer.stop();
#endif // FLAG_IN_FREQUENCY_DIRECTION


#if FLAG_IN_TIME_DIRECTION
  flaggerTimeTimer.start();
  initFlagsTime(filteredData, station); // copy flags to my local format

  for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol++) {
    calcIntegratedPowersTime(filteredData, station, subband, pol);
    sumThresholdFlagger1D(itsPowers[pol], itsFlagsTime[pol], itsBaseSensitivity); // Flag in time direction.
    sumThresholdFlagger1D(itsPowers[pol], itsFlagsTime[pol], itsBaseSensitivity); // Flag twice, the second time with corrected statistics.
  }

  takeUnionOfFlags(itsFlagsTime);

  // Scale-Invariant-Rank operator, to expand the flagged windows a bit, and to fill in the holes.
  unsigned flaggedCountTime = SIROperator(itsFlagsTime[0], 0.4f);

  LOG_DEBUG_STR("samples flagged in time: " << flaggedCountTime);

#if USE_HISTORY_FLAGGER
  if(flaggedCountTime < itsFFTSize) { // If everything was already flagged, skip this entirely.
    // TODO just compute mean inline here, we don't need median and stddev
    float mean0, median0, stdDev0, mean1, median1, stdDev1;
    calculateWinsorizedStatistics(itsPowers[0], itsFlagsTime[0], mean0, median0, stdDev0);
    calculateWinsorizedStatistics(itsPowers[1], itsFlagsTime[0], mean1, median1, stdDev1); // take flags at index 0, they are unified.

    // I have empirically found that the mean of the unflagged samples is a better predictor than the median, for history flagging at least.
    if(addToHistory((mean0 + mean1)/(2.0f * itsIntegrationFactor) /*meanPower*/, itsHistory[station][subband], 10.0f)) {
      LOG_DEBUG_STR("History flagger flagged this second " << globalTime << " for station " << station << " subband " << subband);
      for(unsigned i=0; i<itsFFTSize; i++) {
	itsFlagsTime[0][i] = true;
      }
      flaggedCountTime = itsFFTSize;
    }
  }
#endif // USE_HISTORY_FLAGGER
    
  if(flaggedCountTime > 0) {
    // copy flags from my original format into FilteredData again.
    applyFlagsTime(filteredData, station, subband, flaggedCountTime);
  }
  
  flaggerTimeTimer.stop();
#endif
}
  

void PreCorrelationNoChannelsFlagger::calcIntegratedPowersFrequency(FilteredData* filteredData, unsigned station, unsigned subband, unsigned pol)
{
  (void) subband; // avoids compiler warning

  memset(&itsPowers[pol][0], 0, itsFFTSize * sizeof(float));

  for(unsigned block=0; block<itsIntegrationFactor; block++) {
    unsigned startIndex = block * itsFFTSize;

    for(unsigned minorTime=0; minorTime<itsFFTSize; minorTime++) {
      itsSamples[minorTime] = filteredData->samples[0][station][startIndex + minorTime][pol];
    }

    forwardFFT();

    for (unsigned i = 0; i < itsFFTSize; i++) { // compute powers from FFT-ed data
      unsigned fftShiftedPos = ((itsFFTSize / 2) + i) % itsFFTSize;
      fcomplex sample = itsFFTBuffer[i];
      if (itsCorrectBandPass) {
	sample *= itsBandPass.correctionFactors()[fftShiftedPos]; // do not just use index i, we do an FFTshift...
      }

      float power = real(sample) * real(sample) + imag(sample) * imag(sample);
      itsPowers[pol][fftShiftedPos] += power;
    }
  }
}


void PreCorrelationNoChannelsFlagger::calcIntegratedPowersTime(FilteredData* filteredData, unsigned station, unsigned subband, unsigned pol)
{
  (void) subband; // avoids compiler warning

  memset(&itsPowers[pol][0], 0, itsFFTSize * sizeof(float));
 
  for(unsigned t=0; t<itsNrSamplesPerIntegration; t++) {
    fcomplex sample = filteredData->samples[0][station][t][pol];
    itsPowers[pol][t/itsIntegrationFactor] += real(sample) * real(sample) + imag(sample) * imag(sample);
  }
}


void PreCorrelationNoChannelsFlagger::initFlagsTime(FilteredData* filteredData, unsigned station)
{
  for (unsigned pol=0; pol < NR_POLARIZATIONS; pol++) {
    for (unsigned i = 0; i < itsFFTSize; i++) {
      itsFlagsTime[pol][i] = false;
    }
  }

  // Use the original flags to initialize the flags.
  // This could be done much faster by just iterating over the windows in the sparse flags set.
  for (unsigned time = 0; time < itsNrSamplesPerIntegration; time++) {
    if(filteredData->flags[0][station].test(time)) {
      for (unsigned pol=0; pol < NR_POLARIZATIONS; pol++) {
	itsFlagsTime[pol][time/itsIntegrationFactor] = true;
      }
    }
  }
}


// Do forward FFT; fix samples; backward FFT on the original samples in full resolution. Flags are already set in itsFlagsFrequency.
// FFT followed by an inverse FFT multiplies all samples by N. Thus, we have to divide by N after we are done.
// NOTE, itsFlagsFrequency are stored in FFTshifted order!
void PreCorrelationNoChannelsFlagger::applyFlagsFrequency(FilteredData* filteredData, unsigned globalTime, unsigned station, unsigned subband, unsigned nrFlaggedSamples)
{
#if SAVE_REAL_TIME_FLAGGER_REPLACED_DEBUG
      std::vector<float> tmp;
      tmp.resize(itsFFTSize);
      for(unsigned minorTime=0; minorTime < itsFFTSize; minorTime++) {
	tmp[minorTime] = 0.0f;
      }
#endif

  for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol++) {
    for (unsigned majorTime = 0; majorTime < itsIntegrationFactor; majorTime++) {
      unsigned startIndex = majorTime * itsFFTSize;
      for(unsigned minorTime=0; minorTime < itsFFTSize; minorTime++) {
	itsSamples[minorTime] = filteredData->samples[0][station][startIndex+minorTime][pol];
      }
      forwardFFT();

      fcomplex replacementValue = computeReplacementValueFrequency(station, subband, pol, nrFlaggedSamples);

      // Replace all flagged samples.
      // Note, the flags are stored in the order of the real frequencies (ie. FFTshifted).
      for(unsigned minorTime=0; minorTime < itsFFTSize; minorTime++) {
	unsigned fftShiftedPos = ((itsFFTSize / 2) + minorTime) % itsFFTSize;
	if(itsFlagsFrequency[0][fftShiftedPos]) {
	  itsFFTBuffer[minorTime] = replacementValue;
	  if (itsCorrectBandPass) {
	    itsFFTBuffer[minorTime] /= itsBandPass.correctionFactors()[fftShiftedPos];
	  }
	}
      }
      
#if SAVE_REAL_TIME_FLAGGER_REPLACED_DEBUG
      for(unsigned c=0; c<itsFFTSize; c++) {
	fcomplex sample = itsFFTBuffer[c];
	unsigned fftShiftedPos = ((itsFFTSize / 2) + c) % itsFFTSize;
	if (itsCorrectBandPass) {
	  sample *= itsBandPass.correctionFactors()[fftShiftedPos]; // do not just use index i, we do an FFTshift...
	}
	float power = real(sample) * real(sample) + imag(sample) * imag(sample);
	tmp[fftShiftedPos] += power;
      }
#endif
      
      backwardFFT();
      for(unsigned minorTime=0; minorTime < itsFFTSize; minorTime++) {
	// FFT followed by an inverse FFT multiplies all samples by N. Thus, we have to divide by N after we are done.
	filteredData->samples[0][station][startIndex+minorTime][pol] = makefcomplex(real(itsSamples[minorTime]) / itsFFTSize, imag(itsSamples[minorTime]) / itsFFTSize);
      }
    }

#if SAVE_REAL_TIME_FLAGGER_REPLACED_DEBUG
      unsigned tmpu = true;
      fwrite(&tmpu, sizeof(unsigned), 1, replacedOutputFile);
      fwrite(&globalTime, sizeof(unsigned), 1, replacedOutputFile);
      fwrite(&station, sizeof(unsigned), 1, replacedOutputFile);
      fwrite(&pol, sizeof(unsigned), 1, replacedOutputFile);
      fwrite(&subband, sizeof(unsigned), 1, replacedOutputFile);
      fwrite(&tmp[0], itsFFTSize * sizeof(float), 1, replacedOutputFile);
      fflush(replacedOutputFile);
#endif
  }
}


// I found that time replacement occasionally (only 1x in our dataset) replaces with values that are too high, if some samples were not flagged correctly.
// So, as a sanity check, we verify replacement value against history?
// replace samples. This can be removed if the beamformer / pulsar pipeline correctly handles flags for stokesI.
void PreCorrelationNoChannelsFlagger::applyFlagsTime(FilteredData* filteredData, unsigned station, unsigned subband, unsigned nrFlaggedSamples)
{
  filteredData->resetFlags(); // Wipe original flags
 
  // include data in orgiginal flags
  for (unsigned i = 0; i < itsFFTSize; i++) { 
    if(itsFlagsTime[0][i]) {
      unsigned startIndex = i * itsIntegrationFactor;
      filteredData->flags[0][station].include(startIndex, startIndex+itsIntegrationFactor);
    }
  }

  for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol++) {
    fcomplex replacementValue = computeReplacementValueTime(filteredData, station, subband, pol, nrFlaggedSamples);

    // and replace
    for (unsigned i = 0; i < itsFFTSize; i++) {
      if(itsFlagsTime[0][i]) {
	unsigned startIndex = i * itsIntegrationFactor;
	for(unsigned pos = startIndex; pos < startIndex+itsIntegrationFactor; pos++) {
	  filteredData->samples[0][station][pos][pol] = replacementValue;
	}
      }
    }
  }
}


fcomplex PreCorrelationNoChannelsFlagger::computeReplacementValueFrequency(unsigned station, unsigned subband, unsigned pol, unsigned nrFlaggedSamples)
{
#if REPLACEMENT_METHOD == REPLACE_WITH_ZERO
  (void) station; (void) subband; (void) pol; (void) nrFlaggedSamples; // prevent compiler warning
  return makefcomplex(0.0f, 0.0f);

#elif REPLACEMENT_METHOD == REPLACE_WITH_MEAN
  // take the mean power of unflagged samples, and create a sample where the total power is the same, with imag=0
  (void) station; (void) subband; (void) pol; // prevent compiler warning
  float meanPower = 0.0f;
  for(unsigned i=0; i < itsFFTSize; i++) {
    if(!itsFlagsFrequency[0][i]) { // The union of the flags for both polarizations is at index 0
      meanPower += power(itsFFTBuffer[i]);
    }
  }
  meanPower /= (itsFFTSize - nrFlaggedSamples);
  return makefcomplex(sqrtf(meanPower), 0.0f);

#elif REPLACEMENT_METHOD == REPLACE_WITH_RANDOM
  // take a random, unflagged sample.
  (void) station; (void) subband; (void) pol; (void) nrFlaggedSamples; // prevent compiler warning
  fcomplex replacementValue = makefcomplex(0.0f, 0.0f);
  for(unsigned minorTime=0; minorTime < itsFFTSize; minorTime++) {
    unsigned fftShiftedPos = ((itsFFTSize / 2) + minorTime) % itsFFTSize;
    if(!itsFlagsFrequency[0][fftShiftedPos]) { // The union of the flags for both polarizations is at index 0
      replacementValue = itsFFTBuffer[minorTime];
      if (itsCorrectBandPass) {
	replacementValue *= itsBandPass.correctionFactors()[fftShiftedPos];
      }
      return replacementValue;
    }
  }
  LOG_DEBUG_STR("replace frequency random: no unflagged samples, returning zero");
  return replacementValue;

#elif REPLACEMENT_METHOD == REPLACE_WITH_MEDIAN
  // Take median of unflagged samples.
  (void) station; (void) subband; (void) pol; (void) nrFlaggedSamples; // prevent compiler warning
  fcomplex replacementValue = makefcomplex(0.0f, 0.0f);
  std::vector<float> powers;
  powers.resize(itsFFTSize);
  for (unsigned i = 0; i < itsFFTSize; i++) { // compute powers from FFT-ed data
    unsigned fftShiftedPos = ((itsFFTSize / 2) + i) % itsFFTSize;
    fcomplex sample = itsFFTBuffer[i];
    float power = real(sample) * real(sample) + imag(sample) * imag(sample);
    if (itsCorrectBandPass) {
      sample *= itsBandPass.correctionFactors()[fftShiftedPos]; // do not just use index i, we do an FFTshift...
    }
    
    powers[fftShiftedPos] = power;
  }
  float median;
  unsigned medianIndex = calculateMedian(powers, itsFlagsFrequency[0], median);
  unsigned fftUnshiftedPos = ((itsFFTSize / 2) + medianIndex) % itsFFTSize; // undo fft shift, only works if fft size is even
  assert(!itsFlagsFrequency[0][medianIndex]);
  replacementValue = itsFFTBuffer[fftUnshiftedPos];
  if (itsCorrectBandPass) {
    replacementValue *= itsBandPass.correctionFactors()[medianIndex];
  }

  return replacementValue;
#else
#error unsupported replacement method
#endif
}


fcomplex PreCorrelationNoChannelsFlagger::computeReplacementValueTime(FilteredData* filteredData, unsigned station, unsigned subband, unsigned pol, unsigned nrFlaggedSamples)
{
#if REPLACEMENT_METHOD == REPLACE_WITH_ZERO
  (void) station; (void) subband; (void) pol; (void) nrFlaggedSamples; // prevent compiler warning
  return makefcomplex(0.0f, 0.0f);

#elif REPLACEMENT_METHOD == REPLACE_WITH_MEAN
  float meanPower = 0.0f;
  if(nrFlaggedSamples == itsFFTSize) {
#if USE_HISTORY_FLAGGER
      return computeReplacementValueFromHistoryTime(station, subband);
#else
      return makefcomplex(0.0f, 0.0f);
#endif
  } else {
    // compute mean power of the unflagged samples
    for(unsigned i=0; i < itsFFTSize; i++) {
      if(!itsFlagsTime[0][i]) { // The union of the flags for both polarizations is at index 0
	meanPower += itsPowers[pol][i];
      }
    }
    meanPower /= (itsFFTSize - nrFlaggedSamples) * itsIntegrationFactor;
  }
    
  meanPower = sqrtf(meanPower);
  fcomplex replacementValue = makefcomplex(meanPower, 0.0f);
  return replacementValueTimeSanityCheck(station, subband, replacementValue);

#elif REPLACEMENT_METHOD == REPLACE_WITH_RANDOM
    if(nrFlaggedSamples == itsFFTSize) {
#if USE_HISTORY_FLAGGER
      return computeReplacementValueFromHistoryTime(station, subband);
#else
      return makefcomplex(0.0f, 0.0f);
#endif
    } else {
      // replace with random (or first) sample of random (first) non-flagged block
      fcomplex replacementValue = makefcomplex(0.0f, 0.0f);
      for (unsigned i = 0; i < itsFFTSize; i++) {
	if(!itsFlagsTime[0][i]) {
	  replacementValue = filteredData->samples[0][station][i*itsIntegrationFactor][pol];
	  break;
	}
      }
      return replacementValueTimeSanityCheck(station, subband, replacementValue);
    }

#elif REPLACEMENT_METHOD == REPLACE_WITH_MEDIAN
    if(nrFlaggedSamples == itsFFTSize) {
#if USE_HISTORY_FLAGGER
      return computeReplacementValueFromHistoryTime(station, subband);
#else
      return makefcomplex(0.0f, 0.0f);
#endif
    } else {
      // replace with median of random (or first) non-flagged block
      fcomplex replacementValue = makefcomplex(0.0f, 0.0f);
      for (unsigned i = 0; i < itsFFTSize; i++) {
	if(!itsFlagsTime[0][i]) {
	  std::vector<float> samples;
	  samples.resize(itsIntegrationFactor);
	  unsigned startIndex = i*itsIntegrationFactor;
	  for(unsigned s=0; s<itsIntegrationFactor; s++) {
	    samples[s] = power(filteredData->samples[0][station][startIndex + s][pol]);
	  }
	  float median;
	  int medianIndex = calculateMedian(samples, median);
	  replacementValue = filteredData->samples[0][station][startIndex + medianIndex][pol];
	  break;
	}
      }
      return replacementValueTimeSanityCheck(station, subband, replacementValue);
    }
#else
#error not supported
#endif
}


#if USE_HISTORY_FLAGGER
// This is used in two cases:
// - If all samples are flagged, either by the time flagger, or by the historical flagger.
// - If the sanity check is triggered beacuse the computed replacement is very high compared to the history. 
// In both cases, we cannot compute a (good) median of the non-flagged data in this block.
// So, we take a previous value from the history.
fcomplex PreCorrelationNoChannelsFlagger::computeReplacementValueFromHistoryTime(unsigned station, unsigned subband)
{
  LOG_DEBUG_STR("replace time: replacing with historic data");
  float meanOfMeanPower = itsHistory[station][subband].getMean();
  return makefcomplex(sqrtf(meanOfMeanPower), 0.0f);
}
#endif // USE_HISTORY_FLAGGER


fcomplex PreCorrelationNoChannelsFlagger::replacementValueTimeSanityCheck(unsigned station, unsigned subband, fcomplex replacementValue)
{
#if USE_HISTORY_FLAGGER
    float meanOfMeanPower = itsHistory[station][subband].getMean();
    float stdDev = itsHistory[station][subband].getStdDev();

    if(power(replacementValue) > meanOfMeanPower /*+ 7.0f * stdDev*/) {
      // Replace with meanPower from history.
      LOG_DEBUG_STR("sanity check time flagger, station: " << station << ", subband: " << subband 
		    << ", replacement: " << power(replacementValue) << ", mean history power: " << meanOfMeanPower << ", history stddev: " << stdDev);
      return computeReplacementValueFromHistoryTime(station, subband);
    } else {
      LOG_DEBUG_STR("sanity check time flagger NOT triggered");
    }
#endif // USE_HISTORY_FLAGGER

    if(replacementValue == makefcomplex(0.0f, 0.0f)) {
      LOG_DEBUG_STR("time flagger replaces with zero, station: " << station << ", subband: " << subband);
    }
    return replacementValue;
}


unsigned PreCorrelationNoChannelsFlagger::takeUnionOfFlags(vector<vector<bool> >& flags)
{
  // Compute the union of flags of the polarizations.
  unsigned flaggedCount = 0;
  for(unsigned minorTime=0; minorTime < itsFFTSize; minorTime++) {
    for(unsigned pol=1; pol<NR_POLARIZATIONS; pol++) {
      flags[0][minorTime] = flags[0][minorTime] | flags[pol][minorTime];
    }
    if(flags[0][minorTime]) {
      flaggedCount++;
    }
  }

  return flaggedCount;
}


void PreCorrelationNoChannelsFlagger::initFFT()
{
#if defined HAVE_FFTW3
  itsFFTWforwardPlan =  fftwf_plan_dft_1d(itsFFTSize, (fftwf_complex *) &itsSamples[0], (fftwf_complex *) &itsFFTBuffer[0], FFTW_FORWARD, FFTW_MEASURE);
  itsFFTWbackwardPlan = fftwf_plan_dft_1d(itsFFTSize, (fftwf_complex *) &itsFFTBuffer[0], (fftwf_complex *) &itsSamples[0], FFTW_FORWARD, FFTW_MEASURE);
#elif defined HAVE_FFTW2
  itsFFTWforwardPlan  = fftw_create_plan(itsFFTSize, FFTW_FORWARD,  FFTW_ESTIMATE);
  itsFFTWbackwardPlan = fftw_create_plan(itsFFTSize, FFTW_BACKWARD, FFTW_ESTIMATE);
#endif
}


void PreCorrelationNoChannelsFlagger::forwardFFT()
{
#if defined HAVE_FFTW3
  fftwf_execute(itsFFTWforwardPlan);
#elif defined HAVE_FFTW2
  fftw_one(itsFFTWforwardPlan, (fftw_complex *) &itsSamples[0], (fftw_complex *) &itsFFTBuffer[0]);
#endif
}


void PreCorrelationNoChannelsFlagger::backwardFFT()
{
#if defined HAVE_FFTW3
  fftwf_execute(itsFFTWbackwardPlan);
#elif defined HAVE_FFTW2
  fftw_one(itsFFTWbackwardPlan,( fftw_complex *) &itsFFTBuffer[0], (fftw_complex *) &itsSamples[0]);
#endif
}


#if SAVE_REAL_TIME_FLAGGER_INTERMEDIATE_DEBUG
void PreCorrelationNoChannelsFlagger::saveIntermediate(unsigned globalTime, unsigned station, unsigned subband, unsigned pol, bool isFlagged)
{
  unsigned tmp = isFlagged;
  fwrite(&tmp, sizeof(unsigned), 1, intermediateOutputFile);
  fwrite(&globalTime, sizeof(unsigned), 1, intermediateOutputFile);
  fwrite(&station, sizeof(unsigned), 1, intermediateOutputFile);
  fwrite(&pol, sizeof(unsigned), 1, intermediateOutputFile);
  fwrite(&subband, sizeof(unsigned), 1, intermediateOutputFile);

  for(unsigned c=0; c<itsFFTSize; c++) {
    float val = (isFlagged && itsFlagsFrequency[pol][c]) ? -1.0f : itsPowers[pol][c];
    fwrite(&val, sizeof(float), 1, intermediateOutputFile);
  }

  fflush(intermediateOutputFile);
}
#endif // OUTPUT_REAL_TIME_FLAGGER_DEBUG


PreCorrelationNoChannelsFlagger::~PreCorrelationNoChannelsFlagger()
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

#if SAVE_REAL_TIME_FLAGGER_INTERMEDIATE_DEBUG 
  fclose(intermediateOutputFile);
#endif
#if SAVE_REAL_TIME_FLAGGER_INTERMEDIATE_DEBUG 
  fclose(replacedOutputFile);
#endif
}

} // namespace RTCP
} // namespace LOFAR
