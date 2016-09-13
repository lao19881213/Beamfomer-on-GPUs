//# Always #include <lofar_config.h> first!
#include <lofar_config.h>

#include <Flagger.h>
#include <Common/LofarLogger.h>
#include <Common/Timer.h>

#include <math.h>
#include <algorithm>
#include <string.h>
#include <vector>

#include <boost/lexical_cast.hpp>

#define MAX_SUM_THRESHOLD_ITERS 7

namespace LOFAR {
namespace RTCP {

//using boost::lexical_cast;

static NSTimer RFIStatsTimer("RFI post statistics calculations", true, true);

Flagger::Flagger(const Parset& parset, const unsigned nrStations, const unsigned nrSubbands, const unsigned nrChannels, const float cutoffThreshold, 
		 float baseSentitivity, FlaggerStatisticsType flaggerStatisticsType) :
    itsParset(parset), itsNrStations(nrStations), itsNrSubbands(nrSubbands), itsNrChannels(nrChannels), itsCutoffThreshold(cutoffThreshold), 
  itsBaseSensitivity(baseSentitivity), itsFlaggerStatisticsType(flaggerStatisticsType)
{
}


void Flagger::calculateMeanAndStdDev(const std::vector<float>& powers, float& mean, float& stdDev) {
  stdDev = 0.0f;
  mean = 0.0f;

  // Calculate mean value.
  for (unsigned i = 0; i < powers.size(); i++) {
      mean += powers[i];
  }
  mean /= powers.size();

  // Calculate standard deviation.
  for (unsigned i = 0; i < powers.size(); i++) {
      float diff = powers[i] - mean;
      stdDev += diff * diff;
  }

  stdDev /= powers.size();
  stdDev = sqrtf(stdDev);
}


// TODO, write version that avoid linear scan for index (often not needed)
unsigned Flagger::calculateMedian(const std::vector<float>& powers, float& median) { // calculate median, return position of the element
  // we have to copy the vector, nth_element changes the ordering.
  // TODO create copy without flagged elements
  std::vector<float> copy(powers);

  // calculate median, expensive, but nth_element is guaranteed to be O(n)
  std::vector<float>::iterator it = copy.begin() + (copy.size() / 2);
  std::nth_element(copy.begin(), it, copy.end());
  median = *it;
//  return it - copy.begin(); // Incorrect! nth_element changes ordering, so index does not mean anything!

  for(unsigned i=0; i<powers.size(); i++) {
    if(median == powers[i]) return i;
  }

  // The element was not found! This should not happen.
  LOG_DEBUG_STR("calculateMedian: could not find index, returning 0");
  return 0;
}


// calculate median, return position of the element
// returns -1 if all all elements were flagged
// TODO write version without linear scan for index
unsigned Flagger::calculateMedian(const std::vector<float>& powers, const std::vector<bool>& flags, float& median) { 
  std::vector<float> data;
  data.resize(powers.size());
  unsigned unflaggedCount = 0;
  for(unsigned i=0; i<powers.size(); i++) {
    if(!flags[i]) {
      data[unflaggedCount] = powers[i];
      unflaggedCount++;
    }
  }

//  cout << "unflaggedCount = " << unflaggedCount << endl;

  if(unflaggedCount == 0) {
    median = 0.0f;
    return -1;
  }

  // fast O(n) median
  std::nth_element(&data[0], &data[unflaggedCount/2], &data[unflaggedCount]);
  median = data[unflaggedCount/2];

  for(unsigned i=0; i<powers.size(); i++) {
    if(median == powers[i]) return i;
  }

  // The element was not found! This should not happen.
  LOG_DEBUG_STR("calculateMedian: could not find index, returning 0");
  return 0;
}


void Flagger::calculateMeanAndStdDev(const std::vector<float>& powers, const std::vector<bool>& flags, float& mean, float& stdDev) {
  stdDev = 0.0f;
  mean = 0.0f;
  unsigned count = 0;

  // Calculate mean value.
  for (unsigned i = 0; i < powers.size(); i++) {
    if(!flags[i]) {
      mean += powers[i];
      count++;
    }
  }
  mean /= count;

  // Calculate standard deviation.
  for (unsigned i = 0; i < powers.size(); i++) {
    if(!flags[i]) {
      float diff = powers[i] - mean;
      stdDev += diff * diff;
    }
  }

  stdDev /= count;
  stdDev = sqrtf(stdDev);
}


void Flagger::calculateNormalStatistics(const std::vector<float>& powers, const std::vector<bool>& flags, 
					float& mean, float& median, float& stdDev) {
  // TODO do not count flagged samples
  calculateMeanAndStdDev(powers, flags, mean, stdDev);
  calculateMedian(powers, median);
}


void Flagger::calculateWinsorizedStatistics(const std::vector<float>& powers, const std::vector<bool>& flags, 
					    float& mean, float& median, float& stdDev) {
  std::vector<float> data;
  data.resize(powers.size());
  unsigned unflaggedCount = 0;
  for(unsigned i=0; i<powers.size(); i++) {
    if(!flags[i]) {
      data[unflaggedCount] = powers[i];
      unflaggedCount++;
    }
  }

//  cout << "unflaggedCount = " << unflaggedCount << endl;

  if(unflaggedCount == 0) {
    mean = 0.0f;
    median = 0.0f;
    stdDev = 0.0f;
    return;
  }

  // fast O(n) median
  std::nth_element(&data[0], &data[unflaggedCount/2], &data[unflaggedCount]);
  median = data[unflaggedCount/2];

  unsigned lowIndex = (unsigned) floor(0.1 * unflaggedCount);
  unsigned highIndex = (unsigned) ceil(0.9 * unflaggedCount);
  if(highIndex > 0) highIndex--;
  std::nth_element(&data[0], &data[lowIndex], &data[unflaggedCount]);
  float lowValue = data[lowIndex];
  std::nth_element(&data[0], &data[highIndex], &data[unflaggedCount]);
  float highValue = data[highIndex];

  // Calculate mean
  mean = 0.0f;
  for(unsigned i = 0;i<unflaggedCount;++i) {
    float value = data[i];
    if(value < lowValue) {
      mean += lowValue;
    } else if(value > highValue) {
      mean += highValue;
    } else {
      mean += value;
    }
  }
  mean /= unflaggedCount;
  
  // Calculate variance
  stdDev = 0.0f;
  for(unsigned i = 0;i<unflaggedCount;++i) {
    float value = data[i];
    if(value < lowValue) {
      stdDev += (lowValue-mean)*(lowValue-mean);
    } else if(value > highValue) {
      stdDev += (highValue-mean)*(highValue-mean);
    } else {
      stdDev += (value-mean)*(value-mean);
    }
  }
  stdDev = sqrtf(1.54f * stdDev / unflaggedCount);

//  cout << "mean = " << mean << ", median = " << median << ", stdDev = " << stdDev << endl;
}

  
void Flagger::calculateStatistics(const std::vector<float>& powers, const std::vector<bool>& flags, float& mean, float& median, float& stdDev) {
  RFIStatsTimer.start();

  switch (itsFlaggerStatisticsType) {
  case FLAGGER_STATISTICS_NORMAL:
    calculateNormalStatistics(powers, flags, mean, median, stdDev);
    break;
  case FLAGGER_STATISTICS_WINSORIZED:
    calculateWinsorizedStatistics(powers, flags, mean, median, stdDev);
    break;
  default:
    LOG_INFO_STR("ERROR, illegal FlaggerStatisticsType.");
    return;
  }

  RFIStatsTimer.stop();
}


void Flagger::calculateStatistics(const MultiDimArray<float,2> &powers, MultiDimArray<bool,2> &flags, float& mean, float& median, float& stdDev) {
  RFIStatsTimer.start();

  unsigned size = powers.shape()[0] * powers.shape()[1];

  // convert to 1D
  std::vector<float> powers1D(size);
  memcpy(powers1D.data(), powers.data(), size * sizeof(float));

  // Std uses specialized versions for bools (bit vectors). So, we have to copy manually.
  std::vector<bool> flags1D( flags.shape()[0] *  flags.shape()[1]);
  int idx=0;
  for (unsigned channel = 0; channel < flags.shape()[0]; channel++) {
    for (unsigned time = 0; time < flags.shape()[1]; time++) {
      flags1D[idx++] = flags[channel][time];
    }
  }

  switch (itsFlaggerStatisticsType) {
  case FLAGGER_STATISTICS_NORMAL:
    calculateNormalStatistics(powers1D, flags1D, mean, median, stdDev);
    break;
  case FLAGGER_STATISTICS_WINSORIZED:
    calculateWinsorizedStatistics(powers1D, flags1D, mean, median, stdDev);
    break;
  default:
    LOG_INFO_STR("ERROR, illegal FlaggerStatisticsType.");
    return;
  }

  RFIStatsTimer.stop();
}


float Flagger::evaluateGaussian(const float x, const float sigma) {
    return (float) (1.0 / (sigma * sqrt(2.0 * M_PI)) * exp(-0.5 * x * x / sigma));
}


float Flagger::logBase2(const float x) {
  // log(base 2) x=log(base e) x/log(base e) 2
  return (float) (log(x) / log(2.0));
}


void Flagger::oneDimensionalConvolution(const float* data, const unsigned dataSize, float*dest, const float* kernel, const unsigned kernelSize) {
  for (unsigned i = 0; i < dataSize; ++i) {
    int offset = i - kernelSize / 2;
    unsigned start, end;
    
    if (offset < 0) {
      start = -offset;
    } else {
      start = 0;
    }
    if (offset + kernelSize > dataSize) {
      end = dataSize - offset;
    } else {
      end = kernelSize;
    }

    float sum = 0.0f;
    float weight = 0.0f;
    for (unsigned k = start; k < end; k++) {
      sum += data[k + offset] * kernel[k];
      weight += kernel[k];
    }

    if (weight != 0.0f) {
      dest[i] = sum / weight;
    }
  }
}


void Flagger::oneDimensionalGausConvolution(const float* data, const unsigned dataSize, float* dest, const float sigma) {
  unsigned kernelSize = (unsigned) round(sigma * 3.0);
  if(kernelSize < 1) {
    kernelSize = 1;
  } else if (kernelSize > dataSize) {
    kernelSize = dataSize;
  }

  float kernel[kernelSize];
  for (unsigned i = 0; i < kernelSize; ++i) {
    float x = i - kernelSize / 2.0f;
    kernel[i] = evaluateGaussian(x, sigma);
  }
  oneDimensionalConvolution(data, dataSize, dest, kernel, kernelSize);
}


float Flagger::calcThresholdI(float threshold1, unsigned window, float p) {
  if (p <= 0.0f) {
    p = 1.5f; // according to Andre's RFI paper, this is a good default value
  }
	
  return (float) (threshold1 * pow(p, logBase2(window)) / window);
}


void Flagger::sumThreshold1D(std::vector<float>& powers, std::vector<bool>& flags, const unsigned window, const float threshold) {
  for (unsigned base = 1; base + window < powers.size(); base++) {
    float sum = 0.0f;

    for (unsigned pos = base; pos < base + window; pos++) {
      if (flags[pos]) { // If it was flagged in a previous iteration, replace sample with current threshold
        sum += threshold;
        powers[pos] = threshold; // for stats calc
      } else {
        sum += powers[pos];
      }
    }

    if (sum >= window * threshold) {
      // flag all samples in the sequence!
      for (unsigned pos = base; pos < base + window; pos++) {
        flags[pos] = true;
        powers[pos] = threshold; // for stats calc
      }
    }
  }
}


// in time direction
void Flagger::sumThreshold2DHorizontal(MultiDimArray<float,2> &powers, MultiDimArray<bool,2> &flags, const unsigned window, const float threshold) {
  for(unsigned channel=1; channel<powers.shape()[0]; channel++) {
    for (unsigned base = 0; base + window < powers.shape()[1]; base++) {
      float sum = 0.0f;
      
      for (unsigned time = base; time < base + window; time++) {
	if (flags[channel][time]) { // If it was flagged in a previous iteration, replace sample with current threshold
	  sum += threshold;
	  powers[channel][time] = threshold; // for stats calc
	} else {
	  sum += powers[channel][time];
	}
      }
      
      if (sum >= window * threshold) {
	// flag all samples in the sequence!
	for (unsigned time = base; time < base + window; time++) {
	  flags[channel][time] = true;
	  powers[channel][time] = threshold; // for stats calc
	}
      }
    }
  }
}


// in frequency direction
void Flagger::sumThreshold2DVertical(MultiDimArray<float,2> &powers, MultiDimArray<bool,2> &flags, const unsigned window, const float threshold) {
  for (unsigned time = 0; time < powers.shape()[1]; time++) {
    for(unsigned base=1; base + window <powers.shape()[0]; base++) {
      float sum = 0.0f;
      
      for (unsigned channel = base; channel < base + window; channel++) {
	if (flags[channel][time]) { // If it was flagged in a previous iteration, replace sample with current threshold
	  sum += threshold;
	  powers[channel][time] = threshold; // for stats calc
	} else {
	  sum += powers[channel][time];
	}
      }
      
      if (sum >= window * threshold) {
	// flag all samples in the sequence!
	for (unsigned channel = base; channel < base + window; channel++) {
	  flags[channel][time] = true;
	  powers[channel][time] = threshold; // for stats calc
	}
      }
    }
  }
}


void Flagger::sumThresholdFlagger2D(MultiDimArray<float,2> &powers, MultiDimArray<bool,2> &flags, const float sensitivity) {
  float mean, stdDev, median;
  calculateStatistics(powers, flags, mean, median, stdDev);

  float factor;
  if (stdDev == 0.0f) {
    factor = sensitivity;
  } else {
    factor = stdDev * sensitivity;
  }

  unsigned window = 1;
  for (unsigned iter = 1; iter <= MAX_SUM_THRESHOLD_ITERS; iter++) {
    float thresholdI = median + calcThresholdI(itsCutoffThreshold, iter, 1.5f) * factor;
//    LOG_DEBUG_STR("THRESHOLD in iter " << iter <<", window " << window << " = " << calcThresholdI(itsCutoffThreshold, iter, 1.5f) << ", becomes = " << thresholdI);

    sumThreshold2DHorizontal(powers, flags, window, thresholdI);
    sumThreshold2DVertical(powers, flags, window, thresholdI);

    window *= 2;
  }
}


void Flagger::thresholdingFlagger1D(std::vector<float>& powers, std::vector<bool>& flags) {
  float mean, stdDev, median;

  calculateStatistics(powers, flags, mean, median, stdDev);

  float threshold = median + itsCutoffThreshold * stdDev;

  for (unsigned channel = 0; channel < powers.size(); channel++) {
    if (powers[channel] > threshold) {
      flags[channel] = true;
    }
  }
}


void Flagger::thresholdingFlagger2D(const MultiDimArray<float,2> &powers, MultiDimArray<bool,2> &flags) {
  float mean, stdDev, median;
  calculateStatistics(powers, flags, mean, median, stdDev);

  float threshold = median + itsCutoffThreshold * stdDev;

  for (unsigned channel = 1; channel < powers.shape()[0]; channel++) {
    for (unsigned time = 0; time < powers.shape()[1]; time++) {
      const float power = powers[channel][time];
      if (power > threshold) {
	// flag this sample, both polarizations.
	flags[channel][time] = true;
      }
    }
  }
}


void Flagger::sumThresholdFlagger1D(std::vector<float>& powers, std::vector<bool>& flags, const float sensitivity) {
  float mean, stdDev, median;
  calculateStatistics(powers, flags, mean, median, stdDev);

  float factor;
  if (stdDev == 0.0f) {
    factor = sensitivity;
  } else {
    factor = stdDev * sensitivity;
  }

  unsigned window = 1;
  for (unsigned iter = 1; iter <= MAX_SUM_THRESHOLD_ITERS; iter++) {
    float thresholdI = median + calcThresholdI(itsCutoffThreshold, iter, 1.5f) * factor;
//    LOG_DEBUG_STR("THRESHOLD in iter " << iter <<", window " << window << " = " << calcThresholdI(itsCutoffThreshold, iter, 1.5f) << ", becomes = " << thresholdI);
    sumThreshold1D(powers, flags, window, thresholdI);
    window *= 2;
  }
}


void Flagger::sumThresholdFlagger1DSmoothed(std::vector<float>& powers, std::vector<float>& smoothedPowers, std::vector<float>& powerDiffs, std::vector<bool>& flags) {
  // first do an insensitive sumthreshold
  sumThresholdFlagger1D(powers, flags, 1.0f * itsBaseSensitivity); // sets flags, and replaces flagged samples with threshold
	
  // smooth
  oneDimensionalGausConvolution(powers.data(), powers.size(), smoothedPowers.data(), 0.5f); // last param is sigma, height of the gaussian curve

  // calculate difference
  for (unsigned int i = 0; i < powers.size(); i++) {
    powerDiffs[i] = powers[i] - smoothedPowers[i];
  }
  
  // flag based on difference
  sumThresholdFlagger1D(powerDiffs, flags, 1.0f * itsBaseSensitivity); // sets additional flags
  
  // and one final, more sensitive pass on the flagged power
  sumThresholdFlagger1D(powers, flags, 0.8f * itsBaseSensitivity); // sets flags, and replaces flagged samples with threshold
}


bool Flagger::addToHistory(const float value, FlaggerHistory& history, float historyFlaggingThreshold) {
  if (history.getSize() < MIN_HISTORY_SIZE) {
    history.add(value); // add it, and return, we don't have enough history yet.
    return false;
  }

  float mean = history.getMean();
  float stdDev = history.getStdDev();
  float threshold = mean + historyFlaggingThreshold * stdDev;

  bool flagSecond = value > threshold;
  if (flagSecond) {
      LOG_DEBUG_STR("History flagger flagged this second: value = " << value << ", mean = " << mean << ", stdDev = " << stdDev << ", factor from cuttoff is: " << (value / threshold));
    // this second was flagged, add the threshold value to the history.
    history.add(threshold);
  } else {
    // add data
    history.add(value);
  }

  return flagSecond;
}


void Flagger::sumThresholdFlagger1DWithHistory(std::vector<float>& powers, 
					       std::vector<bool>& flags, const float sensitivity, FlaggerHistory& history) {
  sumThresholdFlagger1D(powers, flags, sensitivity);

  // flag twice, so the second time flags with the corrected statistics
  sumThresholdFlagger1D(powers, flags, sensitivity);
  
  float localMean, localStdDev, localMedian;

  // calculate final statistics (flagged samples were replaced with threshold values)
  calculateStatistics(powers, flags, localMean, localMedian, localStdDev);

  if(addToHistory(localMedian, history)) {
    for (unsigned i = 0; i < powers.size(); i++) {
      flags[i] = true;
    }
  }
}


void Flagger::sumThresholdFlagger2DWithHistory(MultiDimArray<float,2> &powers, MultiDimArray<bool,2> &flags, std::vector<float> &integratedPowers,
					       std::vector<bool> & /*integratedFlags*/, const float sensitivity, 
					       MultiDimArray<FlaggerHistory, 3>& history, unsigned station, unsigned subband, unsigned pol) {
  float localMean, localStdDev, localMedian;

  sumThresholdFlagger2D(powers, flags, sensitivity);

  // initialize integratedFlags
  std::vector<bool> integratedFlags(flags.shape()[0]);
  integratedFlags.clear();
  for (unsigned channel = 0; channel < flags.shape()[0]; channel++) {
    for (unsigned time = 0; time < flags.shape()[1]; time++) {
      if(flags[channel][time]) {
	integratedFlags[channel] = true;
      }
    }
  }

  // Also flag in frequency direction on fully integrated data for maximal signal to noise
  sumThresholdFlagger1D(integratedPowers, integratedFlags, sensitivity);

  // apply flags of step above
  apply1DflagsTo2D(flags, integratedFlags);

  // calculate final statistics for integrated data
  calculateStatistics(integratedPowers, integratedFlags, localMean, localMedian, localStdDev);

  float total = 0.0f;
  for(unsigned i=0; i<integratedPowers.size(); i++) {
    if(!integratedFlags[i]) {
      total += integratedPowers[i];
    }
  }


  string s(" ");
  for (unsigned channel = 0; channel < integratedPowers.size(); channel++) {
    s += boost::lexical_cast<string>(integratedPowers[channel]/total);
    s += " ";
  }

  LOG_DEBUG_STR("HISTORY_FREQ: station " << station << " subband = " << subband << " pol " << pol << " mean = " << localMean << " median " << localMedian << ", stdDev " << localStdDev << "total " << total << " meanTot " << (localMean / total) << " medTot " << (localMedian / total) << " stdDevTot " << (localStdDev / total) << s);


#if 0
  LOG_DEBUG_STR("HISTORY: station = " << station << ", subband = " << subband << " pol = " << pol << ", mean = " << localMean << ", median = " << localMedian << ", stdDev = " << localStdDev << ", total = " << total << ", meanTot = " << (localMean / total) << ", medTot = " << (localMedian / total) << ", stdDevTot = " << (localStdDev / total));
#endif

// if we divide median by the total power, we cancel out bandpass, etc.
//  if(addToHistory(localMean / total, localStdDev / total, localMedian / total, history[station][subband][pol])) {

  if(addToHistory(localMedian, history[station][subband][pol])) {
    for (unsigned channel = 0; channel < flags.shape()[0]; channel++) {
      for (unsigned time = 0; time < flags.shape()[1]; time++) {
	flags[channel][time] = true;
      }
    }
  }
}


void Flagger::apply1DflagsTo2D(MultiDimArray<bool,2> &flags, std::vector<bool> & integratedFlags)
{
  for (unsigned channel = 0; channel < flags.shape()[0]; channel++) {
    if(integratedFlags[channel]) {
      for (unsigned time = 0; time < flags.shape()[1]; time++) {
	flags[channel][time] = true;
      }
    }
  }
}


void Flagger::sumThresholdFlagger1DSmoothedWithHistory(std::vector<float>& powers, std::vector<float>& smoothedPowers, std::vector<float>& powerDiffs, 
						       std::vector<bool>& flags, FlaggerHistory& history) {
  sumThresholdFlagger1DSmoothed(powers, smoothedPowers, powerDiffs, flags);
  
  float localMean, localStdDev, localMedian;

  // calculate final statistics (flagged samples were replaced with threshold values)
  calculateStatistics(powers, flags, localMean, localMedian, localStdDev);

  if(addToHistory(localMedian, history)) {
    for (unsigned i = 0; i < powers.size(); i++) {
      flags[i] = true;
    }
  }
}


/**
 * This is an experimental algorithm that might be slightly faster than
 * the original algorithm by Andre Offringa. Jasper van de Gronde is preparing an article about it.
 * @param [in,out] flags The input array of flags to be dilated that will be overwritten by the dilatation of itself.
 * @param [in] eta The η parameter that specifies the minimum number of good data
 * that any subsequence should have (see class description for the definition).
 */
unsigned Flagger::SIROperator(std::vector<bool> flags, float eta) {
  int old= 0;
  for(unsigned i=0; i<flags.size(); i++) {
    if(flags[i]) old++;
  }

  bool temp[flags.size()];
  float credit = 0.0f;
  for (unsigned i = 0; i < flags.size(); i++) {
    // credit ← max(0, credit) + w(f [i])
    const float w = flags[i] ? eta : eta - 1.0f;
    const float maxcredit0 = credit > 0.0f ? credit : 0.0f;
    credit = maxcredit0 + w;
    temp[i] = (credit >= 0.0f);
  }

  // The same iteration, but now backwards
  credit = 0.0f;
  for (int i = flags.size() - 1; i >= 0; i--) {
    const float w = flags[i] ? eta : eta - 1.0f;
    const float maxcredit0 = credit > 0.0f ? credit : 0.0f;
    credit = maxcredit0 + w;
    flags[i] = (credit >= 0.0f) || temp[i];
  }


  int c= 0;
  for(unsigned i=0; i<flags.size(); i++) {
    if(flags[i]) c++;
  }

  LOG_DEBUG_STR("SIR operator flagged " << (c - old) << " samples");

  return c;
}


FlaggerStatisticsType Flagger::getFlaggerStatisticsType(std::string t) {
  if (t.compare("NORMAL") == 0) {
    return FLAGGER_STATISTICS_NORMAL;
  } else if (t.compare("WINSORIZED") == 0) {
    return FLAGGER_STATISTICS_WINSORIZED;
  } else {
    LOG_DEBUG_STR("unknown flagger statistics type, using default FLAGGER_STATISTICS_WINSORIZED");
    return FLAGGER_STATISTICS_WINSORIZED;
  }
}


std::string Flagger::getFlaggerStatisticsTypeString(FlaggerStatisticsType t) {
  switch(t) {
  case FLAGGER_STATISTICS_NORMAL:
    return "FLAGGER_STATISTICS_NORMAL";
  case FLAGGER_STATISTICS_WINSORIZED:
    return "FLAGGER_STATISTICS_WINSORIZED";
  default:
    return "ILLEGAL FLAGGER STATISTICS TYPE";
  }
}


std::string Flagger::getFlaggerStatisticsTypeString() {
  return getFlaggerStatisticsTypeString(itsFlaggerStatisticsType);
}

} // namespace RTCP
} // namespace LOFAR
