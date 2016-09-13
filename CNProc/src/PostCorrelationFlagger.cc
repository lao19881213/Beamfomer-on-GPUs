//# Always #include <lofar_config.h> first!
#include <lofar_config.h>

#include <Common/Timer.h>

#include <Correlator.h>
#include <PostCorrelationFlagger.h>

namespace LOFAR {
namespace RTCP {

static NSTimer detectBrokenStationsTimer("RFI post DetectBrokenStations", true, true);

// CorrelatedData samples: [nrBaselines][nrChannels][NR_POLARIZATIONS][NR_POLARIZATIONS]

// We have the data for one second, all frequencies in a subband.
// If one of the polarizations exceeds the threshold, flag them all.
// All baselines are flagged completely independently.
// Autocorrelations are ignored.

// TODO: if detectBrokenStations is not enabled, we don't have to wipe/calc summedbaselinePowers
// TODO: if some data was already flagged, take that into account (already done for pre-correlation flagger).

PostCorrelationFlagger::PostCorrelationFlagger(const Parset& parset, const unsigned nrStations, const unsigned nrSubbands, 
					       const unsigned nrChannels, const float cutoffThreshold, float baseSentitivity) :
  Flagger(parset, nrStations, nrSubbands, nrChannels, cutoffThreshold, baseSentitivity,
	    getFlaggerStatisticsType(parset.onlinePostCorrelationFlaggingStatisticsType(getFlaggerStatisticsTypeString(FLAGGER_STATISTICS_WINSORIZED)))), 
    itsFlaggerType(getFlaggerType(parset.onlinePostCorrelationFlaggingType(getFlaggerTypeString(POST_FLAGGER_SMOOTHED_SUM_THRESHOLD_WITH_HISTORY)))),
    itsNrBaselines((nrStations * (nrStations + 1) / 2)) {

  itsPowers.resize(itsNrChannels);
  itsSmoothedPowers.resize(itsNrChannels);
  itsPowerDiffs.resize(nrChannels);
  itsFlags.resize(itsNrChannels);
  itsSummedBaselinePowers.resize(itsNrBaselines);
  itsSummedStationPowers.resize(itsNrStations);
  itsHistory.resize(boost::extents[itsNrBaselines][nrSubbands][NR_POLARIZATIONS][NR_POLARIZATIONS]);

  LOG_DEBUG_STR("post correlation flagging type = " << getFlaggerTypeString()
		<< ", statistics type = " << getFlaggerStatisticsTypeString());
}

void PostCorrelationFlagger::flag(CorrelatedData* correlatedData, unsigned currentSubband) {
  NSTimer flaggerTimer("RFI post flagger", true, true);
  flaggerTimer.start();

  wipeSums();

  for (unsigned baseline = 0; baseline < itsNrBaselines; baseline++) {
    if (Correlator::baselineIsAutoCorrelation(baseline)) {
//      LOG_DEBUG_STR(" baseline " << baseline << " is an autocorrelation, skipping");
      continue;
    }

    wipeFlags();
    for (unsigned pol1 = 0; pol1 < NR_POLARIZATIONS; pol1++) {
      for (unsigned pol2 = 0; pol2 < NR_POLARIZATIONS; pol2++) {
        calculatePowers(baseline, pol1, pol2, correlatedData);

        switch (itsFlaggerType) {
        case POST_FLAGGER_THRESHOLD:
	  thresholdingFlagger1D(itsPowers, itsFlags);
          break;
        case POST_FLAGGER_SUM_THRESHOLD:
          sumThresholdFlagger1D(itsPowers, itsFlags, itsBaseSensitivity);
          break;
	case POST_FLAGGER_SMOOTHED_SUM_THRESHOLD:
          sumThresholdFlagger1DSmoothed(itsPowers, itsSmoothedPowers, itsPowerDiffs, itsFlags);
	  break;
	case POST_FLAGGER_SMOOTHED_SUM_THRESHOLD_WITH_HISTORY:
          sumThresholdFlagger1DSmoothedWithHistory(itsPowers, itsSmoothedPowers, itsPowerDiffs, itsFlags, itsHistory[baseline][currentSubband][pol1][pol2]);
	  break;
        default:
          LOG_INFO_STR("ERROR, illegal FlaggerType. Skipping online post correlation flagger.");
          return;
        }

        calculateSummedbaselinePowers(baseline);
      }
    }

    applyFlags(baseline, correlatedData);
  }
  flaggerTimer.stop();
}

void PostCorrelationFlagger::calculateSummedbaselinePowers(unsigned baseline) {
  for (unsigned channel = 0; channel < itsNrChannels; channel++) {
    if (!itsFlags[channel]) {
      itsSummedBaselinePowers[baseline] += itsPowers[channel];
    }
  }
}

  // TODO: also integrate flags?
void PostCorrelationFlagger::detectBrokenStations() {
  detectBrokenStationsTimer.start();

  // Sum all baselines that involve a station (both horizontally and vertically).

  for (unsigned station = 0; station < itsNrStations; station++) {
    float sum = 0.0f;
    for (unsigned stationH = station+1; stationH < itsNrStations; stationH++) { // do not count autocorrelation
      unsigned baseline = Correlator::baseline(station, stationH);
      sum += itsSummedBaselinePowers[baseline];
    }
    for (unsigned stationV = 0; stationV < station; stationV++) {
      unsigned baseline = Correlator::baseline(stationV, station);
      sum += itsSummedBaselinePowers[baseline];
    }

    itsSummedStationPowers[station] = sum;
  }

  float stdDev;
  float mean;
  //  calculateStdDevAndSum(itsSummedStationPowers.data(), itsSummedStationPowers.size(), mean, stdDev, sum);

  calculateMeanAndStdDev(itsSummedStationPowers, mean, stdDev);

  float median;
  calculateMedian(itsSummedStationPowers, median);
  float threshold = mean + itsCutoffThreshold * stdDev;

  LOG_DEBUG_STR("RFI post detectBrokenStations: mean = " << mean << ", median = " << median << " stdDev = " << stdDev << ", threshold = " << threshold);

  for (unsigned station = 0; station < itsNrStations; station++) {
    LOG_INFO_STR("RFI post detectBrokenStations: station " << station << " total summed power = " << itsSummedStationPowers[station]);
    if (itsSummedStationPowers[station] > threshold) {
      LOG_INFO_STR(
          "RFI post detectBrokenStations: WARNING, station " << station << " seems to be corrupted, total summed power = " << itsSummedStationPowers[station]);
    }
  }

  detectBrokenStationsTimer.stop();
}

void PostCorrelationFlagger::wipeSums() {
  for (unsigned baseline = 0; baseline < itsNrBaselines; baseline++) {
    itsSummedBaselinePowers[baseline] = 0.0f;
  }

  for (unsigned station = 0; station < itsNrStations; station++) {
    itsSummedStationPowers[station] = 0.0f;
  }
}

void PostCorrelationFlagger::wipeFlags() {
  for (unsigned channel = 0; channel < itsNrChannels; channel++) {
    itsFlags[channel] = false;
  }
}

void PostCorrelationFlagger::applyFlags(unsigned baseline, CorrelatedData* correlatedData) {
  for (unsigned channel = 0; channel < itsNrChannels; channel++) {
    if (itsFlags[channel]) {
      correlatedData->setNrValidSamples(baseline, channel, 0);
      // TODO: currently, we can only flag all channels at once! This is a limitation in CorrelatedData.
      //	    correlatedData->flags[station].include(time);
    }
  }
}

void PostCorrelationFlagger::calculatePowers(unsigned baseline, unsigned pol1, unsigned pol2, CorrelatedData* correlatedData) {
  for (unsigned channel = 0; channel < itsNrChannels; channel++) {
    fcomplex sample = correlatedData->visibilities[baseline][channel][pol1][pol2];
    float power = real(sample) * real(sample) + imag(sample) * imag(sample);
    itsPowers[channel] = power;
  }
}

PostCorrelationFlaggerType PostCorrelationFlagger::getFlaggerType(std::string t) {
  if (t.compare("THRESHOLD") == 0) {
    return POST_FLAGGER_THRESHOLD;
  } else if (t.compare("SUM_THRESHOLD") == 0) {
    return POST_FLAGGER_SUM_THRESHOLD;
  } else if (t.compare("SMOOTHED_SUM_THRESHOLD") == 0) {
    return POST_FLAGGER_SMOOTHED_SUM_THRESHOLD;
  } else if (t.compare("SMOOTHED_SUM_THRESHOLD_WITH_HISTORY") == 0) {
    return POST_FLAGGER_SMOOTHED_SUM_THRESHOLD_WITH_HISTORY;
  } else {
    LOG_DEBUG_STR("unknown flagger type, using default SMOOTHED_SUM_THRESHOLD_WITH_HISTORY");
    return POST_FLAGGER_SMOOTHED_SUM_THRESHOLD_WITH_HISTORY;
  }
}

std::string PostCorrelationFlagger::getFlaggerTypeString(PostCorrelationFlaggerType t) {
  switch(t) {
  case POST_FLAGGER_THRESHOLD:
    return "THRESHOLD";
  case POST_FLAGGER_SUM_THRESHOLD:
    return "SUM_THRESHOLD";
  case POST_FLAGGER_SMOOTHED_SUM_THRESHOLD:
    return "SMOOTHED_SUM_THRESHOLD";
  case POST_FLAGGER_SMOOTHED_SUM_THRESHOLD_WITH_HISTORY:
    return "SMOOTHED_SUM_THRESHOLD_WITH_HISTORY";
  default:
    return "ILLEGAL FLAGGER TYPE";
  }
}

std::string PostCorrelationFlagger::getFlaggerTypeString() {
  return getFlaggerTypeString(itsFlaggerType);
}

} // namespace RTCP
} // namespace LOFAR
