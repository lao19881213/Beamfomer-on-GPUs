//# Always #include <lofar_config.h> first!
#include <lofar_config.h>

#include <Common/Timer.h>

#include <PreCorrelationFlagger.h>

// history is kept per subband, as we can get different subbands over time on this compute node.
// Always flag poth polarizations as a unit.

namespace LOFAR {
namespace RTCP {

PreCorrelationFlagger::PreCorrelationFlagger(const Parset& parset, const unsigned nrStations, const unsigned nrSubbands, const unsigned nrChannels, 
					     const unsigned nrSamplesPerIntegration, const float cutoffThreshold)
:
  Flagger(parset, nrStations, nrSubbands, nrChannels, cutoffThreshold, /*baseSentitivity*/ 1.0f, 
	  getFlaggerStatisticsType(parset.onlinePreCorrelationFlaggingStatisticsType(getFlaggerStatisticsTypeString(FLAGGER_STATISTICS_WINSORIZED)))),
  itsFlaggerType(getFlaggerType(parset.onlinePreCorrelationFlaggingType(getFlaggerTypeString(PRE_FLAGGER_THRESHOLD)))), 
  itsNrSamplesPerIntegration(nrSamplesPerIntegration),
  itsIntegrationFactor(parset.onlinePreCorrelationFlaggingIntegration())
{
  if(itsIntegrationFactor == 0) {
    itsIntegrationFactor = itsNrSamplesPerIntegration;
  }

  if(itsNrSamplesPerIntegration % itsIntegrationFactor != 0) {
    LOG_ERROR_STR("preCorrelationFlagger: Illegal integration factor, fully integrating");
    itsIntegrationFactor = itsNrSamplesPerIntegration;
  }

  itsPowers.resize(boost::extents[itsNrChannels][itsNrSamplesPerIntegration]);
  itsFlags.resize(boost::extents[itsNrChannels][itsNrSamplesPerIntegration]);

  switch(itsFlaggerType) {
    // not integrated
  case PRE_FLAGGER_THRESHOLD:
    break;

    // fully integrated
  case PRE_FLAGGER_INTEGRATED_THRESHOLD:
  case PRE_FLAGGER_INTEGRATED_SUM_THRESHOLD:
    itsIntegratedPowers.resize(itsNrChannels);
    itsIntegratedFlags.resize(itsNrChannels);
    break;

    // fully integrated, and smoothed
  case PRE_FLAGGER_INTEGRATED_SMOOTHED_SUM_THRESHOLD:
    itsIntegratedPowers.resize(itsNrChannels);
    itsIntegratedFlags.resize(itsNrChannels);
    itsSmoothedIntegratedPowers.resize(itsNrChannels);
    itsIntegratedPowerDiffs.resize(itsNrChannels);
    break;

    // fully integrated, and history
  case PRE_FLAGGER_INTEGRATED_SUM_THRESHOLD_WITH_HISTORY:
    itsHistory.resize(boost::extents[itsNrStations][itsNrSubbands][NR_POLARIZATIONS]);
    itsIntegratedPowers.resize(itsNrChannels);
    itsIntegratedFlags.resize(itsNrChannels);
    break;

    // fully integrated, smoothed, and history
  case PRE_FLAGGER_INTEGRATED_SMOOTHED_SUM_THRESHOLD_WITH_HISTORY:
    itsHistory.resize(boost::extents[itsNrStations][itsNrSubbands][NR_POLARIZATIONS]);
    itsIntegratedPowers.resize(itsNrChannels);
    itsIntegratedFlags.resize(itsNrChannels);
    itsSmoothedIntegratedPowers.resize(itsNrChannels);
    itsIntegratedPowerDiffs.resize(itsNrChannels);
    break;

    // Partially integrated
  case PRE_FLAGGER_INTEGRATED_THRESHOLD_2D:
  case PRE_FLAGGER_INTEGRATED_SUM_THRESHOLD_2D:
    itsIntegratedPowers2D.resize(boost::extents[itsNrChannels][itsNrSamplesPerIntegration/itsIntegrationFactor]);
    itsIntegratedFlags2D.resize(boost::extents[itsNrChannels][itsNrSamplesPerIntegration/itsIntegrationFactor]);
    break;

    // Partially integrated, with history
  case PRE_FLAGGER_INTEGRATED_SUM_THRESHOLD_2D_WITH_HISTORY:
    itsIntegratedPowers.resize(itsNrChannels); // needed for history
    itsIntegratedFlags.resize(itsNrChannels); // needed for history
    itsIntegratedPowers2D.resize(boost::extents[itsNrChannels][itsNrSamplesPerIntegration/itsIntegrationFactor]);
    itsIntegratedFlags2D.resize(boost::extents[itsNrChannels][itsNrSamplesPerIntegration/itsIntegrationFactor]);
    itsHistory.resize(boost::extents[itsNrStations][itsNrSubbands][NR_POLARIZATIONS]);
    break;

  default:
    LOG_INFO_STR("ERROR, illegal FlaggerType. Skipping online pre correlation flagger.");
    return;
  }

  LOG_DEBUG_STR("pre correlation flagging type = " << getFlaggerTypeString()
		<< ", statistics type = " << getFlaggerStatisticsTypeString()
		<< ", extra integration factor = " << itsIntegrationFactor);
}


void PreCorrelationFlagger::flag(FilteredData* filteredData, unsigned currentSubband)
{
  NSTimer flaggerTimer("RFI pre flagger", true, true);
  flaggerTimer.start();

  for(unsigned station = 0; station < itsNrStations; station++) {
    initFlags(station, filteredData); // copy flags to my local format
    filteredData->resetFlags();       // Wipe original flags

    for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol++) {
      calculatePowers(station, pol, filteredData);

      switch(itsFlaggerType) {
      case PRE_FLAGGER_THRESHOLD:
	thresholdingFlagger2D(itsPowers, itsFlags);
	break;
      case PRE_FLAGGER_INTEGRATED_THRESHOLD:
	integratingThresholdingFlagger(itsPowers, itsFlags, itsIntegratedPowers, itsIntegratedFlags);
	break;
      case PRE_FLAGGER_INTEGRATED_SUM_THRESHOLD:
	integratingSumThresholdFlagger(itsPowers, itsFlags, itsIntegratedPowers, itsIntegratedFlags);
	break;
      case PRE_FLAGGER_INTEGRATED_SUM_THRESHOLD_WITH_HISTORY:
	integratingSumThresholdFlaggerWithHistory(itsPowers, itsFlags, itsIntegratedPowers, itsIntegratedFlags, itsHistory[station][currentSubband][pol]);
	break;
      case PRE_FLAGGER_INTEGRATED_SMOOTHED_SUM_THRESHOLD:
	integratingSumThresholdFlaggerSmoothed(itsPowers, itsFlags, itsIntegratedPowers, itsSmoothedIntegratedPowers, itsIntegratedPowerDiffs, itsIntegratedFlags);
	break;
      case PRE_FLAGGER_INTEGRATED_SMOOTHED_SUM_THRESHOLD_WITH_HISTORY:
	integratingSumThresholdFlaggerSmoothedWithHistory(itsPowers, itsFlags, itsIntegratedPowers, itsSmoothedIntegratedPowers, itsIntegratedPowerDiffs, itsIntegratedFlags, itsHistory[station][currentSubband][pol]);
	break;
      case PRE_FLAGGER_INTEGRATED_THRESHOLD_2D:
	integratingThresholdingFlagger2D(itsPowers, itsFlags, itsIntegratedPowers2D, itsIntegratedFlags2D);
	break;
      case PRE_FLAGGER_INTEGRATED_SUM_THRESHOLD_2D:
	integratingSumThresholdFlagger2D(itsPowers, itsFlags, itsIntegratedPowers2D, itsIntegratedFlags2D);
	break;
      case PRE_FLAGGER_INTEGRATED_SUM_THRESHOLD_2D_WITH_HISTORY:
	integratingSumThresholdFlagger2DWithHistory(itsPowers, itsFlags, itsIntegratedPowers2D, itsIntegratedFlags2D,  itsIntegratedPowers, itsIntegratedFlags, itsHistory, station, currentSubband, pol);
	break;
      default:
	LOG_INFO_STR("ERROR, illegal FlaggerType. Skipping online pre correlation flagger.");
	return;
      }
    }

    applyFlags(station, filteredData); // copy flags from my original format into FilteredData again.
  }

  flaggerTimer.stop();
}


void PreCorrelationFlagger::integratingThresholdingFlagger(const MultiDimArray<float,2> &powers, MultiDimArray<bool,2>& flags,
							   vector<float> &integratedPowers, vector<bool> &integratedFlags)
{
  integratePowers(powers, flags, integratedPowers, integratedFlags);
  thresholdingFlagger1D(integratedPowers, integratedFlags);
}

void PreCorrelationFlagger::integratingThresholdingFlagger2D(const MultiDimArray<float,2> &powers, MultiDimArray<bool,2>& flags,
							     MultiDimArray<float,2>& integratedPowers2D,  
							     MultiDimArray<bool,2>& integratedFlags2D)
{
  integratePowers2D(powers, flags, integratedPowers2D, integratedFlags2D);
  thresholdingFlagger2D(integratedPowers2D, integratedFlags2D);
}


void PreCorrelationFlagger::integratingSumThresholdFlagger(const MultiDimArray<float,2> &powers, MultiDimArray<bool,2>& flags,
							   vector<float> &integratedPowers, vector<bool> &integratedFlags) {
  integratePowers(powers, flags, integratedPowers, integratedFlags);
  sumThresholdFlagger1D(integratedPowers, integratedFlags, itsBaseSensitivity);
}


void PreCorrelationFlagger::integratingSumThresholdFlagger2D(const MultiDimArray<float,2> &powers, MultiDimArray<bool,2>& flags,
							     MultiDimArray<float,2>& integratedPowers2D,  
							     MultiDimArray<bool,2>& integratedFlags2D) {
  integratePowers2D(powers, flags, integratedPowers2D, integratedFlags2D);
  sumThresholdFlagger2D(integratedPowers2D, integratedFlags2D, itsBaseSensitivity);
}


void PreCorrelationFlagger::integratingSumThresholdFlagger2DWithHistory(const MultiDimArray<float,2> &powers, MultiDimArray<bool,2>& flags,
									MultiDimArray<float,2>& integratedPowers2D, MultiDimArray<bool,2>& integratedFlags2D,
									vector<float> &integratedPowers, vector<bool> &integratedFlags, 
									MultiDimArray<FlaggerHistory, 3>& history, unsigned station, unsigned subband, unsigned pol) {
  integratePowers2D(powers, flags, integratedPowers2D, integratedFlags2D);
  integratePowers(powers, flags, integratedPowers, integratedFlags); // for the history
  sumThresholdFlagger2DWithHistory(integratedPowers2D, integratedFlags2D, integratedPowers, integratedFlags, itsBaseSensitivity, history, station, subband, pol);
}


void PreCorrelationFlagger::integratingSumThresholdFlaggerSmoothed(const MultiDimArray<float,2> &powers, MultiDimArray<bool,2>& flags,
								   vector<float> &integratedPowers, vector<float> &smoothedPowers, 
								   vector<float> &powerDiffs, vector<bool> &integratedFlags) {
  integratePowers(powers, flags, integratedPowers, integratedFlags);
  sumThresholdFlagger1DSmoothed(integratedPowers, smoothedPowers, powerDiffs, integratedFlags);
}


void PreCorrelationFlagger::integratingSumThresholdFlaggerWithHistory(const MultiDimArray<float,2> &powers, MultiDimArray<bool,2>& flags,
									      vector<float> &integratedPowers, vector<bool> &integratedFlags, 
									      FlaggerHistory& history) {
  integratePowers(powers, flags, integratedPowers, integratedFlags);
  sumThresholdFlagger1DWithHistory(integratedPowers, integratedFlags, itsBaseSensitivity, history);
}


void PreCorrelationFlagger::integratingSumThresholdFlaggerSmoothedWithHistory(const MultiDimArray<float,2> &powers, MultiDimArray<bool,2>& flags,
									      vector<float> &integratedPowers, vector<float> &smoothedPowers,
									      vector<float> &powerDiffs, vector<bool> &integratedFlags, 
									      FlaggerHistory& history) {
  integratePowers(powers, flags, integratedPowers, integratedFlags);
  sumThresholdFlagger1DSmoothedWithHistory(integratedPowers, smoothedPowers, powerDiffs, integratedFlags, history);
}


void PreCorrelationFlagger::integratePowers(const MultiDimArray<float,2>& powers, MultiDimArray<bool,2>& flags,
					    vector<float>& integratedPowers, vector<bool>& integratedFlags) {
  // sum all powers over time to increase the signal-to-noise-ratio
  for (unsigned channel = 0; channel < itsNrChannels; channel++) {
    float powerSum = 0.0f;
    bool flagged = false;
    for (unsigned time = 0; time < itsNrSamplesPerIntegration; time++) {
      powerSum += powers[channel][time];
      flagged |= flags[channel][time];
    }
    integratedPowers[channel] = powerSum;
    integratedFlags[channel]  = flagged;
  }
}


void PreCorrelationFlagger::integratePowers2D(const MultiDimArray<float,2>& powers, MultiDimArray<bool,2>& flags,
					      MultiDimArray<float,2>& integratedPowers2D, MultiDimArray<bool,2>& integratedFlags2D) {
  // Sum powers over time to increase the signal-to-noise-ratio.
  // We do this in groups of itsIntegrationFactor.
  unsigned nrTimes = itsNrSamplesPerIntegration / itsIntegrationFactor;

  for (unsigned channel = 0; channel < itsNrChannels; channel++) {
    for(unsigned block = 0; block < nrTimes; block++) {
      float powerSum = 0.0f;
      bool flagged = false;
      for (unsigned time = 0; time < itsIntegrationFactor; time++) {
	unsigned globalIndex = block * itsIntegrationFactor + time;
	powerSum += powers[channel][globalIndex];
	flagged |= flags[channel][globalIndex];
      }
      integratedPowers2D[channel][block] = powerSum;
      integratedFlags2D[channel][block]  = flagged;
    }
  }
}


// data:  [nrChannels][nrStations][nrSamplesPerIntegration | 2][NR_POLARIZATIONS]
void PreCorrelationFlagger::calculatePowers(unsigned station, unsigned pol, FilteredData* filteredData) {
  itsTotalPower = 0.0f;
  for (unsigned channel = 0; channel < itsNrChannels; channel++) {
    for (unsigned time = 0; time < itsNrSamplesPerIntegration; time++) {
      fcomplex sample = filteredData->samples[channel][station][time][pol];
      float power = real(sample) * real(sample) + imag(sample) * imag(sample);
      itsPowers[channel][time] = power;
      itsTotalPower += power;
    }
  }
}


// flags: nrStations -> nrSamplesPerIntegration
void PreCorrelationFlagger::initFlags(unsigned station, FilteredData* filteredData) {
  switch(itsFlaggerType) {
  case PRE_FLAGGER_THRESHOLD:
    // not integrated
    for (unsigned channel = 0; channel < itsNrChannels; channel++) {
      for (unsigned time = 0; time < itsNrSamplesPerIntegration; time++) {
	itsFlags[channel][time] = false;
      }
    }

    // Use the original coarse flags to initialize the flags.
    for (unsigned time = 0; time < itsNrSamplesPerIntegration; time++) {
      for (unsigned channel = 0; channel < itsNrChannels; channel++) {
        if(filteredData->flags[channel][station].test(time)) {
	  itsFlags[channel][time] = true;
	}
      }
    }
    break;
  case PRE_FLAGGER_INTEGRATED_THRESHOLD:
  case PRE_FLAGGER_INTEGRATED_SUM_THRESHOLD:
  case PRE_FLAGGER_INTEGRATED_SUM_THRESHOLD_WITH_HISTORY:
  case PRE_FLAGGER_INTEGRATED_SMOOTHED_SUM_THRESHOLD:
  case PRE_FLAGGER_INTEGRATED_SMOOTHED_SUM_THRESHOLD_WITH_HISTORY:
    // The flags of all channels are still the same in the input FilteredData, so we just use channel 1.
    // Flags are kept per channel, since we will do online flagging on FilteredData later.

    // fully integrated
    for (unsigned channel = 0; channel < itsNrChannels; channel++) {
      itsIntegratedFlags[channel] = false;
    }
    // Use the original coarse flags to initialize the flags.
    if(filteredData->flags[itsNrChannels == 1 ? 0 : 1][station].count() > 0) { // We are integrating, so if any sample in time is flagged, everything is flagged.
      for (unsigned channel = 0; channel < itsNrChannels; channel++) {
	itsIntegratedFlags[channel] = true;
      }
    }
    break;
  case PRE_FLAGGER_INTEGRATED_THRESHOLD_2D:
  case PRE_FLAGGER_INTEGRATED_SUM_THRESHOLD_2D:
  case PRE_FLAGGER_INTEGRATED_SUM_THRESHOLD_2D_WITH_HISTORY:
    // Partially integrated
    {
      unsigned nrTimes = itsNrSamplesPerIntegration / itsIntegrationFactor;
      for (unsigned channel = 0; channel < itsNrChannels; channel++) {
	for(unsigned block = 0; block < nrTimes; block++) {
	  itsIntegratedFlags2D[channel][block] = false;
	}
      }

      // Use the original coarse flags to initialize the flags.
      for (unsigned time = 0; time < itsNrSamplesPerIntegration; time++) {
	if(filteredData->flags[itsNrChannels == 1 ? 0 : 1][station].test(time)) {
	  for (unsigned channel = 0; channel < itsNrChannels; channel++) {
	    itsIntegratedFlags2D[channel][time/itsIntegrationFactor] = true;
	  }
	}
      }
    }
    break;
  default:
    LOG_INFO_STR("ERROR, illegal FlaggerType. Skipping online pre correlation flagger.");
    return;
  }
}


void PreCorrelationFlagger::applyFlags(unsigned station, FilteredData* filteredData) {
  switch(itsFlaggerType) {
  case PRE_FLAGGER_THRESHOLD:
    // not integrated
    for (unsigned channel = 0; channel < itsNrChannels; channel++) {
      for (unsigned time = 0; time < itsNrSamplesPerIntegration; time++) {
	if(itsFlags[channel][time]) {
	  flagSample(filteredData, channel, station, time);
	}
      }
    }
    break;
  case PRE_FLAGGER_INTEGRATED_THRESHOLD:
  case PRE_FLAGGER_INTEGRATED_SUM_THRESHOLD:
  case PRE_FLAGGER_INTEGRATED_SUM_THRESHOLD_WITH_HISTORY:
  case PRE_FLAGGER_INTEGRATED_SMOOTHED_SUM_THRESHOLD:
  case PRE_FLAGGER_INTEGRATED_SMOOTHED_SUM_THRESHOLD_WITH_HISTORY:
    // fully integrated
    for (unsigned channel = 0; channel < itsNrChannels; channel++) {
      if(itsIntegratedFlags[channel]) {
	for (unsigned time = 0; time < itsNrSamplesPerIntegration; time++) {
	  flagSample(filteredData, channel, station, time);
	}
      }
    }
    break;
  case PRE_FLAGGER_INTEGRATED_THRESHOLD_2D:
  case PRE_FLAGGER_INTEGRATED_SUM_THRESHOLD_2D:
  case PRE_FLAGGER_INTEGRATED_SUM_THRESHOLD_2D_WITH_HISTORY:
    // Partially integrated
    {
      unsigned nrTimes = itsNrSamplesPerIntegration / itsIntegrationFactor;
      const fcomplex zero = makefcomplex(0, 0);

      for (unsigned channel = 0; channel < itsNrChannels; channel++) {
	for(unsigned block = 0; block < nrTimes; block++) {
	  if(itsIntegratedFlags2D[channel][block]) {
	    unsigned startIndex = block * itsIntegrationFactor;

	    filteredData->flags[channel][station].include(startIndex, startIndex+itsIntegrationFactor);

	    for (unsigned time = 0; time < itsIntegrationFactor; time++) {
	      unsigned globalIndex = block * itsIntegrationFactor + time;
	      for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol++) {
		filteredData->samples[channel][station][globalIndex][pol] = zero;
	      }
	    }
	  }
	}
      }
    }
    break;
  default:
    LOG_INFO_STR("ERROR, illegal FlaggerType. Skipping online pre correlation flagger.");
    return;
  }
}


void PreCorrelationFlagger::flagSample(FilteredData* filteredData, unsigned channel, unsigned station, unsigned time) {
  filteredData->flags[channel][station].include(time);
  const fcomplex zero = makefcomplex(0, 0);
  for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol++) {
    filteredData->samples[channel][station][time][pol] = zero;
  }
}


PreCorrelationFlaggerType PreCorrelationFlagger::getFlaggerType(std::string t) {
  if (t.compare("THRESHOLD") == 0) {
    return PRE_FLAGGER_THRESHOLD;
  } else if (t.compare("INTEGRATED_THRESHOLD") == 0) {
    return PRE_FLAGGER_INTEGRATED_THRESHOLD;
  } else if (t.compare("INTEGRATED_THRESHOLD_2D") == 0) {
    return PRE_FLAGGER_INTEGRATED_THRESHOLD_2D;
  } else if (t.compare("INTEGRATED_SUM_THRESHOLD") == 0) {
    return PRE_FLAGGER_INTEGRATED_SUM_THRESHOLD;
  } else if (t.compare("INTEGRATED_SUM_THRESHOLD_WITH_HISTORY") == 0) {
    return PRE_FLAGGER_INTEGRATED_SUM_THRESHOLD_WITH_HISTORY;
  } else if (t.compare("INTEGRATED_SUM_THRESHOLD_2D") == 0) {
    return PRE_FLAGGER_INTEGRATED_SUM_THRESHOLD_2D;
  } else if (t.compare("INTEGRATED_SUM_THRESHOLD_2D_WITH_HISTORY") == 0) {
    return PRE_FLAGGER_INTEGRATED_SUM_THRESHOLD_2D_WITH_HISTORY;
  } else if (t.compare("INTEGRATED_SMOOTHED_SUM_THRESHOLD") == 0) {
    return PRE_FLAGGER_INTEGRATED_SMOOTHED_SUM_THRESHOLD;
  } else if (t.compare("INTEGRATED_SMOOTHED_SUM_THRESHOLD_WITH_HISTORY") == 0) {
    return PRE_FLAGGER_INTEGRATED_SMOOTHED_SUM_THRESHOLD_WITH_HISTORY;
  } else {
    LOG_DEBUG_STR("unknown flagger type, using default THRESHOLD");
    return PRE_FLAGGER_THRESHOLD;
  }
}


std::string PreCorrelationFlagger::getFlaggerTypeString(PreCorrelationFlaggerType t) {
  switch(t) {
  case PRE_FLAGGER_THRESHOLD:
    return "THRESHOLD";
  case PRE_FLAGGER_INTEGRATED_THRESHOLD:
    return "INTEGRATED_THRESHOLD";
  case PRE_FLAGGER_INTEGRATED_THRESHOLD_2D:
    return "INTEGRATED_THRESHOLD_2D";
  case PRE_FLAGGER_INTEGRATED_SUM_THRESHOLD:
    return "INTEGRATED_SUM_THRESHOLD";
  case PRE_FLAGGER_INTEGRATED_SUM_THRESHOLD_WITH_HISTORY:
    return "INTEGRATED_SUM_THRESHOLD_WITH_HISTORY";
  case PRE_FLAGGER_INTEGRATED_SUM_THRESHOLD_2D:
    return "INTEGRATED_SUM_THRESHOLD_2D";
  case PRE_FLAGGER_INTEGRATED_SUM_THRESHOLD_2D_WITH_HISTORY:
    return "INTEGRATED_SUM_THRESHOLD_2D_WITH_HISTORY";
  case PRE_FLAGGER_INTEGRATED_SMOOTHED_SUM_THRESHOLD:
    return "INTEGRATED_SMOOTHED_SUM_THRESHOLD";
  case PRE_FLAGGER_INTEGRATED_SMOOTHED_SUM_THRESHOLD_WITH_HISTORY:
    return "INTEGRATED_SMOOTHED_SUM_THRESHOLD_WITH_HISTORY";
  default:
    return "ILLEGAL FLAGGER TYPE";
  }
}


std::string PreCorrelationFlagger::getFlaggerTypeString() {
  return getFlaggerTypeString(itsFlaggerType);
}

} // namespace RTCP
} // namespace LOFAR
