#ifndef LOFAR_CNPROC_PRE_CORRELATION_FLAGGER_H
#define LOFAR_CNPROC_PRE_CORRELATION_FLAGGER_H

#include <Flagger.h>
#include <Interface/FilteredData.h>

namespace LOFAR {
namespace RTCP {

enum PreCorrelationFlaggerType {
  PRE_FLAGGER_THRESHOLD,

  PRE_FLAGGER_INTEGRATED_THRESHOLD,
  PRE_FLAGGER_INTEGRATED_SUM_THRESHOLD,
  PRE_FLAGGER_INTEGRATED_SUM_THRESHOLD_WITH_HISTORY,

  PRE_FLAGGER_INTEGRATED_SMOOTHED_SUM_THRESHOLD,
  PRE_FLAGGER_INTEGRATED_SMOOTHED_SUM_THRESHOLD_WITH_HISTORY,

  PRE_FLAGGER_INTEGRATED_THRESHOLD_2D,
  PRE_FLAGGER_INTEGRATED_SUM_THRESHOLD_2D,
  PRE_FLAGGER_INTEGRATED_SUM_THRESHOLD_2D_WITH_HISTORY,
};

// If we only have few channels (e.g., 16), we have to do 2D flagging, otherwise we don't have enough data to do statistics.
// So, we partially integrate in the time direction.
class PreCorrelationFlagger : public Flagger {
  public:
  PreCorrelationFlagger(const Parset& parset, const unsigned nrStations, const unsigned nrSubbands, const unsigned nrChannels, const unsigned nrSamplesPerIntegration, float cutoffThreshold = 7.0f);

  void flag(FilteredData* filteredData, unsigned currentSubband);

  private:

  // Does simple thresholding.
  void integratingThresholdingFlagger(const MultiDimArray<float,2> &powers, MultiDimArray<bool,2>& flags, 
				      vector<float> &integratedPowers, vector<bool> &integratedFlags);

  void integratingThresholdingFlagger2D(const MultiDimArray<float,2> &powers, MultiDimArray<bool,2>& flags, 
					MultiDimArray<float,2>& integratedPowers2D,  MultiDimArray<bool,2>& integratedFlags2D);

  void integratingSumThresholdFlagger(const MultiDimArray<float,2> &powers, MultiDimArray<bool,2>& flags, 
				      vector<float> &integratedPowers, vector<bool> &integratedFlags);

  void integratingSumThresholdFlaggerWithHistory(const MultiDimArray<float,2> &powers, MultiDimArray<bool,2>& flags, 
						 vector<float> &integratedPowers,
						 vector<bool> &integratedFlags, FlaggerHistory& history);

  void integratingSumThresholdFlagger2D(const MultiDimArray<float,2> &powers, MultiDimArray<bool,2>& flags,
					MultiDimArray<float,2>& integratedPowers2D,  
					MultiDimArray<bool,2>& integratedFlags2D);

  void integratingSumThresholdFlagger2DWithHistory(const MultiDimArray<float,2> &powers, MultiDimArray<bool,2>& flags,
						   MultiDimArray<float,2>& integratedPowers2D, MultiDimArray<bool,2>& integratedFlags2D,
						   vector<float> &integratedPowers, vector<bool> &integratedFlags,
						   MultiDimArray<FlaggerHistory, 3>& history, unsigned station, unsigned subband, unsigned pol);

  void integratingSumThresholdFlaggerSmoothed(const MultiDimArray<float,2> &powers, MultiDimArray<bool,2>& flags, 
					      vector<float> &integratedPowers, vector<float> &smoothedPowers, vector<float> &powerDiffs, vector<bool> &integratedFlags);

  void integratingSumThresholdFlaggerSmoothedWithHistory(const MultiDimArray<float,2> &powers, MultiDimArray<bool,2>& flags, 
							 vector<float> &integratedPowers, vector<float> &smoothedPowers, vector<float> &powerDiffs, 
							 vector<bool> &integratedFlags, FlaggerHistory& history);


  void calculatePowers(unsigned station, unsigned pol, FilteredData* filteredData);
  void integratePowers(const MultiDimArray<float,2>& powers, MultiDimArray<bool,2>& flags,
		       vector<float>& integratedPowers, vector<bool>& integratedFlags);
  void integratePowers2D(const MultiDimArray<float,2>& powers, MultiDimArray<bool,2>& flags,
			 MultiDimArray<float,2>& integratedPowers2D,  
			 MultiDimArray<bool,2>& integratedFlags2D);


  void initFlags(unsigned station, FilteredData* filteredData);
  void applyFlags(unsigned station, FilteredData* filteredData);

  void flagSample(FilteredData* filteredData, unsigned channel, unsigned station, unsigned time);
  void wipeFlaggedSamples(unsigned station, FilteredData* filteredData);

  PreCorrelationFlaggerType getFlaggerType(std::string t);
  std::string getFlaggerTypeString(PreCorrelationFlaggerType t);
  std::string getFlaggerTypeString();

  const PreCorrelationFlaggerType itsFlaggerType;
  const unsigned itsNrSamplesPerIntegration;
  unsigned itsIntegrationFactor; 

  float itsTotalPower;

  MultiDimArray<float,2> itsPowers; // [itsNrChannels][itsNrSamplesPerIntegration]
  MultiDimArray<float,2> itsIntegratedPowers2D; // [itsNrChannels][itsNrSamplesPerIntegration/itsIntegrationFactor]
  vector<float> itsIntegratedPowers; // [itsNrChannels]

  MultiDimArray<bool,2> itsFlags; // [itsNrChannels][itsNrSamplesPerIntegration]
  MultiDimArray<bool,2> itsIntegratedFlags2D; // [itsNrChannels][itsNrSamplesPerIntegration/itsIntegrationFactor]
  vector<bool> itsIntegratedFlags; // [itsNrChannels]

  std::vector<float> itsSmoothedIntegratedPowers; // [itsNrChannels]
  std::vector<float> itsIntegratedPowerDiffs; // [itsNrChannels]

  MultiDimArray<FlaggerHistory, 3> itsHistory;   // [nrStations][nrSubbands][NR_POLARIZATIONS]

};

} // namespace RTCP
} // namespace LOFAR

#endif // LOFAR_CNPROC_PRE_CORRELATION_FLAGGER_H
