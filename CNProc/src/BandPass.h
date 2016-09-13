#ifndef LOFAR_CNPROC_BANDPASS_H
#define LOFAR_CNPROC_BANDPASS_H

#include <vector>


namespace LOFAR {
namespace RTCP {

#define STATION_FILTER_LENGTH 16384 // Number of filter taps of the station filters.
#define STATION_FFT_SIZE 1024 // The size of the FFT that the station filter does.

class BandPass {
  public:
			BandPass(bool correct, unsigned nrChannels);

    const float		*correctionFactors() const;

  private:
    void		computeCorrectionFactors(unsigned nrChannels);

    static const float	stationFilterConstants[];
    
    std::vector<float>	factors;
};


inline const float *BandPass::correctionFactors() const
{
  return &factors[0];
}

} // namespace RTCP
} // namespace LOFAR

#endif
