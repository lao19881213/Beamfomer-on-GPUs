//# Always #include <lofar_config.h> first!
#include <lofar_config.h>

#include <FIR.h>
#include <cstring>

#include <math.h>
#include <iostream>
#include <cstring>

#include <Common/LofarLogger.h>


namespace LOFAR {
namespace RTCP {

template <typename FIR_SAMPLE_TYPE> FIR<FIR_SAMPLE_TYPE>::FIR()
{
}


template <typename FIR_SAMPLE_TYPE> void FIR<FIR_SAMPLE_TYPE>::initFilter(FilterBank *filterBank, unsigned channel)
{
  itsFilterBank = filterBank;
  itsChannel = channel;
  itsNrTaps = filterBank->getNrTaps();
  itsWeights = filterBank->getWeights(channel);
  itsCurrentIndex = 0;

  itsDelayLine.resize(itsNrTaps);
  memset(itsDelayLine.data(), 0, sizeof(FIR_SAMPLE_TYPE) * itsNrTaps);
}


template <typename FIR_SAMPLE_TYPE> FIR_SAMPLE_TYPE FIR<FIR_SAMPLE_TYPE>::processNextSample(FIR_SAMPLE_TYPE sample)
{
  FIR_SAMPLE_TYPE *delayPtr = &itsDelayLine[0];
#if 0
  FIR_SAMPLE_TYPE sum = sample * itsWeights[0];
  delayPtr[0] = sample;

  for (int tap = itsNrTaps; -- tap > 0;) {
    sum += itsWeights[tap] * delayPtr[tap];
    delayPtr[tap] = delayPtr[tap - 1];
  }
#else
  FIR_SAMPLE_TYPE sum = 0;
  delayPtr[itsCurrentIndex] = sample;

  for (int tap = itsNrTaps - itsCurrentIndex; -- tap >= 0;)
    sum += delayPtr[itsCurrentIndex + tap] * itsWeights[tap];

  float *weightPtr = &itsWeights[itsNrTaps - itsCurrentIndex];

  for (int tap = 0; tap < itsCurrentIndex; tap ++)
    sum += delayPtr[tap] * weightPtr[tap];

  if (-- itsCurrentIndex < 0)
    itsCurrentIndex += itsNrTaps;
#endif

  return sum;
}

template class FIR<float>;
template class FIR<fcomplex>;

} // namespace RTCP
} // namespace LOFAR
