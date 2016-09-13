#ifndef LOFAR_INTERFACE_INVERSE_FILTERED_DATA_H
#define LOFAR_INTERFACE_INVERSE_FILTERED_DATA_H

#include <Common/lofar_complex.h>
#include <Stream/Stream.h>
#include <Interface/Align.h>
#include <Interface/Config.h>
#include <Interface/MultiDimArray.h>
#include <Interface/SparseSet.h>
#include <Interface/StreamableData.h>

namespace LOFAR {
namespace RTCP {

// This assumes the nrChannels == 1
// We store the data for only 1 beam, 1 polarization.
// nrSamplesPerIntegration is the original nrSamplesPerIntegration, and now becomes the "major time" index.
// The stationFilterSize is the minor time index.

class InverseFilteredData: public SampleData<float,1,1>
{
  public:
    typedef SampleData<float,1,1> SuperType;

    InverseFilteredData(unsigned nrSamplesPerIntegration, unsigned stationFilterSize);

  protected:
    const unsigned              itsNrSamplesPerIntegration;
    const unsigned              itsStationFilterSize;
    
};

inline InverseFilteredData::InverseFilteredData(unsigned nrSamplesPerIntegration, unsigned stationFilterSize)
:
  SuperType::SampleData(boost::extents[nrSamplesPerIntegration*stationFilterSize], boost::extents[1]),
  itsNrSamplesPerIntegration(nrSamplesPerIntegration),
  itsStationFilterSize(stationFilterSize)
{
}

} // namespace RTCP
} // namespace LOFAR

#endif
