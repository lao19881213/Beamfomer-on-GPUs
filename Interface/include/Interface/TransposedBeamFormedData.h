#ifndef LOFAR_CNPROC_TRANSPOSED_BEAMFORMED_DATA_H
#define LOFAR_CNPROC_TRANSPOSED_BEAMFORMED_DATA_H

#include <Common/lofar_complex.h>
#include <Interface/StreamableData.h>

#include <vector>


namespace LOFAR {
namespace RTCP {

// Polarizations are separated, otherwise the buffers do not fit in memory.

class TransposedBeamFormedData: public SampleData<fcomplex,3,1>
{
  public:
    typedef SampleData<fcomplex,3,1> SuperType;

    TransposedBeamFormedData(unsigned nrSubbands, unsigned nrChannels, unsigned nrSamplesPerIntegration);
};


inline TransposedBeamFormedData::TransposedBeamFormedData(unsigned nrSubbands, unsigned nrChannels, unsigned nrSamplesPerIntegration)
:
  SuperType(boost::extents[nrSubbands][nrChannels][nrSamplesPerIntegration | 2], boost::extents[1])
{
}

} // namespace RTCP
} // namespace LOFAR

#endif
