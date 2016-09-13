#ifndef LOFAR_INTERFACE_FILTERED_DATA_H
#define LOFAR_INTERFACE_FILTERED_DATA_H

#include <Common/lofar_complex.h>
#include <Stream/Stream.h>
#include <Interface/Align.h>
#include <Interface/Config.h>
#include <Interface/MultiDimArray.h>
#include <Interface/SparseSet.h>
#include <Interface/StreamableData.h>

namespace LOFAR {
namespace RTCP {

class FilteredData : public SampleData<fcomplex, 4, 2>
{
  public:
    typedef SampleData<fcomplex, 4, 2> SuperType;

    FilteredData(unsigned nrStations, unsigned nrChannels, unsigned nrSamplesPerIntegration, Allocator & = heapAllocator);

    void resetFlags(void);
};


inline FilteredData::FilteredData(unsigned nrStations, unsigned nrChannels, unsigned nrSamplesPerIntegration, Allocator &allocator)
:
  // The "| 2" significantly improves transpose speeds for particular
  // numbers of stations due to cache conflict effects.  The extra memory
  // is not used.
  SuperType::SampleData(boost::extents[nrChannels][nrStations][nrSamplesPerIntegration | 2][NR_POLARIZATIONS], boost::extents[nrChannels][nrStations], allocator)
{
}


inline void FilteredData::resetFlags(void)
{
  for(unsigned c=0; c < flags.shape()[0]; c++) {
    for(unsigned s=0; s < flags.shape()[1]; s++) {
      flags[c][s].reset();
    }
  }
}

} // namespace RTCP
} // namespace LOFAR

#endif
