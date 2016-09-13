#ifndef LOFAR_INTERFACE_FAKE_DATA_H
#define LOFAR_INTERFACE_FAKE_DATA_H

#include <Interface/FilteredData.h>
#include <Interface/BeamFormedData.h>
#include <Interface/Parset.h>
#include <Common/LofarLogger.h>
#include <Common/LofarTypes.h>
#include <cmath>

namespace LOFAR {
namespace RTCP {

class FakeData {
  public:
    FakeData( const Parset &parset ): itsParset(parset) {}

    void fill( FilteredData *data, unsigned subband ) const;
    void check( const FilteredData *data ) const;
    void check( const FinalBeamFormedData *data, unsigned pol ) const;

    void check( const StreamableData *data, OutputType outputType, unsigned streamNr ) const;

  private:
    const Parset &itsParset;
    static const double TOLERANCE = 1e-6;

    template<typename T> bool equal( const T a, const T b ) const { return a == b; }
};

template<> bool FakeData::equal( const float a, const float b ) const {
  return fabsf(a - b) < TOLERANCE;
}

template<> bool FakeData::equal( const double a, const double b ) const {
  return fabs(a - b) < TOLERANCE;
}

template<> bool FakeData::equal( const fcomplex a, const fcomplex b ) const {
  return equal(real(a), real(b)) && equal(imag(a), imag(b));
}

void FakeData::fill( FilteredData *data, unsigned subband ) const
{
  for (unsigned s = 0; s < itsParset.nrStations(); s++) {
    for (unsigned c = 0; c < itsParset.nrChannelsPerSubband(); c++) {
      for (unsigned t = 0; t < itsParset.CNintegrationSteps(); t++) {
        const float base = 1000 * subband;
        data->samples[c][s][t][0] = makefcomplex(base + 1 * t, base + 2 * t);
        data->samples[c][s][t][1] = makefcomplex(base + 3 * t, base + 5 * t);
      }
      data->flags[c][s].reset();
    }
  }
}

void FakeData::check( const FilteredData *data ) const
{
  for (unsigned s = 0; s < itsParset.nrStations(); s++) {
    for (unsigned c = 0; c < itsParset.nrChannelsPerSubband(); c++) {
      for (unsigned t = 0; t < itsParset.CNintegrationSteps(); t++) {
        ASSERT( equal( data->samples[c][s][t][0], makefcomplex(1 * t, 2 * t) ) );
        ASSERT( equal( data->samples[c][s][t][1], makefcomplex(3 * t, 5 * t) ) );
      }
      ASSERT( data->flags[c][s].count() == 0 );
    }
  }
}

void FakeData::check( const FinalBeamFormedData* /* data */, unsigned /* pol */) const
{
  // TODO: support other configurations than just 1 station equal to reference phase center
/*
  for (unsigned t = 0; t < itsParset.CNintegrationSteps(); t++) {
    for (unsigned s = 0; s < itsParset.nrSubbands(); s++) {
      for (unsigned c = 0; c < itsParset.nrChannelsPerSubband(); c++) {
        switch (pol) {
          case 0: // Xr
            ASSERT( equal( data->samples[t][s][c], 1.0f * t ) );
            break;
          case 1: // Xi
            ASSERT( equal( data->samples[t][s][c], 2.0f * t ) );
            break;
          case 2: // Yr
            ASSERT( equal( data->samples[t][s][c], 3.0f * t ) );
            break;
          case 3: // Yi
            ASSERT( equal( data->samples[t][s][c], 5.0f * t ) );
            break;
        }
      }
    }  
  }
*/
}

void FakeData::check( const StreamableData *data, OutputType outputType, unsigned streamNr ) const
{
  switch (outputType) {
    case BEAM_FORMED_DATA:
      check( static_cast<const FinalBeamFormedData *>(data), streamNr % NR_POLARIZATIONS );
      break;

    default:
      return;
  }
}

} // namespace RTCP
} // namespace LOFAR

#endif
