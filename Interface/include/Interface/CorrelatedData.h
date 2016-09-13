#ifndef LOFAR_INTERFACE_CORRELATED_DATA_H
#define LOFAR_INTERFACE_CORRELATED_DATA_H

#include <Common/lofar_complex.h>
#include <Common/DataConvert.h>
#include <Interface/Align.h>
#include <Interface/Allocator.h>
#include <Interface/Config.h>
#include <Interface/StreamableData.h>
#include <Interface/BGPAsm.h>
#include <Interface/MultiDimArray.h>
#include <Stream/Stream.h>


namespace LOFAR {
namespace RTCP {

class CorrelatedData : public StreamableData, public IntegratableData
{
  public:
    CorrelatedData(unsigned nrStations, unsigned nrChannels, unsigned maxNrValidSamples, Allocator & = heapAllocator, unsigned alignment = 512);
    CorrelatedData(unsigned nrStations, unsigned nrChannels, unsigned maxNrValidSamples, std::complex<float> *visibilities, size_t nrVisibilities, Allocator & = heapAllocator, unsigned alignment = 512);

    virtual IntegratableData &operator += (const IntegratableData &);

    unsigned			nrValidSamples(unsigned bl, unsigned ch) const;
    void			setNrValidSamples(unsigned bl, unsigned ch, unsigned value);

    const unsigned              itsAlignment;
    const unsigned              itsNrBaselines;

    MultiDimArray<fcomplex, 4>	visibilities; //[nrBaselines][nrChannels][NR_POLARIZATIONS][NR_POLARIZATIONS]

    const unsigned			itsNrBytesPerNrValidSamples;
    Matrix<uint32_t>		itsNrValidSamples4; //[nrBaselines][nrChannels]
    Matrix<uint16_t>		itsNrValidSamples2; //[nrBaselines][nrChannels]
    Matrix<uint8_t>		itsNrValidSamples1; //[nrBaselines][nrChannels]

  protected:
    virtual void		readData(Stream *);
    virtual void		writeData(Stream *);

  private:
    void init(unsigned nrChannels, Allocator &allocator);
};


inline CorrelatedData::CorrelatedData(unsigned nrStations, unsigned nrChannels, unsigned maxNrValidSamples, Allocator &allocator, unsigned alignment)
:
  itsAlignment(alignment),
  itsNrBaselines(nrStations * (nrStations + 1) / 2),
  visibilities(boost::extents[itsNrBaselines][nrChannels][NR_POLARIZATIONS][NR_POLARIZATIONS], itsAlignment, allocator, true),
  itsNrBytesPerNrValidSamples(maxNrValidSamples < 256 ? 1 : maxNrValidSamples < 65536 ? 2 : 4)
{
  init(nrChannels, allocator);
}


inline CorrelatedData::CorrelatedData(unsigned nrStations, unsigned nrChannels, unsigned maxNrValidSamples, std::complex<float> *visibilities, size_t nrVisibilities, Allocator &allocator, unsigned alignment)
:
  itsAlignment(alignment),
  itsNrBaselines(nrStations * (nrStations + 1) / 2),
  visibilities(boost::extents[itsNrBaselines][nrChannels][NR_POLARIZATIONS][NR_POLARIZATIONS], visibilities, false),
  itsNrBytesPerNrValidSamples(maxNrValidSamples < 256 ? 1 : maxNrValidSamples < 65536 ? 2 : 4)
{
  ASSERT(this->visibilities.num_elements() == nrVisibilities);

  init(nrChannels, allocator); 
}

inline void CorrelatedData::init(unsigned nrChannels, Allocator &allocator)
{
  switch (itsNrBytesPerNrValidSamples) {
    case 4 : itsNrValidSamples4.resize(boost::extents[itsNrBaselines][nrChannels], itsAlignment, allocator, true);
            break;
      
    case 2 : itsNrValidSamples2.resize(boost::extents[itsNrBaselines][nrChannels], itsAlignment, allocator, true);
            break;

    case 1 : itsNrValidSamples1.resize(boost::extents[itsNrBaselines][nrChannels], itsAlignment, allocator, true);
            break;
  }
}


inline unsigned CorrelatedData::nrValidSamples(unsigned bl, unsigned ch) const
{
  switch (itsNrBytesPerNrValidSamples) {
    case 4 : return itsNrValidSamples4[bl][ch];
    case 2 : return itsNrValidSamples2[bl][ch];
    case 1 : return itsNrValidSamples1[bl][ch];
  }

  return 0;
}


inline void CorrelatedData::setNrValidSamples(unsigned bl, unsigned ch, unsigned value)
{
  switch (itsNrBytesPerNrValidSamples) {
    case 4 : itsNrValidSamples4[bl][ch] = value;
	     break;

    case 2 : itsNrValidSamples2[bl][ch] = value;
	     break;

    case 1 : itsNrValidSamples1[bl][ch] = value;
	     break;
  }
}


inline void CorrelatedData::readData(Stream *str)
{
  str->read(visibilities.origin(), align(visibilities.num_elements() * sizeof(fcomplex), itsAlignment));

  switch (itsNrBytesPerNrValidSamples) {
    case 4 : str->read(itsNrValidSamples4.origin(), align(itsNrValidSamples4.num_elements() * sizeof(uint32_t), itsAlignment));
            break;

    case 2 : str->read(itsNrValidSamples2.origin(), align(itsNrValidSamples2.num_elements() * sizeof(uint16_t), itsAlignment));
            break;

    case 1 : str->read(itsNrValidSamples1.origin(), align(itsNrValidSamples1.num_elements() * sizeof(uint8_t), itsAlignment));
            break;
  }
}


inline void CorrelatedData::writeData(Stream *str) 
{
  str->write(visibilities.origin(), align(visibilities.num_elements() * sizeof *visibilities.origin(), itsAlignment));

  switch (itsNrBytesPerNrValidSamples) {
    case 4 : str->write(itsNrValidSamples4.origin(), align(itsNrValidSamples4.num_elements() * sizeof(uint32_t), itsAlignment));
            break;

    case 2 : str->write(itsNrValidSamples2.origin(), align(itsNrValidSamples2.num_elements() * sizeof(uint16_t), itsAlignment));
            break;

    case 1 : str->write(itsNrValidSamples1.origin(), align(itsNrValidSamples1.num_elements() * sizeof(uint8_t), itsAlignment));
            break;
  }
}


template <typename T> inline void addNrValidSamples(T * __restrict__ dst, const T * __restrict__ src, unsigned count)
{
  for (unsigned i = 0; i < count; i ++)
    dst[i] += src[i];
}


template<> inline void addNrValidSamples<uint16_t>(uint16_t * __restrict__ dst, const uint16_t * __restrict__ src, unsigned count)
{
  addNrValidSamples<uint32_t>(reinterpret_cast<uint32_t*>(dst), reinterpret_cast<const uint32_t*>(src), count / 2);

  if (count & 1)
    dst[count - 1] += src[count - 1];
}


template<> inline void addNrValidSamples<uint8_t>(uint8_t * __restrict__ dst, const uint8_t * __restrict__ src, unsigned count)
{
  addNrValidSamples<uint16_t>(reinterpret_cast<uint16_t*>(dst), reinterpret_cast<const uint16_t*>(src), count / 2);

  if (count & 1)
    dst[count - 1] += src[count - 1];
}


inline IntegratableData &CorrelatedData::operator += (const IntegratableData &other_)
{
  const CorrelatedData &other = static_cast<const CorrelatedData &>(other_);

  // add visibilities
  {
    fcomplex	   *dst	 = visibilities.origin();
    const fcomplex *src	 = other.visibilities.origin();
    unsigned	   count = visibilities.num_elements();
#ifdef HAVE_BGP
    unsigned fastcopyfloats = (count * 2) & ~0xF;
    unsigned remainder = count % 8;

    for (unsigned i = 0; i < remainder; i ++)
      dst[i] += src[i];

    if (fastcopyfloats > 0)
      _add_2_single_precision_vectors( reinterpret_cast<float*>(dst + remainder), reinterpret_cast<float*>(dst + remainder), reinterpret_cast<const float*>(src + remainder), fastcopyfloats );
#else
    for (unsigned i = 0; i < count; i ++)
      dst[i] += src[i];
#endif
  }

  // add nr. valid samples
  switch (itsNrBytesPerNrValidSamples) {
    case 4 : addNrValidSamples(itsNrValidSamples4.origin(), other.itsNrValidSamples4.origin(), itsNrValidSamples4.num_elements());
	     break;

    case 2 : addNrValidSamples(itsNrValidSamples2.origin(), other.itsNrValidSamples2.origin(), itsNrValidSamples2.num_elements());
	     break;

    case 1 : addNrValidSamples(itsNrValidSamples1.origin(), other.itsNrValidSamples1.origin(), itsNrValidSamples1.num_elements());
	     break;
  }

  return *this;
}


} // namespace RTCP
} // namespace LOFAR

#endif
