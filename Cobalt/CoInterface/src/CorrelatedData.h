//# CorrelatedData.h
//# Copyright (C) 2008-2013  ASTRON (Netherlands Institute for Radio Astronomy)
//# P.O. Box 2, 7990 AA Dwingeloo, The Netherlands
//#
//# This file is part of the LOFAR software suite.
//# The LOFAR software suite is free software: you can redistribute it and/or
//# modify it under the terms of the GNU General Public License as published
//# by the Free Software Foundation, either version 3 of the License, or
//# (at your option) any later version.
//#
//# The LOFAR software suite is distributed in the hope that it will be useful,
//# but WITHOUT ANY WARRANTY; without even the implied warranty of
//# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//# GNU General Public License for more details.
//#
//# You should have received a copy of the GNU General Public License along
//# with the LOFAR software suite. If not, see <http://www.gnu.org/licenses/>.
//#
//# $Id: CorrelatedData.h 25445 2013-06-24 06:47:16Z mol $

#ifndef LOFAR_INTERFACE_CORRELATED_DATA_H
#define LOFAR_INTERFACE_CORRELATED_DATA_H

#include <Common/lofar_complex.h>
#include <Common/DataConvert.h>
#include <Stream/Stream.h>
#include <CoInterface/Align.h>
#include <CoInterface/Allocator.h>
#include <CoInterface/Config.h>
#include <CoInterface/StreamableData.h>
#include <CoInterface/MultiDimArray.h>
#include <CoInterface/OutputTypes.h>


namespace LOFAR
{
  namespace Cobalt
  {

    class CorrelatedData : public StreamableData, public IntegratableData
    {
    public:
      CorrelatedData(unsigned nrStations, unsigned nrChannels, unsigned maxNrValidSamples, Allocator & = heapAllocator, unsigned alignment = 1);
      CorrelatedData(unsigned nrStations, unsigned nrChannels, unsigned maxNrValidSamples, std::complex<float> *visibilities, size_t nrVisibilities, Allocator & = heapAllocator, unsigned alignment = 1);

      virtual IntegratableData &operator += (const IntegratableData &);

      // Fast access to weights; T = uint32_t, uint16_t, or uint8_t,
      // based on itsNrBytesPerNrValidSamples.
      template<typename T> T &nrValidSamples(unsigned bl, unsigned ch);

      // Slow short-cut functions. Use for testing only!
      unsigned getNrValidSamples(unsigned bl, unsigned ch);
      void setNrValidSamples(unsigned bl, unsigned ch, unsigned value);

      const unsigned itsAlignment;
      const unsigned itsNrBaselines;

      MultiDimArray<fcomplex, 4>  visibilities; //[nrBaselines][nrChannels][NR_POLARIZATIONS][NR_POLARIZATIONS]

      // The size of the nrValidSamples is determined by the maximum value that
      // has to be stored, which fits either in 1, 2, or 4 bytes.
      const unsigned itsNrBytesPerNrValidSamples;     // 1, 2, or 4
    protected:
      virtual void                readData(Stream *, unsigned);
      virtual void                writeData(Stream *, unsigned);

    private:
      void init(unsigned nrChannels, Allocator &allocator);

      Matrix<uint32_t>            itsNrValidSamples4; //[nrBaselines][nrChannels]
      Matrix<uint16_t>            itsNrValidSamples2; //[nrBaselines][nrChannels]
      Matrix<uint8_t>             itsNrValidSamples1; //[nrBaselines][nrChannels]
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
      case 4: itsNrValidSamples4.resize(boost::extents[itsNrBaselines][nrChannels], itsAlignment, allocator, true);
        break;

      case 2: itsNrValidSamples2.resize(boost::extents[itsNrBaselines][nrChannels], itsAlignment, allocator, true);
        break;

      case 1: itsNrValidSamples1.resize(boost::extents[itsNrBaselines][nrChannels], itsAlignment, allocator, true);
        break;
      }

      // zero weights
      for (size_t bl = 0; bl < itsNrBaselines; ++bl) {
        for (size_t ch = 0; ch < nrChannels; ++ch) {
          setNrValidSamples(bl, ch, 0);
        }
      }
    }


    template<> inline uint32_t &CorrelatedData::nrValidSamples<uint32_t>(unsigned bl, unsigned ch)
    {
      return itsNrValidSamples4[bl][ch];
    }


    template<> inline uint16_t &CorrelatedData::nrValidSamples<uint16_t>(unsigned bl, unsigned ch)
    {
      return itsNrValidSamples2[bl][ch];
    }


    template<> inline uint8_t &CorrelatedData::nrValidSamples<uint8_t>(unsigned bl, unsigned ch)
    {
      return itsNrValidSamples1[bl][ch];
    }


    inline unsigned CorrelatedData::getNrValidSamples(unsigned bl, unsigned ch)
    {
      switch (itsNrBytesPerNrValidSamples) {
        case 4: return nrValidSamples<uint32_t>(bl, ch);
        case 2: return nrValidSamples<uint16_t>(bl, ch);
        case 1: return nrValidSamples<uint8_t>(bl, ch);
      }

      // Satisfy compiler
      return 0;
    }


    inline void CorrelatedData::setNrValidSamples(unsigned bl, unsigned ch, unsigned value)
    {
      switch (itsNrBytesPerNrValidSamples) {
        case 4: nrValidSamples<uint32_t>(bl, ch) = value;
          break;

        case 2: nrValidSamples<uint16_t>(bl, ch) = value;
          break;

        case 1: nrValidSamples<uint8_t>(bl, ch) = value;
          break;
      }
    }


    inline void CorrelatedData::readData(Stream *str, unsigned alignment)
    {
      ASSERT(alignment <= itsAlignment);

      str->read(visibilities.origin(), align(visibilities.num_elements() * sizeof(fcomplex), alignment));

      switch (itsNrBytesPerNrValidSamples) {
      case 4: str->read(itsNrValidSamples4.origin(), align(itsNrValidSamples4.num_elements() * sizeof(uint32_t), alignment));
        break;

      case 2: str->read(itsNrValidSamples2.origin(), align(itsNrValidSamples2.num_elements() * sizeof(uint16_t), alignment));
        break;

      case 1: str->read(itsNrValidSamples1.origin(), align(itsNrValidSamples1.num_elements() * sizeof(uint8_t), alignment));
        break;
      }
    }


    inline void CorrelatedData::writeData(Stream *str, unsigned alignment)
    {
      ASSERT(alignment <= itsAlignment);

      str->write(visibilities.origin(), align(visibilities.num_elements() * sizeof *visibilities.origin(), alignment));

      switch (itsNrBytesPerNrValidSamples) {
      case 4: str->write(itsNrValidSamples4.origin(), align(itsNrValidSamples4.num_elements() * sizeof(uint32_t), alignment));
        break;

      case 2: str->write(itsNrValidSamples2.origin(), align(itsNrValidSamples2.num_elements() * sizeof(uint16_t), alignment));
        break;

      case 1: str->write(itsNrValidSamples1.origin(), align(itsNrValidSamples1.num_elements() * sizeof(uint8_t), alignment));
        break;
      }
    }


    template <typename T>
    inline void addNrValidSamples(T * __restrict__ dst, const T * __restrict__ src, unsigned count)
    {
      for (unsigned i = 0; i < count; i++)
        dst[i] += src[i];
    }


    template<>
    inline void addNrValidSamples<uint16_t>(uint16_t * __restrict__ dst, const uint16_t * __restrict__ src, unsigned count)
    {
      addNrValidSamples<uint32_t>(reinterpret_cast<uint32_t*>(dst), reinterpret_cast<const uint32_t*>(src), count / 2);

      if (count & 1)
        dst[count - 1] += src[count - 1];
    }


    template<>
    inline void addNrValidSamples<uint8_t>(uint8_t * __restrict__ dst, const uint8_t * __restrict__ src, unsigned count)
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
        fcomplex       *dst = visibilities.origin();
        const fcomplex *src = other.visibilities.origin();
        unsigned count = visibilities.num_elements();

        for (unsigned i = 0; i < count; i++)
          dst[i] += src[i];
      }

      // add nr. valid samples
      switch (itsNrBytesPerNrValidSamples) {
      case 4: addNrValidSamples(itsNrValidSamples4.origin(), other.itsNrValidSamples4.origin(), itsNrValidSamples4.num_elements());
        break;

      case 2: addNrValidSamples(itsNrValidSamples2.origin(), other.itsNrValidSamples2.origin(), itsNrValidSamples2.num_elements());
        break;

      case 1: addNrValidSamples(itsNrValidSamples1.origin(), other.itsNrValidSamples1.origin(), itsNrValidSamples1.num_elements());
        break;
      }

      return *this;
    }


  } // namespace Cobalt
} // namespace LOFAR

#endif

