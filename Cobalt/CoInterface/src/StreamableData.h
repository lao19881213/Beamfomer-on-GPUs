//# StreamableData.h
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
//# $Id: StreamableData.h 25445 2013-06-24 06:47:16Z mol $

#ifndef LOFAR_INTERFACE_STREAMABLE_DATA_H
#define LOFAR_INTERFACE_STREAMABLE_DATA_H

#include <cstring>

#include <Common/LofarTypes.h>
#include <Common/DataConvert.h>
#include <CoInterface/Parset.h>
#include <CoInterface/MultiDimArray.h>
#include <CoInterface/SparseSet.h>
#include <CoInterface/Allocator.h>
#include <CoInterface/BlockID.h>
#include <Stream/Stream.h>

namespace LOFAR
{
  namespace Cobalt
  {

    // Data which needs to be transported between CN, ION and Storage.
    // Apart from read() and write() functionality, the data is augmented
    // with a sequence number in order to detect missing data. Furthermore,
    // an integration operator += can be defined to reduce the data.

    // Endianness:
    // * Endianness is defined by the correlator.
    // * Both Data and sequence number will have endianness of the
    //   correlator
    //
    // WARNING: We consider all data streams to have the endianness of the
    // correlator. No conversion is done here.

    class IntegratableData
    {
    public:
      virtual ~IntegratableData()
      {
      }

      virtual IntegratableData &operator += (const IntegratableData &) = 0;
    };


    class StreamableData
    {
    public:
      static const uint32_t magic = 0xda7a;
      static const size_t alignment = 512;

      // Freely modified by GPUProc (only)
      struct BlockID blockID;

      // the CPU which fills the datastructure sets the peerMagicNumber,
      // because other CPUs will overwrite it with a read(s,true) call from
      // either disk or network.
      StreamableData() : peerMagicNumber(magic), rawSequenceNumber(0)
      {
      }
      virtual ~StreamableData()
      {
      }

      void read(Stream *, bool withSequenceNumber, unsigned align = 1);
      void write(Stream *, bool withSequenceNumber, unsigned align = 1);

      bool shouldByteSwap() const
      {
        return peerMagicNumber != magic;
      }

      uint32_t sequenceNumber(bool raw = false) const
      {
        if (shouldByteSwap() && !raw) {
          uint32_t seqno = rawSequenceNumber;

          byteSwap32(&seqno);

          return seqno;
        } else {
          return rawSequenceNumber;
        }
      }

      void setSequenceNumber(uint32_t seqno)
      {
        if (shouldByteSwap())
          byteSwap32(&seqno);

        rawSequenceNumber = seqno;
      }

      virtual void setDimensions(unsigned, unsigned, unsigned)
      {
      }

      uint32_t peerMagicNumber;  /// magic number received from peer

    protected:
      // a subclass should override these to marshall its data
      virtual void readData(Stream *, unsigned) = 0;
      virtual void writeData(Stream *, unsigned) = 0;

    private:
      uint32_t rawSequenceNumber; /// possibly needs byte swapping
    };


    // A typical data set contains a MultiDimArray of tuples and a set of flags.
    template <typename T = fcomplex, unsigned DIM = 4, unsigned FLAGS_DIM = 2>
    class SampleData : public StreamableData
    {
    public:
      typedef typename MultiDimArray<T,DIM>::ExtentList ExtentList;
      typedef typename MultiDimArray<SparseSet<unsigned>,FLAGS_DIM>::ExtentList FlagsExtentList;

      SampleData(const ExtentList &extents, const FlagsExtentList &flagsExtents, Allocator & = heapAllocator);

      MultiDimArray<T,DIM>              samples;
      MultiDimArray<SparseSet<unsigned>,FLAGS_DIM>   flags;

    protected:
      virtual void readData(Stream *, unsigned);
      virtual void writeData(Stream *, unsigned);

    private:
      //bool	 itsHaveWarnedLittleEndian;
    };


    inline void StreamableData::read(Stream *str, bool withSequenceNumber, unsigned alignment)
    {
      if (withSequenceNumber) {
        std::vector<char> header(alignment > 2 * sizeof(uint32_t) ? alignment : 2 * sizeof(uint32_t));
        uint32_t          &magicValue = *reinterpret_cast<uint32_t *>(&header[0]);
        uint32_t          &seqNo = *reinterpret_cast<uint32_t *>(&header[sizeof(uint32_t)]);

        str->read(&header[0], header.size());

        peerMagicNumber = magicValue;
        rawSequenceNumber = seqNo;
      }

      readData(str, alignment);
    }


    inline void StreamableData::write(Stream *str, bool withSequenceNumber, unsigned alignment)
    {

      if (withSequenceNumber) {
        /*     std::vector<char> header(alignment > sizeof(uint32_t) ? alignment : sizeof(uint32_t)); */
        std::vector<char> header(alignment > 2 * sizeof(uint32_t) ? alignment : 2 * sizeof(uint32_t));
        uint32_t          &magicValue = *reinterpret_cast<uint32_t *>(&header[0]);
        uint32_t          &seqNo = *reinterpret_cast<uint32_t *>(&header[sizeof(uint32_t)]);

#if defined USE_VALGRIND
        std::memset(&header[0], 0, header.size());
#endif

        magicValue = peerMagicNumber;
        seqNo = rawSequenceNumber;

        str->write(&header[0], header.size());
      }

      writeData(str, alignment);
    }


    template <typename T, unsigned DIM, unsigned FLAGS_DIM>
    inline SampleData<T,DIM,FLAGS_DIM>::SampleData(const ExtentList &extents, const FlagsExtentList &flagsExtents, Allocator &allocator)
      :
      samples(extents, alignment, allocator),
      flags(flagsExtents) // e.g., for FilteredData [nrChannels][nrStations], sparse dimension [nrSamplesPerIntegration]

      //itsHaveWarnedLittleEndian(false)
    {
    }


    template <typename T, unsigned DIM, unsigned FLAGS_DIM>
    inline void SampleData<T,DIM,FLAGS_DIM>::readData(Stream *str, unsigned alignment)
    {
      (void)alignment;

      str->read(samples.origin(), samples.num_elements() * sizeof(T));
    }


    template <typename T, unsigned DIM, unsigned FLAGS_DIM>
    inline void SampleData<T,DIM,FLAGS_DIM>::writeData(Stream *str, unsigned alignment)
    {
      (void)alignment;

      str->write(samples.origin(), samples.num_elements() * sizeof(T));
    }

  } // namespace Cobalt
} // namespace LOFAR

#endif

