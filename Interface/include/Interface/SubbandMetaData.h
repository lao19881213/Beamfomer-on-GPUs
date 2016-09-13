//# SubbandMetaData.h:
//#
//#  Copyright (C) 2007
//#  ASTRON (Netherlands Foundation for Research in Astronomy)
//#  P.O.Box 2, 7990 AA Dwingeloo, The Netherlands, seg@astron.nl
//#
//#  This program is free software; you can redistribute it and/or modify
//#  it under the terms of the GNU General Public License as published by
//#  the Free Software Foundation; either version 2 of the License, or
//#  (at your option) any later version.
//#
//#  This program is distributed in the hope that it will be useful,
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//#  GNU General Public License for more details.
//#
//#  You should have received a copy of the GNU General Public License
//#  along with this program; if not, write to the Free Software
//#  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//#
//#  $Id: SubbandMetaData.h 19195 2011-11-08 10:41:16Z mol $

#ifndef LOFAR_INTERFACE_SUBBAND_META_DATA_H
#define LOFAR_INTERFACE_SUBBAND_META_DATA_H

#include <Interface/Align.h>
#include <Interface/Allocator.h>
#include <Interface/MultiDimArray.h>
#include <Interface/SparseSet.h>
#include <Common/LofarLogger.h>
#include <Stream/Stream.h>

#include <cassert>
#include <cstring>


namespace LOFAR {
namespace RTCP {

// Note: struct must remain copyable to avoid ugly constructions when passing it around
struct SubbandMetaData
{
  public:
    SubbandMetaData(unsigned nrSubbands, unsigned nrBeams, Allocator &allocator = heapAllocator);
    ~SubbandMetaData();

    struct beamInfo {
      float  delayAtBegin, delayAfterEnd;
      double beamDirectionAtBegin[3], beamDirectionAfterEnd[3];
    };

    struct marshalledData {
      unsigned char	flagsBuffer[132];
      unsigned		alignmentShift;

      // nrBeams elements will really be allocated, so this array needs to
      // be the last element. Also, ISO C++ forbids zero-sized arrays, so we use size 1.
      struct beamInfo   beams[1];
    };

    SparseSet<unsigned>	getFlags(unsigned subband) const;
    void		setFlags(unsigned subband, const SparseSet<unsigned> &);

    unsigned            alignmentShift(unsigned subband) const;
    unsigned            &alignmentShift(unsigned subband);

    struct beamInfo     *beams(unsigned subband) const;
    struct beamInfo     *beams(unsigned subband);

    struct marshalledData &subbandInfo(unsigned subband) const;
    struct marshalledData &subbandInfo(unsigned subband);
 
    void read(Stream *str);
    void write(Stream *str) const;

    // size of the information for one subband
    const unsigned      itsSubbandInfoSize;

 private:
    // size of the information for all subbands
    const unsigned      itsMarshalledDataSize;

    // the pointer to all our data, which consists of struct marshalledData[itsNrSubbands],
    // except for the fact that the elements are spaces apart more than sizeof(struct marshalledData)
    // to make room for extra beams which are not defined in the marshalledData structure.
    //
    // Access elements through subbandInfo(subband).
    char		*const itsMarshalledData;

    Allocator           &itsAllocator;
};


inline SubbandMetaData::SubbandMetaData(unsigned nrSubbands, unsigned nrBeams, Allocator &allocator)
: 
  // Size of the data we need to allocate. Note that marshalledData already contains
  // the size of one beamInfo.
  itsSubbandInfoSize(sizeof(struct marshalledData) + (nrBeams - 1) * sizeof(struct beamInfo)),
  itsMarshalledDataSize(align(nrSubbands * itsSubbandInfoSize, 16)),
  itsMarshalledData(static_cast<char*>(allocator.allocate(itsMarshalledDataSize, 16))),
  itsAllocator(allocator)
{
#if defined USE_VALGRIND
  memset(itsMarshalledData, 0, itsMarshalledDataSize);
#endif
}

inline SubbandMetaData::~SubbandMetaData()
{
  itsAllocator.deallocate(itsMarshalledData);
}

inline SparseSet<unsigned> SubbandMetaData::getFlags(unsigned subband) const
{
  SparseSet<unsigned> flags;

  flags.unmarshall(subbandInfo(subband).flagsBuffer);
  return flags;
}

inline void SubbandMetaData::setFlags(unsigned subband, const SparseSet<unsigned> &flags)
{
  ssize_t size = flags.marshall(&subbandInfo(subband).flagsBuffer, sizeof subbandInfo(subband).flagsBuffer);
  
  assert(size >= 0);
}

inline unsigned SubbandMetaData::alignmentShift(unsigned subband) const
{
  return subbandInfo(subband).alignmentShift;
}

inline unsigned &SubbandMetaData::alignmentShift(unsigned subband)
{
  return subbandInfo(subband).alignmentShift;
}

inline struct SubbandMetaData::beamInfo *SubbandMetaData::beams(unsigned subband) const
{
  return &subbandInfo(subband).beams[0];
}

inline struct SubbandMetaData::beamInfo *SubbandMetaData::beams(unsigned subband)
{
  return &subbandInfo(subband).beams[0];
}

inline struct SubbandMetaData::marshalledData &SubbandMetaData::subbandInfo(unsigned subband) const
{
  // calculate the array stride ourself, since C++ does not know the proper size of the marshalledData elements
  return *reinterpret_cast<struct marshalledData*>(itsMarshalledData + (subband * itsSubbandInfoSize));
}

inline struct SubbandMetaData::marshalledData &SubbandMetaData::subbandInfo(unsigned subband)
{
  // calculate the array stride ourself, since C++ does not know the proper size of the marshalledData elements
  return *reinterpret_cast<struct marshalledData*>(itsMarshalledData + (subband * itsSubbandInfoSize));
}

inline void SubbandMetaData::read(Stream *str)
{
  // TODO: endianness

  str->read(itsMarshalledData, itsMarshalledDataSize);
}

inline void SubbandMetaData::write(Stream *str) const
{
  // TODO: endianness

  str->write(itsMarshalledData, itsMarshalledDataSize);
}

} // namespace RTCP
} // namespace LOFAR

#endif 
