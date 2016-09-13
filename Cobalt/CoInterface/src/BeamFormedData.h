//# BeamFormedData.h
//# Copyright (C) 2012-2013  ASTRON (Netherlands Institute for Radio Astronomy)
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
//# $Id: BeamFormedData.h 24388 2013-03-26 11:14:29Z amesfoort $

#ifndef LOFAR_INTERFACE_BEAMFORMED_DATA_H
#define LOFAR_INTERFACE_BEAMFORMED_DATA_H

#include <Common/lofar_complex.h>
#include <Stream/Stream.h>
#include <CoInterface/Align.h>
#include <CoInterface/Config.h>
#include <CoInterface/MultiDimArray.h>
#include <CoInterface/SparseSet.h>
#include <CoInterface/StreamableData.h>

namespace LOFAR
{
  namespace Cobalt
  {

    /*
     * Data flow:
     *
     * BeamFormedData -> PreTransposeBeamFormedData -> TransposedBeamFormedData -> FinalBeamFormedData
     *
     * The separate steps are necessary since the data is required or produced in different orders
     * by different processes. The transpose wants to split beams and polarizations and puts subbands
       & in the highest dimension in exchange. The final data product however wants time to be the
     * highest dimension.
     *
     */

    class BeamFormedData : public SampleData<fcomplex,4,2>
    {
    public:
      typedef SampleData<fcomplex,4,2> SuperType;

      BeamFormedData(unsigned nrBeams, unsigned nrChannels, unsigned nrSamples, Allocator &allocator = heapAllocator);
    };


    class PreTransposeBeamFormedData : public SampleData<float,3,1>
    {
    public:
      typedef SampleData<float,3,1> SuperType;

      PreTransposeBeamFormedData(unsigned nrStokes, unsigned nrChannels, unsigned nrSamples, Allocator &allocator = heapAllocator);
    };


    class TransposedBeamFormedData : public SampleData<float,3,2>
    {
    public:
      typedef SampleData<float,3,2> SuperType;

      TransposedBeamFormedData(unsigned nrSubbands, unsigned nrChannels, unsigned nrSamples, Allocator &allocator = heapAllocator);

      virtual void setDimensions(unsigned nrSubbands, unsigned nrChannels, unsigned nrSamples);
    };


    class FinalBeamFormedData : public SampleData<float,3,2>
    {
    public:
      typedef SampleData<float,3,2> SuperType;

      FinalBeamFormedData(unsigned nrSamples, unsigned nrSubbands, unsigned nrChannels, Allocator & = heapAllocator);

      virtual void setDimensions(unsigned nrSamples, unsigned nrSubbands, unsigned nrChannels);
    };


    inline BeamFormedData::BeamFormedData(unsigned nrBeams, unsigned nrChannels, unsigned nrSamples, Allocator &allocator)
    // The "| 2" significantly improves transpose speeds for particular
    // numbers of stations due to cache conflict effects.  The extra memory
    // is not used.
      :
      SuperType::SampleData(boost::extents[nrBeams][nrChannels][nrSamples | 2][NR_POLARIZATIONS], boost::extents[nrBeams][nrChannels], allocator)
    {
    }


    inline PreTransposeBeamFormedData::PreTransposeBeamFormedData(unsigned nrStokes, unsigned nrChannels, unsigned nrSamples, Allocator &allocator)
      :
      SuperType::SampleData(boost::extents[nrStokes][nrChannels][nrSamples | 2],  boost::extents[nrChannels], allocator)
    {
    }


    inline TransposedBeamFormedData::TransposedBeamFormedData(unsigned nrSubbands, unsigned nrChannels, unsigned nrSamples, Allocator &allocator)
      :
      SuperType(boost::extents[nrSubbands][nrChannels][nrSamples | 2], boost::extents[nrSubbands][nrChannels], allocator)
    {
    }


    inline void TransposedBeamFormedData::setDimensions(unsigned nrSubbands, unsigned nrChannels, unsigned nrSamples)
    {
      samples.resizeInplace(boost::extents[nrSubbands][nrChannels][nrSamples | 2]);
    }


    inline FinalBeamFormedData::FinalBeamFormedData(unsigned nrSamples, unsigned nrSubbands, unsigned nrChannels, Allocator &allocator)
      :
      SuperType(boost::extents[nrSamples | 2][nrSubbands][nrChannels], boost::extents[nrSubbands][nrChannels], allocator)
    {
    }


    inline void FinalBeamFormedData::setDimensions(unsigned nrSamples, unsigned nrSubbands, unsigned nrChannels)
    {
      samples.resizeInplace(boost::extents[nrSamples | 2][nrSubbands][nrChannels]);
    }

  } // namespace Cobalt
} // namespace LOFAR

#endif

