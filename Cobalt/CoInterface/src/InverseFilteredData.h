//# InverseFilteredData.h
//# Copyright (C) 2010-2013  ASTRON (Netherlands Institute for Radio Astronomy)
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
//# $Id: InverseFilteredData.h 24388 2013-03-26 11:14:29Z amesfoort $

#ifndef LOFAR_INTERFACE_INVERSE_FILTERED_DATA_H
#define LOFAR_INTERFACE_INVERSE_FILTERED_DATA_H

#include <CoInterface/MultiDimArray.h>
#include <CoInterface/StreamableData.h>

namespace LOFAR
{
  namespace Cobalt
  {

    // This assumes the nrChannels == 1
    // We store the data for only 1 beam, 1 polarization.
    // nrSamplesPerIntegration is the original nrSamplesPerIntegration, and now becomes the "major time" index.
    // The stationFilterSize is the minor time index.

    class InverseFilteredData : public SampleData<float,1,1>
    {
    public:
      typedef SampleData<float,1,1> SuperType;

      InverseFilteredData(unsigned nrSamplesPerIntegration, unsigned stationFilterSize);

    protected:
      const unsigned itsNrSamplesPerIntegration;
      const unsigned itsStationFilterSize;

    };

    inline InverseFilteredData::InverseFilteredData(unsigned nrSamplesPerIntegration, unsigned stationFilterSize)
      :
      SuperType::SampleData(boost::extents[nrSamplesPerIntegration * stationFilterSize], boost::extents[1]),
      itsNrSamplesPerIntegration(nrSamplesPerIntegration),
      itsStationFilterSize(stationFilterSize)
    {
    }

  } // namespace Cobalt
} // namespace LOFAR

#endif

