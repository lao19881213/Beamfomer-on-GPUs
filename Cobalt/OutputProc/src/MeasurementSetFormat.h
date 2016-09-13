//# MeasurementSetFormat.h: defines the format of the RAW datafile
//# Copyright (C) 2009-2013  ASTRON (Netherlands Institute for Radio Astronomy)
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
//# $Id: MeasurementSetFormat.h 25660 2013-07-13 13:11:46Z mol $

#ifndef LOFAR_STORAGE_MEASUREMENT_SET_FORMAT_H
#define LOFAR_STORAGE_MEASUREMENT_SET_FORMAT_H

#include <string>
#include <vector>

#include <Common/LofarTypes.h>
#include <Common/Thread/Mutex.h>
#include <MSLofar/MSLofar.h>
#include <CoInterface/Parset.h>
#include <CoInterface/SmartPtr.h>
#include "Format.h"

#include <casa/aips.h>
#include <casa/Utilities/DataType.h>
#include <casa/Arrays/IPosition.h>

/*
 * LofarStMan supports multiple versions of the MS, with the following
 * differences in the data on disk:
 *
 * MS version    visibilities    antenna order     support for
 *                               (baseline = 1,2)  bytes/weight
 * -------------------------------------------------------------
 *  1            conjugated      1,2               2
 *  2            conjugated      2,1               1,2,4
 *  3            normal          1,2               1,2,4
 */

static const unsigned LofarStManVersion = 3;

//# Forward Declarations
namespace casa
{
  class MPosition;
  template<class T>
  class Block;
}


namespace LOFAR
{
  namespace Cobalt
  {

    class MeasurementSetFormat : public Format
    {
    public:
      MeasurementSetFormat(const Parset &, uint32 alignment = 1);
      virtual ~MeasurementSetFormat();

      virtual void addSubband(const std::string MSname, unsigned subband, bool isBigEndian);

      // casacore/measurementset mutex
      static Mutex sharedMutex;

    private:
      const Parset &itsPS;

      const std::vector<std::string> stationNames;
      const MultiDimArray<double,2>  antPos;

      const unsigned itsNrAnt;
      uint32 itsNrTimes;

      double itsStartTime;
      double itsTimeStep;


      SmartPtr<MSLofar> itsMS;

      const uint32 itsAlignment;

      void createMSTables(const std::string &MSname, unsigned subband);
      void createMSMetaFile(const std::string &MSname, unsigned subband, bool isBigEndian);

      void fillFeed();
      void fillAntenna(const casa::Block<casa::MPosition>& antMPos);
      void fillField(unsigned subarray);
      void fillPola();
      void fillDataDesc();
      void fillSpecWindow(unsigned subband);
      void fillObs(unsigned subarray);
      void fillHistory();
    };

  } // namespace Cobalt
} // namespace LOFAR

#endif

