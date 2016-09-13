//# StationID.h
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
//# $Id: StationID.h 25598 2013-07-08 12:31:36Z mol $

#ifndef LOFAR_INPUT_PROC_STATIONID_H
#define LOFAR_INPUT_PROC_STATIONID_H

#include <string>
#include <ostream>

#include <Common/LofarTypes.h>

namespace LOFAR
{
  namespace Cobalt
  {

    struct StationID {
      char stationName[64];
      char antennaField[64];

      StationID( const std::string &stationName = "", const std::string &antennaField = "" );

      // The full ('storage') name of the station, f.e.
      // CS001HBA0.
      std::string name() const;

      bool operator==(const struct StationID &other) const;
      bool operator!=(const struct StationID &other) const;

      uint32 hash() const;
    };

    std::ostream& operator<<( std::ostream &str, const struct StationID &s );

  }
}


#endif

