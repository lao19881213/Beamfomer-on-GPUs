//# FinalMetaData.h
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
//# $Id: FinalMetaData.h 24388 2013-03-26 11:14:29Z amesfoort $

#ifndef LOFAR_INTERFACE_FINAL_METADATA_H
#define LOFAR_INTERFACE_FINAL_METADATA_H

#include <cstddef>
#include <string>
#include <vector>
#include <ostream>

#include <Stream/Stream.h>

namespace LOFAR
{
  namespace Cobalt
  {

    class FinalMetaData
    {
    public:
      struct BrokenRCU {
        std::string station; // CS001, etc
        std::string type;  // RCU, LBA, HBA
        size_t seqnr;      // RCU/antenna number
        std::string time;  // date time of break

        BrokenRCU()
        {
        }
        BrokenRCU(const std::string &station, const std::string &type, size_t seqnr, const std::string &time) :
          station(station), type(type), seqnr(seqnr), time(time)
        {
        }

        bool operator==(const BrokenRCU &other) const
        {
          return station == other.station && type == other.type && seqnr == other.seqnr && time == other.time;
        }
      };

      std::vector<BrokenRCU>  brokenRCUsAtBegin, brokenRCUsDuring;

      void write(Stream &s);
      void read(Stream &s);
    };

    std::ostream& operator<<(std::ostream& os, const struct FinalMetaData::BrokenRCU &rcu);

    std::ostream& operator<<(std::ostream& os, const FinalMetaData &finalMetaData);

  } // namespace Cobalt
} // namespace LOFAR

#endif

