//# BlockID.cc
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
//# $Id: BlockID.cc 26758 2013-09-30 11:47:21Z loose $

#include <lofar_config.h>

#include "BlockID.h"

#include <iostream>

namespace LOFAR
{
  namespace Cobalt
  {
    BlockID::BlockID() :
      block(0),
      globalSubbandIdx(0),
      localSubbandIdx(0),
      subbandProcSubbandIdx(0)
    {
    }

    bool BlockID::operator<(const struct BlockID &other) const {
      if (block != other.block)
        return block < other.block;
      return globalSubbandIdx < other.globalSubbandIdx;
    }

    std::ostream &operator<<(std::ostream &str, const struct BlockID &id)
    {
      str << "block " << id.block << " subband " << id.globalSubbandIdx 
          << " (local index " << id.localSubbandIdx << ")";
      return str;
    }

  }
}

