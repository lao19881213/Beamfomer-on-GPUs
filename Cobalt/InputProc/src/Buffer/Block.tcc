//# Block.tcc
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
//# $Id: Block.tcc 25318 2013-06-13 11:48:50Z mol $

#include "Block.h"

namespace LOFAR
{
  namespace Cobalt
  {
    template<typename T> size_t Block<T>::Beamlet::Range::size() const
    {
      return to - from;
    }

    template<typename T> void Block<T>::Beamlet::copy( T *dest ) const
    {
      // copy first range
      memcpy(dest, ranges[0].from, ranges[0].size() * sizeof(T));

      if (nrRanges > 1) {
        // append second range
        memcpy(dest + ranges[0].size(), ranges[1].from, ranges[1].size() * sizeof(T));
      }
    }
  }
}

