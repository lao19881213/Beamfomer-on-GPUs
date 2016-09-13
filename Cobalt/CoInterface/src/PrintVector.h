//# PrintVector.h
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
//# $Id: PrintVector.h 24385 2013-03-26 10:43:55Z amesfoort $

#ifndef LOFAR_INTERFACE_PRINT_VECTOR_H
#define LOFAR_INTERFACE_PRINT_VECTOR_H

#include <vector>
#include <iostream>

namespace LOFAR
{
  namespace Cobalt
  {

    template<typename T>
    inline std::ostream &operator << (std::ostream &str, const std::vector<T> &v)
    {
      str << '[';

      for (typename std::vector<T>::const_iterator it = v.begin(); it != v.end(); it++) {
        if (it != v.begin())
          str << ',';

        str << *it;
      }

      return str << ']';
    }


    template<typename T, typename U>
    inline std::ostream &operator << (std::ostream &str, const std::pair<T,U> &p)
    {
      return str << '(' << p.first << ',' << p.second << ')';
    }


  } // namespace Cobalt
} // namespace LOFAR

#endif

