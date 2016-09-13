//# SampleType.h
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
//# $Id: SampleType.h 25540 2013-07-02 13:20:36Z mol $

#ifndef LOFAR_INPUT_PROC_SAMPLE_TYPE_H 
#define LOFAR_INPUT_PROC_SAMPLE_TYPE_H

#include <ostream>

#include <Common/LofarConstants.h>
#include <Common/lofar_complex.h>

namespace LOFAR
{
  namespace Cobalt
  {


    template<typename T>
    struct SampleType {
      T x, y;

      bool operator == (const struct SampleType &other) const {
        return x == other.x && y == other.y;
      }

      bool operator != (const struct SampleType &other) const {
        return !(*this == other);
      }

      static unsigned bitMode() {
        return 8 * sizeof(T) / 2;
      }
    };

    template struct SampleType<i16complex>;
    template struct SampleType<i8complex>;
    template struct SampleType<i4complex>;

    template<typename T>
    std::ostream &operator <<(std::ostream &str, const struct SampleType<T> &sample)
    {
      // Cast to int to prevent signed chars from being printed as characters
      // instead of numbers.
      str << "(" << (int)(sample.x.real()) << " + " << (int)(sample.x.imag()) << "i, " << (int)(sample.y.real()) << " + " << (int)(sample.y.imag()) << "i)";

      return str;
    }


  }
}

#endif

