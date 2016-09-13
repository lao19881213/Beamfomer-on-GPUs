//# fpequals.h: templated floating point comparison routines with epsilon
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
//# $Id: fpequals.h 25957 2013-08-06 17:49:54Z amesfoort $

#ifndef LOFAR_GPUPROC_FPEQUALS_H
#define LOFAR_GPUPROC_FPEQUALS_H

#include <cmath>
#include <complex>
#include <limits>

namespace LOFAR
{
  namespace Cobalt
  {

    // Inexact floating point comparison routines.
    // The 'tfpequals' test covers these routines
    // and has some more hints on reasonable epsilon values.


    // T can be float, double, long double, or complex of those (overloads below).
    template <typename T>
    bool fpEquals(T x, T y, T eps = std::numeric_limits<T>::epsilon())
    {
      // Despite comparisons below, x==y is still needed to correctly eval the
      // equality of identical inf args: 1.0/0.0==1.0/0.0 and -1.0/0.0==-1.0/0.0
      if (x == y)
      {
        return true;
      }

      // absolute
      if (std::abs(x - y) <= eps)
      {
        return true;
      }

      // relative
      if (std::abs(y) < std::abs(x)) {
        return std::abs((x - y) / x) <= eps;
      } else {
        return std::abs((x - y) / y) <= eps;
      }
    }

    template <typename T>
    bool fpEquals(std::complex<T> x, std::complex<T> y,
                  T eps = std::numeric_limits<T>::epsilon())
    {
      return fpEquals(x.real(), y.real(), eps) &&
             fpEquals(x.imag(), y.imag(), eps);
    }

    template <typename T>
    bool fpEquals(std::complex<T> x, T y, T eps = std::numeric_limits<T>::epsilon())
    {
      return fpEquals(x.real(), y, eps) &&
             fpEquals(x.imag(), (T)0.0, eps);
    }


  }
}

#endif

