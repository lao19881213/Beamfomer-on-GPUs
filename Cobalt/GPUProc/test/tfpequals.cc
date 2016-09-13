//# tfpequals.h
//# Copyright (C) 2013  ASTRON (Netherlands Institute for Radio Astronomy)
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
//# $Id: tfpequals.cc 25957 2013-08-06 17:49:54Z amesfoort $

#include <lofar_config.h>

#include <iomanip>

#include <Common/LofarLogger.h>

#include "fpequals.h"

using namespace LOFAR::Cobalt;

template <typename T>
void testFpEquals(T nan, T posInf, T negInf, T diff, T large, T largeDiff, T eps)
{
  T v0 = 0.0;
  T v1 = 1.0;
  T v2 = -1.0;
  ASSERT(fpEquals(v0, v0, eps));
  ASSERT(fpEquals(v1, v1, eps));
  ASSERT(fpEquals(v2, v2, eps));
  ASSERT(!fpEquals(v0, v1, eps));
  ASSERT(!fpEquals(v2, v0, eps));
  ASSERT(!fpEquals(v1, v2, eps));
  ASSERT(!fpEquals(v2, v1, eps));

  ASSERT(fpEquals(large, large, eps));
  ASSERT(fpEquals(-large, -large, eps));
  ASSERT(!fpEquals(-large, large, eps));

  ASSERT(fpEquals(v0, v0+diff, eps));
  ASSERT(fpEquals(v0, v0-diff, eps));
  ASSERT(fpEquals(v0-diff, v0+diff, eps));
  ASSERT(fpEquals(v1+diff, v1-diff, eps));
  ASSERT(fpEquals(v2-diff, v2+diff, eps));

  ASSERT(fpEquals(large-largeDiff, large, eps));
  ASSERT(fpEquals(-large+largeDiff, -large-largeDiff, eps));

  // NaN, Inf
  // A NaN is inequal to any value, even NaN.
  ASSERT(!fpEquals(nan, nan, eps));
  ASSERT(fpEquals(posInf, posInf, eps));
  ASSERT(fpEquals(negInf, negInf, eps));
  ASSERT(!fpEquals(posInf, negInf, eps));
  ASSERT(!fpEquals(negInf, posInf, eps));
  ASSERT(!fpEquals(posInf, nan, eps));
  ASSERT(!fpEquals(nan, negInf, eps));

  // complex<T>
  T r = 1.0;
  T i = 2.0;
  std::complex<T> c0(r, i);
  ASSERT(fpEquals(c0, c0, eps));

  std::complex<T> c1(r, r);
  std::complex<T> c2(i, i);
  ASSERT(!fpEquals(c0, c1, eps));
  ASSERT(!fpEquals(c0, c2, eps));
  ASSERT(!fpEquals(c1, c2, eps));
}

// A suitable epsilon is calculation dependent, but try to use a reasonable one-size-fits-all (yeah, right...)
// Do some basic tests, also to check behavior wrt NaN and Inf. (One should use binary/hex, not decimal notation...)
int main()
{
  INIT_LOGGER("tfpequals");

  // On Linux x86-64 (gcc-4.6.1), epsilon() returns 1.192092896e-07f and 2.22044604925031308e-16
  // Run gcc -dM -E empty.h | less for the list of predefines.
  LOG_INFO_STR("numeric_limits<float>::epsilon()="  << std::setprecision(9+1)  << std::numeric_limits<float>::epsilon());
  LOG_INFO_STR("numeric_limits<double>::epsilon()=" << std::setprecision(17+1) << std::numeric_limits<double>::epsilon());

  float epsf = 4.0f * std::numeric_limits<float>::epsilon();
  float difff = epsf / 2.0f;
  float largef = 1000000.0f;
  float largeDifff = 1e-1f;
  float nanf =     0.0f/0.0f;
  float posInff =  1.0f/0.0f;
  float negInff = -1.0f/0.0f;
  LOG_INFO("testFpEquals<float>()");
  testFpEquals(nanf, posInff, negInff, difff, largef, largeDifff, epsf);

  double eps = 4.0 * std::numeric_limits<double>::epsilon();
  double diff = eps / 2.0;
  double large = 100000000000.0;
  double largeDiff = 1e-5f;
  double nan =     0.0/0.0;
  double posInf =  1.0/0.0;
  double negInf = -1.0/0.0;
  LOG_INFO("testFpEquals<double>()");
  testFpEquals(nan, posInf, negInf, diff, large, largeDiff, eps);

  LOG_INFO("all cool");
  return 0;
}

