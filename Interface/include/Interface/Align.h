//#  Align.h
//#
//#  Copyright (C) 2006
//#  ASTRON (Netherlands Foundation for Research in Astronomy)
//#  P.O.Box 2, 7990 AA Dwingeloo, The Netherlands, seg@astron.nl
//#
//#  This program is free software; you can redistribute it and/or modify
//#  it under the terms of the GNU General Public License as published by
//#  the Free Software Foundation; either version 2 of the License, or
//#  (at your option) any later version.
//#
//#  This program is distributed in the hope that it will be useful,
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//#  GNU General Public License for more details.
//#
//#  You should have received a copy of the GNU General Public License
//#  along with this program; if not, write to the Free Software
//#  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//#
//#  $Id: Align.h 24020 2013-03-03 12:00:07Z mol $

#ifndef LOFAR_INTERFACE_ALIGN_H
#define LOFAR_INTERFACE_ALIGN_H

#include <cstddef>

namespace LOFAR {
namespace RTCP {


/*
 * Returns true iff n is a power of two.
 */
template <typename T> inline static bool powerOfTwo(T n)
{
  return (n | (n - 1)) == 2 * n - 1;
}


/*
 * Returns the first power of two higher than n.
 */
template <typename T> inline static T nextPowerOfTwo(T n)
{
  T p;

  for (p = 1; p < n; p <<= 1)
    ;

  return p;
}


/*
 * Returns `value' rounded up to `alignment'.
 */
template <typename T> inline static T align(T value, size_t alignment)
{
#if defined __GNUC__
  if (__builtin_constant_p(alignment) && powerOfTwo(alignment))
    return (value + alignment - 1) & ~(alignment - 1);
  else
#endif
    return (value + alignment - 1) / alignment * alignment;
}


/*
 * Returns `value' rounded up to `alignment', in bytes.
 */
template <typename T> inline static T *align(T *value, size_t alignment)
{
  return reinterpret_cast<T *>(align(reinterpret_cast<size_t>(value), alignment));
}


/*
 * Returns true if `value' is aligned to `alignment'.
 */
template <typename T> inline static bool aligned(T value, size_t alignment)
{
  return value % alignment == 0;
}


/*
 * Returns true if `value' is aligned to `alignment', in bytes.
 */
template <typename T> inline static bool aligned(T *value, size_t alignment)
{
  return reinterpret_cast<size_t>(value) % alignment == 0;
}


} // namespace RTCP
} // namespace LOFAR

#endif

