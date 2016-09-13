//#  VectorOps.h
//#
//#  Copyright (C) 2007
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
//#  $Id: PrintVector.h 16765 2010-11-25 13:27:09Z mol $

#ifndef LOFAR_INTERFACE_SET_OPERATIONS_H
#define LOFAR_INTERFACE_SET_OPERATIONS_H

#include <algorithm>

namespace LOFAR {
namespace RTCP {

template <typename T> T operator & (T a, T b)
{
  sort(a.begin(), a.end());
  sort(b.begin(), b.end());

  T c(a.size() + b.size());
  c.resize(set_intersection(a.begin(), a.end(), b.begin(), b.end(), c.begin()) - c.begin());
  return c;
}

template <typename T> T operator | (T a, T b)
{
  sort(a.begin(), a.end());
  sort(b.begin(), b.end());

  T c(a.size() + b.size());
  c.resize(set_union(a.begin(), a.end(), b.begin(), b.end(), c.begin()) - c.begin());
  return c;
}

} // namespace RTCP
} // namespace LOFAR

#endif
