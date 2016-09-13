//#  PrintVector.h
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

#ifndef LOFAR_INTERFACE_PRINT_VECTOR_H
#define LOFAR_INTERFACE_PRINT_VECTOR_H

#include <iostream>
#include <vector>

namespace LOFAR {
namespace RTCP {

template<typename T> inline std::ostream &operator << (std::ostream &str, const std::vector<T> &v)
{
  str << '[';

  for (typename std::vector<T>::const_iterator it = v.begin(); it != v.end(); it ++) {
    if (it != v.begin())
      str << ',';

    str << *it;
  }
  
  return str << ']';
}


template<typename T, typename U> inline std::ostream &operator << (std::ostream &str, const std::pair<T,U> &p)
{
  return str << '(' << p.first << ',' << p.second << ')';
}


} // namespace RTCP
} // namespace LOFAR

#endif
