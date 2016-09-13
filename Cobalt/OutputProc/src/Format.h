//# Format.h: Virtual baseclass
//# Copyright (C) 2009-2013  ASTRON (Netherlands Institute for Radio Astronomy)
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
//# $Id: Format.h 24388 2013-03-26 11:14:29Z amesfoort $

#ifndef LOFAR_STORAGE_FORMAT_H
#define LOFAR_STORAGE_FORMAT_H

#include <string>

namespace LOFAR
{
  namespace Cobalt
  {

    class Format
    {
    public:
      virtual ~Format();

      virtual void addSubband(const std::string MSname, unsigned subband, bool isBigEndian) = 0;
    };

  } // namespace Cobalt
} // namespace LOFAR

#endif

