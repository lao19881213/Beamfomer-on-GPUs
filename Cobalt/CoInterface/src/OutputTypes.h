//# OutputTypes.h
//# Copyright (C) 2011-2013  ASTRON (Netherlands Institute for Radio Astronomy)
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
//# $Id: OutputTypes.h 24388 2013-03-26 11:14:29Z amesfoort $

#ifndef LOFAR_RTCP_INTERFACE_OUTPUT_TYPES_H
#define LOFAR_RTCP_INTERFACE_OUTPUT_TYPES_H

namespace LOFAR
{
  namespace Cobalt
  {

    enum OutputType
    {
      CORRELATED_DATA = 1,
      BEAM_FORMED_DATA,
      TRIGGER_DATA,

      // define LAST and FIRST in the enum to make them valid values within the
      // allocated range for the enum (=minimal number of bits to store all values)
      LAST_OUTPUT_TYPE,
      FIRST_OUTPUT_TYPE = 1
    };

    inline OutputType operator ++ (OutputType &outputType) // prefix ++
    {
      return (outputType = static_cast<OutputType>(outputType + 1));
    }


    inline OutputType operator ++ (OutputType &outputType, int) // postfix ++
    {
      return static_cast<OutputType>((outputType = static_cast<OutputType>(outputType + 1)) - 1);
    }

  } // namespace Cobalt
} // namespace LOFAR

#endif

