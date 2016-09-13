//# MSMriter.cc: Base classs for MS writer
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
//# $Id: MSWriter.cc 24385 2013-03-26 10:43:55Z amesfoort $

#include <lofar_config.h>

#include "MSWriter.h"

#include <algorithm>


namespace LOFAR
{
  namespace Cobalt
  {

    MSWriter::MSWriter()
      :
      itsNrBlocksWritten(0),
      itsNrExpectedBlocks(0)
    {
    }


    MSWriter::~MSWriter()
    {
    }

    void MSWriter::augment(const FinalMetaData &finalMetaData)
    {
      (void)finalMetaData;
    }


    size_t MSWriter::getDataSize()
    {
      return 0;
    }

    ParameterSet MSWriter::configuration() const
    {
      return itsConfiguration;
    }


    /* Returns a percentage based on a current and a target value,
     * with the following rounding:
     *
     * 0     -> current == 0
     * 1..99 -> 0 < current < target
     * 100   -> current == target
     */

    unsigned MSWriter::percentageWritten() const
    {
      size_t current = itsNrBlocksWritten;
      size_t target = itsNrExpectedBlocks;

      if (current == target || target == 0)
        return 100;

      if (current == 0)
        return 0;

      return std::min(std::max(100 * current / target, static_cast<size_t>(1)), static_cast<size_t>(99));
    }


  }
}

