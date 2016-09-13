//# RSPBoards.h
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
//# $Id: RSPBoards.h 24388 2013-03-26 11:14:29Z amesfoort $

#ifndef LOFAR_INPUT_PROC_RSP_BOARDS_H 
#define LOFAR_INPUT_PROC_RSP_BOARDS_H

#include <string>
#include <vector>

#include "Buffer/BufferSettings.h"
#include "WallClockTime.h"

namespace LOFAR
{
  namespace Cobalt
  {

    /* A class that generates or processes a set of data streams of a station. */

    class RSPBoards
    {
    public:
      RSPBoards( const std::string &logPrefix, size_t nrBoards );

      void process();

      void stop();

    protected:
      const std::string logPrefix;
      const size_t nrBoards;

      WallClockTime waiter;

      virtual void processBoard( size_t nr ) = 0;
      virtual void logStatistics() = 0;
    };


  }
}

#endif

