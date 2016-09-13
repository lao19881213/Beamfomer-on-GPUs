//# PerformanceCounter.cc
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
//# $Id: PerformanceCounter.cc 26758 2013-09-30 11:47:21Z loose $

#include <lofar_config.h>

#include "PerformanceCounter.h"

namespace LOFAR
{
  namespace Cobalt
  {
    PerformanceCounter::PerformanceCounter(const gpu::Context &context)
      :
    start(context),
    stop(context)
    {}


    void PerformanceCounter::logTime()
    {
      // get the difference between start and stop. push it on the stats object
      stats.push(stop.elapsedTime(start));
    }
    
  }
}

