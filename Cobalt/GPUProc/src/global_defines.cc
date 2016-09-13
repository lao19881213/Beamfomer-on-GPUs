//# global_defines.cc
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
//# $Id: global_defines.cc 26436 2013-09-09 16:17:42Z mol $
#include <lofar_config.h>

#include "global_defines.h"

#include <cstdlib>
#include <cstdio>

#if defined __linux__
#include <sched.h>
#endif

namespace LOFAR
{
  namespace Cobalt
  {
    bool profiling = false;
    bool gpuProfiling = true;

    inline void set_affinity(unsigned device)
    {
#ifdef __linux__
#if 0
      static const char mapping[1][12] = {
        0,  1,  2,  3,  8,  9, 10, 11,
      };
#else
      static const char mapping[8][12] = {
        { 0,  1,  2,  3,  4,  5, 12, 13, 14, 15, 16, 17, },
        { 0,  1,  2,  3,  4,  5, 12, 13, 14, 15, 16, 17, },
        { 0,  1,  2,  3,  4,  5, 12, 13, 14, 15, 16, 17, },
        { 0,  1,  2,  3,  4,  5, 12, 13, 14, 15, 16, 17, },
        { 6,  7,  8,  9, 10, 11, 18, 19, 20, 21, 22, 23, },
        { 6,  7,  8,  9, 10, 11, 18, 19, 20, 21, 22, 23, },
        { 6,  7,  8,  9, 10, 11, 18, 19, 20, 21, 22, 23, },
        { 6,  7,  8,  9, 10, 11, 18, 19, 20, 21, 22, 23, },
      };
#endif

      cpu_set_t set;

      CPU_ZERO(&set);

      for (unsigned coreIndex = 0; coreIndex < 12; coreIndex++)
        CPU_SET(mapping[device][coreIndex], &set);

      if (sched_setaffinity(0, sizeof set, &set) < 0)
        perror("sched_setaffinity");
#endif // __linux__
    }

  }
}

