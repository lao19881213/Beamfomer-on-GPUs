//# UnitTest.cc
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
//# $Id: UnitTest.cc 25358 2013-06-18 09:05:40Z loose $

#include <lofar_config.h>

#include "UnitTest.h"

#include <limits>
#include <complex>
#include <cmath>

#include <GPUProc/global_defines.h>
#include <GPUProc/opencl/gpu_utils.h>

namespace LOFAR
{
  namespace Cobalt
  {

    UnitTest::UnitTest(const Parset &ps, const char *programName)
      :
      counter(programName != 0 ? programName : "test", profiling)
    {
      createContext(context, devices);
      queue = cl::CommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

      if (programName != 0)
        program = createProgram(ps, context, devices, programName);
    }

  }
}

