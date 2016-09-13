//# tContext.cc: test OpenCL context creation
//# Copyright (C) 2013  ASTRON (Netherlands Institute for Radio Astronomy)
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
//# $Id: tContext.cc 25358 2013-06-18 09:05:40Z loose $

#include <lofar_config.h>

#include <vector>
#include <Common/LofarLogger.h>

#include <GPUProc/opencl/gpu_utils.h>

using namespace LOFAR;
using namespace Cobalt;
using namespace std;

// test OpenCL context creation
void test_create()
{
  cl::Context context;
  vector<cl::Device> devices;

  createContext(context, devices);
}

int main()
{
  INIT_LOGGER( "tContext" );

  test_create();

  return 0;
}

