//# tPerformanceCounter.cc
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
//# $Id: tPerformanceCounter.cc 25358 2013-06-18 09:05:40Z loose $

#include <lofar_config.h>

#include <vector>
#include <Common/LofarLogger.h>

#include <GPUProc/MultiDimArrayHostBuffer.h>
#include <GPUProc/opencl/gpu_utils.h>
#include <GPUProc/PerformanceCounter.h>

using namespace LOFAR;
using namespace Cobalt;
using namespace std;

cl::Context context;
vector<cl::Device> devices;

// test a performance counter without events
void test_simple()
{
  PerformanceCounter counter("test", true);
}

// test a single event
void test_event()
{
  PerformanceCounter counter("test", true);

  // create a buffer and a queue to send the buffer
  cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

  MultiArraySharedBuffer<float, 1> buffer(boost::extents[1024], queue, CL_MEM_WRITE_ONLY, CL_MEM_READ_ONLY);

  // transfer the buffer and record the performance
  buffer.hostToDevice(CL_TRUE);
  counter.doOperation(buffer.event, 0, 0, buffer.bytesize());

  // wait for all scheduled events to pass
  counter.waitForAllOperations();

  struct PerformanceCounter::figures total = counter.getTotal();

  // validate results
  ASSERT(total.nrEvents == 1);

  ASSERT(total.nrBytesRead == 0);
  ASSERT(total.nrBytesWritten == buffer.bytesize());

  ASSERT(total.runtime > 0.0);
}

int main()
{
  INIT_LOGGER( "tPerformanceCounter" );

  createContext(context, devices);

  test_simple();
  test_event();

  return 0;
}

