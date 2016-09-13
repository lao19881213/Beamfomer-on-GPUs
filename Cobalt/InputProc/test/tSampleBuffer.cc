//# tSampleBuffer.cc
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
//# $Id: tSampleBuffer.cc 25540 2013-07-02 13:20:36Z mol $

#include <lofar_config.h>

#include <unistd.h>

#include <Common/LofarTypes.h>
#include <Common/LofarLogger.h>

#include <InputProc/SampleType.h>
#include <InputProc/Buffer/StationID.h>
#include <InputProc/Buffer/SampleBuffer.h>
#include <InputProc/Buffer/BufferSettings.h>

using namespace LOFAR;
using namespace Cobalt;
using namespace std;

template<typename T>
void test( struct BufferSettings &settings, unsigned bitMode )
{
  LOG_INFO_STR("Test " << bitMode << "-bit complex");

  // Should be able to create the buffer
  SampleBuffer< SampleType<T> > buffer_create(settings, SharedMemoryArena::CREATE);

  // Should be able to attach to created buffer
  SampleBuffer< SampleType<T> > buffer_read(settings, SharedMemoryArena::READ);
  SampleBuffer< SampleType<T> > buffer_readwrite(settings, SharedMemoryArena::READWRITE);
}

int main()
{
  INIT_LOGGER( "tSampleBuffer" );

  // Don't run forever if communication fails for some reason
  alarm(10);

  // Fill a BufferSettings object
  struct StationID stationID("RS106", "LBA");
  struct BufferSettings settings(stationID, false);

  // Use a fixed key, so the test suite knows what to clean
  settings.dataKey = 0x10000004;
  removeSampleBuffers(settings);

  // Limit the array in size to work on systems with only 32MB SHM
  settings.nrBoards = 1;
  settings.setBufferSize(0.1);

  // Test various modes
  test<i16complex>(settings, 16);
  test<i8complex>(settings, 8);
  test<i4complex>(settings, 4);

  return 0;
}

