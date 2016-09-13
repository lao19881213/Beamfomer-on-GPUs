//# PacketsToBuffer.cc
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
//# $Id: tPacketsToBuffer.cc 25540 2013-07-02 13:20:36Z mol $

#include <lofar_config.h>

#include <ctime>
#include <string>

#include <Common/LofarTypes.h>
#include <Common/LofarLogger.h>
#include <Stream/FileStream.h>

#include <InputProc/SampleType.h>
#include <InputProc/Station/PacketsToBuffer.h>
#include <InputProc/Buffer/SampleBuffer.h>

using namespace LOFAR;
using namespace Cobalt;
using namespace std;

unsigned clockMHz = 200;

template<typename T>
void test( struct BufferSettings &settings, const std::string &filename )
{
  // Create the buffer to keep it around after transfer.process(), or there
  // will be no subscribers and transfer will delete the buffer automatically,
  // at which point we can't attach anymore.
  SampleBuffer< SampleType<T> > buffer(settings, SharedMemoryArena::CREATE);

  // Read packets from file
  FileStream fs(filename);

  // Set up transfer
  PacketsToBuffer transfer(fs, settings, 0);

  // Do transfer
  transfer.process();

  // Check if the board ended up in the right mode
  const struct BoardMode mode(SampleType<T>::bitMode(), clockMHz);
  ASSERT(mode == *buffer.boards[0].mode);

  // There should be 32 samples in the buffer (16 per packet, 2 packets per
  // file).
  BufferSettings::range_type now = TimeStamp(time(0) + 1, 0, mode.clockHz());
  BufferSettings::flags_type available = buffer.boards[0].available.sparseSet(0, now);
  ASSERT(available.count() == 32);
}


int main()
{
  INIT_LOGGER( "tPacketsToBuffer" );

  // Don't run forever if communication fails for some reason
  alarm(10);

  // Fill a BufferSettings object
  struct StationID stationID("RS106", "LBA");
  struct BufferSettings settings(stationID, false);

  // Use a fixed key, so the test suite knows what to clean
  settings.dataKey = 0x10000002;
  removeSampleBuffers(settings);

  // Limit the array in size to work on systems with only 32MB SHM
  settings.nrBoards = 1;
  settings.setBufferSize(0.1);

  // Test various modes
  LOG_INFO("Test 16-bit complex");
  test<i16complex>(settings, "tPacketsToBuffer.in_16bit");

  LOG_INFO("Test 8-bit complex");
  test<i8complex>(settings, "tPacketsToBuffer.in_8bit");

  return 0;
}

