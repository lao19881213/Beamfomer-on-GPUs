/* tBlockReader.cc
 * Copyright (C) 2013  ASTRON (Netherlands Institute for Radio Astronomy)
 * P.O. Box 2, 7990 AA Dwingeloo, The Netherlands
 *
 * This file is part of the LOFAR software suite.
 * The LOFAR software suite is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * The LOFAR software suite is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the LOFAR software suite. If not, see <http://www.gnu.org/licenses/>.
 *
 * $Id: tBlockReader.cc 25950 2013-08-06 14:32:24Z mol $
 */

#include <lofar_config.h>

#include <ctime>
#include <string>

#include <Common/LofarTypes.h>
#include <Common/LofarLogger.h>
#include <Stream/FileStream.h>

#include <InputProc/SampleType.h>
#include <InputProc/Station/PacketsToBuffer.h>
#include <InputProc/Buffer/BlockReader.h>
#include <InputProc/Buffer/SampleBuffer.h>

#include <UnitTest++.h>

using namespace LOFAR;
using namespace Cobalt;
using namespace std;

// A BufferSettings object to be used for all tests
struct StationID stationID("RS106", "LBA");
struct BufferSettings settings(stationID, false);
struct BoardMode mode(16, 200);

TEST(Basic) {
  for (size_t nrBeamlets = 1; nrBeamlets < settings.nrBoards * mode.nrBeamletsPerBoard(); nrBeamlets <<= 1) {
    for (size_t blockSize = 1; blockSize < settings.nrSamples(16); blockSize <<= 1) {
      // Create a buffer
      SampleBuffer< SampleType<i16complex> > buffer(settings, SharedMemoryArena::CREATE);
      buffer.boards[0].changeMode(mode);

      // Read the beamlets
      std::vector<size_t> beamlets(nrBeamlets);
      for (size_t b = 0; b < beamlets.size(); ++b) {
        beamlets[b] = nrBeamlets - b;
      }

      BlockReader< SampleType<i16complex> > reader(settings, mode, beamlets);

      // Read a few blocks -- from the distant past to prevent unnecessary
      // waiting.
      const TimeStamp from(0, 0, mode.clockHz());
      const TimeStamp to(from + 10 * blockSize);
      for (TimeStamp current = from; current + blockSize < to; current += blockSize) {
        SmartPtr<struct BlockReader< SampleType<i16complex> >::LockedBlock> block(reader.block(current, current + blockSize, std::vector<ssize_t>(nrBeamlets, 0)));

        // Validate the block
        ASSERT(block->beamlets.size() == beamlets.size());

        for (size_t b = 0; b < beamlets.size(); ++b) {
          struct Block< SampleType<i16complex> >::Beamlet &ib = block->beamlets[b];

          // Beamlets should be provided in the same order
          CHECK_EQUAL(beamlets[b], ib.stationBeamlet);

          switch (ib.nrRanges) {
            case 1:
              CHECK(ib.ranges[0].from < ib.ranges[0].to);

              CHECK_EQUAL(blockSize, ib.ranges[0].size());
              break;

            case 2:
              CHECK(ib.ranges[0].from < ib.ranges[0].to);
              CHECK(ib.ranges[1].from < ib.ranges[1].to);

              CHECK_EQUAL(blockSize, ib.ranges[0].size() + ib.ranges[1].size());
              break;

            default:
              ASSERTSTR(false, "nrRanges must be 1 or 2");
              break;
          }
          
          // No samples should be available
          CHECK_EQUAL((uint64)blockSize, ib.flagsAtBegin.count());
        }
      }
    }
  }
}

template<typename T>
void test( struct BufferSettings &settings, struct BoardMode &mode, const std::string &filename )
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

  // Determine the timestamps of the packets we've just written
  BufferSettings::range_type now = (uint64)TimeStamp(time(0) + 1, 0, mode.clockHz());
  BufferSettings::flags_type available = buffer.boards[0].available.sparseSet(0, now);

  ASSERT(available.getRanges().size() > 0);

  const TimeStamp from(available.getRanges()[0].begin, mode.clockHz());

  // Read some of the beamlets
  std::vector<size_t> beamlets(2);
  for (size_t b = 0; b < beamlets.size(); ++b)
    beamlets[b] = b;

  BlockReader< SampleType<T> > reader(settings, mode, beamlets);

  // Read the block, plus 16 unavailable samples
  SmartPtr<struct BlockReader< SampleType<T> >::LockedBlock> block(reader.block(from, from + available.count() + 16, std::vector<ssize_t>(beamlets.size(),0)));

  // Validate the block
  for (size_t b = 0; b < beamlets.size(); ++b) {
    // We should only detect the 16 unavailable samples
    ASSERT(block->beamlets[b].flagsAtBegin.count() == 16);
  }
}


int main()
{
  INIT_LOGGER( "tBlockReader" );

  // Don't run forever if communication fails for some reason
  alarm(10);

  // Use a fixed key, so the test suite knows what to clean
  settings.dataKey = 0x10000001;
  removeSampleBuffers(settings);

  // Limit the array in size to work on systems with only 32MB SHM
  settings.nrBoards = 1;
  settings.setBufferSize(0.1);

  // Test various modes
  {
    LOG_INFO("Test 16-bit complex");
    struct BoardMode mode(16, 200);
    test<i16complex>(settings, mode, "tBlockReader.in_16bit");
  }

  {
    LOG_INFO("Test 8-bit complex");
    struct BoardMode mode(8, 200);
    test<i8complex>(settings, mode, "tBlockReader.in_8bit");
  }

  return UnitTest::RunAllTests() > 0;
}

