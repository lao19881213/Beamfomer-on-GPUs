//# tRSPTimeStamp.cc
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
//# $Id: tRSPTimeStamp.cc 25498 2013-06-27 05:43:41Z mol $

#include <lofar_config.h>


#include <stdint.h>
#include <UnitTest++.h>

#include <Common/LofarLogger.h>

#include <InputProc/RSPTimeStamp.h>

#define SAMPLERATE 195312.5
// start the test at INT32_MAX * SAMPLERATE
#define TESTSTART static_cast<int64>(0x7fffffff * SAMPLERATE)
// we don't want the test to take too long
#define TESTEND   static_cast<int64>(0x7fffff00 * SAMPLERATE)

using namespace LOFAR;
using LOFAR::Cobalt::TimeStamp;

TEST(One) {
  unsigned clock = static_cast<unsigned>(1024 * SAMPLERATE);

  for (int64 timecounter = TESTSTART; timecounter >= TESTEND; timecounter--) {
    TimeStamp one(timecounter, clock);
    TimeStamp other(one.getSeqId(), one.getBlockId(), clock);

    CHECK_EQUAL(one, other);
  }
}

TEST(Two) {
  unsigned clock = 200 * 1000 * 1000;

  TimeStamp ts(0, 0, clock);

  for (int64 i = 0; i < clock * 3; i += 100, ts += 100) {
    CHECK_EQUAL(i, (int64)ts);
    CHECK_EQUAL(1024 * i / clock,        ts.getSeqId());
    CHECK_EQUAL(1024 * i % clock / 1024, ts.getBlockId());
  }
}

int main()
{
  INIT_LOGGER("tRSPTimeStamp");

  return UnitTest::RunAllTests() > 0;
}

