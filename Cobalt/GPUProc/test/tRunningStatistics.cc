//# tRunningStatistics.cc : Unit tests for RunningStatistics 
//#
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
//# $Id: tRunningStatistics.cc 26143 2013-08-21 12:36:16Z klijn $

#include <lofar_config.h>

#include <iostream>
#include <GPUProc/RunningStatistics.h>
#include <Common/LofarLogger.h>

#include <UnitTest++.h>
#include <math.h> 

using namespace std;
using namespace LOFAR::Cobalt;

TEST(AddSingleValue)
{
  RunningStatistics stats;

  stats.push(2.0);

  CHECK(stats.count() == 1);
  CHECK(stats.mean() == 2.0);
  CHECK(stats.variance() == 0.0);
  CHECK(stats.stDev() == 0.0);
}

TEST(AddTwoSameValue)
{
  RunningStatistics stats;

  stats.push(2.0);
  stats.push(2.0);

  CHECK(stats.count() == 2);
  CHECK(stats.mean() == 2.0);
  CHECK(stats.variance() == 0.0);
  CHECK(stats.stDev() == 0.0);
}

TEST(AddTwoDifValue)
{
  RunningStatistics stats;

  stats.push(2.0);
  stats.push(3.0);

  CHECK(stats.count() == 2);
  CHECK(stats.mean() == 2.5);
  CHECK(stats.variance() == 0.5);
  CHECK(stats.stDev() == sqrt(0.5));
}

TEST(AddThreeDifValue)
{
  RunningStatistics stats;

  stats.push(2.0);
  stats.push(3.0);
  stats.push(4.0);

  CHECK(stats.count() == 3);
  CHECK(stats.mean() == 3.0);
  CHECK(stats.variance() == 1.0);
  CHECK(stats.stDev() == sqrt(1));
}


TEST(AddNoneValid)
{
  RunningStatistics stats;

  CHECK(stats.count() == 0);
  CHECK(stats.mean() == 0.0);
  CHECK(stats.variance() == 0.0);
  CHECK(stats.stDev() == 0.0);
}

TEST(AddTwoStatsAndAssign)
{
  RunningStatistics stats2;
  RunningStatistics stats3;
  stats2.push(2.0);
  stats3.push(3.0);
  stats3.push(4.0);

  
  RunningStatistics stats1 = stats2 + stats3;

  CHECK(stats1.count() == 3);
  CHECK(stats1.mean() == 3.0);
  CHECK(stats1.variance() == 1.0);
  CHECK(stats1.stDev() == sqrt(1));
}

TEST(plusis)
{
  RunningStatistics stats2;
  RunningStatistics stats3;
  stats2.push(2.0);
  stats3.push(3.0);
  stats3.push(4.0);

  
  stats2 += stats3;

  CHECK(stats2.count() == 3);
  CHECK(stats2.mean() == 3.0);
  CHECK(stats2.variance() == 1.0);
  CHECK(stats2.stDev() == sqrt(1));
}

TEST(minMax)
{
  RunningStatistics stats;
  stats.push(2.0);
  stats.push(3.0);
  stats.push(4.0);

  CHECK(stats.min() == 2.0);
  CHECK(stats.max() == 4.0);
}

int main()
{
  INIT_LOGGER("tRunningStatistics");
  return UnitTest::RunAllTests() > 0;
}



