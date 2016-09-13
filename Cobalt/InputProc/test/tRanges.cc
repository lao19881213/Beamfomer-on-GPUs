//# tRanges.cc
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
//# $Id: tRanges.cc 25606 2013-07-09 19:36:21Z mol $

#include <lofar_config.h>

#include <vector>
#include <sstream>

#include <Common/LofarTypes.h>
#include <Common/LofarLogger.h>

#include <InputProc/Buffer/Ranges.h>
#include <UnitTest++.h>

using namespace LOFAR;
using namespace Cobalt;
using namespace std;

const size_t nrRanges = 5;
size_t clockHz = 200 * 1000 * 1000;
size_t history = clockHz / 1024;
vector<char> buf(Ranges::size(nrRanges));
Ranges r(&buf[0], buf.size(), history, true);

string toStr() {
  ostringstream os;
  os << r;
  return os.str();
}

struct Fixture {
  Fixture() {
    r.clear();
  }
};

SUITE(Basic) {
  TEST_FIXTURE(Fixture, Clear) {
    r.include(10, 20);
    r.include(25, 30);

    r.clear();

    CHECK_EQUAL("", toStr());
  }
}

SUITE(InOrder) {
  TEST_FIXTURE(Fixture, OneRange) {
    bool result = r.include(10, 20);
    CHECK(result);

    CHECK_EQUAL("[10, 20)", toStr());
  }

  TEST_FIXTURE(Fixture, TwoRanges) {
    r.include(10, 20);
    r.include(30, 40);

    CHECK_EQUAL("[10, 20) [30, 40)", toStr());
  }

  TEST_FIXTURE(Fixture, TooManyRanges) {
    bool result;

    result = r.include( 10,  20);
    CHECK(result);
    result = r.include( 30,  40);
    CHECK(result);
    result = r.include( 50,  60);
    CHECK(result);
    result = r.include( 70,  80);
    CHECK(result);
    result = r.include( 90, 100);
    CHECK(result);
    result = r.include(110, 120);
    CHECK(!result);

    // we can insert once the buffer's content
    // is ancient history
    result = r.include(20 + history, 30 + history);
    CHECK(result);
  }

  TEST_FIXTURE(Fixture, Append) {
    r.include(10, 20);
    r.include(30, 40);
    r.include(40, 50);

    CHECK_EQUAL("[10, 20) [30, 50)", toStr());
  }

  TEST_FIXTURE(Fixture, ExcludeBefore) {
    r.include(10, 20);
    r.include(30, 50);
    r.include(60, 70);

    r.excludeBefore(35);

    CHECK_EQUAL("[35, 50) [60, 70)", toStr());

    r.excludeBefore(70);

    CHECK_EQUAL("", toStr());
  }
}

SUITE(OutOfOrder) {
  TEST_FIXTURE(Fixture, PrependStart) {
    r.include(10, 20);
    r.include(30, 40);

    r.include(5, 10);

    CHECK_EQUAL("[5, 20) [30, 40)", toStr());
  }

  TEST_FIXTURE(Fixture, PrependMiddle) {
    r.include(10, 20);
    r.include(30, 40);

    r.include(25, 30);

    CHECK_EQUAL("[10, 20) [25, 40)", toStr());
  }

  TEST_FIXTURE(Fixture, AppendMiddle) {
    r.include(10, 20);
    r.include(30, 40);

    r.include(20, 25);

    CHECK_EQUAL("[10, 25) [30, 40)", toStr());
  }

  TEST_FIXTURE(Fixture, Insert) {
    r.include(10, 20);
    r.include(30, 40);
    r.include(50, 60);

    r.include(25, 28);

    CHECK_EQUAL("[10, 20) [25, 28) [30, 40) [50, 60)", toStr());
  }

  TEST_FIXTURE(Fixture, InsertOldDataWhenFull) {
    r.include(10, 20);
    r.include(30, 40);
    r.include(50, 60);
    r.include(70, 80);
    r.include(90, 100);

    // full, can't insert data
    bool result = r.include(25, 28);
    CHECK(!result);
  }

  TEST_FIXTURE(Fixture, InsertNewDataWhenFull) {
    r.include(10, 20);
    r.include(30, 40);
    r.include(50 + history, 60 + history);
    r.include(70 + history, 80 + history);
    r.include(90 + history, 100 + history);

    // full, but can insert data sufficiently new
    bool result = r.include(25 + history, 28 + history);
    CHECK(result);
  }

  TEST_FIXTURE(Fixture, FillHole) {
    r.include(10, 20);
    r.include(30, 40);

    r.include(20, 30);

    CHECK_EQUAL("[10, 40)", toStr());
  }
}

SUITE(Subsets) {
  TEST_FIXTURE(Fixture, AnythingBetween) {
    r.include(10, 20);
    r.include(30, 40);

    // check exact boundaries
    CHECK(!r.anythingBetween( 0, 10));
    CHECK( r.anythingBetween(10, 20));
    CHECK(!r.anythingBetween(20, 30));
    CHECK( r.anythingBetween(30, 40));
    CHECK(!r.anythingBetween(40, 50));

    // check subsets
    CHECK(!r.anythingBetween( 2,  5));
    CHECK( r.anythingBetween(12, 15));
    CHECK(!r.anythingBetween(22, 25));
    CHECK( r.anythingBetween(32, 35));
    CHECK(!r.anythingBetween(42, 45));
  }

  TEST_FIXTURE(Fixture, SparseSet) {
    r.include(10, 20);
    r.include(30, 40);
    
    BufferSettings::flags_type s = r.sparseSet(0, 100);

    CHECK_EQUAL(20U, s.count());
    CHECK(!s.test( 9));
    CHECK( s.test(10));
    CHECK( s.test(19));
    CHECK(!s.test(20));
    CHECK(!s.test(29));
    CHECK( s.test(30));
    CHECK( s.test(39));
    CHECK(!s.test(40));
  }
}

int main()
{
  INIT_LOGGER( "tRanges" );

  return UnitTest::RunAllTests() > 0;
}

