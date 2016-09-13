//#  tRSPTimeStamp.cc: test for the RSPTimeStamp
//#
//#  Copyright (C) 2006
//#  ASTRON (Netherlands Foundation for Research in Astronomy)
//#  P.O.Box 2, 7990 AA Dwingeloo, The Netherlands, seg@astron.nl
//#
//#  This program is free software; you can redistribute it and/or modify
//#  it under the terms of the GNU General Public License as published by
//#  the Free Software Foundation; either version 2 of the License, or
//#  (at your option) any later version.
//#
//#  This program is distributed in the hope that it will be useful,
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//#  GNU General Public License for more details.
//#
//#  You should have received a copy of the GNU General Public License
//#  along with this program; if not, write to the Free Software
//#  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//#
//#  $Id: tRSPTimeStamp.cc 16497 2010-10-07 11:56:13Z mol $

#include <lofar_config.h>
#include <Common/LofarLogger.h>
#include <stdint.h>

#include <Interface/RSPTimeStamp.h>

#define SAMPLERATE 195312.5
// start the test at INT32_MAX * SAMPLERATE
#define TESTSTART static_cast<int64>(0x7fffffff * SAMPLERATE)
// we don't want the test to take too long
#define TESTEND   static_cast<int64>(0x7fffff00 * SAMPLERATE) 

using namespace LOFAR;
using LOFAR::RTCP::TimeStamp;

int main()
{
  unsigned clock = static_cast<unsigned>(1024 * SAMPLERATE);

  for (int64 timecounter = TESTSTART; timecounter >= TESTEND; timecounter--) {
    TimeStamp one(timecounter, clock);
    TimeStamp other(one.getSeqId(), one.getBlockId(), clock);
    ASSERTSTR(one == other, one << " == " << other << " counter was " << timecounter);
  }

  return 0;
}
