//# TimeSync.h
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
//# $Id: TimeSync.h 25015 2013-05-22 22:35:17Z amesfoort $

#ifndef LOFAR_INPUT_PROC_TIME_SYNC_H
#define LOFAR_INPUT_PROC_TIME_SYNC_H

#include <ctime>

#include <Common/LofarTypes.h>
#include <Common/Thread/Mutex.h>
#include <Common/Thread/Condition.h>

namespace LOFAR
{

  class TimeSync
  {
  public:
    TimeSync();

    // set to `val'
    void set( int64 val );

    // wait for the value to be at least `val'

    // wait for the value to be at least `val' (and
    // return true), or until there is no more data
    // or a timeout (return false).
    bool wait( int64 val );
    bool wait( int64 val, struct timespec &timeout );

    // signal no more data
    void noMoreData();

  private:
    bool stop;
    int64 timestamp;
    int64 waitFor;

    Mutex mutex;
    Condition cond;
  };

  TimeSync::TimeSync()
    :
    stop(false),
    timestamp(0),
    waitFor(0)
  {
  }

  void TimeSync::set( int64 val )
  {
    ScopedLock sl(mutex);

    timestamp = val;

    if (waitFor != 0 && timestamp > waitFor)
      cond.signal();
  }

  bool TimeSync::wait( int64 val )
  {
    ScopedLock sl(mutex);

    waitFor = val;

    while (timestamp <= val && !stop)
      cond.wait(mutex);

    waitFor = 0;

    return timestamp <= val;
  }

  bool TimeSync::wait( int64 val, struct timespec &timeout )
  {
    ScopedLock sl(mutex);

    waitFor = val;

    while (timestamp <= val && !stop)
      if( !cond.wait(mutex, timeout) )
        break;

    waitFor = 0;

    return timestamp <= val;
  }

  void TimeSync::noMoreData()
  {
    ScopedLock sl(mutex);

    stop = true;
    cond.signal();
  }

}

#endif

