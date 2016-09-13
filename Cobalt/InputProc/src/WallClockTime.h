//# WallClockTime.h
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
//# $Id: WallClockTime.h 25497 2013-06-27 05:36:14Z mol $

#ifndef LOFAR_INPUT_PROC_WALL_CLOCK_TIME_H
#define LOFAR_INPUT_PROC_WALL_CLOCK_TIME_H

//# Never #include <config.h> or #include <lofar_config.h> in a header file!

#include <ctime>
#include <cerrno>

#include <Common/Thread/Mutex.h>
#include <Common/Thread/Condition.h>
#include <InputProc/RSPTimeStamp.h>


namespace LOFAR
{
  namespace Cobalt
  {


    class WallClockTime
    {
    public:
      WallClockTime();

      bool      waitUntil(const struct timespec &);
      bool waitUntil(time_t);
      bool      waitUntil(const TimeStamp &);
      void      waitForever();

      void      cancelWait();

    private:
      Mutex itsMutex;
      Condition itsCondition;
      bool itsCancelled;
    };


    inline WallClockTime::WallClockTime()
      :
      itsCancelled(false)
    {
    }


    inline bool WallClockTime::waitUntil(const struct timespec &timespec)
    {
      ScopedLock scopedLock(itsMutex);

      while (!itsCancelled && itsCondition.wait(itsMutex, timespec))
        ;

      return !itsCancelled;
    }


    inline bool WallClockTime::waitUntil(time_t timestamp)
    {
      struct timespec timespec = { timestamp, 0 };

      return waitUntil(timespec);
    }


    inline bool WallClockTime::waitUntil(const TimeStamp &timestamp)
    {
      return waitUntil(static_cast<struct timespec>(timestamp));
    }

    inline void WallClockTime::waitForever()
    {
      ScopedLock scopedLock(itsMutex);

      while (!itsCancelled)
        itsCondition.wait(itsMutex);
    }

    inline void WallClockTime::cancelWait()
    {
      ScopedLock scopedLock(itsMutex);

      itsCancelled = true;
      itsCondition.broadcast();
    }


  } // namespace Cobalt
} // namespace LOFAR

#endif

