//# Copyright (C) 2007
//# ASTRON (Netherlands Foundation for Research in Astronomy)
//# P.O.Box 2, 7990 AA Dwingeloo, The Netherlands, seg@astron.nl
//#
//# This program is free software; you can redistribute it and/or modify
//# it under the terms of the GNU General Public License as published by
//# the Free Software Foundation; either version 2 of the License, or
//# (at your option) any later version.
//#
//# This program is distributed in the hope that it will be useful,
//# but WITHOUT ANY WARRANTY; without even the implied warranty of
//# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//# GNU General Public License for more details.
//#
//# You should have received a copy of the GNU General Public License
//# along with this program; if not, write to the Free Software
//# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//#
//# $Id: WallClockTime.h 22199 2012-10-03 13:27:48Z mol $

#ifndef LOFAR_IONPROC_WALL_CLOCK_TIME_H
#define LOFAR_IONPROC_WALL_CLOCK_TIME_H

//# Never #include <config.h> or #include <lofar_config.h> in a header file!

#include <Interface/RSPTimeStamp.h>
#include <Common/Thread/Condition.h>
#include <Common/Thread/Mutex.h>

#include <errno.h>
#include <time.h>


namespace LOFAR {
namespace RTCP {


class WallClockTime
{
  public:
	      WallClockTime();

    bool      waitUntil(const struct timespec &);
    bool      waitUntil(time_t);
    bool      waitUntil(const TimeStamp &);
    void      waitForever();

    void      cancelWait();

  private:
    Mutex     itsMutex;
    Condition itsCondition;
    bool      itsCancelled;
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


} // namespace RTCP
} // namespace LOFAR

#endif
