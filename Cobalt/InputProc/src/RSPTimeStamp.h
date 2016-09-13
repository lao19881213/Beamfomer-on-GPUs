//# RSPTimeStamp.h: Small class to hold the timestamps from RSP
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
//# $Id: RSPTimeStamp.h 26336 2013-09-03 10:01:41Z mol $

#ifndef LOFAR_COBALT_INPUTPROC_RSPTIMESTAMP_H
#define LOFAR_COBALT_INPUTPROC_RSPTIMESTAMP_H

#include <Common/lofar_iosfwd.h>
#include <Common/LofarTypes.h>
#include <Common/LofarLogger.h>

#define EVEN_SECOND_HAS_MORE_SAMPLES

namespace LOFAR
{
  namespace Cobalt
  {

    class TimeStamp
    {
    public:
      TimeStamp(); // empty constructor to be able to create vectors of TimeStamps
      TimeStamp(uint64 time); // for conversion from ints, used to convert values like 0x7FFFFFFF and 0x0 for special cases.
      TimeStamp(uint64 time, unsigned clockSpeed);
      TimeStamp(unsigned seqId, unsigned blockId, unsigned clockSpeed);

      TimeStamp     &setStamp(unsigned seqId, unsigned blockId);
      unsigned      getSeqId() const;
      unsigned      getBlockId() const;
      unsigned      getClock() const
      {
        return itsClockSpeed;
      }

      template <typename T>
      TimeStamp &operator += (T increment);
      template <typename T>
      TimeStamp &operator -= (T decrement);
      TimeStamp operator ++ (int);                        // postfix
      TimeStamp &operator ++ ();                          // prefix
      TimeStamp operator -- (int);
      TimeStamp &operator -- ();

      template <typename T>
      TimeStamp operator +  (T) const;
      template <typename T>
      TimeStamp operator -  (T) const;
      uint64 operator -  (const TimeStamp &) const;

      bool operator >  (const TimeStamp &) const;
      bool operator <  (const TimeStamp &) const;
      bool operator >= (const TimeStamp &) const;
      bool operator <= (const TimeStamp &) const;
      bool operator == (const TimeStamp &) const;
      bool operator != (const TimeStamp &) const;
      bool operator !  () const;

      double getSeconds() const;
      operator uint64 () const;
      operator struct timespec () const;

      friend ostream &operator << (ostream &os, const TimeStamp &ss);

    protected:
      uint64 itsTime;
      unsigned itsClockSpeed;
    };

    inline TimeStamp::TimeStamp() :
      itsTime(0),
      itsClockSpeed(1)
    {
    }

    inline TimeStamp::TimeStamp(uint64 time) :
      itsTime(time),
      itsClockSpeed(1)
    {
    }

    inline TimeStamp::TimeStamp(uint64 time, unsigned clockSpeed) :
      itsTime(time),
      itsClockSpeed(clockSpeed)
    {
      ASSERT(clockSpeed > 0);
    }

    inline TimeStamp::TimeStamp(unsigned seqId, unsigned blockId, unsigned clockSpeed)
    {
      ASSERT(clockSpeed > 0);

      itsClockSpeed = clockSpeed;

#ifdef EVEN_SECOND_HAS_MORE_SAMPLES
      itsTime = ((uint64) seqId * itsClockSpeed + 512) / 1024 + blockId;
#else
      itsTime = ((uint64) seqId * itsClockSpeed) / 1024 + blockId;
#endif
    }

    inline TimeStamp &TimeStamp::setStamp(unsigned seqId, unsigned blockId)
    {
#ifdef EVEN_SECOND_HAS_MORE_SAMPLES
      itsTime = ((uint64) seqId * itsClockSpeed + 512) / 1024 + blockId;
#else
      itsTime = ((uint64) seqId * itsClockSpeed) / 1024 + blockId;
#endif
      return *this;
    }

    inline unsigned TimeStamp::getSeqId() const
    {
#ifdef EVEN_SECOND_HAS_MORE_SAMPLES
      return (unsigned) (1024 * itsTime / itsClockSpeed);
#else
      return (unsigned) ((1024 * itsTime + 512) / itsClockSpeed);
#endif
    }

    inline unsigned TimeStamp::getBlockId() const
    {
#ifdef EVEN_SECOND_HAS_MORE_SAMPLES
      return (unsigned) (1024 * itsTime % itsClockSpeed / 1024);
#else
      return (unsigned) ((1024 * itsTime + 512) % itsClockSpeed / 1024);
#endif
    }

    template <typename T>
    inline TimeStamp &TimeStamp::operator += (T increment)
    {
      itsTime += increment;
      return *this;
    }

    template <typename T>
    inline TimeStamp &TimeStamp::operator -= (T decrement)
    {
      itsTime -= decrement;
      return *this;
    }

    inline TimeStamp &TimeStamp::operator ++ ()
    {
      ++itsTime;
      return *this;
    }

    inline TimeStamp TimeStamp::operator ++ (int)
    {
      TimeStamp tmp = *this;
      ++itsTime;
      return tmp;
    }

    inline TimeStamp &TimeStamp::operator -- ()
    {
      --itsTime;
      return *this;
    }

    inline TimeStamp TimeStamp::operator -- (int)
    {
      TimeStamp tmp = *this;
      --itsTime;
      return tmp;
    }

    template <typename T>
    inline TimeStamp TimeStamp::operator + (T increment) const
    {
      return TimeStamp(itsTime + increment, itsClockSpeed);
    }

    template <typename T>
    inline TimeStamp TimeStamp::operator - (T decrement) const
    {
      return TimeStamp(itsTime - decrement, itsClockSpeed);
    }

    inline uint64 TimeStamp::operator - (const TimeStamp &other) const
    {
      return itsTime - other.itsTime;
    }

    inline bool TimeStamp::operator ! () const
    {
      return itsTime == 0;
    }

    inline double TimeStamp::getSeconds() const
    {
      return 1.0 * itsTime * 1024 / itsClockSpeed;
    }

    inline TimeStamp::operator uint64 () const
    {
      return itsTime;
    }

    inline TimeStamp::operator struct timespec () const
    {
      uint64 ns = (uint64) (getSeconds() * 1e9);
      struct timespec ts;

      ts.tv_sec = ns / 1000000000ULL;
      ts.tv_nsec = ns % 1000000000ULL;

      return ts;
    }

    inline bool TimeStamp::operator > (const TimeStamp &other) const
    {
      return itsTime > other.itsTime;
    }

    inline bool TimeStamp::operator >= (const TimeStamp &other) const
    {
      return itsTime >= other.itsTime;
    }

    inline bool TimeStamp::operator < (const TimeStamp &other) const
    {
      return itsTime < other.itsTime;
    }

    inline bool TimeStamp::operator <= (const TimeStamp &other) const
    {
      return itsTime <= other.itsTime;
    }

    inline bool TimeStamp::operator == (const TimeStamp &other) const
    {
      return itsTime == other.itsTime;
    }

    inline bool TimeStamp::operator != (const TimeStamp &other) const
    {
      return itsTime != other.itsTime;
    }

  } // namespace Cobalt

} // namespace LOFAR

#endif

