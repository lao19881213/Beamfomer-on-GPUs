//#  RSPTimeStamp.h: Small class to hold the timestamps from RSP
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
//#  $Id: RSPTimeStamp.h 22201 2012-10-03 14:04:42Z mol $

#ifndef LOFAR_INTERFACE_RSPTIMESTAMP_H
#define LOFAR_INTERFACE_RSPTIMESTAMP_H

#include <Common/lofar_iosfwd.h>
#include <Common/LofarTypes.h>
#include <Common/LofarLogger.h>

#define EVEN_SECOND_HAS_MORE_SAMPLES

namespace LOFAR {
  namespace RTCP {

    class TimeStamp {
    public:
      TimeStamp(); // empty constructor to be able to create vectors of TimeStamps
      TimeStamp(int64 time); // for conversion from ints, used to convert values like 0x7FFFFFFF and 0x0 for special cases.
      TimeStamp(int64 time, unsigned clockSpeed);
      TimeStamp(unsigned seqId, unsigned blockId, unsigned clockSpeed);

      TimeStamp	    &setStamp(unsigned seqId, unsigned blockId);
      unsigned	    getSeqId() const;
      unsigned	    getBlockId() const;
      unsigned	    getClock() const { return itsClockSpeed; }

      template <typename T> TimeStamp &operator += (T increment);
      template <typename T> TimeStamp &operator -= (T decrement);
			    TimeStamp  operator ++ (int); // postfix
			    TimeStamp &operator ++ ();	  // prefix
			    TimeStamp  operator -- (int);
			    TimeStamp &operator -- ();

      template <typename T> TimeStamp  operator +  (T) const;
      template <typename T> TimeStamp  operator -  (T) const;
			    int64      operator -  (const TimeStamp &) const;

			    bool       operator >  (const TimeStamp &) const;
			    bool       operator <  (const TimeStamp &) const;
			    bool       operator >= (const TimeStamp &) const;
			    bool       operator <= (const TimeStamp &) const;
			    bool       operator == (const TimeStamp &) const;
			    bool       operator != (const TimeStamp &) const;
                            bool       operator !  () const;

				       operator int64 () const;
				       operator struct timespec () const;

      friend ostream &operator << (ostream &os, const TimeStamp &ss);

    protected:
      int64	      itsTime;
      unsigned        itsClockSpeed;
    };

    inline TimeStamp::TimeStamp():
      itsTime(0),
      itsClockSpeed(0)
      {
      }

    inline TimeStamp::TimeStamp(int64 time):
      itsTime(time),
      itsClockSpeed(0)
      {
      }

    inline TimeStamp::TimeStamp(int64 time, unsigned clockSpeed):
      itsTime(time),
      itsClockSpeed(clockSpeed)
      {
      }

    inline TimeStamp::TimeStamp(unsigned seqId, unsigned blockId, unsigned clockSpeed)
      {
        itsClockSpeed = clockSpeed;

#ifdef EVEN_SECOND_HAS_MORE_SAMPLES
	itsTime = ((int64) seqId * itsClockSpeed + 512) / 1024 + blockId;
#else
	itsTime = ((int64) seqId * itsClockSpeed) / 1024 + blockId;
#endif
      }

    inline TimeStamp &TimeStamp::setStamp(unsigned seqId, unsigned blockId)
      {
#ifdef EVEN_SECOND_HAS_MORE_SAMPLES
	itsTime = ((int64) seqId * itsClockSpeed + 512) / 1024 + blockId;
#else
	itsTime = ((int64) seqId * itsClockSpeed) / 1024 + blockId;
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

    template <typename T> inline TimeStamp &TimeStamp::operator += (T increment)
      { 
	itsTime += increment;
	return *this;
      }

    template <typename T> inline TimeStamp &TimeStamp::operator -= (T decrement)
      { 
	itsTime -= decrement;
	return *this;
      }

    inline TimeStamp &TimeStamp::operator ++ ()
      { 
	++ itsTime;
	return *this;
      }

    inline TimeStamp TimeStamp::operator ++ (int)
      { 
        TimeStamp tmp = *this;
	++ itsTime;
	return tmp;
      }

    inline TimeStamp &TimeStamp::operator -- ()
      { 
	-- itsTime;
	return *this;
      }

    inline TimeStamp TimeStamp::operator -- (int)
      { 
	TimeStamp tmp = *this;
	-- itsTime;
	return tmp;
      }

    template <typename T> inline TimeStamp TimeStamp::operator + (T increment) const
      { 
	return TimeStamp(itsTime + increment, itsClockSpeed);
      }

    template <typename T> inline TimeStamp TimeStamp::operator - (T decrement) const
      { 
	return TimeStamp(itsTime - decrement, itsClockSpeed);
      }

    inline int64 TimeStamp::operator - (const TimeStamp &other) const
      { 
	return itsTime - other.itsTime;
      }

    inline bool TimeStamp::operator ! () const
      {
	return itsTime == 0;
      }

    inline TimeStamp::operator int64 () const
      {
	return itsTime;
      }

    inline TimeStamp::operator struct timespec () const
      {
	int64		ns = (int64) (itsTime * 1024 * 1e9 / itsClockSpeed);
	struct timespec ts;

	ts.tv_sec  = ns / 1000000000ULL;
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

  } // namespace RTCP

} // namespace LOFAR

#endif

