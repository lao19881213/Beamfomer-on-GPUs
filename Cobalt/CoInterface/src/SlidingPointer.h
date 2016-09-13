//# SlidingPointer.h
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
//# $Id: SlidingPointer.h 24630 2013-04-17 12:15:57Z mol $

#ifndef LOFAR_COINTERFACE_SLIDING_POINTER_H
#define LOFAR_COINTERFACE_SLIDING_POINTER_H

//# Never #include <config.h> or #include <lofar_config.h> in a header file!

#include <set>

#include <Common/LofarLogger.h>
#include <Common/Thread/Condition.h>
#include <Common/Thread/Mutex.h>

namespace LOFAR
{
  namespace Cobalt
  {


    template <typename T>
    class SlidingPointer
    {
    public:
      SlidingPointer(T = 0);
      SlidingPointer(const SlidingPointer &other);

      void advanceTo(T);
      void waitFor(T);

      T value();

    private:
      struct WaitCondition {
        WaitCondition(T value, std::set<WaitCondition *> &set) : value(value), set(set)
        {
          set.insert(this);
        }
        ~WaitCondition()
        {
          set.erase(this);
        }

        T value;
        Condition valueReached;
        std::set<WaitCondition *> &set;
      };

      T itsValue;
      Mutex itsMutex;
      std::set<WaitCondition *> itsWaitList;
    };


    template <typename T>
    inline SlidingPointer<T>::SlidingPointer(T value)
      :
      itsValue(value)
    {
    }


    template <typename T>
    inline SlidingPointer<T>::SlidingPointer(const SlidingPointer &other)
      :
      itsValue(other.itsValue)
    {
    }


    template <typename T>
    inline void SlidingPointer<T>::advanceTo(T value)
    {
      ScopedLock lock(itsMutex);

      if (value > itsValue) {
        itsValue = value;

        for (typename std::set<WaitCondition *>::iterator it = itsWaitList.begin(); it != itsWaitList.end(); it++)
          if (value >= (*it)->value)
            (*it)->valueReached.signal();
      }
    }


    template <typename T>
    inline void SlidingPointer<T>::waitFor(T value)
    {
      ScopedLock lock(itsMutex);

      while (itsValue < value) {
        WaitCondition waitCondition(value, itsWaitList);
        waitCondition.valueReached.wait(itsMutex);
      }
    }

    template <typename T>
    inline T SlidingPointer<T>::value()
    {
      ScopedLock lock(itsMutex);

      return itsValue;
    }

  } // namespace Cobalt
} // namespace LOFAR

#endif

