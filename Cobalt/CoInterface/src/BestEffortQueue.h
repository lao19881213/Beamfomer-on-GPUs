//# BestEffortQueue.h
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
//# $Id: BestEffortQueue.h 27103 2013-10-27 09:51:53Z mol $

#ifndef LOFAR_GPUPROC_BEST_EFFORT_QUEUE_H
#define LOFAR_GPUPROC_BEST_EFFORT_QUEUE_H

#include <Common/Thread/Queue.h>
#include <Common/Thread/Semaphore.h>

namespace LOFAR
{
  namespace Cobalt
  {
    /*
     * Implements a best-effort queue. The queue has a maximum size,
     * at which point further append()s are blocked until an item
     * is removed. If `dropIfFull` is set, append() will not block,
     * but discard the item instead.
     *
     * The noMore() function signals the end-of-stream, at which point
     * an 0 or NULL element is inserted into the stream. The reader
     * thus has to consider the value 0 as end-of-stream.
     */
    template<typename T>
    class BestEffortQueue : public Queue<T>
    {
    public:
      // Create a best-effort queue with room for `maxSize' elements.
      // If `drop' is true, appends are dropped if the queue
      // has reached `maxSize', or if no remove() has been posted yet.
      BestEffortQueue(size_t maxSize, bool drop);

      // Add an element. Returns true if append succeeded, false if element
      // was dropped.
      bool append(T&);

      // Remove an element -- 0 or NULL signals end-of-stream.
      T remove();

      // Signal end-of-stream.
      void noMore();

    private:
      const size_t maxSize;
      const bool drop;

      // true if we're removing elements, that is,
      // remove() has been called at least once.
      bool removing;

      // contains the amount of free space in the queue
      Semaphore freeSpace;

      // true if the queue is being flushed
      bool flushing;
    };
  }
}

#include "BestEffortQueue.tcc"

#endif

