//# BestEffortQueue.tcc
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
//# $Id: BestEffortQueue.tcc 27103 2013-10-27 09:51:53Z mol $

namespace LOFAR
{
  namespace Cobalt 
  {

template <typename T> inline BestEffortQueue<T>::BestEffortQueue(size_t maxSize, bool drop)
:
  maxSize(maxSize),
  drop(drop),
  //removing(false), // <-- this will prevent append() if noone is remove()ing. Disabled for now, because
                     // it causes tests to fail, and even if a thread is remove()ing, objects can still
                     // pile up in the queue.
  removing(true),
  freeSpace(maxSize),
  flushing(false)
{
}


template <typename T> inline bool BestEffortQueue<T>::append(T &element)
{
  bool canAppend;

  // can't append if we're emptying the queue
  if (flushing)
    return false;

  // can't append if we're not removing elements
  if (!removing && drop)
    return false;

  // determine whether we can append
  if (drop) {
    canAppend = freeSpace.tryDown();
  } else {
    canAppend = freeSpace.down();
  }

  // append if possible
  if (canAppend) {
    Queue<T>::append(element);
  }

  return canAppend;
}


template <typename T> inline T BestEffortQueue<T>::remove()
{
  removing = true;

  T element = Queue<T>::remove();

  // freed up one spot
  freeSpace.up();

  return element;
}


template <typename T> inline void BestEffortQueue<T>::noMore()
{
  if (flushing)
    return;

  // mark queue as flushing
  flushing = true;

  // prevent writer from blocking
  freeSpace.noMore();

  // signal end-of-stream to reader
  Queue<T>::append(0);
}


  }
}

