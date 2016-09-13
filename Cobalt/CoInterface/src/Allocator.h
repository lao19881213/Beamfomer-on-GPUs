//# Allocator.h
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
//# $Id: Allocator.h 24385 2013-03-26 10:43:55Z amesfoort $

#ifndef LOFAR_INTERFACE_ALLOCATOR_H
#define LOFAR_INTERFACE_ALLOCATOR_H

#include <map>

#include <Common/Thread/Mutex.h>
#include <CoInterface/SparseSet.h>

namespace LOFAR
{
  namespace Cobalt
  {

    // There is a strict separation between a memory allocator and the physical
    // memory (arena) that it manages.

    /*
     * Arena owns a chunk of memory for SparseSetAllocator to play in.
     */
    class Arena
    {
    public:
      void        *begin() const
      {
        return itsBegin;
      }
      size_t      size() const
      {
        return itsSize;
      }

    protected:
      void        *itsBegin;
      size_t itsSize;
    };


    /*
     * MallocedArena allocates memory using malloc.
     */
    class MallocedArena : public Arena
    {
    public:
      MallocedArena(size_t size, size_t alignment);
      ~MallocedArena();
    };


    /*
     * FixedArena represents an externally allocated
     * piece of memory.
     */
    class FixedArena : public Arena
    {
    public:
      FixedArena(void *begin, size_t size);
    };


    /*
     * An Allocator can both allocate and deallocate pointers.
     */
    class Allocator
    {
    public:
      virtual ~Allocator();

      virtual void                *allocate(size_t size, size_t alignment = 1) = 0;
      virtual void                deallocate(void *) = 0;

      /*
       * Allows TYPE *foo = allocator.allocateTyped() without type-casting.
       */
      class TypedAllocator
      {
      public:
        TypedAllocator(Allocator &allocator, size_t alignment) : allocator(allocator), alignment(alignment)
        {
        }

        // cast-operator overloading is the only way to let C++ automatically deduce the type that we want
        // to return.
        template<typename T>
        operator T* ()
        {
          return static_cast<T*>(allocator.allocate(sizeof(T), alignment));
        }
      private:
        Allocator &allocator;
        const size_t alignment;
      };

      TypedAllocator allocateTyped(size_t alignment = 1)
      {
        return TypedAllocator(*this, alignment);
      }
    };


    /*
     * Allocates memory on the heap.
     */
    class HeapAllocator : public Allocator
    {
    public:
      virtual ~HeapAllocator();

      virtual void                *allocate(size_t size, size_t alignment = 1);
      virtual void                deallocate(void *);
    };

    extern HeapAllocator heapAllocator;


    /*
     * Allocates memory within an Arena, using a simple
     * memory manager based on a SparseSet.
     *
     * The allocator is deterministic.
     */
    class SparseSetAllocator : public Allocator
    {
    public:
      SparseSetAllocator(const Arena &);

      virtual void                *allocate(size_t size, size_t alignment = 1);
      virtual void                deallocate(void *);

      bool                        empty()
      {
        ScopedLock sl(mutex);
        return sizes.empty();
      }

    private:
      Mutex mutex;

      SparseSet<void *>           freeList;
      std::map<void *, size_t>    sizes;
    };

  } // namespace Cobalt
} // namespace LOFAR

#endif

