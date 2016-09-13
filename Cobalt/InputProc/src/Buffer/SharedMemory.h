//# SharedMemory.h
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
//# $Id: SharedMemory.h 26051 2013-08-14 19:34:37Z mol $

#ifndef LOFAR_INPUT_PROC_SHARED_MEMORY_H
#define LOFAR_INPUT_PROC_SHARED_MEMORY_H

#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string>

#include <Common/Exception.h>
#include <CoInterface/Allocator.h>

namespace LOFAR
{
  namespace Cobalt
  {

    /*
     * A memory region manager for shared memory, to be used by
     * allocators defined in CoInterface/Allocator.h
     */
    class SharedMemoryArena : public FixedArena
    {
    public:
      EXCEPTION_CLASS(TimeOutException, LOFAR::Exception);

      enum Mode {
        CREATE,
        CREATE_EXCL,
        READ,
        READWRITE
      };

      /* Create a shared memory region, or attach to an existing one. The timeout
       * specifies how long the constructor will wait for the memory region to
       * appear if mode == READ or mode == READWRITE.
       */
      SharedMemoryArena( key_t key, size_t size, Mode mode = CREATE, time_t timeout = 60 );
      ~SharedMemoryArena();

      template <typename T>
      T* ptr( size_t offset = 0 ) const
      {
        return reinterpret_cast<T*>(reinterpret_cast<char*>(itsBegin) + offset);
      }

      // Return the maximum size for shared-memory buffers,
      // or 0 if the maximum could not be determined.
      static size_t maxSize();

      // Remove an existing SHM area (no-op if no SHM area has been registered
      // under the given key). If quiet = false, a warning is printed if the
      // SHM area is found and removed.
      static void remove( key_t key, bool quiet = false );

    private:
      const key_t key;
      const Mode mode;

      // Whether the memory region existed before we tried to create it.
      bool preexisting;

      // Try to open the region indicated by `this->key', and store the pointer
      // in `itsBegin' on success. Used by the constructor.
      //
      // If timeout is false, no errors are silenced. If timeout is true,
      // some errors are ignored to allow subsequent attempts.
      //
      // Returns:
      //   true:  region was opened/created succesfully.
      //   false: region does not exist (and open_flags do not contain CREATE).
      //   throws SystemCallException: a system call failed.
      bool open( int open_flags, int attach_flags, bool timeout);

      std::string modeStr( Mode mode ) const;
    };

    /*
     * Provides an interface for any struct stored as a shared memory region.
     */
    template<typename T>
    class SharedStruct
    {
    public:
      SharedStruct( key_t key, bool create = false, time_t timeout = 60 );

      T &get()
      {
        return *data.ptr<T>();
      }

      T &get() const
      {
        return *data.ptr<T>();
      }

    private:
      SharedMemoryArena data;

      SharedStruct( const SharedStruct & );
      SharedStruct &operator=( const SharedStruct & );
    };


    template<typename T>
    SharedStruct<T>::SharedStruct( key_t key, bool create, time_t timeout )
      :
      data(key, sizeof(T), create ? SharedMemoryArena::CREATE : SharedMemoryArena::READ, timeout)
    {
    }

  }
}

#endif

