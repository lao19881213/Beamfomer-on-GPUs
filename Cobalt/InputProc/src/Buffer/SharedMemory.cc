//# SharedMemory.cc
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
//# $Id: SharedMemory.cc 26484 2013-09-11 09:45:23Z mol $

#include <lofar_config.h>

#include "SharedMemory.h"

#include <ctime>
#include <fstream>
#include <iomanip>
#include <sys/mman.h>
#include <fcntl.h>
#include <boost/format.hpp>

#include <Common/Exception.h>
#include <Common/SystemCallException.h>
#include <Common/LofarLogger.h>
#include <Common/Thread/Mutex.h>
#include <Stream/FileDescriptorBasedStream.h>
#include <CoInterface/Allocator.h>

#ifdef HAVE_MPI
#include <mpi.h>
#include <InputProc/Transpose/MPIUtil.h>
#endif

// If defined, disable the use of shared memory, and fake it with
// local memory instead.
#define FAKE_SHARED_MEM

using namespace std;
using boost::format;

namespace LOFAR
{
  namespace Cobalt
  {
    // Prevent multiple threads from creating/deleting the same region at the same
    // time.
    static Mutex shmMutex;

#ifdef FAKE_SHARED_MEM
    std::map<key_t, void *> fakeSharedMem;
#endif

    // TODO: key_t can be replaced with std::string, but that would require
    // updates across SampleBuffer/StationID/BufferSettings/SharedStruct, and
    // all tests, etc.

    SharedMemoryArena::SharedMemoryArena( key_t key, size_t size, Mode mode, time_t timeout )
      :
      FixedArena(NULL, size),
      key(key),
      mode(mode),
      preexisting(false)
    {
      time_t deadline = time(0) + timeout;

      int open_flags = 0, mmap_flags = 0;

      // Check whether the size is allowed
      size_t max_shm_size = maxSize();

      if (max_shm_size > 0) {
        ASSERTSTR(size <= max_shm_size, "Requested " << size << " bytes of shared memory, but system max is " << max_shm_size << " bytes.");
      }

      // Derive the open flags based on the provided mode
      switch (mode) {
        case CREATE_EXCL:
          open_flags |= O_EXCL;
          // fall-through
        case CREATE:
          open_flags |= O_CREAT;
          // fall-through
        case READWRITE:
        default:
          open_flags |= O_RDWR;
          mmap_flags |= PROT_READ | PROT_WRITE;
          break;

        case READ:
          open_flags |= O_RDONLY;
          mmap_flags |= PROT_READ;
          break;

      }

      // Keep attempting to create/attach until the deadline
      for(;;) {
        LOG_DEBUG_STR("SHM: " << modeStr(mode) << " 0x" << hex << key << " (" << dec << size << " bytes): TRYING");

        // Try to open the buffer
        if (open(open_flags, mmap_flags, timeout > 0))
          break;

        // try again until the deadline
        if (time(0) >= deadline)
          throw TimeOutException("shared memory", THROW_ARGS);

        if (usleep(999999) < 0)
          THROW_SYSCALL("usleep");
      }

      LOG_DEBUG_STR("SHM: " << modeStr(mode) << " 0x" << hex << key << " (" << dec << size << " bytes): SUCCESS");
    }

    namespace {
      void *allocate(size_t size) {
#if 0
        // Allocate HugeTLB pages
        void *addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_HUGETLB | MAP_ANONYMOUS | MAP_PRIVATE, 0, 0);

        if (addr == MAP_FAILED)
          THROW_SYSCALL("mmap");

        return addr;
#endif

#ifdef HAVE_MPI
        // Don't use MPI_Alloc_mem if MPI wasn't initialised
        if (MPI_Initialised()) {
          return mpiAllocator.allocate(size);
        } else {
          return heapAllocator.allocate(size, 256);
        }
#else
        return heapAllocator.allocate(size, 256);
#endif
      }

      void deallocate(void *ptr) {
#ifdef HAVE_MPI
        // NOTE: We assume that MPI_Init was either called before the use of
        // SharedMemory or will never be called at all.

        // Don't use MPI_Free_mem if MPI wasn't initialised
        if (MPI_Initialised()) {
          mpiAllocator.deallocate(ptr);
        } else {
          heapAllocator.deallocate(ptr);
        }
#else
        heapAllocator.deallocate(ptr);
#endif
      }
    }


    bool SharedMemoryArena::open( int open_flags, int mmap_flags, bool timeout )
    {
#ifdef FAKE_SHARED_MEM
      (void)mmap_flags;

      //LOG_WARN_STR("Faking shared memory!");

      ScopedLock sl(shmMutex);

      preexisting = true;

      if (fakeSharedMem.find(key) == fakeSharedMem.end()) {
        if ((open_flags & O_CREAT) > 0) {
          preexisting = false;
          fakeSharedMem[key] = allocate(itsSize);
        } else {
          if (!timeout)
            THROW_SYSCALL("(dummy)");

          return false;
        }
      }

      itsBegin = fakeSharedMem[key];
      return true;
#else
      ScopedLock sl(shmMutex);

      string keystr = str(format("/%x") % key);

      // check if the SHM already exists, and mark it as such
      int tmpid;
      if ((tmpid = shm_open(keystr.c_str(), O_RDONLY, 0666)) >= 0) {
        (void)close(tmpid);

        preexisting = true;
      }

      // open the SHM
      FileDescriptorBasedStream fd(shm_open(keystr.c_str(), open_flags, 0666 ));

      if (fd.fd == -1) {
        // No timeout means we're not keeping silent about ENOENT/ENOEXIST
        if (!timeout)
          THROW_SYSCALL("shm_open");
      } else {
        if (mode == CREATE || mode == CREATE_EXCL) {
          // set correct size
          LOG_DEBUG_STR("Resizing " << keystr << " to " << itsSize << " bytes");
          if (ftruncate(fd.fd, itsSize) < 0)
            THROW_SYSCALL("ftruncate");
        }

        // attach
        itsBegin = mmap(NULL, itsSize, mmap_flags, MAP_SHARED | MAP_LOCKED | MAP_POPULATE | MAP_NORESERVE, fd.fd, 0);

        if (itsBegin == (void*)MAP_FAILED)
          THROW_SYSCALL("mmap");

        return true; // success!
      }

      return false;
#endif
    }


    size_t SharedMemoryArena::maxSize()
    {
#ifdef FAKE_SHARED_MEM
      return 0;
#else
      size_t max_shm_size = 0;

#ifdef __linux__
      // Linux provides the system limit for the shared-memory size in
      // /proc/sys/kernel/shmmax. See also 'sysctl kernel.shmmax'.
      {
        ifstream shmmax("/proc/sys/kernel/shmmax");
        shmmax >> max_shm_size;

        // If everything went ok, we got our maximum
        if (shmmax.good())
          return max_shm_size;
      }
#else
      (void)max_shm_size;
#endif

      return 0;
#endif
    }

    void SharedMemoryArena::remove( key_t key, bool quiet )
    {
      (void)quiet;

#ifdef FAKE_SHARED_MEM
      (void)key;
#else
      string keystr = str(format("/%x") % key);

      shm_unlink(keystr.c_str());
#endif
    }


    SharedMemoryArena::~SharedMemoryArena()
    {
      ScopedLock sl(shmMutex);

      try {
#ifdef FAKE_SHARED_MEM
        // destroy, if we created it
        if (!preexisting && (mode == CREATE || mode == CREATE_EXCL)) {
          deallocate(itsBegin);

          fakeSharedMem.erase(key);
        }
#else
        // detach
        if (munmap(itsBegin, itsSize) < 0)
          THROW_SYSCALL("munmap");

        // destroy, if we created it
        if (!preexisting && (mode == CREATE || mode == CREATE_EXCL)) {
          string keystr = str(format("/%x") % key);

          if (shm_unlink(keystr.c_str()) < 0)
            THROW_SYSCALL("shm_unlink");
        }
#endif
      } catch (Exception &ex) {
        LOG_ERROR_STR("Exception in destructor: " << ex);
      }
    }

    string SharedMemoryArena::modeStr( Mode mode ) const {
      switch(mode) {
        case CREATE_EXCL:
          return "CREATE_EXCL";
        case CREATE:
          return "CREATE";
        case READ:
          return "READ";
        case READWRITE:
          return "READWRITE";
        default:
          return "INVALID";
      }
    }

  }
}

