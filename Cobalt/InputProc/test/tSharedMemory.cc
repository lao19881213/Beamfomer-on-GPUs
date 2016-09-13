//# tSharedMemory.cc
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
//# $Id: tSharedMemory.cc 24444 2013-03-28 09:16:26Z mol $

#include <lofar_config.h>

#include <unistd.h>

#include <Common/LofarLogger.h>
#include <Common/Thread/Thread.h>
#include <Common/Thread/Semaphore.h>

#include <InputProc/Buffer/SharedMemory.h>

#define DATAKEY 0x10000006

using namespace LOFAR;
using namespace Cobalt;

Semaphore semaphore;

class A
{
public:
  void creator()
  {
    sleep(1);

    SharedMemoryArena m( DATAKEY, 1024, SharedMemoryArena::CREATE, 0 );

    LOG_INFO("Memory area created");

    // wait for done
    semaphore.down();
  }

  void reader()
  {
    LOG_INFO("Waiting for memory area");

    SharedMemoryArena m( DATAKEY, 1024, SharedMemoryArena::READ, 2 );

    LOG_INFO("Memory area attached");

    // signal done
    semaphore.up();
  }
};

int main()
{
  INIT_LOGGER( "tSharedMemory" );

  /* Create a shared memory region */
  {
    LOG_INFO("Create shared memory region");

    SharedMemoryArena m( DATAKEY, 1024, SharedMemoryArena::CREATE, 0 );
  }

  /* Create a shared memory region and access it */
  {
    LOG_INFO("Create shared memory region and access it");

    SharedMemoryArena x( DATAKEY, 1024, SharedMemoryArena::CREATE, 0 );

    SharedMemoryArena y( DATAKEY, 1024, SharedMemoryArena::READ, 0 );
  }

  /* Access a non-existing shared memory region */
  {
    LOG_INFO("Access a non-existing shared memory region");

    bool caught_exception = false;

    try {
      SharedMemoryArena y( DATAKEY, 1024, SharedMemoryArena::READ, 0 );
    } catch(SystemCallException &e) {
      caught_exception = true;
    }

    ASSERT(caught_exception);
  }

  /* Access a non-existing shared memory region, with timeout */
  {
    LOG_INFO("Access a non-existing shared memory region");

    bool caught_exception = false;

    try {
      SharedMemoryArena y( DATAKEY, 1024, SharedMemoryArena::READ, 1 );
    } catch(SharedMemoryArena::TimeOutException &e) {
      caught_exception = true;
    }

    ASSERT(caught_exception);
  }

#ifdef USE_THREADS
  LOG_INFO("Debugging concurrent access");

  {
    /* Start reader before creator */
    A obj;

    // delayed creation of memory region
    Thread creator(&obj, &A::creator);

    // wait for access
    obj.reader();
  }
#endif

  /* Check whether memory access works as expected */
  {
    LOG_INFO("Checking memory access through SharedStruct");

    SharedStruct<int> writer( DATAKEY, true, 0 );

    SharedStruct<int> reader( DATAKEY, false, 0 );

    writer.get() = 42;
    ASSERT( reader.get() == 42 );
  }

  return 0;
}

