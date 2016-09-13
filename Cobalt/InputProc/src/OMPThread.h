//# OMPThread.h
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
//# $Id: OMPThread.h 26015 2013-08-10 20:35:23Z amesfoort $

#ifndef LOFAR_INPUT_PROC_OMP_THREAD_H 
#define LOFAR_INPUT_PROC_OMP_THREAD_H

#include <ctime>
#include <csignal>
#include <pthread.h>

#include <Common/LofarLogger.h>
#include <Common/SystemCallException.h>

namespace LOFAR
{

  /*
   * Represents an OpenMP thread. To use,
   * call start() and stop() at the beginning and end
   * of an OpenMP thread. The kill() command can then
   * be used by another thread to kill the OpenMP thread.
   *
   * The thread is killed by sending SIGHUP to it until
   * stop() is called by the thread.
   *
   * To be able to use this class properly, please call
   * OMPThread::init() to clear the SIGHUP handler.
   *
   * Note: do NOT use this on threads which continue operating
   * through omp 'nowait' parallelism, due to race conditions.
   */
  class OMPThread
  {
  public:
    OMPThread() : id(0), stopped(false)
    {
    }

    // Register the current thread as killable
    void start()
    {
      id = pthread_self();
    }

    // Unregister the current thread
    void stop()
    {
      id = 0;
      stopped = true;
    }

    // Kill the registered thread. If no thread is registered,
    // kill() will wait.
    void kill()
    {
      while (!stopped) {
        // interrupt blocking system calls (most notably, read())
        // note that the thread will stick around until the end
        // of pragma parallel, so the thread id is always valid
        // once it has been set.
        pthread_t oldid = id;

        if (oldid > 0) {
          // Do not use THROW_SYSCALL(), because pthread_*() does not set errno,
          // but returns it.
          int error = pthread_kill(oldid, SIGHUP);
          if (error != 0)
            throw SystemCallException("pthread_kill", error, THROW_ARGS);
        }

        // sleep for 100ms - do NOT let us get killed here,
        // because we're maintaining integrity
        const struct timespec ts = { 1, 200 * 1000 };
        while (nanosleep( &ts, NULL ) == -1 && errno == EINTR)
          ;
      }
    }

    class ScopedRun
    {
    public:
      ScopedRun( OMPThread &thread ) : thread(thread)
      {
        thread.start();
      }

      ~ScopedRun()
      {
        thread.stop();
      }

    private:
      OMPThread &thread;
    };

    static void init()
    {
      signal(SIGHUP, sighandler);
      siginterrupt(SIGHUP, 1);
    }

  private:
    volatile pthread_t id;
    volatile bool stopped;

    static void sighandler(int)
    {
      /* no-op. We use SIGHUP only
       * to interrupt system calls.
       */
    }
  };

}

#endif

