//# RSPBoards.cc
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
//# $Id: RSPBoards.cc 26336 2013-09-03 10:01:41Z mol $

#include <lofar_config.h>

#include "RSPBoards.h"

#include <omp.h>

#include <Common/LofarLogger.h>

#include "OMPThread.h"


namespace LOFAR
{
  namespace Cobalt
  {


    RSPBoards::RSPBoards( const std::string &logPrefix, size_t nrBoards )
      :
      logPrefix(logPrefix),
      nrBoards(nrBoards)
    {
    }

    void RSPBoards::process()
    {
      // References to all threads that will need aborting
      std::vector<OMPThread> threads(nrBoards + 1);

      ASSERT(nrBoards > 0);

      LOG_DEBUG_STR( logPrefix << "Start" );

# pragma omp parallel sections num_threads(3)
      {
        // Board threads
#   pragma omp section
        {
          // start all boards
          LOG_INFO_STR( logPrefix << "Starting all boards" );
#     pragma omp parallel for num_threads(nrBoards)
          for (size_t i = 0; i < nrBoards; ++i) {
            OMPThread::ScopedRun sr(threads[i]);

            try {
              processBoard(i);
            } catch(Exception &ex) {
              LOG_ERROR_STR("Caught exception: " << ex);
            }
          }

          // we're done
          stop();
        }

        // Log threads
#   pragma omp section
        {
          // start log statistics
          LOG_DEBUG_STR( logPrefix << "Starting log statistics" );

          OMPThread::ScopedRun sr(threads[0 + nrBoards]);

          try {
            for(;; ) {
              if (usleep(999999) == -1 && errno == EINTR)
                // got killed
                break;

              logStatistics();
            }
          } catch(Exception &ex) {
            LOG_ERROR_STR("Caught exception: " << ex);
          }
        }

        // Watcher thread
#   pragma omp section
        {
          // wait until we have to stop
          LOG_DEBUG_STR( logPrefix << "Waiting for stop signal" );
          waiter.waitForever();

          // kill all boards
          LOG_INFO_STR( logPrefix << "Stopping all boards" );
#     pragma omp parallel for num_threads(threads.size())
          for (size_t i = 0; i < threads.size(); ++i)
            try {
              threads[i].kill();
            } catch(Exception &ex) {
              LOG_ERROR_STR("Caught exception: " << ex);
            }
        }
      }

      LOG_DEBUG_STR( logPrefix << "End" );
    }

    void RSPBoards::stop()
    {
      waiter.cancelWait();
    }


  }
}

