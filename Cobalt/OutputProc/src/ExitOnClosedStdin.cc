//# ExitOnClosedStdin.cc
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
//# $Id: ExitOnClosedStdin.cc 25574 2013-07-04 16:02:41Z amesfoort $

#include <lofar_config.h>

#include "ExitOnClosedStdin.h"

#include <sys/select.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <Common/LofarLogger.h>
#include <Common/SystemCallException.h>
#include <CoInterface/Exceptions.h>

namespace LOFAR
{
  namespace Cobalt
  {


    ExitOnClosedStdin::ExitOnClosedStdin()
      :
      itsThread(this, &ExitOnClosedStdin::mainLoop, "[obs unknown] [stdinWatcherThread] ", 65535)
    {
    }


    ExitOnClosedStdin::~ExitOnClosedStdin()
    {
      itsThread.cancel();
    }


    void ExitOnClosedStdin::mainLoop()
    {
      // an empty read on stdin means the SSH connection closed, which indicates that we should abort

      while (true) {
        fd_set fds;

        FD_ZERO(&fds);
        FD_SET(0, &fds);

        struct timeval timeval;

        timeval.tv_sec = 1;
        timeval.tv_usec = 0;

        switch (select(1, &fds, 0, 0, &timeval)) {
        case -1: THROW_SYSCALL("select");
        case  0: continue;
        }

        char buf[1];
        ssize_t numbytes;
        numbytes = ::read(0, buf, sizeof buf);

        if (numbytes == 0) {
          LOG_FATAL("Lost stdin -- aborting"); // this most likely won't arrive, since stdout/stderr are probably closed as well
          exit(1);
        } else {
          // slow down reading data (IONProc will be spamming us with /dev/zero)
          if (usleep(999999) < 0)
            THROW_SYSCALL("usleep");
        }
      }
    }

  }
}

